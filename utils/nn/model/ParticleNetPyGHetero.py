"""
Rewriting ParticleNet (https://github.com/hqucms/weaver/blob/master/utils/nn/model/ParticleNet.py)
in PyTorch Geometric 2.

This model particularly leverage the heterogeneous graph functionalities.

Author: Raghav Kansal
"""

from typing import Union, Tuple
from torch_geometric.typing import OptTensor, PairTensor, PairOptTensor, Adj

import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F

from torch_geometric.data import HeteroData
from torch_geometric.nn.conv import MessagePassing, HeteroConv
from torch_geometric.nn import global_mean_pool

# import torch_geometric.transforms as T
from torch_cluster import knn, knn_graph

# from torch_geometric.nn.pool import knn, knn_graph

import numpy as np


class LinearNet(nn.Module):
    r"""
    Module for simple fully connected networks, with ReLU activations and optional batch norm
    (TODO: try dropout?)

    Args:
        layers (list): list with layers of the fully connected network, optionally containing the
          input and output sizes inside e.g. ``[input_size, ... hidden layers ..., output_size]``
        input_size (list, optional): size of input, if 0 or unspecified, first element of ``layers``
          will be treated as the input size
        output_size (list, optional): size of output, if 0 or unspecified, last element of
          ``layers`` will be treated as the output size
        batch_norm (bool, optional): use batch norm or not
    """

    def __init__(
        self, layers: list, input_size: int = 0, output_size: int = 0, batch_norm: bool = False
    ):

        super(LinearNet, self).__init__()

        self.batch_norm = batch_norm

        layers = layers.copy()
        if input_size:
            layers.insert(0, input_size)
        if output_size:
            layers.append(output_size)

        self.net = nn.ModuleList()
        if batch_norm:
            self.bn = nn.ModuleList()
        for i in range(len(layers) - 1):
            linear = nn.Linear(layers[i], layers[i + 1])
            self.net.append(linear)
            if batch_norm:
                self.bn.append(nn.BatchNorm1d(layers[i + 1]))

    def forward(self, x: Tensor):
        """
        Runs input `x` through linear layers and returns output

        Args:
            x (torch.Tensor): input tensor of shape [batch size, # input features]
        """
        for i in range(len(self.net)):
            x = self.net[i](x)
            if self.batch_norm:
                x = self.bn[i](x)
            x = F.relu(x)

        return x

    def __repr__(self):
        return f"{self.__class__.__name__}(net = {self.net})"


class ParticleNetEdgeNet(MessagePassing):
    def __init__(
        self,
        source_in_feat: int,
        dest_in_feat: int,
        out_feats: list,
        use_edge_feats: bool = False,
        edge_feats: dict = {
            "deltaR": True,
            "m2": True,
            "kT": True,
            "z": True,
        },
        batch_norm: bool = True,
        aggr: str = "mean",
        **kwargs,
    ):
        super(ParticleNetEdgeNet, self).__init__(aggr=aggr, **kwargs)
        self.edge_feats = edge_feats

        self.num_edge_feats = 0
        for key in edge_feats:
            if edge_feats[key]:
                self.num_edge_feats += 1

        self.use_edge_feats = use_edge_feats and self.num_edge_feats > 0

        self.nn = LinearNet(
            out_feats,
            source_in_feat + dest_in_feat + (self.num_edge_feats * int(self.use_edge_feats)),
            batch_norm=batch_norm,
        )

        # 1d conv to make input dims -> output dims if not already equal for final shortcut
        # connection
        self.sc = (
            None
            if dest_in_feat == out_feats[-1]
            else nn.Sequential(
                *[
                    nn.Conv1d(dest_in_feat, out_feats[-1], kernel_size=1, bias=False),
                    nn.BatchNorm1d(out_feats[-1]),
                ]
            )
        )

        # print(self.nn)

    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: PairTensor = (x, x)
        # propagate_type: (x: PairTensor)
        out = self.propagate(edge_index, x=x, size=None)

        sc = x[1] if not self.sc else self.sc(x[1].unsqueeze(2)).squeeze()

        # summing input and output as a shortcut connection, as described in paper
        return F.relu(out + sc)

    def message(self, x_i: Tensor, x_j: Tensor) -> Tensor:
        return self.nn(torch.cat([x_i, x_j], dim=-1))

    def __repr__(self):
        return "{}(nn={})".format(self.__class__.__name__, self.nn)


def knn_hack(pf_points, sv_points, pf_sv_k, sv_pf_k, pf_batch, sv_batch):
    """quick hack for knn while waiting for fix for https://github.com/pyg-team/pytorch_geometric/issues/3441"""

    # issue occurs when no SVs in the last batch
    if sv_batch[-1] != pf_batch[-1]:
        # find index of last batch and remove all nodes in that batch from pf tensors
        last_batch_idx = torch.searchsorted(pf_batch, sv_batch[-1] + 1)
        temp_pf_batch = pf_batch[:last_batch_idx]
        temp_pf_points = pf_points[:last_batch_idx]

        pf_sv_edge_index = knn(sv_points, temp_pf_points, pf_sv_k, sv_batch, temp_pf_batch)[[1, 0]]
        sv_pf_edge_index = knn(temp_pf_points, sv_points, sv_pf_k, temp_pf_batch, sv_batch)[[1, 0]]

        return pf_sv_edge_index, sv_pf_edge_index
    else:
        pf_sv_edge_index = knn(sv_points, pf_points, pf_sv_k, sv_batch, pf_batch)[[1, 0]]
        sv_pf_edge_index = knn(pf_points, sv_points, sv_pf_k, pf_batch, sv_batch)[[1, 0]]

        return pf_sv_edge_index, sv_pf_edge_index


class ParticleNetTaggerPyGHetero(nn.Module):
    """
    Tagger module, forward pass takes an input of particle flow (pf) candidates and secondary
    vertices (sv), and outputs the tagger scores for each class

    Args:
        pf_features_dim (int): dimension of pf candidate features
        sv_features_dim (int): dimension of sv features
        num_classes (int): number of output classes
        conv_params (list of tuples of tuples, optional): parameters for each edge net layer,
          formatted per layer as
          ``(# nearest neighbours for pf -> pf, #NN sv -> sv, #NN sv -> pf, #NN pf -> sv), (# of output features per layer)``
        fc_params (list of tuples, optional): layer sizes and dropout rates for final fully
          connected network, formatted per layer as ``(layer size, dropout rate)``
        use_fusion (bool, optional): use all intermediate edge conv layer outputs for final output
        use_ftns_bn (bool, optional): use initial batch norm layers on the input
        for_inference (bool, optional): for inference i.e. whether to use softmax at the end or not
        use_edge_feats (bool): use edge features in dynamic edge conv or not
    """

    def __init__(
        self,
        pf_features_dims: int,
        sv_features_dims: int,
        num_classes: int,
        conv_params: list = [
            ((16, 7, 1, 16), (64, 64, 64)),
            ((16, 7, 1, 16), (128, 128, 128)),
            ((16, 7, 1, 16), (256, 256, 256)),
            ((16, 7, 1, 16), (256, 256, 256)),
        ],
        fc_params: list = [(128, 0.1)],
        use_fusion: bool = True,
        use_fts_bn: bool = True,
        pf_input_dropout: bool = None,
        sv_input_dropout: bool = None,
        for_inference: bool = False,
        use_edge_feats: bool = False,
        **kwargs,
    ):
        super(ParticleNetTaggerPyGHetero, self).__init__(**kwargs)
        self.use_edge_feats = use_edge_feats

        self.pf_input_dropout = nn.Dropout(pf_input_dropout) if pf_input_dropout else None
        self.sv_input_dropout = nn.Dropout(sv_input_dropout) if sv_input_dropout else None

        self.use_fts_bn = use_fts_bn
        if self.use_fts_bn:
            self.pf_bn_fts = nn.BatchNorm1d(pf_features_dims)
            self.sv_bn_fts = nn.BatchNorm1d(sv_features_dims)

        self.conv_params = conv_params

        self.edge_convs = torch.nn.ModuleList()
        for idx, layer_params in enumerate(conv_params):
            _, channels = layer_params
            channels = list(channels)
            pf_ins = pf_features_dims if idx == 0 else conv_params[idx - 1][1][-1]
            sv_ins = sv_features_dims if idx == 0 else conv_params[idx - 1][1][-1]

            # Separate EdgeNets for every type of interaction
            conv = HeteroConv(
                {
                    ("pfs", "edge", "pfs"): ParticleNetEdgeNet(
                        pf_ins, pf_ins, channels, aggr="mean"
                    ),
                    ("pfs", "edge", "svs"): ParticleNetEdgeNet(
                        pf_ins, sv_ins, channels, aggr="mean"
                    ),
                    ("svs", "edge", "svs"): ParticleNetEdgeNet(
                        sv_ins, sv_ins, channels, aggr="mean"
                    ),
                    ("svs", "edge", "pfs"): ParticleNetEdgeNet(
                        sv_ins, pf_ins, channels, aggr="mean"
                    ),
                },
                aggr="sum",
            )
            self.edge_convs.append(conv)

        self.use_fusion = use_fusion
        if self.use_fusion:
            in_chn = sum(x[-1] for _, x in conv_params)
            out_chn = np.clip((in_chn // 128) * 128, 128, 1024)
            self.pf_fusion_block = nn.Sequential(
                nn.Conv1d(in_chn, out_chn, kernel_size=1, bias=False),
                nn.BatchNorm1d(out_chn),
                nn.ReLU(),
            )
            self.sv_fusion_block = nn.Sequential(
                nn.Conv1d(in_chn, out_chn, kernel_size=1, bias=False),
                nn.BatchNorm1d(out_chn),
                nn.ReLU(),
            )

        fcs = []
        for idx, layer_param in enumerate(fc_params):
            layer_size, drop_rate = layer_param
            if idx == 0:
                in_chn = out_chn if self.use_fusion else conv_params[-1][1][-1]
            else:
                in_chn = fc_params[idx - 1][0]

            fcs.append(
                nn.Sequential(nn.Linear(in_chn, layer_size), nn.ReLU(), nn.Dropout(drop_rate))
            )

        fcs.append(nn.Linear(fc_params[-1][0], num_classes))

        self.fc = nn.Sequential(*fcs)

        self.for_inference = for_inference

        print("Using torch geometric")

    def forward(
        self,
        pf_points: Tensor,
        pf_features: Tensor,
        pf_mask: Tensor,
        sv_points: Tensor,
        sv_features: Tensor,
        sv_mask: Tensor,
    ):
        """
        Runs pf candidates and svs through ParticleNet and outputs multi-class tagger scores

        Args:
            pf_points (Tensor): pf candidate coordinates of shape ``[batch size, 2, num particles]``
            pf_features (Tensor): pf candidate features of shape
              ``[batch size, num features, num particles]``
            pf_mask (Tensor): pf candidate masks of shape ``[batch size, num 1, num particles]``
            sv_*: same as for pfs
        """

        if self.pf_input_dropout:
            pf_mask = (self.pf_input_dropout(pf_mask) != 0).float()

        if self.sv_input_dropout:
            sv_mask = (self.sv_input_dropout(sv_mask) != 0).float()

        pf_points, pf_features, pf_batch = self._convert_to_pyg_graphs(
            pf_points, pf_features, pf_mask
        )

        sv_points, sv_features, sv_batch = self._convert_to_pyg_graphs(
            sv_points, sv_features, sv_mask
        )

        if self.use_edge_feats:
            pf_pee, sv_pee = self._get_pee(pf_points, pf_features, sv_points, sv_features)

        pf_fts = pf_features if not self.use_fts_bn else self.pf_bn_fts(pf_features)
        sv_fts = sv_features if not self.use_fts_bn else self.sv_bn_fts(sv_features)

        data = HeteroData()
        data["pfs"].x = pf_fts
        data["svs"].x = sv_fts

        # if using fusion, saving outputs of each EdgeNet layer for 'fusion' step
        if self.use_fusion:
            pf_outputs = []
            sv_outputs = []

        # message passing time
        for idx, conv in enumerate(self.edge_convs):
            # knn-ing
            pf_k, sv_k, pf_sv_k, sv_pf_k = self.conv_params[idx][0]

            # separate knn for each interaction
            if idx == 0:
                pf_edge_index = knn_graph(pf_points, pf_k, pf_batch, loop=True)
                sv_edge_index = knn_graph(sv_points, sv_k, sv_batch, loop=True)
                # knn goes from pfs -> sv but for message passing we want sv -> pf so edge index is
                # inverted (same for pf -> sv message passing).

                # pf_sv_edge_index = knn(sv_points, pf_points, pf_sv_k, sv_batch, pf_batch)[[1, 0]]
                # sv_pf_edge_index = knn(pf_points, sv_points, sv_pf_k, pf_batch, sv_batch)[[1, 0]]

                pf_sv_edge_index, sv_pf_edge_index = knn_hack(
                    pf_points, sv_points, pf_sv_k, sv_pf_k, pf_batch, sv_batch
                )
            else:
                pf_edge_index = knn_graph(data["pfs"].x, pf_k, pf_batch)
                sv_edge_index = knn_graph(data["svs"].x, sv_k, sv_batch)

                # pf_sv_edge_index = knn(data["svs"].x, data["pfs"].x, pf_sv_k, sv_batch, pf_batch)[
                #     [1, 0]
                # ]
                # sv_pf_edge_index = knn(data["pfs"].x, data["svs"].x, sv_pf_k, pf_batch, sv_batch)[
                #     [1, 0]
                # ]

                pf_sv_edge_index, sv_pf_edge_index = knn_hack(
                    data["pfs"].x, data["svs"].x, pf_sv_k, sv_pf_k, pf_batch, sv_batch
                )

            data["pfs", "edge", "pfs"].edge_index = pf_edge_index
            data["svs", "edge", "svs"].edge_index = sv_edge_index
            data["svs", "edge", "pfs"].edge_index = pf_sv_edge_index
            data["pfs", "edge", "svs"].edge_index = sv_pf_edge_index

            out = conv(data.x_dict, data.edge_index_dict)
            for key in out:
                data[key].x = out[key]

            if self.use_fusion:
                pf_outputs.append(data["pfs"].x)
                sv_outputs.append(data["svs"].x)

        # fusion step i.e. use all intermediate outputs for final output
        if self.use_fusion:
            pf_fts = self.pf_fusion_block(torch.cat(pf_outputs, dim=1).unsqueeze(2)).squeeze()
            sv_fts = self.sv_fusion_block(torch.cat(sv_outputs, dim=1).unsqueeze(2)).squeeze()

        # do a global mean pool on the combined pf + sv graph
        x = global_mean_pool(
            torch.cat((pf_fts, sv_fts), dim=0), torch.cat((pf_batch, sv_batch), dim=0)
        )

        output = self.fc(x)
        if self.for_inference:
            output = torch.softmax(output, dim=1)

        return output

    def _convert_to_pyg_graphs(self, points: Tensor, features: Tensor, mask: Tensor):
        """
        Converts 3D ``points`` and ``features`` tensors, of shape
        ``[num_jets, num_features, num_particles]``, to a single large disconnected graph of shape
        ``[sum of num_particles in each jet, num_features]``, excluding the masked particles.

        particles <--> jet mapping is retained in the ``batch`` tensor of shape
        ``[sum of num_particles in each jet]`` which stores the jet # of each particle.

        Returns the reformatted ``points`` and ``features`` tensors, and the ``batch`` tensor.
        """
        batch_size = points.size(0)
        num_nodes = points.size(2)

        if mask is None:
            mask = features.abs().sum(dim=1, keepdim=True) != 0  # [batch size, 1, num nodes]

        mask = mask.view(-1).bool()

        points = points.permute(0, 2, 1).reshape(batch_size * num_nodes, -1)[mask]
        features = features.permute(0, 2, 1).reshape(batch_size * num_nodes, -1)[mask]
        zeros = torch.zeros(batch_size * num_nodes, dtype=int, device=points.device)
        zeros[torch.arange(batch_size) * num_nodes] = 1
        # batch = [0 x num nodes in jet 1 ... 1 x num nodes in jet 2 ... (N - 1) x num nodes]
        batch = (torch.cumsum(zeros, 0) - 1)[mask]

        return points, features, batch

    def _get_pee(self, pf_points, pf_features, sv_points, sv_features):
        """Calculates the pT, E, and |\eta| for each pf cand and sv"""
        pf_pt = torch.exp((pf_features[:, 0] / 0.5) + 1.0)
        pf_e = torch.exp((pf_features[:, 1] / 0.5) + 1.3)
        pf_abseta = (pf_features[:, 9] / 1.6) + 0.6
        pf_pee = torch.stack((pf_pt, pf_e, pf_abseta), dim=1)

        sv_pt = torch.exp((sv_features[:, 0] / 0.6) + 4.0)
        sv_m = (sv_features[:, 1] / 0.3) + 1.2
        sv_abseta = (sv_features[:, 4] / 1.6) + 0.5
        sv_p = sv_pt * torch.cosh(sv_abseta)
        sv_e = torch.sqrt((sv_m ** 2) + (sv_p ** 2))
        sv_pee = torch.stack((sv_pt, sv_e, sv_abseta), dim=1)

        return pf_pee, sv_pee

    def __repr__(self):
        # fusion_str = f",\npf_fusion = {self.fnd_layer}" if self.use_fusion else ""
        return f"{self.__class__.__name__}(EdgeNets = {self.edge_convs}\nFC = {self.fc})"
