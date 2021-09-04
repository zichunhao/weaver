"""
Rewriting ParticleNet (https://github.com/hqucms/weaver/blob/master/utils/nn/model/ParticleNet.py) in PyTorch Geometric

Author: Raghav Kansal
"""

import torch
from torch import nn
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn import global_mean_pool
from torch import Tensor
import torch.nn.functional as F
from torch_cluster import knn_graph

import numpy as np


class LinearNet(nn.Module):
    r"""
    Module for simple fully connected networks, with ReLU activations and optional batch norm (TODO: try dropout?)

    Args:
        layers (list): list with layers of the fully connected network, optionally containing the input and output sizes inside e.g. [input_size, ... hidden layers ..., output_size]
        input_size (list, optional): size of input, if 0 or unspecified, first element of `layers` will be treated as the input size
        output_size (list, optional): size of output, if 0 or unspecified, last element of `layers` will be treated as the output size
        batch_norm (bool, optional): use batch norm or not
    """

    def __init__(self,
                 layers: list,
                 input_size: int = 0,
                 output_size: int = 0,
                 batch_norm: bool = False
                 ):

        super(LinearNet, self).__init__()

        self.batch_norm = batch_norm

        layers = layers.copy()
        if input_size: layers.insert(0, input_size)
        if output_size: layers.append(output_size)

        self.net = nn.ModuleList()
        if batch_norm: self.bn = nn.ModuleList()
        for i in range(len(layers) - 1):
            linear = nn.Linear(layers[i], layers[i + 1])
            self.net.append(linear)
            if batch_norm: self.bn.append(nn.BatchNorm1d(layers[i + 1]))

    def forward(self, x: Tensor):
        """
        Runs input `x` through linear layers and returns output

        Args:
            x (torch.Tensor): input tensor of shape [batch size, # input features]
        """
        for i in range(len(self.net)):
            x = self.net[i](x)
            if self.batch_norm: x = self.bn[i](x)
            x = F.relu(x)

        return x

    def __repr__(self):
        # TODO!
        return ""


class ParticleNetDynamicEdgeConv(MessagePassing):
    r"""Modified PyG implementation of DynamicEdgeConv (below) to include a shortcut connection.

    The dynamic edge convolutional operator from the `"Dynamic Graph CNN
    for Learning on Point Clouds" <https://arxiv.org/abs/1801.07829>`_ paper
    (see :class:`torch_geometric.nn.conv.EdgeConv`), where the graph is
    dynamically constructed using nearest neighbors in the feature space.

    Args:
        k (int): Number of nearest neighbors for edge conv.
        in_feat (int): Number of input features
        out_feat (list): # of output features of each edge network layer
        aggr (string): The aggregation operator to use (:obj:`"add"`, :obj:`"mean"`, :obj:`"max"`). (default: :obj:`"mean"`)
        **kwargs (optional): Additional arguments of :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    def __init__(self,
                 k: int,
                 in_feat: int,
                 out_feats: list,
                 batch_norm: bool = True,
                 aggr: str = 'mean',
                 **kwargs):

        super(ParticleNetDynamicEdgeConv, self).__init__(aggr=aggr, flow='target_to_source', **kwargs)
        self.nn = LinearNet(out_feats, in_feat * 2, batch_norm=batch_norm)
        self.k = k

        # 1d conv to make input dims -> output dims if not already equal for final shortcut connection
        self.sc = None if in_feat == out_feats[-1] else nn.Sequential(*[nn.Conv1d(in_feat, out_feats[-1], kernel_size=1, bias=False), nn.BatchNorm1d(out_feats[-1])])


    def forward(self, x: Tensor, batch: Tensor, coords: Tensor = None):
        """
        Run x through DynamicEdgeConv then add original features to output (shortcut connection)
        Inputs must be in PyG format, see args below.

        Args:
            x (Tensor): input tensor of shape [batch size * num nodes per batch, num features]
            batch (Tensor): tensor listing batch of each node in x i.e. = [0 x num nodes, 1 x num nodes, ..., (N - 1) x num nodes]
            coords (Tensor, optional): tensor of coordinates to use for knn, only if not using x features itself [batch size * num nodes per batch, num coordinates]
        """

        edge_index = knn_graph(x if coords is None else coords, self.k, batch)  # gets edges to nearest neighbours
        out = self.propagate(edge_index, x=x, size=None)  # message passing and aggregation

        sc = x if not self.sc else self.sc(x.unsqueeze(2)).squeeze()  # shortcut connection, described in paper

        return F.relu(out + sc)


    def message(self, x_i: Tensor, x_j: Tensor) -> Tensor:
        return self.nn(torch.cat([x_i, x_j - x_i], dim=-1))

    def __repr__(self):
        return '{}(nn={}, k={})'.format(self.__class__.__name__, self.nn, self.k)


class ParticleNetPyG(nn.Module):
    """
    ParticleNet model

    Args:
        input_dims (int): input node feature size
        num_classes (int): number of output classes
        conv_params (list of tuples of tuples, optional): parameters for each graph convolutional layer, formatted per layer as (# nearest neighbours, (# of output features per layer))
        fc_params (list of tuples, optional): layer sizes and dropout rates for final fully connected network, formatted per layer as (layer size, dropout rate)
        use_fusion (bool, optional): use all intermediate edge conv layer outputs for final output (see forward pass)
        use_ftns_bn (bool, optional): use initial batch norm layers on the input
        use_counts (bool, optional): when averaging divide by actual # of particles or zero-padded # - SECOND ONE DOESN'T MAKE SENSE SO THIS IS NO LONGER IMPLEMENTED
        for_inference (bool, optional): for inference i.e. whether to use softmax at the end or not
        for_segmentation (bool, optional): for segmentation - not sure the use case for this - NOT IMPLEMENTED PROPERLY YET
    """

    def __init__(self,
                 input_dims: int,
                 num_classes: int,
                 conv_params: list = [(7, (32, 32, 32)), (7, (64, 64, 64))],
                 fc_params: list = [(128, 0.1)],
                 use_fusion: bool = True,
                 use_fts_bn: bool = True,
                 use_counts: bool = True,
                 for_inference: bool = False,
                 for_segmentation: bool = False,
                 **kwargs):
        super(ParticleNetPyG, self).__init__(**kwargs)

        self.use_fts_bn = use_fts_bn
        if self.use_fts_bn:
            self.bn_fts = nn.BatchNorm1d(input_dims)

        self.use_counts = use_counts

        self.edge_convs = nn.ModuleList()
        for idx, layer_param in enumerate(conv_params):
            k, channels = layer_param
            in_feat = input_dims if idx == 0 else conv_params[idx - 1][1][-1]
            self.edge_convs.append(ParticleNetDynamicEdgeConv(k=k, in_feat=in_feat, out_feats=list(channels)))

        self.use_fusion = use_fusion
        if self.use_fusion:
            in_chn = sum(x[-1] for _, x in conv_params)
            out_chn = np.clip((in_chn // 128) * 128, 128, 1024)
            self.fusion_block = nn.Sequential(nn.Conv1d(in_chn, out_chn, kernel_size=1, bias=False), nn.BatchNorm1d(out_chn), nn.ReLU())

        self.for_segmentation = for_segmentation

        fcs = []
        for idx, layer_param in enumerate(fc_params):
            channels, drop_rate = layer_param
            if idx == 0: in_chn = out_chn if self.use_fusion else conv_params[-1][1][-1]
            else: in_chn = fc_params[idx - 1][0]

            if self.for_segmentation: fcs.append(nn.Sequential(nn.Conv1d(in_chn, channels, kernel_size=1, bias=False), nn.BatchNorm1d(channels), nn.ReLU(), nn.Dropout(drop_rate)))
            else: fcs.append(nn.Sequential(nn.Linear(in_chn, channels), nn.ReLU(), nn.Dropout(drop_rate)))

        if self.for_segmentation: fcs.append(nn.Conv1d(fc_params[-1][0], num_classes, kernel_size=1))
        else: fcs.append(nn.Linear(fc_params[-1][0], num_classes))

        self.fc = nn.Sequential(*fcs)

        self.for_inference = for_inference

    def forward(self, points, features, mask=None):
        """
        runs nodes through ParticleNet and outputs multi-class tagger scores

        Args:
            points (Tensor): node coordinates of shape [batch size, 2, num nodes]
            features (Tensor): pf candidate features of shape [batch size, num features, num nodes]
            mask (Tensor, optional): pf candidate masks of shape [batch size, 1, num nodes]
        """

        # convert to PyG format:
        # points: [batch size * num nodes per jet, num node coordinates], x: [batch size * num nodes per jet, num node features], batch = [0 x num nodes in jet 1 ... 1 x num nodes in jet 2 ... (N - 1) x num nodes]
        # allows naturally for variable-sized particle clouds

        batch_size = points.size(0)
        num_nodes = points.size(2)

        if mask is None: mask = (features.abs().sum(dim=1, keepdim=True) != 0)  # [batch size, 1, num nodes]
        mask = mask.view(-1).bool()

        points = points.permute(0, 2, 1).reshape(batch_size * num_nodes, -1)[mask]
        features = features.permute(0, 2, 1).reshape(batch_size * num_nodes, -1)[mask]
        zeros = torch.zeros(batch_size * num_nodes, dtype=int, device=points.device)
        zeros[torch.arange(batch_size) * num_nodes] = 1
        batch = (torch.cumsum(zeros, 0) - 1)[mask]  # batch = [0 x num nodes in jet 1 ... 1 x num nodes in jet 2 ... (N - 1) x num nodes]


        # feature batch norm
        fts = features if not self.use_fts_bn else self.bn_fts(features)

        # EdgeConv layers - if using fusion, saving outputs of each layer for 'fusion' step
        if self.use_fusion: outputs = []
        for idx, conv in enumerate(self.edge_convs):
            fts = conv(fts, batch, None if idx > 0 else points)
            if self.use_fusion: outputs.append(fts)

        # fusion step i.e. use all intermediate outputs for final output
        if self.use_fusion: fts = self.fusion_block(torch.cat(outputs, dim=1).unsqueeze(2)).squeeze()

        if self.for_segmentation: x = fts  # still need to reshape this back into a [batch size, num features, num nodes] shape tensor if actually doing segmentation
        else: x = global_mean_pool(fts, batch)  #

        output = self.fc(x)
        if self.for_inference: output = torch.softmax(output, dim=1)

        return output


class FeatureConv(nn.Module):
    def __init__(self, in_chn, out_chn, **kwargs):
        super(FeatureConv, self).__init__(**kwargs)
        self.conv = nn.Sequential(
            nn.BatchNorm1d(in_chn),
            nn.Conv1d(in_chn, out_chn, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_chn),
            nn.ReLU()
        )

    def forward(self, x):
        return self.conv(x)


class ParticleNetTaggerPyG(nn.Module):
    """
    Tagger module, forward pass takes an input of particle flow (pf) candidates and secondary vertices (sv) and outputs the tagger scores for each class

    Args:
        pf_features_dim (int): dimension of pf candidate features
        sv_features_dim (int): dimension of sv features
        num_classes (int): number of output classes
        conv_params (list of tuples of tuples): parameters for each graph convolutional layer, formatted per layer as (# nearest neighbours, (# of nodes in ))
    """

    def __init__(self,
                 pf_features_dims,
                 sv_features_dims,
                 num_classes,
                 conv_params=[(7, (32, 32, 32)), (7, (64, 64, 64))],
                 fc_params=[(128, 0.1)],
                 use_fusion=True,
                 use_fts_bn=True,
                 use_counts=True,
                 pf_input_dropout=None,
                 sv_input_dropout=None,
                 for_inference=False,
                 **kwargs):
        super(ParticleNetTaggerPyG, self).__init__(**kwargs)
        self.pf_input_dropout = nn.Dropout(pf_input_dropout) if pf_input_dropout else None
        self.sv_input_dropout = nn.Dropout(sv_input_dropout) if sv_input_dropout else None

        self.pf_conv = FeatureConv(pf_features_dims, 32)
        self.sv_conv = FeatureConv(sv_features_dims, 32)

        self.pn = ParticleNetPyG(input_dims=32,
                              num_classes=num_classes,
                              conv_params=conv_params,
                              fc_params=fc_params,
                              use_fusion=use_fusion,
                              use_fts_bn=use_fts_bn,
                              use_counts=use_counts,
                              for_inference=for_inference)

        print("Using torch geometric")


    def forward(self, pf_points, pf_features, pf_mask, sv_points, sv_features, sv_mask):
        """
        runs pf candidates and svs through ParticleNet and outputs multi-class tagger scores

        Args:
            pf_points (Tensor): pf candidate coordinates of shape [batch size, 2, num particles]
            pf_features (Tensor): pf candidate features of shape [batch size, num features, num particles]
            pf_mask (Tensor): pf candidate masks of shape [batch size, num 1, num particles]
            sv_*: same as for pfs
        """

        if self.pf_input_dropout:
            pf_mask = (self.pf_input_dropout(pf_mask) != 0).float()
            pf_points *= pf_mask
            pf_features *= pf_mask

        if self.sv_input_dropout:
            sv_mask = (self.sv_input_dropout(sv_mask) != 0).float()
            sv_points *= sv_mask
            sv_features *= sv_mask

        points = torch.cat((pf_points, sv_points), dim=2)
        features = torch.cat((self.pf_conv(pf_features * pf_mask) * pf_mask, self.sv_conv(sv_features * sv_mask) * sv_mask), dim=2)
        mask = torch.cat((pf_mask, sv_mask), dim=2)
        return self.pn(points, features, mask)
