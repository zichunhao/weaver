import itertools
import numpy as np
import torch
import torch.nn as nn

# only vertex-particle branch
class INTagger(nn.Module):

    def __init__(self,
                 pf_dims,
                 sv_dims,
                 num_classes,
                 pf_features_dims,
                 sv_features_dims,
                 hidden, De, Do,
                 **kwargs):
        super(INTagger, self).__init__(**kwargs)
        self.P = pf_features_dims
        self.N = pf_dims
        self.S = sv_features_dims
        self.Nv = sv_dims
        self.Nr = self.N * (self.N - 1)
        self.Nt = self.N * self.Nv
        self.De = De
        self.Do = Do
        self.n_targets = num_classes
        self.hidden = hidden
        self.assign_matrices()
        self.assign_matrices_SV()
        
        self.fr = nn.Sequential(nn.Linear(2 * self.P, self.hidden),
                                nn.ReLU(),
                                nn.Linear(self.hidden, self.hidden),
                                nn.ReLU(),
                                nn.Linear(self.hidden, self.De),
                                nn.ReLU())

        self.fr_pv = nn.Sequential(nn.Linear(self.S + self.P, self.hidden),
                                nn.ReLU(),
                                nn.Linear(self.hidden, self.hidden),
                                nn.ReLU(),
                                nn.Linear(self.hidden, self.De),
                                nn.ReLU())
        
        self.fo = nn.Sequential(nn.Linear(self.P + (2 * self.De), self.hidden),
                                nn.ReLU(),
                                nn.Linear(self.hidden, self.hidden),
                                nn.ReLU(),
                                nn.Linear(self.hidden, self.Do),
                                nn.ReLU())
        
        self.fc_fixed = nn.Linear(self.Do, self.n_targets)

    def assign_matrices(self):
        self.Rr = torch.zeros(self.N, self.Nr)
        self.Rs = torch.zeros(self.N, self.Nr)
        receiver_sender_list = [i for i in itertools.product(range(self.N), range(self.N)) if i[0]!=i[1]]
        for i, (r, s) in enumerate(receiver_sender_list):
            self.Rr[r, i] = 1
            self.Rs[s, i] = 1

    def assign_matrices_SV(self):
        self.Rk = torch.zeros(self.N, self.Nt)
        self.Rv = torch.zeros(self.Nv, self.Nt)
        receiver_sender_list = [i for i in itertools.product(range(self.N), range(self.Nv))]
        for i, (k, v) in enumerate(receiver_sender_list):
            self.Rk[k, i] = 1
            self.Rv[v, i] = 1

    def edge_conv(self, x):
        Orr = torch.einsum('bij,ik->bkj', x, self.Rr.to(device=x.device))
        Ors = torch.einsum('bij,ik->bkj', x, self.Rs.to(device=x.device))
        B = torch.cat([Orr, Ors], dim=-1)
        E = self.fr(B)
        Ebar_pp = torch.einsum('bij,ki->bkj', E, self.Rr.to(device=x.device))
        return Ebar_pp

    def edge_conv_SV(self, x, y):
        Ork = torch.einsum('bij,ik->bkj', x, self.Rk.to(device=x.device))
        Orv = torch.einsum('bij,ik->bkj', y, self.Rv.to(device=x.device))
        B = torch.cat([Ork, Orv], dim=-1)
        E = self.fr_pv(B)
        Ebar_pv = torch.einsum('bij,ki->bkj', E, self.Rk.to(device=x.device))
        return Ebar_pv
        
    def forward(self, x, y):
        x = torch.transpose(x, -1, -2).contiguous() # [batch, pf_dims, pf_feature_dims]
        y = torch.transpose(y, -1, -2).contiguous() # [batch, sv_dims, sv_feature_dims]
        
        # pf - pf
        print('x',torch.isnan(x).any())
        Ebar_pp = self.edge_conv(x)
        print('Ebar_pp',torch.isnan(Ebar_pp).any())
        
        # sv - pf
        print('y',torch.isnan(y).any())
        Ebar_pv = self.edge_conv_SV(x, y)
        print('Ebar_pv',torch.isnan(Ebar_pv).any())
        
        # Final output matrix
        C = torch.cat([x, Ebar_pp, Ebar_pv], dim=-1)
        O = self.fo(C)
        print('O',torch.isnan(O).any())
        
        # Taking the sum of over each particle/vertex
        N = torch.sum(O, dim=-2)

        # Classification MLP
        N = self.fc_fixed(N)
        print('N',torch.isnan(N).any())

        return N
