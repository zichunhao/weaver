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
        print(self.N,self.Nv)
        print(self.P,self.S)
        self.Nr = self.N * (self.N - 1)
        self.Nt = self.N * self.Nv
        self.Dr = 0
        self.De = De
        self.Dx = 0
        self.Do = Do
        self.n_targets = num_classes
        self.hidden = hidden
        self.assign_matrices()
        self.assign_matrices_SV()

        self.Ra = torch.ones(self.Dr, self.Nr)
        self.fr1 = nn.Linear(2 * self.P + self.Dr, self.hidden).cuda()
        self.fr2 = nn.Linear(self.hidden, int(self.hidden)).cuda()
        self.fr3 = nn.Linear(int(self.hidden), self.De).cuda()
        self.fr1_pv = nn.Linear(self.S + self.P + self.Dr, self.hidden).cuda()
        self.fr2_pv = nn.Linear(self.hidden, int(self.hidden)).cuda()
        self.fr3_pv = nn.Linear(int(self.hidden), self.De).cuda()

        self.fo1 = nn.Linear(self.P + self.Dx + (2 * self.De), self.hidden).cuda()
        self.fo2 = nn.Linear(self.hidden, int(self.hidden)).cuda()
        self.fo3 = nn.Linear(int(self.hidden), self.Do).cuda()

        self.fc_fixed = nn.Linear(self.Do, self.n_targets).cuda()

    def assign_matrices(self):
        self.Rr = torch.zeros(self.N, self.Nr)
        self.Rs = torch.zeros(self.N, self.Nr)
        receiver_sender_list = [i for i in itertools.product(range(self.N), range(self.N)) if i[0]!=i[1]]
        for i, (r, s) in enumerate(receiver_sender_list):
            self.Rr[r, i] = 1
            self.Rs[s, i] = 1
        self.Rr = (self.Rr).cuda()
        self.Rs = (self.Rs).cuda()

    def assign_matrices_SV(self):
        self.Rk = torch.zeros(self.N, self.Nt)
        self.Rv = torch.zeros(self.Nv, self.Nt)
        receiver_sender_list = [i for i in itertools.product(range(self.N), range(self.Nv))]
        for i, (k, v) in enumerate(receiver_sender_list):
            self.Rk[k, i] = 1
            self.Rv[v, i] = 1
        self.Rk = (self.Rk).cuda()
        self.Rv = (self.Rv).cuda()

    def tmul(self, x, y):  #Takes (I * J * K)(K * L) -> I * J * L 
        x_shape = x.size()
        y_shape = y.size()
        return torch.mm(x.view(-1, x_shape[2]), y).view(-1, x_shape[1], y_shape[1])

    def forward(self, x, y):
        # pf - pf
        Orr = self.tmul(x, self.Rr)
        Ors = self.tmul(x, self.Rs)
        B = torch.cat([Orr, Ors], dim=1)
        B = torch.transpose(B, 1, 2).contiguous()
        B = nn.functional.relu(self.fr1(B.view(-1, 2 * self.P + self.Dr)))
        B = nn.functional.relu(self.fr2(B))
        E = nn.functional.relu(self.fr3(B).view(-1, self.Nr, self.De))
        del B
        E = torch.transpose(E, 1, 2).contiguous()
        Ebar_pp = self.tmul(E, torch.transpose(self.Rr, 0, 1).contiguous())
        del E

        # sv - pf
        Ork = self.tmul(x, self.Rk)
        Orv = self.tmul(y, self.Rv)
        B = torch.cat([Ork, Orv], 1)
        B = torch.transpose(B, 1, 2).contiguous()
        B = nn.functional.relu(self.fr1_pv(B.view(-1, self.S + self.P + self.Dr)))
        B = nn.functional.relu(self.fr2_pv(B))
        E = nn.functional.relu(self.fr3_pv(B).view(-1, self.Nt, self.De))
        del B
        E = torch.transpose(E, 1, 2).contiguous()
        Ebar_pv = self.tmul(E, torch.transpose(self.Rk, 0, 1).contiguous())
        Ebar_vp = self.tmul(E, torch.transpose(self.Rv, 0, 1).contiguous())
        del E

        # Final output matrix 
        C = torch.cat([x, Ebar_pp, Ebar_pv], 1)
        del Ebar_pp
        del Ebar_pv
        C = torch.transpose(C, 1, 2).contiguous()
        C = nn.functional.relu(self.fo1(C.view(-1, self.P + self.Dx + (2 * self.De))))
        C = nn.functional.relu(self.fo2(C))
        O = nn.functional.relu(self.fo3(C).view(-1, self.N, self.Do))
        del C

        # Taking the sum of over each particle/vertex
        N = torch.sum(O, dim=1)
        del O

        # Classification MLP
        N = self.fc_fixed(N)

        return N
