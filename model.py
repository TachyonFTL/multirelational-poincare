import numpy as np
import torch.nn.functional as F
from torch import nn
from utils import *


class MuRP(torch.nn.Module):
    def __init__(self, d, dim):
        super(MuRP, self).__init__()
        self.Eh = torch.nn.Embedding(len(d.entities), dim, padding_idx=0)
        self.Eh.weight.data = (1e-3 * torch.randn((len(d.entities), dim), dtype=torch.double, device="cuda"))
        self.rvh = torch.nn.Embedding(len(d.relations), dim, padding_idx=0)
        self.rvh.weight.data = (1e-3 * torch.randn((len(d.relations), dim), dtype=torch.double, device="cuda"))
        self.Wu = torch.nn.Parameter(torch.tensor(np.random.uniform(-1, 1, (len(d.relations), 
                                        dim)), dtype=torch.double, requires_grad=True, device="cuda"))
        #self.bs = torch.nn.Parameter(torch.zeros(len(d.entities), dtype=torch.double, requires_grad=True, device="cuda"))
        #self.bo = torch.nn.Parameter(torch.zeros(len(d.entities), dtype=torch.double, requires_grad=True, device="cuda"))
        self.loss = torch.nn.BCEWithLogitsLoss()

    def forward(self, u_idx, r_idx, v_idx):
        u = self.Eh.weight[u_idx]
        v = self.Eh.weight[v_idx]
        Ru = self.Wu[r_idx]
        rvh = self.rvh.weight[r_idx]


        u = torch.where(torch.norm(u, 2, dim=-1, keepdim=True) >= 1, 
                        u/(torch.norm(u, 2, dim=-1, keepdim=True)-1e-5), u)
        v = torch.where(torch.norm(v, 2, dim=-1, keepdim=True) >= 1, 
                        v/(torch.norm(v, 2, dim=-1, keepdim=True)-1e-5), v)
        rvh = torch.where(torch.norm(rvh, 2, dim=-1, keepdim=True) >= 1, 
                          rvh/(torch.norm(rvh, 2, dim=-1, keepdim=True)-1e-5), rvh)   
        u_e = p_log_map(u)
        u_W = u_e * Ru
        u_m = p_exp_map(u_W)
        v_m = p_sum(v, rvh)
        u_m = torch.where(torch.norm(u_m, 2, dim=-1, keepdim=True) >= 1, 
                          u_m/(torch.norm(u_m, 2, dim=-1, keepdim=True)-1e-5), u_m)
        v_m = torch.where(torch.norm(v_m, 2, dim=-1, keepdim=True) >= 1, 
                          v_m/(torch.norm(v_m, 2, dim=-1, keepdim=True)-1e-5), v_m)
        
        sqdist = (2.*artanh(torch.clamp(torch.norm(p_sum(-u_m, v_m), 2, dim=-1), 1e-10, 1-1e-5)))**2

        return -sqdist
               # + self.bs[u_idx] + self.bo[v_idx]
# Test:
# Number of data points: 5842
# Hits @10: 0.9984594317014721
# Hits @3: 0.984594317014721
# Hits @1: 0.8623758986648408
# Mean rank: 1.2516261554262238
# Mean reciprocal rank: 0.923340773888485



class MuRE(torch.nn.Module):
    def __init__(self, d, dim):
        super(MuRE, self).__init__()
        self.E = torch.nn.Embedding(len(d.entities), dim, padding_idx=0)
        self.E.weight.data = self.E.weight.data.double()
        self.E.weight.data = (1e-3 * torch.randn((len(d.entities), dim), dtype=torch.double, device="cuda"))
        self.Wu = torch.nn.Parameter(torch.tensor(np.random.uniform(-1, 1, (len(d.relations), 
                                        dim)), dtype=torch.double, requires_grad=True, device="cuda"))
        print(self.Wu)
        self.rv = torch.nn.Embedding(len(d.relations), dim, padding_idx=0)
        self.rv.weight.data = self.rv.weight.data.double()
        self.rv.weight.data = (1e-3 * torch.randn((len(d.relations), dim), dtype=torch.double, device="cuda"))
        self.bs = torch.nn.Parameter(torch.zeros(len(d.entities), dtype=torch.double, requires_grad=True, device="cuda"))
        self.bo = torch.nn.Parameter(torch.zeros(len(d.entities), dtype=torch.double, requires_grad=True, device="cuda"))
        self.loss = torch.nn.BCEWithLogitsLoss()
       
    def forward(self, u_idx, r_idx, v_idx):
        u = self.E.weight[u_idx]
        v = self.E.weight[v_idx]
        Ru = self.Wu[r_idx]
        rv = self.rv.weight[r_idx]
        u_size = u.size()
        print(u.size())
        print(Ru.size())
        exit()
        u_W = u * Ru

        sqdist = torch.sum(torch.pow(u_W - (v+rv), 2), dim=-1)
        return -sqdist + self.bs[u_idx] + self.bo[v_idx]


class EMuRP(torch.nn.Module):
    def __init__(self, d, dim, dim_hidden, dim_euclid, pretrain_embedding):
        super(EMuRP, self).__init__()
        self.Eh = torch.nn.Embedding(len(d.entities), dim, padding_idx=0)
        self.Eh.weight.data = (1e-3 * torch.randn((len(d.entities), dim), dtype=torch.double, device="cuda"))
        self.rvh = torch.nn.Embedding(len(d.relations), dim, padding_idx=0)
        self.rvh.weight.data = (1e-3 * torch.randn((len(d.relations), dim), dtype=torch.double, device="cuda"))
        self.Wu = torch.nn.Parameter(torch.tensor(np.random.uniform(-1, 1, (len(d.relations),
                                                                            dim)), dtype=torch.double,
                                                  requires_grad=True, device="cuda"))
        # self.bs = torch.nn.Parameter(torch.zeros(len(d.entities), dtype=torch.double, requires_grad=True, device="cuda"))
        # self.bo = torch.nn.Parameter(torch.zeros(len(d.entities), dtype=torch.double, requires_grad=True, device="cuda"))
        self.euclidean = nn.Embedding(len(d.entities), dim_euclid, padding_idx=0)
        self.embedding.weight.data.copy_(pretrain_embedding)
        self.embedding.weight.requires_grad = False
        self.euclidean_map_0 = nn.Linear(dim_euclid, dim_hidden)
        self.euclidean_map_1 = nn.Linear(dim_hidden, dim)

        # self.projection = nn.Linear(dim_euclid, dim)

        self.loss = torch.nn.BCEWithLogitsLoss()
        self.loss_1 = torch.nn.MSELoss()

    def forward(self, u_idx, r_idx, v_idx, phase=1):
        if phase == 1:
            u = self.Eh.weight[u_idx]
            v = self.Eh.weight[v_idx]
            return self.forward_1(u, r_idx, v)
        elif phase == 2:
            return self.forward_2(u_idx, r_idx, v_idx)
        else:
            u, v = self.forward_2(u_idx, r_idx, v_idx)
            return self.forward_1(u, r_idx, v)

    def forward_1(self, u, r_idx, v):
        Ru = self.Wu[r_idx]
        rvh = self.rvh.weight[r_idx]

        u = torch.where(torch.norm(u, 2, dim=-1, keepdim=True) >= 1,
                        u / (torch.norm(u, 2, dim=-1, keepdim=True) - 1e-5), u)
        v = torch.where(torch.norm(v, 2, dim=-1, keepdim=True) >= 1,
                        v / (torch.norm(v, 2, dim=-1, keepdim=True) - 1e-5), v)
        rvh = torch.where(torch.norm(rvh, 2, dim=-1, keepdim=True) >= 1,
                          rvh / (torch.norm(rvh, 2, dim=-1, keepdim=True) - 1e-5), rvh)
        u_e = p_log_map(u)
        u_W = u_e * Ru
        u_m = p_exp_map(u_W)
        v_m = p_sum(v, rvh)
        u_m = torch.where(torch.norm(u_m, 2, dim=-1, keepdim=True) >= 1,
                          u_m / (torch.norm(u_m, 2, dim=-1, keepdim=True) - 1e-5), u_m)
        v_m = torch.where(torch.norm(v_m, 2, dim=-1, keepdim=True) >= 1,
                          v_m / (torch.norm(v_m, 2, dim=-1, keepdim=True) - 1e-5), v_m)

        sqdist = (2. * artanh(torch.clamp(torch.norm(p_sum(-u_m, v_m), 2, dim=-1), 1e-10, 1 - 1e-5))) ** 2
        return -sqdist

    def forward_2(self, u_idx, r_idx, v_idx):
        u = self.euclidean(u_idx)
        v = self.euclidean(v_idx)
        return self.euclidean_map_1(F.relu(self.euclidean_map_0(u))), \
               self.euclidean_map_1(F.relu(self.euclidean_map_0(v)))
