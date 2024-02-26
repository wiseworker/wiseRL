import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from .utils import orthogonal_init
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch.distributions import Categorical

class NNActor(nn.Module):
    def __init__(self, net_cfg,cfg):
        super(NNActor, self).__init__()
        state_dim = net_cfg.get("state_dim") 
        action_dim = net_cfg.get("action_dim") 
        self.fc1 = nn.Linear(state_dim, cfg.hidden_width)
        self.fc2 = nn.Linear(cfg.hidden_width, cfg.hidden_width)
        self.fc3 = nn.Linear(cfg.hidden_width, action_dim)
        self.activate_func = [nn.ReLU(), nn.Tanh()][cfg.use_tanh]  # Trick10: use tanh

        if cfg.use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.fc3, gain=0.01)

    def forward(self, s):
        s = self.activate_func(self.fc1(s))
        s = self.activate_func(self.fc2(s))
        a_prob = torch.softmax(self.fc3(s), dim=1)
        return a_prob

    def get_dist(self,s):
        probs = self.forward(s)
        dist = Categorical(probs)
        a = dist.sample()
        a_logprob = dist.log_prob(a)
        return a.numpy()[0], a_logprob.numpy()[0]

class NNCritic(nn.Module):
    def __init__(self, net_cfg,cfg):
        super(NNCritic, self).__init__()
        state_dim = net_cfg.get("state_dim") 
        action_dim = net_cfg.get("action_dim") 
        self.fc1 = nn.Linear(state_dim, cfg.hidden_width)
        self.fc2 = nn.Linear(cfg.hidden_width, cfg.hidden_width)
        self.fc3 = nn.Linear(cfg.hidden_width, 1)
        self.activate_func = [nn.ReLU(), nn.Tanh()][cfg.use_tanh]  # Trick10: use tanh

        if cfg.use_orthogonal_init:
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.fc3)

    def forward(self, s):
        s = self.activate_func(self.fc1(s))
        s = self.activate_func(self.fc2(s))
        v_s = self.fc3(s)
        return v_s