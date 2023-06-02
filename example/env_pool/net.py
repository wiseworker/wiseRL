import example.env_pool.config as config
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

def orthogonal_init(layer, gain=1.0):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0)

class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(config.N_STATE, config.HIDDEN_WIDTH)
        self.fc2 = nn.Linear(config.HIDDEN_WIDTH, config.HIDDEN_WIDTH)
        self.fc3 = nn.Linear(config.HIDDEN_WIDTH, config.N_ACTION)
        self.activate_func = [nn.ReLU(), nn.Tanh()][config.USE_TANH]  # Trick10: use tanh

        if config.USE_ORTHOGONAL_INIT:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.fc3, gain=0.01)

    def forward(self, s):
        s = self.activate_func(self.fc1(s))
        s = self.activate_func(self.fc2(s))
        a_prob = torch.softmax(self.fc3(s), dim=1)
        return a_prob


class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(config.N_STATE, config.HIDDEN_WIDTH)
        self.fc2 = nn.Linear(config.HIDDEN_WIDTH, config.HIDDEN_WIDTH)
        self.fc3 = nn.Linear(config.HIDDEN_WIDTH, 1)
        self.activate_func = [nn.ReLU(), nn.Tanh()][config.USE_TANH]  # Trick10: use tanh

        if config.USE_ORTHOGONAL_INIT:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.fc3)

    def forward(self, s):
        s = self.activate_func(self.fc1(s))
        s = self.activate_func(self.fc2(s))
        v_s = self.fc3(s)
        return v_s