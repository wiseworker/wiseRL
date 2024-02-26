
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
# 2. Define the network used in both target net and the net for training
class DNNNet(nn.Module):
    def __init__(self,n_state,n_action):
        super(DQNNet, self).__init__()
        # Define the structure of fully connected network
        self.fc1 = nn.Linear(n_state, 512)  # layer 1
        self.fc1.weight.data.normal_(0, 0.1) # in-place initilization of weights of fc1
        self.out = nn.Linear(512, n_action) # layer 2
        self.out.weight.data.normal_(0, 0.1) # in-place initilization of weights of fc2
        
        
    def forward(self, x):
        # Define how the input data pass inside the network
        x = self.fc1(x)
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value