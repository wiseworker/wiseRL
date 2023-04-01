import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import ray

from example.dqn.net import Net
import example.dqn.config as config
from core.wise_rl  import WiseRL
from core.action import Action


class ActionActor(Action):
    def __init__(self):
        super().__init__()
        self.eval_net= Net()
        self.value =0

    def updateModel(self, x):
        self.eval_net.load_state_dict(x)


    def choseAction(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x), 0) # add 1 dimension to input state x
        if np.random.uniform() < config.EPSILON:   
            actions_value = self.eval_net.forward(x)
            action = torch.max(actions_value, 1)[1].data.numpy()
            action = action[0] if config.ENV_A_SHAPE == 0 else action.reshape(config.ENV_A_SHAPE)  # return the argmax index
        else:   # random
            action = np.random.randint(0, config.N_ACTIONS)
            action = action if config.ENV_A_SHAPE == 0 else action.reshape(config.ENV_A_SHAPE)
        return action      
