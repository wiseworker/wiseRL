import torch
import torch.nn as nn
from torch.autograd import Variable
from wiseRL.core.action import Action
from example.ddp_ppo.net import Actor
from torch.distributions import Categorical

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device",device)
class ActionActor(Action):
    def __init__(self):
        self.actor = Actor().to(device)

    def choseAction(self, s):
        s = torch.unsqueeze(torch.tensor(s, dtype=torch.float), 0).to(device)
        with torch.no_grad():
            dist = Categorical(probs=self.actor(s))
            a = dist.sample()
            a_logprob = dist.log_prob(a)
        return a.cpu().numpy()[0], a_logprob.cpu().numpy()[0]

    def updateModel(self, param ):
        for name, mm in param.items():
	        param[name]= mm.to(device)
        self.actor.load_state_dict(param)
       