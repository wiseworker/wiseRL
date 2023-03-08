
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import ray
from example.dqn.net import Net
import example.dqn.config as config
from  example.dqn.mem_store import MemoryStore
from core.wise_rl  import WiseRL
from core.learner import Learner
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class LearnerActor(Learner):
    def __init__(self):
        super().__init__()
        self.eval_net, self.target_net = Net().to(device), Net().to(device)
        self.learn_step_counter = 0 
        self.memory_store = MemoryStore(config.MEMORY_CAPACITY, config.N_STATES * 2 + 2) 
        self.memory = np.zeros(()) 
        #------- Define the optimizer------#
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=config.LR)
        
        # ------Define the loss function-----#
        self.loss_func = nn.MSELoss()
        # param = self.eval_net.state_dict()
        # self.fire('action', param )

    def update(self,  s, a, r, s_):
        self.memory_store.push( s, a, r, s_)
        if self.memory_store.memory_counter < config.MEMORY_CAPACITY:
            return 
        # update the target network every fixed steps
      
        if self.learn_step_counter % config.TARGET_NETWORK_REPLACE_FREQ == 0:
            # Assign the parameters of eval_net to target_net
            param = self.eval_net.state_dict()
            self.target_net.load_state_dict(param)
    
        self.learn_step_counter += 1
        b_s, b_a, b_r, b_s_ = self.memory_store.sample(config.BATCH_SIZE,config.N_STATES)
        b_a = b_a.to(device)
        b_s = b_s.to(device)
        b_r= b_r.to(device)
        b_s_ = b_s_.to(device)
        # calculate the Q value of state-action pair
        q_eval = self.eval_net(b_s).gather(1, b_a) # (batch_size, 1)
        #print(q_eval)
        # calculate the q value of next state
        q_next = self.target_net(b_s_).detach() # detach from computational graph, don't back propagate
        # select the maximum q value
        #print(q_next)
        # q_next.max(1) returns the max value along the axis=1 and its corresponding index
        q_target = b_r + config.GAMMA * q_next.max(1)[0].view(config.BATCH_SIZE, 1) # (batch_size, 1)
        loss = self.loss_func(q_eval, q_target)
        
        self.optimizer.zero_grad() # reset the gradient to zero
        loss.backward()
        self.optimizer.step() # execute back propagation for one step
        if self.learn_step_counter % config.ACTION_FIRE == 0:
            param = self.eval_net.state_dict()
            if device.type != "cpu":
                for name, mm in param.items():
	                param[name]= mm.cpu()
            self.fire('action', param )
   
      
