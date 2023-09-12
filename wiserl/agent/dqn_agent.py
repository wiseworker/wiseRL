
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import ray
from  wiserl.core.agent import Agent
from  wiserl.store.mem_store import MemoryStore
import wiserl.agent.config as cfg

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DQNAgent(Agent):
    def __init__(self,net_class,n_states,n_actions,config=None, sync=True):
        super().__init__(sync)
        self.config = cfg
        if config != None:
            self.config=config
        self.n_actions = n_actions
        self.n_states = n_states
        self.eval_net, self.target_net = net_class(n_states,n_actions).to(device), net_class(n_states,n_actions).to(device)
        self.learn_step_counter = 0 
        self.memory_store = MemoryStore(8000,n_states * 2 + 2) 
        self.memory = np.zeros(()) 
        #------- Define the optimizer------#
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=self.config.LR)
        
        # ------Define the loss function-----#
        self.loss_func = nn.MSELoss()
        # param = self.eval_net.state_dict()
        # self.fire('action', param )

    def update(self,  s, a, r, s_):
        self.memory_store.push( s, a, r, s_)
        if self.memory_store.memory_counter < self.config.MEMORY_CAPACITY:
            return 
        # update the target network every fixed steps
      
        if self.learn_step_counter % self.config.TARGET_NETWORK_REPLACE_FREQ == 0:
            # Assign the parameters of eval_net to target_net
            param = self.eval_net.state_dict()
            self.target_net.load_state_dict(param)
    
        self.learn_step_counter += 1
        b_s, b_a, b_r, b_s_ = self.memory_store.sample(self.config.BATCH_SIZE,self.n_states)
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
        q_target = b_r + self.config.GAMMA * q_next.max(1)[0].view(self.config.BATCH_SIZE, 1) # (batch_size, 1)
        loss = self.loss_func(q_eval, q_target)
        
        self.optimizer.zero_grad() # reset the gradient to zero
        loss.backward()
        self.optimizer.step() # execute back propagation for one step
        if self.sync == False:
            self._syncModel()
        
   
    def choseAction(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x).to(device), 0) # add 1 dimension to input state x
        if np.random.uniform() < self.config.EPSILON:   
            actions_value = self.eval_net.forward(x).cpu()
            action = torch.max(actions_value, 1)[1].data.numpy()
            action = action[0] #if self.config.ENV_A_SHAPE == 0 else action.reshape(self.config.ENV_A_SHAPE)  # return the argmax index
        else:   # random
            action = np.random.randint(0, self.n_actions)
            action = action #if self.config.ENV_A_SHAPE == 0 else action.reshape(self.config.ENV_A_SHAPE)
        return action    

    def _syncModel(self):
        param = self.eval_net.state_dict()
        if device.type != "cpu":
            for name, mm in param.items():
                param[name]= mm.cpu()
        self._fire(param)
    
    def _updateModel(self,param):
        self.eval_net.load_state_dict(param)
        self.target_net.load_state_dict(param)

