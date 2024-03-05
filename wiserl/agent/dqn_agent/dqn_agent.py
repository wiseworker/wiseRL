
import torch
import torch.nn as nn
import numpy as np
from wiserl.core.agent import Agent
from wiserl.net.nn_net import QNet
from wiserl.utils.mem_store import MemoryStore
from wiserl.agent.agent_utils import get_optimizer
from wiserl.agent.config import Config

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class DqnAgent(Agent):
    def __init__(self,  config , sync=True):
        super().__init__(sync)
        self.config = Config(config)
        self.config.add_config()
        self.actor = QNet(self.config.net_dims, self.config.state_dim, self.config.action_dim).to(device)
        self.actor_target = QNet(self.config.net_dims, self.config.state_dim, self.config.action_dim).to(device)
        self.actor_optimizer = get_optimizer(self.config.optimizer, self.actor, self.config.lr_a )
        self.mseLoss = nn.MSELoss()
        self.state_dim = self.config.state_dim
        self.action_dim = self.config.action_dim
        self.memory_store = MemoryStore(self.config.memory_capacity, self.config.state_dim * 2 + 1 + 1 + 1)
        self.memory = np.zeros(())
        self.learn_step_counter = 0 
        self.epsilon = self.config.epsilon
    
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon * self.config.epsilon_decay, self.config.epsilon_min)
    
    def choose_action(self, s):
        self.decay_epsilon()
        s = torch.unsqueeze(torch.tensor(s, dtype=torch.float), 0).to(device)
        if np.random.uniform() > self.epsilon:
            # forward feed the observation and get q value for every actions
            actions_value = self.actor(s)
            action = [np.argmax(actions_value.detach().cpu().numpy())][0]
        else:
            action = np.random.randint(0, self.action_dim)
        return action

    def update(self, s, a, r, s_, done):
        self.memory_store.push(s, a, r, s_, done)
        #print("s",s,a,r,s_,done)
        if self.memory_store.memory_counter < self.config.memory_capacity:
            return 
        if self.learn_step_counter % self.config.target_network_replace_freq == 0:
            # Assign the parameters of eval_net to target_net
            param = self.actor.state_dict()
            self.actor_target.load_state_dict(param)
    
        self.learn_step_counter += 1
        b_s, b_a, b_r, b_s_, dones = self.memory_store.sample(self.config.batch_size, self.config.state_dim)
    
        b_a = b_a.to(device)
        b_s = b_s.to(device)
        b_r= b_r.to(device)
        b_s_ = b_s_.to(device)
        dones = dones.to(device)
        # print("b_s",b_s_)
        #print("dones",dones)
        # calculate the Q value of state-action pair
        q_eval = self.actor(b_s).gather(1, b_a) # (batch_size, 1)
        # calculate the q value of next state
        q_next = self.actor_target(b_s_).detach() # detach from computational graph, don't back propagate
        # select the maximum q value
        # q_next.max(1) returns the max value along the axis=1 and its corresponding index
        q_target = b_r + self.config.gamma * q_next.max(1)[0].view(self.config.batch_size, 1) * (1 - dones)# (batch_size, 1)
        loss = self.mseLoss(q_eval, q_target)
        
        self.actor_optimizer.zero_grad() # reset the gradient to zero
        loss.backward()
        self.actor_optimizer.step() # execute back propagation for one step
        if self.sync == False:
            self._syncModel()


    def _sync_model(self):
        param = self.actor.state_dict()
        if device.type != "cpu":
            for name, mm in param.items():
                param[name]= mm.cpu()
        self._fire(param)
    
    def _update_model(self,param):
        self.actor.load_state_dict(param)
        self.actor_target.load_state_dict(param)

