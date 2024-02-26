import torch
import torch.nn as nn
import numpy as np
import copy
from  wiserl.core.agent import Agent
import torch.nn.functional as F
from wiserl.agent.agent_utils import  get_optimizer,make_actor_net,make_critic_net
from  wiserl.agent.ppo_agent.ppo_config import init_params
import wiserl.agent.ppo_agent.ppo_config  as config 
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch.distributions import Categorical
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DDPGAgent(Agent):
    def __init__(self, net_cfg,cfg=None,sync=True):
        # if cfg != None:
        #     init_params(cfg)
        self.lr_a = config.lr_a  # Learning rate of actor
        self.lr_c = config.lr_c  # Learning rate of critic
        self.gamma = config.gamma  # Discount factor
        self.lamda = config.lamda  # GAE parameter
        self.use_lr_decay = config.use_lr_decay
        self.use_adv_norm = config.use_adv_norm
        self.set_adam_eps = config.set_adam_eps
        self.sync = sync
        self.actor = make_actor_net("nn",net_cfg ,config)
        self.critic =make_critic_net("nn",net_cfg ,config)
        self.actor_target = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)

        if self.set_adam_eps:  # Trick 9: set Adam epsilon=1e-5
            self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr_a, eps=1e-5)
            self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=self.lr_c, eps=1e-5)
        else:
            self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr_a)
            self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=self.lr_c)
     
    def evaluate(self, s):  # When evaluating the policy, we select the action with the highest probability
        s = torch.unsqueeze(torch.tensor(s, dtype=torch.float), 0)
        a_prob = self.actor(s).detach().numpy().flatten()
        a = np.argmax(a_prob)
        return a

    def choose_action(self, s):
        s = torch.unsqueeze(torch.tensor(s, dtype=torch.float), 0)
        with torch.no_grad():
            dist = Categorical(probs=self.actor(s))
            a = dist.sample()
            a_logprob = dist.log_prob(a)
        return a.numpy()[0], a_logprob.numpy()[0]

    def update(self, replay_buffer, total_steps):
        batch_s, batch_a, a_logprob,batch_r, batch_s_, batch_dw ,done= replay_buffer.numpy_to_tensor()   # Sample a batch

        # Compute the target Q
        with torch.no_grad():  # target_Q has no gradient
            Q_ = self.critic_target(batch_s_, self.actor_target(batch_s_))
            target_Q = batch_r + self.GAMMA * (1 - batch_dw) * Q_

        # Compute the current Q and the critic loss
        current_Q = self.critic(batch_s, batch_a)
        critic_loss = self.MseLoss(target_Q, current_Q)
        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Freeze critic networks so you don't waste computational effort
        for params in self.critic.parameters():
            params.requires_grad = False

        # Compute the actor loss
        actor_loss = -self.critic(batch_s, self.actor(batch_s)).mean()
        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Unfreeze critic networks
        for params in self.critic.parameters():
            params.requires_grad = True

        # Softly update the target networks
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.TAU * param.data + (1 - self.TAU) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.TAU * param.data + (1 - self.TAU) * target_param.data)

        if self.use_lr_decay:  # Trick 6:learning rate Decay
            self.lr_decay(total_steps)
        if self.sync == False:
            self._sync_model()

    def lr_decay(self, total_steps):
        lr_a_now = self.lr_a * (1 - total_steps / self.max_train_steps)
        lr_c_now = self.lr_c * (1 - total_steps / self.max_train_steps)
        for p in self.optimizer_actor.param_groups:
            p['lr'] = lr_a_now
        for p in self.optimizer_critic.param_groups:
            p['lr'] = lr_c_now

    def _sync_model(self):
        actor_param = self.actor.state_dict()
        if device.type != "cpu":
            for name, mm in actor_param.items():
                actor_param[name]= mm.cpu()
        critic_param = self.critic.state_dict()
        if device.type != "cpu":
            for name, mm in critic_param.items():
                critic_param[name]= mm.cpu()
        self._fire(actor_param,critic_param)
    
    def _update_model(self,actor_param,critic_param):
        self.actor.load_state_dict(actor_param)
        self.critic.load_state_dict(critic_param)
