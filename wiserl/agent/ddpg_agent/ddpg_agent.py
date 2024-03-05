import torch
import torch.nn as nn
import numpy as np
import copy
from  wiserl.core.agent import Agent
import torch.nn.functional as F
from wiserl.utils.replay_buffer import Offpolicy_ReplayBuffer as ReplayBuffer
from wiserl.agent.agent_utils import get_optimizer, make_actor_net, make_critic_net
from  wiserl.agent.ppo_agent.ppo_config import init_params
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch.distributions import Categorical

class DDPGAgent(Agent):
    def __init__(self, config=None, sync=True):
        self.config = config
        self.lr_a = config.lr_a   # Learning rate of actor
        self.lr_c = config.lr_c   # Learning rate of critic
        self.gamma = config.gamma # Discount factor
        self.sigma = config.sigma # Gaussian factor
        self.tau = config.tau     # soft-update factor
        self.use_lr_decay = config.use_lr_decay
        self.use_adv_norm = config.use_adv_norm
        self.set_adam_eps = config.set_adam_eps
        self.sync = sync
        self.replay_buffer = ReplayBuffer(config.buffer_size)
        self.actor = make_actor_net("ddpg_nn", config)
        self.critic = make_critic_net("ddpg_nn", config)
        self.target_actor = make_actor_net("ddpg_nn", config)
        self.target_critic = make_critic_net("ddpg_nn", config)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.target_actor.load_state_dict(self.actor.state_dict())
        # self.target_actor = copy.deepcopy(self.actor)
        # self.target_critic = copy.deepcopy(self.critic)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if self.set_adam_eps:  # Trick 9: set Adam epsilon=1e-5
            self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr_a, eps=1e-5)
            self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=self.lr_c, eps=1e-5)
        else:
            self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr_a)
            self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=self.lr_c)

    def choose_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        action = self.actor(state).item()
        # add noise
        action = action + self.sigma * np.random.randn(self.config.action_dim)
        return action

    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)

    def update(self, s, a, r, s_, done):
        self.replay_buffer.add(s, a, r, s_, done)
        if self.replay_buffer.size() < self.config.batch_size:
            return
        
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.config.batch_size)
    
        states = torch.tensor(states, dtype=torch.float).to(self.device)
        actions = torch.tensor(actions, dtype=torch.float).view(-1, 1).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float).view(-1, 1).to(self.device)
        # print("*************actions:", actions)
        next_q_values = self.target_critic(next_states, self.target_actor(next_states))
        # print("*************next_q_values:", next_q_values)
        q_targets = rewards + self.gamma * next_q_values * (1 - dones)
        # print("*************q_targets:", q_targets)
        critic_loss = torch.mean(F.mse_loss(self.critic(states, actions), q_targets))
        # print("*****critic_loss:", critic_loss)
        self.optimizer_critic.zero_grad()
        critic_loss.backward()
        self.optimizer_critic.step()
        # print("Action:", self.actor(states))
        actor_loss = -torch.mean(self.critic(states, self.actor(states)))
        # print("*****loss:", actor_loss)
        self.optimizer_actor.zero_grad()
        actor_loss.backward()
        self.optimizer_actor.step()

        self.soft_update(self.actor, self.target_actor)  # soft-update for actor
        self.soft_update(self.critic, self.target_critic)  # soft-update for critic

        #if self.sync == False:
        #    self._sync_model()

    def lr_decay(self, total_steps):
        lr_a_now = self.lr_a * (1 - total_steps / self.max_train_steps)
        lr_c_now = self.lr_c * (1 - total_steps / self.max_train_steps)
        for p in self.optimizer_actor.param_groups:
            p['lr'] = lr_a_now
        for p in self.optimizer_critic.param_groups:
            p['lr'] = lr_c_now

    def _sync_model(self):
        actor_param = self.actor.state_dict()
        if self.device.type != "cpu":
            for name, mm in actor_param.items():
                actor_param[name]= mm.cpu()
        critic_param = self.critic.state_dict()
        if self.device.type != "cpu":
            for name, mm in critic_param.items():
                critic_param[name]= mm.cpu()
        self._fire(actor_param,critic_param)
    