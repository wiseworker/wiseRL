
import torch
import torch.nn as nn
import numpy as np
from  wiserl.core.agent import Agent
from wise_rl.net.nn_net import QNet
from  wiserl.store.mem_store import MemoryStore
from wiserl.agent.agent_utils import  get_optimizer,make_net
from wiserl.agent.config import Config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DoulDqnAgent(Agent):
    def __init__(self,   args: None , sync=True):
        super().__init__(sync)
        self.config = Config(args)
        self.actor = make_net(QNet, sefl.config.net_dims, self.config.state_dim, self.config.action_dim).to(device)
        self.critic = make_net(QNet, sefl.config.net_dims, self.config.state_dim, self.config.action_dim).to(device).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)
        self.actor_optimizer = get_optimizer(self.config.optimizer, self.actor, self.config.lr_a )
        self.critic_optimizer = get_optimizer(self.config.optimizer, self.critic, self.config.lr_c )
        self.mseLoss = nn.MSELoss()

    def choose_action(self, s):
        s = torch.unsqueeze(torch.tensor(s, dtype=torch.float), 0)
        adv = self.actor(s)
		v = self.critic(s)
		q = v + (adv - torch.mean(adv, dim=-1, keepdim=True))  # Q(s,a)=V(s)+A(s,a)-mean(A(s,a))
        a = self.actor(s)..argmax().item() data.numpy().flatten()
        return a

    def learn(self, relay_buffer):
        batch_s, batch_a, batch_r, batch_s_, batch_dw = relay_buffer.sample(self.batch_size)  # Sample a batch

        # Compute the target Q
        with torch.no_grad():  # target_Q has no gradient
            Q_ = self.critic_target(batch_s_, self.actor_target(batch_s_))
            target_Q = batch_r + self.GAMMA * (1 - batch_dw) * Q_

        # Compute the current Q and the critic loss
        current_Q = self.critic(batch_s, batch_a)
        critic_loss = self.mseLoss(target_Q, current_Q)
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


    def _sync_model(self):
        param = self.eval_net.state_dict()
        if device.type != "cpu":
            for name, mm in param.items():
                param[name]= mm.cpu()
        self._fire(param)
    
    def _update_model(self,param):
        self.ev_al_net.load_state_dict(param)
        self.target_net.load_state_dict(param)

