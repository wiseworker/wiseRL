import torch
import torch.nn as nn
from torch import optim
from torch.distributions import Normal
import numpy as np
from wiserl.net.nn_net import Actor, Critic, ActorNetwork, CriticNetwork
from wiserl.agent.agent_utils import get_optimizer, make_actor_net, make_critic_net
from wiserl.core.agent import Agent
from wiserl.utils.mem_store import ReplayBuffer
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SingleAgent(Agent):
    def __init__(self, sync=True, agent_idx=0, **kwargs):
        super().__init__(sync)
        self.__dict__.update(kwargs)
    # def __init__(self, actor_dims, critic_dims, n_actions, n_agents, agent_idx, chkpt_dir='tmp/maddpg/',
    #                 alpha=0.01, beta=0.01, fc1=64,
    #                 fc2=64, gamma=0.95, tau=0.01):
        self.agent_name = 'agent_%s' % agent_idx
        self.actor = ActorNetwork(self.alpha, self.actor_dims[agent_idx], self.fc1, self.fc2, self.n_actions,
                                  chkpt_dir=self.chkpt_dir,  name=self.agent_name+'_actor')
        self.critic = CriticNetwork(self.beta, self.critic_dims,
                            self.fc1, self.fc2, self.n_agents, self.n_actions,
                            chkpt_dir=self.chkpt_dir, name=self.agent_name+'_critic')
        self.target_actor = ActorNetwork(self.alpha, self.actor_dims[agent_idx], self.fc1, self.fc2, self.n_actions,
                                        chkpt_dir=self.chkpt_dir,
                                        name=self.agent_name+'_target_actor')
        self.target_critic = CriticNetwork(self.beta, self.critic_dims,
                                            self.fc1, self.fc2, self.n_agents, self.n_actions,
                                            chkpt_dir=self.chkpt_dir,
                                            name=self.agent_name+'_target_critic')
        self.update_network_parameters(tau=1)

    def choose_action(self, observation):
        state = torch.tensor([observation], dtype=torch.float).to(self.actor.device)
        actions = self.actor.forward(state)
        noise = torch.rand(self.n_actions).to(self.actor.device)
        action = (actions + noise).detach().cpu().numpy()[0]
        return action.clip(min=0, max=1)

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        target_actor_params = self.target_actor.named_parameters()
        actor_params = self.actor.named_parameters()

        target_actor_state_dict = dict(target_actor_params)
        actor_state_dict = dict(actor_params)
        for name in actor_state_dict:
            actor_state_dict[name] = tau*actor_state_dict[name].clone() + \
                    (1-tau)*target_actor_state_dict[name].clone()

        self.target_actor.load_state_dict(actor_state_dict)

        target_critic_params = self.target_critic.named_parameters()
        critic_params = self.critic.named_parameters()

        target_critic_state_dict = dict(target_critic_params)
        critic_state_dict = dict(critic_params)
        for name in critic_state_dict:
            critic_state_dict[name] = tau*critic_state_dict[name].clone() + \
                    (1-tau)*target_critic_state_dict[name].clone()

        self.target_critic.load_state_dict(critic_state_dict)

class DDPGAgent(Agent):
    def __init__(self, sync=True, **kwargs):
        super().__init__(sync)
        self.__dict__.update(kwargs)
        self.actor, self.actor_target = make_actor_net("ddpg_nn", dict({"net_dims":self.net_dims,"state_dim":self.state_dim,"action_dim":self.action_dim,"action_bound":self.action_bound})),\
                                        make_actor_net("ddpg_nn", dict({"net_dims":self.net_dims,"state_dim":self.state_dim,"action_dim":self.action_dim,"action_bound":self.action_bound}))
        self.critic, self.critic_target = make_critic_net("ddpg_nn", dict({"net_dims":self.net_dims,"state_dim":self.state_dim,"action_dim":self.action_dim})),\
                                          make_critic_net("ddpg_nn", dict({"net_dims":self.net_dims,"state_dim":self.state_dim,"action_dim":self.action_dim}))

        self.replay_buffer = ReplayBuffer(self.buffer_size, self.state_dim, self.action_dim)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr_a)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr_c)
        
        self.num_transition = 0  # pointer of replay buffer
        self.num_training = 1

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)

        os.makedirs('./DDPG_model/', exist_ok=True)

    def choose_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        action = self.actor(state)
        return action.detach().cpu().numpy()[0]

    def update(self, s, a, r, s_, d):
        self.replay_buffer.store(s, a, r, s_, done=d)

        # Sample a random minibatch from the replay buffer
        buffer = self.replay_buffer.sample(self.batch_size)
        state_batch = buffer['state'].to(torch.float32).to(device)
        action_batch = buffer['action'].to(torch.float32).to(device)
        reward_batch = buffer['reward'].to(torch.float32).to(device)
        next_state_batch = buffer['next_state'].to(torch.float32).to(device)
        done_batch = buffer['done'].to(torch.float32).to(device)

        # Compute Q targets
        next_action = self.actor_target(next_state_batch)
        target_Q = self.critic_target(next_state_batch, next_action)
        target_Q = reward_batch + (1 - done_batch) * self.gamma * target_Q.detach()

        # Update critic
        current_Q = self.critic(state_batch, action_batch)
        critic_loss = nn.MSELoss()(current_Q, target_Q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update actor
        predicted_action = self.actor(state_batch)
        actor_loss = -self.critic(state_batch, predicted_action).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update target networks
        self.update_target(self.actor, self.actor_target, self.tau)
        self.update_target(self.critic, self.critic_target, self.tau)

        self.num_training += 1

    def update_target(self, current_model, target_model, tau):
        for param, target_param in zip(current_model.parameters(), target_model.parameters()):
            target_param.data.copy_((1 - tau) * target_param.data + tau * param.data)

    def save(self):
        torch.save(self.actor.state_dict(), './DDPG_model/actor_net.pth')
        torch.save(self.actor_target.state_dict(), './DDPG_model/actor_target_net.pth')
        torch.save(self.critic.state_dict(), './DDPG_model/critic_net.pth')
        torch.save(self.critic_target.state_dict(), './DDPG_model/critic_target_net.pth')
        print("====================================")
        print("Model has been saved.")
        print("====================================")

    def load(self):
        torch.load(self.actor.state_dict(), './DDPG_model/actor_net.pth')
        torch.load(self.actor_target.state_dict(), './DDPG_model/actor_target_net.pth')
        torch.load(self.critic.state_dict(), './DDPG_model/critic_net.pth')
        torch.load(self.critic_target.state_dict(), './DDPG_model/critic_target_net.pth')
        print("====================================")
        print("Model has been loaded.")
        print("====================================")
    
    def _sync_model(self):
        param = self.actor.state_dict()
        if device.type != "cpu":
            for name, mm in param.items():
                param[name]= mm.cpu()
        self._fire(param)
    
    def _update_model(self,param):
        self.actor.load_state_dict(param)
        self.actor_target.load_state_dict(param)