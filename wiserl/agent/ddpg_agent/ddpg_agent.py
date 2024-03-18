import torch
import torch.nn as nn
from torch import optim
import numpy as np
from wiserl.agent.agent_utils import make_actor_net, make_critic_net
from wiserl.core.agent import Agent
from wiserl.store.mem_store import ReplayBuffer
import os

class DDPGAgent(Agent):
    def __init__(self, config=None, sync=True):
        super(DDPGAgent, self).__init__(sync)
        if config != None:
            self.config = config
        self.n_agents = config.n_agents
        self.n_rollout_threads = config.n_rollout_threads
        self.actor, self.actor_target = make_actor_net("ddpg_nn", config), make_actor_net("ddpg_nn", config)
        self.critic, self.critic_target = make_critic_net("ddpg_nn", config), make_critic_net("ddpg_nn", config)
        self.replay_buffer = ReplayBuffer(self.config.buffer_size, self.config.state_space, self.config.action_space, \
                                          n_rollout_threads=self.n_rollout_threads)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.config.lr_a)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.config.lr_c)
        self.num_transition = 0  # pointer of replay buffer
        self.num_training = 1
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)
        os.makedirs('./DDPG_model/', exist_ok=True)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def choose_action(self, state):
        # state = torch.tensor(state, dtype=torch.float).to(self.device)
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action = self.actor(state)
        return action.detach().cpu().numpy()[0]

    def update(self, s, a, r, s_, d):
        self.replay_buffer.store(s, a, r, s_, done=d)
        # Sample a random minibatch from the replay buffer
        buffer = self.replay_buffer.sample(self.config.batch_size)
        state_batch = buffer['state']
        action_batch = buffer['action']
        reward_batch = buffer['reward']
        next_state_batch = buffer['next_state']
        done_batch = buffer['done']

        state_batch = np.reshape(state_batch, (-1, state_batch.shape[-1]))
        action_batch = np.stack(action_batch)
        action_batch = np.reshape(action_batch, (-1, action_batch.shape[-1]))
        reward_batch = np.stack(reward_batch)
        reward_batch = np.reshape(reward_batch, (-1, 1))
        next_state_batch = np.reshape(next_state_batch, (-1, next_state_batch.shape[-1]))
        done_batch = np.stack(done_batch)
        done_batch = np.reshape(done_batch, (-1, done_batch.shape[-1]))

        state_batch = torch.tensor(state_batch, dtype=torch.float).to(self.device)
        action_batch = torch.tensor(action_batch, dtype=torch.float).view(-1, 1).to(self.device)
        reward_batch = torch.tensor(reward_batch, dtype=torch.float).view(-1, 1).to(self.device)
        next_state_batch = torch.tensor(next_state_batch, dtype=torch.float).to(self.device)
        done_batch = torch.tensor(done_batch, dtype=torch.float).view(-1, 1).to(self.device)

        # Compute Q targets
        next_action = self.actor_target(next_state_batch)
        target_Q = self.critic_target(next_state_batch, next_action)
        target_Q = reward_batch + (1 - done_batch) * self.config.gamma * target_Q.detach()

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
        self.update_target(self.actor, self.actor_target, self.config.tau)
        self.update_target(self.critic, self.critic_target, self.config.tau)

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
    