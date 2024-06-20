import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from wiserl.core.agent import Agent
from wiserl.agent.agent_utils import get_optimizer, make_actor_net, make_critic_net
from wiserl.store.mem_store import ReplayBuffer
import os

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class TD3Agent(Agent):
    def __init__(self, sync=False, **kwargs):
        super().__init__(sync)
        self.__dict__.update(kwargs)

        self.actor, self.actor_target = make_actor_net("td3_nn", dict({"state_dim": self.state_dim, "action_dim": self.action_dim, "action_bound": self.action_bound})), \
                                        make_actor_net("td3_nn", dict({"state_dim": self.state_dim, "action_dim": self.action_dim, "action_bound": self.action_bound}))
        self.critic, self.critic_target = make_critic_net("td3_nn", dict({"state_dim": self.state_dim, "action_dim": self.action_dim})), \
                                          make_critic_net("td3_nn", dict({"state_dim": self.state_dim, "action_dim": self.action_dim}))

        self.replay_buffer = ReplayBuffer(self.MEMORY_CAPACITY, self.state_dim, self.action_dim)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.learning_rate)
        self.num_transition = 0  # pointer of replay buffer
        self.num_training = 1
        self.total_it = 0

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)

    def choose_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        action = self.actor(state)
        return action.detach().cpu().numpy()[0]

    def update(self, s, a, r, s_, d):
        self.replay_buffer.store(s, a, r, s_, d)

        if self.num_training % 500 == 0:
            print("Training ... {} ".format(self.num_training))

        # Sample a random minibatch from the replay buffer
        buffer = self.replay_buffer.sample(self.batch_size)
        state = buffer['state'].to(torch.float32).to(device)
        action = buffer['action'].to(torch.float32).to(device)
        reward = buffer['reward'].to(torch.float32).to(device)
        next_state = buffer['next_state'].to(torch.float32).to(device)
        done = buffer['done'].to(torch.float32).to(device)

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_state) + noise).clamp(-self.action_bound, self.action_bound).to(device)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            a = (~torch.tensor(done, dtype=torch.bool)).numpy()
            target_Q = reward + (~torch.tensor(done, dtype=torch.bool, device=device)) * self.discount * target_Q

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        # print('critic_loss:', critic_loss)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:
            # q1, q2 = self.critic(state, self.actor(state))
            # Compute actor loss
            q1, _ = self.critic(state, self.actor(state))
            actor_loss = -q1.mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            with torch.no_grad():
                # Update the frozen target models
                self.update_target(self.actor, self.actor_target, self.tau)
                self.update_target(self.critic, self.critic_target, self.tau)

        self.num_training += 1

    def update_target(self, current_model, target_model, tau):
        for param, target_param in zip(current_model.parameters(), target_model.parameters()):
            target_param.data.copy_((1 - tau) * target_param.data + tau * param.data)

    def _sync_model(self):
        param = self.policy_net.state_dict()
        if device.type != "cpu":
            for name, mm in param.items():
                param[name] = mm.cpu()
        self._fire(param)

    # def _update_model(self,param):
    #     self.actor.load_state_dict(param)
    #     self.actor_target.load_state_dict(param)