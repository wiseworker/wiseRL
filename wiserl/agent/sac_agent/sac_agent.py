import torch
import torch.nn as nn
from torch import optim
from torch.distributions import Normal
from wiserl.core.agent import Agent
import wiserl.agent.sac_agent.config as cfg
from wiserl.store.mem_store import ReplayBuffer
import os

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class SACAgent(Agent):
    def __init__(self, Actor, Critic, Q_net, state_dim, action_dim, config=None, sync=False):
        super(SACAgent, self).__init__(sync)

        self.policy_net = Actor(state_dim).to(device)
        self.value_net = Critic(state_dim).to(device)
        self.Q_net = Q_net(state_dim, action_dim).to(device)
        self.Target_value_net = Critic(state_dim).to(device)

        self.replay_buffer = ReplayBuffer(cfg.MEMORY_CAPACITY, state_dim, action_dim)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=cfg.learning_rate)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=cfg.learning_rate)
        self.Q_optimizer = optim.Adam(self.Q_net.parameters(), lr=cfg.learning_rate)
        self.num_transition = 0  # pointer of replay buffer
        self.num_training = 1

        self.value_criterion = nn.MSELoss()
        self.Q_criterion = nn.MSELoss()
        self.min_Val = torch.tensor(1e-7).float()
        for target_param, param in zip(self.Target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(param.data)

        os.makedirs('./SAC_model/', exist_ok=True)

    def choseAction(self, state):
        state = torch.FloatTensor(state).to(device)
        mu, log_sigma = self.policy_net(state)
        sigma = torch.exp(log_sigma)
        dist = Normal(mu, sigma)
        z = dist.sample()
        action = torch.tanh(z).detach().cpu().numpy()
        # return action.item()  # return a scalar, float32
        return action

    def get_action_log_prob(self, state):

        batch_mu, batch_log_sigma = self.policy_net(state)
        batch_sigma = torch.exp(batch_log_sigma)
        dist = Normal(batch_mu, batch_sigma)
        z = dist.sample()
        action = torch.tanh(z)
        log_prob = dist.log_prob(z) - torch.log(1 - action.pow(2) + cfg.min_Val)
        return action, log_prob, z, batch_mu, batch_log_sigma

    def update(self, s, a, r, s_, d):
        self.replay_buffer.store(s, a, r, s_, d)
        #print(self.replay_buffer.memory_counter)

        if self.num_training % 500 == 0:
            print("Training ... {} ".format(self.num_training))

        if self.replay_buffer.memory_counter >= cfg.MEMORY_CAPACITY:
            for _ in range(cfg.gradient_steps):

                buffer = self.replay_buffer.sample(cfg.batch_size)
                bn_s = buffer['state'].to(torch.float32).to(device)
                bn_a = buffer['action'].to(torch.float32).to(device)
                bn_r = buffer['reward'].to(torch.float32).to(device)
                bn_s_ = buffer['next_state'].to(torch.float32).to(device)
                bn_d = buffer['done'].to(torch.float32).to(device)
                target_value = self.Target_value_net(bn_s_)
                next_q_value = bn_r + (1 - bn_d) * cfg.gamma * target_value

                excepted_value = self.value_net(bn_s)
                excepted_Q = self.Q_net(bn_s, bn_a)

                sample_action, log_prob, z, batch_mu, batch_log_sigma = self.get_action_log_prob(bn_s)
                excepted_new_Q = self.Q_net(bn_s, sample_action)
                next_value = excepted_new_Q - log_prob

                # !!!Note that the actions are sampled according to the current policy,
                # instead of replay buffer. (From original paper)

                V_loss = self.value_criterion(excepted_value, next_value.detach())  # J_V
                V_loss = V_loss.mean()

                # Single Q_net this is different from original paper!!!
                Q_loss = self.Q_criterion(excepted_Q, next_q_value.detach())  # J_Q
                Q_loss = Q_loss.mean()

                log_policy_target = excepted_new_Q - excepted_value

                pi_loss = log_prob * (log_prob - log_policy_target).detach()
                pi_loss = pi_loss.mean()

                # mini batch gradient descent
                self.value_optimizer.zero_grad()
                V_loss.backward(retain_graph=True)
                nn.utils.clip_grad_norm_(self.value_net.parameters(), 0.5)
                self.value_optimizer.step()

                self.Q_optimizer.zero_grad()
                Q_loss.backward(retain_graph=True)
                nn.utils.clip_grad_norm_(self.Q_net.parameters(), 0.5)
                self.Q_optimizer.step()

                self.policy_optimizer.zero_grad()
                pi_loss.backward(retain_graph=True)
                nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.5)
                self.policy_optimizer.step()

                # soft update
                for target_param, param in zip(self.Target_value_net.parameters(), self.value_net.parameters()):
                    target_param.data.copy_(target_param * (1 - cfg.tau) + param * cfg.tau)

                self.num_training += 1

    def save(self):
        torch.save(self.policy_net.state_dict(), './SAC_model/policy_net.pth')
        torch.save(self.value_net.state_dict(), './SAC_model/value_net.pth')
        torch.save(self.Q_net.state_dict(), './SAC_model/Q_net.pth')
        print("====================================")
        print("Model has been saved...")
        print("====================================")

    def load(self):
        torch.load(self.policy_net.state_dict(), './SAC_model/policy_net.pth')
        torch.load(self.value_net.state_dict(), './SAC_model/value_net.pth')
        torch.load(self.Q_net.state_dict(), './SAC_model/Q_net.pth')
        print()
