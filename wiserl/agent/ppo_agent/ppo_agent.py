import torch
import torch.nn as nn
import numpy as np
from wiserl.core.agent import Agent
import torch.nn.functional as F
from wiserl.agent.agent_utils import get_optimizer, make_actor_net, make_critic_net
from wiserl.agent.ppo_agent.ppo_config import init_params
import wiserl.agent.ppo_agent.ppo_config as config 
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch.distributions import Categorical
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class PPOAgent(Agent):
    def __init__(self, sync=True, **kwargs):
        super().__init__(sync)
        self.__dict__.update(kwargs)
        # config.net_dims, config.state_dim, config.action_dims

        self.actor = make_actor_net("ppo_nn", dict({"net_dims":self.net_dims,"state_dim":self.state_dim,"action_dim":self.action_dim})) # [dis_nn, nn]
        self.critic = make_critic_net("ppo_nn", dict({"net_dims":self.net_dims,"state_dim":self.state_dim,"action_dim":self.action_dim}))   # [nn]
        if self.set_adam_eps:  # Trick 9: set Adam epsilon=1e-5
            self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr_a, eps=1e-5)
            self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=self.lr_c, eps=1e-5)
        else:
            self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr_a)
            self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=self.lr_c)
        print("ppo agent done ")

    def evaluate(self, s):  # When evaluating the policy, we select the action with the highest probability
        s = torch.unsqueeze(torch.tensor(s, dtype=torch.float), 0)
        a_prob = self.actor(s).detach().numpy().flatten()
        a = np.argmax(a_prob)
        return a

    def choose_action(self, s):
        s = torch.unsqueeze(torch.tensor(s, dtype=torch.float), 0).to(device)
        with torch.no_grad():
           a, p = self.actor.get_action(s)
           return a.item(), p 

    def compute_advantage(self, gamma, lmbda, td_delta):
        td_delta = td_delta.detach().numpy()
        advantage_list = []
        advantage = 0.0
        for delta in td_delta[::-1]:
            advantage = gamma * lmbda * advantage + delta
            advantage_list.append(advantage)
        advantage_list.reverse()
        return torch.tensor(advantage_list, dtype=torch.float)

    def update(self, replay_buffer, total_steps):
        s, a, a_logprob, r, s_, dw, done = replay_buffer.numpy_to_tensor()  # Get training data
        s = s.to(device)
        a = a.to(device)
        a_logprob = a_logprob.to(device)
        r = r.to(device)
        s_ = s_.to(device)
        dw = dw.to(device)
        """
            Calculate the advantage using GAE
            'dw=True' means dead or win, there is no next state s'
            'done=True' represents the terminal of an episode(dead or win or reaching the max_episode_steps). When calculating the adv, if done=True, gae=0
        """
        adv = []
        if self.use_reward_norm:  # Trick 3:reward normalization
            normalized_rewards = np.zeros_like(r, dtype=np.float32)
            cumulative_reward = 0.0

            for i in reversed(range(len(r))):
                cumulative_reward = r[i] + self.gamma * cumulative_reward
                normalized_rewards[i] = cumulative_reward

            normalized_rewards = (normalized_rewards - np.mean(normalized_rewards)) / (np.std(normalized_rewards) + 1e-8)
            r = torch.tensor(normalized_rewards)
        with torch.no_grad():  # adv and v_target have no gradient
            vs = self.critic(s).unsqueeze(1)
            vs_ = self.critic(s_).unsqueeze(1)
            v_target = r + self.gamma * (1.0 - dw) * vs_
            deltas = v_target - vs
            adv = self.compute_advantage(self.gamma, self.lamda, deltas)

            if self.use_adv_norm:  # Trick 1:advantage normalization
                adv = ((adv - adv.mean()) / (adv.std() + 1e-5))

        old_log_prob = torch.log(self.actor(s).gather(1, a)).detach()
        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            # Random sampling and no repetition. 'False' indicates that training will continue even if the number of samples in the last time is less than mini_batch_size
            for index in BatchSampler(SubsetRandomSampler(range(self.batch_size)), self.mini_batch_size, False):
                # a_logprob_now = dist_now.log_prob(a[index].squeeze()).view(-1, 1)  # shape(mini_batch_size X 1)
                a_logprob_now = torch.log(self.actor(s[index]).gather(1, a[index]))
                ratios = torch.exp(a_logprob_now - old_log_prob[index])  # shape(mini_batch_size X 1)

                surr1 = ratios * adv[index]  # Only calculate the gradient of 'a_logprob_now' in ratios
                surr2 = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * adv[index]
                actor_loss = torch.mean(-torch.min(surr1, surr2))  # shape(mini_batch_size X 1)- self.entropy_coef * dist_entropy
                # Update actor
                self.optimizer_actor.zero_grad()
                actor_loss.backward()
                if self.use_grad_clip:  # Trick 7: Gradient clip
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                self.optimizer_actor.step()

                v_s = self.critic(s[index]).unsqueeze(1)
                
                critic_loss = F.mse_loss(v_target[index], v_s)
                #print("!!!:",v_target[index].shape,v_s.shape)
                # Update critic
                self.optimizer_critic.zero_grad()
                critic_loss.backward()
                if self.use_grad_clip:  # Trick 7: Gradient clip
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.optimizer_critic.step()

        if self.use_lr_decay:  # Trick 6:learning rate Decay
            self.lr_decay(total_steps)

    def lr_decay(self, total_steps):
        lr_a_now = self.lr_a * (1 - total_steps / self.max_train_steps)
        lr_c_now = self.lr_c * (1 - total_steps / self.max_train_steps)
        for p in self.optimizer_actor.param_groups:
            p['lr'] = lr_a_now
        for p in self.optimizer_critic.param_groups:
            p['lr'] = lr_c_now

    def _sync_model(self):
        param = self.actor.state_dict()
        if device.type != "cpu":
            for name, mm in param.items():
                param[name]= mm.cpu()
        self._fire(param)
    
    def _update_model(self,param):
        self.actor.load_state_dict(param)
        self.actor_target.load_state_dict(param)