import torch
import torch.nn as nn
import numpy as np
from  wiserl.core.agent import Agent
import torch.nn.functional as F
from wiserl.agent.agent_utils import  get_optimizer,make_actor_net,make_critic_net
from  wiserl.agent.ppo_agent.ppo_config import init_params
import wiserl.agent.ppo_agent.ppo_config  as config 
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch.distributions import Categorical
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class PPO2Agent(Agent):
    def __init__(self, net_cfg,cfg=None,sync=True):
        # if cfg != None:
        #     init_params(cfg)

        self.batch_size = config.batch_size
        self.mini_batch_size = config.mini_batch_size
        self.max_train_steps = config.max_train_steps
        self.lr_a = config.lr_a  # Learning rate of actor
        self.lr_c = config.lr_c  # Learning rate of critic
        self.gamma = config.gamma  # Discount factor
        self.lamda = config.lamda  # GAE parameter
        self.epsilon = config.epsilon  # PPO clip parameter
        self.K_epochs = config.K_epochs  # PPO parameter
        self.entropy_coef = config.entropy_coef  # Entropy coefficient
        self.set_adam_eps = config.set_adam_eps
        self.use_grad_clip = config.use_grad_clip
        self.use_lr_decay = config.use_lr_decay
        self.use_adv_norm = config.use_adv_norm
        self.sync = sync

        self.actor = make_actor_net("nn",net_cfg ,config)
        self.critic =make_critic_net("nn",net_cfg ,config)
        if self.set_adam_eps:  # Trick 9: set Adam epsilon=1e-5
            self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr_a, eps=1e-5)
            self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=self.lr_c, eps=1e-5)
        else:
            self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr_a)
            self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=self.lr_c)
        print("ppo2 agent done ")

        
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
        s, a, a_logprob, r, s_, dw, done = replay_buffer.numpy_to_tensor()  # Get training data
        """
            Calculate the advantage using GAE
            'dw=True' means dead or win, there is no next state s'
            'done=True' represents the terminal of an episode(dead or win or reaching the max_episode_steps). When calculating the adv, if done=True, gae=0
        """
        adv = []
        gae = 0
        with torch.no_grad():  # adv and v_target have no gradient
            vs = self.critic(s)
            vs_ = self.critic(s_)
            deltas = r + self.gamma * (1.0 - dw) * vs_ - vs
            for delta, d in zip(reversed(deltas.flatten().numpy()), reversed(done.flatten().numpy())):
                gae = delta + self.gamma * self.lamda * gae * (1.0 - d)
                adv.insert(0, gae)
            adv = torch.tensor(adv, dtype=torch.float).view(-1, 1)
            v_target = adv + vs
            if self.use_adv_norm:  # Trick 1:advantage normalization
                adv = ((adv - adv.mean()) / (adv.std() + 1e-5))

        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            # Random sampling and no repetition. 'False' indicates that training will continue even if the number of samples in the last time is less than mini_batch_size
            for index in BatchSampler(SubsetRandomSampler(range(self.batch_size)), self.mini_batch_size, False):
                dist_now = Categorical(probs=self.actor(s[index]))
                dist_entropy = dist_now.entropy().view(-1, 1)  # shape(mini_batch_size X 1)
                a_logprob_now = dist_now.log_prob(a[index].squeeze()).view(-1, 1)  # shape(mini_batch_size X 1)
                # a/b=exp(log(a)-log(b))
                ratios = torch.exp(a_logprob_now - a_logprob[index])  # shape(mini_batch_size X 1)

                surr1 = ratios * adv[index]  # Only calculate the gradient of 'a_logprob_now' in ratios
                surr2 = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * adv[index]
                actor_loss = -torch.min(surr1, surr2) - self.entropy_coef * dist_entropy  # shape(mini_batch_size X 1)
               
                # Update actor
                self.optimizer_actor.zero_grad()
                actor_loss.mean().backward()
                if self.use_grad_clip:  # Trick 7: Gradient clip
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                self.optimizer_actor.step()

                v_s = self.critic(s[index])
                critic_loss = F.mse_loss(v_target[index], v_s)
                # Update critic
                self.optimizer_critic.zero_grad()
                critic_loss.backward()
                if self.use_grad_clip:  # Trick 7: Gradient clip
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.optimizer_critic.step()

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
