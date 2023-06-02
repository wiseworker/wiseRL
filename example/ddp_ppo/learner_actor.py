
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import ray
import os
import time
from example.ddp_ppo.net import Actor, Critic
import example.ddp_ppo.config as config
from wiseRL.core.wise_rl  import WiseRL
from wiseRL.core.learner import Learner
from torch.distributions import Categorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch.nn.parallel import DistributedDataParallel as DDP

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device",device)
class LearnerActor(Learner):
    def __init__(self):
        self.batch_size = config.BATCH_SIZE
        self.mini_batch_size = config.MINI_BATCH_SIZE
        self.max_train_steps = config.MAX_TRAIN_STEPS
        self.lr_a = config.LR_A  # Learning rate of actor
        self.lr_c = config.LR_C  # Learning rate of critic
        self.gamma = config.GAMMA  # Discount factor
        self.lamda = config.LAMDA  # GAE parameter
        self.epsilon = config.EPSILON  # PPO clip parameter
        self.K_epochs = config.K_EPOCHS  # PPO parameter
        self.entropy_coef = config.ENTROPY_COEF  # Entropy coefficient
        self.set_adam_eps = config.SET_ADAM_EPS
        self.use_grad_clip = config.USE_GRAD_CLIP
        self.use_lr_decay = config.USE_LR_DECAY
        self.use_adv_norm = config.USE_ADV_NORM

       
    
    def start(self):
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '12345'
        print("==================rank", self.rank)
        torch.distributed.init_process_group(backend='gloo',rank=self.rank,
                                         world_size=3)
        torch.cuda.set_device(0)
        a_net = Actor().to(device)
        c_net = Critic().to(device)
        self.actor = DDP(a_net,
            device_ids=[0],
            output_device=0).to(device)
        self.critic = DDP(c_net,
            device_ids=[0],
            output_device=0).to(device)
        if self.set_adam_eps:  # Trick 9: set Adam epsilon=1e-5
            self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr_a, eps=1e-5)
            self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=self.lr_c, eps=1e-5)
        else:
            self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr_a)
            self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=self.lr_c)

    def evaluate(self, s):  # When evaluating the policy, we select the action with the highest probability
        s = torch.unsqueeze(torch.tensor(s, dtype=torch.float), 0).to(device)
        a_prob = self.actor(s).detach().numpy().flatten()
        a = np.argmax(a_prob)
        return a

    def choseAction(self, s):
        s = torch.unsqueeze(torch.tensor(s, dtype=torch.float), 0).to(device)
        with torch.no_grad():
            dist = Categorical(probs=self.actor(s))
            a = dist.sample()
            a_logprob = dist.log_prob(a)
        return a.cpu().numpy()[0], a_logprob.cpu().numpy()[0]

    def update(self, replay_buffer, total_steps):
        #print("buffer_id2",buffer_id)
        #replay_buffer=ray.get(buffer_id)
        s, a, a_logprob, r, s_, dw, done = replay_buffer.numpy_to_tensor()  # Get training data
        """
            Calculate the advantage using GAE
            'dw=True' means dead or win, there is no next state s'
            'done=True' represents the terminal of an episode(dead or win or reaching the max_episode_steps). When calculating the adv, if done=True, gae=0
        """
        adv = []
        gae = 0
        s = s.to(device)
        s_ = s_.to(device)
        r = r.to(device)
        dw = dw.to(device)
        a = a.to(device)
        a_logprob = a_logprob.to(device)
        with torch.no_grad():  # adv and v_target have no gradient
            vs = self.critic(s)
            vs_ = self.critic(s_)
            deltas = r + self.gamma * (1.0 - dw) * vs_ - vs
            for delta, d in zip(reversed(deltas.cpu().flatten().numpy()), reversed(done.flatten().numpy())):
                gae = delta + self.gamma * self.lamda * gae * (1.0 - d)
                adv.insert(0, gae)
            adv = torch.tensor(adv, dtype=torch.float).view(-1, 1).to(device)
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
        
       

    def lr_decay(self, total_steps):
        lr_a_now = self.lr_a * (1 - total_steps / self.max_train_steps)
        lr_c_now = self.lr_c * (1 - total_steps / self.max_train_steps)
        for p in self.optimizer_actor.param_groups:
            p['lr'] = lr_a_now
        for p in self.optimizer_critic.param_groups:
            p['lr'] = lr_c_now