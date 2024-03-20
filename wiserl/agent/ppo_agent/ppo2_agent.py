from wiserl.net.nn_net import CriticPPO2, ActorContinuousPPO
from wiserl.agent.agent_utils import get_optimizer, make_actor_net, make_critic_net
from wiserl.core.agent import Agent
import numpy as np
import copy
import torch
import math
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class PPO2Agent(Agent):
	def __init__(self, sync=True, **kwargs):
		super().__init__(sync)
		# Init hyperparameters for PPO agent, just like "self.gamma = opt.gamma, self.lambd = opt.lambd, ..."
		self.__dict__.update(kwargs)
		self.dvc = device
		# Choose distribution for the actor
		
		self.actor = make_actor_net("ppo2_nn", dict({"state_dim":self.state_dim,"action_dim":self.action_dim,"net_width":self.net_width}))
		
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.a_lr)

		# Build Critic
		self.critic = make_critic_net("ppo2_nn", dict({"state_dim":self.state_dim, "net_width":self.net_width}))
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.c_lr)

		# Build Trajectory holder
		self.s_hoder = np.zeros((self.T_horizon, self.state_dim),dtype=np.float32)
		self.a_hoder = np.zeros((self.T_horizon, self.action_dim),dtype=np.float32)
		self.r_hoder = np.zeros((self.T_horizon, 1),dtype=np.float32)
		self.s_next_hoder = np.zeros((self.T_horizon, self.state_dim),dtype=np.float32)
		self.logprob_a_hoder = np.zeros((self.T_horizon, self.action_dim),dtype=np.float32)
		self.done_hoder = np.zeros((self.T_horizon, 1),dtype=np.bool_)
		self.dw_hoder = np.zeros((self.T_horizon, 1),dtype=np.bool_)

	def choose_action(self, state, deterministic):
		with torch.no_grad():
			state = torch.FloatTensor(state.reshape(1, -1)).to(self.dvc)
			if deterministic:
				# only used when evaluate the policy.Making the performance more stable
				a = self.actor.deterministic_act(state)
				return a.cpu().numpy()[0], None  # action is in shape (adim, 0)
			else:
				# only used when interact with the env
				dist = self.actor.get_dist(state)
				a = dist.sample()
				a = torch.clamp(a, 0, 1)
				logprob_a = dist.log_prob(a).cpu().numpy().flatten()
				return a.cpu().numpy()[0], logprob_a # both are in shape (adim, 0)

	def update(self):
		self.entropy_coef*=self.entropy_coef_decay

		'''Prepare PyTorch data from Numpy data'''
		s = torch.from_numpy(self.s_hoder).to(self.dvc)
		a = torch.from_numpy(self.a_hoder).to(self.dvc)
		r = torch.from_numpy(self.r_hoder).to(self.dvc)
		s_next = torch.from_numpy(self.s_next_hoder).to(self.dvc)
		logprob_a = torch.from_numpy(self.logprob_a_hoder).to(self.dvc)
		done = torch.from_numpy(self.done_hoder).to(self.dvc)
		dw = torch.from_numpy(self.dw_hoder).to(self.dvc)

		''' Use TD+GAE+LongTrajectory to compute Advantage and TD target'''
		with torch.no_grad():
			vs = self.critic(s)
			vs_ = self.critic(s_next)

			'''dw for TD_target and Adv'''
			deltas = r + self.gamma * vs_ * (~dw) - vs
			deltas = deltas.cpu().flatten().numpy()
			adv = [0]

			'''done for GAE'''
			for dlt, mask in zip(deltas[::-1], done.cpu().flatten().numpy()[::-1]):
				advantage = dlt + self.gamma * self.lambd * adv[-1] * (~mask)
				adv.append(advantage)
			adv.reverse()
			adv = copy.deepcopy(adv[0:-1])
			adv = torch.tensor(adv).unsqueeze(1).float().to(self.dvc)
			td_target = adv + vs
			adv = (adv - adv.mean()) / ((adv.std()+1e-4))  #sometimes helps


		"""Slice long trajectopy into short trajectory and perform mini-batch PPO update"""
		a_optim_iter_num = int(math.ceil(s.shape[0] / self.a_optim_batch_size))
		c_optim_iter_num = int(math.ceil(s.shape[0] / self.c_optim_batch_size))
		for i in range(self.K_epochs):

			#Shuffle the trajectory, Good for training
			perm = np.arange(s.shape[0])
			np.random.shuffle(perm)
			perm = torch.LongTensor(perm).to(self.dvc)
			s, a, td_target, adv, logprob_a = \
				s[perm].clone(), a[perm].clone(), td_target[perm].clone(), adv[perm].clone(), logprob_a[perm].clone()

			'''update the actor'''
			for i in range(a_optim_iter_num):
				index = slice(i * self.a_optim_batch_size, min((i + 1) * self.a_optim_batch_size, s.shape[0]))
				distribution = self.actor.get_dist(s[index])
				dist_entropy = distribution.entropy().sum(1, keepdim=True)
				logprob_a_now = distribution.log_prob(a[index])
				ratio = torch.exp(logprob_a_now.sum(1,keepdim=True) - logprob_a[index].sum(1,keepdim=True))  # a/b == exp(log(a)-log(b))

				surr1 = ratio * adv[index]
				surr2 = torch.clamp(ratio, 1 - self.clip_rate, 1 + self.clip_rate) * adv[index]
				a_loss = -torch.min(surr1, surr2) - self.entropy_coef * dist_entropy

				self.actor_optimizer.zero_grad()
				a_loss.mean().backward()
				torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 40)
				self.actor_optimizer.step()

			'''update the critic'''
			for i in range(c_optim_iter_num):
				index = slice(i * self.c_optim_batch_size, min((i + 1) * self.c_optim_batch_size, s.shape[0]))
				c_loss = (self.critic(s[index]) - td_target[index]).pow(2).mean()
				# print("!!!:",self.critic(s[index]).shape, td_target[index].shape)
				for name,param in self.critic.named_parameters():
					if 'weight' in name:
						c_loss += param.pow(2).sum() * self.l2_reg

				self.critic_optimizer.zero_grad()
				c_loss.backward()
				self.critic_optimizer.step()

	def put_data(self, s, a, r, s_next, logprob_a, done, dw, idx):
		self.s_hoder[idx] = s
		self.a_hoder[idx] = a
		self.r_hoder[idx] = r
		self.s_next_hoder[idx] = s_next
		self.logprob_a_hoder[idx] = logprob_a
		self.done_hoder[idx] = done
		self.dw_hoder[idx] = dw

	def save(self,EnvName, timestep):
		torch.save(self.actor.state_dict(), "./model/{}_actor{}.pth".format(EnvName,timestep))
		torch.save(self.critic.state_dict(), "./model/{}_q_critic{}.pth".format(EnvName,timestep))

	def load(self,EnvName, timestep):
		self.actor.load_state_dict(torch.load("./model/{}_actor{}.pth".format(EnvName, timestep)))
		self.critic.load_state_dict(torch.load("./model/{}_q_critic{}.pth".format(EnvName, timestep)))
	
	def _sync_model(self):
		param = self.actor.state_dict()
		if device.type != "cpu":
			for name, mm in param.items():
				param[name]= mm.cpu()
		self._fire(param)
    
	def _update_model(self,param):
		self.actor.load_state_dict(param)
