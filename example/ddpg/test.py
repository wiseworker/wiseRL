import torch
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim
# from tensorboardX import SummaryWriter
import gym
import numpy as np
import random
from torch.distributions import Normal
from itertools import count

# Define hyperparameters
ENV_NAME = "Pendulum-v1" # gym envs
BATCH_SIZE = 100 # mini-batch size when sampled from buffer
MEM_CAPACTIY = 10000 # Replay buffer size
EPISODES = 200
STEPS = 200
GAMMA = 0.9 # discount factor
LEARNING_RATE = 1e-3 # learning rate of optimizer
TAU = 0.01 # update the target net parameter smoothly
RANDOM_SEED = 9527 # fix the random seed
# SAMPLE_FREQ = 2000 
NOISE_VAR = 0.1
RENDER = False
device = 'cuda' if torch.cuda.is_available() else 'cpu' # use GPU to train
print(device)

env = gym.make(ENV_NAME) 
ACTION_DIM = env.action_space.shape[0] #dim=1
STATE_DIM = env.observation_space.shape[0] # dim=3
ACTION_BOUND = env.action_space.high[0] # action interval [-2,2]
np.random.seed(RANDOM_SEED) # fix the random seed
directory = '.\\exp\\'

class ReplayBuffer():
    def __init__(self, max_size=MEM_CAPACTIY):
        self.storage = [] #empty list
        self.max_size = max_size
        self.pointer= 0
    def store_transition(self, transition):
        if len(self.storage) == self.max_size: # replace the old data
            self.storage[self.pointer] = transition
            self.pointer = (self.pointer + 1) % self.max_size # point to next position
        else:
            self.storage.append(transition)
    def sample(self, batch_size):
        # Define the array of indices for random sampling from storage
        # the size of this array equals to batch_size
        ind_array = np.random.randint(0, len(self.storage),size=batch_size)
        s, a, r, s_, d = [], [], [], [], []
        
        for i in ind_array:
            S, A, R, S_, D = self.storage[i]
            s.append(np.array(S, copy=False))
            a.append(np.array(A, copy=False))
            r.append(np.array(R, copy=False))
            s_.append(np.array(S_, copy=False))
            d.append(np.array(D, copy=False))
        return np.array(s), np.array(a), np.array(r).reshape(-1, 1), np.array(s_), np.array(d).reshape(-1, 1)

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, 30)
        self.l1.weight.data.normal_(0, 0.3)   # initialization
        self.l2 = nn.Linear(30, action_dim)
        self.l2.weight.data.normal_(0, 0.3) # initialization

        self.max_action = max_action
    def forward(self, x):
        x = F.relu(self.l1(x))
        x = self.max_action * torch.tanh(self.l2(x)) # the range of tanh is [-1, 1]
        
        return x

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic,self).__init__()
        self.l1 = nn.Linear(state_dim + action_dim, 30)
        self.l1.weight.data.normal_(0, 0.3)   # initialization
        self.l2 = nn.Linear(30, 1)
        self.l2.weight.data.normal_(0, 0.3) # initialization

    def forward(self, x, a):
        x = F.relu(self.l1(torch.cat([x, a], 1)))
        x = self.l2(x)
        return x
class DDPG(object):
    def __init__(self, state_dim, action_dim, max_action):
        # network, optimizer for actor
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LEARNING_RATE)
        
        # network, optimizer for critic
        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=LEARNING_RATE)
        # create replay buffer object
        self.replay_buffer = ReplayBuffer()
    def select_action(self, state):
        # select action based on actor network and add some noise on it for exploration
       
        state = torch.FloatTensor(state).to(device)
        action = self.actor(state).cpu().data.numpy().flatten()
        noise = np.random.normal(0, NOISE_VAR, size=ACTION_DIM).clip(
            env.action_space.low, env.action_space.high)
        action = action + noise
        return action
    def update(self):
        for i in range(EPISODES):
            s, a, r, s_, d = self.replay_buffer.sample(BATCH_SIZE)
            # transfer these tensors to GPU
            state = torch.FloatTensor(s).to(device)
            action = torch.FloatTensor(a).to(device)
            reward = torch.FloatTensor(r).to(device)
            next_state = torch.FloatTensor(s_).to(device)
            done = torch.FloatTensor(d).to(device)
            # compute the target Q value
            target_Q = self.critic_target(next_state, self.actor_target(next_state))
            target_Q = reward + (done * GAMMA * target_Q).detach()
            # Get the current Q value
            current_Q = self.critic(state, action)
            # compute critic loss by MSE
            critic_loss = F.mse_loss(current_Q, target_Q)
            # use optimizer to update the critic network
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
            
            # compute the actor loss and its gradient to update the parameters
            actor_loss = -self.critic(state,self.actor(state)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            # update the target network of actor and critic
            # zip() constructs tuple from iterable object
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(TAU * param.data + (1-TAU) * target_param.data)
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(TAU * param.data + (1-TAU) * target_param.data)
    
        def save(self):
            torch.save(self.actor.state_dict(), directory + 'actor.pth')
            torch.save(self.critic.state_dict(), directory + 'critic.pth')
        
        def load(self):
            self.actor.load_state_dict(torch.load(directory + 'actor.pth'))
            self.critic.load_state_dict(torch.load(directory + 'critic.pth'))
def main():
    agent = DDPG(STATE_DIM, ACTION_DIM, ACTION_BOUND)
    ep_r = 0
    total_step = 0
    for i in range(EPISODES):
        total_reward = 0
        step = 0
        state = env.reset()[0]
        for t in range(200):
            if RENDER == True and i > 100: env.render() # Render is unnecessary
            action = agent.select_action(state)
            
            # get the next transition by using current action
            next_state, reward, done, info,_ = env.step(action)
            # store the transition to the buffer
            agent.replay_buffer.store_transition((state, action, reward / 10, next_state, np.float64(done)))
            
            state = next_state
            if done:
                break
            step += 1
            total_reward += reward
        total_step += step+1
        print("Total T:{} Episode: \t{} Total Reward: \t{:0.2f}".format(total_step, i, total_reward))
        agent.update()
        #NOISE_VAR *= 0.99
    env.close()

if __name__ == '__main__':
    main()