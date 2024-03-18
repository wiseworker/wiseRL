import torch
import numpy as np
import random
import collections

class SingleReplayBuffer:
    def __init__(self, config, Discrete=True):
        self.s = np.zeros((config.batch_size, config.state_dim))
        self.a = np.zeros((config.batch_size, config.action_dim))
        self.a_logprob = np.zeros((config.batch_size, 1))
        self.r = np.zeros((config.batch_size, 1))
        self.s_ = np.zeros((config.batch_size, config.state_dim))
        self.dw = np.zeros((config.batch_size, 1))
        self.done = np.zeros((config.batch_size, 1))
        self.count = 0
        self.Discrete = Discrete

    def store(self, s, a, a_logprob, r, s_, dw, done):
        self.s[self.count] = s
        self.a[self.count] = a
        if self.Discrete:
            self.a_logprob[self.count] = a_logprob
        self.r[self.count] = r
        self.s_[self.count] = s_
        self.dw[self.count] = dw
        self.done[self.count] = done
        self.count += 1

    def numpy_to_tensor(self):
        s = torch.tensor(self.s, dtype=torch.float)
        a = torch.tensor(self.a, dtype=torch.long)  # In discrete action space, 'a' needs to be torch.long
        r = torch.tensor(self.r, dtype=torch.float)
        s_ = torch.tensor(self.s_, dtype=torch.float)
        dw = torch.tensor(self.dw, dtype=torch.float)
        done = torch.tensor(self.done, dtype=torch.float)
        if self.Discrete:
            a_logprob = torch.tensor(self.a_logprob, dtype=torch.float)
            return s, a, a_logprob, r, s_, dw, done
        return s, a, r, s_, dw, done

class ReplayBuffer:
    def __init__(self, config, Discrete=True):
        self.s = np.zeros((config.batch_size, config.n_rollout_threads, config.state_space))
        self.a = np.zeros((config.batch_size, config.n_rollout_threads, 1))
        self.a_logprob = np.zeros((config.batch_size, config.n_rollout_threads, 1))
        self.r = np.zeros((config.batch_size, config.n_rollout_threads, 1))
        self.s_ = np.zeros((config.batch_size, config.n_rollout_threads, config.state_space))
        self.dw = np.zeros((config.batch_size, config.n_rollout_threads, 1))
        self.done = np.zeros((config.batch_size, config.n_rollout_threads,  1))
        self.count = 0
        self.Discrete = Discrete
        self.n_agnets = 1

    def store(self, s, a, a_logprob, r, s_, dw, done):
        for i in range(self.n_agnets):
            self.s[self.count] = s[:, i, :]
            self.a[self.count] = a[:, i, :]
            if self.Discrete:
                self.a_logprob[self.count] = a_logprob[:, i, :]
            self.s_[self.count] = s_[:, i, :]
        self.dw[self.count] = dw
        self.done[self.count] = done
        self.r[self.count] = np.reshape(r, (r.shape[0], 1))
        self.count += 1

    def numpy_to_tensor(self):
        s = torch.tensor(self.s, dtype=torch.float)
        a = torch.tensor(self.a, dtype=torch.long)  # In discrete action space, 'a' needs to be torch.long
        r = torch.tensor(self.r, dtype=torch.float)
        s_ = torch.tensor(self.s_, dtype=torch.float)
        dw = torch.tensor(self.dw, dtype=torch.float)
        done = torch.tensor(self.done, dtype=torch.float)
        if self.Discrete:
            a_logprob = torch.tensor(self.a_logprob, dtype=torch.float)
            return s, a, a_logprob, r, s_, dw, done
        return s, a, r, s_, dw, done

class Offpolicy_ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity) 

    def add(self, state, action, reward, next_state, done): 
        self.buffer.append((state, action, reward, next_state, done)) 

    def sample(self, batch_size): 
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done 

    def size(self): 
        return len(self.buffer)