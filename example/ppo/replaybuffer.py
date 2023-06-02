import torch
import numpy as np
import example.ppo.config as config
class ReplayBuffer:
    def __init__(self):
        self.s = np.zeros((config.BATCH_SIZE, config.N_STATE))
        self.a = np.zeros((config.BATCH_SIZE, 1))
        self.a_logprob = np.zeros((config.BATCH_SIZE, 1))
        self.r = np.zeros((config.BATCH_SIZE, 1))
        self.s_ = np.zeros((config.BATCH_SIZE, config.N_STATE))
        self.dw = np.zeros((config.BATCH_SIZE, 1))
        self.done = np.zeros((config.BATCH_SIZE, 1))
        self.count = 0

    def store(self, s, a, a_logprob, r, s_, dw, done):
        self.s[self.count] = s
        self.a[self.count] = a
        self.a_logprob[self.count] = a_logprob
        self.r[self.count] = r
        self.s_[self.count] = s_
        self.dw[self.count] = dw
        self.done[self.count] = done
        self.count += 1

    def numpy_to_tensor(self):
        s = torch.tensor(self.s, dtype=torch.float)
        a = torch.tensor(self.a, dtype=torch.long)  # In discrete action space, 'a' needs to be torch.long
        a_logprob = torch.tensor(self.a_logprob, dtype=torch.float)
        r = torch.tensor(self.r, dtype=torch.float)
        s_ = torch.tensor(self.s_, dtype=torch.float)
        dw = torch.tensor(self.dw, dtype=torch.float)
        done = torch.tensor(self.done, dtype=torch.float)
        return s, a, a_logprob, r, s_, dw, done