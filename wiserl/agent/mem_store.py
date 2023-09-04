import numpy as np
from torch.autograd import Variable
import torch

class MemoryStore:
    def __init__(self, capacity, size):
        self.memory = np.zeros((capacity, size)) 
        self.memory_counter =0
        self.capacity = capacity
        self.size = size
    
    def push(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_)) # horizontally stack these vectors
        index = self.memory_counter % self.capacity
        self.memory[index, :] = transition
        self.memory_counter += 1  

    def sample(self ,batch_size, N_STATES):
        sample_index = np.random.choice(self.capacity, batch_size) # randomly select some data from buffer
        # extract experiences of batch size from buffer.
        b_memory = self.memory[sample_index, :]
        # extract vectors or matrices s,a,r,s_ from batch memory and convert these to torch Variables
        # that are convenient to back propagation
        b_s = Variable(torch.FloatTensor(b_memory[:, :N_STATES]))
        # convert long int type to tensor
        b_a = Variable(torch.LongTensor(b_memory[:, N_STATES:N_STATES+1].astype(int)))
        b_r = Variable(torch.FloatTensor(b_memory[:, N_STATES+1:N_STATES+2]))
        b_s_ = Variable(torch.FloatTensor(b_memory[:, -N_STATES:]))
        return b_s,b_a,b_r,b_s_

    def sampleppo(self):
        l_s, l_a, l_r, l_s_, l_done = [], [], [], [], []
        for n in self.memory_counter:
            s, a, r, s_, done = self.memory[n]
            l_s.append(torch.tensor([s], dtype=torch.float))
            l_a.append(torch.tensor([[a]], dtype=torch.float))
            l_r.append(torch.tensor([[r]], dtype=torch.float))
            l_s_.append(torch.tensor([s_], dtype=torch.float))
            l_done.append(torch.tensor([[done]], dtype=torch.float))
        s = torch.cat(l_s, dim=0)
        a = torch.cat(l_a, dim=0)
        r = torch.cat(l_r, dim=0)
        s_ = torch.cat(l_s_, dim=0)
        done = torch.cat(l_done, dim=0)
        return s, a, r, s_ , done
    
    def put(self):
        return ray.put(self.memory)
