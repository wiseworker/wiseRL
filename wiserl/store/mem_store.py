import numpy as np
from torch.autograd import Variable
import torch
import random
import ray


class MemoryStore:
    def __init__(self, capacity, size):
        self.memory = np.zeros((capacity, size))
        self.memory_counter = 0
        self.capacity = capacity
        self.size = size

    def push(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))  # horizontally stack these vectors
        index = self.memory_counter % self.capacity
        self.memory[index, :] = transition
        self.memory_counter += 1

    def sample(self, batch_size, N_STATES):
        sample_index = np.random.choice(self.capacity, batch_size)  # randomly select some data from buffer
        # extract experiences of batch size from buffer.
        b_memory = self.memory[sample_index, :]
        # extract vectors or matrices s,a,r,s_ from batch memory and convert these to torch Variables
        # that are convenient to back propagation
        b_s = Variable(torch.FloatTensor(b_memory[:, :N_STATES]))
        # convert long int type to tensor
        b_a = Variable(torch.LongTensor(b_memory[:, N_STATES:N_STATES + 1].astype(int)))
        b_r = Variable(torch.FloatTensor(b_memory[:, N_STATES + 1:N_STATES + 2]))
        b_s_ = Variable(torch.FloatTensor(b_memory[:, -N_STATES:]))
        return b_s, b_a, b_r, b_s_

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
        return s, a, r, s_, done

    def put(self):
        return ray.put(self.memory)


class ReplayBuffer:
    def __init__(self, capacity, state_dim, action_dim=1, log_prob_dim=1, reward_dim=1):
        self.capacity = capacity
        self.memory_counter = 0

        self.state = np.zeros((capacity, state_dim))
        self.next_state = np.zeros((capacity, state_dim))
        self.action = np.zeros((capacity, action_dim))
        self.log_prob = np.zeros((capacity, log_prob_dim))
        self.reward = np.zeros((capacity, reward_dim))
        self.done = np.zeros((capacity, 1))

    def store(self, s, a, r, s_, log_prob=None, done=None):
        index = self.memory_counter % self.capacity

        self.state[index] = s
        self.action[index] = a
        self.reward[index] = r
        self.next_state[index] = s_

        if log_prob is not None:
            self.log_prob[index] = log_prob

        if done is not None:
            self.done[index] = done

        self.memory_counter += 1

    def sample(self, batch_size):
        sample_index = np.random.choice(min(self.capacity, self.memory_counter), batch_size)
        return {
            'state': torch.tensor(self.state[sample_index]),
            'action': torch.tensor(self.action[sample_index]),
            'reward': torch.tensor(self.reward[sample_index]),
            'next_state': torch.tensor(self.next_state[sample_index]),
            'done': torch.tensor(self.done[sample_index]),
            'log_prob': torch.tensor(self.log_prob[sample_index])
        }


class MAReplayBuffer:
    def __init__(self, args):
        self.N = args.N
        self.obs_dim = args.obs_dim
        self.state_dim = args.state_dim
        self.episode_limit = args.episode_limit
        self.batch_size = args.batch_size
        self.episode_num = 0
        self.buffer = None
        self.reset_buffer()
        # create a buffer (dictionary)

    def reset_buffer(self):
        self.buffer = {'obs_n': np.empty([self.batch_size, self.episode_limit, self.N, self.obs_dim]),
                       's': np.empty([self.batch_size, self.episode_limit, self.state_dim]),
                       'v_n': np.empty([self.batch_size, self.episode_limit + 1, self.N]),
                       'a_n': np.empty([self.batch_size, self.episode_limit, self.N]),
                       'a_logprob_n': np.empty([self.batch_size, self.episode_limit, self.N]),
                       'r_n': np.empty([self.batch_size, self.episode_limit, self.N]),
                       'done_n': np.empty([self.batch_size, self.episode_limit, self.N])
                       }
        self.episode_num = 0

    def store_transition(self, episode_step, obs_n, s, v_n, a_n, a_logprob_n, r_n, done_n):
        self.buffer['obs_n'][self.episode_num][episode_step] = obs_n
        self.buffer['s'][self.episode_num][episode_step] = s
        self.buffer['v_n'][self.episode_num][episode_step] = v_n
        self.buffer['a_n'][self.episode_num][episode_step] = a_n
        self.buffer['a_logprob_n'][self.episode_num][episode_step] = a_logprob_n
        self.buffer['r_n'][self.episode_num][episode_step] = r_n
        self.buffer['done_n'][self.episode_num][episode_step] = done_n

    def store_last_value(self, episode_step, v_n):
        self.buffer['v_n'][self.episode_num][episode_step] = v_n
        self.episode_num += 1

    def get_training_data(self):
        batch = {}
        for key in self.buffer.keys():
            if key == 'a_n':
                batch[key] = torch.tensor(self.buffer[key], dtype=torch.long)
            else:
                batch[key] = torch.tensor(self.buffer[key], dtype=torch.float32)
        return batch