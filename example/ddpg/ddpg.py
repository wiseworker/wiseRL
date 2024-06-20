import torch
from wiserl.core.runner import Runner
from wiserl.agent.ddpg_agent.ddpg_agent import DDPGAgent
from wiserl.core.wise_rl import WiseRL
from gym.wrappers import RescaleAction
import matplotlib.pyplot as plt 
import argparse
import numpy as np
import gym
import time
import ray

use_ray = True
if use_ray:
    wise_rl = WiseRL()

class GymRunner(Runner):
    def __init__(self, args, local_rank=0):
        self.env = gym.make('Pendulum-v1')
        self.config = args
        # self.env = RescaleAction(self.env, min_action=-1.0, max_action=1.0)
        print("rank=", local_rank)

        # config setting
        setattr(self.config, 'state_dim', self.env.observation_space.shape[0])
        setattr(self.config, 'action_dim', self.env.action_space.shape[0])
        setattr(self.config, 'action_bound', self.env.action_space.high[0])
        self.rank = local_rank
        if use_ray:
            if self.rank == 0:
                wise_rl.make2_agent(name='ddpg_agent', agent_class=DDPGAgent, sync=True, **vars(self.config))
                self.agent = wise_rl.getAgent('ddpg_agent')    
            self.agent = wise_rl.getAgent('ddpg_agent')
        else:
            self.agent = DDPGAgent(**vars(self.config))
        
    def run(self):
        print_interval = 10
        print("====================================")
        print("Collection Experience...")
        print("====================================")
        start_time = time.time()
        ep_r = 0
        train_info = []
        plt_list = []
        for i in range(self.config.max_epoch):
            state, _ = self.env.reset()
            for t in range(self.config.max_step):
                action = self.agent.choose_action(state)
                next_state, reward, done, _, _ = self.env.step(action)
                ep_r += reward
                self.agent.update(state, action, reward, next_state, done)
                state = next_state
                if done or t == self.config.max_step-1:
                    break
            train_info.append(ep_r)
            ep_r = 0
            if i % 2 == 0: 
                print("Episode:{}--train_info:{}".format(i, np.mean(train_info[-10:])))
                plt_list.append(np.mean(train_info[-10:]))
                x = range(len(plt_list))
                plt.plot(x, plt_list)
                plt.title('Rewards-DDPG')
                plt.xlabel('Time')
                plt.ylabel('Rewards')
                plt.grid(True)
                plt.savefig('DDPG-Rewards')

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameter Setting for DDPG")
    parser.add_argument("--net_dims", default=(256, 128), help="The number of neurons in hidden layers of the neural network")
    parser.add_argument("--print_interval", type=int, default=10, help="Print_interval")
    parser.add_argument("--max_epoch", type=int, default=10000, help="max epoch for training")
    parser.add_argument("--max_step", type=int, default=199, help="max step for each episode")
    parser.add_argument("--evaluate_freq", type=float, default=5e3, help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--save_freq", type=int, default=20, help="Save frequency")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--buffer_size", type=int, default=10000, help="Buffer size")
    parser.add_argument("--lr_a", type=float, default=3e-4, help="Learning rate of actor")
    parser.add_argument("--lr_c", type=float, default=3e-4, help="Learning rate of critic")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--tau", type=float, default=0.005, help="Factor for soft-update")
 
    args = parser.parse_args()

    if use_ray:
        runners = wise_rl.makeRunner("runner", GymRunner, args, num=2)
        wise_rl.startAllRunner(runners)
    else:
        runners = GymRunner()
        runners.run()