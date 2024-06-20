import torch
from wiserl.core.runner import Runner
from wiserl.agent.sarsa_agent.sarsa_agent import SARSAAgent
from wiserl.core.wise_rl import WiseRL
from gymnasium.wrappers import RescaleAction
import matplotlib.pyplot as plt
import numpy as np
import argparse
import gymnasium as gym
import time
import ray

class GymRunner(Runner):
    def __init__(self, args, local_rank=0):

        self.env = gym.make("CliffWalking-v0")
        self.config = args
        # self.env = RescaleAction(self.env, min_action=-1.0, max_action=1.0)
        print("rank=", local_rank)
        setattr(self.config, 'nrow', 4)
        setattr(self.config, 'ncol', 12)
        setattr(self.config, 'n_action', 4)
        # setattr(self.config, 'max_action', float(self.env.action_space.high[0]))
        # setattr(self.config, 'max_steps', int(self.env._max_episode_steps))

        self.rank = local_rank
        if self.rank == 0:
            wise_rl.make2_agent(name="sarsa_agent", agent_class=SARSAAgent, sync=True, **vars(self.config))
        self.agent = wise_rl.getAgent("sarsa_agent")

    def run(self):

        for i in range(self.config.num_episodes):
            state, _ = self.env.reset()
            ep_r = 0
            for t in range(self.config.max_steps):
                action = self.agent.choose_action(state)
                # next_state, reward, done, _, _ = env.step(np.float32(action))
                next_state, reward, done, _, _ = self.env.step(action)

                next_action = self.agent.choose_action(next_state)
                ep_r += reward
                self.agent.update(state, action, reward, next_state, next_action)
                state = next_state
                if done or t == 199:
                    break
            if i % 20 == 0:
                print("Episode:{}-----Reward:{}".format(i,ep_r))

if __name__ == '__main__':
    print("use GPU to train" if torch.cuda.is_available() else "use CPU to train")
    parser = argparse.ArgumentParser("Hyperparameter Setting for SARSA")
    parser.add_argument('--epsilon', type=float, default=0.1, help='hyper-parameters')
    parser.add_argument('--gamma', type=float, default=0.9, help='hyper-parameters')
    parser.add_argument('--alpha', type=float, default=0.1, help='hyper-parameters')
    parser.add_argument('--num_episodes', type=int, default=5000, help='hyper-parameters')
    parser.add_argument('--max_steps', type=int, default=50, help='max_steps')

#    ncol = 12
#    nrow = 4

    args = parser.parse_args()
    print(args)
    wise_rl = WiseRL()
    runners = wise_rl.makeRunner("runner", GymRunner, args, num=2)
    wise_rl.startAllRunner(runners)
