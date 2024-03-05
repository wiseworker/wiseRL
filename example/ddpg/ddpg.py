import torch
import numpy as np
import gym
from tqdm import tqdm
import argparse

import time
import configparser
import os
from wiserl.core.runner import Runner
from wiserl.agent.ddpg_agent.ddpg_agent import DDPGAgent
from wiserl.net.dnn_net import DNNNet
from wiserl.core.wise_rl import WiseRL
from wiserl.env import make_env
from wiserl.utils.normalization import Normalization, RewardScaling
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

# wise_rl = WiseRL()
use_ray = False
if use_ray:
    wise_rl = WiseRL()


class GymRunner(Runner):
    def __init__(self, args=None, local_rank=0):
        self.wsenv = make_env("Pendulum-v1")
        # self.env_evaluate = gym.make("CartPole-v1")  # When evaluating the policy, we need to rebuild an environment
        self.agent_name = "ddpg_agent"
        self.config= args
        self.local_rank = local_rank
        # config setting
        setattr(self.config, 'state_dim', self.wsenv.state_dim)
        setattr(self.config, 'action_dim', self.wsenv.action_dim)
        setattr(self.config, 'action_bound', float(self.wsenv.env.action_space.high[0]))
 
        # Set random seed
        seed = 0
        # self.env.seed(seed)
        # self.env.action_space.seed(seed)
        # self.env_evaluate.seed(seed)
        # self.env_evaluate.action_space.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if use_ray:
            if self.local_rank == 0:
                wise_rl.make_agent(name=self.agent_name, agent_class=DDPGAgent, config=self.config, sync=False)
            self.agent = wise_rl.get_agent(self.agent_name)
        else:
            self.agent = DDPGAgent(self.config, sync=False)

        # if self.local_rank == 0:
        #     wise_rl.make_agent(name=self.agent_name, agent_class=DDPGAgent, config=self.config, sync=False)
        # self.agent = wise_rl.get_agent(self.agent_name)
        self.state_norm = Normalization(shape=self.config.state_dim)  # Trick 2:state normalization
        
    # print("max_episode_steps={}".format(args.max_episode_steps))
    def run(self):
        num_episodes = 200
        # return_list = []
        for i in range(50):
            with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i) as pbar:
                return_list = []
                for i_episode in range(int(num_episodes/10)):
                    episode_return = 0
                    state = self.wsenv.reset()[0]
                    done = False
                    for i in range(self.config.max_episode_steps):
                        if done:
                            break
                        action = self.agent.choose_action(state)
                        next_state, reward, done, info, _ = self.wsenv.step(action)
                        episode_return += reward
                        state = next_state
                        self.agent.update(state, action, reward, next_state, done)
                    return_list.append(episode_return)
                    if (i_episode+1) % 10 == 0:
                        print("reward:", episode_return)
                        pbar.set_postfix({'episode': '%d' % (num_episodes/10 * i + i_episode+1), 'return': '%.3f' % np.mean(return_list[-10:])})
                    pbar.update(1)

    def evaluate_policy(self, arg, env, agent, state_norm):
        times = 3
        evaluate_reward = 0
        for _ in range(times):
            s = env.reset()[0]
            if arg.use_state_norm:  # During the evaluating,update=False
                s = state_norm(s, update=False)
            done = False
            episode_reward = 0
            episode_step = 0
            while not done:
                episode_step += 1
                a = agent.choose_action(s)  # We use the deterministic policy during the evaluating
                s_, r, done, info, _ = env.step(a)
                if arg.use_state_norm:
                    s_ = state_norm(s_, update=False)
                episode_reward += r
                s = s_
                if done or episode_step > arg.max_episode_steps:
                    done = True
                    break
            evaluate_reward += episode_reward

        return evaluate_reward / times

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameter Setting for DDPG")
    parser.add_argument("--max_train_steps", type=int, default=int(2e6), help=" Maximum number of training steps")
    parser.add_argument("--max_episode_steps", type=int, default=int(200), help=" Maximum number of episode steps")
    parser.add_argument("--net_dims", default=(256, 128), help="The number of neurons in hidden layers of the neural network")
    parser.add_argument("--evaluate_freq", type=float, default=5e2, help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--save_freq", type=int, default=20, help="Save frequency")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--buffer_size", type=int, default=10000, help="Buffer size")
    parser.add_argument("--hidden_dim", type=int, default=64, help="The number of neurons in hidden layers of the neural network")
    parser.add_argument("--lr_a", type=float, default=3e-4, help="Learning rate of actor")
    parser.add_argument("--lr_c", type=float, default=3e-3, help="Learning rate of critic")
    parser.add_argument("--gamma", type=float, default=0.98, help="Discount factor")
    parser.add_argument("--sigma", type=float, default=0.01, help="Gaussian factor")
    parser.add_argument("--tau", type=float, default=0.005, help="factor for soft-update")
    parser.add_argument("--use_adv_norm", type=bool, default=True, help="Trick 1:advantage normalization")
    parser.add_argument("--use_state_norm", type=bool, default=False, help="Trick 2:state normalization")
    parser.add_argument("--use_reward_norm", type=bool, default=False, help="Trick 3:reward normalization")
    parser.add_argument("--use_reward_scaling", type=bool, default=False, help="Trick 4:reward scaling")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="Trick 5: policy entropy")
    parser.add_argument("--use_lr_decay", type=bool, default=True, help="Trick 6:learning rate Decay")
    parser.add_argument("--use_grad_clip", type=bool, default=False, help="Trick 7: Gradient clip")
    parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Trick 8: orthogonal initialization")
    parser.add_argument("--set_adam_eps", type=float, default=False, help="Trick 9: set Adam epsilon=1e-5")
    parser.add_argument("--use_tanh", type=float, default=True, help="Trick 10: tanh activation function")

    args = parser.parse_args()
    if use_ray:
        runners = wise_rl.make_runner("runner", GymRunner, args, num=3)
        wise_rl.start_all_runner(runners)
    else:
        runners = GymRunner(args)
        runners.run() 

    # runners = wise_rl.make_runner("runner", GymRunner,args, num=5)
    # wise_rl.start_all_runner(runners)