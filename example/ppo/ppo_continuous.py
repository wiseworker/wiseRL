import torch
from wiserl.core.runner import Runner
from wiserl.agent.ppo_agent.ppo2_agent import PPO2Agent as PPO_agent
from wiserl.core.wise_rl import WiseRL
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import argparse
import gymnasium as gym
import time
import ray
import os, shutil
import numpy as np
import copy
import math

use_ray = True
if use_ray:
    wise_rl = WiseRL()

def str2bool(v):
    '''transfer str to bool for argparse'''
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'True','true','TRUE', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'False','false','FALSE', 'f', 'n', '0'):
        return False
    else:
        print('Wrong Input.')
        raise

class GymRunner(Runner):
    def __init__(self, args, local_rank=0):
        self.env = gym.make('Pendulum-v1')
        self.eval_env = gym.make('Pendulum-v1')
        self.config = args
        
        # self.env = RescaleAction(self.env, min_action=-1.0, max_action=1.0)
        print("rank=", local_rank)

        # config setting
        setattr(self.config, 'state_dim', self.env.observation_space.shape[0])
        setattr(self.config, 'action_dim', self.env.action_space.shape[0])
        setattr(self.config, 'max_action', float(self.env.action_space.high[0]))
        setattr(self.config, 'max_steps', float(self.env._max_episode_steps))
       
        # Seed Everything
        self.env_seed = self.config.seed
        torch.manual_seed(self.config.seed)
        torch.cuda.manual_seed(self.config.seed)
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
        print("Random Seed: {}".format(self.config.seed))

        self.rank = local_rank
        if use_ray:
            if self.rank == 0:
                wise_rl.make2_agent(name='ppo_agent', agent_class=PPO_agent, sync=True, **vars(self.config))
                self.agent = wise_rl.getAgent('ppo_agent')    
            self.agent = wise_rl.getAgent('ppo_agent')
        else:
            self.agent = PPO_agent(sync=True, **vars(self.config))
        
    def run(self):
        EnvName = ['Pendulum-v1']
        BrifEnvName = ['PV1']
        train_info = []
        plt_list = []
        if self.config.render:
            while True:
                ep_r = evaluate_policy(self.env, self.agent, self.config.max_action, 1)
                print(f'Env:{EnvName[self.config.EnvIdex]}, Episode Reward:{ep_r}')
        else:
            traj_lenth, total_steps, episode_num = 0, 0, 0
            while total_steps < self.config.Max_train_steps:
                episode_num += 1
                s, info = self.env.reset(seed=self.env_seed) # Do not use self.config.seed directly, or it can overfit to self.config.seed
                self.env_seed += 1
                done = False
                ep_r = 0
                '''Interact & trian'''
                while not done:
                    '''Interact with Env'''
                    a, logprob_a = self.agent.choose_action(s, deterministic=False) # use stochastic when training
                    act = self.Action_adapter(a,self.config.max_action) #[0,1] to [-max,max]
                    s_next, r, dw, tr, info = self.env.step(act) # dw: dead&win; tr: truncated
                    r = self.Reward_adapter(r, self.config.EnvIdex)
                    done = (dw or tr)
                    ep_r += r
                    '''Store the current transition'''
                    self.agent.put_data(s, a, r, s_next, logprob_a, done, dw, idx = traj_lenth)
                    s = s_next

                    traj_lenth += 1
                    total_steps += 1

                    '''Update if its time'''
                    if traj_lenth % self.config.T_horizon == 0:
                        self.agent.update()
                        traj_lenth = 0

                    '''Record & log'''
                    if total_steps % self.config.eval_interval == 0:
                        score = self.evaluate_policy(self.eval_env, self.agent, self.config.max_action, turns=3) # evaluate the policy for 3 times, and get averaged result
                        if self.config.write: writer.add_scalar('ep_r', score, global_step=total_steps)
                        print('EnvName:', EnvName[self.config.EnvIdex],'seed:',self.config.seed,'steps: {}k'.format(int(total_steps/1000)),'score:', score)

                    '''Save model'''
                    if total_steps % self.config.save_interval==0:
                        self.agent.save(BrifEnvName[self.config.EnvIdex], int(total_steps/1000))
                train_info.append(ep_r)
                if episode_num % 2 == 0: 
                    print("Episode:{}--train_info:{}".format(episode_num, np.mean(train_info[-10:])))
                    plt_list.append(np.mean(train_info[-10:]))
                    x = range(len(plt_list))
                    plt.plot(x, plt_list)
                    plt.title('Rewards-PPO')
                    plt.xlabel('Time')
                    plt.ylabel('Rewards')
                    plt.grid(True)
                    plt.savefig('PPO-Rewards')
            self.env.close()
            self.eval_env.close()

    def Action_adapter(self, a,max_action):
        #from [0,1] to [-max,max]
        return  2*(a-0.5)*max_action

    def Reward_adapter(self, r, EnvIdex):
        # For BipedalWalker
        if EnvIdex == 0 or EnvIdex == 1:
            if r <= -100: r = -1
        # For Pendulum-v0
        elif EnvIdex == 3:
            r = (r + 8) / 8
        return r

    def evaluate_policy(self, env, agent, max_action, turns):
        total_scores = 0
        for j in range(turns):
            s, info = env.reset()
            done = False
            while not done:
                a, logprob_a = agent.choose_action(s, deterministic=True) # Take deterministic actions when evaluation
                act = self.Action_adapter(a, max_action)  # [0,1] to [-max,max]
                s_next, r, dw, tr, info = env.step(act)
                done = (dw or tr)

                total_scores += r
                s = s_next

        return total_scores/turns

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameter Setting for PPO-Continuous")
    parser.add_argument('--dvc', type=str, default='cuda', help='running device: cuda or cpu')
    parser.add_argument('--EnvIdex', type=int, default=0, help='PV1, Lch_Cv2, Humanv4, HCv4, BWv3, BWHv3')
    parser.add_argument('--write', type=str2bool, default=False, help='Use SummaryWriter to record the training')
    parser.add_argument('--render', type=str2bool, default=False, help='Render or Not')
    parser.add_argument('--Loadmodel', type=str2bool, default=False, help='Load pretrained model or Not')
    parser.add_argument('--ModelIdex', type=int, default=100, help='which model to load')

    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--T_horizon', type=int, default=2048, help='lenth of long trajectory')
    parser.add_argument('--Distribution', type=str, default='Beta', help='Should be one of Beta ; GS_ms  ;  GS_m')
    parser.add_argument('--Max_train_steps', type=int, default=int(5e7), help='Max training steps')
    parser.add_argument('--save_interval', type=int, default=int(5e5), help='Model saving interval, in steps.')
    parser.add_argument('--eval_interval', type=int, default=int(5e3), help='Model evaluating interval, in steps.')

    parser.add_argument('--gamma', type=float, default=0.99, help='Discounted Factor')
    parser.add_argument('--lambd', type=float, default=0.95, help='GAE Factor')
    parser.add_argument('--clip_rate', type=float, default=0.2, help='PPO Clip rate')
    parser.add_argument('--K_epochs', type=int, default=10, help='PPO update times')
    parser.add_argument('--net_width', type=int, default=150, help='Hidden net width')
    parser.add_argument('--a_lr', type=float, default=2e-4, help='Learning rate of actor')
    parser.add_argument('--c_lr', type=float, default=2e-4, help='Learning rate of critic')
    parser.add_argument('--l2_reg', type=float, default=1e-3, help='L2 regulization coefficient for Critic')
    parser.add_argument('--a_optim_batch_size', type=int, default=64, help='lenth of sliced trajectory of actor')
    parser.add_argument('--c_optim_batch_size', type=int, default=64, help='lenth of sliced trajectory of critic')
    parser.add_argument('--entropy_coef', type=float, default=1e-3, help='Entropy coefficient of Actor')
    parser.add_argument('--entropy_coef_decay', type=float, default=0.99, help='Decay rate of entropy_coef')
    args = parser.parse_args()
    args.dvc = torch.device(args.dvc) # from str to torch.device
    print(args)

    if use_ray:
        runners = wise_rl.makeRunner("runner", GymRunner, args, num=1)
        wise_rl.startAllRunner(runners)
    else:
        runners = GymRunner(args)
        runners.run()