# -- coding: utf-8 --
import gym
import sys
import os

# 获取当前文件所在目录的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# 将 util 文件夹的父目录（project_root）添加到 sys.path
sys.path.append(current_dir)

from envs.mpe.MPE_env import MPEEnv
from envs.env_wrappers import SubprocVecEnv, DummyVecEnv

class WsEnv(object):
    def __init__(self,env,name,action_dim=5,state_dim=6, discrete=False):
        self.name = name
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.discrete = True 
        self.env = env
    
    def step(self,s):
        return self.env.step(s)

    def reset(self):
        return self.env.reset()

def make_train_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "MPE":
                env = MPEEnv(all_args)
            else:
                print("Can not support the " +
                      all_args.env_name + "environment.")
                raise NotImplementedError
            env.seed(all_args.seed + rank * 1000)
            return env
        return init_env
    if all_args.n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])

def make_env(name, render_mode=None):
    from pettingzoo.mpe import simple_spread_v3, simple_tag_v3, simple_adversary_v3
    from pettingzoo.mpe import simple_v3
    if name == "CartPole-v1":
        env = gym.make("CartPole-v1")
        action_dim = env.action_space.n  # 2 actions
        state_dim = env.observation_space.shape[0]  # 4 states
        return WsEnv(env,name,action_dim,state_dim)
    elif name == "Acrobot-v1":
        env = gym.make("Acrobot-v1")
        action_dim = env.action_space.n  # 3 actions
        state_dim = env.observation_space.shape[0]  # 6 states
        return WsEnv(env,name,action_dim,state_dim)
    elif name == 'simple_spread_v3':
        env = simple_spread_v3.parallel_env(N=3, render_mode=render_mode, continuous_actions=False)
        return WsEnv(env, name)
    elif name == 'simple_tag_v3':
        env = simple_tag_v3.parallel_env(num_good=1, num_adversaries=3, num_obstacles=2, max_cycles=35, continuous_actions=True)
        return WsEnv(env, name)
    elif name == 'simple_adversary_v3':
        env = simple_adversary_v3.parallel_env(N=2, max_cycles=35, continuous_actions=True)
        return WsEnv(env, name)
    elif name == 'simple_v3':
        env = simple_v3.parallel_env(render_mode=render_mode, max_cycles=35, continuous_actions=True)
        action_dim = 5
        state_dim = 4
        return WsEnv(env,name,action_dim,state_dim)
    elif name == 'Pendulum-v1':
        env = gym.make("Pendulum-v1")
        action_dim = env.action_space.shape[0]
        state_dim = env.observation_space.shape[0]
        return WsEnv(env,name,action_dim,state_dim)
    else:
        raise Exception("No such environment.")



def save_gym_state(env,i):
    next_state = env.render(mode='rgb_array')
    plt.imsave('./shy/state/state{}.png'.format(i),preprocess_frame(next_state)) 
    next_state = plt.imread('./state/state{}.png'.format(i))
    return next_state


def preprocess_frame(self,frame):
    gray = rgb2gray(frame)
    #crop the frame
    #cropped_frame = gray[:,:]
    normalized_frame = gray/255.0
    preprocessed_frame = transform.resize(normalized_frame, [84,84])
    return preprocessed_frame