import torch
import numpy as np
import gym
import argparse
from wiserl.utils.replay_buffer import ReplayBuffer
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

wise_rl = WiseRL()

def evaluate_policy(args, env, agent, state_norm):
    times = 3
    evaluate_reward = 0
    for _ in range(times):
        s = env.reset()
        if args.use_state_norm:  # During the evaluating,update=False
            s = state_norm(s, update=False)
        done = False
        episode_reward = 0
        while not done:
            a, a_logprob  = agent.choose_action(s)  # We use the deterministic policy during the evaluating
            s_, r, done, _ = env.step(a)
            if args.use_state_norm:
                s_ = state_norm(s_, update=False)
            episode_reward += r
            s = s_
        evaluate_reward += episode_reward

    return evaluate_reward / times


class GymRunner(Runner):
    def __init__(self, args=None, local_rank=0):
        self.env = gym.make("Pendulum-v1")
        self.env_evaluate = gym.make("Pendulum-v1")  # When evaluating the policy, we need to rebuild an environment
        self.cfg= args
        self.agent_name = "ddpg_agent"
        # Set random seed
        seed = 0
        # self.env.seed(seed)
        # self.env.action_space.seed(seed)
        # self.env_evaluate.seed(seed)
        # self.env_evaluate.action_space.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]
        max_action = float(self.env.action_space.high[0])
        self.net_cfg = {
            "net":  "DNNNet",
            "action_dim": state_dim,
            "state_dim": state_dim
        }
        conf = configparser.ConfigParser()
        conf.read("/workspace/zs/wiseRL/example/ppo/config.ini")
        self.config = conf['ppo']
        self.cfg.state_dim = state_dim
        self.cfg.action_dim = action_dim
        self.cfg.max_episode_steps =self.env._max_episode_steps  # Maximum number of steps per episode
        if local_rank == 0:
            wise_rl.make_agent(name=self.agent_name, agent_class=DDPGAgent, net_cfg=self.net_cfg,cfg=args,sync=False)
            self.agent = wise_rl.get_agent(self.agent_name)
        else:
            self.agent = wise_rl.get_agent(self.agent_name)
        self.state_norm = Normalization(shape=self.net_cfg['state_dim'])  # Trick 2:state normalization


    # args.state_dim = env.observation_space.shape[0]
    # args.action_dim = env.action_space.n
    # args.max_episode_steps = env._max_episode_steps  # Maximum number of steps per episode
    # print("env={}".format(env_name))
    # print("state_dim={}".format(args.state_dim))
    # print("action_dim={}".format(args.action_dim))
    # print("max_episode_steps={}".format(args.max_episode_steps))
    def run(self):
        evaluate_num = 0  # Record the number of evaluations
        evaluate_rewards = []  # Record the rewards during the evaluating
        total_steps = 0  # Record the total steps during the training
        batch_size = 2048  if self.config["batch_size"] == None else int(self.config["batch_size"])
        replay_buffer = ReplayBuffer(batch_size,self.net_cfg['state_dim'])
        if self.cfg.use_reward_norm:  # Trick 3:reward normalization
            reward_norm = Normalization(shape=1)
        elif self.cfg.use_reward_scaling:  # Trick 4:reward scaling
            reward_scaling = RewardScaling(shape=1, gamma=self.cfg.gamma)

        while total_steps < args.max_train_steps:
            s = self.env.reset()[0]
            if self.cfg.use_state_norm:
                s = self.state_norm(s)
            if self.cfg.use_reward_scaling:
                reward_scaling.reset()
            episode_steps = 0
            done = False
            while not done:
                episode_steps += 1
                print("s",s)
                a, a_logprob = self.agent.choose_action(s)  # Action and the corresponding log probability
                print("a",a)
                s_, r, done, _ ,_= self.env.step(a)

                if self.cfg.use_state_norm:
                    s_ = self.state_norm(s_)
                if args.use_reward_norm:
                    r = reward_norm(r)
                elif self.cfg.use_reward_scaling:
                    r = reward_scaling(r)

                # When dead or win or reaching the max_episode_steps, done will be Ture, we need to distinguish them;
                # dw means dead or win,there is no next state s';
                # but when reaching the max_episode_steps,there is a next state s' actually.
                if done and episode_steps != self.cfg.max_episode_steps:
                    dw = True
                else:
                    dw = False

                replay_buffer.store(s, a, a_logprob, r, s_, dw, done)
                s = s_
                total_steps += 1

                # When the number of transitions in buffer reaches batch_size,then update
                if replay_buffer.count == self.cfg.batch_size:
                    print("replay_buffer",replay_buffer)
                    self.agent.update(replay_buffer, total_steps)
                    replay_buffer.count = 0

                # Evaluate the policy every 'evaluate_freq' steps
                if total_steps % self.cfg.evaluate_freq == 0:
                    evaluate_num += 1
                    evaluate_reward = evaluate_policy(self.cfg, self.env_evaluate, self.agent, self.state_norm)
                    evaluate_rewards.append(evaluate_reward)
                    print("evaluate_num:{} \t evaluate_reward:{} \t".format(evaluate_num, evaluate_reward))
                    #writer.add_scalar('step_rewards_{}'.format(env_name), evaluate_rewards[-1], global_step=total_steps)
                    # Save the rewards
                    # if evaluate_num % self.cfg.save_freq == 0:
                    #     np.save('./data_train/PPO_discrete_env_{}_number_{}_seed_{}.npy'.format(env_name, number, seed), np.array(evaluate_rewards))


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameter Setting for PPO-discrete")
    parser.add_argument("--max_train_steps", type=int, default=int(2e5), help=" Maximum number of training steps")
    parser.add_argument("--evaluate_freq", type=float, default=5e3, help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--save_freq", type=int, default=20, help="Save frequency")
    parser.add_argument("--batch_size", type=int, default=2048, help="Batch size")
    parser.add_argument("--mini_batch_size", type=int, default=64, help="Minibatch size")
    parser.add_argument("--hidden_width", type=int, default=64, help="The number of neurons in hidden layers of the neural network")
    parser.add_argument("--lr_a", type=float, default=3e-4, help="Learning rate of actor")
    parser.add_argument("--lr_c", type=float, default=3e-4, help="Learning rate of critic")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lamda", type=float, default=0.95, help="GAE parameter")
    parser.add_argument("--epsilon", type=float, default=0.2, help="PPO clip parameter")
    parser.add_argument("--K_epochs", type=int, default=10, help="PPO parameter")
    parser.add_argument("--use_adv_norm", type=bool, default=True, help="Trick 1:advantage normalization")
    parser.add_argument("--use_state_norm", type=bool, default=True, help="Trick 2:state normalization")
    parser.add_argument("--use_reward_norm", type=bool, default=False, help="Trick 3:reward normalization")
    parser.add_argument("--use_reward_scaling", type=bool, default=True, help="Trick 4:reward scaling")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="Trick 5: policy entropy")
    parser.add_argument("--use_lr_decay", type=bool, default=True, help="Trick 6:learning rate Decay")
    parser.add_argument("--use_grad_clip", type=bool, default=True, help="Trick 7: Gradient clip")
    parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Trick 8: orthogonal initialization")
    parser.add_argument("--set_adam_eps", type=float, default=True, help="Trick 9: set Adam epsilon=1e-5")
    parser.add_argument("--use_tanh", type=float, default=True, help="Trick 10: tanh activation function")

    args = parser.parse_args()
    runners = wise_rl.make_runner("runner", GymRunner,args, num=1)
    wise_rl.start_all_runner(runners)