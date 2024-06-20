import torch
from wiserl.core.runner import Runner
from wiserl.agent.td3_agent.td3_agent import TD3Agent
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

        self.env = gym.make("Pendulum-v1")
        self.config = args
        # self.env = RescaleAction(self.env, min_action=-1.0, max_action=1.0)
        print("rank=", local_rank)
        setattr(self.config, 'state_dim', self.env.observation_space.shape[0])
        setattr(self.config, 'action_dim', self.env.action_space.shape[0])
        setattr(self.config, 'action_bound', float(self.env.action_space.high[0]))

        # Seed Everything
        self.env_seed = self.config.seed
        torch.manual_seed(self.config.seed)
        torch.cuda.manual_seed(self.config.seed)

        self.rank = local_rank
        if self.rank == 0:
            # net = DQNNet(N_STATES,N_ACTIONS)
            wise_rl.make2_agent(name="td3_agent", agent_class=TD3Agent, sync=True, **vars(self.config))
        self.agent = wise_rl.getAgent("td3_agent")

    def run(self):
        print("====================================")
        print("Collection Experience...")
        print("====================================")
        start_time = time.time()
        train_info, plt_list = [], []
        ep_r = 0
        for i in range(self.config.MAX_EPOCH):
            state, _ = self.env.reset()
            for t in range(self.config.max_steps):
                action = self.agent.choose_action(state)
                # next_state, reward, done, _, _ = env.step(np.float32(action))
                next_state, reward, done, _, _ = self.env.step(action)
                ep_r += reward
                self.agent.update(state, action, reward, next_state, done)
                state = next_state
                if done or t == 199:
                    break
            # if i % cfg.log_interval == 0:
            #     self.agent.save()
            train_info.append(ep_r)
            ep_r = 0
            if i % 2 == 0: 
                print("Episode:{}--train_info:{}".format(i, np.mean(train_info[-10:])))
                plt_list.append(np.mean(train_info[-10:]))
                if np.mean(train_info[-10:]) > -500:
                    x = range(len(plt_list))
                    plt.plot(x, plt_list)
                    plt.title('Rewards-TD3')
                    plt.xlabel('Time')
                    plt.ylabel('Rewards')
                    plt.grid(True)
                    plt.savefig('TD3-Rewards')


if __name__ == '__main__':
    print("use GPU to train" if torch.cuda.is_available() else "use CPU to train")
    parser = argparse.ArgumentParser("Hyperparameter Setting for TD3")
    parser.add_argument('--tau', type=float, default=0.005, help='hyper-parameters')
    parser.add_argument('--gamma', type=float, default=0.99, help='hyper-parameters')
    parser.add_argument('--MEMORY_CAPACITY', type=int, default=10000, help='hyper-parameters')
    parser.add_argument('--MAX_EPOCH', type=int, default=4000, help='hyper-parameters')
    parser.add_argument('--LR', type=float, default=0.01, help='hyper-parameters')
    parser.add_argument('--gradient_steps', type=int, default=1, help='hyper-parameters')
    parser.add_argument('--discount', type=float, default=0.99, help='hyper-parameters')
    parser.add_argument('--batch_size', type=int, default=256, help='hyper-parameters')
    parser.add_argument('--ACTION_FIRE', type=int, default=64, help='hyper-parameters')
    parser.add_argument('--TARGET_NETWORK_REPLACE_FREQ', type=int, default=20, help='hyper-parameters')
    parser.add_argument('--total_steps', type=int, default=0, help='count for the number of steps')
    parser.add_argument('--policy_noise', type=float, default=0.2, help='policy_noise')
    parser.add_argument('--max_steps', type=int, default=200, help='max episode steps')
    parser.add_argument('--log_interval', type=int, default=20, help='hyper-parameters')
    parser.add_argument('--hidden_dim', type=int, default=512, help='hyper-parameters')
    parser.add_argument('--learning_rate', type=float, default=3e-4, help='hyper-parameters')
    parser.add_argument('--noise_clip', type=float, default=0.5, help='hyper-parameters')
    parser.add_argument('--explore_steps', type=int, default=0, help='hyper-parameters')
    parser.add_argument('--policy_freq', type=int, default=2, help='hyper-parameters')
    parser.add_argument('--EPSILON', type=float, default=0.9, help='hyper-parameters')
    parser.add_argument('--model_path', type=str, default="./model/td3", help='hyper-parameters')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    
    args = parser.parse_args()
    print(args)
    # ray.init(address="auto")
    # ray.init(local_mode=True)
    wise_rl = WiseRL()
    runners = wise_rl.makeRunner("runner", GymRunner, args, num=1)
    wise_rl.startAllRunner(runners)
