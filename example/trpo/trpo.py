from wiserl.core.runner import Runner
from wiserl.agent.trpo_agent.trpo_agent import TRPOAgent
from wiserl.net.dnn_net import DNNNet
from wiserl.core.wise_rl import WiseRL
from wiserl.env import make_env
from wiserl.utils.replay_buffer import ReplayBuffer
from wiserl.utils.normalization import Normalization, RewardScaling
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import argparse
import os

wise_rl = WiseRL()

class GymRunner(Runner):
    def __init__(self, args, local_rank=0):
        self.wsenv = make_env("CartPole-v1")
        print("rank=", local_rank)
        self.agent_name = "trpo_agent"
        self.local_rank = local_rank
        self.config = args
        # config setting
        setattr(self.config, 'state_dim', self.wsenv.state_dim)
        setattr(self.config, 'action_dim', self.wsenv.action_dim)

        if self.local_rank == 0:
            wise_rl.make2_agent(name=self.agent_name, agent_class=TRPOAgent, sync=True, **vars(self.config))
        self.agent = wise_rl.getAgent(self.agent_name)
        self.state_norm = Normalization(shape=self.config.state_dim)  # Trick 2:state normalization

    def run(self):
        return_list = []
        num_episodes = 500
        for i in range(1):
            with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
                for i_episode in range(int(num_episodes / 10)):
                    episode_return = 0
                    transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
                    state, _ = self.wsenv.reset()
                    done = False
                    while not done:
                        action = self.agent.choose_action(state)
                        next_state, reward, done, _, _ = self.wsenv.step(action)
                        transition_dict['states'].append(state)
                        transition_dict['actions'].append(action)
                        transition_dict['next_states'].append(next_state)
                        transition_dict['rewards'].append(reward)
                        transition_dict['dones'].append(done)
                        state = next_state
                        episode_return += reward
                    return_list.append(episode_return)
                    self.agent.update(transition_dict)
                    if (i_episode + 1) % 10 == 0:
                        pbar.set_postfix({'episode': '%d' % (num_episodes / 10 * i + i_episode + 1),
                                          'return': '%.3f' % np.mean(return_list[-10:])})
                    pbar.update(1)

    def evaluate_policy(self, env):
        times = 3
        evaluate_reward = 0
        for _ in range(times):
            state = env.reset()[0]
            done = False
            if self.config.use_state_norm:  # During the evaluating,update=False
                state = self.state_norm(state, update=False)

            episode_reward = 0
            while not done:
                return_value = self.agent.choose_action(state)  # We use the deterministic policy during the evaluating
                action = return_value[0]
                action_logprob = return_value[1]
                state_, reward, done, info, _= env.step(action)
                if self.config.use_state_norm:
                    state_ = self.state_norm(state_, update=False)
                episode_reward += reward
                state = state_
            evaluate_reward += episode_reward

        return evaluate_reward / times

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameter Setting for TRPO")
    parser.add_argument("--print_interval", type=int, default=10, help="Print_interval")
    parser.add_argument("--max_train_steps", type=int, default=int(2e6), help=" Maximum number of training steps")
    parser.add_argument("--evaluate_freq", type=float, default=5e2, help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--save_freq", type=int, default=20, help="Save frequency")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--mini_batch_size", type=int, default=6, help="Mini Batch size")
    parser.add_argument("--net_dims", default=(256, 128), help="The number of neurons in hidden layers of the neural network")
    parser.add_argument("--optimizer", default="Adam", help="Optimizer")
    parser.add_argument("--lr_a", type=float, default=1e-3, help="Learning rate of actor")
    parser.add_argument("--lr_c", type=float, default=1e-2, help="Learning rate of critic")
    parser.add_argument("--gamma", type=float, default=0.98, help="Discount factor")
    parser.add_argument("--lmbda", type=float, default=0.95, help="GAE parameter")
    parser.add_argument("--alpha", type=float, default=0.5, help="parameter")
    parser.add_argument("--kl_constraint", type=float, default=0.0005, help="kl_constraint")
    parser.add_argument("--K_epochs", type=int, default=10, help="epoch_num")
    parser.add_argument("--scale_factor", type=int, default=2, help="scale_factor")
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
    #runners = wise_rl.makeRunner("runner", GymRunner, num=5,resource={"resources":{'custom_env':1}})
    runners = wise_rl.makeRunner("runner", GymRunner, args, num=2)
    wise_rl.startAllRunner(runners)
