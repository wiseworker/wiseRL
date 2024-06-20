import torch
import torch.nn.functional as F
import numpy as np
import argparse
import time
from torch.utils.tensorboard import SummaryWriter
from wiserl.utils.replay_buffer import MultiAgentReplayBuffer
from wiserl.core.runner import Runner
from wiserl.agent.ddpg_agent.ddpg_agent import DDPGAgent, SingleAgent
from wiserl.core.wise_rl import WiseRL
from wiserl.env import make_env
from wiserl.utils.normalization import Normalization, RewardScaling
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import array
import matplotlib.pyplot as plt
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_ray = False
if use_ray:
    wise_rl = WiseRL()

writer = SummaryWriter("Log/Avg_socre")
class GymRunner(Runner):
    def __init__(self, args=None, local_rank=0):
        self.ws_env = make_env("simple_adversary_v3")
        self.eval_env = make_env("simple_adversary_v3")
        self.cfg = args
        # Set random seed
        self.seed = 0
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        actor_dims, critic_dims = [], []

        setattr(self.cfg, 'n_agents', self.ws_env.env.max_num_agents)
        setattr(self.cfg, 'n_actions', 5)
        actor_dims = [8, 10, 10]
        # for i in range(self.cfg.n_agents):
        #     actor_dims.append(self.ws_env.env.observation_space[i].shape[0])
        critic_dims = sum(actor_dims)
        setattr(self.cfg, 'actor_dims', actor_dims)
        setattr(self.cfg, 'critic_dims', critic_dims)
        self.maddpg_agents = []

        self.memory = MultiAgentReplayBuffer(1000000, critic_dims, actor_dims, self.cfg.n_actions, self.cfg.n_agents, batch_size=1024)
        if use_ray:
            if local_rank == 0:
                for i in range(self.cfg.num_agent):
                    wise_rl.make2_agent(name=self.agent_name + str(i), agent_class=SingleAgent, sync=True,
                                        **vars(self.cfg))
                    self.maddpg_agents.append(wise_rl.getAgent(self.agent_name + str(i)))
            else:
                for i in range(self.cfg.num_agent):
                    self.maddpg_agents.append(wise_rl.getAgent(self.agent_name + str(i)))
        else:
            for i in range(self.cfg.n_agents):
                self.maddpg_agents.append(SingleAgent(sync=True, agent_idx=i, **vars(self.cfg)))

    def run(self):
        total_steps, best_score = 0, 0
        score_history = []

        for i in range(self.cfg.N_GAMES):
            obs = self.ws_env.env.reset()[0]
            score = 0
            done = [False] * self.cfg.n_agents
            episode_step = 0
            for _ in range (self.cfg.MAX_STEPS):
                actions = self.get_actions(obs)
                obs_, reward, done, info, _ = self.ws_env.env.step(actions)
                state = obs_list_to_state_vector(obs)
                state_ = obs_list_to_state_vector(obs_)
                if episode_step + 1 >= self.cfg.MAX_STEPS:
                    done = [True] * self.cfg.n_agents
                self.memory.store_transition(obs, state, actions, reward, obs_, state_, done)
                if total_steps % 200 == 0:
                    ###########
                    self.agent_update(self.memory)
                obs = obs_
                score += list(reward.values())[0]
                episode_step += 1
                total_steps += 1


            score_history.append(score)
            avg_score = np.mean(score_history[-100:])
            writer.add_scalar("Log/Avg_socre", avg_score, i)
            if avg_score > best_score:
                best_score = avg_score

            if i % self.cfg.PRINT_INTERVAL == 0 and i > 0:

                print('episode', i, 'average score {:.1f}'.format(avg_score))
                self.evaluate()
    def evaluate(self):
        re = 0
        for i in range(3):
            obs = self.eval_env.env.reset()[0]
            score = 0
            done = [False] * self.cfg.n_agents
            episode_step = 0
            for _ in range (self.cfg.MAX_STEPS):

                actions = self.get_actions(obs)
                obs_, reward, done, info, _ = self.eval_env.env.step(actions)
                state = obs_list_to_state_vector(obs)
                state_ = obs_list_to_state_vector(obs_)
                if episode_step + 1 >= self.cfg.MAX_STEPS:
                    done = [True] * self.cfg.n_agents
                self.memory.store_transition(obs, state, actions, reward, obs_, state_, done)
                obs = obs_
                score += sum(reward.values())
                episode_step += 1
            re += score
        print('evaluate_score:', re / 3)
        #self.eval_env.env.close()

    def get_actions(self, observations):
        actions = {}
        for index, agent in enumerate(observations):
            obs = observations[agent]
            actions[agent] = self.maddpg_agents[index].choose_action(obs)
        return actions

    def agent_update(self, memory):
        if not memory.ready():
            return

        actor_states, states, actions, rewards, \
        actor_new_states, states_, dones = memory.sample_buffer()

        states = torch.tensor(states, dtype=torch.float).to(device)
        actions = torch.tensor(actions, dtype=torch.float).to(device)
        rewards = torch.tensor(rewards).to(device)
        states_ = torch.tensor(states_, dtype=torch.float).to(device)
        dones = torch.tensor(dones).to(device)

        all_agents_new_actions = []
        all_agents_new_mu_actions = []
        old_agents_actions = []

        for agent_idx, agent in enumerate(self.maddpg_agents):
            new_states = torch.tensor(actor_new_states[agent_idx],
                                  dtype=torch.float).to(device)

            new_pi = agent.target_actor.forward(new_states)

            all_agents_new_actions.append(new_pi)
            mu_states = torch.tensor(actor_states[agent_idx],
                                 dtype=torch.float).to(device)
            pi = agent.actor.forward(mu_states)
            all_agents_new_mu_actions.append(pi)
            old_agents_actions.append(actions[agent_idx])

        new_actions = torch.cat([acts for acts in all_agents_new_actions], dim=1)
        mu = torch.cat([acts for acts in all_agents_new_mu_actions], dim=1)
        old_actions = torch.cat([acts for acts in old_agents_actions], dim=1)

        for agent_idx, agent in enumerate(self.maddpg_agents):
            critic_value_ = agent.target_critic.forward(states_, new_actions).flatten()
            critic_value_[dones[:, 0]] = 0.0
            critic_value = agent.critic.forward(states, old_actions).flatten()

            target = rewards[:, agent_idx] + agent.gamma * critic_value_
            critic_loss = F.mse_loss(target, critic_value)
            agent.critic.optimizer.zero_grad()
            critic_loss.backward(retain_graph=True)
            agent.critic.optimizer.step()

            actor_loss = agent.critic.forward(states, mu).flatten()
            actor_loss = -torch.mean(actor_loss)
            agent.actor.optimizer.zero_grad()
            actor_loss.backward(retain_graph=True)
            agent.actor.optimizer.step()

            agent.update_network_parameters()

def obs_list_to_state_vector(observation):
    state = np.array([])
    for index, agent in enumerate(observation):
        obs = observation[agent]
        state = np.concatenate([state, obs])
        # for obs in observation:
        #     s = state
        #     o = obs
        #     state = np.concatenate([state, obs])
    return state

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameter Setting for PPO-discrete")
    parser.add_argument("--net_dims", default=(256, 256),
                        help="The number of neurons in hidden layers of the neural network")
    parser.add_argument("--max_train_steps", type=int, default=int(2e7), help=" Maximum number of training steps")
    parser.add_argument("--evaluate_freq", type=float, default=5e3,
                        help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--PRINT_INTERVAL", type=int, default=500, help="Print frequency")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--mini_batch_size", type=int, default=64, help="Minibatch size")
    parser.add_argument("--net_width", type=int, default=256,
                        help="The number of neurons in hidden layers of the neural network")
    parser.add_argument("--lr_a", type=float, default=7e-4, help="Learning rate of actor")
    parser.add_argument("--lr_c", type=float, default=7e-4, help="Learning rate of critic")
    parser.add_argument("--alpha", type=float, default=0.01, help="parameter")
    parser.add_argument("--beta", type=float, default=0.01, help="parameter")
    parser.add_argument("--tau", type=float, default=0.01, help="parameter")
    parser.add_argument("--gamma", type=float, default=0.99, help="parameter")
    parser.add_argument("--fc1", type=float, default=64, help="parameter")
    parser.add_argument("--fc2", type=float, default=64, help="parameter")
    parser.add_argument("--N_GAMES", type=int, default=50000, help="num of episodes")
    parser.add_argument("--MAX_STEPS", type=int, default=25, help="max_episode_step")
    parser.add_argument("--chkpt_dir", type=str, default='tmp/maddpg/', help="checkpoint-dir")

    args = parser.parse_args()
    if use_ray:
        runners = wise_rl.makeRunner("runner", GymRunner, args, resource={"num_cpus": 1}, num=2)
        wise_rl.startAllRunner(runners)
    else:
        runners = GymRunner(args)
        runners.run()


