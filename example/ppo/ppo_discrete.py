from wiserl.core.runner import Runner
from wiserl.agent.ppo_agent.ppo_agent import PPOAgent
from wiserl.net.dnn_net import DNNNet
from wiserl.core.wise_rl import WiseRL
from wiserl.env import make_env
from wiserl.utils.replay_buffer import ReplayBuffer
import gym
import time
import configparser
import os
from wiserl.utils.normalization import Normalization, RewardScaling
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
wise_rl = WiseRL()

class GymRunner(Runner):
    def __init__(self, args=None, local_rank=0):
        self.wsenv = make_env("CartPole-v1")
        self.net_cfg = {
            "net":  "DNNNet",
            "action_dim": self.wsenv.action_dim,
            "state_dim": self.wsenv.state_dim
        }
        print("rank=", local_rank)
        self.agent_name = "ppo_agent"
        self.local_rank = local_rank
        conf = configparser.ConfigParser()
        conf.read("/workspace/zs/wiseRL/example/ppo/config.ini")
        self.config = conf['ppo']
       
        if local_rank == 0:
            wise_rl.make_agent(name=self.agent_name, agent_class=PPOAgent, net_cfg=self.net_cfg,cfg=self.config,sync=True)
            self.agent = wise_rl.get_agent(self.agent_name)
        else:
            self.agent = wise_rl.get_agent(self.agent_name)
        self.state_norm = Normalization(shape=self.net_cfg['state_dim'])  # Trick 2:state normalization

    def run(self):
        env = self.wsenv.env
        evaluate_num = 0  # Record the number of evaluations
        evaluate_rewards = []  # Record the rewards during the evaluating
        total_steps = 0  # Record the total steps during the training
        batch_size = 2048  if self.config["batch_size"] == None else int(self.config["batch_size"])
        replay_buffer = ReplayBuffer(batch_size,self.net_cfg['state_dim'])
        print("reward_scaling",  self.config["use_reward_norm"],self.config["use_reward_scaling"] )
        if self.config.getboolean("use_reward_norm"):  # Trick 3:reward normalization
            reward_norm = Normalization(shape=1)
        elif self.config.getboolean("use_reward_scaling"):  # Trick 4:reward scaling
            reward_scaling = RewardScaling(shape=1, gamma=self.config.getfloat("gamma"))
        while total_steps < self.config.getint('max_train_steps'):
            s = env.reset()
            if self.config.getboolean("use_state_norm"):
                s = self.state_norm(s)
            if self.config.getboolean("use_reward_scaling"):
                reward_scaling.reset()
            episode_steps = 0
            done = False
            while not done:
                episode_steps += 1
                re = self.agent.choose_action(s)  # Action and the corresponding log probability
                a = re[0]
                a_logprob = re[1]
                
                s_, r, done, _ = env.step(a)
                if self.config.getboolean("use_state_norm"):
                    s_ = self.state_norm(s_)
                if self.config.getboolean("use_reward_norm"):
                    r = reward_norm(r)
                elif self.config.getboolean("use_reward_scaling"):
                    r = reward_scaling(r)

                if done and episode_steps != self.config.getint("max_episode_steps"):
                    dw = True
                else:
                    dw = False
                replay_buffer.store(s, a, a_logprob, r, s_, dw, done)
                s = s_
                total_steps += 1

                # When the number of transitions in buffer reaches batch_size,then update
                if replay_buffer.count == self.config.getint("batch_size"):
                    self.agent.update(replay_buffer, total_steps)
                    replay_buffer.count = 0

                # Evaluate the policy every 'evaluate_freq' steps
                if total_steps % self.config.getint("evaluate_freq") == 0:
                    evaluate_num += 1
                    evaluate_reward = self.evaluate_policy(env )
                    evaluate_rewards.append(evaluate_reward)
                    done = True
                    print("evaluate_num:{} \t evaluate_reward:{} \t".format(evaluate_num, evaluate_reward))


    def evaluate_policy(self,env):
        times = 3
        evaluate_reward = 0
        for _ in range(times):
            s = env.reset()
            done = False
            if self.config.getboolean("use_state_norm"):  # During the evaluating,update=False
                s = self.state_norm(s, update=False)

            episode_reward = 0
            while not done:
                re = self.agent.choose_action(s)  # We use the deterministic policy during the evaluating
                a = re[0]
                a_logprob = re[1]
                s_, r, done, _= env.step(a)
                if self.config.getboolean("use_state_norm"):
                    s_ = self.state_norm(s_, update=False)
                episode_reward += r
                s = s_
            evaluate_reward += episode_reward

        return evaluate_reward / times

if __name__ == '__main__':
    #runners = wise_rl.makeRunner("runner", GymRunner, num=5,resource={"resources":{'custom_env':1}})
    runners = wise_rl.make_runner("runner", GymRunner, num=5)
    wise_rl.start_all_runner(runners)
