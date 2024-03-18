import torch
import numpy as np
import argparse
from wiserl.utils.replay_buffer import ReplayBuffer
from wiserl.core.runner import Runner
from wiserl.agent.ppo_agent.ppo2_agent import PPO2Agent
from wiserl.core.wise_rl import WiseRL
from wiserl.envs.env import make_env
from wiserl.utils.normalization import Normalization, RewardScaling

wise_rl = WiseRL()




class GymRunner(Runner):
    def __init__(self, args=None, local_rank=0):
        self.ws_env = make_env("simple_spread_v3")
        self.env_evaluate = make_env("simple_spread_v3").env  # When evaluating the policy, we need to rebuild an environment
        self.cfg= args
        self.agent_name = "agent"
        # Set random seed
        seed = 0
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.net_cfg = {
            "net":  "DNNNet",
            "action_dim": self.ws_env.action_dim,
            "state_dim": self.ws_env.state_dim
        }
        self.replay_buffer = []
        self.agent = None
        for i in range(self.cfg.num_agent):
            self.replay_buffer.append(ReplayBuffer(self.cfg.batch_size*3,self.net_cfg['state_dim']))
      
        if local_rank == 0:
            wise_rl.make_agent(name=self.agent_name, agent_class=PPO2Agent, net_cfg=self.net_cfg,cfg=args,sync=True)
            self.agent = wise_rl.get_agent(self.agent_name)
        else:
            self.agent = wise_rl.get_agent(self.agent_name)
        self.state_norm = Normalization(shape=self.net_cfg['state_dim'])  # Trick 2:state normalization
    def run(self):
        evaluate_num = 0  # Record the number of evaluations
        evaluate_rewards = []  # Record the rewards during the evaluating
        total_steps = 0  # Record the total steps during the training
        replay_buffer = []
       
        if self.cfg.use_reward_norm:  # Trick 3:reward normalization
            reward_norm = Normalization(shape=1)
        elif self.cfg.use_reward_scaling:  # Trick 4:reward scaling
            reward_scaling = RewardScaling(shape=1, gamma=self.cfg.gamma)

        while total_steps < self.cfg.max_train_steps:
            observations, infos = self.ws_env.env.reset()
            if self.cfg.use_state_norm:
                for agent in observations:
                    observations[agent] = self.state_norm(observations[agent])
            if self.cfg.use_reward_scaling:
                reward_scaling.reset()
            episode_steps = 0
            done = False
            while not done:
                episode_steps += 1
                actions , logprobs =self.get_actions(observations)
                observations_, rewards, terminations, truncations, infos = self.ws_env.env.step(actions)
                #print("rewards",rewards)
                if self.cfg.use_state_norm:
                    for agent in observations_:
                        #print("observations_",observations_)
                        observations_[agent]= self.state_norm(observations_[agent])
                if args.use_reward_norm:
                    rewards = reward_norm(rewards)
                elif self.cfg.use_reward_scaling:
                    for agent in rewards:
                        rewards[agent] = reward_scaling(rewards[agent])
                
                done = False if self.ws_env.env.agents  else True
                #print("done", done,self.ws_env.envs.agents)
                # When dead or win or reaching the max_episode_steps, done will be Ture, we need to distinguish them;
                # dw means dead or win,there is no next state s';
                # but when reaching the max_episode_steps,there is a next state s' actually.
                if done and episode_steps != self.cfg.max_episode_steps:
                    dw = True
                else:
                    dw = False

                self.store(observations, actions, logprobs, rewards, observations_, dw, done)
                observations = observations_
                total_steps += 1


                # When the number of transitions in buffer reaches batch_size,then update
                if self.replay_buffer[0].count == self.cfg.batch_size*3:
                    print("update=========================")
                    self.agent_update(total_steps)
                if total_steps % 2000 ==0:
                    evaluate_num += 1
                    evaluate_reward = self.evaluate_policy(self.cfg, self.env_evaluate, self.agent, self.state_norm)
                    evaluate_rewards.append(evaluate_reward)
                    print("evaluate_num:{} \t evaluate_reward:{} \t".format(evaluate_num, evaluate_reward))
                    #writer.add_scalar('step_rewards_{}'.format(env_name), evaluate_rewards[-1], global_step=total_steps)
                    # Save the rewards
                    # if evaluate_num % self.cfg.save_freq == 0:
                    #     np.save('./data_train/PPO_discrete_env_{}_number_{}_seed_{}.npy'.format(env_name, number, seed), np.array(evaluate_rewards))

    def get_actions(self,observations):
        actions ={}
        logprobs = {}
        for index, agent in enumerate(observations): 
            a, a_logprob = self.agent.choose_action(observations[agent])  # Action and the corresponding log probability
            actions[agent]=a
            logprobs[agent]=a_logprob
        return actions, logprobs

    def store(self,observations, actions, logprobs, rewards, observations_, dw, done ):
        for index, agent in enumerate(observations):
            s_ = None if len(observations_) ==0 else observations_[agent]
            self.replay_buffer[0].store(observations[agent], actions[agent], logprobs[agent], rewards[agent], s_, dw, done)

    def agent_update(self,total_steps):
        #for i in range(self.cfg.num_agent): 
        #print("update")
        self.agent.update(self.replay_buffer[0], total_steps)
        self.replay_buffer[0].count = 0
    def evaluate_policy(self,args, env, agents, state_norm):
        times = 3
        evaluate_reward = 0
        for _ in range(times):
            observations, infos = env.reset()
            if args.use_state_norm:  # During the evaluating,update=False
                for agent in observations:
                    observations[agent] = state_norm(observations[agent], update=False)
            done = False
            episode_reward = 0
            while not done:
                actions , logprobs =self.get_actions(observations)
                observations_, rewards, terminations, truncations, infos = env.step(actions)
                if args.use_state_norm:
                    for agent in observations_:
                        observations_[agent] = state_norm(observations_[agent], update=False)
                        episode_reward += rewards[agent]
                observations = observations_
                done = False if env.agents else True
                #print("done", done,envs.agents)
            evaluate_reward += episode_reward
        return  evaluate_reward/times

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameter Setting for PPO-discrete")
    parser.add_argument("--max_train_steps", type=int, default=int(2e7), help=" Maximum number of training steps")
    parser.add_argument("--evaluate_freq", type=float, default=5e3, help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--save_freq", type=int, default=20, help="Save frequency")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--mini_batch_size", type=int, default=64, help="Minibatch size")
    parser.add_argument("--hidden_width", type=int, default=64, help="The number of neurons in hidden layers of the neural network")
    parser.add_argument("--lr_a", type=float, default=5e-2, help="Learning rate of actor")
    parser.add_argument("--lr_c", type=float, default=5e-3, help="Learning rate of critic")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lamda", type=float, default=0.95, help="GAE parameter")
    parser.add_argument("--epsilon", type=float, default=0.2, help="PPO clip parameter")
    parser.add_argument("--K_epochs", type=int, default=10, help="PPO parameter")
    parser.add_argument("--max_episode_steps", type=int, default=100, help="max_episode_step")
    parser.add_argument("--num_agent", type=int, default=3, help="num_agent")
    parser.add_argument("--use_adv_norm", type=bool, default=True, help="Trick 1:advantage normalization")
    parser.add_argument("--use_state_norm", type=bool, default=True, help="Trick 2:state normalization")
    parser.add_argument("--use_reward_norm", type=bool, default=False, help="Trick 3:reward normalization")
    parser.add_argument("--use_reward_scaling", type=bool, default=False, help="Trick 4:reward scaling")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="Trick 5: policy entropy")
    parser.add_argument("--use_lr_decay", type=bool, default=True, help="Trick 6:learning rate Decay")
    parser.add_argument("--use_grad_clip", type=bool, default=True, help="Trick 7: Gradient clip")
    parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Trick 8: orthogonal initialization")
    parser.add_argument("--set_adam_eps", type=float, default=True, help="Trick 9: set Adam epsilon=1e-5")
    parser.add_argument("--use_tanh", type=float, default=True, help="Trick 10: tanh activation function")

    args = parser.parse_args()
    runners = wise_rl.make_runner("runner", GymRunner,args, resource={"num_cpus":1},num=1)
    wise_rl.start_all_runner(runners)


  