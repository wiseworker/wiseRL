from wiserl.core.runner import Runner
from wiserl.agent.ppo_agent.ppo_agent import PPOAgent
from wiserl.core.wise_rl import WiseRL
from wiserl.envs.env_tool import *
from wiserl.utils.replay_buffer import ReplayBuffer
import argparse
from wiserl.utils.normalization import Normalization

class GymRunner(Runner):
    def __init__(self, args, local_rank=0):
        print("rank=", local_rank)
        self.agent_name = "ppo_agent"
        self.env_args = "CartPole-v1"
        self.local_rank = local_rank
        self.config = args
        self.n_rollout_threads = args.n_rollout_threads
        self.seed = args.seed
        self.use_ray = args.use_ray
        self.n_agents = 1
        self.wsenv = make_threads_env("gym",  self.n_rollout_threads, self.env_args)
        self.test_env = make_threads_env("gym",  self.n_rollout_threads, self.env_args)
        # config setting
        setattr(self.config, 'state_space', self.wsenv.state_space)
        setattr(self.config, 'action_space', self.wsenv.action_space)
        setattr(self.config, 'n_agents', self.n_agents)
        setattr(self.config, 'n_rollout_threads', self.n_rollout_threads)
        if self.use_ray:
            if self.local_rank == 0:
                wise_rl.make_agent(name=self.agent_name, agent_class=PPOAgent, config=self.config, sync=True)
            self.agent = wise_rl.get_agent(self.agent_name)
        else:
            self.agent = PPOAgent(self.config, sync=False)
        self.state_norm = Normalization(shape=self.config.state_space)  # Trick 2:state normalization

    def run(self):
        evaluate_num = 0  # Record the number of evaluations
        evaluate_rewards = []  # Record the rewards during the evaluating
        total_steps = 0  # Record the total steps during the training
        replay_buffer = ReplayBuffer(self.config, Discrete=True)

        while total_steps < self.config.max_train_steps:
            state = self.wsenv.reset()[0]
            done = False
            episode_reward = 0
            while not done:
                action, action_logprob = self.agent.choose_action(state)  # Action and the corresponding log probability
                state_, observation_obs, reward, done, info, available_actions = self.wsenv.step(action)
                if self.config.use_state_norm:
                    state_ = self.state_norm(state_)
                if self.config.use_reward_scaling: # Trick 4: reward scaling
                    reward *= self.config.scale_factor
                reward_env = np.mean(reward)
                episode_reward += reward_env
                done = np.any(done)
                if done and total_steps != self.config.max_train_steps:
                    dw = True
                    if self.local_rank == 0:
                        print(self.local_rank, ' Ep: ', total_steps, ' |', 'Ep_r: ', episode_reward)
                else:
                    dw = False
                replay_buffer.store(state, action, action_logprob, reward, state_, dw, done)
                replay_buffer.count += 1
                state = state_
                total_steps += 1
                # When the number of transitions in buffer reaches batch_size,then update
                if replay_buffer.count == self.config.batch_size:
                    self.agent.update(replay_buffer, total_steps)
                    replay_buffer.count = 0
                # Evaluate the policy every 'evaluate_freq' steps
                if total_steps % self.config.evaluate_freq == 0:
                    evaluate_num += 1
                    evaluate_reward = self.evaluate_policy()
                    evaluate_rewards.append(evaluate_reward)
                    done = True
                    print("evaluate_num:{} \t evaluate_reward:{} \t".format(evaluate_num, evaluate_reward))

    def evaluate_policy(self):
        times = 3
        evaluate_reward = 0
        for _ in range(times):
            state = self.test_env.reset()[0]
            done = False
            if self.config.use_state_norm:  # During the evaluating,update=False
                state = self.state_norm(state, update=False)
            episode_reward = 0
            while not done:
                action, action_logprob = self.agent.choose_action(state)  # We use the deterministic policy during the evaluating
                state_, observation_obs, reward, done, info, available_actions = self.test_env.step(action)
                if self.config.use_state_norm:
                    state_ = self.state_norm(state_, update=False)
                if self.config.use_reward_scaling: # Trick 4: reward scaling
                    reward *= self.config.scale_factor
                reward_env = np.mean(reward)
                episode_reward += reward_env
                done = np.any(done)
                state = state_
            evaluate_reward += episode_reward
        return evaluate_reward / times

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameter Setting for PPO-discrete")
    parser.add_argument("--max_train_steps", type=int, default=int(2e6), help=" Maximum number of training steps")
    parser.add_argument("--evaluate_freq", type=float, default=5e2, help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--save_freq", type=int, default=20, help="Save frequency")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--mini_batch_size", type=int, default=6, help="Mini Batch size")
    parser.add_argument("--net_dims", default=(256, 128), help="The number of neurons in hidden layers of the neural network")
    parser.add_argument("--optimizer", default="Adam", help="Optimizer")
    parser.add_argument("--lr_a", type=float, default=3e-4, help="Learning rate of actor")
    parser.add_argument("--lr_c", type=float, default=3e-4, help="Learning rate of critic")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lamda", type=float, default=0.95, help="GAE parameter")
    parser.add_argument("--epsilon", type=float, default=0.2, help="PPO clip parameter")
    parser.add_argument("--K_epochs", type=int, default=10, help="epoch_num")
    parser.add_argument("--scale_factor", type=int, default=2, help="scale_factor")
    parser.add_argument("--seed", type=int, default=0, help="seed")
    parser.add_argument("--use_ray", type=bool, default=False, help="use ray or not")
    parser.add_argument("--ray_num", type=int, default=1, help="num of ray")
    parser.add_argument("--n_rollout_threads", type=int, default=1, help="num of threads")
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
    wise_rl = WiseRL(args.use_ray)
    if args.use_ray:
        runners = wise_rl.make_runner("runner", GymRunner, args, num=args.ray_num)
        wise_rl.start_all_runner(runners)
    else:
        runners = GymRunner(args)
        runners.run()
