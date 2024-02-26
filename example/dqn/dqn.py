from wiserl.core.runner import Runner
from wiserl.agent.dqn_agent.dqn_agent import DqnAgent
from wiserl.core.wise_rl import WiseRL
from wiserl.env import make_env, WsEnv
import time
import argparse
import configparser

wise_rl = WiseRL()
class GymRunner(Runner):
    def __init__(self, args,local_rank=0):
        self.local_rank = local_rank
        self.agent_name ="dqn_agent"
        self.env = make_env("CartPole-v1")
        self.config = args
        setattr(self.config, 'state_dim', self.env.state_dim)
        setattr(self.config, 'action_dim', self.env.action_dim)
        self.total_steps = 0
        if local_rank == 0:
            wise_rl.make_agent(name=self.agent_name, agent_class=DqnAgent,config = self.config)
            self.agent = wise_rl.get_agent(self.agent_name)
        else:
            self.agent = wise_rl.get_agent(self.agent_name)

    def run(self):
        start = time.time()
        while self.total_steps < args.max_train_steps:
            s = self.env.reset()[0]
            ep_r = 0
            while True:
                a = self.agent.choose_action(s)
                s_, r, done, info, _ = self.env.step(a)
                x, x_dot, theta, theta_dot = s_
                r1 = (self.env.env.x_threshold - abs(x)) / self.env.env.x_threshold - 0.8
                r2 = (self.env.env.theta_threshold_radians - abs(theta)) / self.env.env.theta_threshold_radians - 0.5
                r = r1 + r2
                ep_r += r
                #print("r", r,done)
                self.agent.update(s, a, r, s_, done)
                if done:
                    r = -10
                    end = time.time()
                    print(self.local_rank, 'time', round((end - start), 2), ' Ep: ', self.total_steps, ' |', 'Ep_r: ',
                          round(ep_r, 2))
                    break
                s = s_
                self.total_steps += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameter Setting for PPO-discrete")
    parser.add_argument("--max_train_steps", type=int, default=int(2e6), help=" Maximum number of training steps")
    parser.add_argument("--evaluate_freq", type=float, default=5e3, help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--save_freq", type=int, default=20, help="Save frequency")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--net_dims", default=(256,256), help="The number of neurons in hidden layers of the neural network")
    parser.add_argument("--lr_a", type=float, default=3e-4, help="Learning rate of actor")
    parser.add_argument("--lr_c", type=float, default=3e-4, help="Learning rate of critic")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lamda", type=float, default=0.95, help="GAE parameter")
    parser.add_argument("--epsilon", type=float, default=0.2, help="PPO clip parameter")
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
    runners = wise_rl.make_runner("runner", GymRunner,args, num=5)
    wise_rl.start_all_runner(runners)