from wiserl.core.runner import Runner
from wiserl.agent.dqn_agent import DQNAgent
from wiserl.net.dqn_net import DQNNet
from wiserl.core.wise_rl import  WiseRL
import gym
import time
import ray
wise_rl = WiseRL()

class GymRunner(Runner):
    def __init__(self,local_rank=0):
        self.env = gym.make("CartPole-v1")
        print("rank=",local_rank)
        N_ACTIONS = self.env.action_space.n  # 2 actions
        N_STATES = self.env.observation_space.shape[0] # 4 states
        self.local_rank = local_rank
        if local_rank ==0:
            #net = DQNNet(N_STATES,N_ACTIONS)
            wise_rl.makeAgent(name ="dqn_agent",agent_class=DQNAgent, net_class=DQNNet,n_states=N_STATES,n_actions=N_ACTIONS,sync=False)
            self.agent = wise_rl.getAgent("dqn_agent")
        else:
            self.agent = wise_rl.getAgent("dqn_agent")
       
    def run(self):
        print("run")
        start = time.time()
        for i_episode in range(1000):
            s = self.env.reset()[0]
            ep_r = 0
            while True:
                a = self.agent.choseAction(s)
                s_, r, done, info, _ = self.env.step(a)
                x, x_dot, theta, theta_dot = s_
                r1 = (self.env.x_threshold - abs(x)) / self.env.x_threshold - 0.8
                r2 = (self.env.theta_threshold_radians - abs(theta)) / self.env.theta_threshold_radians - 0.5
                r = r1 + r2
                self.agent.update(s, a, r, s_)
                ep_r += r
                if done:
                    r = -10
                    end = time.time()
                    print(self.local_rank, 'time', round((end-start),2),' Ep: ', i_episode, ' |', 'Ep_r: ', round(ep_r, 2))
                    break
                s = s_  

if __name__=='__main__':
   
    runners = wise_rl.makeRunner("runner",GymRunner,num=5)
    wise_rl.startAllRunner(runners)