import gym
import time
import ray
import example.dqn.config as config
from core.wise_rl  import WiseRL
from core.env import Env
#@WiseRL.actor
class EnvActor(Env):
    def __init__(self):
        super().__init__()
        self.env = config.env
        
    def run(self):

        action = self.getActor("action")
        learner = self.getActor("learner")
        print("config.MAX_EPOCH", config.MAX_EPOCH)
        start = time.time()
        for i_episode in range(config.MAX_EPOCH):
            s = self.env.reset()[0]
            ep_r = 0
            while True:
                a = ray.get(action.choseAction.remote(s))
                s_, r, done, info, _ = self.env.step(a)
                x, x_dot, theta, theta_dot = s_
                r1 = (self.env.x_threshold - abs(x)) / self.env.x_threshold - 0.8
                r2 = (self.env.theta_threshold_radians - abs(theta)) / self.env.theta_threshold_radians - 0.5
                r = r1 + r2
                ray.get(learner.update.remote(s, a, r, s_))
                ep_r += r
                if done:
                    r = -10
                    end = time.time()
                    print('time', round((end-start),2),' rank',self.getRank(),' Ep: ', i_episode, ' |', 'Ep_r: ', round(ep_r, 2))
                    break
              
                s = s_  
