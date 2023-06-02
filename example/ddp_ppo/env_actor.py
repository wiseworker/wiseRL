import gym
import time
import ray
import example.ddp_ppo.config as config
from wiseRL.core.wise_rl  import WiseRL
from wiseRL.core.env import Env
from net import Critic,Actor 
from normalization import Normalization, RewardScaling
from replaybuffer import ReplayBuffer
import time

class EnvActor(Env):

    def __init__(self):
        super().__init__()
        self.env = config.env
        self.start_time = time.time()

    def _toList(self,tuple):
        states = []
        for index,value in enumerate(tuple):
            states.append(value)
        return states
    def run(self):
        print("get action")
        learner = self.getActor("learner")
        env = config.env
        print("state_dim={}".format(config.N_STATE))
        print("action_dim={}".format(config.N_ACTION))
        print("max_episode_steps={}".format(config.MAX_EPISODE_STEPS))

        evaluate_num = 0  # Record the number of evaluations
        total_steps = 0  # Record the total steps during the training

        
        state_norm = Normalization(config.N_STATE)  # Trick 2:state normalization
        if config.USE_REWARD_NORM:  # Trick 3:reward normalization
            reward_norm = Normalization(shape=1)
        elif config.USE_REWARD_SCALING:  # Trick 4:reward scaling
            reward_scaling = RewardScaling(shape=1, gamma=config.GAMMA)
        while total_steps < config.MAX_EPISODE_STEPS:
            s = env.reset()
            replay_buffers = [ ReplayBuffer() for x in range(config.NUM_ENVS)]
            #print("s1",len(s),s[1])
            #s = self._toList(s)
            dones = {}
            s = s[0]
            episode_steps = 0
            evaluate_reward = 0
            while len(dones) < config.NUM_ENVS:
                episode_steps += 1
                a, a_logprob = learner.choseAction(s)  # Action and the corresponding log probability
                
                s_, r, done, info, _ = env.step(a)
                if config.USE_STATE_NORM:
                    s_ = state_norm(s_)
                if config.USE_REWARD_NORM:
                    r = reward_norm(r)
                elif config.USE_REWARD_SCALING:
                    r = reward_scaling(r)
                evaluate_reward += r  
                # When dead or win or reaching the max_episode_steps, done will be Ture, we need to distinguish them;
                # dw means dead or win,there is no next state s';
                # but when reaching the max_episode_steps,there is a next state s' actually.
                for i in range(len(done)):
                    d = done[i]
                    if d == True:
                        if  i not  in  dones:
                            dones[i]=True
                            dw = True
                            replay_buffers[i].store(s[i], a[i], a_logprob[i], r[i], s_[i], dw, done[i])
                            buffer_id = ray.put(replay_buffers[i])
                            learner.update(buffer_id, episode_steps)
                    else:
                        dw = False
                        replay_buffers[i].store(s[i], a[i], a_logprob[i], r[i], s_[i], dw, done[i])                
                if episode_steps == config.BATCH_SIZE:
                    for i in range(len(done)):
                        if i not  in dones:
                            dones[i]=True
                            buffer_id = ray.put(replay_buffers[i])
                            learner.update(buffer_id, episode_steps)
            evaluate_num += 1
            end = time.time()
            print("evaluate_num:{} \t evaluate_reward:{} {}\t".format(evaluate_num, evaluate_reward,
            (end-self.start_time) ))
                    # Save the rewards
                    # if evaluate_num % config.SAVE_FREQ == 0:
                    #     np.save('./data_train/PPO_discrete_env_{}_number_{}_seed_{}.npy'.format(env_name, number, seed), np.array(evaluate_rewards))
        print("done",total_steps)