import gym
import time
import ray
import example.ppo.config as config
from wiseRL.core.wise_rl  import WiseRL
from wiseRL.core.env import Env
from net import Critic,Actor 
from normalization import Normalization, RewardScaling
from replaybuffer import ReplayBuffer

class EnvActor(Env):

    def __init__(self):
        super().__init__()
        self.env = config.env

    def run(self):
        learner = self.getActor("learner")
        env = gym.make(config.ENV_NAME)
        env_evaluate = gym.make(config.ENV_NAME)  # When evaluating the policy, we need to rebuild an environment
        # Set random seed
        # env.seed(seed)
        # env.action_space.seed(seed)
        # env_evaluate.seed(seed)
        # env_evaluate.action_space.seed(seed)
        # np.random.seed(seed)
        # torch.manual_seed(seed)

        print("state_dim={}".format(config.N_STATE))
        print("action_dim={}".format(config.N_ACTION))
        print("max_episode_steps={}".format(config.MAX_EPISODE_STEPS))

        evaluate_num = 0  # Record the number of evaluations
        evaluate_rewards = []  # Record the rewards during the evaluating
        total_steps = 0  # Record the total steps during the training

        replay_buffer = ReplayBuffer()
        state_norm = Normalization(config.N_STATE)  # Trick 2:state normalization
        if config.USE_REWARD_NORM:  # Trick 3:reward normalization
            reward_norm = Normalization(shape=1)
        elif config.USE_REWARD_SCALING:  # Trick 4:reward scaling
            reward_scaling = RewardScaling(shape=1, gamma=config.GAMMA)

        while total_steps < config.MAX_EPISODE_STEPS:
            s = env.reset()[0]
            if config.USE_STATE_NORM:
                s = state_norm(s)
            if config.USE_REWARD_SCALING:
                reward_scaling.reset()
            episode_steps = 0
            done = False
            evaluate_reward =0 
            while not done:
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
                if done and episode_steps != config.MAX_EPISODE_STEPS:
                    dw = True
                else:
                    dw = False

                replay_buffer.store(s, a, a_logprob, r, s_, dw, done)
                s = s_
                total_steps += 1

                # When the number of transitions in buffer reaches batch_size,then update
                if replay_buffer.count == config.BATCH_SIZE:
                    buffer_id = ray.put(replay_buffer)
                    learner.update(buffer_id, total_steps)
                    replay_buffer.count = 0

                # Evaluate the policy every 'evaluate_freq' steps
                if total_steps % config.EVALUATE_FREQ == 0:
                    evaluate_num += 1
                    print("evaluate_num:{} \t evaluate_reward:{} \t".format(evaluate_num, evaluate_reward))
                    # Save the rewards
                    # if evaluate_num % config.SAVE_FREQ == 0:
                    #     np.save('./data_train/PPO_discrete_env_{}_number_{}_seed_{}.npy'.format(env_name, number, seed), np.array(evaluate_rewards))
        print("done",total_steps)