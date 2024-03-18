# -- coding: utf-8 --
import gym

class WsEnv(object):
    def __init__(self, env, name, action_space, state_space, discrete=False):
        self.name = name
        self.action_space = action_space
        self.state_space = state_space
        self.observation_space = state_space
        self.discrete = discrete
        self.env = env
        self.n_agents = 1
    
    def step(self, a):
        obs, reward, done, _, info = self.env.step(a)
        observation_obs = obs
        available_actions = None
        return (obs, observation_obs, reward, done,  info, available_actions)

    def reset(self):
        obs, info = self.env.reset()
        observation_obs = obs
        return (obs, observation_obs, info)
    

def make_env(name):
    # from pettingzoo.mpe import simple_spread_v3
    if name == "CartPole-v1":
        env = gym.make("CartPole-v1")
        action_space = env.action_space.n  # 2 actions
        state_space = env.observation_space.shape[0]  # 4 states
        discrete = True
        return WsEnv(env, name, action_space, state_space, discrete=discrete)
    # elif name == 'simple_spread_v3':
    #     env = simple_spread_v3.parallel_env(render_mode=None)
    #     action_space = 5
    #     state_space = 18
    #     return WsEnv(env, name , action_space, state_space)
    elif name == 'Pendulum-v1':
        env = gym.make("Pendulum-v1")
        action_space = env.action_space.shape[0]
        state_space = env.observation_space.shape[0]
        discrete = False
        return WsEnv(env, name, action_space, state_space, discrete=discrete)
    raise Exception("No such environment.")



def save_gym_state(env,i):
    next_state = env.render(mode='rgb_array')
    plt.imsave('./shy/state/state{}.png'.format(i),preprocess_frame(next_state)) 
    next_state = plt.imread('./state/state{}.png'.format(i))
    return next_state


def preprocess_frame(self,frame):
    gray = rgb2gray(frame)
    #crop the frame
    #cropped_frame = gray[:,:]
    normalized_frame = gray/255.0
    preprocessed_frame = transform.resize(normalized_frame, [84,84])
    return preprocessed_frame