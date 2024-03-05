# -- coding: utf-8 --
import gym

class WsEnv(object):
    def __init__(self,env,name,action_dim,state_dim, discrete=False):
        self.name = name
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.discrete = True 
        self.env = env
    
    def step(self,s):
        return self.env.step(s)

    def reset(self):
        return self.env.reset()
    

def make_env(name):
    from pettingzoo.mpe import simple_spread_v3
    if name == "CartPole-v1":
        env = gym.make("CartPole-v1")
        action_dim = env.action_space.n  # 2 actions
        state_dim = env.observation_space.shape[0]  # 4 states
        return WsEnv(env,name,action_dim,state_dim)
    elif name == 'simple_spread_v3':
        env = simple_spread_v3.parallel_env(render_mode=None)
        action_dim = 5
        state_dim = 18
        return WsEnv(env,name ,action_dim,state_dim)
    elif name == 'Pendulum-v1':
        env = gym.make("Pendulum-v1")
        action_dim = env.action_space.shape[0]
        state_dim = env.observation_space.shape[0]
        return WsEnv(env,name,action_dim,state_dim)
    else:
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