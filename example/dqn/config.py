
import gym

BATCH_SIZE = 32     # batch size of sampling process from buffer
LR = 0.01           # learning rate
EPSILON = 0.9       # epsilon used for epsilon greedy approach
GAMMA = 0.9         # discount factor
ACTION_FIRE = 64
MAX_EPOCH = 4000
TARGET_NETWORK_REPLACE_FREQ = 20       # How frequently target netowrk updates
MEMORY_CAPACITY = 8000                  # The capacity of experience replay buffer

env = gym.make("CartPole-v1") # Use cartpole game as environment
env = env.unwrapped
N_ACTIONS = env.action_space.n  # 2 actions
N_STATES = env.observation_space.shape[0] # 4 states
ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample().shape     #
