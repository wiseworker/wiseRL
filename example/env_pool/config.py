
import gym
import envpool

ENV_NAME="LunarLander-v2"
NUM_ENVS =10
env = envpool.make(ENV_NAME, env_type="gym", num_envs=NUM_ENVS)
env = env.unwrapped
MAX_EPISODE_STEPS = 10000000  # Maximum number of steps per episode
N_ACTION = env.action_space.n  # 2 actions
N_STATE = env.observation_space.shape[0] # 4 states
MAX_TRAIN_STEPS=2e5
EVALUATE_FREQ = 5e3
SAVE_FREQ =20
BATCH_SIZE=2048
MINI_BATCH_SIZE=64
HIDDEN_WIDTH=64
LR_A=3e-3
LR_C=3e-3
GAMMA=0.99
LAMDA=0.95
EPSILON=0.2
K_EPOCHS=10
USE_ADV_NORM=True
USE_STATE_NORM =True
USE_REWARD_NORM = False
USE_REWARD_SCALING= False
ENTROPY_COEF=0.01
USE_LR_DECAY=True
USE_GRAD_CLIP=True
USE_ORTHOGONAL_INIT= True
SET_ADAM_EPS=True
USE_TANH= True