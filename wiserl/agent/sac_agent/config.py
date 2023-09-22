# hyper-parameters
tau = 0.005
gamma = 0.99
MEMORY_CAPACITY = 10000
#MEMORY_CAPACITY = 500
MAX_EPOCH = 100000
learning_rate = 3e-4
gradient_steps = 1
min_Val = 1e-7
batch_size = 256
ACTION_FIRE = 64

TARGET_NETWORK_REPLACE_FREQ = 20  # How frequently target netowrk updates
# The capacity of experience replay buffer

# run const
total_steps = 0
random_steps = 150
max_episode_steps = 150
log_interval = 2000
# net parameter
hidden_dim = 512

# hyper-parameters
value_lr = 3e-4
soft_q_lr = 3e-4
policy_lr = 3e-4
max_steps = 200  # Pendulum needs 150 steps per episode to learn well, cannot handle 20
frame_idx = 0
explore_steps = 0
rewards = []
reward_scale = 10.0
model_path = './model/sac'
