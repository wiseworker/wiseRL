batch_size = 64     # batch size of sampling process from buffer
LR = 0.01           # learning rate
EPSILON = 0.9       # epsilon used for epsilon greedy approach
ACTION_FIRE = 64
MAX_EPOCH = 4000
TARGET_NETWORK_REPLACE_FREQ = 20       # How frequently target netowrk updates
MEMORY_CAPACITY = 10000                # The capacity of experience replay buffer

learning_rate = 3e-4

log_interval = 20
# hyper-parameters
max_steps = 200  # Pendulum needs 150 steps per episode to learn well, cannot handle 20
gamma=0.99       # discount factor
tau=0.005