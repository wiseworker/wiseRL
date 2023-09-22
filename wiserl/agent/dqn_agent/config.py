BATCH_SIZE = 32     # batch size of sampling process from buffer
LR = 0.01           # learning rate
EPSILON = 0.9       # epsilon used for epsilon greedy approach
GAMMA = 0.9         # discount factor
ACTION_FIRE = 64
MAX_EPOCH = 4000
TARGET_NETWORK_REPLACE_FREQ = 20       # How frequently target netowrk updates
MEMORY_CAPACITY = 8000                  # The capacity of experience replay buffer
