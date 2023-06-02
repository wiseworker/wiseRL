from example.env_pool.env_actor import EnvActor
from example.env_pool.action_actor import ActionActor
from example.env_pool.learner_actor import LearnerActor
from wiseRL.core.wise_rl  import WiseRL

import config
cfg = [
    {
        "name": "learner",
        "actor": LearnerActor,
        "num": 1,
        "num_gpus": 0.5
    },
    {
        "name": "action",
        "actor": ActionActor,
        "num": 1,
        "num_gpus": 0.5
    },
    {
        "name": "env",
        "actor": EnvActor,
        "num": 10
    },

]
def train():
    wiseRl = WiseRL(cfg, num_gpus = 1)
    wiseRl.run()

if __name__=='__main__':
    train()