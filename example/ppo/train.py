from example.ppo.env_actor import EnvActor
from example.ppo.learner_actor import LearnerActor
from wiseRL.core.wise_rl  import WiseRL

import config
cfg = [
    {
        "name": "learner",
        "actor": LearnerActor,
        "num": 1,
        "num_gpus": 1
    },
    {
        "name": "env",
        "actor": EnvActor,
        "num": 18
    },

]
def train():
    wiseRl = WiseRL(cfg, num_gpus = 1)
    wiseRl.run()

if __name__=='__main__':
    train()