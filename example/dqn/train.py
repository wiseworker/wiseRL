from example.dqn.env_actor import EnvActor
from example.dqn.action_actor import ActionActor
from example.dqn.learner_actor import LearnerActor
from core.wise_rl  import WiseRL

import config
cfg = [
    {
        "name": "action",
        "actor": ActionActor,
        "num": 1
    },
    {
        "name": "learner",
        "actor": LearnerActor,
        "num": 1,
        "num_gpus": 1
    },
    {
        "name": "env",
        "actor": EnvActor,
        "num": 1
    },

]
def train():
    wiseRl = WiseRL(cfg, num_gpus = 1)
    wiseRl.run()

if __name__=='__main__':
    train()