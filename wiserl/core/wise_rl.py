import ray
import time
from  .registre import Registre
from  .runner import Runner
from  .agent import Agent

#ray.init()
registre =  Registre.remote()
# class WiseRL:
#     def __init__(self, actor_cfg, num_gpus:0):
#         ray.init(num_gpus=num_gpus)
#         self.
#         for cfg in actor_cfg:
#             actor_num = 1
#             if "num" in cfg:
#                actor_num = cfg['num']
            
#             num_gpus = 0
#             if "num_gpus" in cfg:
#                num_gpus = cfg['num_gpus']
         
#             for n in range(actor_num):
#                 actor = None
#                 if num_gpus ==0:
#                     actor =ray.remote(cfg['actor']).remote()
#                 else:
#                     print("num_gpus=",num_gpus)
#                     actor =ray.remote(cfg['actor']).options(num_gpus=num_gpus).remote()
#                 actor.setRegistre.remote(self.registre)
#                 actor.setRank.remote(n)
#                 self.registre.addActor.remote(cfg['name'], actor ) 
#                 if issubclass( cfg['actor'], Env):
#                     self.registre.addEnv.remote(cfg['name'], actor )  

    # def run(self):
    #     envs_s = ray.get(registre.getAllEnv.remote())
    #     refs = []
    #     for envs in envs_s:
    #         for env in envs:
    #             ref = env.run.remote()
    #             refs.append(ref)
    #     results = ray.get(refs)
    #     print("results",results)
    #     time.sleep(10)

def makeRunner(Runner,num =1):
    for i in range(num):
        print("i",i)
        runner =ray.remote(Runner).remote(local_rank=i)
        registre.addRunner.remote(Runner.__name__,runner)
    retref = registre.getAllRunner.remote(Runner.__name__)
    return ray.get(retref)

def getRunner(Runner):
    return ray.get(registre.getRunner.remote(Runner.__name__))

def makeAgent(Agent,net,n_states ,n_actions,config=None,num=1) :
    print("num=============",num)
    for i in range(num):
        agent =ray.remote(Agent).remote(net,n_states,n_actions,config)
        registre.addAgent.remote(Agent.__name__,agent)
    retref = registre.getAllAgent.remote(Agent.__name__)
    return ray.get(retref)

def getAgent(Agent):
    while True:
        agent =ray.get(registre.getAgent.remote(Agent.__name__))
        if agent != None:
            return agent
        time.sleep(2)
