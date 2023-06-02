import ray
import time
from  wiseRL.core.registre import Registre
from  wiseRL.core.env import Env
class WiseRL:
    def __init__(self, actor_cfg, num_gpus:0):
        ray.init()
        self.registre =  Registre.remote()
        for cfg in actor_cfg:
            actor_num = 1
            if "num" in cfg:
               actor_num = cfg['num']
            
            num_gpus = 0
            num_custom_env = 0
            if "num_gpus" in cfg:
               num_gpus = cfg['num_gpus']
            if "num_custom_env" in cfg:
               num_custom_env = cfg['num_custom_env']
         
            for n in range(actor_num):
                actor = None
                if num_gpus == 0 and num_custom_env == 0:
                    actor =ray.remote(cfg['actor']).remote()
                else:
                    actor =ray.remote(cfg['actor']).options(num_gpus=num_gpus, resources={'custom_env':num_custom_env}).remote()
                actor.setRegistre.remote(self.registre)
                actor.setRank.remote(n)
                actor.start.remote()
                self.registre.addActor.remote(cfg['name'], actor ) 
                if issubclass( cfg['actor'], Env):
                    self.registre.addEnv.remote(cfg['name'], actor )  

    def run(self):
        envs_s = ray.get(self.registre.getAllEnv.remote())
        refs = []
        for envs in envs_s:
            for env in envs:
                ref = env.run.remote()
                refs.append(ref)
        results = ray.get(refs)
        print("results",results)
        time.sleep(10)

         