import ray
import time
from  core.registre import Registre
from  core.env import Env
class WiseRL:
    def __init__(self, actor_cfg, num_gpus:0):
        ray.init(num_gpus=num_gpus)
        self.registre =  Registre.remote()
        for cfg in actor_cfg:
            actor_num = 1
            if "num" in cfg:
               actor_num = cfg['num']
            
            num_gpus = 0
            if "num_gpus" in cfg:
               num_gpus = cfg['num_gpus']
         
            for n in range(actor_num):
                actor = None
                if num_gpus ==0:
                    actor =ray.remote(cfg['actor']).remote()
                else:
                    print("num_gpus=",num_gpus)
                    actor =ray.remote(cfg['actor']).options(num_gpus=num_gpus).remote()
                actor.setRegistre.remote(self.registre)
                actor.setRank.remote(n)
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

         