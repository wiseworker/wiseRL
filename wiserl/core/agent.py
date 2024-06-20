# -- coding: utf-8 --

import ray
from wiserl.core.actor import Actor
class Agent(Actor):

    def __init__(self, sync):
        super().__init__()
        self.copy_name = None
        self.sync = sync 
  
    def choose_action(self, *args,**kwargs):
        pass

    def update(self,*args,**kwargs):
        pass

    def get_copy_name(self):
        return self.copy_name
    
    def set_copy_name(self,copy_name):
        self.copy_name = copy_name

    def _sync_model(self):
        pass

    def _update_model(self,param):
        pass

    def get_all_agents(self, name):
        agents =ray.get(self.registre.getAllAgent.remote(name))
        return agents

    def _fire(self,*args,**kwargs):
        agents = self.get_all_agents(self.copy_name)
        refs = []
        for copy_agent in  agents:
            ref = copy_agent._update_model(*args, **kwargs)
       
        