# -- coding: utf-8 --

import ray
from wiserl.core.actor import Actor
class Agent(Actor):

    def __init__(self, sync):
        super().__init__()
        self.copy_name = None
        self.sync = sync 
  
    def choseAction(self, *args,**kwargs):
        pass

    def update(self,*args,**kwargs):
        pass

    def getCopyName(self):
        return self.copy_name
    
    def setCopyName(self,copy_name):
        self.copy_name = copy_name

    def _syncModel(self):
        pass

    def _updateModel(self,param):
        pass

    def getAllAgents(self, name):
        agents =ray.get(self.registre.getAllAgent.remote(name))
        return agents

    def _fire(self,*args,**kwargs):
        agents = self.getAllAgents(self.copy_name)
        refs = []
        for copy_agent in  agents:
            ref = copy_agent._updateModel(*args, **kwargs)
       
        