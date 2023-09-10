# -- coding: utf-8 --
import ray

class AgentProxy(object):

    def __init__(self, agent):
        self.agent = agent
       
    def choseAction(self, *args,**kwargs):
        return ray.get(self.agent.choseAction.remote(*args , **kwargs))

    def update(self,*args,**kwargs):
        ray.get(self.agent.update.remote(*args, **kwargs))
    
    def _updateModel(self,*args, **kwargs):
        ray.get(self.agent._updateModel.remote(*args, **kwargs))