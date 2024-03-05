# -- coding: utf-8 --
import ray

class AgentProxy(object):
    def __init__(self, agent,copy_agent=None):
        self.agent = agent
        self.copy_agent = copy_agent
       
    def choose_action(self, *args,**kwargs):
        if self.copy_agent != None:
            return ray.get(self.copy_agent.choose_action.remote(*args , **kwargs))
        else:
            return ray.get(self.agent.choose_action.remote(*args , **kwargs))

    def update(self,*args,**kwargs):
        ray.get(self.agent.update.remote(*args, **kwargs))
    
    def _update_model(self,*args, **kwargs):
        ray.get(self.agent._update_model.remote(*args, **kwargs))