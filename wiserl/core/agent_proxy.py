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
    def get_values(self, *args,**kwargs):
        if self.copy_agent != None:
            return ray.get(self.copy_agent.get_values.remote(*args, **kwargs))
        else:
            return ray.get(self.agent.get_values.remote(*args, **kwargs))

    def get_actions(self, *args,**kwargs):
        if self.copy_agent != None:
            return ray.get(self.copy_agent.get_actions.remote(*args, **kwargs))
        else:
            return ray.get(self.agent.get_actions.remote(*args, **kwargs))

    def normalizer(self, *args,**kwargs):
        if self.copy_agent != None:
            return ray.get(self.copy_agent.normalizer.remote(*args, **kwargs))
        else:
            return ray.get(self.agent.normalizer.remote(*args, **kwargs))

    def train(self, *args,**kwargs):
        if self.copy_agent != None:
            return ray.get(self.copy_agent.train.remote(*args, **kwargs))
        else:
            return ray.get(self.agent.train.remote(*args, **kwargs))

    def update(self,*args,**kwargs):
        ray.get(self.agent.update.remote(*args, **kwargs))

    def prep_rollout(self, *args, **kwargs):
        ray.get(self.agent.prep_rollout.remote(*args, **kwargs))

    def prep_training(self, *args, **kwargs):
        ray.get(self.agent.prep_training.remote(*args, **kwargs))

    def put_data(self,*args,**kwargs):
        ray.get(self.agent.put_data.remote(*args, **kwargs))
    
    def _update_model(self,*args, **kwargs):
        ray.get(self.agent._update_model.remote(*args, **kwargs))