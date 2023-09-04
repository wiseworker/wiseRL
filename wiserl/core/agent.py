import ray
from wiserl.core.base import Base
class Agent(Base):
    def __init__(self):
        super().__init__()
  
    def choseAction(self, *args,**kwargs):
        pass

    def update(self,*args,**kwargs):
        pass

    def fire(self, name,*args,**kwargs):
        actors = self.getAllActors(name)
        refs = []
        for learner in actors:
            ref = learner.updateModel.remote(*args, **kwargs)
            refs.append(ref)
        ray.get(refs)
        