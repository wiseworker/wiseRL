import ray
from wiseRL.core.base import Base
class Learner(Base):
    def __init__(self):
        super().__init__()
  
    def update(self,*args,**kwargs):
        pass

    def fire(self, name,*args,**kwargs):
        actors = self.getAllActors(name)
        refs = []
        for learner in actors:
            ref = learner.updateModel.remote(*args, **kwargs)
            refs.append(ref)
        ray.get(refs)
        