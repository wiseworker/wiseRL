import ray
from  core.env import Env
@ray.remote
class Registre(object):
    def __init__(self):
        self.actor_dict = {}
        self.actor_index_dict = {}
        self.env_dict = {}

    def addActor(self,name, actor):
        if (name in self.actor_dict.keys()):
            self.actor_dict[name].append(actor)
        else:
            self.actor_dict[name]=[actor]
            self.actor_index_dict[name] = 0

    def getActor(self, name):
        actors = self.actor_dict[name]
        index = self.actor_index_dict[name]
        index += 1
        if index >= len(actors):
            index =0
        self.actor_index_dict[name] = index
        return actors[index]

    def getActorsByName(self, name):
        actors = self.actor_dict[name]
        return actors

    def getAllActors(self):
        return self.actors_dict.values()    
    
    def addEnv(self,name, env):
        if (name in self.env_dict.keys()):
            self.env_dict[name].append(env)
        else:
            self.env_dict[name]=[env]

    def getEnvByName(self, name):
        return self.env_dict[name]    
    
    def getAllEnv(self):
        return self.env_dict.values()    
    