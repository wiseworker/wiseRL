import ray
from wiserl.core.remote import Remote
class Base(object):
    def __init__(self):
        self.registre = None
        self.rank =0

    def start(self):
        pass

    def setRank(self,rank):
        self.rank = rank
    
    def getRank(self):
        return self.rank

    def setRegistre(self,registre):
        self.registre = registre

    def getRegistre(self):
        return self.registre

    def getActor(self, name):
        actor =ray.get(self.registre.getActor.remote(name))
        remoteActor = self._createRemoteActor(actor)
        return remoteActor

    def getAllActors(self, name):
        actors =ray.get(self.registre.getActorsByName.remote(name))
        return actors
    
    def _createRemoteActor(self,actor):
        remoteActor = Remote(actor)
        return remoteActor
