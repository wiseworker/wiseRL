import ray

class Base(object):
    def __init__(self):
        self.registre = None
        self.rank =0

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
        return actor

    def getAllActors(self, name):
        actors =ray.get(self.registre.getActorsByName.remote(name))
        return actors
    
