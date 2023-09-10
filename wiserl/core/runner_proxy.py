# -- coding: utf-8 --
import ray 

class RunnerProxy(object):

    def __init__(self, runner):
        self.runner = runner
    
    def getRunner():
        return self.runner

    def run(self):
        return self.runner.run.remote()

    def setRank(self,rank):
        ray.get(self.runner.setRank.remote(rank))
    
    def getRank(self):
        ray.get(self.runner.getRank.remote())

