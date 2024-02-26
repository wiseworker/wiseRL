# -- coding: utf-8 --
import ray 

class RunnerProxy(object):

    def __init__(self, runner):
        self.runner = runner
    
    def get_runner():
        return self.runner

    def run(self):
        return self.runner.run.remote()

    def set_rank(self,rank):
        ray.get(self.runner.set_rank.remote(rank))
    
    def get_rank(self):
        ray.get(self.runner.get_rank.remote())

