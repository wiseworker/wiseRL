# -- coding: utf-8 --

from wiserl.core.actor import Actor

class Runner(Actor):

    def __init__(self):
        super().__init__()
     
    def run(self):
        pass

    def set_rank(self,rank):
        self.rank = rank
    
    def get_rank(self):
        return self.rank
