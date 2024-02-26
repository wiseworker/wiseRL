# -- coding: utf-8 --

import ray
from wiserl.core.remote import Remote

class Actor(object):
    
    def __init__(self):
        self.registre = None

    def set_registre(self,registre):
        self.registre = registre

    def get_registre(self):
        return self.registre

     
    def _createRemoteActor(self,actor):
        remoteActor = Remote(actor)
        return remoteActor
