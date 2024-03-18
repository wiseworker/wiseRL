# -- coding: utf-8 --
import ray
import time
import uuid
from  .registre_server import RegistreServer
from  .runner import Runner
from  .agent import Agent
import configparser
import os


#ray.init()

class WiseRL(object):
    def __init__(self, use_ray=False):
        self.use_ray = use_ray
        if self.use_ray:
            self.registre = RegistreServer.remote()
        else:
            self.registre = None

    def make_runner(self, name, Runner, args=None, num=1,resource=None):
        for i in range(num):
            runner =ray.remote(Runner)
            if resource != None:
                runner = runner.options(**resource)
            runner =runner.remote(args,local_rank=i)
            self.registre.add_runner.remote(name,runner)
            runner.set_registre.remote(self.registre)
        retref = self.registre.get_all_runner.remote(name)
        return ray.get(retref)

    def get_runner(self, name):
        return ray.get(self.registre.get_runner.remote(name))

    def make_agent(self,name,agent_class,config=None,num=1, sync=True, resource=None) :
        copy_name= None
        copy_agent = None
        if sync == False:
            copy_name="_wise_copy_" + name + str(uuid.uuid1())
            copy_agent =ray.remote(agent_class)
            if resource != None:
                copy_agent = copy_agent.options(**resource)
            copy_agent = copy_agent.remote(config,sync)
            self.registre.add_agent.remote(copy_name,copy_agent,copy_agent)
            copy_agent.set_registre.remote(self.registre) 

        for i in range(num):
            agent = ray.remote(agent_class)
            if resource != None:
                agent = agent.options(**resource)
            agent = agent.remote(config,sync)
            self.registre.add_agent.remote(name,agent,copy_agent)
            agent.set_registre.remote(self.registre)
            if sync == False:
                agent.set_copy_name.remote(copy_name)
        retref = self.registre.get_all_agent.remote(name)
        return ray.get(retref)

    def get_agent(self, name):
        for i in range(100):
            agent =ray.get(self.registre.get_agent.remote(name))
            if agent != None:
                return agent
            time.sleep(1)
        raise ValueError(name + " agent not found ,please check that the name is correct")

    def start_all_runner(self, runners):
        results =[]
        for runner in runners:
            ref = runner.run.remote()
            results.append(ref)
        ray.get(results)