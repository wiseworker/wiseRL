# -- coding: utf-8 --

import ray
import time
import uuid
from  .registre_server import RegistreServer

#ray.init()

class WiseRL(object):
    def __init__(self):
        self.registre = RegistreServer.remote()

    def make_agent(self,name,agent_class,config=None,num=1, sync=True, resource=None) :
        copy_name= None
        copy_agent = None
        if sync == False:
            copy_name="_wise_copy_" + name + str(uuid.uuid1())
            copy_agent =ray.remote(agent_class)
            if resource != None:
                copy_agent = copy_agent.options(**resource)
            copy_agent = copy_agent.remote(config,sync)
            self.registre.addAgent.remote(copy_name,copy_agent,copy_agent)
            copy_agent.set_registre.remote(self.registre) 

        for i in range(num):
            agent =ray.remote(agent_class)
            if resource != None:
                agent = agent.options(**resource)
            agent = agent.remote(config,sync)
            import pdb;pdb.set_trace()
            self.registre.addAgent.remote(name,agent,copy_agent)
            agent.set_registre.remote(self.registre)
            if sync == False:
                agent.set_copy_name.remote(copy_name)
        retref = self.registre.getAllAgent.remote(name)
        return ray.get(retref)
    
    def make2_agent(self,name,agent_class,num=1,sync=True,resource=None, **kwargs) :
        copy_name= None
        copy_agent = None
        if sync == False:
            copy_name="_wise_copy_" + name + str(uuid.uuid1())
            copy_agent =ray.remote(agent_class)
            if resource != None:
                copy_agent = copy_agent.options(**resource)
            copy_agent = copy_agent.remote(sync=sync, **kwargs)
            self.registre.addAgent.remote(copy_name,copy_agent,copy_agent)
            copy_agent.set_registre.remote(self.registre) 

        for i in range(num):
            agent =ray.remote(agent_class)
            if resource != None:
                agent = agent.options(**resource)
            agent = agent.remote(sync=sync, **kwargs)
            self.registre.addAgent.remote(name,agent,copy_agent)
            agent.set_registre.remote(self.registre)
            if sync == False:
                agent.set_copy_name.remote(copy_name)
        retref = self.registre.getAllAgent.remote(name)
        return ray.get(retref)

    def make_ragent(self,name,agent_class,args=None,policy=None,sync=True,num=1,resource=None) :
        copy_name= None
        copy_agent = None
        if sync == False:
            copy_name="_wise_copy_" + name + str(uuid.uuid1())
            copy_agent =ray.remote(agent_class)
            if resource != None:
                copy_agent = copy_agent.options(**resource)
            copy_agent = copy_agent.remote(args=args, policy=policy,sync=sync)
            self.registre.addAgent.remote(copy_name,copy_agent,copy_agent)
            copy_agent.set_registre.remote(self.registre)

        for i in range(num):
            agent =ray.remote(agent_class)
            if resource != None:
                agent = agent.options(**resource)
            agent = agent.remote(args=args, policy=policy, sync=sync)
            self.registre.addAgent.remote(name,agent,copy_agent)
            agent.set_registre.remote(self.registre)
            if sync == False:
                agent.set_copy_name.remote(copy_name)
        retref = self.registre.getAllAgent.remote(name)
        return ray.get(retref)

    def make_runner(self, name,Runner,args=None,num =1,resource=None):
        for i in range(num):
            runner =ray.remote(Runner)
            if resource != None:
                runner = runner.options(**resource)
            runner =runner.remote(args,local_rank=i)
            self.registre.add_runner.remote(name,runner)
            runner.set_registre.remote(self.registre)
        retref = self.registre.get_all_runner.remote(name)
        return ray.get(retref)

    def makeRunner(self, name, Runner, args=None, num =1, resource=None):
        for i in range(num):
            #import pdb;pdb.set_trace()
            runner = ray.remote(Runner)
            #########################################
            if resource != None:
                runner = runner.options(**resource)
            #########################################
            runner = runner.remote(args, local_rank=i)
            self.registre.addRunner.remote(name,runner)
            runner.set_registre.remote(self.registre)
        retref = self.registre.getAllRunner.remote(name)
        return ray.get(retref)

    def getRunner(self, name):
        return ray.get(self.registre.getRunner.remote(name))

    def makeDDPGAgent(self, name, agent_class, config=None, num=1, sync=True):
        copy_name = None
        if sync is not True:
            copy_name = "_wise_copy_" + name + str(uuid.uuid1())
        for _ in range(num):
            agent = ray.remote(agent_class).remote(config, sync)
            self.registre.addAgent.remote(name, agent)
            agent.set_registre.remote(self.registre)
            if sync is not True:
                agent.setCopyName.remote(copy_name)
                copy_agent = ray.remote(agent_class).remote(config, sync)
                self.registre.addAgent.remote(copy_name, copy_agent)
                copy_agent.set_registre.remote(self.registre)
        retref = self.registre.getAllAgent.remote(name)
        return ray.get(retref)

    def getAgent(self, name):
        for i in range(100):
            agent =ray.get(self.registre.getAgent.remote(name))
            if agent != None:
                return agent
            time.sleep(1)
        raise ValueError(name + " agent not found ,please check that the name is correct")

    def startAllRunner(self, runners):
        results =[]
        for runner in runners:
            ref = runner.run.remote()
            results.append(ref)
        ray.get(results)