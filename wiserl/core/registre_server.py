# -- coding: utf-8 --

import ray
from wiserl.core.remote import Remote
from .runner_proxy import RunnerProxy
from .agent_proxy import AgentProxy

@ray.remote
class RegistreServer(object):

    def __init__(self):
        self.agent_dict = {}
        self.agent_index = {}
        self.runner_dict = {}
        self.runner_index = {}

    def addAgent(self,name,agent):
        if name in self.agent_dict.keys():
            self.agent_dict[name].append(agent)
        else:
            self.agent_dict[name]=[]
            self.agent_dict[name].append(agent)
        if name not in self.agent_index.keys():
            self.agent_index[name] = 0
  
    def getAgent(self, name):
        if name not in self.agent_dict or  name not in self.agent_dict:
            return None
        index =self.agent_index[name]
        agent = self.agent_dict[name][index]
        index += 1
        if index>=len(self.agent_dict[name]):
            index = 0
            self.agent_index[name]=index
        agent = self._createRemoteAgent(agent)
        return agent

    def getAllAgent(self,name):
        agents = self.agent_dict[name]
        remote_agents = []
        for agent in agents:
            agent = self._createRemoteAgent(agent)
            remote_agents.append(agent)
        return remote_agents
    
    def addRunner(self,name, runner):
        if name in self.runner_dict:
            self.runner_dict[name].append(runner)
        else:
            self.runner_dict[name]=[]
            self.runner_dict[name].append(runner)

        if name not in self.runner_index:
            self.runner_index[name] = 0
        return None


    def getRunner(self, name):
        index =self.runner_index[name]
        runner = self.runner_dict[index]
        index += 1
        if index>=len(self.runner_dict[name]):
            index = 0
            runner = self.runner_index[index]
        runner = self._createRemoteRunner(runner)
        return runner
 
    
    def getAllRunner(self,name):
        print("runner",self.runner_dict)
        runners = self.runner_dict[name]
        # remote_runners = []
        # for runner in runners:
        #     #runner = self._createRemoteRunner(runner)
        #     remote_runners.append(runner)
        return runners

    
      
    def _createRemoteAgent(self,actor):
        remoteAgent = AgentProxy(actor)
        return remoteAgent

    def _createRemoteRunner(self,actor):
        remoteRunner = RunnerProxy(actor)
        return remoteRunner