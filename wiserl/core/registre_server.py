# -- coding: utf-8 --

import ray
from .runner_proxy import RunnerProxy
from .agent_proxy import AgentProxy

@ray.remote
class RegistreServer(object):

    def __init__(self):
        self.agent_dict = {}
        self.agent_copy_dict = {}
        self.agent_index = {}
        self.runner_dict = {}
        self.runner_index = {}

    def add_agent(self,name,agent, copy_agent=None):
        if name in self.agent_dict.keys():
            self.agent_dict[name].append(agent)
        else:
            self.agent_dict[name]=[]
            self.agent_dict[name].append(agent)
        self.agent_copy_dict[name] = copy_agent
        if name not in self.agent_index.keys():
            self.agent_index[name] = 0
  
    def get_agent(self, name):
        if name not in self.agent_dict or  name not in self.agent_dict:
            return None
        index =self.agent_index[name]
        agent = self.agent_dict[name][index]
        index += 1
        if index>=len(self.agent_dict[name]):
            index = 0
            self.agent_index[name]=index
        copy_agent = None
        if name in self.agent_copy_dict.keys():
            copy_agent = self.agent_copy_dict[name]
        agent = self._create_remote_agent(agent, copy_agent)
        return agent

    def get_all_agent(self,name):
        agents = self.agent_dict[name]
        remote_agents = []
        copy_agent = None
        if name in self.agent_copy_dict.keys():
            copy_agent = self.agent_copy_dict[name]
        for agent in agents:
            agent = self._create_remote_agent(agent, copy_agent)
            remote_agents.append(agent)
        return remote_agents
    
    def add_runner(self,name, runner):
        if name in self.runner_dict:
            self.runner_dict[name].append(runner)
        else:
            self.runner_dict[name]=[]
            self.runner_dict[name].append(runner)

        if name not in self.runner_index:
            self.runner_index[name] = 0
        return None


    def get_runner(self, name):
        index =self.runner_index[name]
        runner = self.runner_dict[index]
        index += 1
        if index>=len(self.runner_dict[name]):
            index = 0
            runner = self.runner_index[index]
        runner = self._create_remote_runner(runner)
        return runner

    def get_all_runner(self,name):
        print("runner",self.runner_dict)
        runners = self.runner_dict[name]
        return runners

    def _create_remote_agent(self,agent, copy_agent):
        remoteAgent = AgentProxy(agent ,copy_agent)
        return remoteAgent

    def _create_remote_runner(self,runner):
        remoteRunner = RunnerProxy(runner)
        return remoteRunner