import torch
from wiserl.core.runner import Runner
from wiserl.agent.sac_agent.sac_agent import SACAgent
from wiserl.net.sac_net import SACActor, SACCritic, SACQnet
from wiserl.core.wise_rl import WiseRL
from gymnasium.wrappers import RescaleAction
import wiserl.agent.sac_agent.config as cfg
import gymnasium as gym
import time
import ray


class GymRunner(Runner):
    def __init__(self, local_rank=0):
        super().__init__()
        self.env = gym.make("Pendulum-v1")
        self.env = RescaleAction(self.env, min_action=-1.0, max_action=1.0)
        print("rank=", local_rank)
        self.n_actions = self.env.action_space.shape[0]
        self.n_states = self.env.observation_space.shape[0]
        self.rank = local_rank
        if self.rank == 0:
            # net = DQNNet(N_STATES,N_ACTIONS)
            wise_rl.makeSACAgent(name="sac_agent", agent_class=SACAgent, actor_net=SACActor,
                                 value_net=SACCritic, q_net=SACQnet, n_states=self.n_states,
                                 n_actions=self.n_actions, sync=True)
        self.agent = wise_rl.getAgent("sac_agent")

    def run(self):

        print("====================================")
        print("Collection Experience...")
        print("====================================")
        start_time = time.time()
        ep_r = 0
        for i in range(cfg.MAX_EPOCH):
            state, _ = self.env.reset()
            for t in range(cfg.max_steps):
                action = self.agent.choseAction(state)
                # next_state, reward, done, _, _ = env.step(np.float32(action))
                next_state, reward, done, _, _ = self.env.step(action)
                ep_r += reward
                self.agent.update(state, action, reward, next_state, done)
                state = next_state
                if done or t == 199:
                    if self.rank == 0:
                        print("Ep_i: {},  ep_r: {}, time_step: {}".format(i, ep_r, t))
                    # print("Ep_i: {},  ep_r: {}, time_step: {}".format(i, ep_r, t))
                    break
                # print(step)
            if ep_r > -200:
                if self.rank == 0:
                    print("train finish")
                    print("training time= ", time.time() - start_time)
                break
            # if i % cfg.log_interval == 0:
            #     self.agent.save()
            ep_r = 0


if __name__ == '__main__':
    print("use GPU to train" if torch.cuda.is_available() else "use CPU to train")
    ray.init(address="auto")
    # ray.init(local_mode=True)
    wise_rl = WiseRL()
    runners = wise_rl.makeRunner("runner", GymRunner, num=1)
    wise_rl.startAllRunner(runners)
