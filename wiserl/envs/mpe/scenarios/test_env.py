import numpy as np
import math as M
from onpolicy.envs.mpe.test_core import World, Agent, Landmark
from onpolicy.envs.mpe.scenario import BaseScenario


class Scenario(BaseScenario):
    def make_world(self, args):
        world = World()
        world.world_length = args.episode_length
        # set any world properties first

        world.relatively_x = args.relatively_x
        world.relatively_y = args.relatively_y
        world.relatively_z = args.relatively_z
        world.altitude = args.altitude
        world.start_t = args.start_t

        world.a = world.altitude
        world.n = np.sqrt(world.mu / world.a ** 3)
        world.n2 = world.n ** 2
        world.Tc = 2 * M.pi / world.n  # %   主星轨道周期
        # %  主星绝对运动（赤道圆轨道）
        world.ac = world.a  # ; %   主星半长轴/km
        world.MT = np.array([[0, 0, 0, 1, 0, 0],
                             [0, 0, 0, 0, 1, 0],
                             [0, 0, 0, 0, 0, 1],
                             [3 * world.n2, 0, 0, 0, 2 * world.n, 0],
                             [0, 0, 0, -2 * world.n, 0, 0],
                             [0, 0, -world.n2, 0, 0, 0]], dtype=np.float64)

        world.e = int(world.Tc * world.start_t)

        world.dim_c = 3
        world.num_agents = 1
        world.num_landmarks = 1
        world.collaborative = True
        # add agents
        world.agents = [Agent() for i in range(world.num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.size = 0.15
        # add landmarks
        world.landmarks = [Landmark() for i in range(world.num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # random properties for agents
        world.assign_agent_colors()

        world.assign_landmark_colors()

        x_pos = world.relatively_x
        y_pos = world.relatively_y
        z_pos = world.relatively_z

        # set random initial states
        for agent in world.agents:
            # agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            # agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.p_pos = np.array([x_pos, y_pos, z_pos])
            agent.state.p_vel = np.array([world.n * y_pos / 2, -2 * world.n * x_pos, world.n * y_pos])
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np.zeros(world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)

    def benchmark_data(self, agent, world):
        rew = 0
        collisions = 0
        occupied_landmarks = 0
        min_dists = 0
        for l in world.landmarks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos)))
                     for a in world.agents]
            min_dists += min(dists)
            rew -= min(dists)
            if min(dists) < 0.1:
                occupied_landmarks += 1
        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 1
                    collisions += 1
        return (rew, collisions, min_dists, occupied_landmarks)

    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions
        rew = 0
        for l in world.landmarks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos)))
                     for a in world.agents]
            if min(dists) >= 2:
                rew -= min(dists)
            elif min(dists) < 2:
                rew -= min(dists) * 100

        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 1
        return rew

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:  # world.entities:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        # entity colors
        entity_color = []
        for entity in world.landmarks:  # world.entities:
            entity_color.append(entity.color)
        # communication of all other agents
        comm = []
        other_pos = []
        for other in world.agents:
            if other is agent:
                continue
            comm.append(other.state.c)
            other_pos.append(other.state.p_pos - agent.state.p_pos)
        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + comm)
