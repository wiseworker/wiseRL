import numpy as np
import math as M
import seaborn as sns

# physical/external base state of all entites
class EntityState(object):
    def __init__(self):
        # physical position
        self.p_pos = None
        # physical velocity
        self.p_vel = None

# state of agents (including communication and internal/mental state)
class AgentState(EntityState):
    def __init__(self):
        super(AgentState, self).__init__()
        # communication utterance
        self.c = None

# action of the agent
class Action(object):
    def __init__(self):
        # physical action
        self.u = None
        # communication action
        self.c = None

# properties of wall entities
class Wall(object):
    def __init__(self, orient='H', axis_pos=0.0, endpoints=(-1, 1), width=0.1,
                 hard=True):
        # orientation: 'H'orizontal or 'V'ertical
        self.orient = orient
        # position along axis which wall lays on (y-axis for H, x-axis for V)
        self.axis_pos = axis_pos
        # endpoints of wall (x-coords for H, y-coords for V)
        self.endpoints = np.array(endpoints)
        # width of wall
        self.width = width
        # whether wall is impassable to all agents
        self.hard = hard
        # color of wall
        self.color = np.array([0.0, 0.0, 0.0])


# properties and state of physical world entity
class Entity(object):
    def __init__(self):
        # index among all entities (important to set for distance caching)
        self.i = 0
        # name
        self.name = ''
        # properties:
        self.size = 0.050
        # entity can move / be pushed
        self.movable = False
        # entity collides with others
        self.collide = True
        # entity can pass through non-hard walls
        self.ghost = False
        # material density (affects mass)
        self.density = 25.0
        # color
        self.color = None
        # max speed and accel
        self.max_speed = None
        self.accel = None
        # state: including internal/mental state p_pos, p_vel
        self.state = EntityState()
        # mass
        self.initial_mass = 1.0
        # commu channel
        self.channel = None

    @property
    def mass(self):
        return self.initial_mass

# properties of landmark entities
class Landmark(Entity):
    def __init__(self):
        super(Landmark, self).__init__()

# properties of agent entities
class Agent(Entity):
    def __init__(self):
        super(Agent, self).__init__()
        # agent are adversary
        self.adversary = False
        # agent are dummy
        self.dummy = False
        # agents are movable by default
        self.movable = True
        # cannot send communication signals
        self.silent = False
        # cannot observe the world
        self.blind = False
        # physical motor noise amount
        self.u_noise = None
        # communication noise amount
        self.c_noise = None
        # control range
        self.u_range = 1.0
        # state: including communication state(communication utterance) c and internal/mental state p_pos, p_vel
        self.state = AgentState()
        # action: physical action u & communication action c
        self.action = Action()
        # script behavior to execute
        self.action_callback = None
        # zoe 20200420
        self.goal = None

# multi agents world
class World(object):
    def __init__(self):
        # list of agents and entities (can change at execution-time!)
        self.agents = []
        self.landmarks = []
        self.walls = []
        # communication channel dimensionality
        self.dim_c = 0
        # position dimensionality
        self.dim_p = 3
        # color dimensionality
        self.dim_color = 3
        # simulation timestep
        self.dt = 1.0
        # physical damping（阻尼）
        self.damping = 0.01
        # contact response parameters
        self.contact_force = 1e+2
        self.contact_margin = 1e-3
        # cache distances between all agents (not calculated by default)
        self.cache_dists = False
        self.cached_dist_vect = None
        self.cached_dist_mag = None
        # zoe 20200420
        self.world_length = 20
        self.world_step = 0
        self.num_agents = 0
        self.num_landmarks = 0

        # cw paras
        self.mu = 398600.0
        self.relatively_x = 0.0
        self.relatively_y = 0.0
        self.relatively_z = 0.0
        self.altitude = 0.0
        self.start_t = 0.0

        # self.a = self.altitude  # 主星轨道半长轴
        self.a = 0.0
        # self.n = M.sqrt(self.mu / self.a ** 3)
        self.n = 0.0
        # self.n2 = self.n ** 2
        self.n2 = 0.0
        # self.Tc = 2 * M.pi / self.n  # %   主星轨道周期
        self.Tc = 0.0
        # %  主星绝对运动（赤道圆轨道）
        # self.ac = self.a  # ; %   主星半长轴/km
        self.ac = 0.0
        # % ！！！除半长轴外轨道六要素可以进行更改
        self.ec = 0  # 主星偏心率
        self.ic = 0 * M.pi / 180  # %   主星轨道倾角
        self.oc = 0  # 赤经
        self.wc = 0  # ;  %   近地点幅角
        self.fc = 0  # ; %   真近点角
        self.uc = self.wc + self.fc  # ; %   纬度幅角
        # self.MT = np.array([[0, 0, 0, 1, 0, 0],
        #                     [0, 0, 0, 0, 1, 0],
        #                     [0, 0, 0, 0, 0, 1],
        #                     [3 * self.n2, 0, 0, 0, 2 * self.n, 0],
        #                     [0, 0, 0, -2 * self.n, 0, 0],
        #                     [0, 0, -self.n2, 0, 0, 0]])
        self.MT = np.zeros((6, 6), dtype=np.float64)

        self.dt = 1.0
        # self.e = int(self.Tc * self.start_t)
        self.e = 0

    # return all entities in the world
    @property
    def entities(self):
        return self.agents + self.landmarks

    # return all agents controllable by external policies
    @property
    def policy_agents(self):
        return [agent for agent in self.agents if agent.action_callback is None]

    # return all agents controlled by world scripts
    @property
    def scripted_agents(self):
        return [agent for agent in self.agents if agent.action_callback is not None]

    def calculate_distances(self):
        if self.cached_dist_vect is None:
            # initialize distance data structure
            self.cached_dist_vect = np.zeros((len(self.entities),
                                              len(self.entities),
                                              self.dim_p))
            # calculate minimum distance for a collision between all entities （size相加�?
            self.min_dists = np.zeros((len(self.entities), len(self.entities)))
            for ia, entity_a in enumerate(self.entities):
                for ib in range(ia + 1, len(self.entities)):
                    entity_b = self.entities[ib]
                    min_dist = entity_a.size + entity_b.size
                    self.min_dists[ia, ib] = min_dist
                    self.min_dists[ib, ia] = min_dist

        for ia, entity_a in enumerate(self.entities):
            for ib in range(ia + 1, len(self.entities)):
                entity_b = self.entities[ib]
                delta_pos = entity_a.state.p_pos - entity_b.state.p_pos
                self.cached_dist_vect[ia, ib, :] = delta_pos
                self.cached_dist_vect[ib, ia, :] = -delta_pos

        self.cached_dist_mag = np.linalg.norm(self.cached_dist_vect, axis=2)

        self.cached_collisions = (self.cached_dist_mag <= self.min_dists)

    def assign_agent_colors(self):
        n_dummies = 0
        if hasattr(self.agents[0], 'dummy'):
            n_dummies = len([a for a in self.agents if a.dummy])
        n_adversaries = 0
        if hasattr(self.agents[0], 'adversary'):
            n_adversaries = len([a for a in self.agents if a.adversary])
        n_good_agents = len(self.agents) - n_adversaries - n_dummies
        # r g b
        dummy_colors = [(0.25, 0.75, 0.25)] * n_dummies
        # sns.color_palette("OrRd_d", n_adversaries)
        adv_colors = [(0.75, 0.25, 0.25)] * n_adversaries
        # sns.color_palette("GnBu_d", n_good_agents)
        good_colors = [(0.25, 0.25, 0.75)] * n_good_agents
        colors = dummy_colors + adv_colors + good_colors
        for color, agent in zip(colors, self.agents):
            agent.color = color

    # landmark color
    def assign_landmark_colors(self):
        for landmark in self.landmarks:
            landmark.color = np.array([0.25, 0.25, 0.25])

    # update state of the world
    def step(self):
        self.world_step += 1
        for agent in self.scripted_agents:
            agent.action = agent.action_callback(agent, self)

        fc_k = self.n * (self.world_step + self.e - 1)
        rc_0 = self.ac * (1 - self.ec ** 2) / (1 + self.ec * M.cos(fc_k))
        rc_k = rc_0
        Rco_k = np.array([rc_0, 0, 0])

        uc_k = self.wc + fc_k
        Loi = [[M.cos(uc_k) * M.cos(self.oc) - M.sin(uc_k) * M.cos(self.ic) * M.sin(self.oc),
                M.cos(uc_k) * M.sin(self.oc) + M.sin(uc_k) * M.cos(self.ic) * M.cos(self.oc),
                M.sin(uc_k) * M.sin(self.ic)],
               [-M.sin(uc_k) * M.cos(self.oc) - M.cos(uc_k) * M.cos(self.ic) * M.sin(self.oc),
                -M.sin(uc_k) * M.sin(self.oc) + M.cos(uc_k) * M.cos(self.ic) * M.cos(self.oc),
                M.cos(uc_k) * M.sin(self.ic)],
               [M.sin(self.ic) * M.sin(self.oc),
                -M.sin(self.ic) * M.cos(self.oc),
                M.cos(self.ic)]]
        Rci_k = np.dot(Loi, Rco_k)
        Rcui_k = Rci_k / np.linalg.norm(Rci_k)

        #     无控制量
        Y_k = np.array([0, 0, 0, 0, 0, 0])
        Y_k1 = np.array([0, 0, 0, 0, 0, 0])
        B = np.array([0, 0, 0, 0, 0, 0])
        pos_k = np.array([0, 0, 0])
        vel_k = np.array([0, 0, 0])
        act_k = np.array([0, 0, 0])
        for i, agent in enumerate(self.agents):
            if i == 0:
                pos_k = agent.state.p_pos
                vel_k = agent.state.p_vel
                act_k = agent.action.u

        Y_k[0:3] = pos_k
        Y_k[3:] = vel_k
        B[3:] = act_k
        dYk1 = np.dot(self.MT, Y_k) + B
        dYk2 = np.dot(self.MT, (Y_k + (self.dt * dYk1 / 2))) + B
        dYk3 = np.dot(self.MT, (Y_k + (self.dt * dYk2 / 2))) + B
        dYk4 = np.dot(self.MT, (Y_k + (self.dt * dYk3))) + B
        Y_k1 = Y_k + self.dt / 6 * (dYk1 + 2 * dYk2 + 2 * dYk3 + dYk4)
        for i, agent in enumerate(self.agents):
            if i == 0:
                agent.state.p_pos = Y_k1[0:3]
                agent.state.p_vel = Y_k1[3:]





