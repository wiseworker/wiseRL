<h1 align="center">
  <a href="https://github.com/wiseworker/wiseRL/blob/bxq"><img src="https://raw.githubusercontent.com/wiseworker/wiseRL/bxq/logo.png" width=700  alt="Wiseworker"></a>
</h1>


<h4 align="center">A State-of-the-art Distributed Open Source Framework.</h4>

<p align="center">
    <a href="https://github.com/wiseworker/wiseRL/commits">
    <img src="https://img.shields.io/github/last-commit/wiseworker/wiseRL.svg?style=flat-square&logo=github&logoColor=white"
         alt="GitHub last commit">
    <a href="https://github.com/wiseworker/wiseRL/issues">
    <img src="https://img.shields.io/github/issues-raw/wiseworker/wiseRL.svg?style=flat-square&logo=github&logoColor=white"
         alt="GitHub issues">
    <a href="https://github.com/wiseworker/wiseRL/pulls">
    <img src="https://img.shields.io/github/issues-pr-raw/wiseworker/wiseRL.svg?style=flat-square&logo=github&logoColor=white"
         alt="GitHub pull requests">
</p>
<p align="center">
  <a href="#bulb-about">About</a> •
  <a href="#hello world">Hello World</a> •
  <a href="#classical_building-framework-architecture">Framework Architecture</a> •
  <a href="#computer-installation">Installation</a> •
  <a href="#personalization">Personalization</a> •
  <a href="#lock_with_ink_pen-algorithm">Algorithm</a> •
  <a href="#credits">Credits</a> •
  <a href="#support">Support</a> •
  <a href="#license">License</a>
</p>


---

## :bulb: About

The main objective of our open-source framework **wiseRL** is facilitate **distributed reinforcement learning** algorithms by implementing distributed data sampling and centralized training. It serves to elucidate the principles of  interpret the principles of [**Reinforcement Learning**]([Reinforcement Learning (mit.edu)](https://mitpress.mit.edu/9780262039246/reinforcement-learning/)).

> Reinforcement learning is a computational approach to learning whereby an agent tries to maximize the total amount of reward it receives while interacting with a complex, uncertain environment.  
> -- Sutton and Barto

 In the main function, WiseRL is utilized to create and manage agents for **single or multi-agent algorithms**. **Parallel processing** is implemented using the Ray library for training, and **multi-processing** is used for interaction with the environment. This framework combines reinforcement learning algorithm implementation with Ray's parallel processing capabilities and multi-processing, simplifying the development and experimentation of distributed reinforcement learning algorithms.

## Hello World

In this section, we'll use "Hello World" to guide you through WiseRL tasks and actors, as well as how to work with instance objects.

### Getting Started

The first step is to import and initialize Ray:

```
import ray
ray.init()
```

### Running a task

WiseRL lets you run functions as remote tasks in the cluster. To do this, you decorate your function with `@ray.remote` to declare that you want to run this function remotely. Then, you call that function with `.remote()` instead of calling it normally. This remote call returns a future, a so-called Ray *object reference*, that you can then fetch with `ray.get`:

```
# Define a print task.
@ray.remote
def run(x):
	return "Hello World"

# Launch three parallel print tasks.
rst = [run.remote(i) for i in range(3)]

# Retrieve results.
print(ray.get(rst))
# -> ["Hello World", "Hello World", "Hello World"]
```

### Calling a Runner

Ray provides actors to allow you to parallelize computation across multiple actor instances. When you instantiate a class that is a Ray actor, Ray will start a remote instance of that class in the cluster. This actor can then execute remote method calls and maintain its own internal state:

```
# Define the GymRunner actor.
@ray.remote
class GymRunner:
    def __init__(self):
        self.i = 0

    def run(self, value):
        self.i += value
        return "Hello World " + str(self.i) 

# Create a GymRunner actor.
g = GymRunner.remote()

# Submit calls to the actor. These calls run asynchronously but in 
# submission order on the remote actor process.
for _ in range(100):
    g.run.remote(1)

# Retrieve final actor state.
print(ray.get(g.run.remote()))
# -> Hello World 100
```

## :classical_building: Installation

##### Downloading and installing steps:

_At the Terminal_

1. Initialize your Python interpreter with a version of Python >= 3.7 and install the corresponding PyTorch.

   ```
   conda create -n wise python=3.8
   conda activate wise
   ```

   You can choose either the CPU version of PyTorch or the CUDA version of PyTorch, depending on your device.

   Follow the installation instructions for PyTorch **[here](https://pytorch.org)**.

2. Enter the command:

   `git clone https://github.com/wiseworker/wiseRL.git`

   > [!NOTE]
   >
   > Remember to turn off your vpn when using command 'git clone'

_Or you can choose_

1. **[Download](https://github.com/wiseworker/wiseRL/archive/master.zip)** the latest version of wiseRL.
2. Open the _archive_ and **extract** the contents of the `wiseRL-main` folder into the root path folder:
   `wiseRL-main/`

_Then_

enter the command at the terminal

```
cd wiseRL
pip install pettingzoo pygame
pip install -e . # run setup.py
cd example
ray start --head --port=6379
python ddpg.py 
```

## Nanny Tutorial

You have the freedom to develop your own algorithms within our framework.

### Start with GymRunner

#### Initialization 

1. **Start by adding a new algorithm folder in the `example` directory**

   - create a `.py` file within it to serve as the main function. 

   - In the `GymRunner` initialization file, create the environment (`env`):

     ```
     self.env = gym.make("Acrobot-v1")
     ```

     You can obtain environment information using APIs like:

     ```
     self.env.observation_space.shape[0] # dimension of the observation space
     self.env.action_space.shape[0]      # dimension of the action space
     self.env.action_space.high[0]		# maximum value of the action space
     ```

     These may vary depending on the environment wrapper. You can also customize your own environment.

   

   

   

   

   - For improved sampling efficiency, consider wrapping the environment for multiprocessing to enhance the interaction and collection of environment information. 

     > If your algorithm utilizes a **Replay Buffer**, be mindful of the **dimension** of the information collected, especially due to the parallel sampling of environments. 

2. **Create your agent and get its information**

   - Use the `make_agent` function to create an agent when `self.rank == 0`.

     ```
     if self.rank == 0:
     	wise_rl.make_agent(name='ddpg_agent', agent_class=DDPGAgent, sync=True, **vars(self.config))
     ```

     Here, you need to pass along several pieces of information, including the agent class (DDPGAgent, which will be discussed later), a synchronization or asynchrony argument, and all hyperparameters as a dictionary (**vars). This will be beneficial for your usage in the Agent class.

     > We create an agent only once and retrieve its information in every process.

   - Then use `get_agent` to access the agent's model information. 

     ```
     self.agent = wise_rl.getAgent('ddpg_agent')
     ```

     This `self.agent` can be directly invoked later on.











#### Run

1. **Initialize the Training Environment for Each Episode**

   Each new episode starts with resetting the environment and obtaining the initial state.

   ```
   state = self.env.reset()[0]
   ```

   This step ensures that each episode begins from a fresh starting point in the environment, allowing the agent to learn from a diverse range of situations.

2. **Running the Training Loop**

   Within each episode, perform the following steps in a loop until the episode ends or the maximum number of steps (`max_step`) is reached:

   ```
   for t in range(self.config.max_step):
   	action = self.agent.choose_action(state)
     next_state, reward, done, info, _ = self.env.step(action)
     ep_r += reward
     self.agent.update(state, action, reward, next_state, done)
     state = next_state
     if done or t == self.config.max_step - 1:
       break
   ```

   - Action Selection: The agent selects an action based on the current state.
   - Environment Interaction: The agent performs the action in the environment, which returns the next state, reward, and a boolean indicating whether the episode has ended (`done`).
   - Agent Update: The agent updates its policy based on the transition (current state, action, reward, next state).
   - State Transition: The current state is updated to the next state for the next iteration.
   - Episode Termination: The loop breaks if the episode ends (`done` is `True`) or the maximum number of steps is reached.

3. **Monitoring Progress**

   After each episode, update the training information and periodically report progress:

   ```
   print("Episode:{}--train_info:{}".format(i, np.mean(train_info[-20:])))
   plt_list.append(np.mean(train_info[-20:]))
   ```

   - After each episode, the average reward of the last 20 episodes is printed. This helps monitor the training progress and evaluate the agent's performance over time.
   - The average rewards are appended to a list (`plt_list`) for plotting the training progress graphically.

### Make your own Agent Class

1. **Initialization**

   To begin, navigate to the `wiserl/agent` directory. Here, you'll create a new agent class that inherits from the base `Agent` class. This base class should support both synchronous and asynchronous model parameter updates, which are crucial for distributed learning scenarios.

   **Create the DDPGAgent Class**:

   ```
   class DDPGAgent(Agent):
   	def __init__(self, sync=True, **kwargs):
     	super().__init__(sync)
       self.__dict__.update(kwargs)
   ```

   In this class:

   - The constructor initializes the agent, optionally synchronizing model parameters.

   - `self.__dict__.update(kwargs)` allows for flexible parameter initialization, accommodating any number of additional keyword arguments.

   **Initialize Components**:

   ```
   self.actor = make_actor_net()
   self.critic = make_critic_net()
   self.replay_buffer = ReplayBuffer(self.buffer_size, self.state_dim, self.action_dim)
   self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr_a)
   self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr_c)
   ```

   > Here, you initialize the actor and critic networks, a replay buffer for storing experiences, and optimizers for training the networks. Custom network definitions should be placed in the `net.py` file.

2. **Choose Action**

   For selecting actions based on the current state:

   ```
   state = torch.FloatTensor(state).unsqueeze(0).to(device)
   action = self.actor(state)
   return action.detach().cpu().numpy()[0]
   ```

   This method transforms the current state into a tensor, processes it through the actor network to obtain the action, and then returns the action as a NumPy array.

3. **Update**

   The update method is where the core of the DDPG algorithm's learning logic is implemented. It generally involves sampling transitions from the replay buffer and performing gradient updates on both the actor and critic networks.

   - Sample Transitions:

   Start by sampling a batch of transitions from the replay buffer.

   - Algorithm Customization:

   Implement the specific learning updates for the DDPG algorithm, including calculating the loss for both the actor and critic networks and performing backpropagation.

   - Synchronize Models (if using asynchronous updates):

   ```
   def _sync_model(self):
     param = self.actor.state_dict()
     if device.type != "cpu":
     	for name, mm in param.items():
       	param[name]= mm.cpu()
     self._fire(param)
   ```

   This method synchronizes the model parameters across different instances or nodes, ensuring consistency in distributed setups.

### Main

#### Hyperparameter

The `argparse` library in Python is commonly used for handling command-line arguments. It's particularly useful in machine learning projects for setting and adjusting hyperparameters externally without modifying the source code.

**_First_**, initialize the Parser:

```
parser = argparse.ArgumentParser("Hyperparameter Setting for DDPG")
parser.add_argument("--net_dims", default=(256, 128), help="The number of neurons in hidden layers of the neural network")
```

In this snippet, we initialize an `ArgumentParser` object with a description. Then, we add a hyperparameter `--net_dims` that expects two integers, representing the number of neurons in the hidden layers of a neural network. The `default=(256, 128)` specifies default values if none are provided externally.

**_Then_**:

```
args = parser.parse_args()
```

Here, `parse_args()` processes the command-line arguments. The `args` object now has an attribute `net_dims`, accessible via `args.net_dims`, containing the provided values for the neural network dimensions.

**_Later_**, you might encounter situations where you need to add more hyperparameters after the initial parsing. You can add these dynamically using `setattr`

```
setattr(args, 'state_dim', self.env.observation_space.shape[0])
```

In this example, `setattr` is used to add a new attribute `state_dim` to the `args` object. This is particularly useful for parameters like `state_dim`, which depends on the environment's observation space and can only be set after the environment is initialized.

#### Runner

You can create a set of runner objects (`runners`) for executing RL environments:

```
runners = wise_rl.makeRunner("runner", GymRunner, args, num=2)
wise_rl.startAllRunner(runners)
```

By adjusting the `num` parameter, you can control the number of parallel processes. This makes scaling your experiments as simple as modifying a single argument, provided you have the resources to support the increased parallelism.

**_Alternatively_**, you can choose to run your code locally without Ray. 

> If you choose not to use Ray, your code will execute on a single machine, running one process at a time if not otherwise parallelized. This approach is simpler and doesn't require managing a distributed system, making it attractive for development, debugging, or when resources are limited.

```
runners = GymRunner()
runners.run()
```

## :computer: Framework Architecture

```
WiseRL
├── __init__.py
├── example
│   ├── ddpg
│   │   └── ddpg.py
│   ├── dqn
│   │   ├── dqn.py
│   │   └── dqn_cnn.py
│   ├── mappo
│   │   ├── mpe.py
│   │   └── mpe2.py
│   ├── ppo
│   │   ├── ppo_continuous.py
│   │   ├── ppo_discrete.py
│   └── sac
│       └── sac.py
├── setup.py
├── wiserl
│   ├── agent
│   │   ├── ddpg_agent
│   │   │   ├── __pycache__
│   │   │   ├── config.py
│   │   │   └── ddpg_agent.py
│   │   ├── dqn_agent
│   │   │   ├── doul_dqn_agent.py
│   │   │   └── dqn_agent.py
│   │   ├── ppo_agent
│   │   │   ├── ppo2_agent.py
│   │   │   ├── ppo_agent.py
│   │   └── sac_agent
│   │       └── sac_agent.py
│   ├── core
│   │   ├── __init__.py
│   │   ├── actor.py
│   │   ├── agent.py
│   │   ├── agent_proxy.py
│   │   ├── registre_server.py
│   │   ├── remote.py
│   │   ├── runner.py
│   │   ├── runner_proxy.py
│   │   └── wise_rl.py
│   ├── net
│   │   ├── nn_net.py
│   └── utils
│       ├── __pycache__
│       ├── mem_store.py
│       ├── normalization.py
│       └── replay_buffer.py
└── wiserl.egg-info
```

## :lock_with_ink_pen: Algorithm

| Algorithm      | Multi-processing | Parallel-processing |
| -------------- | :--------------: | :-----------------: |
| DQN            |        ✔️         |          ✔️          |
| PPO-discrete   |        ✔️         |          ✔️          |
| PPO-continuous |        ✔️         |          ✔️          |
| DDPG           |        ✔️         |          ✔️          |
| SAC            |        ✔️         |          ✔️          |
| TRPO           |        ✔️         |          ✔️          |
| TD3            |        ✔️         |          ✔️          |
| MADDPG         |        ❌         |          ✔️          |
| MAPPO          |        ❌         |          ✔️          |
| QMIX           |        ❌         |          ✔️          |
| VDN            |        ❌         |          ✔️          |

## Credits

| <img src="https://raw.githubusercontent.com/wiseworker/wiseRL/bxq/Wise.png" width = 150 height = 150 /> |
| :----------------------------------------------------------: |
|                        **Wiseworker**                        |

## Support

Reach out to me via the **[profile addresses](https://github.com/wiseworker)**.

## License

[![License: CC0-1.0](https://img.shields.io/badge/License-CC0%201.0-lightgrey.svg)](https://tldrlegal.com/license/creative-commons-cc0-1.0-universal)