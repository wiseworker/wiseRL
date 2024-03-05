import torch
from wiserl.net.nn_net import ActorDiscretePPO, ActorPPO, ActorDDPG, CriticDDPG
from wiserl.net.nn_net import CriticPPO
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_optimizer(optimizer, model, lr):
    if optimizer == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr)
    if optimizer == "mbgd":
        return torch.optim.SGD(model.parameters(), lr=lr)
    if optimizer == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr)
    if optimizer == "adagrad":
        return torch.optim.Adagrad(model.parameters(), lr=lr)
    if optimizer == "momentum":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    if optimizer == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr)
    raise Exception("no such optimizer")
   
def make_actor_net(net_name, config):
    if net_name == "dis_nn":
        return ActorDiscretePPO(config.net_dims, config.state_dim, config.action_dim)
    if net_name == "nn":
        return ActorPPO(config.net_dims, config.state_dim, config.action_dim)
    if net_name == "ddpg_nn":
        return ActorDDPG(config.state_dim, config.hidden_dim, config.action_dim, config.action_bound).to(device)
    raise Exception("no such actor network")

def make_critic_net(net_name, config):
    if net_name == "nn":
        return CriticPPO(config.net_dims, config.state_dim, config.action_dim)
    if net_name == "ddpg_nn":
        return CriticDDPG(config.state_dim, config.hidden_dim, config.action_dim).to(device)
    raise Exception("no such critic network")