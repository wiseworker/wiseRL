import torch
from wiserl.net.nn_net import ActorDiscretePPO, ActorContinuousPPO, ActorDDPG, ActorSAC, ActorTD3, ActorTRPO
from wiserl.net.nn_net import CriticPPO, CriticPPO2, CriticDDPG, CriticSAC, SACQnet, CriticTD3, CriticTRPO
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

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
    if net_name == "ppo_nn":
        return ActorDiscretePPO(config["net_dims"], config["state_dim"], config["action_dim"]).to(device)
    if net_name == "ppo2_nn":
        return ActorContinuousPPO(config["state_dim"], config["action_dim"], config["net_width"]).to(device)
    if net_name == "ddpg_nn":
        return ActorDDPG(config["net_dims"], config["state_dim"], config["action_dim"], config["action_bound"]).to(device)
    if net_name == "sac_nn":
        return ActorSAC(config["state_dim"]).to(device)
    if net_name == "td3_nn":
        return ActorTD3(config["state_dim"], config["action_dim"], config["action_bound"]).to(device)
    if net_name == "trpo_nn":
        return ActorTRPO(config["state_dim"], config["action_dim"]).to(device)
    raise Exception("no such actor network")

def make_critic_net(net_name, config):
    if net_name == "ppo_nn":
        return CriticPPO(config["net_dims"], config["state_dim"], config["action_dim"]).to(device)
    if net_name == "ppo2_nn":
        return CriticPPO2(config["state_dim"], config["net_width"]).to(device)
    if net_name == "ddpg_nn":
        return CriticDDPG(config["net_dims"], config["state_dim"], config["action_dim"]).to(device)
    if net_name == "sac_nn":
        return CriticSAC(config["state_dim"]).to(device)
    if net_name == "q_nn":
        return SACQnet(config["state_dim"], config["action_dim"]).to(device)
    if net_name == "td3_nn":
        return CriticTD3(config["state_dim"], config["action_dim"]).to(device)
    if net_name == "trpo_nn":
        return CriticTRPO(config["state_dim"]).to(device)
    raise Exception("no such critic network")