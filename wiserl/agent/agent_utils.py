import torch

def get_optimizer(optimizer,model,lr):
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
   