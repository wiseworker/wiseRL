net_dims = (256, 128)
max_train_steps = 200000
evaluate_freq = 5000
save_freq = 20
batch_size = 256
mini_batch_size = 64
hidden_width = 128
optimizer = "Adam"
lr_a = 0.0003
lr_c = 0.0003
gamma = 0.99
lamda  = 0.95
epsilon = 0.2
K_epochs = 10
use_adv_norm=True  #"Trick 1:advantage normalization"
use_state_norm = True #"Trick 2:state normalization"
use_reward_norm=False #"Trick 3:reward normalization"
use_reward_scaling = True #"Trick 4:reward scaling"
entropy_coef =0.01 #Trick 5: policy entropy"
use_lr_decay=True #"Trick 6:learning rate Decay"
use_grad_clip=True #"Trick 7: Gradient clip"
use_orthogonal_init=True #"Trick 8: orthogonal initialization"
set_adam_eps=True #"Trick 9: set Adam epsilon=1e-5"
use_tanh = True #"Trick 10: tanh activation function"

def init_params(config):
    net_dims = (256, 128) if config.get("net_dims") == None else config.get("net_dims")
    max_train_steps = 200000 if config.get("max_train_steps") == None else config.get("max_train_steps")
    evaluate_freq = 5000 if config.get("evaluate_freq") == None else config.get("evaluate_freq")
    save_freq = 20 if config.get("save_freq") == None else  config.get("save_freq")
    batch_size = 2048 if config.get("batch_size") == None else  config.get("batch_size")
    mini_batch_size = 64 if config.get("mini_batch_size") == None else  config.get("mini_batch_size")
    hidden_width = 128 if config.get("hidden_width") == None else  config.get("hidden_width")
    optimizer = "Adam" if config.get("optimizer") == None else config.get("optimizer")
    lr_a = 0.003 if config.get("lr_a") == None else  config.get("lr_a")
    lr_c = 0.003 if config.get("lr_c") == None else  config.get("lr_c")
    gamma =  0.99 if config.get("gamma") == None else  config.get("gamma")
    lamda =  0.95 if config.get("lamda") == None else  config.get("lamda")
    epsilon = 0.2 if config.get("epsilon") == None else  config.get("epsilon")
    K_epochs = 10 if config.get("K_epochs") == None else  config.get("K_epochs")
    use_adv_norm = True if config.get("use_adv_norm") == None else  config.get("use_adv_norm")
    use_state_norm = True if config.get("use_state_norm") == None else  config.get("use_state_norm")
    use_reward_norm = False if config.get("use_reward_norm") == None else  config.get("use_reward_norm")
    entropy_coef = True if config.get("entropy_coef") == None else  config.get("entropy_coef")
    use_lr_decay = 0.01  if config.get("use_lr_decay") == None else  config.get("use_lr_decay")
    use_grad_clip = True if config.get("use_grad_clip") == None else  config.get("use_grad_clip")
    use_orthogonal_init = True if config.get("use_orthogonal_init") == None else  config.get("use_orthogonal_init")
    set_adam_eps = True if config.get("set_adam_eps") == None else  config.get("set_adam_eps")
    use_tanh = F if config.get("use_tanh") == None else  config.get("use_tanh")