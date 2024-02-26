#ALL
class Config(object):
    def __init__(self, args=None):
        self.gamma = 0.99
        self.args = args
        self.lr_a = 0.01 
        self.lr_c = 0.01 
        self.optimizer = 'adam' 
        self.net_dims = (256, 128) 
        self.epsilon=  0.9  
        self.epsilon_decay = 0.995 
        self.epsilon_min = 0.01
        self.memory_capacity = 2000 
        self.batch_size = 64 
        self.target_network_replace_freq = 200

    def add_config(self):
        if self.args != None:
            for attr in dir(self.args):
                if not attr.startswith('__'):
                    v = getattr(self.args, attr)
                    setattr(self, attr, v)
       