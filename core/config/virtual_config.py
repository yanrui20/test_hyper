from .config import Config

class VirtualConfig(Config):
    def __init__(self, args, **extra_kwargs):
        super().__init__(args, **extra_kwargs)

    def search_config(self, **extra_kwargs):
        self.order = ""
        self.tp = 0
        self.pp = 0
        self.dp = 0
        self.vpp = 0
        self.micro_batch_size = 0
        self.split_flows = 0
        self.scheduling_policy = ""
        self.all_gather = ""
        self.reduce_scatter = ""
        self.flash_attention = ""
        self.recompute = ""
        self.rmsnorm = ""