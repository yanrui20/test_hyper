from .config import Config, get_factors

class OptunaConfig(Config):
    def __init__(self, args, **extra_kwargs):
        super().__init__(args, **extra_kwargs)

    def search_config(self, **extra_kwargs):
        trial = extra_kwargs.get("trial")
        if self.model_config["model_name"] == "test":
            tp_index = trial.suggest_int('tp_index', 0, 2) ## 无用
            self.tp = 4
            self.pp = 2
            self.dp = 2
            self.vpp = 2
            self.micro_batch_size = 1
            self.all_gather = 'ring'
            self.reduce_scatter = 'ring'
            return

        ## 选择tp, 暂时只在机内搜索
        tp_factors = [1, 2, 4, 8, 16]
        tp_index = trial.suggest_int('tp_index', 0, len(tp_factors) - 1)
        self.tp = tp_factors[tp_index]
        ## 选择pp
        pp_factors = [3, 4, 6, 8, 12, 16, 24, 48]
        pp_index = trial.suggest_int('pp_index', 0, len(pp_factors) - 1)
        self.pp = pp_factors[pp_index]
        ## 得到dp
        self.dp = self.world_size // (self.tp * self.pp)
        ## 选择vpp
        vpp_factors = [2, 3, 4, 6, 8]
        vpp_index = trial.suggest_int('vpp_index', 0, len(vpp_factors) - 1)
        self.vpp = vpp_factors[vpp_index]
        ## 选择micro_batch_size
        mbs_factors = [1, 2, 3, 4]
        mbs_index = trial.suggest_int('mbs_index', 0, len(mbs_factors) - 1)
        self.micro_batch_size = mbs_factors[mbs_index]
        ## 选择集合通信算法
        cc_algorithms = ['ring', 'tree']
        all_gather_index = trial.suggest_int('all_gather_index', 0, len(cc_algorithms) - 1)
        self.all_gather = cc_algorithms[all_gather_index]
        reduce_scatter_index = trial.suggest_int('reduce_scatter_index', 0, len(cc_algorithms) - 1)
        self.reduce_scatter = cc_algorithms[reduce_scatter_index]