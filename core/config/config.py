import os
from collections.abc import MutableMapping

def flatten_dict(d, parent_key='', sep='.'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def get_factors(n):
    factors = set()
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            factors.add(i)
            factors.add(n // i)
    return sorted(factors)

class Config:
    @property
    def flatten_dict(self):
        return flatten_dict(self.__dict__)

    def init_dirs(self, args):
        self.main_dir = args.main_dir
        self.gen_workload_exec_dir = os.path.join(self.main_dir, "extern/aicb-plus")
        self.simulator_bin = os.path.join(self.main_dir, "simulator/htsim_astra_flow")
        self.trace_dir = os.path.join(self.main_dir, "trace_dir")
        self.log_dir = args.log_dir
        self.train_log_dir = args.train_log_dir
        self.system_dir = os.path.join(self.main_dir, "system/")
        self.memory_dir = os.path.join(self.main_dir, "memory/")
        self.topology_dir = os.path.join(self.main_dir, "topology/")
        self.result_dir = os.path.join(self.main_dir, "result/")
        self.workload_dir = os.path.join(self.main_dir, "workload/")
        self.aiob_comp_dir = os.path.join(self.workload_dir, "aiob_outputs/")
    
    def __init__(self, args, **extra_kwargs):
        self.init_dirs(args)
        self.sim_id = extra_kwargs.get("sim_id")
        self.topo_file = args.topo_file
        self.world_size = args.world_size
        self.model_config = args.model_config
        self.num_layers = self.model_config.get("num_layers")
        self.global_batch_size = args.global_batch_size
        self.topo_config = args.topo_config
        self.gpu_type = args.gpu_type
        self.gpu_memory = args.gpu_memory
        self.search_func = args.search_func
        self.default_algs = "direct" ## 默认集合通信算法, pp默认算法
        self.search_config(**extra_kwargs)
    
    def search_config(self, **extra_kwargs):
        pass
    
    @property
    def search_space(self):
        return [
            "sim_id",
            ## search space
            "tp",
            "pp",
            "dp",
            "vpp",
            "micro_batch_size",
            # "split_flows",
            # "all_reduce",
            "all_gather",
            "reduce_scatter",
            # "flash_attention",
            # "recompute",
            # "rmsnorm",
            ## info
            "sim_time",
            "error",
        ]
    
    @property
    def search_space_dict(self):
        flatten_dict = self.flatten_dict
        return {k: flatten_dict.get(k) for k in self.search_space}
