import optuna
from .simulate import Simulator
from rich import print
from rich.pretty import Pretty
import csv
import os
import time

class Search:
    def __init__(self, **args):
        self.search_func = args.get('search_func')
        self.main_dir = args.get('main_dir')
        self.topo_file = args.get('topo_file')
        self.topo_config = args.get('topo_config')
        self.gpu_type = args.get('gpu_type')
        self.gpu_memory = args.get('gpu_memory')
        self.world_size = args.get('world_size')
        self.model_config = args.get('model_config')
        self.global_batch_size = self.model_config.get('global_batch_size')
        self.all_trials = args.get('all_trials')
        self.study_name = args.get('study_name')
        self.parallel = args.get('parallel')
        self.time_str = time.strftime("%Y%m%d_%H_%M_%S", time.localtime())
        self.log_dir = os.path.join(self.main_dir, f"log/{self.search_func}_{self.time_str}_{self.study_name}")
        os.makedirs(self.log_dir, exist_ok=True)
        self.train_log_dir = os.path.join(self.log_dir, "train_log")
        os.makedirs(self.train_log_dir, exist_ok=True)
        self.log_file_prefix = f"{self.search_func}_{self.time_str}_{self.study_name}_config_log"
        print("Search Config:", Pretty(self.__dict__))
    
    def search(self):
        if self.search_func == 'optuna':
            run_optuna_real(self, 0)
    
    def test(self):
        pass 

def init_writer(log_io, args):
    virtual_config = Simulator(args, sim_id=0, virtual_init=True)
    writer = csv.DictWriter(log_io, fieldnames=virtual_config.search_space_dict.keys())
    writer.writeheader()
    return writer

def run_optuna_real(args: Search, index):
    log_file = os.path.join(args.log_dir, f"{args.log_file_prefix}_{index}.csv")
    with open(log_file, mode='w', newline='', encoding='utf-8', buffering=1) as log_io:
        writer = init_writer(log_io, args)
        def objective(trial):
            nonlocal writer
            simulator = Simulator(args, sim_id=trial.number, trial=trial)
            simulator.run()
            writer.writerow(simulator.search_space_dict)
            return simulator.sim_time

        study = optuna.create_study(direction='minimize', study_name=args.study_name)
        study.optimize(objective, n_trials=args.all_trials // args.parallel)