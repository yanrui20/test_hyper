from .config import MultiConfig
import subprocess
import re
import os

class Simulator(MultiConfig):
    def __init__(self, args, **extra_kwargs):
        super().__init__(args, **extra_kwargs)
        ## prepared
        self.sim_count = 0
        self.sim_time = float('inf')
        self.error = ""
    
    def check_restrict(self):
        if self.dp == 0:
            self.error += "self.dp == 0;"
            return
        if self.tp * self.dp * self.pp != self.world_size:
            self.error += "self.tp * self.dp * self.pp != self.world_size; "
        if self.global_batch_size % (self.dp * self.micro_batch_size) != 0:
            self.error += "self.global_batch_size % (self.dp * self.micro_batch_size) != 0; "
        if self.num_layers % (self.pp * self.vpp) != 0:
            self.error += "self.num_layers % (self.pp * self.vpp) != 0; "
        num_microbatch_per_dp = self.global_batch_size // (self.micro_batch_size * self.dp)
        if num_microbatch_per_dp % (self.pp * self.vpp) != 0:
            self.error += "num_microbatch_per_dp % (self.pp * self.vpp) != 0; "
        if self.topo_config["gpus_per_server"] % self.tp != 0:
            self.error += "gpus_per_server % tp != 0; "
    
    def megatron_cmd(self):
        return \
f"""
cd /opt/tiger/Megatron-LM/ && \\
bash -x examples/pretrain_gpt_distributed_with_mp_13B.sh \\
    --num-attention-heads {self.model_config["num_attention_heads"]} \\
    --tensor-model-parallel-size {self.tp} \\
    --pipeline-model-parallel-size {self.pp} \\
    --micro-batch-size {self.micro_batch_size} \\
    --num-layers-per-virtual-pipeline-stage {self.vpp} \\
    --micro-batch-size {self.micro_batch_size} \\
    --global-batch-size {self.model_config["global_batch_size"]} \\
    --sequence-parallel \\
    --use-flash-attn \\
    --use-distributed-optimizer \\
    --train-iters 5 \\
    --eval-iters 0 2>&1 | tee {self.train_log_dir}/train_id_{self.sim_id}.log
"""
    def megatron_env(self):
        env = dict(os.environ)
        env["NCCL_DEBUG_FILE"] = f"{self.log_dir}/nccl_debug.%h.%p"
        env["NCCL_DEBUG"] = "INFO"
        env["NCCL_DEBUG_SUBSYS"] = "INIT,COLL"
        env["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
        env["NCCL_ALGO"] = f"allgather:{self.all_gather};reducescatter:{self.reduce_scatter}"
        return env

    def run(self):
        self.check_restrict()
        if self.error:
            return
        result = subprocess.run(
            self.megatron_cmd(), 
            env=self.megatron_env(), 
            shell=True, 
            executable='/bin/bash', 
            capture_output=True, 
            text=True,
        )
        iteration_times = re.findall(r'elapsed time per iteration \(ms\):\s*([\d.]+)', result.stdout)
        if iteration_times:
            self.sim_time = sum(map(float, iteration_times)) / len(iteration_times)
        else:
            print(f"::::::::DEBUG_START_trial_{self.sim_id}::::::::")
            print(f"::::::::CMD_trial_{self.sim_id}::::::::")
            print(self.megatron_cmd())
            print(f"::::::::ENV_trial_{self.sim_id}::::::::")
            print(self.megatron_env())
            print(f"::::::::STDOUT_trial_{self.sim_id}::::::::")
            print(result.stdout)
            print(f"::::::::STDERR_trial_{self.sim_id}::::::::")
            print(result.stderr)
            print(f"::::::::DEBUG_END_trial_{self.sim_id}::::::::")