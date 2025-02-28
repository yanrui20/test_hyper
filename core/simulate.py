from .config import MultiConfig
import subprocess
import re

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
        cmd = \
f"""
cd /opt/tiger/Megatron-LM/ && \\
NCCL_DEBUG_FILE=`pwd`/nccl_debug.%h.%p \\
LD_PRELOAD=/opt/tiger/msccl/build/lib/libnccl.so.2.23.4 \\
NCCL_DEBUG=INFO \\
NCCL_DEBUG_SUBSYS=INIT,COLL \\
NCCL_ALGO="allgather:{self.all_gather};reducescatter:{self.reduce_scatter}" \\
bash -x examples/pretrain_gpt_distributed_with_mp_13B.sh \\
    --num-attention-heads {self.model_config["num_attention_heads"]} \\
    --tensor-model-parallel-size {self.tp} \\
    --pipeline-model-parallel-size {self.pp} \\
    --micro-batch-size {self.micro_batch_size} \\
    --num-layers-per-virtual-pipeline-stage {self.vpp} \\
    --micro_batch_size {self.micro_batch_size} \\
    --global-batch-size {self.model_config["global_batch_size"]} \\
    --sequence-parallel \\
    --use-flash-attn \\
    --use-distributed-optimizer \\
    --train-iters 5 \\
    --eval-iters 0 2>&1 | tee {self.train_log_dir}/train_id_{self.sim_id}.log
"""
        return cmd

    def run(self):
        self.check_restrict()
        if self.error:
            return
        cmd = self.megatron_cmd()
        result = subprocess.run(cmd, shell=True, executable='/bin/bash', capture_output=True, text=True)
        iteration_times = re.findall(r'elapsed time per iteration \(ms\):\s*([\d.]+)', result.stdout)
        if iteration_times:
            self.sim_time = sum(map(float, iteration_times)) / len(iteration_times)
        else:
            print(f"::::::::DEBUG_START_trial_{self.sim_id}::::::::")
            print(f"::::::::STDOUT_trial_{self.sim_id}::::::::")
            print(result.stdout)
            print(f"::::::::STDERR_trial_{self.sim_id}::::::::")
            print(result.stderr)
            print(f"::::::::DEBUG_END_trial_{self.sim_id}::::::::")