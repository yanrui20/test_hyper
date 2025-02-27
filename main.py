import os
from core.search import Search
import argparse

# Config Settings
model_config = {
    "gpt3_175B":{
        "model_name": "gpt3_175B",
        "frame": "Megatron",
        "model_size": 175 * 1e9,
        "num_layers": 96,
        "hidden_size": 12288,
        "ffn_hidden_size": 12288*4,
        "num_attention_heads": 96,
        "seq_length": 2048,
        "max_position_embeddings": 2048,
        "vocab_size": 50257,
        "global_batch_size": 1536,
    },
    "test":{
        "model_name": "test",
        "frame": "Megatron",
        "model_size": 999999999,
        "num_layers": 16,
        "hidden_size": 4096,
        "ffn_hidden_size": 4096*4,
        "num_attention_heads": 16,
        "seq_length": 2048,
        "max_position_embeddings": 2048,
        "vocab_size": 32000,
        "global_batch_size": 64,
    },
}

topo_config = {
    "gpus_per_server": 16,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--world_size", type=int, default=1024)
    parser.add_argument("--model", type=str, default="gpt3_175B")
    parser.add_argument("--trial", type=int, default=8100)
    args = parser.parse_args()

    search = Search(
        main_dir=os.getcwd(),
        topo_file=None,
        world_size=args.world_size,
        model_config=model_config[args.model],
        topo_config=topo_config,
        search_sql=None,
        study_name="real-1024-gpu",
        gpu_type="L20",
        gpu_memory=None,
        all_trials=args.trial,
        parallel=1,
        search_func="optuna",
    )
    search.search()

