from .config import Config
from .optuna_config import OptunaConfig
from .virtual_config import VirtualConfig
from .llm_config import LlmConfig

class MultiConfig(Config):
    def __new__(cls, args, **extra_kwargs):
        if extra_kwargs.get("virtual_init"):
            cls = type('DynamicVirtualConfig', (VirtualConfig, cls), {})
        elif args.search_func == "optuna":
            cls = type('DynamicOptunaConfig', (OptunaConfig, cls), {})
        elif args.search_func == "llm":
            cls = type('DynamicLlmConfig', (LlmConfig, cls), {})
        else:
            raise ValueError(f"Unknown search_func value: {args.search_func}")
        instance = super().__new__(cls)
        return instance
       
    def __init__(self, args, **extra_kwargs):
        super().__init__(args, **extra_kwargs)
