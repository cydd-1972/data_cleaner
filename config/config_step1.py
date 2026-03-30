import os
from pathlib import Path

# ==================== 本地模型配置 ====================
MODEL_CONFIG = {
    "model_path": "/mnt_16T/zy/models/Qwen3-8B",
    "temperature": 1.0,
    "max_tokens": 4096,
    "device": [1, 2],  # 使用的GPU设备
    "dtype": "float16",
}

# ==================== vLLM 配置 ====================
VLLM_CONFIG = {
    "enabled": True,
    "engine_config": {
        "gpu_memory_utilization": 0.85,
        "max_num_seqs": 8,  # 并发序列数，可根据显存调整
        "max_model_len": 8192,
    }
}

# ==================== 数据路径配置 ====================
DATA_DIR = Path("/mnt_16T/zy/data/locomo/train_left")
OUTPUT_DIR = Path("/mnt_16T/zy/data/cleaned_data_left")

# ==================== 文件选择策略 ====================
FILE_SELECTION = {
    "strategy": "custom",  # 可选: "all" 或 "custom"   
    # custom模式：精确匹配指定的文件
    "custom_patterns": [
        "evaluation_metrics_0.jsonl"
    ],  
    # all模式：在 DATA_DIR 下 glob 一次（不含子集编号）
    "all_pattern": "*.jsonl",
}

# ==================== 生成参数配置 ====================
GENERATION_CONFIG = {
    "k": 128,
    "batch_size": 4,
    "max_retries": 3,
    "temperature": 1.0,
    "max_tokens": 4096,
    "enable_thinking": False,
    "top_p": 0.95,
    "top_k": 50,
}

# ==================== 输出配置 ====================
OUTPUT_CONFIG = {
    "generated_answers_dir": OUTPUT_DIR / "generated_answers",
    "shuffle": False,  # 是否打乱数据
    "seed": 42,
}

# ==================== 日志配置 ====================
LOG_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": OUTPUT_DIR / "step1_generation.log",
}

# ==================== 断点续传配置 ====================
RESUME_CONFIG = {
    "enabled": True,  # 是否启用断点续传
    "save_interval": 10,  # 每处理多少个样本保存一次（用于flush）
}