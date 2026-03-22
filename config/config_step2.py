import os
from pathlib import Path

# ==================== 本地模型配置 ====================
MODEL_CONFIG = {
    "judge_model_path": "/mnt_16T/zy/models/Qwen3-14B",
    "judge_device": [1, 2],  # 评估模型使用的GPU设备
    "judge_max_tokens": 4096,
    "torch_dtype": "float16",
}

# ==================== vLLM 配置 ====================
VLLM_CONFIG = {
    "enabled": True,
    "engine_config": {
        "gpu_memory_utilization": 0.85,
        "max_num_seqs": 8,
        "max_model_len": 8192,
    }
}

# ==================== 数据路径配置 ====================
DATA_DIR = Path("/mnt_16T/zy/data/locomo/train_filtered_emb_augmented")
OUTPUT_DIR = Path("/mnt_16T/zy/data/cleaned_data")

# 第一步生成答案的目录
GENERATED_ANSWERS_DIR = OUTPUT_DIR / "generated_answers"

# ==================== 文件选择策略 ====================
FILE_SELECTION = {
    "strategy": "custom",  # 可选: "all" 或 "custom"
    
    # custom模式：精确匹配指定的文件
    "custom_patterns": [
        # "0_0000_qwen3-14b_qwen3-14b-70_clm_10_generated.jsonl",
    ],
    
    # all模式：匹配所有文件
    "all_pattern": "*_generated.jsonl",
}

# ==================== 评估配置 ====================
EVALUATION_CONFIG = {
    # 评估参数
    "batch_size": 64,          
    "temperature": 0.1,        
    "early_stop_for_medium": True, 
    "max_tokens": 4096,        
    
    # 输出配置
    "output_dir": OUTPUT_DIR / "evaluate",
    "output_suffix": "_evaluated.jsonl",
    
    # 断点续传
    "resume_enabled": True,
}

# ==================== 分类配置 ====================
CLASSIFICATION_CONFIG = {
    "hard_threshold": 0,        # pass@k = 0 为难例
    "easy_threshold": 128,      # pass@k = 128 为简单题
    # 介于两者之间的为中等题
}

# ==================== 日志配置 ====================
LOG_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": OUTPUT_DIR / "step2_evaluation.log",
}

# ==================== 输出目录 ====================
# 分类结果保存目录
HARD_DIR = OUTPUT_DIR / "hard"
MEDIUM_DIR = OUTPUT_DIR / "medium"
EASY_DIR = OUTPUT_DIR / "easy"