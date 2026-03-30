"""
LoCoMo数据集清洗 - 第三步配置（中等题去重 + 按比例保留）
"""
from pathlib import Path

# ==================== 数据路径配置 ====================
OUTPUT_DIR = Path("/mnt_16T/zy/data/cleaned_data_left")

# 第一步生成的分类目录
EASY_DIR = OUTPUT_DIR / "easy"
MEDIUM_DIR = OUTPUT_DIR / "medium"
HARD_DIR = OUTPUT_DIR / "hard"

# 中等题去重后的目录
MEDIUM_DEDUP_DIR = OUTPUT_DIR / "medium_dedup"

# 按比例保留后的目录
EASY_PRO_DIR = OUTPUT_DIR / "easy_pro"
MEDIUM_PRO_DIR = OUTPUT_DIR / "medium_pro"
HARD_PRO_DIR = OUTPUT_DIR / "hard_pro"

# ==================== 分类比例配置 ====================
CLASSIFICATION_CONFIG = {
    "easy_ratio": 0.10,    # 简单题占比 10%
    "medium_ratio": 0.85,  # 中等题占比 85%
    "hard_ratio": 0.05,    # 难题占比 5%
}

# ==================== 输出配置 ====================
OUTPUT_CONFIG = {
    "total_target": 1360,  # 目标总样本数
    "seed": 42,            # 随机种子
}