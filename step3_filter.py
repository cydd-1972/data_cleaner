import json
import random
import shutil
import re
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

import config_step3 as config


def extract_memory_counts_standard(content: str) -> Dict[str, int]:
    """处理标准格式: Memories for user X:"""
    memory_counts = {}
    pattern = r'Memories for user (\w+):\s*\n\s*\[(.*?)\]\s*(?=\n\s*\n\s*Memories for user|\n\s*\n\s*Question:|$)'
    matches = re.findall(pattern, content, re.DOTALL)
    
    for user_name, memory_content in matches:
        quoted_items = re.findall(r'"[^"]*"', memory_content)
        timestamp_items = re.findall(r'\d+:\d+\s+(?:am|pm)\s+on\s+\d+\s+\w+,\s+\d{4}:', memory_content)
        memory_count = max(len(quoted_items), len(timestamp_items))
        
        if memory_count > 0:
            memory_counts[user_name] = memory_count
    
    return memory_counts


def extract_memory_counts_nano(content: str) -> Dict[str, int]:
    """处理Nano格式: Memories for the Users:"""
    memory_counts = {}
    
    pattern = r'Memories for the Users:\s*\n\s*\[(.*?)\]\s*(?=\n\s*\n\s*Insights from the Search Agent|\n\s*\n\s*Question:|$)'
    match = re.search(pattern, content, re.DOTALL)
    
    if match:
        memory_content = match.group(1)
        
        # 匹配所有带用户标识的条目: "[User] ..." 或 '[User] ...'
        user_pattern = r'["\']\[(\w+)\][^"\']*["\']'
        memory_items = re.findall(user_pattern, memory_content)
        
        if not memory_items:
            user_pattern2 = r'\[(\w+)\]'
            memory_items = re.findall(user_pattern2, memory_content)
        
        if memory_items:
            user_counts = defaultdict(int)
            for user in memory_items:
                user_counts[user] += 1
            
            for user, count in user_counts.items():
                memory_counts[user] = count
    
    return memory_counts


def extract_memory_counts(content: str) -> Dict[str, int]:
    """尝试两种格式提取记忆数量"""
    counts = extract_memory_counts_standard(content)
    if counts:
        return counts
    
    counts = extract_memory_counts_nano(content)
    if counts:
        return counts
    
    return {}


def get_sample_key_for_dedup(sample: Dict) -> Tuple:
    """
    生成用于去重的key：基于(question, 每个用户的记忆长度)
    返回 (question, tuple(sorted(user_memory_counts.items())))
    """
    question = sample.get("question", "")
    if not question:
        return ("[No Question]", ())
    
    # 获取prompt内容
    prompt_list = sample.get("prompt", [])
    if not prompt_list:
        return (question, ())
    
    content = prompt_list[0].get("content", "")
    if not content:
        return (question, ())
    
    # 提取记忆数量
    memory_counts = extract_memory_counts(content)
    
    if not memory_counts:
        return (question, ())
    
    # 按用户排序后转为tuple，确保可哈希
    sorted_counts = tuple(sorted(memory_counts.items()))
    
    return (question, sorted_counts)


def deduplicate_medium():
    """
    中等题去重：从 medium/ 读取，去重后保存到 medium_dedup/
    基于 (question, 每个用户的记忆长度) 去重
    """
    source_dir = config.MEDIUM_DIR
    target_dir = config.MEDIUM_DEDUP_DIR
    
    if not source_dir.exists():
        print(f"目录不存在: {source_dir}")
        return []
    
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # 收集所有JSON文件
    json_files = list(source_dir.glob("*.json"))
    
    if not json_files:
        return []
    
    print(f"中等题原始数量: {len(json_files)}")
    
    # 按key分组
    groups = defaultdict(list)
    failed_count = 0
    
    for file_path in json_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                sample = json.load(f)
            
            key = get_sample_key_for_dedup(sample)
            groups[key].append((file_path, sample))
            
        except Exception:
            failed_count += 1
            continue
    
    print(f"解析失败: {failed_count}")
    print(f"分组数量: {len(groups)}")
    
    # 去重：每组随机保留一个
    deduped_samples = []
    
    for key, items in groups.items():
        # 随机选择一个保留
        random.shuffle(items)
        keep_file, keep_sample = items[0]
        
        # 复制到目标目录
        shutil.copy2(keep_file, target_dir / keep_file.name)
        deduped_samples.append(keep_sample)
    
    print(f"去重后数量: {len(deduped_samples)}")
    print(f"移除重复: {len(json_files) - len(deduped_samples) - failed_count}")
    
    return deduped_samples


def load_samples_from_dir(directory: Path) -> List[Dict]:
    """从目录加载所有样本"""
    samples = []
    if not directory.exists():
        return samples
    
    for file_path in directory.glob("*.json"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                samples.append(json.load(f))
        except Exception:
            continue
    
    return samples


def random_select_and_save(samples: List[Dict], target_count: int, target_dir: Path):
    """随机选择指定数量的样本并保存到目标目录"""
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # 随机选择
    if len(samples) <= target_count:
        selected = samples
    else:
        selected = random.sample(samples, target_count)
    
    # 保存到目录
    for i, sample in enumerate(selected):
        # 生成文件名
        source_file = sample.get("extra_info", {}).get("source_file", "unknown")
        source_file = source_file.replace('.jsonl', '')
        output_file = target_dir / f"{source_file}_{i}.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(sample, f, ensure_ascii=False, indent=2)
    
    return selected


def main():
    """主函数"""
    random.seed(config.OUTPUT_CONFIG["seed"])
    
    print("=" * 50)
    print("第三步：中等题去重 + 按比例保留")
    print("=" * 50)
    
    # 1. 中等题去重
    print("\n[1/3] 中等题去重...")
    medium_deduped = deduplicate_medium()
    
    # 2. 加载样本
    print("\n[2/3] 加载样本...")
    easy_samples = load_samples_from_dir(config.EASY_DIR)
    hard_samples = load_samples_from_dir(config.HARD_DIR)
    medium_samples = load_samples_from_dir(config.MEDIUM_DEDUP_DIR)
    
    print(f"  简单题: {len(easy_samples)}")
    print(f"  中等题: {len(medium_samples)}")
    print(f"  难题: {len(hard_samples)}")
    
    # 3. 计算目标数量
    total_target = config.OUTPUT_CONFIG["total_target"]
    easy_ratio = config.CLASSIFICATION_CONFIG["easy_ratio"]
    medium_ratio = config.CLASSIFICATION_CONFIG["medium_ratio"]
    hard_ratio = config.CLASSIFICATION_CONFIG["hard_ratio"]
    
    target_counts = {
        "easy": int(total_target * easy_ratio),
        "medium": int(total_target * medium_ratio),
        "hard": int(total_target * hard_ratio)
    }
    
    # 调整总和
    diff = total_target - sum(target_counts.values())
    if diff > 0:
        target_counts["medium"] += diff
    
    print(f"\n目标数量 (总计{total_target}):")
    print(f"  简单题: {target_counts['easy']} ({easy_ratio*100:.0f}%)")
    print(f"  中等题: {target_counts['medium']} ({medium_ratio*100:.0f}%)")
    print(f"  难题: {target_counts['hard']} ({hard_ratio*100:.0f}%)")
    
    # 4. 按比例随机选择并保存
    print("\n[3/3] 随机选择并保存...")
    
    selected_easy = random_select_and_save(
        easy_samples, target_counts["easy"], config.EASY_PRO_DIR
    )
    selected_medium = random_select_and_save(
        medium_samples, target_counts["medium"], config.MEDIUM_PRO_DIR
    )
    selected_hard = random_select_and_save(
        hard_samples, target_counts["hard"], config.HARD_PRO_DIR
    )
    
    print(f"\n完成!")
    print(f"  简单题: {len(selected_easy)} -> {config.EASY_PRO_DIR}")
    print(f"  中等题: {len(selected_medium)} -> {config.MEDIUM_PRO_DIR}")
    print(f"  难题: {len(selected_hard)} -> {config.HARD_PRO_DIR}")


if __name__ == "__main__":
    main()