import json
import logging
import argparse
import re
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm

import config_step2 as config

try:
    from vllm_client import VLLMClient
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    print("警告: vLLM未安装，请先安装")

# 配置日志
logging.basicConfig(
    level=getattr(logging, config.LOG_CONFIG["level"]),
    format=config.LOG_CONFIG["format"],
    handlers=[
        logging.FileHandler(config.LOG_CONFIG["file"], encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class AnswerEvaluator:
    """答案评估器"""
    def __init__(self, batch_size: int = 64, temperature: float = 0.1, 
                 early_stop: bool = True, max_tokens: int = None):
        
        self.model_path = config.MODEL_CONFIG["judge_model_path"]
        self.device = config.MODEL_CONFIG["judge_device"]
        self.max_tokens = max_tokens or config.MODEL_CONFIG.get("judge_max_tokens", 4096)
        self.temperature = temperature
        self.batch_size = batch_size
        self.early_stop = early_stop
        
        self.stats = {
            "total_samples": 0, "total_answers": 0, "correct_answers": 0,
            "model_calls": 0, "batches": 0, "early_stopped": 0
        }
        self.classification = {"hard": 0, "medium": 0, "easy": 0}
        
        self._init_vllm()
    
    def _init_vllm(self):
        use_vllm = config.VLLM_CONFIG.get("enabled", False) and VLLM_AVAILABLE
        
        if use_vllm:
            vllm_config = config.VLLM_CONFIG.get("engine_config", {})
            device_list = self.device if isinstance(self.device, list) else [self.device]

            self.vllm_client = VLLMClient(
                model_path=self.model_path,
                device=device_list,
                dtype=config.MODEL_CONFIG.get("dtype", config.MODEL_CONFIG.get("torch_dtype", "float16")),
                max_model_len=vllm_config.get("max_model_len", 8192),
                gpu_memory_utilization=vllm_config.get("gpu_memory_utilization", 0.85),
                max_num_seqs=vllm_config.get("max_num_seqs", 8),
                tensor_parallel_size=vllm_config.get("tensor_parallel_size"),
            )
            
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        else:
            raise ImportError("vLLM未启用或不可用")
    
    def _build_judge_prompt(self, question: str, ground_truth: List[str], answer: str) -> str:
        """构建判断prompt"""
        if len(ground_truth) == 1:
            gt_text = ground_truth[0]
        else:
            gt_text = "\n".join([f"{i+1}. {gt}" for i, gt in enumerate(ground_truth)])
        
        user_content = f"""问题：{question}

标准答案：
{gt_text}

生成的答案：
{answer}

这个答案正确吗？请只回答"正确"或"错误"。"""
        
        messages = [
            {"role": "system", "content": "你是一个公正的答案评判员。请判断生成的答案是否与标准答案意思一致，只回答'正确'或'错误'。"},
            {"role": "user", "content": user_content}
        ]
        
        try:
            if hasattr(self.tokenizer, 'apply_chat_template'):
                return self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
        except:
            pass
        
        return f"""你是一个公正的答案评判员。请判断以下生成的答案是否与标准答案意思一致。

【问题】
{question}

【标准答案】
{gt_text}

【生成的答案】
{answer}

请只回答"正确"或"错误"。"""
    
    def _remove_think_tags(self, text: str) -> str:
        if not text:
            return text
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        text = re.sub(r'</?think>', '', text)
        return text.strip()
    
    def _parse_judge_result(self, result_text: str) -> bool:
        if not result_text:
            return False
        
        cleaned_text = self._remove_think_tags(result_text)
        result_lower = cleaned_text.lower().strip()
        
        if any(word in result_lower for word in ["正确", "correct", "yes", "true", "对"]):
            return True
        elif any(word in result_lower for word in ["错误", "incorrect", "no", "false", "错"]):
            return False
        return False
    
    def judge_answers_batch(self, questions: List[str], ground_truths: List[List[str]], 
                           answers: List[str]) -> List[bool]:
        if not answers:
            return []
        
        prompts = [self._build_judge_prompt(q, gt, ans) for q, gt, ans in zip(questions, ground_truths, answers)]
        
        try:
            results = self.vllm_client.generate(
                prompts, temperature=self.temperature, max_tokens=self.max_tokens, n=1
            )
            self.stats["model_calls"] += len(prompts)
            self.stats["batches"] += 1
            # vLLM 返回每条 prompt 一个 list（n 个候选），此处 n=1
            texts = [row[0] if isinstance(row, list) and row else row for row in results]
            return [self._parse_judge_result(t) for t in texts]
        except Exception as e:
            logger.error(f"vLLM批量判断失败: {e}")
            raise
    
    def evaluate_sample(self, sample_data: Dict) -> Dict:
        sample = sample_data["sample"]
        generated_answers = sample_data["generated_answers"]
        
        question = sample.get("question", "")
        ground_truth = sample["reward_model"]["ground_truth"]["fixed"]
        sample_key = sample_data["sample_key"]
        
        total = len(generated_answers)
        correct_flags = []
        has_correct, has_incorrect, early_stopped = False, False, False
        
        for i in range(0, total, self.batch_size):
            batch_answers = generated_answers[i:i+self.batch_size]
            batch_questions = [question] * len(batch_answers)
            batch_ground_truths = [ground_truth] * len(batch_answers)
            
            batch_results = self.judge_answers_batch(batch_questions, batch_ground_truths, batch_answers)
            correct_flags.extend(batch_results)
            
            if any(batch_results):
                has_correct = True
            if not all(batch_results):
                has_incorrect = True
            
            if self.early_stop and has_correct and has_incorrect:
                early_stopped = True
                self.stats["early_stopped"] += 1
                break
        
        total_judged = len(correct_flags)
        total_correct = sum(correct_flags)
        
        hard_threshold = config.CLASSIFICATION_CONFIG["hard_threshold"]
        easy_threshold = config.CLASSIFICATION_CONFIG["easy_threshold"]
        
        if total_correct <= hard_threshold:
            difficulty = "hard"
        elif total_correct >= easy_threshold and total_judged == total:
            difficulty = "easy"
        else:
            difficulty = "medium"
        
        self.stats["total_samples"] += 1
        self.stats["total_answers"] += total_judged
        self.stats["correct_answers"] += total_correct
        self.classification[difficulty] += 1
        
        return {
            "sample_key": sample_key,
            "original_sample": sample,
            "difficulty": difficulty,
            "correct_count": total_correct,
            "judged_count": total_judged,
            "total_answers": total,
            "early_stopped": early_stopped
        }
    
    def print_stats(self):
        logger.info("=" * 50)
        logger.info(f"总样本: {self.stats['total_samples']}, 答案: {self.stats['total_answers']}, 正确: {self.stats['correct_answers']}")
        logger.info(f"难例: {self.classification['hard']}, 中等: {self.classification['medium']}, 简单: {self.classification['easy']}")
        logger.info(f"提前停止: {self.stats['early_stopped']}")


def get_files_to_process() -> List[Path]:
    """获取要处理的文件列表，支持all和custom两种模式"""
    generated_dir = config.GENERATED_ANSWERS_DIR
    
    if not generated_dir.exists():
        logger.error(f"目录不存在: {generated_dir}")
        return []
    
    strategy = config.FILE_SELECTION["strategy"]
    
    if strategy == "all":
        pattern = config.FILE_SELECTION.get("all_pattern", "*_generated.jsonl")
        files = list(generated_dir.glob(pattern))
        logger.info(f"All模式: 找到 {len(files)} 个文件")
        return files
        
    elif strategy == "custom":
        patterns = config.FILE_SELECTION.get("custom_patterns", [])
        files = [f for f in [generated_dir / p for p in patterns] if f.exists()]
        logger.info(f"Custom模式: 找到 {len(files)} 个文件")
        return files
        
    else:
        raise ValueError(f"不支持的策略: {strategy}")


def load_samples_from_file(file_path: Path) -> List[Dict]:
    """从文件加载样本"""
    samples = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    samples.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logger.error(f"解析失败 {file_path.name}: {e}")
    return samples


def save_sample_to_category(original_sample: Dict, difficulty: str, sample_key: str):
    """保存样本到对应类别文件夹"""
    category_map = {"hard": config.HARD_DIR, "medium": config.MEDIUM_DIR, "easy": config.EASY_DIR}
    category_dir = category_map.get(difficulty, config.MEDIUM_DIR)
    category_dir.mkdir(parents=True, exist_ok=True)
    
    dest_file = category_dir / f"{sample_key}.json"
    with open(dest_file, 'w', encoding='utf-8') as f:
        json.dump(original_sample, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description="评估答案并分类")
    parser.add_argument("--batch-size", type=int, help="批量大小")
    parser.add_argument("--no-early-stop", action="store_true", help="禁用提前停止")
    parser.add_argument("--no-resume", action="store_true", help="禁用断点续传")
    args = parser.parse_args()
    
    logger.info("开始评估答案并分类")
    
    eval_config = config.EVALUATION_CONFIG
    batch_size = args.batch_size or eval_config.get("batch_size", 64)
    early_stop = eval_config.get("early_stop_for_medium", True) and not args.no_early_stop
    resume_enabled = eval_config.get("resume_enabled", True) and not args.no_resume
    
    output_dir = eval_config.get("output_dir", config.OUTPUT_DIR / "evaluate")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_suffix = eval_config.get("output_suffix", "_evaluated.jsonl")
    
    # 创建分类目录
    config.HARD_DIR.mkdir(parents=True, exist_ok=True)
    config.MEDIUM_DIR.mkdir(parents=True, exist_ok=True)
    config.EASY_DIR.mkdir(parents=True, exist_ok=True)
    
    files_to_process = get_files_to_process()
    if not files_to_process:
        logger.error("没有找到要处理的文件")
        return
    
    logger.info(f"批量大小: {batch_size}, 提前停止: {early_stop}")
    
    evaluator = AnswerEvaluator(
        batch_size=batch_size, temperature=eval_config.get("temperature", 0.1),
        early_stop=early_stop, max_tokens=eval_config.get("max_tokens", 4096)
    )
    
    total_new = 0
    saved_counts = {"hard": 0, "medium": 0, "easy": 0}
    
    for input_file in files_to_process:
        logger.info(f"处理文件: {input_file.name}")
        
        output_file = output_dir / input_file.name.replace('_generated.jsonl', output_suffix)
        samples = load_samples_from_file(input_file)
        
        if not samples:
            continue
        
        # 断点续传
        completed_keys = set()
        if resume_enabled and output_file.exists():
            with open(output_file, 'r') as f:
                for line in f:
                    try:
                        completed_keys.add(json.loads(line)["sample_key"])
                    except:
                        pass
        
        samples_to_process = [s for s in samples if s["sample_key"] not in completed_keys]
        if not samples_to_process:
            continue
        
        logger.info(f"待评估: {len(samples_to_process)}/{len(samples)}")
        file_results = []
        
        for sample_data in tqdm(samples_to_process, desc=input_file.name):
            try:
                result = evaluator.evaluate_sample(sample_data)
                file_results.append(result)
                
                difficulty = result["difficulty"]
                save_sample_to_category(result["original_sample"], difficulty, result["sample_key"])
                saved_counts[difficulty] += 1
                total_new += 1
                
            except Exception as e:
                logger.error(f"评估失败 {sample_data.get('sample_key', 'unknown')}: {e}")
        
        if file_results:
            with open(output_file, 'a', encoding='utf-8') as f:
                for r in file_results:
                    f.write(json.dumps(r, ensure_ascii=False) + '\n')
    
    logger.info("=" * 50)
    logger.info(f"完成! 新增评估: {total_new} 个样本")
    logger.info(f"难例: {saved_counts['hard']}, 中等: {saved_counts['medium']}, 简单: {saved_counts['easy']}")
    evaluator.print_stats()


if __name__ == "__main__":
    main()