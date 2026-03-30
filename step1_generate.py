import json
import logging
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List
import torch
from transformers import AutoTokenizer
from tqdm import tqdm
import config_step1 as config
from vllm_client import VLLMClient

# 创建日志文件目录（如果不存在）
log_file = Path(config.LOG_CONFIG["file"])
log_file.parent.mkdir(parents=True, exist_ok=True)

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


class AnswerGenerator:
    """答案生成器"""
    
    def __init__(self):
        """初始化生成器"""
        # 加载配置
        self.model_path = config.MODEL_CONFIG["model_path"]
        self.temperature = config.GENERATION_CONFIG["temperature"]
        self.max_tokens = config.GENERATION_CONFIG["max_tokens"]
        self.k = config.GENERATION_CONFIG["k"]
        self.batch_size = config.GENERATION_CONFIG["batch_size"]
        self.max_retries = config.GENERATION_CONFIG["max_retries"]
        self.enable_thinking = config.GENERATION_CONFIG["enable_thinking"]
        self.top_p = config.GENERATION_CONFIG.get("top_p", 0.95)
        self.top_k = config.GENERATION_CONFIG.get("top_k", 50)
        
        # 初始化vLLM客户端
        logger.info(f"使用vLLM加速生成，模型: {self.model_path}")
        engine_config = config.VLLM_CONFIG["engine_config"]
        self.vllm_client = VLLMClient(
            model_path=self.model_path,
            device=config.MODEL_CONFIG["device"],
            dtype=config.MODEL_CONFIG.get("dtype", config.MODEL_CONFIG.get("torch_dtype", "float16")),
            max_model_len=engine_config["max_model_len"],
            gpu_memory_utilization=engine_config["gpu_memory_utilization"],
            max_num_seqs=engine_config["max_num_seqs"],
        )
        
        # 初始化tokenizer（用于apply_chat_template）
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, 
            trust_remote_code=True
        )
        
        # 创建输出目录
        self.output_dir = config.OUTPUT_CONFIG["generated_answers_dir"]
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 统计信息
        self.stats = {
            "total_samples": 0,
            "processed_samples": 0,
            "failed_samples": 0,
            "total_generated_answers": 0,
        }
        
        logger.info(f"答案生成器初始化完成")
        logger.info(f"配置: k={self.k}, batch_size={self.batch_size}")
        logger.info(f"文件选择策略: {config.FILE_SELECTION['strategy']}")
    
    def load_data(self, subset: int) -> List[Dict]:
        """
        加载指定子集的数据
        支持 "all" 和 "custom" 两种策略
        """
        subset_dir = config.DATA_DIR
        files = []
        
        strategy = config.FILE_SELECTION["strategy"]
        
        if strategy == "all":
            # all模式：加载所有匹配的文件
            pattern = config.FILE_SELECTION.get("all_pattern", "{subset}_*.jsonl")
            pattern = pattern.format(subset=subset)
            files = list(subset_dir.glob(pattern))            
            logger.info(f"All模式: 找到 {len(files)} 个匹配 {pattern} 的文件")
            if files:
                logger.debug(f"文件列表: {[f.name for f in files[:5]]}{'...' if len(files) > 5 else ''}")
                
        elif strategy == "custom":
            # custom模式：精确匹配指定的文件
            patterns = config.FILE_SELECTION.get("custom_patterns", [])            
            for pattern in patterns: # 之前用的先匹配{subset}前缀，不太好，直接用patterns就行
                formatted_pattern = pattern
                file_path = subset_dir / formatted_pattern
                if file_path.exists():
                    files.append(file_path)
                    logger.debug(f"找到文件: {file_path.name}")
                else:
                    logger.warning(f"自定义模式文件不存在: {file_path}")
            logger.info(f"Custom模式: 找到 {len(files)} 个匹配的文件")
            
        else:
            raise ValueError(f"不支持的文件选择策略: {strategy}，请使用 'all' 或 'custom'")
        
        if not files:
            logger.warning(f"子集 {subset} 未找到数据文件")
            return []
        
        # 加载所有文件，并记录行号
        all_samples = []
        for file_path in files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    file_samples = []
                    for line_num, line in enumerate(f, start=1):
                        line = line.strip()
                        if line:
                            sample = json.loads(line)
                            if "extra_info" not in sample: # 这里的extra_info是外层的
                                sample["extra_info"] = {}
                            sample["extra_info"]["source_file"] = file_path.name
                            sample["extra_info"]["line_number"] = line_num
                            file_samples.append(sample)
                    
                    all_samples.extend(file_samples)
                    logger.info(f"从 {file_path.name} 加载了 {len(file_samples)} 条数据（行号 1-{len(file_samples)}）")
                    
            except Exception as e:
                logger.error(f"加载文件 {file_path} 失败: {e}")
        
        logger.info(f"子集 {subset} 共加载 {len(all_samples)} 条样本")
        return all_samples
    
    def get_sample_key(self, sample: Dict) -> str:
        """
        生成样本的唯一标识
        格式: {文件名去掉.jsonl后缀}_{行号}
        例如: 0_0000_qwen3-14b_qwen3-14b-70_clm_10_123
        """
        source_file = sample.get("extra_info", {}).get("source_file", "unknown.jsonl")
        file_basename = source_file.replace('.jsonl', '')
        line_number = sample.get("extra_info", {}).get("line_number", 0)
        sample_key = f"{file_basename}_{line_number}"
        return sample_key
    
    def format_prompt(self, sample: Dict) -> str:
        """
        格式化prompt，使用模型的chat template
        根据模型的tokenizer配置，自动将对话列表转换为模型需要的格式
        添加特殊标记（如 <|im_start|>、<|im_end|> 等）
        添加生成提示（add_generation_prompt=True 会在最后添加 <|im_start|>assistant\n，告诉模型开始回答）
        """
        prompt_data = sample.get("prompt", [])
        
        if not prompt_data:
            logger.warning("样本中没有prompt字段")
            return ""
        
        # 应用chat template
        try:  # 检查tokenizer是否支持chat template
            if hasattr(self.tokenizer, 'apply_chat_template'):
                import inspect
                sig = inspect.signature(self.tokenizer.apply_chat_template)
                if 'enable_thinking' in sig.parameters:
                    formatted_prompt = self.tokenizer.apply_chat_template(
                        prompt_data,
                        tokenize=False,   # 返回文本，不进行tokenization
                        add_generation_prompt=True,   # 添加助手回复的提示符
                        enable_thinking=self.enable_thinking
                    )
                else:  # 不支持enable_thinking的旧版本
                    formatted_prompt = self.tokenizer.apply_chat_template(
                        prompt_data,
                        tokenize=False,
                        add_generation_prompt=True
                    )
            else:
                formatted_prompt = prompt_data[-1].get("content", "") if prompt_data else ""
        except Exception as e:
            logger.warning(f"apply_chat_template失败: {e}，使用原始内容")
            formatted_prompt = prompt_data[-1].get("content", "") if prompt_data else ""
        
        return formatted_prompt
    
    def generate_answers_batch(self, samples: List[Dict]) -> List[List[str]]:
        """
        批量生成多个样本的答案
        """
        if not samples:
            return []
        
        # 格式化所有prompts
        prompts = []
        for sample in samples:
            prompt = self.format_prompt(sample)
            prompts.append(prompt)
        
        # 使用vLLM批量生成
        try:
            # vLLM支持每个prompt生成n个答案
            results = self.vllm_client.generate(
                prompts,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                n=self.k,
                top_p=self.top_p,  
                top_k=self.top_k,  
            )
            
            return results
            
        except Exception as e:
            logger.error(f"批量生成失败: {e}")
            return [[""] * self.k for _ in samples]
    
    def remove_think_part(self, text: str) -> str:
        """从生成的文本中移除思考部分，减少评估阶段输入的token，同时方便debug"""
        if not text:
            return text
        
        think_patterns = [
            r'<think>.*?</think>',  # 基本上是这个
            r'\[think\].*?\[/think\]',
            r'\[思考\].*?\[/思考\]',
            r'思考：.*?\n',
            r'Think:.*?\n',
            r'分析：.*?\n',
            r'推理：.*?\n',
        ]
        
        cleaned_text = text
        for pattern in think_patterns:
            cleaned_text = re.sub(pattern, '', cleaned_text, flags=re.DOTALL | re.IGNORECASE)
        
        cleaned_text = re.sub(r'\n\s*\n', '\n', cleaned_text)
        cleaned_text = cleaned_text.strip()
        
        return cleaned_text if cleaned_text else text
    
    def process_file(self, samples: List[Dict], output_file: Path):
        """
        处理单个文件的数据
        
        Args:
            samples: 样本列表
            output_file: 输出文件路径
        """
        if not samples:
            logger.warning("没有样本需要处理")
            return
        
        # 断点续传：检查已处理的样本
        processed_keys = set()
        if config.RESUME_CONFIG["enabled"] and output_file.exists():
            try:
                with open(output_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            data = json.loads(line)
                            processed_keys.add(data.get("sample_key", ""))
                logger.info(f"已处理 {len(processed_keys)} 条，将跳过")
            except Exception as e:
                logger.warning(f"读取已有结果失败: {e}")
        
        # 过滤未处理的样本
        samples_to_process = []
        for sample in samples:
            sample_key = self.get_sample_key(sample)
            if sample_key not in processed_keys:
                samples_to_process.append(sample)
        
        if not samples_to_process:
            logger.info(f"文件 {output_file.name} 已全部处理完成")
            return
        
        logger.info(f"待处理样本数: {len(samples_to_process)}")
        
        # 批量处理
        with open(output_file, 'a', encoding='utf-8') as f_out:
            pbar = tqdm(total=len(samples_to_process), desc=f"处理 {output_file.name}")
            
            for batch_idx in range(0, len(samples_to_process), self.batch_size):
                batch = samples_to_process[batch_idx:batch_idx + self.batch_size]
                
                try:
                    # 批量生成答案
                    batch_results = self.generate_answers_batch(batch)
                    
                    # 处理每个样本的结果
                    for sample, generated_answers in zip(batch, batch_results):
                        sample_key = self.get_sample_key(sample)
                        
                        # 清洗答案
                        cleaned_answers = [self.remove_think_part(ans) for ans in generated_answers]
                        
                        # 构建输出
                        result = {
                            "sample_key": sample_key,
                            "sample": {
                                "data_source": sample.get("data_source", ""),
                                "prompt": sample.get("prompt", []),
                                "ability": sample.get("ability", ""),
                                "reward_model": sample.get("reward_model", {
                                    "ground_truth": {
                                        "raw": "",
                                        "fixed": []
                                    },
                                    "style": ""
                                }),
                                "extra_info": sample.get("extra_info", {}),
                                "question": sample.get("question", "")
                            },
                            "generated_answers": cleaned_answers,
                            "extra_info": {
                                "original_sample_key": sample_key,
                                "line_number": sample.get("extra_info", {}).get("line_number", 0)
                            }
                        }
                        
                        f_out.write(json.dumps(result, ensure_ascii=False) + '\n')
                        f_out.flush()
                        
                        self.stats["processed_samples"] += 1
                        self.stats["total_generated_answers"] += len(cleaned_answers)
                        pbar.update(1)
                    
                except Exception as e:
                    logger.error(f"批次 {batch_idx//self.batch_size + 1} 处理失败: {e}")
                    self.stats["failed_samples"] += len(batch)
                    pbar.update(len(batch))
                    continue
            
            pbar.close()
        
        logger.info(f"文件处理完成: {output_file.name}")
    
    def run(self):
        """运行：生成答案"""
        logger.info("=" * 60)
        logger.info("第一步：批量生成答案")
        logger.info("=" * 60)
        
        # 验证文件选择策略
        strategy = config.FILE_SELECTION["strategy"]
        if strategy not in ["all", "custom"]:
            logger.error(f"不支持的文件选择策略: {strategy}")
            logger.error("请使用 'all' 或 'custom'")
            return
        
        # 加载所有数据
        all_samples = []
        for subset in config.TRAIN_SUBSETS:
            logger.info(f"加载子集 {subset}...")
            samples = self.load_data(subset)
            all_samples.extend(samples)
        
        if not all_samples:
            logger.error("未加载到任何数据")
            return
        
        self.stats["total_samples"] = len(all_samples)
        logger.info(f"共加载 {len(all_samples)} 条样本")
        
        # 按源文件分组处理
        samples_by_file = defaultdict(list)
        for sample in all_samples:
            source_file = sample.get("extra_info", {}).get("source_file", "unknown.jsonl")
            samples_by_file[source_file].append(sample)
        
        logger.info(f"共 {len(samples_by_file)} 个源文件需要处理")
        
        # 打印文件列表和样本数量
        for source_file, file_samples in samples_by_file.items():
            first_line = file_samples[0].get("extra_info", {}).get("line_number", 0)
            last_line = file_samples[-1].get("extra_info", {}).get("line_number", 0)
            logger.info(f"  - {source_file}: {len(file_samples)} 条样本 (行号范围: {first_line}-{last_line})")
        
        # 处理每个文件
        for source_file, file_samples in samples_by_file.items():
            output_filename = source_file.replace('.jsonl', '_generated.jsonl')
            output_file = self.output_dir / output_filename
            
            logger.info(f"\n处理文件: {source_file}")
            logger.info(f"样本数: {len(file_samples)}, 输出: {output_file}")
            
            # 按行号排序
            file_samples.sort(key=lambda x: x.get("extra_info", {}).get("line_number", 0))
            
            # 可选：打乱样本顺序
            if config.OUTPUT_CONFIG["shuffle"]:
                random.seed(config.OUTPUT_CONFIG["seed"])
                random.shuffle(file_samples)
                logger.info("已打乱样本顺序")
            
            # 处理文件
            self.process_file(file_samples, output_file)
        
        # 输出统计信息
        logger.info("\n" + "=" * 60)
        logger.info("第一步完成！统计信息:")
        logger.info(f"总样本数: {self.stats['total_samples']}")
        logger.info(f"已处理: {self.stats['processed_samples']}")
        logger.info(f"失败: {self.stats['failed_samples']}")
        logger.info(f"生成答案总数: {self.stats['total_generated_answers']}")
        if self.stats['processed_samples'] > 0:
            logger.info(f"平均每样本答案数: {self.stats['total_generated_answers'] / self.stats['processed_samples']:.1f}")
        logger.info("=" * 60)


def main():
    """主函数"""
    try:
        generator = AnswerGenerator()
        generator.run()
    except KeyboardInterrupt:
        logger.info("用户中断")
    except Exception as e:
        logger.error(f"运行失败: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()