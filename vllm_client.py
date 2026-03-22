import logging
from typing import List, Optional, Union
import torch

logger = logging.getLogger(__name__)

try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    logger.error("vLLM未安装，请运行: pip install vllm")


class VLLMClient:
    """vLLM客户端，支持高效的批量生成"""
    
    def __init__(self, model_path: str, device: List[int] = None, 
                 dtype: str = "float16", max_model_len: int = 8192,
                 gpu_memory_utilization: float = 0.85, max_num_seqs: int = 8):
        if not VLLM_AVAILABLE:
            raise ImportError("vLLM未安装")
        
        # 设置GPU设备
        if device is not None:
            import os
            if isinstance(device, list):
                os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, device))
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = str(device)
        
        # 初始化LLM引擎
        self.llm = LLM(
            model=model_path,
            dtype=dtype,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            max_num_seqs=max_num_seqs,
            trust_remote_code=True,
        )
        
        logger.info(f"vLLM引擎初始化完成，模型: {model_path}")
    
    def generate(self, prompts: Union[str, List[str]], 
                temperature: float = 1.0,
                max_tokens: int = 4096,
                n: int = 1,
                top_p: float = 0.95,
                top_k: int = 50,
                stop: Optional[List[str]] = None) -> List[List[str]]:

        # 统一处理为列表
        if isinstance(prompts, str):
            prompts = [prompts]
        
        # 设置采样参数
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            n=n,
            top_p=top_p,
            top_k=top_k,
            stop=stop,
        )
        
        # 批量生成
        outputs = self.llm.generate(prompts, sampling_params)
        
        # 提取生成的文本
        results = []
        for output in outputs:
            # 每个output有n个候选答案
            generated_texts = [output.outputs[i].text.strip() for i in range(len(output.outputs))]
            results.append(generated_texts)
        
        return results
    
    def generate_batch(self, prompts: List[str], 
                      temperature: float = 1.0,
                      max_tokens: int = 4096,
                      n: int = 1) -> List[List[str]]:
        return self.generate(prompts, temperature, max_tokens, n)
    
    def close(self):
        """释放资源"""
        if hasattr(self, 'llm') and self.llm:
            # vLLM会自动管理GPU内存，这里只是清空引用
            self.llm = None
            logger.info("vLLM引擎已释放")
    
    def __del__(self):
        """析构函数"""
        self.close()