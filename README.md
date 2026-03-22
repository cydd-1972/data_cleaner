# LoCoMo 数据清洗

本仓库将原始对话样本经三步处理：批量生成候选答案、用大模型评判并划分难度、对中等题去重后按比例抽样，得到用于训练的清洗数据。下文说明如何一键或分步运行，以及各步数据约定。

---

## 一、快速开始

### 1.1 一步清洗

**配置**：运行前在 `config/config_step1.py`、`config/config_step2.py`、`config/config_step3.py` 中设置数据目录、模型路径、`TRAIN_SUBSETS`、文件选择策略、保留数量target_counts等。

在**项目根目录**执行（需与本仓库中的 `config/`、`vllm_client.py` 及三步脚本放在一起）：

```bash
python run_clean_pipeline.py
```

脚本会**顺序**拉起三个子进程：`step1_generate.py` → `step2_evaluate.py` → `step3_filter.py`。每步在独立进程中结束，便于释放显存，避免 8B 生成模型与 14B 评判模型同时占用 GPU。

**常用参数**

- `--skip-step1` | 跳过生成（已有 `generated_answers/` 时） |
- `--skip-step2` | 跳过评估（已有 `easy/`、`medium/`、`hard/` 时） |
- `--skip-step3` | 跳过去重与按比例抽样 |

第二步（评估）支持的参数与 `step2_evaluate.py` 相同；写在命令行末尾、且不属于本入口脚本的参数，会原样传给第二步，例如：

```bash
python run_clean_pipeline.py --batch-size 64
python run_clean_pipeline.py --skip-step1 --no-early-stop
```

### 1.2 分步清洗

需让 Python 能导入 `config_step1` / `config_step2` / `config_step3`（位于 `config/`）以及根目录的 `vllm_client.py`。在**项目根目录**执行，并设置 `PYTHONPATH`（项目根 + `config`）。

Linux / macOS：

```bash
export PYTHONPATH="$(pwd):$(pwd)/config:${PYTHONPATH}"
python step1_generate.py
python step2_evaluate.py
python step3_filter.py
```

Windows PowerShell：

```powershell
$env:PYTHONPATH = "$PWD;$PWD\config"
python step1_generate.py
python step2_evaluate.py
python step3_filter.py
```

---

## 二、项目结构

| 路径 | 作用 |
|------|------|
| `run_clean_pipeline.py` | 一键串联三步的入口 |
| `step1_generate.py` | 第一步：Qwen3-8B（vLLM）为每条样本生成 k 个答案 |
| `step2_evaluate.py` | 第二步：Qwen3-14B 判对错，按难度写入 `easy/`、`medium/`、`hard/` |
| `step3_filter.py` | 第三步：中等题去重，再按比例抽样到 `*_pro/` |
| `vllm_client.py` | vLLM 封装，供第一步与第二步共用 |
| `config/config_step1.py` | 第一步的配置文件 |
| `config/config_step2.py` | 第二步的配置文件 |
| `config/config_step3.py` | 第三步的配置文件 |


日志路径由各步 `LOG_CONFIG` 或脚本输出决定（例如 `OUTPUT_DIR` 下的 `step1_generation.log`、`step2_evaluation.log`）。

---

## 三、各步说明

### 3.1 生成答案

- **模型**：Qwen3-8B  
- **能力**：vLLM 加速、批量生成、断点续传  

每条样本生成 k 个答案；输出为 JSONL，**一行一个样本**。字段含义如下。

**sample_key**  
样本唯一标识，一般由「源文件名去掉后缀」与「行号」等组成，例如 `0_14714705_gpt-4.1-mini_qwen3-14b-30_clm_10_4`。  
说明：不宜仅用 `data_source` 区分样本（多行可能共享同一 `data_source`）。显式 `sample_key` 便于断点续跑、排查问题，以及后续评估与分类溯源。

**sample（原始样本）**

| 字段 | 含义 |
|------|------|
| `data_source` | 数据来源标识，例如 `0_14714705_gpt-4` |
| `prompt` | 对话提示；`content` 为完整上下文（系统指令、记忆、问题等），`role` 多为 `user` |
| `ability` | 任务类型，例如记忆检索时为 `memory` |
| `reward_model` | `ground_truth` 含 `raw` / `fixed` 等；`style` 为规则类型，如 `rule` |
| `extra_info` | `index`、`split`、`group_id`、`qa_id`、`category`、`source_file` 等 |
| `question` | 用户问题文本 |

示例结构：

```json
{
  "sample_key": "0_0000_qwen3-14b_qwen3-14b-70_clm_10_123",
  "sample": {
    "data_source": "locomo",
    "prompt": [
      {
        "content": "请回答以下问题...",
        "role": "user"
      }
    ],
    "ability": "qa",
    "reward_model": {
      "ground_truth": {
        "raw": "原始答案",
        "fixed": ["标准答案1", "标准答案2"]
      },
      "style": "qa"
    },
    "extra_info": {
      "index": 0,
      "split": "train",
      "group_id": "group_123",
      "qa_id": "qa_456",
      "category": 0,
      "source_file": "0_0000_qwen3-14b_qwen3-14b-70_clm_10.jsonl",
      "line_number": 123
    },
    "question": "具体的问题内容"
  },
  "generated_answers": ["答案1", "答案2", "（共 k 条）"],
  "extra_info": {
    "original_sample_key": "0_0000_qwen3-14b_qwen3-14b-70_clm_10_123",
    "line_number": 123
  }
}
```

### 3.2 评估答案正确性

1. 对每条样本的每个生成答案调用评判模型，输出「正确」或「错误」。  
2. 根据正确数量划分难度：难例 / 中等 / 简单（阈值见 `config_step2`）。  
3. 将整条原始样本写入对应难度目录下的 JSON 文件。  

**模型**：Qwen3-14B  

**提示词（Chat 模板）**

```python
messages = [
    {
        "role": "system",
        "content": "你是一个公正的答案评判员。请判断生成的答案是否与标准答案意思一致，只回答'正确'或'错误'。",
    },
    {
        "role": "user",
        "content": f"""问题：{question}
标准答案：
{gt_text}
生成的答案：
{answer}
这个答案正确吗？请只回答"正确"或"错误"。"""",
    },
]
```

**提示词（纯文本回退）**

```text
你是一个公正的答案评判员。请判断以下生成的答案是否与标准答案意思一致。

【问题】
{question}

【标准答案】
{gt_text}

【生成的答案】
{answer}

请只回答"正确"或"错误"。
```

### 3.3 分类、去重与抽样

中等题去重时需兼容两种记忆块格式：

- **标准格式**（常见）：`Memories for user X`
- **Nano 格式**：`Memories for the Users`，有单引号和双引号

**目录含义**

| 目录 | 说明 |
|------|------|
| `easy/`、`medium/`、`hard/` | 第二步输出的全量简单 / 中等 / 难题样本 |
| `medium_dedup/` | 在 `medium/` 上按规则去重后的中等题 |
| `easy_pro/`、`medium_pro/`、`hard_pro/` | 第三步按配置比例与目标总量抽样后的子集 |

**去重与抽样记录：**

**最开始：**
- Easy 样本数: 136
- Medium 样本数: 1756---->1438
- Hard 样本数: 439
- 总分类样本数: 2331

**中等题去重后：**
- Easy：136 / 2013 ≈ 6.8%
- Medium：1438 / 2013 ≈ 71.4%
- Hard：439 / 2013 ≈ 21.8%

**按比例保留后：**
最大保留数方案  N=1360，Easy 全留，Hard 留 68，Medium 留 1156。
- Easy: 保留 136（占 10%）
- Medium: 保留 1156（占 85%）
- Hard: 保留 68（占 5%）
- 总样本数 1360（相比原 2013，删除了 653 个）
