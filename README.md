# locomo数据清洗
## 一 快速清洗
### 1.1一步清洗


### 1.2分步清洗



## 二 项目结构




## 三 步骤细节说明
### 3.1.生成答案
- 模型：Qwen3-8B
- 支持vLLM加速和批量生成
- 支持断点续传
每个样本将生成k个答案，保存文件中将一行一个样本，样本结构如下：  

（1）sample_key：作为样本唯一标识，由源样本名（去掉后缀）+ 行号组成，例如："0_14714705_gpt-4.1-mini_qwen3-14b-30_clm_10_4"，注意原样本名不是data_source，因为很多文件都来自同一个data_source，不能保证sample_key的唯一性，增加该key是为了断点续传、debug、后续步骤评估和分类溯源

（2）sample：原样本内容:  
- data_source
        数据来源的模型或配置，示例："0_14714705_gpt-4" 
  
- prompt
        content：完整的提示词，包含系统指令、记忆内容、问题等 role：这里是 "user"。

- ability
        字符串，表示任务能力类型，这里是 "memory"（记忆检索）。

- reward_model
        ground_truth：答案（raw 和 fixed 列表两种格式）
        style：规则类型，这里是 "rule"

- extra_info
        index：在数据集中的序号
        split：数据集划分（"train"）
        group_id：对话组 ID
        qa_id：问答对 ID
        category：类别编号
        source_file：来源文件名
- question
        字符串，用户的问题。

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
  "generated_answers": ["答案1", "答案2", ..., "答案128"],
  "extra_info": {
    "original_sample_key": "0_0000_qwen3-14b_qwen3-14b-70_clm_10_123",
    "line_number": 123
  }
}
```



### 3.2评估答案正确性
模型：用Qwen3-14B评估
提示词：
- chat模板
```python
    messages = [
        {"role": "system", "content": "你是一个公正的答案评判员。请判断生成的答案是否与标准答案意思一致，只回答'正确'或'错误'。"},
        {"role": "user", "content": 
        f"""问题：{question}
            标准答案：
            {gt_text}
            生成的答案：
            {answer}
            这个答案正确吗？请只回答"正确"或"错误"。"""
        }
    ]
```
- 纯文本模板
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
### 3.3样本分类并整合

