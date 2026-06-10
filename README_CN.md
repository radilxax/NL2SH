# NL2SH: 改进小模型的自然语言到 Bash 命令翻译能力

[English Version](README.md)

基于 NAACL 2025 论文 [LLM-Supported Natural Language to Bash Translation](https://aclanthology.org/2025.naacl-long.555/)，对小参数量模型（0.5B-3B）在 NL2SH 任务上的表现进行复现与改进。

## 目标

复现论文中 Qwen2.5-Coder 系列的 Base / Parse / ICL / IWL 指标，并通过改进推理策略和训练方法提升小模型的 Bash 命令生成能力。

## 当前结果

| 模型 | Base | Parse | ICL (completion) | ICL (chat) | 论文 ICL |
|------|------|-------|-------------------|------------|----------|
| qwen2.5-coder-0.5b-instruct | 0.110 | 0.337 | - | 0.000 | 0.36 |
| qwen2.5-coder-1.5b-instruct | 0.220 | 0.497 | **0.503** | - | 0.44 |
| qwen2.5-coder-3b-instruct | 0.260 | 0.583 | **0.597** | - | 0.50 |

> 评分器：InterCode-ALFA (icalfa)，exec + mxbai-embed 余弦相似度 >= 0.75

## 关键发现：ICL 推理方式对小模型影响显著

论文使用 chat template 做 ICL 推理，但 0.5B 模型在 chat template 下会复读 ICL examples 而非为测试查询生成命令。改用 raw completion（纯文本续写，不经过 chat template）后，1.5B 和 3B 的 ICL 分数显著超过论文报告值。

- **chat template**: `apply_chat_template` -> system role + user role -> `model.generate()`
- **raw completion**: 纯文本 prompt -> `model.generate()` -> 取输出第一行

## 快速开始

### 环境准备

```bash
# 依赖
pip install torch transformers datasets tqdm icalfa accelerate

# Docker（icalfa 评分需要）
sudo usermod -aG docker $USER
sg docker bash

# 构建 icalfa Docker 镜像（首次，约 5-10 分钟）
bash redili_Reproduce/build_icalfa_images.sh

# 拉取嵌入模型（icalfa 评分需要）
ollama pull mxbai-embed-large
```

### 下载模型

```bash
export HF_ENDPOINT=https://hf-mirror.com
hf download Qwen/Qwen2.5-Coder-0.5B-Instruct --local-dir ~/models/qwen2.5-coder-0.5b-instruct
hf download Qwen/Qwen2.5-Coder-1.5B-Instruct --local-dir ~/models/qwen2.5-coder-1.5b-instruct
hf download Qwen/Qwen2.5-Coder-3B-Instruct   --local-dir ~/models/qwen2.5-coder-3b-instruct
```

### 运行评估

```bash
cd redili_Reproduce

# 选 ICL 示例（首次，约 3 分钟）
python 0_precompute_icl.py

# 跑 Base / Parse / ICL（ICL 默认用 completion 模式）
python 1_reproduce_no_train.py --models ~/models/qwen2.5-coder-1.5b-instruct

# 用 chat template 跑 ICL（与论文对齐，0.5B 会崩）
python 1_reproduce_no_train.py --models ~/models/qwen2.5-coder-0.5b-instruct --icl_mode chat --force

# 查看所有结果汇总
python summarize_results.py
```

## 项目结构

```
NL2SH/
├── paper/                          # 论文 LaTeX 源码
├── code/                           # 论文原始代码（notebooks）
├── nl2sh_alfa/                     # 数据集（本地 CSV）
│   ├── train.csv                   # 训练集（40,939 条）
│   └── test.csv                    # 测试集（300 条）
├── redili_Reproduce/               # 复现与改进代码
│   ├── 0_precompute_icl.py         # ICL 示例选择（mxbai-embed + k-means）
│   ├── 1_reproduce_no_train.py     # Base / Parse / ICL 评估
│   ├── 2_train_and_eval_iwl.py     # SFT 训练 + IWL 评估
│   ├── build_icalfa_images.sh      # 构建 Docker 评分镜像
│   ├── icl_examples.json           # 25 个 ICL 示例
│   ├── summarize_results.py        # 结果汇总脚本
│   ├── results/                    # 评估结果 CSV
│   └── PLAN.md                     # 复现与涨点计划
└── AGENTS.md
```

## 评估指标说明

| 指标 | Prompt | 后处理 | 模型 |
|------|--------|--------|------|
| Base | `prompt-baseline`（含 "no markdown" 约束） | 无 | 原始 Instruct 模型 |
| Parse | `prompt-other` | `parse_bash()` 去 markdown | 原始 Instruct 模型 |
| ICL | `prompt-other` + 25 个 few-shot 示例 | `parse_bash_completion()` | 原始 Instruct 模型 |
| IWL | `prompt-other` | 无 | SFT 后的模型 |

## 改进方向

1. **ICL 推理优化**：raw completion 替代 chat template，提升 few-shot 效果
2. **LoRA 微调**：用较小的 rank（r=8-16）避免小模型训飞
3. **训练策略**：减少 epoch 数（3-5 vs 论文的 10），加 eval loss 早停
4. **数据筛选**：用训练集高质量子集而非全量 40k 数据

## 引用

```bibtex
@inproceedings{westenfelder-etal-2025-llm,
    title = "{LLM}-Supported Natural Language to Bash Translation",
    author = "Westenfelder, Finnian and Hemberg, Erik and Moskal, Stephen and O'Reilly, Una-May and Chiricescu, Silviu",
    booktitle = "Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers)",
    month = apr,
    year = "2025",
    address = "Albuquerque, New Mexico",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.naacl-long.555/",
    pages = "11135--11147",
}
```

## 致谢

- 原始论文：[LLM-Supported Natural Language to Bash Translation](https://aclanthology.org/2025.naacl-long.555/) (NAACL 2025)
- 评分器：[InterCode-ALFA](https://github.com/westenfelder/InterCode-ALFA)
- 数据集：[NL2SH-ALFA](https://huggingface.co/datasets/westenfelder/NL2SH-ALFA)
