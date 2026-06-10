# 复现 + 涨点计划

## 目标
复现论文 `tab:model_results`（`paper/paper.tex:317-329`）中 Qwen2.5-Coder 系列
**Base / Parse / ICL / IWL** 四个指标，并尝试通过改进训练策略提升 IWL。

CD 跳过——论文对 Qwen 报告 CD 全部负收益（paper.tex:322-325，0.5B CD=0.05 vs Base=0.10）。

## 评分器（已统一为 icalfa）
- **icalfa = paper 主指标**（paper.tex:241, 292："InterCode-ALFA, exec+mxbai-embed F1=0.90"）
- `icalfa.submit_command(index, command, eval_mode="embed", eval_param=0.75)`
  - exec 在 Docker 沙盒里跑命令
  - 输出用 mxbai-embed-large-v1 算余弦相似度，≥ 0.75 判正确
- **跑前准备**（一次性）：
  1. `sg docker bash` （让 docker 组有效）
  2. `bash /home/redili/github/NL2SH/redili_Reproduce/build_icalfa_images.sh` （构建 5 个 intercode-bash 镜像，约 5-10 分钟）
  3. `ollama pull mxbai-embed-large` （669MB，3-5 分钟）
- 脚本不再有 `--no_docker` 开关（论文就是用 icalfa，去掉 fallback 后才能严格复现）

## 文件
| 文件 | 作用 |
|---|---|
| `0_precompute_icl.py` | 一次性：从 train.csv 用 mxbai-embed + k-means 选 25 个 ICL 示例，存 `icl_examples.json` |
| `1_reproduce_no_train.py` | 跑 Base / Parse / ICL（**不训练**，直接用 HF 拉的 instruct 模型，icalfa 评分） |
| `2_train_and_eval_iwl.py` | 跑 full / LoRA SFT，icalfa 评分得 IWL 指标 |
| `build_icalfa_images.sh` | 一次性构建 5 个 intercode-bash Docker 镜像（`--network=host` 绕过容器网络） |
| `icl_examples.json` | 25 个 ICL 示例（被上面两个脚本共享） |
| `results/` | 输出 CSV：每模型每模式一个文件 |

## 跑法（推荐 3 个小模型，全是 Qwen2.5-Coder-Instruct，避开 Llama 因镜像无权限）

```bash
sg docker bash
cd /home/redili/github/NL2SH/redili_Reproduce

# 0. 一次性：拉模型（如未下）
export HF_ENDPOINT=https://hf-mirror.com
hf download Qwen/Qwen2.5-Coder-0.5B-Instruct --local-dir ~/models/qwen2.5-coder-0.5b-instruct
hf download Qwen/Qwen2.5-Coder-1.5B-Instruct --local-dir ~/models/qwen2.5-coder-1.5b-instruct
hf download Qwen/Qwen2.5-Coder-3B-Instruct   --local-dir ~/models/qwen2.5-coder-3b-instruct

# 1. 选 25 个 ICL 示例（首次必跑，~3 min，仅离线 mxbai，不调 icalfa）
python 0_precompute_icl.py

# 2. 跑 Base/Parse/ICL（~30 min/模型）
python 1_reproduce_no_train.py \
    --models ~/models/qwen2.5-coder-0.5b-instruct \
             ~/models/qwen2.5-coder-1.5b-instruct \
             ~/models/qwen2.5-coder-3b-instruct

# 3. SFT + 跑 IWL（先 LoRA，全参数容易把 0.5B 训坏）
python 2_train_and_eval_iwl.py \
    --model ~/models/qwen2.5-coder-0.5b-instruct \
    --method lora \
    --output_dir ./checkpoints/qwen2.5-coder-0.5b-lora
```

## 关键设计决策
1. **只用 Qwen2.5-Coder-Instruct**（HF 镜像可达 + Llama 系列被 403 拦截，AGENTS.md:99-108）
2. **ICL 选例 = mxbai-embed 命令嵌入 + k-means(k=25) 取每簇最近样本**（paper.tex:259-261）
3. **Parse/ICL 都过 `parse_bash()`**（保守做法，避免 markdown 干扰）
4. **IWL = prompt-other + 不过 parse**（`finetuned_model_comparison.ipynb:160-161` 的做法）
5. **默认 LoRA r=16/α=32/dropout=0.05**（比 paper 的 r=64/α=32 更稳定，paper 自己 IWL 反而比 Parse 差说明 r=64 训飞了）
6. **评分器统一为 icalfa**（paper 主指标，无 fallback）

## 评估指标对应
| 指标 | Prompt | 后处理 | 模型 |
|---|---|---|---|
| Base | `prompt-baseline`（带 "no markdown"） | 无 | Instruct 原模型 |
| Parse | `prompt-other`（不带 "no markdown"） | `parse_bash()` | Instruct 原模型 |
| ICL | `prompt-icl`（prompt-other + 25 例） | `parse_bash()` | Instruct 原模型 |
| IWL | `prompt-other` | 无 | **SFT 后的模型** |

## 涨点思路
如果 SFT 训出的 IWL 仍然低于 Parse（paper 里 0.5B 情况：IWL 0.27 < Parse 0.35），按下面顺序调：

1. **降 LoRA rank** 到 r=8 或 r=16（r=64 容易在小模型上训飞）
2. **少训**（3-5 epochs，paper 的 10 epochs 太多）
3. **用 train.csv 的小子集**（如 5000 条高质量样本）—— paper 自己也承认 40k 没全验证
4. **加 eval loss 早停**：当前 finetune.py 没早停，训到过拟合
5. **换 base** 用 `Qwen2.5-Coder-1.5B-Instruct` 训，再跟 0.5B 的 Parse 比（paper.tex:354 提到 1.5B-训后 ≈ 3B-base 的 Parse）

## 文件-代码-Prompt 对照
- `parse_bash()` → `1_reproduce_no_train.py`（仿 `code/model_comparison.ipynb:168`）
- ICL 25 例选法 → `0_precompute_icl.py`（仿 `paper.tex:259-261`）
- SFT 训练循环 → `2_train_and_eval_iwl.py`（基于 `code/finetune/finetune.py`，加 peft 支持）
- IWL 评估 → `2_train_and_eval_iwl.py`（仿 `code/finetuned_model_comparison.ipynb:138-166`）
