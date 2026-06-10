# NL2SH — Agent Guide

NAACL 2025 paper: LLM-Supported Natural Language to Bash Translation. Full finetuning (not LoRA) of small LMs (0.5B–8B) on HuggingFace dataset `westenfelder/NL2SH-ALFA`, evaluated via `icalfa` (embedding similarity).

## Quick commands

```bash
# finetune (in code/finetune/)
pip install -r requirements.txt
python finetune.py > finetune.log 2>&1 & disown

# benchmark a finetuned model (editable notebook)
# run code/finetuned_model_comparison.ipynb
```

## Dataset

- `westenfelder/NL2SH-ALFA` on HuggingFace
- Fields: `nl` (prompt), `bash` (ground truth command)
- **Load convention:** config arg selects train/test, `split` is always `"train"`:

```python
train_dataset = load_dataset("westenfelder/NL2SH-ALFA", "train", split="train")
test_dataset  = load_dataset("westenfelder/NL2SH-ALFA", "test",  split="train")
```

- Test set = 300 examples (in the actual CSV; dataset card says 600 but file disagrees — trust `len()`)
- Train set = 40,639 examples (unverified), test = 300 (manually verified)

## Evaluation

```python
from icalfa import submit_command
correct = submit_command(index=index, command=model_command, eval_mode="embed", eval_param=0.75)
```

- Embeds the predicted command and checks cosine similarity ≥ 0.75 vs ground truth.
- Results are cached per-model in `ft_model_results/{model_short_name}.csv` — delete to re-run.

## Inference quirks

- **Tokenizer:** always `clean_up_tokenization_spaces=False`, `pad_token = eos_token`.
- **Model:** `device_map="cuda"`, `torch_dtype=torch.bfloat16` (or `dtype=torch.bfloat16` in newer transformers).
- **Generation:** greedy (`do_sample=False`, `max_new_tokens=100`).
- **Llama models:** pass `eos_token_id=[eos_id, <|eot_id|>]` + `pad_token_id=eos`.
- **Qwen models:** omit `eos_token_id`/`pad_token_id` in `model.generate()`.
- **Output parser:** strip markdown with `parse_bash()` (regex: triple-backtick → inline-backtick → raw text).

## Finetuning details (`code/finetune/finetune.py`)

- **Chat template format:** system + user + assistant (no generation prompt).
- **Padding masked from loss:** `labels = -100 where input_id == pad_token_id`.
- **Truncation:** 150 tokens max.
- **Hyperparams:** 10 epochs, lr=1e-5, batch=15/15, grad_accum=5, bf16, weight_decay=0.01, max_grad_norm=2.
- **Script pushes to HF hub** at the end — change `westenfelder/NL2SH` to your own repo in `trainer.save_model(...)` / `push_to_hub(...)`.

## Model-specific notes

- Original paper used **Qwen2.5-Coder** (0.5B–7B) and **Llama 3.2/3.1** (1B–8B).
- No existing fine-tuned Qwen3.5 models yet. To try Qwen3.5-0.8B (or any new base), you will need to:
  1. Update `model_id` in `finetune.py`.
  2. Check `apply_chat_template` output format and generation kwargs — Qwen3.5 may differ from Qwen2.5.
  3. Re-benchmark after finetuning.

## Loading models and datasets on this server

This server cannot reach `huggingface.co` directly. Use the mirror `https://hf-mirror.com`. The dataset is already vendored locally at `nl2sh_alfa/` in the project root; load it from there to avoid the network entirely.

### 1. `os.environ["HF_ENDPOINT"]` in Jupyter is unreliable

`transformers.AutoModel.from_pretrained` (and similar `from_pretrained` calls) does not always read `HF_ENDPOINT` set via `os.environ[...]` in a notebook cell, depending on import order / cell order. Don't rely on it.

### 2. Load from a local path — the most reliable approach

```python
# dataset (already vendored in this repo)
from datasets import load_dataset
base = "/home/redili/github/NL2SH/nl2sh_alfa"
train_dataset = load_dataset("csv", data_files=f"{base}/train.csv", split="train")
test_dataset  = load_dataset("csv", data_files=f"{base}/test.csv",  split="train")

# model — pre-download with `hf download ... --local-dir ...`, then load from disk
model_id = "/home/redili/models/qwen2.5-coder-0.5b"   # example
AutoModelForCausalLM.from_pretrained(model_id, device_map="cuda", dtype=torch.bfloat16)
```

### 3. `huggingface-cli` is deprecated — use `hf`

```bash
# install if missing
pip install -U "huggingface_hub[cli]"

# use the new `hf` command (HF_ENDPOINT env var IS respected by the CLI)
export HF_ENDPOINT="https://hf-mirror.com"
hf download <org>/<repo> --local-dir ~/models/<name>
```

### 4. `meta-llama/*` models are NOT on the mirror

`hf-mirror.com` returns 403 for all Meta-Llama repos (gated, license forbids redistribution). To use Llama you must:
- Have a HuggingFace account and have accepted Meta's license on the model page.
- Run from a host that can reach `huggingface.co` directly (this server cannot).
- `huggingface-cli login` (or `hf auth login`) and provide a token.

For this server, **prefer Qwen** (open license, fully mirrored). Good choices that match the paper's baselines:
- `Qwen/Qwen2.5-Coder-0.5B-Instruct` (~1 GB, fast)
- `Qwen/Qwen2.5-Coder-1.5B-Instruct` (~3 GB)
- `Qwen/Qwen3.5-0.8B` (~1.6 GB, newest)

## Docker for icalfa evaluation

`icalfa.submit_command()` requires Docker — it executes commands in a sandboxed
container. The current user must be in the `docker` group, or every benchmark
call fails with `PermissionError(13, 'Permission denied')`.

```bash
sudo usermod -aG docker $USER   # add self to docker group
# Then log out of SSH and log back in (or `exec sg docker -c "bash"` to test
# in the current shell without re-logging-in)
docker ps                       # should not error
```

First benchmark run will pull 5 Docker images (~500 MB total, one-time cost).
There is no Docker-free mode in `icalfa`; even `eval_mode="embed"` still spawns
containers (it just changes the comparison from string-match to cosine-similarity
of command **outputs**, not the commands themselves).

## Conda env gotcha

The user's conda env (`rdlpytorch`) ships with `transformers` and `torch` but **not** `accelerate`. `from_pretrained(..., device_map="cuda")` will fail with a `ValueError: ... requires accelerate`. Fix:

```bash
pip install accelerate
```

## .gitignore is highly restrictive

Root `.gitignore` = `*` then un-ignores specific dirs/files. Any new file not explicitly listed in `.gitignore` will be invisible to git. Add entries if you create new scripts or outputs. In particular: notes, training logs, vendored datasets, and downloaded model weights are all untracked by default — keep that in mind for reproducibility.

## File map

```
code/
  example.ipynb                 → starter: load data + run model + benchmark
  model_comparison.ipynb        → reproduce best model (+ parser) results
  finetuned_model_comparison.ipynb  → reproduce fine-tuned model results
  feh_comparison.ipynb          → reproduce FEH comparison results
  requirements.txt              → all dependencies
  finetune/
    finetune.py                 → full finetuning script (notebook-free, standalone)
    Modelfile                   → Ollama model config for GGUF deployment
    README.md                   → server setup, GGUF conversion, Ollama push
nl2sh_alfa/                     → vendored dataset (CSV) — load from here, not HF

## Hard rules (NEVER violate)

### 工作目录约束
- **禁止写入 `/tmp/`**、系统根目录的任何位置、或 conda 环境目录里的 `site-packages`
- 所有用户可见/可恢复的文件（日志、构建产物、icalfa 缓存、下载的模型、实验结果）**必须放在 `/home/redili/...` 下**
- 例外：read 临时只读操作（`/proc`、`/sys` 之类）允许，写入一律禁止
- 唯一允许的 `/tmp` 使用：通过 OpenCode tool 的 `/tmp/opencode`（pre-approved），但这仅用于 tool 自身的临时工作，**不用于存储用户实验数据**

### icalfa Docker 规则
- 跑 icalfa 前必须先 `sg docker bash` 进入带 docker 组的 shell
- icalfa 首次跑会从 5 个本地 Dockerfile 构建镜像，**耗时 5-10 分钟，不要 Ctrl-C**
- 如果 Docker 构建报 `apt-get ... returned a non-zero code: 100`，是 Docker build 上下文无网络访问外网仓库（与主机 `hf-mirror.com` 是否可达无关——Docker daemon 走自己的网络栈）
- **本服务器的容器网络是坏的**：所有 HTTP 出站都被中间 nginx（Server: nginx/1.18.0 Ubuntu）拦截并返回 404。Dockerfile 默认的 `archive.ubuntu.com`、`mirrors.aliyun.com`、`mirrors.tuna.tsinghua.edu.cn` 全部 404
- **修法**：在 sg docker bash 里用 `--network=host` 手动构建 5 个镜像。脚本在 `redili_Reproduce/build_icalfa_images.sh`，会构建 `intercode-bash-{1..5}:latest`（icalfa 期望的 tag），icalfa 检测到镜像存在就跳过构建
- **ollama 嵌入模型**：icalfa 调 `http://localhost:11434/api/embeddings` 用 `mxbai-embed-large` 做余弦相似度，本服务器 ollama 缺这个模型，要在 sg docker bash 里 `ollama pull mxbai-embed-large`（669MB，3-5 分钟）

### 修改 AGENTS.md 后的告知
- 每次往 AGENTS.md 写新规则后，必须**显式告诉用户**"已写入，下次会话会自动加载"，避免规则被悄悄覆盖丢失
```
