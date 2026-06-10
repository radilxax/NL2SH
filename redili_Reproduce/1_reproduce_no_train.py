"""
1_reproduce_no_train.py

复现论文 Base / Parse / ICL 三个指标（不动模型权重，直接用 HF Instruct 模型）。

用法:
    sg docker bash
    python 1_reproduce_no_train.py --models ~/models/qwen2.5-coder-0.5b-instruct

模式定义（与 paper.tex:309 一致）:
  Base : prompt-baseline + 不过 parse_bash
  Parse: prompt-other    + parse_bash()
  ICL  : prompt-icl(prompt-other + 25 examples) + parse_bash()

评分器: icalfa (InterCode-ALFA) = paper 主指标
  - icalfa = InterCode (exec in Docker) + mxbai-embed-large 输出余弦 ≥ 0.75
  - paper.tex:241, 292: 论文主 FEH 就是 icalfa
  - 调用: icalfa.submit_command(index, command, eval_mode="embed", eval_param=0.75)
  - 跑前确保: 5 个 intercode-bash 镜像已构建 (build_icalfa_images.sh) +
              ollama 已 pull mxbai-embed-large
"""
import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# 抑制 jupyter-only 依赖
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

# 路径
TEST_CSV = "/home/redili/github/NL2SH/nl2sh_alfa/test.csv"
ICL_JSON = "/home/redili/github/NL2SH/redili_Reproduce/icl_examples.json"
RESULTS_DIR = Path("/home/redili/github/NL2SH/redili_Reproduce/results")

# prompt 模板（与 paper/paper.tex:441-545 一致）
SYSTEM_BASELINE = ("Your task is to translate a natural language instruction to a Bash command. "
                   "You will receive an instruction in English and output a Bash command that can "
                   "be run in a Linux terminal. You will not output markdown or other formatting. "
                   "You will not include additional information.")
SYSTEM_OTHER = ("Your task is to translate a natural language instruction to a Bash command. "
                "You will receive an instruction in English and output a Bash command that can "
                "be run in a Linux terminal.")


# ----------------------------- 工具函数 ----------------------------- #

def parse_bash(text: str) -> str:
    """与 paper sec:markdown-parser 一致：按 ```bash -> ``` -> ` 顺序匹配第一个 code block。
    增加对未闭合代码块的处理（输出被截断时 ```bash 后没有闭合 ```）。"""
    # 1) 尝试匹配闭合的代码块
    for pat in [r"```bash\s*(.*?)\s*```", r"```(.*?)```"]:
        m = re.search(pat, text, re.DOTALL)
        if m:
            return m.group(1).strip()
    # 2) 未闭合的 ```bash：取 ```bash 之后的所有内容
    m = re.search(r"```bash\s*(.*)", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    # 3) 行内 backtick（非 triple-backtick）
    m = re.search(r"(?<!`)`(?!`)(.*?)(?<!`)`(?!`)", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return text


def parse_bash_completion(text: str) -> str:
    """ICL raw completion 模式专用解析。
    模型输出形如 'ls\\n\\nlist files in the /path folder\\nls /path\\n...'
    只取第一行作为命令（后续行是模型继续生成的 examples）。"""
    text = text.strip()
    if not text:
        return text
    # 取第一个非空行
    first_line = text.split("\n")[0].strip()
    return first_line


def build_icl_user_msg(query: str, icl_examples: list) -> str:
    """paper.tex:486-538 格式：每个 example 是 "<nl>\\n<bash>"（单换行），pairs 之间双换行，最后拼测试 query。"""
    pairs = []
    for ex in icl_examples:
        pairs.append(f"{ex['nl']}\n{ex['bash']}")
    pairs.append(query)
    return "\n\n".join(pairs)


def load_model(path: str):
    """Qwen 专用加载（AGENTS.md 强调的 quirk）。"""
    tok = AutoTokenizer.from_pretrained(path, clean_up_tokenization_spaces=False)
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        path, device_map="cuda", dtype=torch.bfloat16
    )
    model.eval()
    return tok, model


def generate(tok, model, system: str, user: str, max_new_tokens: int = 256) -> str:
    """Chat template 模式（Base / Parse）。Qwen 不传 eos_token_id/pad_token_id。"""
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    inputs = tok(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            temperature=None, top_p=None, top_k=None,
        )
    return tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)


def generate_completion(tok, model, prompt: str, max_new_tokens: int = 256) -> str:
    """Raw completion 模式（ICL）。不用 chat template，直接续写文本。
    论文 ICL 用纯文本 few-shot prompt，chat template 的 special tokens 会破坏 pattern。"""
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            temperature=None, top_p=None, top_k=None,
        )
    return tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)


# ----------------------------- 评分器 ----------------------------- #

class IcalfaScorer:
    """icalfa = paper 主指标 (InterCode-ALFA, paper.tex:241,292)。
    需要 Docker + ollama mxbai-embed-large。"""
    def __init__(self):
        from icalfa import submit_command
        self.submit_command = submit_command

    def __call__(self, index: int, command: str) -> int:
        return self.submit_command(index=index, command=command,
                                   eval_mode="embed", eval_param=0.75)


# ----------------------------- 主流程 ----------------------------- #

def evaluate_model(model_path: str, test_ds, icl_examples, scorer, modes, icl_mode: str, force: bool):
    short = model_path.rstrip("/").split("/")[-1]
    model_dir = RESULTS_DIR / short
    model_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n========== {short} ==========")
    print(f"[load] {model_path}")
    tok, model = load_model(model_path)

    # 准备每个 mode 的结果文件
    handles = {}
    writers = {}
    for mode in modes:
        out_path = model_dir / f"{mode}.csv"
        if out_path.exists() and not force:
            print(f"  [skip] {out_path} already exists")
            continue
        f = open(out_path, "w", newline="")
        w = csv.writer(f)
        w.writerow(["index", "nl", "ground_truth", "model_raw", "model_command", "correct"])
        handles[mode] = f
        writers[mode] = w

    if not handles:
        print(f"  all {len(modes)} modes already done for {short}, skipping inference")
        del model
        torch.cuda.empty_cache()
    else:
        pbar = tqdm(enumerate(test_ds), total=len(test_ds), desc=short)
        for idx, row in pbar:
            nl = row["nl"]
            gt = row["bash"]

            for mode in modes:
                if mode == "base":
                    system, user = SYSTEM_BASELINE, nl
                    raw = generate(tok, model, system, user)
                    cmd = raw
                elif mode == "parse":
                    system, user = SYSTEM_OTHER, nl
                    raw = generate(tok, model, system, user)
                    cmd = parse_bash(raw)
                elif mode == "icl":
                    if icl_mode == "completion":
                        icl_prompt = SYSTEM_OTHER + "\n\n" + build_icl_user_msg(nl, icl_examples) + "\n"
                        raw = generate_completion(tok, model, icl_prompt)
                        cmd = parse_bash_completion(raw)
                    else:
                        icl_user = build_icl_user_msg(nl, icl_examples)
                        raw = generate(tok, model, SYSTEM_OTHER, icl_user)
                        cmd = parse_bash(raw)
                else:
                    raise ValueError(f"unknown mode {mode}")

                ok = scorer(idx, cmd)
                writers[mode].writerow([idx, nl, gt, raw, cmd, ok])

        for f in handles.values():
            f.close()

        del model
        torch.cuda.empty_cache()

    # 打 accuracy
    print(f"\n[results] {short}")
    for mode in modes:
        out_path = model_dir / f"{mode}.csv"
        if not out_path.exists():
            continue
        with open(out_path) as f:
            r = csv.reader(f); next(r)
            correct = sum(1 for row in r if row[5] == "1")
        acc = correct / 300
        print(f"  {mode:6s} = {acc:.3f}  ({correct}/300)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True, help="本地模型路径（已 hf download 过的）")
    ap.add_argument("--modes", nargs="+", default=["base", "parse", "icl"],
                    choices=["base", "parse", "icl"])
    ap.add_argument("--test_csv", default=TEST_CSV)
    ap.add_argument("--icl_json", default=ICL_JSON)
    ap.add_argument("--icl_mode", default="completion", choices=["completion", "chat"],
                    help="completion: raw text续写(效果更好); chat: chat template(与论文对齐)")
    ap.add_argument("--force", action="store_true", help="覆盖已有结果")
    args = ap.parse_args()

    print(f"[load test] {args.test_csv}")
    test_ds = load_dataset("csv", data_files=args.test_csv, split="train")
    print(f"  {len(test_ds)} test rows")

    if any(m == "icl" for m in args.modes):
        if not Path(args.icl_json).exists():
            print(f"[err] ICL 模式需要 {args.icl_json}，先跑 0_select_icl_examples.py")
            sys.exit(1)
        with open(args.icl_json) as f:
            icl_examples = json.load(f)
        print(f"[icl] loaded {len(icl_examples)} examples from {args.icl_json}")
    else:
        icl_examples = []

    print(f"[scorer] icalfa (InterCode-ALFA, paper 主指标)")
    if "icl" in args.modes:
        print(f"[icl_mode] {args.icl_mode}")
    scorer = IcalfaScorer()

    for m in args.models:
        evaluate_model(m, test_ds, icl_examples, scorer, args.modes, args.icl_mode, args.force)

    print("\n[done] see", RESULTS_DIR)


if __name__ == "__main__":
    main()
