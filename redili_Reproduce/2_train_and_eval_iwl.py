"""
2_train_and_eval_iwl.py

训练一个小 instruct 模型的 SFT（full 或 LoRA），然后在测试集上评估 IWL 指标。

IWL 定义（与 finetuned_model_comparison.ipynb:160-161 一致）:
    模型 = SFT 后的模型
    prompt = prompt-other（不带 "no markdown" 后缀）
    不过 parse_bash（finetuned_model_comparison.ipynb 没调）

用法:
    sg docker bash
    python 2_train_and_eval_iwl.py --model ~/models/qwen2.5-coder-0.5b-instruct \
        --output_dir ~/runs/iwl-0.5b-lora --method lora

评分器: icalfa (InterCode-ALFA) = paper 主指标
  - icalfa = InterCode (exec in Docker) + mxbai-embed-large 输出余弦 ≥ 0.75
  - paper.tex:241, 292: 论文主 FEH 就是 icalfa
"""
import argparse
import csv
import json
import os
import re
import shutil
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          Trainer, TrainingArguments)

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

TRAIN_CSV = "/home/redili/github/NL2SH/nl2sh_alfa/train.csv"
TEST_CSV = "/home/redili/github/NL2SH/nl2sh_alfa/test.csv"
RESULTS_DIR = Path("/home/redili/github/NL2SH/redili_Reproduce/results")

SYSTEM_OTHER = ("Your task is to translate a natural language instruction to a Bash command. "
                "You will receive an instruction in English and output a Bash command that can "
                "be run in a Linux terminal.")


# ====================== 训练数据 ====================== #

def build_dataset(tok, train_csv, max_length):
    train = load_dataset("csv", data_files=train_csv, split="train")

    def chat(row):
        msgs = [
            {"role": "system", "content": SYSTEM_OTHER},
            {"role": "user", "content": row["nl"]},
            {"role": "assistant", "content": row["bash"]},
        ]
        text = tok.apply_chat_template(msgs, add_generation_prompt=False, tokenize=False)
        return {"text": text}

    def tok_fn(row):
        out = tok(row["text"], padding="max_length", truncation=True, max_length=max_length)
        out["labels"] = [-100 if t == tok.pad_token_id else t for t in out["input_ids"]]
        return out

    ds = train.map(chat, remove_columns=train.column_names)
    ds = ds.map(tok_fn, remove_columns=["text"])
    return ds


# ====================== 模型装载（含 LoRA） ====================== #

def load_model_with_lora(model_path, lora_r, lora_alpha, lora_dropout, lora_targets):
    from peft import LoraConfig, get_peft_model, TaskType
    base = AutoModelForCausalLM.from_pretrained(
        model_path, device_map="cuda", dtype=torch.bfloat16
    )
    cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
        target_modules=lora_targets, bias="none",
    )
    model = get_peft_model(base, cfg)
    model.print_trainable_parameters()
    return model


def load_model_full(model_path):
    return AutoModelForCausalLM.from_pretrained(
        model_path, device_map="cuda", dtype=torch.bfloat16
    )


def save_model(model, tok, output_dir, method):
    if method == "lora":
        # 合并 LoRA 到 base，存标准 HF 格式（方便后续直接 AutoModel.from_pretrained）
        model = model.merge_and_unload()
    model.save_pretrained(output_dir)
    tok.save_pretrained(output_dir)
    print(f"[save] {output_dir}")


# ====================== 评估 ====================== #

def parse_bash(text):
    for pat in [r"```bash\s*(.*?)\s*```", r"```(.*?)```", r"`(.*?)`"]:
        m = re.search(pat, text, re.DOTALL)
        if m:
            return m.group(1).strip()
    return text


def generate(tok, model, user):
    msgs = [{"role": "system", "content": SYSTEM_OTHER}, {"role": "user", "content": user}]
    text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    inputs = tok(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=100, do_sample=False,
            temperature=None, top_p=None, top_k=None,
        )
    return tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)


def evaluate_iwl(model_path, test_ds, scorer, force):
    short = Path(model_path).name
    out_csv = RESULTS_DIR / f"{short}_iwl.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if out_csv.exists() and not force:
        print(f"[skip] {out_csv} exists, delete to re-run")
        return _read_acc(out_csv)

    print(f"[load ft model] {model_path}")
    tok = AutoTokenizer.from_pretrained(model_path, clean_up_tokenization_spaces=False)
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path, device_map="cuda", dtype=torch.bfloat16
    )
    model.eval()

    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["index", "nl", "ground_truth", "model_command", "correct"])
        for idx, row in tqdm(enumerate(test_ds), total=len(test_ds), desc=f"eval {short}"):
            raw = generate(tok, model, row["nl"])
            # IWL: 不过 parse_bash（与 finetuned_model_comparison.ipynb 一致）
            ok = scorer(idx, raw)
            w.writerow([idx, row["nl"], row["bash"], raw, ok])

    del model
    torch.cuda.empty_cache()
    return _read_acc(out_csv)


def _read_acc(csv_path):
    with open(csv_path) as f:
        r = csv.reader(f); next(r)
        c = sum(1 for row in r if row[4] == "1")
    return c / 300


def make_scorer():
    """icalfa (InterCode-ALFA, paper 主指标)。需 Docker + ollama mxbai-embed-large。"""
    from icalfa import submit_command
    return lambda i, c: submit_command(index=i, command=c, eval_mode="embed", eval_param=0.75)


# ====================== main ====================== #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="base instruct 模型的本地路径")
    ap.add_argument("--method", choices=["lora", "full"], default="lora")
    ap.add_argument("--output_dir", required=True, help="SFT 后模型的保存路径")
    ap.add_argument("--train_csv", default=TRAIN_CSV)
    ap.add_argument("--test_csv", default=TEST_CSV)

    # 训练超参
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--batch", type=int, default=15)
    ap.add_argument("--grad_accum", type=int, default=5)
    ap.add_argument("--max_length", type=int, default=150)
    ap.add_argument("--seed", type=int, default=123)

    # LoRA 超参（默认比 paper 的 r=64 更小更稳定）
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--lora_targets", nargs="+",
                    default=["q_proj","k_proj","v_proj","o_proj",
                             "gate_proj","up_proj","down_proj"])

    ap.add_argument("--skip_train", action="store_true",
                    help="output_dir 已有训好的模型时只跑评估")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    # 训练
    if not args.skip_train and not Path(args.output_dir).joinpath("config.json").exists():
        print(f"\n========== TRAIN ({args.method}) ==========")
        print(f"  base = {args.model}")
        print(f"  out  = {args.output_dir}")
        print(f"  epochs={args.epochs} lr={args.lr} batch={args.batch}x{args.grad_accum} "
              f"max_len={args.max_length} seed={args.seed}")
        if args.method == "lora":
            print(f"  LoRA r={args.lora_r} alpha={args.lora_alpha} dropout={args.lora_dropout}")

        torch.manual_seed(args.seed)
        tok = AutoTokenizer.from_pretrained(args.model, clean_up_tokenization_spaces=False)
        tok.pad_token = tok.eos_token

        if args.method == "lora":
            model = load_model_with_lora(args.model, args.lora_r, args.lora_alpha,
                                         args.lora_dropout, args.lora_targets)
        else:
            model = load_model_full(args.model)

        ds = build_dataset(tok, args.train_csv, args.max_length)
        print(f"  train rows: {len(ds)}")

        targs = TrainingArguments(
            output_dir=Path(args.output_dir) / "_ckpt",
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.lr,
            weight_decay=0.01, max_grad_norm=2.0,
            bf16=True, seed=args.seed,
            logging_steps=50, save_strategy="no", report_to=[],
        )
        Trainer(model=model, args=targs, train_dataset=ds,
                processing_class=tok).train()

        save_model(model, tok, args.output_dir, args.method)

        del model
        torch.cuda.empty_cache()
    else:
        print(f"[skip_train] {args.output_dir} already exists")

    # 评估
    print(f"\n========== EVAL IWL ==========")
    test_ds = load_dataset("csv", data_files=args.test_csv, split="train")
    print(f"  test rows: {len(test_ds)}")
    scorer = make_scorer()
    print(f"  scorer: icalfa (InterCode-ALFA, paper 主指标)")
    acc = evaluate_iwl(args.output_dir, test_ds, scorer, args.force)
    print(f"\n[result] IWL accuracy = {acc:.3f}  (model={args.output_dir})")

    # 自动对比
    short = Path(args.model).name
    base_csv = RESULTS_DIR / short / "base.csv"
    if base_csv.exists():
        base_acc = _read_acc(base_csv)
        print(f"  vs Base ({short}) = {base_acc:.3f}  ->  delta = {acc - base_acc:+.3f}")


if __name__ == "__main__":
    main()
