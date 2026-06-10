"""
0_precompute_icl.py  (旧名 0_select_icl_examples.py)

一次性**预处理**脚本：生成 25 个 ICL 示例并存到 icl_examples.json。
**不跑评估**，跑过一次后被 1_reproduce_no_train.py / 2_train_and_eval_iwl.py 复用。

两种模式:
  默认:  --from_paper      ← 严格复现：直接用 paper.tex:486-537 列出的 25 条
  备选:  --from_kmeans k=25 ← 探索性：用 mxbai-embed + k-means 从 train.csv 选

⚠️ 注意: paper 的文字（tex:261）说"k-means 选"，但图里列的 25 条有 11 条
不在 train.csv 里（含占位路径 xx.sh / /path / nonsense_dir / foobar/test_file
等），显然是手挑的。严格复现请用 --from_paper。
"""
import json
import argparse
from pathlib import Path

import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.cluster import MiniBatchKMeans


TRAIN_CSV = "/home/redili/github/NL2SH/nl2sh_alfa/train.csv"
EMBED_MODEL = "mixedbread-ai/mxbai-embed-large-v1"
DEFAULT_OUT = "/home/redili/github/NL2SH/redili_Reproduce/icl_examples.json"


# paper.tex:486-537 完整复制（25 条，与 fig:prompt-icl 一一对应）
PAPER_ICL_EXAMPLES = [
    ("Show logged-in users info", "w"),
    ('Print the contents of "xx.sh"', "cat xx.sh"),
    ('Change owner to "root" and group to "www-data" of "/foobar/test_file"',
     "chown root:www-data /foobar/test_file"),
    ("delete all the text files in the current folder",
     "find . -type f -name \"*.txt\" -delete"),
    ("find all the files in the /path folder and delete them",
     "find /path -type f -delete"),
    ("Print the exit status of the last executed command", "echo $?"),
    ("Display a tree of processes", "pstree"),
    ("Display information about all CPUs", "lscpu"),
    ("Make an HTTPS GET request to example.com and dump the contents in `stdout`",
     "curl https://example.com"),
    ("Display system memory", "free"),
    ("List all files, including hidden files", "ls -a"),
    ("Print a sequence from 1 to 10", "seq 10"),
    ("Get the properties of all the user limits", "ulimit -a"),
    ("List the name and status of all services", "service --status-all"),
    ("Display a calendar for the current month", "cal"),
    ("Show the environment", "env"),
    ("create directory TestProject", "mkdir TestProject"),
    ("Query the default name server for the IP address of example.com",
     "nslookup example.com"),
    ("Print Hello World", "echo \"Hello World\""),
    ("List all bound commands and their hotkeys", "bind -p"),
    ("Display the openssl version", "openssl version"),
    ("Print current time, uptime, number of logged-in users", "uptime"),
    ("Print file system disk space usage", "df"),
    ("List all configuration values available", "getconf -a"),
    ("Delete empty folder 'nonsense_dir'.", "rmdir nonsense_dir"),
]


def from_paper(out_path: Path):
    data = [{"nl": nl, "bash": b} for nl, b in PAPER_ICL_EXAMPLES]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"[done] wrote {len(data)} paper-canonical ICL examples to {out_path}")


def from_kmeans(out_path: Path, k: int, seed: int):
    print(f"[load] {TRAIN_CSV}")
    train = load_dataset("csv", data_files=TRAIN_CSV, split="train")
    print(f"  {len(train)} training rows")

    print(f"[embed] {EMBED_MODEL}")
    embed = SentenceTransformer(EMBED_MODEL)
    commands = [r["bash"] for r in train]
    embs = embed.encode(commands, batch_size=64, show_progress_bar=True,
                        normalize_embeddings=True, convert_to_numpy=True)
    print(f"  embs shape = {embs.shape}")

    print(f"[kmeans] k={k}")
    km = MiniBatchKMeans(n_clusters=k, random_state=seed, n_init=10, batch_size=1024)
    km.fit(embs)

    selected = []
    for c in km.cluster_centers_:
        d = np.linalg.norm(embs - c, axis=1)
        idx = int(d.argmin())
        selected.append({"nl": train[idx]["nl"], "bash": train[idx]["bash"]})

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(selected, f, ensure_ascii=False, indent=2)
    print(f"[done] wrote {len(selected)} k-means ICL examples to {out_path}")
    print("  ⚠️ 这 25 条跟 paper 图里的 25 条不同（k-means 随机性 + 11 条 paper 例不在 train.csv）")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--from_paper", action="store_true", default=True,
                   help="用 paper.tex:486-537 列出的 25 条（默认，严格复现）")
    p.add_argument("--from_kmeans", type=int, default=None, metavar="K",
                   help="用 k-means 从 train.csv 选 K 条（探索性，结果跟 paper 不同）")
    p.add_argument("--out", default=DEFAULT_OUT)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--force", action="store_true",
                   help="覆盖已存在的 icl_examples.json")
    args = p.parse_args()

    out = Path(args.out)
    if out.exists() and not args.force:
        with open(out) as f:
            data = json.load(f)
        print(f"[skip] {out} exists ({len(data)} examples). 删掉它或加 --force 重生。")
        return

    if args.from_kmeans is not None:
        from_kmeans(out, args.from_kmeans, args.seed)
    else:
        from_paper(out)


if __name__ == "__main__":
    main()
