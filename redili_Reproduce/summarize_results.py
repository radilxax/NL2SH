"""
summarize_results.py — 统计 results/ 下所有模型的 base/parse/icl 指标

用法:
    python summarize_results.py
    python summarize_results.py --out results/summary.txt   # 同时写文件
"""
import argparse
import csv
from pathlib import Path

RESULTS_DIR = Path("/home/redili/github/NL2SH/redili_Reproduce/results")
MODES = ["base", "parse", "icl"]


def count_correct(csv_path: Path) -> tuple[int, int]:
    """返回 (correct, total)。"""
    with open(csv_path, newline="") as f:
        r = csv.reader(f)
        next(r)
        total = 0
        correct = 0
        for row in r:
            total += 1
            if row[5] == "1":
                correct += 1
    return correct, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default=None, help="同时写入文件")
    args = ap.parse_args()

    models = sorted([d for d in RESULTS_DIR.iterdir() if d.is_dir()])

    if not models:
        print("results/ 下没有模型目录")
        return

    lines = []
    for model_dir in models:
        name = model_dir.name
        lines.append(f"[results] {name}")
        for mode in MODES:
            csv_path = model_dir / f"{mode}.csv"
            if csv_path.exists():
                correct, total = count_correct(csv_path)
                acc = correct / total if total > 0 else 0
                lines.append(f"  {mode:6s} = {acc:.3f}  ({correct}/{total})")
            else:
                lines.append(f"  {mode:6s} = N/A  (文件不存在)")
        lines.append("")

    output = "\n".join(lines)
    print(output)

    if args.out:
        Path(args.out).write_text(output)
        print(f"已写入 {args.out}")


if __name__ == "__main__":
    main()
