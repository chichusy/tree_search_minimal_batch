from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def convert_gsm8k(df: pd.DataFrame):
    rows = df.to_dict(orient="records")
    out = []
    for i, row in enumerate(rows):
        item = {
            "id": i,
            "dataset": "gsm8k",
            "question": row["question"],
            "answer": row["answer"],
        }
        out.append(item)
    return out


def convert_math500(df: pd.DataFrame):
    rows = df.to_dict(orient="records")
    out = []
    for i, row in enumerate(rows):
        item = {
            "id": i,
            "dataset": "math500",
            "question": row["problem"],
            "answer": row["solution"],   # 为了和你现有框架兼容，统一放到 answer
            "solution": row["solution"],
            "level": row["level"],
            "subject": row["type"],      # type 改名成 subject，更直观
        }
        out.append(item)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_path", type=str, required=True)
    ap.add_argument("--output_path", type=str, required=True)
    ap.add_argument("--dataset_type", type=str, required=True, choices=["gsm8k", "math500"])
    args = ap.parse_args()

    input_path = Path(args.input_path)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(input_path)

    if args.dataset_type == "gsm8k":
        data = convert_gsm8k(df)
    elif args.dataset_type == "math500":
        data = convert_math500(df)
    else:
        raise ValueError(f"Unsupported dataset_type: {args.dataset_type}")

    with open(output_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"[OK] converted {len(data)} rows")
    print(f"input : {input_path}")
    print(f"output: {output_path}")


if __name__ == "__main__":
    main()