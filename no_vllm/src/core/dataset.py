# src/core/dataset.py
from __future__ import annotations

import csv
import json
from typing import Any, Dict, List


def load_dataset(path: str, fmt: str) -> List[Dict[str, Any]]:
    """
    支持两种格式：
      - jsonl: 每行一个 json，至少包含 question/prompt/input/query/text/problem 之一
      - csv:  表头包含 question/prompt/input/query/text/problem 之一
    """
    fmt = fmt.lower().strip()
    samples: List[Dict[str, Any]] = []

    if fmt == "jsonl":
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if not isinstance(obj, dict):
                    obj = {"_raw": obj}
                samples.append(obj)

    elif fmt == "csv":
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                samples.append(dict(row))

    else:
        raise ValueError(f"Unsupported dataset format: {fmt}")

    return samples


def extract_question(sample: Dict[str, Any]) -> str:
    """
    尽量兼容不同字段名。
    如果没有明确字段，回退为整个 sample 的 json 字符串。
    """
    for k in ("question", "problem", "prompt", "input", "query", "text"):
        v = sample.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()

    return json.dumps(sample, ensure_ascii=False)