from __future__ import annotations

import re
from typing import Optional, Dict, Any, Tuple

_NUM_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")
_BOXED_RE = re.compile(r"\\boxed\{([^{}]+)\}")
_FINAL_PATTERNS = [
    re.compile(r"final\s+answer\s*(?:is|:)?\s*([^\n\r.]*)", re.IGNORECASE),
    re.compile(r"the\s+answer\s*(?:is|:)?\s*([^\n\r.]*)", re.IGNORECASE),
    re.compile(r"answer\s*(?:is|:)?\s*([^\n\r.]*)", re.IGNORECASE),
]


def normalize_num_str(x: Optional[str]) -> Optional[str]:
    """
    保留原有 GSM8K 数值归一化逻辑，不改行为。
    """
    if x is None:
        return None
    s = str(x).strip()
    if not s:
        return None
    s = s.replace(",", "").replace("$", "").replace("%", "")
    nums = _NUM_RE.findall(s)
    if not nums:
        return None
    try:
        val = float(nums[-1])
    except Exception:
        return nums[-1].strip()
    if abs(val - round(val)) < 1e-9:
        return str(int(round(val)))
    return f"{val:.12f}".rstrip("0").rstrip(".")


def extract_gsm8k_gt_from_answer_field(answer_text: str) -> Optional[str]:
    """
    保留原有 GSM8K GT 提取逻辑，不改行为。
    只认 answer 字段中的 #### xxx
    """
    if not answer_text:
        return None
    m = re.search(r"####\s*([^\n\r]+)", answer_text)
    if not m:
        return None
    return normalize_num_str(m.group(1).strip())


# =========================
# 新增：更通用 / MATH 兼容逻辑
# =========================

def _strip_outer_math_delimiters(s: str) -> str:
    s = s.strip()
    if len(s) >= 2 and s.startswith("$") and s.endswith("$"):
        s = s[1:-1].strip()
    return s


def _normalize_symbolic_str(x: Optional[str]) -> Optional[str]:
    """
    给 MATH 类答案做“轻量规范化”：
    - 不做数值强行抽取
    - 尽量保留表达式语义
    """
    if x is None:
        return None
    s = str(x).strip()
    if not s:
        return None

    s = _strip_outer_math_delimiters(s)
    s = s.strip().rstrip("。．.,;:，；：")
    s = s.replace("\\left", "").replace("\\right", "")
    s = re.sub(r"\s+", "", s)

    return s or None


def _canonicalize_answer(x: Optional[str]) -> Optional[str]:
    """
    通用答案规范化：
    - 若是纯数值答案，沿用 normalize_num_str
    - 若不是纯数值，保留其轻量规范化后的表达式形式
    """
    if x is None:
        return None

    s = str(x).strip()
    if not s:
        return None

    s = _strip_outer_math_delimiters(s)
    s = s.strip().rstrip("。．.,;:，；：")

    # 尝试纯数值判断；只有在“整体就是一个数字”的情况下才走数值归一化
    numeric_candidate = s.replace(",", "").replace("$", "").replace("%", "").strip()
    if re.fullmatch(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", numeric_candidate):
        return normalize_num_str(s)

    return _normalize_symbolic_str(s)


def _find_last_boxed_content(text: str) -> Optional[str]:
    """
    提取最后一个 \\boxed{...} 的内容。
    支持嵌套花括号，例如 \\boxed{\\frac{1}{2}}
    """
    text = text or ""
    starts = list(re.finditer(r"\\boxed\s*\{", text))
    if not starts:
        return None

    start_match = starts[-1]
    i = start_match.end()  # 位于 '{' 后的第一个字符
    depth = 1
    buf = []

    while i < len(text):
        ch = text[i]
        if ch == "{":
            depth += 1
            buf.append(ch)
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return "".join(buf).strip()
            buf.append(ch)
        else:
            buf.append(ch)
        i += 1

    return None


def extract_math_gt_from_answer_field(answer_text: str) -> Optional[str]:
    """
    MATH / MATH500 的 GT 提取：
    1. 优先取最后一个 \\boxed{...}
    2. 再退到 final answer phrase
    3. 最后再退到最后一个数字
    """
    if not answer_text:
        return None

    raw = _find_last_boxed_content(answer_text)
    if raw:
        return _canonicalize_answer(raw)

    pred, _, _ = extract_pred_answer(answer_text)
    return pred


def extract_gt_from_answer_field(answer_text: str, dataset_name: Optional[str]) -> Optional[str]:
    """
    统一 GT 提取入口：
    - gsm8k: 完全保持原逻辑
    - math / math500: 使用 boxed / expression 兼容逻辑
    - 其他：尽量走通用逻辑
    """
    ds = (dataset_name or "").strip().lower()

    if ds == "gsm8k":
        return extract_gsm8k_gt_from_answer_field(answer_text)

    if ds in {"math", "math500"}:
        return extract_math_gt_from_answer_field(answer_text)

    # fallback：尽量保守兼容
    gt = extract_gsm8k_gt_from_answer_field(answer_text)
    if gt is not None:
        return gt
    return extract_math_gt_from_answer_field(answer_text)


def _extract_boxed_answer(text: str) -> Tuple[Optional[str], Optional[str]]:
    raw = _find_last_boxed_content(text or "")
    if not raw:
        return None, None
    return _canonicalize_answer(raw), raw


def _extract_final_phrase_answer(text: str) -> Tuple[Optional[str], Optional[str]]:
    text = text or ""
    for pat in _FINAL_PATTERNS:
        matches = list(pat.finditer(text))
        if not matches:
            continue
        raw = matches[-1].group(1).strip()
        if raw:
            norm = _canonicalize_answer(raw)
            if norm is not None:
                return norm, raw
    return None, None


def _extract_last_number(text: str) -> Tuple[Optional[str], Optional[str]]:
    nums = _NUM_RE.findall(text or "")
    if not nums:
        return None, None
    raw = nums[-1].strip()
    return normalize_num_str(raw), raw


def extract_pred_answer(text: str) -> Tuple[Optional[str], str, Optional[str]]:
    """
    预测答案提取：
    1. boxed
    2. final phrase
    3. last number
    """
    pred, raw = _extract_boxed_answer(text)
    if pred is not None:
        return pred, "boxed", raw

    pred, raw = _extract_final_phrase_answer(text)
    if pred is not None:
        return pred, "final_phrase", raw

    pred, raw = _extract_last_number(text)
    if pred is not None:
        return pred, "last_number", raw

    return None, "none", None


def detect_answer_signal(*, step_text: str = "", full_text: str = "") -> Tuple[bool, str, Optional[str]]:
    """
    终止信号检测逻辑保持原风格：
    - 只把 boxed / final_phrase 当作“强答案信号”
    - 仅出现普通数字不算终止
    """
    for candidate_text, reason_prefix in [(step_text, "step"), (full_text, "full")]:
        pred, answer_type, _ = extract_pred_answer(candidate_text)
        if pred is None:
            continue
        if answer_type == "boxed":
            return True, f"{reason_prefix}_boxed", pred
        if answer_type == "final_phrase":
            return True, f"{reason_prefix}_final_phrase", pred
    return False, "", None


def evaluate_against_gt(full_text: str, gt_answer: Optional[str]) -> Dict[str, Any]:
    """
    通用离线评估：
    - pred 用新的通用提取
    - gt 不再强制只按数字处理
    - 因此可以兼容 GSM8K 和 MATH
    """
    pred, answer_type, matched = extract_pred_answer(full_text)
    gt = _canonicalize_answer(gt_answer)
    correct = None if gt is None else (pred == gt)
    return {
        "pred": pred,
        "gt": gt,
        "correct_bool": correct,
        "answer_type": answer_type,
        "answer_matched_text": matched,
    }