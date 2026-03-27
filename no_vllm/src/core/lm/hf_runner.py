from __future__ import annotations

import time
from typing import Optional, List, Any, Dict

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from .results import LMResult


def _get_gpu_mem_snapshot() -> Dict[str, Optional[float]]:
    try:
        if not torch.cuda.is_available():
            return {
                "gpu_mem_free_mb": None,
                "gpu_mem_used_mb": None,
                "gpu_mem_total_mb": None,
                "gpu_mem_used_ratio": None,
            }

        free_bytes, total_bytes = torch.cuda.mem_get_info()
        used_bytes = total_bytes - free_bytes
        return {
            "gpu_mem_free_mb": float(free_bytes / 1024.0 / 1024.0),
            "gpu_mem_used_mb": float(used_bytes / 1024.0 / 1024.0),
            "gpu_mem_total_mb": float(total_bytes / 1024.0 / 1024.0),
            "gpu_mem_used_ratio": float(used_bytes / total_bytes) if total_bytes > 0 else None,
        }
    except Exception:
        return {
            "gpu_mem_free_mb": None,
            "gpu_mem_used_mb": None,
            "gpu_mem_total_mb": None,
            "gpu_mem_used_ratio": None,
        }


def _merge_stop_sequences(user_stop: Optional[List[str]]) -> Optional[List[str]]:
    default_stop = [
        "<STEP_END>",
        "\n\n",
        "\nStep ",
        "\nNext step:",
        "\nProblem:",
        "\nCurrent reasoning steps:",
    ]

    merged: List[str] = []
    for s in (user_stop or []) + default_stop:
        if s and s not in merged:
            merged.append(s)
    return merged or None


def _resolve_dtype(dtype: Optional[str]):
    if dtype is None:
        return None
    if not isinstance(dtype, str):
        return dtype
    dtype = dtype.strip().lower()
    if dtype in {"", "auto"}:
        return "auto"
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if dtype not in mapping:
        raise ValueError(f"Unsupported torch dtype: {dtype}")
    return mapping[dtype]


class HFRunner:
    """
    Batch-first Hugging Face runner that mirrors the old LMResult contract.
    """

    def __init__(
        self,
        model_ckpt: str,
        seed: int = 0,
        max_batch_size: int = 16,
        dtype: str = "auto",
        device_map: str = "auto",
        attn_implementation: Optional[str] = None,
        max_model_len: int = 8192,
        logprobs: int = 1,
        enable_metrics: bool = False,
    ):
        if max_batch_size <= 0:
            raise ValueError("max_batch_size must be >= 1")
        if max_model_len <= 0:
            raise ValueError("max_model_len must be >= 1")

        self.model_ckpt = model_ckpt
        self.seed = int(seed)
        self.max_batch_size = int(max_batch_size)
        self.max_model_len = int(max_model_len)
        self.logprobs = int(logprobs) if logprobs is not None else 0
        self.enable_metrics = bool(enable_metrics)
        self.device_map = device_map
        self.attn_implementation = attn_implementation
        self.torch_dtype = _resolve_dtype(dtype)
        self._last_prompt_token_ids: Optional[List[int]] = None
        self._num_generate_calls = 0
        self._total_prompt_tokens = 0
        self._total_output_tokens = 0

        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

        self.tokenizer = AutoTokenizer.from_pretrained(model_ckpt, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        model_kwargs: Dict[str, Any] = {
            "trust_remote_code": True,
            "device_map": self.device_map,
        }
        if self.torch_dtype != "auto":
            model_kwargs["torch_dtype"] = self.torch_dtype
        if self.attn_implementation:
            model_kwargs["attn_implementation"] = self.attn_implementation

        self.model = AutoModelForCausalLM.from_pretrained(model_ckpt, **model_kwargs).eval()
        self.input_device = next(self.model.parameters()).device

        print(
            "[HFRunner] "
            f"model_ckpt={model_ckpt}, "
            f"seed={self.seed}, "
            f"max_batch_size={self.max_batch_size}, "
            f"dtype={dtype}, "
            f"device_map={self.device_map}, "
            f"attn_implementation={self.attn_implementation}, "
            f"logprobs={self.logprobs}, "
            f"max_model_len={self.max_model_len}"
        )

    def _collect_engine_metrics_snapshot(self) -> Dict[str, float]:
        if not self.enable_metrics:
            return {}

        snap: Dict[str, float] = {
            "hf:num_generate_calls": float(self._num_generate_calls),
            "hf:total_prompt_tokens": float(self._total_prompt_tokens),
            "hf:total_output_tokens": float(self._total_output_tokens),
        }
        if torch.cuda.is_available():
            try:
                snap["hf:cuda_memory_allocated_mb"] = float(
                    torch.cuda.memory_allocated() / 1024.0 / 1024.0
                )
                snap["hf:cuda_memory_reserved_mb"] = float(
                    torch.cuda.memory_reserved() / 1024.0 / 1024.0
                )
                snap["hf:cuda_max_memory_allocated_mb"] = float(
                    torch.cuda.max_memory_allocated() / 1024.0 / 1024.0
                )
            except Exception:
                pass
        return snap

    @staticmethod
    def _diff_metric_snapshots(before: Dict[str, float], after: Dict[str, float]) -> Dict[str, float]:
        delta: Dict[str, float] = {}
        all_keys = sorted(set(before.keys()) | set(after.keys()))
        for k in all_keys:
            b = before.get(k)
            a = after.get(k)
            if isinstance(b, (int, float)) and isinstance(a, (int, float)):
                delta[k] = float(a - b)
            elif isinstance(a, (int, float)):
                delta[k] = float(a)
        return delta

    def _trim_by_stop_sequences(
        self,
        text: str,
        token_ids: List[int],
        logprobs: Optional[List[Optional[float]]],
        stop_sequences: Optional[List[str]],
    ) -> tuple[str, List[int], Optional[List[Optional[float]]], Optional[str]]:
        if not text:
            return text, token_ids, logprobs, None

        finish_reason = None
        cutoff = None
        matched_stop = None
        for stop in stop_sequences or []:
            idx = text.find(stop)
            if idx >= 0 and (cutoff is None or idx < cutoff):
                cutoff = idx
                matched_stop = stop

        if cutoff is None:
            return text, token_ids, logprobs, finish_reason

        trimmed_text = text[:cutoff]
        trimmed_ids = self.tokenizer.encode(trimmed_text, add_special_tokens=False)
        trimmed_logprobs = logprobs[: len(trimmed_ids)] if logprobs is not None else None
        finish_reason = f"stop:{matched_stop}"
        return trimmed_text, trimmed_ids, trimmed_logprobs, finish_reason

    def _extract_logprobs(
        self,
        scores: List[torch.Tensor],
        output_token_ids: List[int],
        batch_index: int,
    ) -> Optional[List[Optional[float]]]:
        if self.logprobs <= 0 or not scores or not output_token_ids:
            return None

        values: List[Optional[float]] = []
        steps = min(len(scores), len(output_token_ids))
        for step in range(steps):
            logits = scores[step][batch_index]
            token_id = int(output_token_ids[step])
            try:
                logprob = F.log_softmax(logits, dim=-1)[token_id]
                values.append(float(logprob.detach().cpu().item()))
            except Exception:
                values.append(None)
        return values

    def generate_batch(
        self,
        prompt_texts: List[str],
        max_new_tokens: Optional[int] = 64,
        temperature: float = 0.7,
        top_p: float = 0.95,
        **kwargs,
    ) -> List[LMResult]:
        if not prompt_texts:
            return []
        if len(prompt_texts) > self.max_batch_size:
            raise ValueError(
                f"Batch size {len(prompt_texts)} exceeds max_batch_size={self.max_batch_size}"
            )

        stop_sequences = _merge_stop_sequences(kwargs.get("stop"))
        enc = self.tokenizer(
            prompt_texts,
            padding=True,
            truncation=True,
            max_length=self.max_model_len,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(self.input_device)
        attention_mask = enc["attention_mask"].to(self.input_device)
        prompt_lens = attention_mask.sum(dim=1).tolist()
        padded_input_len = int(input_ids.shape[1])

        do_sample = bool(temperature and temperature > 0.0)
        gen_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": max(1, int(max_new_tokens or 64)),
            "do_sample": do_sample,
            "use_cache": True,
            "return_dict_in_generate": True,
            "output_scores": bool(self.logprobs > 0),
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if do_sample:
            gen_kwargs["temperature"] = float(max(temperature, 1e-5))
            gen_kwargs["top_p"] = float(top_p)

        metrics_before = self._collect_engine_metrics_snapshot()
        t0 = time.time()
        with torch.inference_mode():
            outputs = self.model.generate(**gen_kwargs)
        t1 = time.time()

        batch_latency_ms = float((t1 - t0) * 1000.0)
        batch_metrics_raw_before = metrics_before

        sequences = outputs.sequences
        scores = list(outputs.scores) if getattr(outputs, "scores", None) is not None else []

        results: List[LMResult] = []
        batch_size = len(prompt_texts)

        total_prompt_tokens_batch = 0
        total_output_tokens_batch = 0

        for batch_index, prompt_text in enumerate(prompt_texts):
            prompt_len = int(prompt_lens[batch_index])
            prompt_token_ids = (
                input_ids[batch_index][attention_mask[batch_index].bool()].detach().cpu().tolist()
            )
            seq = sequences[batch_index].detach().cpu()
            output_token_ids = seq[padded_input_len:].tolist()

            pad_token_id = self.tokenizer.pad_token_id
            while output_token_ids and pad_token_id is not None and output_token_ids[-1] == pad_token_id:
                output_token_ids.pop()

            output_text = self.tokenizer.decode(output_token_ids, skip_special_tokens=False)
            logprobs = self._extract_logprobs(scores, output_token_ids, batch_index)
            trimmed_text, trimmed_ids, trimmed_logprobs, stop_finish_reason = self._trim_by_stop_sequences(
                output_text,
                output_token_ids,
                logprobs,
                stop_sequences,
            )

            output_text = trimmed_text
            output_token_ids = trimmed_ids
            logprobs = trimmed_logprobs

            finish_reason = stop_finish_reason
            if finish_reason is None:
                if output_token_ids and self.tokenizer.eos_token_id is not None and output_token_ids[-1] == self.tokenizer.eos_token_id:
                    finish_reason = "eos"
                elif len(output_token_ids) >= max(1, int(max_new_tokens or 64)):
                    finish_reason = "length"

            cumulative_logprob = None
            avg_logprob = None
            valid_logprobs = [x for x in (logprobs or []) if x is not None]
            if valid_logprobs:
                cumulative_logprob = float(sum(valid_logprobs))
                avg_logprob = float(cumulative_logprob / len(valid_logprobs))

            output_len = len(output_token_ids)
            total_prompt_tokens_batch += prompt_len
            total_output_tokens_batch += output_len

            tokens_per_sec = float(output_len / (batch_latency_ms / 1000.0)) if batch_latency_ms > 0 else None

            num_cached_tokens = 0
            if self._last_prompt_token_ids:
                max_prefix = min(len(self._last_prompt_token_ids), len(prompt_token_ids))
                while num_cached_tokens < max_prefix and self._last_prompt_token_ids[num_cached_tokens] == prompt_token_ids[num_cached_tokens]:
                    num_cached_tokens += 1
            cached_ratio = float(num_cached_tokens / prompt_len) if prompt_len > 0 else None
            self._last_prompt_token_ids = list(prompt_token_ids)

            gpu_snap = _get_gpu_mem_snapshot()
            results.append(
                LMResult(
                    output_text=output_text,
                    output_tokens_len=output_len,
                    num_cached_tokens=num_cached_tokens,
                    prompt_tokens_len=prompt_len,
                    prompt_token_ids=prompt_token_ids,
                    output_token_ids=output_token_ids,
                    finish_reason=finish_reason,
                    logprobs=logprobs,
                    cumulative_logprob=cumulative_logprob,
                    avg_logprob=avg_logprob,
                    latency_ms=batch_latency_ms,
                    tokens_per_sec=tokens_per_sec,
                    cached_ratio=cached_ratio,
                    gpu_mem_free_mb=gpu_snap.get("gpu_mem_free_mb"),
                    gpu_mem_used_mb=gpu_snap.get("gpu_mem_used_mb"),
                    gpu_mem_total_mb=gpu_snap.get("gpu_mem_total_mb"),
                    gpu_mem_used_ratio=gpu_snap.get("gpu_mem_used_ratio"),
                    batch_index=batch_index,
                    batch_size=batch_size,
                    batch_latency_ms=batch_latency_ms,
                )
            )

        self._num_generate_calls += 1
        self._total_prompt_tokens += total_prompt_tokens_batch
        self._total_output_tokens += total_output_tokens_batch

        batch_metrics_raw_after = self._collect_engine_metrics_snapshot()
        batch_metrics_raw = {
            "before": batch_metrics_raw_before,
            "after": batch_metrics_raw_after,
            "batch_size": batch_size,
            "batch_latency_ms": batch_latency_ms,
            "batch_prompt_tokens": total_prompt_tokens_batch,
            "batch_output_tokens": total_output_tokens_batch,
            "engine": "hf",
        }
        batch_metrics_delta = self._diff_metric_snapshots(
            batch_metrics_raw_before,
            batch_metrics_raw_after,
        )
        batch_metrics_delta.update(
            {
                "hf:batch_prompt_tokens": float(total_prompt_tokens_batch),
                "hf:batch_output_tokens": float(total_output_tokens_batch),
                "hf:batch_size": float(batch_size),
            }
        )

        for res in results:
            res.engine_metrics_raw = batch_metrics_raw
            res.engine_metrics_delta = batch_metrics_delta

        return results

    def generate(
        self,
        prompt_text: str,
        max_new_tokens: Optional[int] = 64,
        temperature: float = 0.7,
        top_p: float = 0.95,
        **kwargs,
    ) -> LMResult:
        return self.generate_batch(
            [prompt_text],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            **kwargs,
        )[0]
