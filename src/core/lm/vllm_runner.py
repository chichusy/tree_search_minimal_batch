from __future__ import annotations

import time
from typing import Optional, List, Any, Dict

from .results import LMResult

WATCH_METRICS = {
    "vllm:num_preemptions",
    "vllm:prefix_cache_hits",
    "vllm:prefix_cache_queries",
    "vllm:prompt_tokens_cached",
    "vllm:prompt_tokens_recomputed",
    "vllm:kv_cache_usage_perc",
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
}


def _get_gpu_mem_snapshot() -> Dict[str, Optional[float]]:
    try:
        import torch

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
        if not s:
            continue
        if s not in merged:
            merged.append(s)
    return merged or None


def _parse_kv_cache_memory_bytes(x: Optional[Any]) -> Optional[int]:
    if x is None:
        return None
    if isinstance(x, int):
        return x

    s = str(x).strip()
    if s == "":
        return None

    s_low = s.lower()

    units = {
        "ki": 1024,
        "mi": 1024**2,
        "gi": 1024**3,
        "ti": 1024**4,
        "k": 1024,
        "m": 1024**2,
        "g": 1024**3,
        "t": 1024**4,
    }

    for u in ["ki", "mi", "gi", "ti", "k", "m", "g", "t"]:
        if s_low.endswith(u):
            num = float(s_low[:-len(u)].strip())
            return int(num * units[u])

    return int(s_low)


class VLLMRunner:
    """
    Batch-first vLLM runner.
    主接口是 generate_batch()，单条 generate() 只是一个薄包装。
    """

    def __init__(
        self,
        model_ckpt: str,
        tensor_parallel_size: int = 1,
        seed: int = 0,
        max_num_seqs: int = 16,
        max_num_batched_tokens: Optional[int] = None,
        num_gpu_blocks_override: Optional[int] = None,
        dtype: str = "auto",
        gpu_memory_utilization: float = 0.25,
        enable_prefix_caching: bool = True,
        logprobs: int = 1,
        max_model_len: int = 8192,
        enable_metrics: bool = False,
        kv_cache_metrics: bool = False,
        kv_cache_metrics_sample: float = 1.0,
        kv_cache_memory_bytes: Optional[Any] = None,
    ):
        from vllm import LLM  # lazy import
        from transformers import AutoTokenizer

        if max_num_seqs <= 0:
            raise ValueError("max_num_seqs must be >= 1")
        if max_num_batched_tokens is not None and max_num_batched_tokens <= 0:
            raise ValueError("max_num_batched_tokens must be >= 1 when specified")
        if num_gpu_blocks_override is not None and num_gpu_blocks_override <= 0:
            raise ValueError("num_gpu_blocks_override must be >= 1 when specified")
        if not (0.0 < gpu_memory_utilization <= 1.0):
            raise ValueError("gpu_memory_utilization must be in (0, 1].")
        if max_model_len <= 0:
            raise ValueError("max_model_len must be >= 1")

        self.model_ckpt = model_ckpt
        self.logprobs = int(logprobs) if logprobs is not None else 0
        self.seed = int(seed)
        self.max_num_seqs = int(max_num_seqs)
        self.max_num_batched_tokens = (
            int(max_num_batched_tokens) if max_num_batched_tokens is not None else None
        )
        self.num_gpu_blocks_override = (
            int(num_gpu_blocks_override) if num_gpu_blocks_override is not None else None
        )
        self.dtype = dtype
        self.gpu_memory_utilization = float(gpu_memory_utilization)
        self.enable_prefix_caching = bool(enable_prefix_caching)
        self.max_model_len = int(max_model_len)
        self.kv_cache_memory_bytes = _parse_kv_cache_memory_bytes(kv_cache_memory_bytes)

        self.enable_metrics = bool(enable_metrics)
        self.kv_cache_metrics = bool(kv_cache_metrics)
        self.kv_cache_metrics_sample = float(kv_cache_metrics_sample)

        if not (0.0 < self.kv_cache_metrics_sample <= 1.0):
            raise ValueError("kv_cache_metrics_sample must be in (0, 1].")

        self.tokenizer = AutoTokenizer.from_pretrained(model_ckpt, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(
            "[VLLMRunner] "
            f"model_ckpt={model_ckpt}, "
            f"tp={tensor_parallel_size}, "
            f"seed={self.seed}, "
            f"max_num_seqs={self.max_num_seqs}, "
            f"max_num_batched_tokens={self.max_num_batched_tokens}, "
            f"num_gpu_blocks_override={self.num_gpu_blocks_override}, "
            f"dtype={self.dtype}, "
            f"gpu_memory_utilization={self.gpu_memory_utilization}, "
            f"kv_cache_memory_bytes={self.kv_cache_memory_bytes}, "
            f"enable_prefix_caching={self.enable_prefix_caching}, "
            f"logprobs={self.logprobs}, "
            f"max_model_len={self.max_model_len}"
        )

        llm_kwargs = dict(
            model=model_ckpt,
            trust_remote_code=True,
            tensor_parallel_size=tensor_parallel_size,
            seed=self.seed,
            dtype=self.dtype,
            max_num_seqs=self.max_num_seqs,
            gpu_memory_utilization=self.gpu_memory_utilization,
            enable_prefix_caching=self.enable_prefix_caching,
            max_model_len=self.max_model_len,
        )

        if self.max_num_batched_tokens is not None:
            llm_kwargs["max_num_batched_tokens"] = self.max_num_batched_tokens
        if self.num_gpu_blocks_override is not None:
            llm_kwargs["num_gpu_blocks_override"] = self.num_gpu_blocks_override
        if self.kv_cache_memory_bytes is not None:
            llm_kwargs["kv_cache_memory_bytes"] = self.kv_cache_memory_bytes

        if self.enable_metrics:
            llm_kwargs["disable_log_stats"] = False
            if self.kv_cache_metrics:
                llm_kwargs["kv_cache_metrics"] = True
                llm_kwargs["kv_cache_metrics_sample"] = self.kv_cache_metrics_sample
        else:
            llm_kwargs["disable_log_stats"] = True

        self.llm = LLM(**llm_kwargs)

    def _metric_obj_to_scalar(self, metric: Any):
        if hasattr(metric, "value"):
            value = getattr(metric, "value")
            if isinstance(value, (int, float)):
                return float(value)
        return None

    def _collect_vllm_metrics_snapshot(self) -> Dict[str, float]:
        if not self.enable_metrics:
            return {}

        try:
            metrics = self.llm.get_metrics()
        except Exception as e:
            print(f"[VLLMRunner] get_metrics() failed: {e}")
            return {}

        snap: Dict[str, float] = {}
        for metric in metrics:
            name = getattr(metric, "name", None)
            if not name or name not in WATCH_METRICS:
                continue
            value = self._metric_obj_to_scalar(metric)
            if value is not None:
                snap[name] = value
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

    def _extract_prompt_token_ids(self, req0, prompt_text: str) -> List[int]:
        prompt_ids = getattr(req0, "prompt_token_ids", None)
        if prompt_ids is not None:
            try:
                return list(prompt_ids)
            except Exception:
                pass
        return list(self.tokenizer.encode(prompt_text)) if self.tokenizer is not None else []

    def _extract_token_logprobs(
        self,
        raw_logprobs: Any,
        output_token_ids: List[int],
    ) -> Optional[List[Optional[float]]]:
        if raw_logprobs is None:
            return None

        try:
            raw_items = list(raw_logprobs)
        except Exception:
            return None

        if not raw_items:
            return []

        normalized: List[Optional[float]] = []

        for i, item in enumerate(raw_items):
            chosen_id = output_token_ids[i] if i < len(output_token_ids) else None
            val: Optional[float] = None

            try:
                if isinstance(item, dict):
                    candidate = None
                    if chosen_id is not None:
                        if chosen_id in item:
                            candidate = item[chosen_id]
                        elif str(chosen_id) in item:
                            candidate = item[str(chosen_id)]

                    if candidate is not None:
                        if hasattr(candidate, "logprob"):
                            val = float(candidate.logprob)
                        elif isinstance(candidate, (int, float)):
                            val = float(candidate)
                        elif isinstance(candidate, dict) and "logprob" in candidate:
                            val = float(candidate["logprob"])
                    else:
                        val = None

                elif hasattr(item, "logprob"):
                    val = float(item.logprob)
                elif isinstance(item, (int, float)):
                    val = float(item)
                elif isinstance(item, dict) and "logprob" in item:
                    val = float(item["logprob"])
                else:
                    val = None
            except Exception:
                val = None

            normalized.append(val)

        return normalized

    def _build_sampling_params(
        self,
        *,
        max_new_tokens: Optional[int],
        temperature: float,
        top_p: float,
        stop: Optional[List[str]],
    ):
        from vllm import SamplingParams

        return SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max(1, int(max_new_tokens or 64)),
            stop=stop,
            logprobs=self.logprobs if self.logprobs and self.logprobs > 0 else None,
        )

    def _build_lm_result_from_request(
        self,
        *,
        req0,
        out0,
        prompt_text: str,
        latency_ms: float,
        batch_index: int,
        batch_size: int,
        batch_metrics_raw: Optional[Dict[str, Any]],
        batch_metrics_delta: Optional[Dict[str, Any]],
    ) -> LMResult:
        prompt_token_ids = self._extract_prompt_token_ids(req0, prompt_text)
        prompt_len = len(prompt_token_ids)

        num_cached = getattr(req0, "num_cached_tokens", 0)
        num_cached = int(num_cached) if num_cached is not None else 0

        out_token_ids = list(getattr(out0, "token_ids", []) or [])
        output_len = len(out_token_ids)

        finish_reason = getattr(out0, "finish_reason", None)

        cumulative_logprob = getattr(out0, "cumulative_logprob", None)
        if cumulative_logprob is not None:
            try:
                cumulative_logprob = float(cumulative_logprob)
            except Exception:
                cumulative_logprob = None

        avg_logprob = None
        if cumulative_logprob is not None and output_len > 0:
            avg_logprob = float(cumulative_logprob / output_len)

        raw_logprobs = getattr(out0, "logprobs", None)
        norm_logprobs = self._extract_token_logprobs(raw_logprobs, out_token_ids)

        tokens_per_sec = None
        if latency_ms > 0:
            tokens_per_sec = float(output_len / (latency_ms / 1000.0))

        cached_ratio = None
        if prompt_len > 0:
            cached_ratio = float(num_cached / prompt_len)

        gpu_snap = _get_gpu_mem_snapshot()

        return LMResult(
            output_text=out0.text,
            output_tokens_len=output_len,
            num_cached_tokens=num_cached,
            prompt_tokens_len=prompt_len,
            prompt_token_ids=prompt_token_ids,
            output_token_ids=out_token_ids,
            finish_reason=finish_reason,
            logprobs=norm_logprobs,
            cumulative_logprob=cumulative_logprob,
            avg_logprob=avg_logprob,
            latency_ms=latency_ms,
            tokens_per_sec=tokens_per_sec,
            cached_ratio=cached_ratio,
            gpu_mem_free_mb=gpu_snap.get("gpu_mem_free_mb"),
            gpu_mem_used_mb=gpu_snap.get("gpu_mem_used_mb"),
            gpu_mem_total_mb=gpu_snap.get("gpu_mem_total_mb"),
            gpu_mem_used_ratio=gpu_snap.get("gpu_mem_used_ratio"),
            batch_index=batch_index,
            batch_size=batch_size,
            batch_latency_ms=latency_ms,
            vllm_metrics_raw=batch_metrics_raw,
            vllm_metrics_delta=batch_metrics_delta,
        )

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

        user_stop = kwargs.get("stop", None)
        stop = _merge_stop_sequences(user_stop)
        sp = self._build_sampling_params(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
        )

        metrics_before = self._collect_vllm_metrics_snapshot()

        t0 = time.time()
        outputs = self.llm.generate(prompt_texts, sp, use_tqdm=False)
        t1 = time.time()

        metrics_after = self._collect_vllm_metrics_snapshot()
        metrics_delta = self._diff_metric_snapshots(metrics_before, metrics_after)
        latency_ms = float((t1 - t0) * 1000.0)

        batch_metrics_raw = {
            "before": metrics_before,
            "after": metrics_after,
        }

        results: List[LMResult] = []
        batch_size = len(outputs)

        for batch_index, (prompt_text, req0) in enumerate(zip(prompt_texts, outputs)):
            out0 = req0.outputs[0]
            results.append(
                self._build_lm_result_from_request(
                    req0=req0,
                    out0=out0,
                    prompt_text=prompt_text,
                    latency_ms=latency_ms,
                    batch_index=batch_index,
                    batch_size=batch_size,
                    batch_metrics_raw=batch_metrics_raw,
                    batch_metrics_delta=metrics_delta,
                )
            )

        return results

    def generate(
        self,
        prompt_text: str,
        max_new_tokens: Optional[int] = 64,
        temperature: float = 0.7,
        top_p: float = 0.95,
        **kwargs,
    ) -> LMResult:
        results = self.generate_batch(
            [prompt_text],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            **kwargs,
        )
        return results[0]