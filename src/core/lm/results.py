from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, Dict, Any


@dataclass
class LMResult:
    output_text: str
    output_tokens_len: int

    # request-level cache stats
    num_cached_tokens: int = 0
    prompt_tokens_len: Optional[int] = None

    # exact token ids
    prompt_token_ids: Optional[List[int]] = None
    output_token_ids: Optional[List[int]] = None

    # generation termination / confidence
    finish_reason: Optional[str] = None
    logprobs: Optional[List[Optional[float]]] = None
    cumulative_logprob: Optional[float] = None
    avg_logprob: Optional[float] = None

    # timing / throughput
    latency_ms: Optional[float] = None
    tokens_per_sec: Optional[float] = None

    # derived cache ratio
    cached_ratio: Optional[float] = None

    # gpu mem snapshot
    gpu_mem_free_mb: Optional[float] = None
    gpu_mem_used_mb: Optional[float] = None
    gpu_mem_total_mb: Optional[float] = None
    gpu_mem_used_ratio: Optional[float] = None

    # batch info
    batch_index: Optional[int] = None
    batch_size: Optional[int] = None
    batch_latency_ms: Optional[float] = None

    # official vLLM metrics (batch-level snapshot/delta)
    vllm_metrics_raw: Optional[Dict[str, Any]] = None
    vllm_metrics_delta: Optional[Dict[str, Any]] = None