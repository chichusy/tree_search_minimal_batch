from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Any, Dict, Optional, List


@dataclass
class CallRecord:
    # identity / structure
    call_id: int
    rollout_id: int
    node_id: int
    parent_node_id: Optional[int]
    purpose: str  # "expand" | "simulate" | "select" | "reward_model" | "rollout_summary" | "final"

    # token-level prompt/decoding sizes
    input_len: int = 0
    output_len: int = 0
    prompt_len: Optional[int] = None

    # lightweight identity for prompt tokens
    input_ids_digest: str = ""

    # locality vs LAST call (time locality)
    lcp_last: int = 0
    reuse_last: int = 0
    miss_last: int = 0

    # locality vs PARENT node's prompt (tree locality)
    lcp_parent: int = 0
    reuse_parent: int = 0
    miss_parent: int = 0

    # Optional timing
    t_start_ms: Optional[int] = None
    t_end_ms: Optional[int] = None
    latency_ms: Optional[float] = None
    tokens_per_sec: Optional[float] = None

    # generation / finish info
    finish_reason: Optional[str] = None
    cumulative_logprob: Optional[float] = None
    avg_logprob: Optional[float] = None
    cached_ratio: Optional[float] = None

    # Extra info (keep flexible)
    meta: Optional[Dict[str, Any]] = None

    # Optional: save raw texts for debugging
    prompt_text: Optional[str] = None
    output_text: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SequenceRecord:
    call_id: int
    rollout_id: int
    node_id: int
    parent_node_id: Optional[int]
    purpose: str

    prompt_token_ids: List[int] = field(default_factory=list)
    output_token_ids: List[int] = field(default_factory=list)
    full_token_ids: List[int] = field(default_factory=list)
    delta_token_ids_vs_parent: List[int] = field(default_factory=list)

    prompt_len: int = 0
    output_len: int = 0
    full_len: int = 0
    delta_len: int = 0

    prompt_ids_digest: str = ""
    output_ids_digest: str = ""
    full_ids_digest: str = ""
    delta_ids_digest: str = ""

    lcp_full_parent: int = 0

    # 新增：生成级统计
    finish_reason: Optional[str] = None
    cumulative_logprob: Optional[float] = None
    avg_logprob: Optional[float] = None
    latency_ms: Optional[float] = None
    tokens_per_sec: Optional[float] = None
    cached_ratio: Optional[float] = None
    logprobs: Optional[List[Optional[float]]] = None

    meta: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class LogicalBlock:
    block_index: int
    start_token_idx: int
    end_token_idx: int   # exclusive
    num_tokens: int
    block_id: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BlockTraceRecord:
    call_id: int
    rollout_id: int
    node_id: int
    parent_node_id: Optional[int]
    purpose: str
    block_size: int

    prompt_blocks: List[LogicalBlock] = field(default_factory=list)
    output_blocks: List[LogicalBlock] = field(default_factory=list)
    full_blocks: List[LogicalBlock] = field(default_factory=list)
    delta_blocks_vs_parent: List[LogicalBlock] = field(default_factory=list)

    lcp_full_parent: int = 0

    meta: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)