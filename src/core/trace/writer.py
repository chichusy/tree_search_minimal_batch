from __future__ import annotations

import hashlib
import os
from typing import Optional, List, Dict

from src.core.trace.jsonl_writer import JsonlWriter
from src.core.trace.schema import (
    CallRecord,
    SequenceRecord,
    LogicalBlock,
    BlockTraceRecord,
)


class TraceLogger:
    """
    对外仍然是一个 logger；
    对内 fan-out 到:
      - trace_calls.jsonl
      - trace_sequences.jsonl
      - trace_blocks.jsonl

    NEW:
      - 支持 sample_context
      - 每条 trace 在落盘前会自动把 sample_context merge 到 meta 里
    """

    def __init__(
        self,
        out_path: str,
        mode: str = "w",
        enable_sequence_trace: bool = True,
        enable_block_trace: bool = True,
        block_size: int = 16,
    ):
        sample_dir = os.path.dirname(out_path)

        self.call_writer = JsonlWriter(out_path, mode=mode)
        self.sequence_writer = (
            JsonlWriter(os.path.join(sample_dir, "trace_sequences.jsonl"), mode=mode)
            if enable_sequence_trace
            else None
        )
        self.block_writer = (
            JsonlWriter(os.path.join(sample_dir, "trace_blocks.jsonl"), mode=mode)
            if enable_block_trace
            else None
        )

        self.enable_sequence_trace = enable_sequence_trace
        self.enable_block_trace = enable_block_trace
        self.block_size = int(block_size)

        self._call_id = 0

        # last prompt input ids for time locality
        self._last_input_ids: Optional[List[int]] = None

        # node_id -> full token ids of that node sequence
        self._node_full_token_ids: Dict[int, List[int]] = {}

        # NEW: per-sample metadata context, e.g.
        # {"sample_id": 17, "difficulty": "medium"}
        self.sample_context: Dict[str, object] = {}

    # -----------------------------
    # NEW: sample context support
    # -----------------------------
    def set_sample_context(self, sample_context: Optional[dict]) -> None:
        if sample_context is None:
            self.sample_context = {}
        else:
            self.sample_context = dict(sample_context)

    def _merge_meta(self, meta: Optional[dict]) -> Optional[dict]:
        """
        Merge order:
          sample_context -> meta
        so trace-specific meta can override sample_context keys if needed.
        """
        if not self.sample_context and not meta:
            return None

        merged = {}
        if self.sample_context:
            merged.update(self.sample_context)
        if meta:
            merged.update(meta)
        return merged

    @property
    def next_call_id(self) -> int:
        self._call_id += 1
        return self._call_id

    def close(self) -> None:
        self.call_writer.close()
        if self.sequence_writer is not None:
            self.sequence_writer.close()
        if self.block_writer is not None:
            self.block_writer.close()

    def log_call(self, rec: CallRecord) -> None:
        rec.meta = self._merge_meta(rec.meta)
        self.call_writer.write(rec)

    def log_generation_bundle(
        self,
        call_rec: CallRecord,
        seq_rec: Optional[SequenceRecord] = None,
        block_rec: Optional[BlockTraceRecord] = None,
    ) -> None:
        call_rec.meta = self._merge_meta(call_rec.meta)

        if seq_rec is not None:
            seq_rec.meta = self._merge_meta(seq_rec.meta)

        if block_rec is not None:
            block_rec.meta = self._merge_meta(block_rec.meta)

        self.call_writer.write(call_rec)
        if self.sequence_writer is not None and seq_rec is not None:
            self.sequence_writer.write(seq_rec)
        if self.block_writer is not None and block_rec is not None:
            self.block_writer.write(block_rec)

    def register_node_sequence(self, node_id: int, full_token_ids: Optional[List[int]]) -> None:
        if full_token_ids is None:
            return
        self._node_full_token_ids[node_id] = list(full_token_ids)

    def get_node_sequence(self, node_id: Optional[int]) -> Optional[List[int]]:
        if node_id is None:
            return None
        ids = self._node_full_token_ids.get(node_id)
        return list(ids) if ids is not None else None

    @staticmethod
    def _digest_token_ids(token_ids: List[int]) -> str:
        if not token_ids:
            return ""
        b = ",".join(map(str, token_ids)).encode("utf-8")
        return hashlib.sha1(b).hexdigest()

    @staticmethod
    def _lcp_len(a: Optional[List[int]], b: Optional[List[int]]) -> int:
        if not a or not b:
            return 0
        m = min(len(a), len(b))
        i = 0
        while i < m and a[i] == b[i]:
            i += 1
        return i

    def compute_locality(
        self,
        *,
        node_id: int,
        parent_node_id: Optional[int],
        input_ids: Optional[List[int]],
        remember_for_node: bool = True,
    ) -> dict:
        """
        兼容现有 call-level trace 的 locality 逻辑。
        注意这里的 input_ids 是“本次调用的 prompt ids”。
        """
        if not input_ids:
            self._last_input_ids = input_ids
            if remember_for_node:
                self._node_full_token_ids[node_id] = input_ids or []
            return {
                "input_len": 0,
                "input_ids_digest": "",
                "lcp_last": 0,
                "reuse_last": 0,
                "miss_last": 0,
                "lcp_parent": 0,
                "reuse_parent": 0,
                "miss_parent": 0,
            }

        digest = self._digest_token_ids(input_ids)

        lcp_last = self._lcp_len(self._last_input_ids, input_ids)
        reuse_last = lcp_last
        miss_last = len(input_ids) - reuse_last

        parent_ids = self._node_full_token_ids.get(parent_node_id) if parent_node_id is not None else None
        lcp_parent = self._lcp_len(parent_ids, input_ids)
        reuse_parent = lcp_parent
        miss_parent = len(input_ids) - reuse_parent

        self._last_input_ids = list(input_ids)
        if remember_for_node:
            self._node_full_token_ids[node_id] = list(input_ids)

        return {
            "input_len": len(input_ids),
            "input_ids_digest": digest,
            "lcp_last": int(lcp_last),
            "reuse_last": int(reuse_last),
            "miss_last": int(miss_last),
            "lcp_parent": int(lcp_parent),
            "reuse_parent": int(reuse_parent),
            "miss_parent": int(miss_parent),
        }

    def build_sequence_record(
        self,
        *,
        call_id: int,
        rollout_id: int,
        node_id: int,
        parent_node_id: Optional[int],
        purpose: str,
        prompt_token_ids: Optional[List[int]],
        output_token_ids: Optional[List[int]],
        full_token_ids: Optional[List[int]],
        finish_reason: Optional[str] = None,
        cumulative_logprob: Optional[float] = None,
        avg_logprob: Optional[float] = None,
        latency_ms: Optional[float] = None,
        tokens_per_sec: Optional[float] = None,
        cached_ratio: Optional[float] = None,
        logprobs: Optional[List[Optional[float]]] = None,
        meta: Optional[dict] = None,
    ) -> Optional[SequenceRecord]:
        if not self.enable_sequence_trace:
            return None
        if prompt_token_ids is None or full_token_ids is None:
            return None

        prompt_ids = list(prompt_token_ids)
        output_ids = list(output_token_ids or [])
        full_ids = list(full_token_ids)

        parent_full = self.get_node_sequence(node_id)
        # 注意：这里 node_id 是“本次调用发生在哪个 node 上”
        # 当前调用创建的是它的 child，因此 delta_vs_parent 应该对齐当前 node 的 full ids
        lcp_full_parent = self._lcp_len(parent_full, full_ids)
        delta_ids = full_ids[lcp_full_parent:]

        return SequenceRecord(
            call_id=call_id,
            rollout_id=rollout_id,
            node_id=node_id,
            parent_node_id=parent_node_id,
            purpose=purpose,
            prompt_token_ids=prompt_ids,
            output_token_ids=output_ids,
            full_token_ids=full_ids,
            delta_token_ids_vs_parent=delta_ids,
            prompt_len=len(prompt_ids),
            output_len=len(output_ids),
            full_len=len(full_ids),
            delta_len=len(delta_ids),
            prompt_ids_digest=self._digest_token_ids(prompt_ids),
            output_ids_digest=self._digest_token_ids(output_ids),
            full_ids_digest=self._digest_token_ids(full_ids),
            delta_ids_digest=self._digest_token_ids(delta_ids),
            lcp_full_parent=int(lcp_full_parent),
            finish_reason=finish_reason,
            cumulative_logprob=cumulative_logprob,
            avg_logprob=avg_logprob,
            latency_ms=latency_ms,
            tokens_per_sec=tokens_per_sec,
            cached_ratio=cached_ratio,
            logprobs=list(logprobs) if logprobs is not None else None,
            meta=self._merge_meta(meta),
        )

    def _block_id(self, token_ids: List[int]) -> str:
        return self._digest_token_ids(token_ids)

    def _token_ids_to_blocks(self, token_ids: List[int]) -> List[LogicalBlock]:
        if not token_ids:
            return []

        blocks: List[LogicalBlock] = []
        bs = self.block_size
        for start in range(0, len(token_ids), bs):
            end = min(start + bs, len(token_ids))
            chunk = token_ids[start:end]
            blocks.append(
                LogicalBlock(
                    block_index=start // bs,
                    start_token_idx=start,
                    end_token_idx=end,
                    num_tokens=len(chunk),
                    block_id=self._block_id(chunk),
                )
            )
        return blocks

    def build_block_record(self, seq_rec: Optional[SequenceRecord]) -> Optional[BlockTraceRecord]:
        if not self.enable_block_trace or seq_rec is None:
            return None

        return BlockTraceRecord(
            call_id=seq_rec.call_id,
            rollout_id=seq_rec.rollout_id,
            node_id=seq_rec.node_id,
            parent_node_id=seq_rec.parent_node_id,
            purpose=seq_rec.purpose,
            block_size=self.block_size,
            prompt_blocks=self._token_ids_to_blocks(seq_rec.prompt_token_ids),
            output_blocks=self._token_ids_to_blocks(seq_rec.output_token_ids),
            full_blocks=self._token_ids_to_blocks(seq_rec.full_token_ids),
            delta_blocks_vs_parent=self._token_ids_to_blocks(seq_rec.delta_token_ids_vs_parent),
            lcp_full_parent=seq_rec.lcp_full_parent,
            meta=self._merge_meta(seq_rec.meta),
        )