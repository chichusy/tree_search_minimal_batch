from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Dict, Optional, List, Any, Sequence

from src.search.node_minimal import Node
from src.core.eval_minimal import evaluate_against_gt, detect_answer_signal
from src.core.trace.schema import CallRecord
from src.core.trace.writer import TraceLogger

_STEP_PREFIX_RE = re.compile(r"^\s*Step\s*\d+\s*:\s*", re.IGNORECASE)


def _pick_search_value(node: Node, mode: str) -> float:
    mode = (mode or "last").lower()
    if mode == "min":
        return float(node.prm_min_score)
    if mode == "mean":
        return float(node.prm_mean_score)
    if mode == "prod":
        return float(node.prm_prod_score)
    return float(node.prm_last_score)


@dataclass
class _CandidateState:
    step_text: str
    steps: List[str]
    text: str
    depth: int
    total_generated_tokens: int
    finish_reason: Optional[str]
    prompt_text: str
    prompt_token_ids: Optional[List[int]]
    prompt_len: int
    output_tokens_len: int
    output_text_raw: str
    lm_res: Any
    prm_result: Any
    search_value: float
    is_terminal: bool
    terminal_reason: Optional[str]
    is_answer_candidate: bool
    pred_answer: Optional[str]
    batch_index: int
    batch_size: int


@dataclass
class _CandidateBatchResult:
    prompt_text: str
    prompt_token_ids: Optional[List[int]]
    prompt_len: int
    requested_count: int
    returned_count: int
    valid_count: int
    empty_count: int
    batch_latency_ms: Optional[float]
    batch_metrics_raw: Optional[Dict[str, Any]]
    batch_metrics_delta: Optional[Dict[str, Any]]
    candidates: List[_CandidateState]


class MCTSMinimal:
    def __init__(
        self,
        lm_runner,
        prm_scorer,
        trace_logger: TraceLogger,
        tokenizer=None,
        max_depth: int = 6,
        branch_factor: int = 3,
        simulation_branch_factor: int = 3,
        num_rollouts: int = 10,
        max_new_tokens_per_step: int = 96,
        max_total_new_tokens: int = 256,
        exploration_weight: float = 1.4,
        unvisited_bonus: float = 1e9,
        discount: float = 1.0,
        temperature: float = 0.7,
        top_p: float = 0.95,
        stop_sequences: Optional[List[str]] = None,
        prm_value_mode: str = "last",
        final_select_mode: str = "min",
        save_prompt_text: bool = False,
        save_output_text: bool = False,
        save_prompt_max_chars: int = 0,
        save_output_max_chars: int = 0,
    ):
        self.lm = lm_runner
        self.prm = prm_scorer
        self.logger = trace_logger
        self.tokenizer = tokenizer

        self.max_depth = max_depth
        self.branch_factor = branch_factor
        self.simulation_branch_factor = simulation_branch_factor
        self.num_rollouts = num_rollouts
        self.max_new_tokens_per_step = max_new_tokens_per_step
        self.max_total_new_tokens = max_total_new_tokens

        self.exploration_weight = exploration_weight
        self.unvisited_bonus = float(unvisited_bonus)
        self.discount = discount
        self.temperature = temperature
        self.top_p = top_p
        self.stop_sequences = stop_sequences or ["<STEP_END>"]
        self.prm_value_mode = prm_value_mode
        self.final_select_mode = final_select_mode

        self.save_prompt_text = save_prompt_text
        self.save_output_text = save_output_text
        self.save_prompt_max_chars = save_prompt_max_chars
        self.save_output_max_chars = save_output_max_chars

        self.nodes: Dict[int, Node] = {}
        self.next_node_id = 0

        self.root_question: str = ""
        self.gt_answer: Optional[str] = None
        self.root_id: Optional[int] = None

        self.rollout_summaries: List[Dict[str, Any]] = []
        self.final_summary: Dict[str, Any] = {}

    def _next_call_id(self) -> int:
        return self.logger.next_call_id

    def _tokenize_text(self, text: str, add_special_tokens: bool = True) -> Optional[List[int]]:
        if self.tokenizer is None:
            return None
        try:
            return list(self.tokenizer.encode(text, add_special_tokens=add_special_tokens))
        except TypeError:
            try:
                return list(self.tokenizer.encode(text))
            except Exception:
                return None
        except Exception:
            return None

    def _truncate_prompt_text(self, text: str) -> Optional[str]:
        if not self.save_prompt_text:
            return None
        if self.save_prompt_max_chars and len(text) > self.save_prompt_max_chars:
            return text[-self.save_prompt_max_chars:]
        return text

    def _truncate_output_text(self, text: str) -> Optional[str]:
        if not self.save_output_text:
            return None
        if self.save_output_max_chars and len(text) > self.save_output_max_chars:
            return text[: self.save_output_max_chars]
        return text

    def _format_steps(self, steps: List[str]) -> str:
        if not steps:
            return "(none yet)"
        return "\n\n".join([f"Step {i+1}: {s}" for i, s in enumerate(steps)])

    def _steps_to_text(self, steps: List[str]) -> str:
        return "\n\n".join(
            [f"Step {i+1}: {s.strip()}" for i, s in enumerate(steps) if s and s.strip()]
        ).strip()

    def _build_prompt(self, question: str, steps: List[str]) -> str:
        formatted_steps = self._format_steps(steps)
        return (
            "You are solving a math problem step by step.\n\n"
            f"Problem:\n{question}\n\n"
            f"Current reasoning steps:\n{formatted_steps}\n\n"
            "Write exactly ONE next step.\n"
            "Rules:\n"
            "1. Output exactly ONE step.\n"
            "2. Make concrete progress.\n"
            "3. If this is the final step, clearly state the final answer, preferably with \\boxed{}.\n"
            "4. End your output with <STEP_END>.\n\n"
            "Next step:"
        )

    def _strip_step_end_marker(self, text: str) -> str:
        return (text or "").replace("<STEP_END>", "").strip()

    def _normalize_generated_step(self, out_text: str) -> str:
        text = self._strip_step_end_marker(out_text or "")
        if not text:
            return ""
        text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
        text = _STEP_PREFIX_RE.sub("", text)
        return text.strip()

    def _is_terminal_state(
        self,
        *,
        depth: int,
        total_generated_tokens: int,
        finish_reason: Optional[str],
        step_text: str = "",
        full_text: str = "",
    ) -> bool:
        answer_flag, _, _ = detect_answer_signal(step_text=step_text, full_text=full_text)
        if answer_flag:
            return True
        if depth >= self.max_depth:
            return True
        if total_generated_tokens >= self.max_total_new_tokens:
            return True
        if finish_reason is not None and str(finish_reason).lower() == "eos":
            return True
        return False

    def _terminal_reason(
        self,
        *,
        depth: int,
        total_generated_tokens: int,
        finish_reason: Optional[str],
        step_text: str = "",
        full_text: str = "",
    ) -> str:
        answer_flag, reason, _ = detect_answer_signal(step_text=step_text, full_text=full_text)
        if answer_flag:
            return reason or "answer_detected"
        if depth >= self.max_depth:
            return "max_depth"
        if total_generated_tokens >= self.max_total_new_tokens:
            return "max_total_new_tokens"
        if finish_reason is not None and str(finish_reason).lower() == "eos":
            return "eos"
        return ""

    def _new_node(self, **kwargs) -> int:
        self.next_node_id += 1
        nid = self.next_node_id
        self.nodes[nid] = Node(node_id=nid, **kwargs)
        return nid

    def init_root(self, question: str) -> int:
        prompt = self._build_prompt(question, [])
        prompt_ids = self._tokenize_text(prompt, add_special_tokens=True)
        prompt_len = len(prompt_ids) if prompt_ids is not None else max(1, len(prompt.split()))
        nid = self._new_node(
            parent_id=None,
            depth=0,
            action_text="",
            steps=[],
            text="",
            prompt_tokens_len=prompt_len,
            cum_generated_tokens=0,
            finish_reason=None,
            terminal_reason=None,
            is_terminal=False,
            is_fully_expanded=False,
        )
        if prompt_ids is not None:
            self.logger.register_node_sequence(nid, prompt_ids)
        return nid

    def _get_children(self, node_id: int) -> List[int]:
        return self.nodes[node_id].children

    def _is_fully_expanded(self, node_id: int) -> bool:
        return bool(self.nodes[node_id].is_fully_expanded)

    def _uct(self, parent_id: int, child_id: int) -> float:
        child = self.nodes[child_id]
        parent = self.nodes[parent_id]
        if child.N == 0:
            return float(child.search_value + self.unvisited_bonus)
        exploit = child.Q / max(1, child.N)
        explore = self.exploration_weight * child.search_value * math.sqrt(
            math.log(max(2, parent.N + 1)) / (child.N + 1)
        )
        return exploit + explore

    def _path_to_root(self, node_id: int) -> List[int]:
        path = []
        cur = node_id
        while cur is not None:
            path.append(cur)
            cur = self.nodes[cur].parent_id
        path.reverse()
        return path

    def _run_generate_batch(self, prompt_texts: List[str]):
        try:
            return self.lm.generate_batch(
                prompt_texts,
                max_new_tokens=self.max_new_tokens_per_step,
                temperature=self.temperature,
                top_p=self.top_p,
                stop=self.stop_sequences if self.stop_sequences else None,
            )
        except TypeError:
            return self.lm.generate_batch(
                prompt_texts,
                max_new_tokens=self.max_new_tokens_per_step,
                temperature=self.temperature,
                top_p=self.top_p,
            )

    def _score_paths_with_prm(self, steps_batch: Sequence[Sequence[str]]):
        if not steps_batch:
            return []
        return self.prm.score_paths(self.root_question, steps_batch)

    def _log_llm_call(
        self,
        *,
        rollout_id: int,
        node_id: int,
        parent_node_id: Optional[int],
        purpose: str,
        prompt_text: str,
        step_text: str,
        res,
        meta: Dict[str, Any],
        remember_for_node: bool = True,
        include_vllm_metrics: bool = False,
    ) -> None:
        prompt_token_ids = getattr(res, "prompt_token_ids", None)
        output_token_ids = getattr(res, "output_token_ids", None)

        loc = self.logger.compute_locality(
            node_id=node_id,
            parent_node_id=parent_node_id,
            input_ids=list(prompt_token_ids) if prompt_token_ids is not None else None,
            remember_for_node=remember_for_node,
        )

        merged_meta = dict(meta or {})

        if include_vllm_metrics:
            if getattr(res, "vllm_metrics_raw", None) is not None:
                merged_meta["vllm_metrics_raw"] = res.vllm_metrics_raw
            if getattr(res, "vllm_metrics_delta", None) is not None:
                merged_meta["vllm_metrics_delta"] = res.vllm_metrics_delta

        rec = CallRecord(
            call_id=self._next_call_id(),
            rollout_id=rollout_id,
            node_id=node_id,
            parent_node_id=parent_node_id,
            purpose=purpose,
            input_len=loc["input_len"],
            output_len=(
                len(output_token_ids)
                if output_token_ids is not None
                else int(getattr(res, "output_tokens_len", 0) or 0)
            ),
            prompt_len=getattr(res, "prompt_tokens_len", None),
            input_ids_digest=loc["input_ids_digest"],
            lcp_last=loc["lcp_last"],
            reuse_last=loc["reuse_last"],
            miss_last=loc["miss_last"],
            lcp_parent=loc["lcp_parent"],
            reuse_parent=loc["reuse_parent"],
            miss_parent=loc["miss_parent"],
            latency_ms=getattr(res, "latency_ms", None),
            tokens_per_sec=getattr(res, "tokens_per_sec", None),
            finish_reason=getattr(res, "finish_reason", None),
            cumulative_logprob=getattr(res, "cumulative_logprob", None),
            avg_logprob=getattr(res, "avg_logprob", None),
            cached_ratio=getattr(res, "cached_ratio", None),
            meta=merged_meta,
            prompt_text=self._truncate_prompt_text(prompt_text),
            output_text=self._truncate_output_text(step_text),
        )

        self.logger.log_call(rec)

    def _log_batch_summary(
        self,
        *,
        rollout_id: int,
        node_id: int,
        parent_node_id: Optional[int],
        purpose: str,
        batch_result: _CandidateBatchResult,
        extra_meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        meta = {
            "batch_prompt_len": batch_result.prompt_len,
            "batch_size_requested": batch_result.requested_count,
            "batch_size_returned": batch_result.returned_count,
            "batch_size_valid": batch_result.valid_count,
            "batch_size_empty": batch_result.empty_count,
            "batch_latency_ms": batch_result.batch_latency_ms,
            "vllm_metrics_raw": batch_result.batch_metrics_raw,
            "vllm_metrics_delta": batch_result.batch_metrics_delta,
        }
        if extra_meta:
            meta.update(extra_meta)

        self.logger.log_call(
            CallRecord(
                call_id=self._next_call_id(),
                rollout_id=rollout_id,
                node_id=node_id,
                parent_node_id=parent_node_id,
                purpose=purpose,
                meta=meta,
                prompt_text=self._truncate_prompt_text(batch_result.prompt_text),
            )
        )

    def _log_prm_call(
        self,
        *,
        rollout_id: int,
        node_id: int,
        parent_node_id: Optional[int],
        prm_result,
        value_mode: str,
        purpose: str = "reward_model",
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        merged_meta = {
            "reward_type": "prm",
            "prm_value_mode": value_mode,
            "prm_step_scores": prm_result.step_scores,
            "prm_last_score": prm_result.last_score,
            "prm_min_score": prm_result.min_score,
            "prm_mean_score": prm_result.mean_score,
            "prm_prod_score": prm_result.prod_score,
        }
        if meta:
            merged_meta.update(meta)

        self.logger.log_call(
            CallRecord(
                call_id=self._next_call_id(),
                rollout_id=rollout_id,
                node_id=node_id,
                parent_node_id=parent_node_id,
                purpose=purpose,
                meta=merged_meta,
            )
        )

    def _select(self, root_id: int) -> List[int]:
        path = []
        cur = root_id
        while True:
            path.append(cur)
            node = self.nodes[cur]
            if node.is_terminal:
                return path
            if not node.is_fully_expanded:
                return path
            children = self._get_children(cur)
            if not children:
                return path
            cur = max(children, key=lambda c: self._uct(cur, c))

    def _prepare_candidate_from_generation(
        self,
        *,
        base_steps: List[str],
        base_depth: int,
        base_total_generated_tokens: int,
        prompt_text: str,
        prompt_token_ids: Optional[List[int]],
        prompt_len: int,
        res: Any,
        prm_result: Any,
        batch_index: int,
        batch_size: int,
    ) -> Optional[_CandidateState]:
        raw_text = getattr(res, "output_text", "") or ""
        step_text = self._normalize_generated_step(raw_text)
        if not step_text:
            return None

        steps = list(base_steps) + [step_text]
        text = self._steps_to_text(steps)
        out_len = int(getattr(res, "output_tokens_len", 0) or 0)
        total_generated_tokens = int(base_total_generated_tokens + out_len)
        finish_reason = getattr(res, "finish_reason", None)
        depth = base_depth + 1

        answer_flag, _, pred_answer = detect_answer_signal(step_text=step_text, full_text=text)
        is_terminal = self._is_terminal_state(
            depth=depth,
            total_generated_tokens=total_generated_tokens,
            finish_reason=finish_reason,
            step_text=step_text,
            full_text=text,
        )
        terminal_reason = (
            self._terminal_reason(
                depth=depth,
                total_generated_tokens=total_generated_tokens,
                finish_reason=finish_reason,
                step_text=step_text,
                full_text=text,
            )
            if is_terminal
            else None
        )

        temp_node = Node(
            node_id=-1,
            parent_id=None,
            depth=depth,
            action_text=step_text,
            steps=list(steps),
            text=text,
            prompt_tokens_len=prompt_len,
            cum_generated_tokens=total_generated_tokens,
            finish_reason=finish_reason,
            terminal_reason=terminal_reason,
            is_terminal=is_terminal,
            prm_step_scores=list(prm_result.step_scores),
            prm_last_score=float(prm_result.last_score),
            prm_min_score=float(prm_result.min_score),
            prm_mean_score=float(prm_result.mean_score),
            prm_prod_score=float(prm_result.prod_score),
            search_value=0.0,
            is_answer_candidate=bool(answer_flag),
            pred_answer=pred_answer,
        )
        temp_node.search_value = _pick_search_value(temp_node, self.prm_value_mode)

        return _CandidateState(
            step_text=step_text,
            steps=list(steps),
            text=text,
            depth=depth,
            total_generated_tokens=total_generated_tokens,
            finish_reason=finish_reason,
            prompt_text=prompt_text,
            prompt_token_ids=list(prompt_token_ids) if prompt_token_ids is not None else None,
            prompt_len=prompt_len,
            output_tokens_len=out_len,
            output_text_raw=raw_text,
            lm_res=res,
            prm_result=prm_result,
            search_value=float(temp_node.search_value),
            is_terminal=bool(is_terminal),
            terminal_reason=terminal_reason,
            is_answer_candidate=bool(answer_flag),
            pred_answer=pred_answer,
            batch_index=batch_index,
            batch_size=batch_size,
        )

    def _generate_candidates(
        self,
        *,
        base_steps: List[str],
        base_depth: int,
        base_total_generated_tokens: int,
        num_candidates: int,
    ) -> _CandidateBatchResult:
        requested_count = max(0, int(num_candidates))
        prompt_text = self._build_prompt(self.root_question, base_steps)
        prompt_token_ids = self._tokenize_text(prompt_text, add_special_tokens=True)
        prompt_len = len(prompt_token_ids) if prompt_token_ids is not None else max(1, len(prompt_text.split()))

        if requested_count <= 0:
            return _CandidateBatchResult(
                prompt_text=prompt_text,
                prompt_token_ids=prompt_token_ids,
                prompt_len=prompt_len,
                requested_count=0,
                returned_count=0,
                valid_count=0,
                empty_count=0,
                batch_latency_ms=None,
                batch_metrics_raw=None,
                batch_metrics_delta=None,
                candidates=[],
            )

        prompt_texts = [prompt_text for _ in range(requested_count)]
        results = self._run_generate_batch(prompt_texts)

        batch_metrics_raw = results[0].vllm_metrics_raw if results else None
        batch_metrics_delta = results[0].vllm_metrics_delta if results else None
        batch_latency_ms = results[0].batch_latency_ms if results else None
        batch_size = len(results)

        valid_items: List[Dict[str, Any]] = []
        empty_count = 0

        for batch_index, res in enumerate(results):
            raw_text = getattr(res, "output_text", "") or ""
            step_text = self._normalize_generated_step(raw_text)
            if not step_text:
                empty_count += 1
                continue
            valid_items.append(
                {
                    "batch_index": batch_index,
                    "res": res,
                    "step_text": step_text,
                }
            )

        candidates: List[_CandidateState] = []
        if valid_items:
            steps_batch = [list(base_steps) + [x["step_text"]] for x in valid_items]
            prm_results = self._score_paths_with_prm(steps_batch)

            for item, prm_result in zip(valid_items, prm_results):
                candidate = self._prepare_candidate_from_generation(
                    base_steps=base_steps,
                    base_depth=base_depth,
                    base_total_generated_tokens=base_total_generated_tokens,
                    prompt_text=prompt_text,
                    prompt_token_ids=prompt_token_ids,
                    prompt_len=prompt_len,
                    res=item["res"],
                    prm_result=prm_result,
                    batch_index=item["batch_index"],
                    batch_size=batch_size,
                )
                if candidate is not None:
                    candidates.append(candidate)

        return _CandidateBatchResult(
            prompt_text=prompt_text,
            prompt_token_ids=prompt_token_ids,
            prompt_len=prompt_len,
            requested_count=requested_count,
            returned_count=batch_size,
            valid_count=len(candidates),
            empty_count=empty_count,
            batch_latency_ms=batch_latency_ms,
            batch_metrics_raw=batch_metrics_raw,
            batch_metrics_delta=batch_metrics_delta,
            candidates=candidates,
        )

    def _expand_batch(self, node_id: int, rollout_id: int) -> List[int]:
        parent = self.nodes[node_id]
        batch_result = self._generate_candidates(
            base_steps=list(parent.steps),
            base_depth=int(parent.depth),
            base_total_generated_tokens=int(parent.cum_generated_tokens),
            num_candidates=self.branch_factor,
        )

        self._log_batch_summary(
            rollout_id=rollout_id,
            node_id=node_id,
            parent_node_id=parent.parent_id,
            purpose="expand_batch_summary",
            batch_result=batch_result,
            extra_meta={
                "stage": "expand",
                "parent_depth": parent.depth,
                "parent_num_children_before": len(parent.children),
            },
        )

        created_child_ids: List[int] = []
        parent.is_fully_expanded = True

        if not batch_result.candidates:
            if not parent.is_terminal:
                parent.is_terminal = True
                parent.terminal_reason = "expand_no_valid_children"
            return created_child_ids

        for candidate in batch_result.candidates:
            child_prompt = self._build_prompt(self.root_question, candidate.steps)
            child_prompt_ids = self._tokenize_text(child_prompt, add_special_tokens=True)
            child_prompt_len = (
                len(child_prompt_ids) if child_prompt_ids is not None else max(1, len(child_prompt.split()))
            )

            child_id = self._new_node(
                parent_id=node_id,
                depth=candidate.depth,
                action_text=candidate.step_text,
                steps=list(candidate.steps),
                text=candidate.text,
                prompt_tokens_len=child_prompt_len,
                cum_generated_tokens=candidate.total_generated_tokens,
                finish_reason=candidate.finish_reason,
                terminal_reason=candidate.terminal_reason,
                is_terminal=candidate.is_terminal,
                prm_step_scores=list(candidate.prm_result.step_scores),
                prm_last_score=float(candidate.prm_result.last_score),
                prm_min_score=float(candidate.prm_result.min_score),
                prm_mean_score=float(candidate.prm_result.mean_score),
                prm_prod_score=float(candidate.prm_result.prod_score),
                search_value=float(candidate.search_value),
                is_answer_candidate=bool(candidate.is_answer_candidate),
                pred_answer=candidate.pred_answer,
                is_fully_expanded=False,
            )
            self.nodes[node_id].children.append(child_id)
            created_child_ids.append(child_id)

            if child_prompt_ids is not None:
                self.logger.register_node_sequence(child_id, child_prompt_ids)

            self._log_llm_call(
                rollout_id=rollout_id,
                node_id=node_id,
                parent_node_id=parent.parent_id,
                purpose="expand",
                prompt_text=candidate.prompt_text,
                step_text=candidate.step_text,
                res=candidate.lm_res,
                meta={
                    "expand_candidate_index": candidate.batch_index,
                    "expand_batch_size": candidate.batch_size,
                    "created_child_id": child_id,
                    "depth": parent.depth,
                    "generated_step_text": candidate.step_text,
                    "child_depth": candidate.depth,
                    "child_cum_generated_tokens": candidate.total_generated_tokens,
                    "child_is_terminal": candidate.is_terminal,
                    "child_terminal_reason": candidate.terminal_reason,
                    "child_is_answer_candidate": candidate.is_answer_candidate,
                    "child_pred_answer": candidate.pred_answer,
                    "prm_last_score": float(candidate.prm_result.last_score),
                    "prm_mean_score": float(candidate.prm_result.mean_score),
                    "prm_min_score": float(candidate.prm_result.min_score),
                    "prm_prod_score": float(candidate.prm_result.prod_score),
                    "search_value": float(candidate.search_value),
                    "empty_step": False,
                },
                include_vllm_metrics=False,
            )
            self._log_prm_call(
                rollout_id=rollout_id,
                node_id=child_id,
                parent_node_id=node_id,
                prm_result=candidate.prm_result,
                value_mode=self.prm_value_mode,
                meta={
                    "stage": "expand",
                    "expand_candidate_index": candidate.batch_index,
                    "expand_batch_size": candidate.batch_size,
                },
            )

        return created_child_ids

    def _pick_best_child_for_simulation(self, child_ids: List[int]) -> Optional[int]:
        if not child_ids:
            return None
        return max(
            child_ids,
            key=lambda nid: (
                float(self.nodes[nid].search_value),
                float(self.nodes[nid].prm_mean_score),
                -int(self.nodes[nid].depth),
            ),
        )

    def _simulate_greedy(
        self,
        *,
        start_child_id: int,
        rollout_id: int,
    ) -> Dict[str, Any]:
        formal_start = self.nodes[start_child_id]
        full_tree_path = self._path_to_root(start_child_id)

        current_steps = list(formal_start.steps)
        current_text = formal_start.text
        current_depth = int(formal_start.depth)
        current_total_generated_tokens = int(formal_start.cum_generated_tokens)
        current_finish_reason = formal_start.finish_reason
        current_term_reason = formal_start.terminal_reason
        current_search_value = float(formal_start.search_value)
        current_prm_step_scores = list(formal_start.prm_step_scores)
        current_prm_last_score = float(formal_start.prm_last_score)
        current_prm_min_score = float(formal_start.prm_min_score)
        current_prm_mean_score = float(formal_start.prm_mean_score)
        current_prm_prod_score = float(formal_start.prm_prod_score)
        current_pred_answer = formal_start.pred_answer

        num_sim_steps = 0

        if formal_start.is_terminal:
            return {
                "start_node_id": start_child_id,
                "final_node_id": start_child_id,
                "final_depth": current_depth,
                "final_text": current_text,
                "final_steps": list(current_steps),
                "num_growth_steps_after_start": 0,
                "total_generated_tokens": current_total_generated_tokens,
                "finish_reason": current_finish_reason,
                "term_reason": current_term_reason,
                "full_tree_path": full_tree_path,
                "search_value": current_search_value,
                "prm_step_scores": list(current_prm_step_scores),
                "prm_last_score": current_prm_last_score,
                "prm_min_score": current_prm_min_score,
                "prm_mean_score": current_prm_mean_score,
                "prm_prod_score": current_prm_prod_score,
                "pred_answer": current_pred_answer,
            }

        while True:
            batch_result = self._generate_candidates(
                base_steps=list(current_steps),
                base_depth=current_depth,
                base_total_generated_tokens=current_total_generated_tokens,
                num_candidates=self.simulation_branch_factor,
            )

            self._log_batch_summary(
                rollout_id=rollout_id,
                node_id=start_child_id,
                parent_node_id=formal_start.parent_id,
                purpose="simulate_batch_summary",
                batch_result=batch_result,
                extra_meta={
                    "stage": "simulate",
                    "simulate_step_index": num_sim_steps + 1,
                    "anchor_formal_child_id": start_child_id,
                    "temp_depth_before": current_depth,
                },
            )

            if not batch_result.candidates:
                current_term_reason = current_term_reason or "simulate_no_valid_children"
                break

            for candidate in batch_result.candidates:
                self._log_llm_call(
                    rollout_id=rollout_id,
                    node_id=start_child_id,
                    parent_node_id=formal_start.parent_id,
                    purpose="simulate",
                    prompt_text=candidate.prompt_text,
                    step_text=candidate.step_text,
                    res=candidate.lm_res,
                    remember_for_node=False,
                    meta={
                        "simulate_candidate_index": candidate.batch_index,
                        "simulate_batch_size": candidate.batch_size,
                        "simulate_step_index": num_sim_steps + 1,
                        "anchor_formal_child_id": start_child_id,
                        "temp_depth": candidate.depth,
                        "temp_cum_generated_tokens": candidate.total_generated_tokens,
                        "temp_is_terminal": candidate.is_terminal,
                        "temp_terminal_reason": candidate.terminal_reason,
                        "temp_pred_answer": candidate.pred_answer,
                        "prm_last_score": float(candidate.prm_result.last_score),
                        "prm_mean_score": float(candidate.prm_result.mean_score),
                        "prm_min_score": float(candidate.prm_result.min_score),
                        "prm_prod_score": float(candidate.prm_result.prod_score),
                        "search_value": float(candidate.search_value),
                        "empty_step": False,
                    },
                    include_vllm_metrics=False,
                )
                self._log_prm_call(
                    rollout_id=rollout_id,
                    node_id=start_child_id,
                    parent_node_id=formal_start.parent_id,
                    prm_result=candidate.prm_result,
                    value_mode=self.prm_value_mode,
                    meta={
                        "stage": "simulate",
                        "simulate_candidate_index": candidate.batch_index,
                        "simulate_batch_size": candidate.batch_size,
                        "simulate_step_index": num_sim_steps + 1,
                        "anchor_formal_child_id": start_child_id,
                    },
                )

            best = max(
                batch_result.candidates,
                key=lambda c: (
                    float(c.search_value),
                    float(c.prm_result.mean_score),
                    -int(c.depth),
                ),
            )

            self.logger.log_call(
                CallRecord(
                    call_id=self._next_call_id(),
                    rollout_id=rollout_id,
                    node_id=start_child_id,
                    parent_node_id=formal_start.parent_id,
                    purpose="simulate_choice",
                    meta={
                        "simulate_step_index": num_sim_steps + 1,
                        "anchor_formal_child_id": start_child_id,
                        "chosen_step_text": best.step_text,
                        "chosen_depth": best.depth,
                        "chosen_total_generated_tokens": best.total_generated_tokens,
                        "chosen_is_terminal": best.is_terminal,
                        "chosen_terminal_reason": best.terminal_reason,
                        "chosen_search_value": float(best.search_value),
                        "chosen_prm_last_score": float(best.prm_result.last_score),
                        "chosen_prm_min_score": float(best.prm_result.min_score),
                        "chosen_prm_mean_score": float(best.prm_result.mean_score),
                        "chosen_prm_prod_score": float(best.prm_result.prod_score),
                        "chosen_batch_index": best.batch_index,
                        "chosen_batch_size": best.batch_size,
                    },
                )
            )

            current_steps = list(best.steps)
            current_text = best.text
            current_depth = int(best.depth)
            current_total_generated_tokens = int(best.total_generated_tokens)
            current_finish_reason = best.finish_reason
            current_term_reason = best.terminal_reason
            current_search_value = float(best.search_value)
            current_prm_step_scores = list(best.prm_result.step_scores)
            current_prm_last_score = float(best.prm_result.last_score)
            current_prm_min_score = float(best.prm_result.min_score)
            current_prm_mean_score = float(best.prm_result.mean_score)
            current_prm_prod_score = float(best.prm_result.prod_score)
            current_pred_answer = best.pred_answer
            num_sim_steps += 1

            if best.is_terminal:
                break

        return {
            "start_node_id": start_child_id,
            "final_node_id": start_child_id,
            "final_depth": current_depth,
            "final_text": current_text,
            "final_steps": list(current_steps),
            "num_growth_steps_after_start": num_sim_steps,
            "total_generated_tokens": current_total_generated_tokens,
            "finish_reason": current_finish_reason,
            "term_reason": current_term_reason,
            "full_tree_path": full_tree_path,
            "search_value": current_search_value,
            "prm_step_scores": list(current_prm_step_scores),
            "prm_last_score": current_prm_last_score,
            "prm_min_score": current_prm_min_score,
            "prm_mean_score": current_prm_mean_score,
            "prm_prod_score": current_prm_prod_score,
            "pred_answer": current_pred_answer,
        }

    def _evaluate_rollout(self, rollout_result: Dict[str, Any]) -> Dict[str, Any]:
        return evaluate_against_gt(rollout_result["final_text"], self.gt_answer)

    def _backpropagate(self, full_tree_path: List[int], leaf_value: float) -> None:
        cur_value = float(leaf_value)
        for nid in reversed(full_tree_path):
            self.nodes[nid].Q += cur_value
            self.nodes[nid].N += 1
            cur_value *= self.discount

    def _build_tree_stats(self) -> Dict[str, Any]:
        nonleaf_nodes = [n for n in self.nodes.values() if len(n.children) > 0]
        terminal_nodes = [n for n in self.nodes.values() if n.is_terminal]
        avg_branching = (
            sum(len(n.children) for n in nonleaf_nodes) / len(nonleaf_nodes) if nonleaf_nodes else 0.0
        )
        max_depth = max((n.depth for n in self.nodes.values()), default=0)
        num_fully_expanded = sum(1 for n in self.nodes.values() if n.is_fully_expanded)
        return {
            "num_nodes": len(self.nodes),
            "num_terminal_nodes": len(terminal_nodes),
            "num_nonleaf_nodes": len(nonleaf_nodes),
            "num_fully_expanded_nodes": num_fully_expanded,
            "max_tree_depth": max_depth,
            "avg_branching_factor_nonleaf": avg_branching,
        }

    def _build_rollout_stats(self) -> Dict[str, Any]:
        if not self.rollout_summaries:
            return {
                "num_rollouts": 0,
                "num_correct_rollouts_offline": 0,
                "mean_search_value": 0.0,
                "terminal_reason_counts": {},
            }
        n = len(self.rollout_summaries)
        term_counts: Dict[str, int] = {}
        for x in self.rollout_summaries:
            r = str(x.get("term_reason", "") or "unknown")
            term_counts[r] = term_counts.get(r, 0) + 1
        num_correct = sum(1 for x in self.rollout_summaries if x.get("correct_bool") is True)
        mean_search_value = sum(float(x.get("search_value", 0.0)) for x in self.rollout_summaries) / n
        return {
            "num_rollouts": n,
            "num_correct_rollouts_offline": num_correct,
            "mean_search_value": mean_search_value,
            "terminal_reason_counts": term_counts,
        }

    def _best_terminal_summary(self) -> Optional[Dict[str, Any]]:
        if not self.rollout_summaries:
            return None
        mode = (self.final_select_mode or "min").lower()
        key_map = {
            "last": "prm_last_score",
            "min": "prm_min_score",
            "mean": "prm_mean_score",
            "prod": "prm_prod_score",
        }
        key_name = key_map.get(mode, "prm_min_score")
        return max(
            self.rollout_summaries,
            key=lambda x: (
                float(x.get(key_name, 0.0)),
                float(x.get("prm_mean_score", 0.0)),
                -int(x.get("final_depth", 0)),
            ),
        )

    def initialize(self, question: str, gt_answer: Optional[str]) -> None:
        self.nodes = {}
        self.next_node_id = 0
        self.rollout_summaries = []
        self.final_summary = {}
        self.root_question = question
        self.gt_answer = gt_answer
        self.root_id = self.init_root(question)

    def run_one_rollout(self, rollout_id: int) -> Dict[str, Any]:
        if self.root_id is None:
            raise RuntimeError("MCTSMinimal is not initialized. Call initialize() first.")

        root_id = self.root_id
        select_path = self._select(root_id)
        leaf_id = select_path[-1]
        leaf = self.nodes[leaf_id]

        self.logger.log_call(
            CallRecord(
                call_id=self._next_call_id(),
                rollout_id=rollout_id,
                node_id=leaf_id,
                parent_node_id=leaf.parent_id,
                purpose="select",
                meta={
                    "selected_leaf_id": leaf_id,
                    "selected_path": list(select_path),
                    "selected_depth": leaf.depth,
                    "selected_leaf_is_terminal": leaf.is_terminal,
                    "selected_leaf_num_children": len(leaf.children),
                    "selected_leaf_is_fully_expanded": leaf.is_fully_expanded,
                    "selected_leaf_prm_last_score": leaf.prm_last_score,
                    "selected_leaf_search_value": leaf.search_value,
                },
            )
        )

        expanded_child_ids: List[int] = []
        start_node_id: Optional[int] = None

        if leaf.is_terminal:
            rollout_result = {
                "start_node_id": leaf_id,
                "final_node_id": leaf_id,
                "final_depth": leaf.depth,
                "final_text": leaf.text,
                "final_steps": list(leaf.steps),
                "num_growth_steps_after_start": 0,
                "total_generated_tokens": leaf.cum_generated_tokens,
                "finish_reason": leaf.finish_reason,
                "term_reason": leaf.terminal_reason,
                "full_tree_path": self._path_to_root(leaf_id),
                "search_value": float(leaf.search_value),
                "prm_step_scores": list(leaf.prm_step_scores),
                "prm_last_score": float(leaf.prm_last_score),
                "prm_min_score": float(leaf.prm_min_score),
                "prm_mean_score": float(leaf.prm_mean_score),
                "prm_prod_score": float(leaf.prm_prod_score),
                "pred_answer": leaf.pred_answer,
            }
        else:
            expanded_child_ids = self._expand_batch(leaf_id, rollout_id)
            start_node_id = self._pick_best_child_for_simulation(expanded_child_ids)

            self.logger.log_call(
                CallRecord(
                    call_id=self._next_call_id(),
                    rollout_id=rollout_id,
                    node_id=leaf_id,
                    parent_node_id=leaf.parent_id,
                    purpose="expand_summary",
                    meta={
                        "expanded_child_ids": list(expanded_child_ids),
                        "num_expanded_children": len(expanded_child_ids),
                        "selected_simulation_start_node_id": start_node_id,
                        "node_marked_fully_expanded": self.nodes[leaf_id].is_fully_expanded,
                        "node_is_terminal_after_expand": self.nodes[leaf_id].is_terminal,
                        "node_terminal_reason_after_expand": self.nodes[leaf_id].terminal_reason,
                    },
                )
            )

            if start_node_id is None:
                leaf = self.nodes[leaf_id]
                rollout_result = {
                    "start_node_id": leaf_id,
                    "final_node_id": leaf_id,
                    "final_depth": leaf.depth,
                    "final_text": leaf.text,
                    "final_steps": list(leaf.steps),
                    "num_growth_steps_after_start": 0,
                    "total_generated_tokens": leaf.cum_generated_tokens,
                    "finish_reason": leaf.finish_reason,
                    "term_reason": leaf.terminal_reason,
                    "full_tree_path": self._path_to_root(leaf_id),
                    "search_value": float(leaf.search_value),
                    "prm_step_scores": list(leaf.prm_step_scores),
                    "prm_last_score": float(leaf.prm_last_score),
                    "prm_min_score": float(leaf.prm_min_score),
                    "prm_mean_score": float(leaf.prm_mean_score),
                    "prm_prod_score": float(leaf.prm_prod_score),
                    "pred_answer": leaf.pred_answer,
                }
            else:
                rollout_result = self._simulate_greedy(
                    start_child_id=start_node_id,
                    rollout_id=rollout_id,
                )

        offline_eval = self._evaluate_rollout(rollout_result)
        search_value = float(rollout_result["search_value"])
        full_tree_path = rollout_result["full_tree_path"]
        self._backpropagate(full_tree_path, search_value)

        rollout_summary = {
            "rollout_id": rollout_id,
            "search_value": search_value,
            "selected_path": list(select_path),
            "selected_path_len": len(select_path),
            "full_tree_path": list(full_tree_path),
            "full_tree_path_len": len(full_tree_path),
            "expanded_child_ids": list(expanded_child_ids),
            "num_expanded_children": len(expanded_child_ids),
            "start_node_id": rollout_result["start_node_id"],
            "final_node_id": rollout_result["final_node_id"],
            "final_depth": rollout_result["final_depth"],
            "final_text": rollout_result["final_text"],
            "num_steps_total": len(rollout_result["final_steps"]),
            "num_growth_steps_after_start": rollout_result["num_growth_steps_after_start"],
            "total_generated_tokens": rollout_result["total_generated_tokens"],
            "finish_reason": rollout_result["finish_reason"],
            "term_reason": rollout_result["term_reason"],
            "prm_step_scores": rollout_result["prm_step_scores"],
            "prm_last_score": rollout_result["prm_last_score"],
            "prm_min_score": rollout_result["prm_min_score"],
            "prm_mean_score": rollout_result["prm_mean_score"],
            "prm_prod_score": rollout_result["prm_prod_score"],
            "pred": offline_eval["pred"],
            "gt": offline_eval["gt"],
            "correct_bool": offline_eval["correct_bool"],
            "answer_type": offline_eval["answer_type"],
            "answer_matched_text": offline_eval["answer_matched_text"],
        }
        self.rollout_summaries.append(rollout_summary)

        self.logger.log_call(
            CallRecord(
                call_id=self._next_call_id(),
                rollout_id=rollout_id,
                node_id=rollout_result["final_node_id"],
                parent_node_id=self.nodes[rollout_result["final_node_id"]].parent_id if rollout_result["final_node_id"] in self.nodes else None,
                purpose="rollout_summary",
                meta=rollout_summary,
            )
        )
        return rollout_summary

    def finalize(self) -> Dict[str, Any]:
        if self.root_id is None:
            raise RuntimeError("MCTSMinimal is not initialized. Call initialize() first.")
        self.final_summary = {
            "tree_stats": self._build_tree_stats(),
            "rollout_stats": self._build_rollout_stats(),
            "best_rollout_by_prm": self._best_terminal_summary(),
            "num_rollouts": len(self.rollout_summaries),
        }
        self.logger.log_call(
            CallRecord(
                call_id=self._next_call_id(),
                rollout_id=-1,
                node_id=self.root_id,
                parent_node_id=None,
                purpose="final",
                meta=self.final_summary,
            )
        )
        return dict(self.final_summary)

    def run(self, question: str, gt_answer: Optional[str]) -> Dict[str, Any]:
        self.initialize(question, gt_answer)
        for rollout_id in range(self.num_rollouts):
            self.run_one_rollout(rollout_id)
        return self.finalize()

    def get_final_summary(self) -> Dict[str, Any]:
        return dict(self.final_summary or {})