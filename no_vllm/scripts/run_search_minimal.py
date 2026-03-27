from __future__ import annotations

import argparse
import json
from pathlib import Path

from no_vllm.src.core.dataset import load_dataset, extract_question
from no_vllm.src.core.lm import HFRunner
from no_vllm.src.core.trace.writer import TraceLogger
from no_vllm.src.core.eval_minimal import extract_gt_from_answer_field
from no_vllm.src.core.prm_remote import RemotePRMScorer
from no_vllm.src.search.mcts_minimal import MCTSMinimal


def infer_dataset_name(sample, dataset_path: str, dataset_name_arg: str = "auto") -> str:
    if isinstance(dataset_name_arg, str) and dataset_name_arg.strip().lower() != "auto":
        return dataset_name_arg.strip().lower()

    if isinstance(sample, dict):
        ds = sample.get("dataset", None)
        if isinstance(ds, str) and ds.strip():
            return ds.strip().lower()

    name = Path(dataset_path).name.lower()
    if "gsm8k" in name:
        return "gsm8k"
    if "math500" in name:
        return "math500"
    if "math" in name:
        return "math"
    return "unknown"


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--dataset_path", type=str, required=True)
    ap.add_argument("--dataset_format", type=str, default="jsonl")
    ap.add_argument("--dataset_name", type=str, default="auto")
    ap.add_argument("--max_samples", type=int, default=1)

    ap.add_argument("--runner", type=str, default="hf", choices=["hf"])
    ap.add_argument("--model_ckpt", type=str, required=True)
    ap.add_argument("--hf_max_batch_size", type=int, default=16)
    ap.add_argument("--torch_dtype", type=str, default="auto")
    ap.add_argument("--hf_seed", type=int, default=0)
    ap.add_argument("--hf_device_map", type=str, default="auto")
    ap.add_argument("--attn_implementation", type=str, default=None)
    ap.add_argument("--hf_logprobs", type=int, default=1)
    ap.add_argument("--max_model_len", type=int, default=1024)
    ap.add_argument("--hf_enable_metrics", action="store_true")

    ap.add_argument("--prm_url", type=str, default="http://127.0.0.1:18080")
    ap.add_argument("--prm_timeout_seconds", type=float, default=120.0)
    ap.add_argument("--prm_num_retries", type=int, default=2)
    ap.add_argument("--prm_retry_sleep_seconds", type=float, default=1.0)
    ap.add_argument("--prm_value_mode", type=str, default="last", choices=["last", "min", "mean", "prod"])
    ap.add_argument("--final_select_mode", type=str, default="min", choices=["last", "min", "mean", "prod"])

    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--num_rollouts", type=int, default=10)
    ap.add_argument("--max_depth", type=int, default=6)
    ap.add_argument("--branch_factor", type=int, default=3)
    ap.add_argument("--simulation_branch_factor", type=int, default=3)
    ap.add_argument("--max_new_tokens_per_step", type=int, default=96)
    ap.add_argument("--max_total_new_tokens", type=int, default=256)
    ap.add_argument("--exploration_weight", type=float, default=1.4)
    ap.add_argument("--unvisited_bonus", type=float, default=1e9)
    ap.add_argument("--discount", type=float, default=1.0)

    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.95)

    ap.add_argument("--save_prompt_text", action="store_true")
    ap.add_argument("--save_output_text", action="store_true")
    ap.add_argument("--save_prompt_max_chars", type=int, default=0)
    ap.add_argument("--save_output_max_chars", type=int, default=0)

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    samples = load_dataset(args.dataset_path, args.dataset_format)[: args.max_samples]

    min_needed_batch = max(args.branch_factor, args.simulation_branch_factor)
    if args.hf_max_batch_size < min_needed_batch:
        print(
            f"[run_search_minimal] bump hf_max_batch_size: "
            f"{args.hf_max_batch_size} -> {min_needed_batch}"
        )
        args.hf_max_batch_size = min_needed_batch

    lm = HFRunner(
        model_ckpt=args.model_ckpt,
        seed=args.hf_seed,
        max_batch_size=args.hf_max_batch_size,
        dtype=args.torch_dtype,
        device_map=args.hf_device_map,
        attn_implementation=args.attn_implementation,
        max_model_len=args.max_model_len,
        logprobs=args.hf_logprobs,
        enable_metrics=args.hf_enable_metrics,
    )
    tokenizer = getattr(lm, "tokenizer", None)

    prm = RemotePRMScorer(
        base_url=args.prm_url,
        timeout_seconds=args.prm_timeout_seconds,
        num_retries=args.prm_num_retries,
        retry_sleep_seconds=args.prm_retry_sleep_seconds,
    )
    health = prm.health_check()
    print(f"[PRM] connected: {health}")

    sample_states = []
    for i, sample in enumerate(samples):
        q = extract_question(sample)
        raw_sample_id = sample.get("id", i) if isinstance(sample, dict) else i
        answer_text = sample.get("answer", None) if isinstance(sample, dict) else None
        difficulty = sample.get("difficulty", "unknown") if isinstance(sample, dict) else "unknown"
        dataset_name = infer_dataset_name(sample, args.dataset_path, args.dataset_name)

        parsed_gt = None
        if isinstance(answer_text, str):
            parsed_gt = extract_gt_from_answer_field(answer_text, dataset_name)

        sample_dir = out_dir / "samples" / f"{i:05d}"
        sample_dir.mkdir(parents=True, exist_ok=True)

        trace_logger = TraceLogger(str(sample_dir / "trace_calls.jsonl"), mode="w")
        trace_logger.set_sample_context(
            {
                "sample_id": raw_sample_id,
                "difficulty": difficulty,
                "dataset": dataset_name,
                "engine": "hf",
            }
        )

        mcts = MCTSMinimal(
            lm_runner=lm,
            prm_scorer=prm,
            trace_logger=trace_logger,
            tokenizer=tokenizer,
            max_depth=args.max_depth,
            branch_factor=args.branch_factor,
            simulation_branch_factor=args.simulation_branch_factor,
            num_rollouts=args.num_rollouts,
            max_new_tokens_per_step=args.max_new_tokens_per_step,
            max_total_new_tokens=args.max_total_new_tokens,
            exploration_weight=args.exploration_weight,
            unvisited_bonus=args.unvisited_bonus,
            discount=args.discount,
            temperature=args.temperature,
            top_p=args.top_p,
            prm_value_mode=args.prm_value_mode,
            final_select_mode=args.final_select_mode,
            save_prompt_text=args.save_prompt_text,
            save_output_text=args.save_output_text,
            save_prompt_max_chars=args.save_prompt_max_chars,
            save_output_max_chars=args.save_output_max_chars,
        )
        mcts.initialize(question=q, gt_answer=parsed_gt)

        sample_states.append(
            {
                "index": i,
                "raw_sample_id": raw_sample_id,
                "difficulty": difficulty,
                "dataset": dataset_name,
                "question": q,
                "answer": answer_text,
                "parsed_gt": parsed_gt,
                "sample_dir": sample_dir,
                "trace_logger": trace_logger,
                "mcts": mcts,
            }
        )

    for rollout_id in range(args.num_rollouts):
        print(f"[ROUND] rollout_id={rollout_id}")
        for state in sample_states:
            state["mcts"].run_one_rollout(rollout_id)

    for state in sample_states:
        final_summary = state["mcts"].finalize()
        sample_info = {
            "sample_id": state["raw_sample_id"],
            "dataset": state["dataset"],
            "difficulty": state["difficulty"],
            "question": state["question"],
            "answer": state["answer"],
            "parsed_gt": state["parsed_gt"],
            "search_mode": "batch_hf_expand_and_simulate",
            "prm_url": args.prm_url,
            "mcts_config": {
                "branch_factor": args.branch_factor,
                "simulation_branch_factor": args.simulation_branch_factor,
                "num_rollouts": args.num_rollouts,
                "max_depth": args.max_depth,
                "max_new_tokens_per_step": args.max_new_tokens_per_step,
                "max_total_new_tokens": args.max_total_new_tokens,
                "exploration_weight": args.exploration_weight,
                "unvisited_bonus": args.unvisited_bonus,
                "discount": args.discount,
                "prm_value_mode": args.prm_value_mode,
                "final_select_mode": args.final_select_mode,
                "hf_max_batch_size": args.hf_max_batch_size,
                "torch_dtype": args.torch_dtype,
                "hf_seed": args.hf_seed,
                "hf_device_map": args.hf_device_map,
                "attn_implementation": args.attn_implementation,
                "max_model_len": args.max_model_len,
            },
            "final_summary": final_summary,
        }
        with open(state["sample_dir"] / "sample_info.json", "w", encoding="utf-8") as f:
            json.dump(sample_info, f, ensure_ascii=False, indent=2)

        state["trace_logger"].close()
        print(f"[OK] sample {state['index']} -> {state['sample_dir']}")

    print("[DONE]")


if __name__ == "__main__":
    main()

