# no_vllm

This directory contains a Hugging Face native version of the tree-search pipeline.

Key changes from the root implementation:
- policy generation uses `transformers` + `torch` instead of `vllm`
- PRM serving remains the same shape and still uses Hugging Face models
- output directories, trace files, rollout summaries, and sample summaries are kept compatible
- engine metrics are preserved, but they are now HF-native metrics gathered from `torch` and runner-side counters

Run the PRM server:

```bash
python -m no_vllm.scripts.prm_server --prm_model_ckpt /path/to/prm
```

Run search:

```bash
python -m no_vllm.scripts.run_search_minimal \
  --model_ckpt /path/to/policy \
  --dataset_path data/gsm8k_single_longer_89.jsonl \
  --out_dir outputs/no_vllm_run \
  --hf_enable_metrics
```

Assumptions:
- `transformers`, `torch`, and PRM dependencies are available in the environment
- datasets can still be read from the repository root `data/` directory
- the HF policy model can be loaded with `AutoModelForCausalLM`

