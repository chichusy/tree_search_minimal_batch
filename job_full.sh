#!/bin/bash
#SBATCH -J ts_full
#SBATCH -p A100
#SBATCH --qos=normal
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --mem=256G
#SBATCH --gres=gpu:2
#SBATCH -t 24:00:00
#SBATCH -o /data2/group_谈海生/wsy/outputs/%x-%j.out
#SBATCH -e /data2/group_谈海生/wsy/outputs/%x-%j.err

set -euo pipefail

WORKDIR=/home/wsy/projects/tree_search_minimal
PRM_PY=/home/wsy/.conda/envs/prm/bin/python
MAIN_PY=/home/wsy/.conda/envs/wsy_env1/bin/python

PRM_MODEL=/data2/group_谈海生/wsy/models/Qwen2.5-Math-PRM-7B
POLICY_MODEL=/data2/group_谈海生/wsy/models/Qwen2.5-7B-Instruct
DATASET=data/gsm8k_difficulty_eval_90.jsonl
OUTROOT=/data2/group_谈海生/wsy/outputs
OUTDIR=${OUTROOT}/full_run_${SLURM_JOB_ID}

mkdir -p "${OUTDIR}"
cd "${WORKDIR}"

echo "===== job info ====="
echo "job id: ${SLURM_JOB_ID}"
echo "node list: ${SLURM_JOB_NODELIST}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "workdir: $(pwd)"
echo "start time: $(date)"
hostname
nvidia-smi

cleanup() {
  if [[ -n "${PRM_PID:-}" ]]; then
    echo "killing PRM server pid=${PRM_PID}"
    kill "${PRM_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

echo "===== start prm server ====="
CUDA_VISIBLE_DEVICES=1 "${PRM_PY}" -m scripts.prm_server \
  --host 127.0.0.1 \
  --port 18080 \
  --prm_model_ckpt "${PRM_MODEL}" \
  --prm_device cuda:0 \
  --prm_dtype bfloat16 \
  --prm_batch_size 8 \
  --max_model_len 4096 \
  > "${OUTDIR}/prm_server.log" 2>&1 &

PRM_PID=$!
echo "PRM_PID=${PRM_PID}"

echo "===== wait prm server ====="
READY=0
for i in $(seq 1 120); do
  if ss -ltn | grep -q ':18080 '; then
    READY=1
    break
  fi
  sleep 2
done

if [[ "${READY}" -ne 1 ]]; then
  echo "PRM server did not become ready in time."
  echo "==== prm_server.log ===="
  tail -n 200 "${OUTDIR}/prm_server.log" || true
  exit 1
fi

echo "===== start main search ====="
CUDA_VISIBLE_DEVICES=0 "${MAIN_PY}" -m scripts.run_search_minimal \
  --runner vllm \
  --model_ckpt "${POLICY_MODEL}" \
  --dataset_path "${DATASET}" \
  --dataset_format jsonl \
  --max_samples 90 \
  --out_dir "${OUTDIR}/run_search" \
  --num_rollouts 20 \
  --max_depth 8 \
  --branch_factor 5 \
  --simulation_branch_factor 2 \
  --max_new_tokens_per_step 128 \
  --max_total_new_tokens 1024 \
  --temperature 0.7 \
  --top_p 0.95 \
  --prm_value_mode last \
  --final_select_mode min \
  --save_prompt_text \
  --save_output_text

echo "===== job finished ====="
date