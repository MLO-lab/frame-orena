#!/bin/bash
#SBATCH --job-name=frame-train
#SBATCH --gres=gpu:h100:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%j_%x.out
#SBATCH --error=logs/%j_%x.err
#
# Qwen3.5-9B + rank-8 vision LoRA on the FRAME mix. 2x H100, 4 epochs.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"

: "${PYTHON:?set PYTHON to the interpreter (see README)}"
: "${TORCHRUN:=$(dirname "$PYTHON")/torchrun}"
: "${HF_HOME:?set HF_HOME to the cache holding Qwen/Qwen3.5-9B}"
: "${EXPORT:=$ROOT/sft_export}"
: "${RUN_NAME:=frame-9b-8r-vision-4ep}"
: "${NPROC:=2}"
export HF_HOME HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
mkdir -p logs checkpoints

TRAIN="$EXPORT/train.jsonl"; MONITOR="$EXPORT/monitor.jsonl"
[[ -f "$TRAIN" ]] || { echo "FATAL: $TRAIN missing -- run slurm/1_prepare_data.sbatch"; exit 1; }
echo "training rows: $(wc -l < "$TRAIN")"

echo "Node: $(hostname)"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
export SMOKE_BATCH=4 SMOKE_TRAIN_FILE="$TRAIN" LORA_VISION=1
$PYTHON orena_sft/preflight_vision_lora.py

$TORCHRUN --standalone --nproc_per_node="$NPROC" \
    orena_sft/sft_train_qwen_frame_ddp.py \
    --train-file "$TRAIN" --eval-file "$MONITOR" \
    --run-name "$RUN_NAME" \
    --prompt-style direct --lora-vision \
    --lora-r 8 --lora-alpha 16 \
    --epochs 4 --lr 1e-4 --seed 42 \
    --batch-size 4 --grad-accum 4 \
    --eval-steps 250 --save-steps 533 --save-total-limit 10
