#!/bin/bash
#SBATCH --job-name=frame-prep
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%j_%x.out
#SBATCH --error=logs/%j_%x.err
#
# Build the FRAME training set: 34,006 examples (20,000 official + 14,006 generated).
# CPU only. Writes $EXPORT/train.jsonl.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"

: "${PYTHON:?set PYTHON to the interpreter (see README)}"
: "${ORENA_DATA_ROOT:?set ORENA_DATA_ROOT to the directory holding heico/ lapchole/}"
: "${EXPORT:=$ROOT/sft_export}"
: "${GEN:=$ROOT/generated}"
: "${ORENA_ANNOT_DIR:=$ROOT/annotations/private/raw}"
: "${LAPCHOLE_CSV_DIR:=$ROOT/annotations/private/raw/lapchole_annotations}"
: "${MESH_CSV_DIR:=$ROOT/annotations/public/raw/hernia_mesh_annotations}"
: "${MESH_FRAMES_DIR:=$ROOT/annotations/public/frames}"
export ORENA_DATA_ROOT ORENA_ANNOT_DIR MESH_FRAMES_DIR FRAME_GEN_DIR="$GEN"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
mkdir -p "$EXPORT" "$GEN" logs

echo "############ 1/5  official export ############"
$PYTHON orena_sft/build_frame_sft_dataset.py --datasets heico lapchole \
    --root-dir "$ORENA_DATA_ROOT" --out-dir "$EXPORT"
wc -l "$EXPORT"/*.jsonl

echo; echo "############ 2/5  questions derived from existing labels ############"
$PYTHON -m frame_data.derive --out "$GEN/derived_all.jsonl"

echo; echo "############ 3/5  interpolated questions ############"
$PYTHON -m frame_data.interpolate --out "$GEN/interp_all.jsonl"

echo; echo "############ 4/5  questions from our own annotation sheets ############"
$PYTHON -m frame_data.lapchole_questions --csv-dir "$LAPCHOLE_CSV_DIR" \
    --out "$GEN/lapchole_questions.jsonl"
$PYTHON -m frame_data.mesh_questions --csv-dir "$MESH_CSV_DIR" \
    --out "$GEN/mesh_questions.jsonl"

echo; echo "############ 5/5  merge into the official export ############"
$PYTHON orena_sft/merge_generated_frame.py --export "$EXPORT" \
    --files derived_all.jsonl interp_all.jsonl lapchole_questions.jsonl mesh_questions.jsonl
wc -l "$EXPORT/train.jsonl"
echo "=== prep done ==="
