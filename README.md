# MLO-Lab — ORena FOCUS 2026, FRAME track

Training code for the FRAME submission: **Qwen3.5-9B with a rank-8 vision LoRA**,
trained on 34,006 single-frame examples.

## Layout

```
annotations/       our frame-level annotations + the VQA pairs generated from them
frame_data/        question generation
orena_sft/         dataset builder, prompts, preflight, trainer
slurm/             the two jobs that run the pipeline
```

The annotation data is not committed. It comes in two halves.

**Public** — HeiCo and hernia questions, the hernia annotation sheets, and the hernia
frames. Published as a Hugging Face dataset:

```bash
python annotations/fetch.py
```

which downloads
[`Machine-Learning-Oncology/orena-frame-annotations`](https://huggingface.co/datasets/Machine-Learning-Oncology/orena-frame-annotations)
into `annotations/public/`.

**Private** — the LapChole questions and sheets. Not hosted: the LapChole-FOCUS data usage
agreement forbids publishing any derivative until the organisers release that dataset.
Available **on request** as a small archive; unpack it into `annotations/private/`:

```bash
mkdir -p annotations/private
tar xzf orena-frame-lapchole-annotations_*.tar.gz \
    -C annotations/private --strip-components=1
```

Either half is optional — the pipeline uses whatever is present. The expected layout is:

```
annotations/
├── public/     vqa/*.jsonl, raw/hernia_mesh_annotations/, frames/, SOURCES.md
└── private/    vqa/lapchole.jsonl, raw/lapchole_annotations/, raw/crop_boxes.json
```

See `annotations/README.md` for the schemas and the reasoning behind the split.

## Requirements

* Python 3.12, 2× H100 80 GB for training; CPU only for data preparation.
* `pip install -r requirements.txt` — install `torch` from the CUDA 12.8 index first.
* `Qwen/Qwen3.5-9B` cached in `$HF_HOME`. Training runs with `HF_HUB_OFFLINE=1`.
* The FOCUS dataset root, containing `heico/` and `lapchole/`.

## Configuration

Both jobs are configured through environment variables. No paths are hardcoded.

| variable | required | meaning |
|---|---|---|
| `PYTHON` | yes | interpreter of the environment built from `requirements.txt` |
| `ORENA_DATA_ROOT` | yes | directory holding `heico/` and `lapchole/` |
| `HF_HOME` | training | cache containing `Qwen/Qwen3.5-9B` |
| `EXPORT` | no | dataset output directory (default `./sft_export`) |
| `GEN` | no | generated-question directory (default `./generated`) |
| `LAPCHOLE_CSV_DIR` | no | LapChole sheets (default `annotations/private/raw/lapchole_annotations`) |
| `MESH_CSV_DIR` | no | hernia sheets (default `annotations/public/raw/hernia_mesh_annotations`) |
| `MESH_FRAMES_DIR` | no | hernia frames (default `annotations/public/frames`) |
| `RUN_NAME` | no | checkpoint directory name |
| `NPROC` | no | GPUs per node (default `2`) |

## 1. Build the dataset

```bash
export PYTHON=/path/to/venv/bin/python
export ORENA_DATA_ROOT=/path/to/orena

python annotations/fetch.py
sbatch slurm/1_prepare_data.sbatch
```

Five steps, writing `$EXPORT/train.jsonl`:

| step | output | rows |
|---|---|---:|
| 1 | official export | 20,000 |
| 2 | derived from existing labels → `derived_all.jsonl` | 7,022 |
| 3 | interpolated between labelled anchors → `interp_all.jsonl` | 3,744 |
| 4 | our annotation sheets → `lapchole_questions.jsonl`, `mesh_questions.jsonl` | 1,334 + 1,950 |
| 5 | merge and deduplicate → **`train.jsonl`** | **34,006** |

If the private half is absent, the LapChole questions are simply not generated and the
merge produces a correspondingly smaller file.

## 2. Train

```bash
export PYTHON=/path/to/venv/bin/python
export HF_HOME=/path/to/hf_cache

sbatch slurm/2_train.sbatch
```

| | |
|---|---|
| base model | `Qwen/Qwen3.5-9B` |
| adapter | LoRA r=8, α=16, applied to the language tower **and the vision tower** |
| prompt style | `direct` |
| batch | per-device 4 × grad-accum 4 × 2 GPUs = effective 32 |
| schedule | lr 1e-4, bf16, seed 42, 4 epochs = 4,252 steps |
| checkpoints | every 533 steps → `checkpoints/$RUN_NAME/` |

The job runs `orena_sft/preflight_vision_lora.py` first, which checks that gradients reach
the vision tower before committing to the full run.

### Selected checkpoint

The run trains for 4 epochs, and **`checkpoint-3198` (epoch 3.0) is the one submitted** —
not the final checkpoint.

## 3. Package for submission

Merge the adapter from `checkpoints/$RUN_NAME/checkpoint-3198` into the base model, then
build the container.

## Notes

* `--lora-vision` is required: the vision tower is adapted, not frozen. This differs from
  the SEGMENT track, where only the language tower carries the adapter.
* `orena_sft/build_frame_sft_dataset.py` defaults `--root-dir` to a path that may not exist
  on your cluster; set `ORENA_DATA_ROOT` or pass `--root-dir` explicitly.

## Acknowledgements

We are grateful to Dr. Todd S. Harris of California Hernia Specialists, who kindly allowed us
to use twelve of his publicly available laparoscopic hernia-repair videos in this work. His
permission covers non-commercial research use and the release of the annotated frames; the
original videos remain his and are not redistributed. The video titles and links are listed
in `annotations/public/SOURCES.md`.
