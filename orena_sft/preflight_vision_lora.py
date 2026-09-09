"""Pre-flight for --lora-vision: confirms the vision adapters actually receive
gradients (the use_reentrant=True checkpointing failure mode is silent) and
reports peak memory at the real batch size before a multi-hour run is launched."""

import os
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

SFT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SFT_DIR))
from prompts import build_system_prompt  # noqa: E402
from sft_train_qwen_frame_ddp import (  # noqa: E402
    LINEAR_ATTN_TARGET_MODULES,
    LM_TARGET_MODULES,
    VISION_TARGET_MODULES,
    build_collate_fn,
)

BATCHES = [int(b) for b in os.environ.get("SMOKE_BATCH", "32").split(",")]
MODEL_ID = os.environ.get("SMOKE_MODEL_ID", "Qwen/Qwen3.5-9B")
WANT_VISION = os.environ.get("LORA_VISION", "0") == "1"
WANT_LINEAR_ATTN = os.environ.get("LORA_LINEAR_ATTN", "0") == "1"
MIN_PIXELS = int(os.environ.get("MIN_PIXELS", "0"))

processor = AutoProcessor.from_pretrained(MODEL_ID)
if MIN_PIXELS:
    # Must match the training override: an area floor upscales every frame to
    # ~the same token count, so the sweep sizes the real 2x-resolution load.
    processor.image_processor.size["shortest_edge"] = MIN_PIXELS
model = Qwen3_5ForConditionalGeneration.from_pretrained(
    MODEL_ID, dtype=torch.bfloat16, device_map={"": 0},
)
target_modules = (LM_TARGET_MODULES
                  + (LINEAR_ATTN_TARGET_MODULES if WANT_LINEAR_ATTN else [])
                  + (VISION_TARGET_MODULES if WANT_VISION else []))
model = get_peft_model(model, LoraConfig(
    r=8, lora_alpha=16, lora_dropout=0.05,
    target_modules=target_modules,
    exclude_modules=["patch_embed.proj"] if WANT_VISION else None,
    task_type="CAUSAL_LM",
))
model.print_trainable_parameters()

# Same as training: gradient checkpointing on, non-reentrant, inputs require grad.
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
model.enable_input_require_grads()
model.train()

system_prompt = build_system_prompt(False, style="direct")
# Must be the file the run actually trains on: the sizing sweep is only valid if
# it sees the same frames. Not every cluster has the default combined/ export.
TRAIN_FILE = os.environ.get("SMOKE_TRAIN_FILE",
                            str(SFT_DIR / "sft_export" / "combined" / "train.jsonl"))
if not os.path.exists(TRAIN_FILE):
    raise SystemExit(f"preflight: no train file at {TRAIN_FILE} (set SMOKE_TRAIN_FILE)")
ds = load_dataset("json", data_files={"train": TRAIN_FILE})["train"]
collate = build_collate_fn(processor, system_prompt)

# Size against the worst case, not the first N rows: visual tokens scale with
# source resolution (1280x720 -> 880 tokens vs 960x540 -> 510), so the largest
# frames carry ~1.7x the activations of the smallest. The first rows are all
# heico, which would pass here and then OOM hours in on a lapchole-heavy batch.
if os.environ.get("SMOKE_WORST_CASE", "1") == "1":
    from PIL import Image as _I

    def _px(i):
        try:
            w, h = _I.open(ds[i]["messages"][0]["content"][0]["image"]).size
            return w * h
        except Exception:
            return 0

    probe = sorted(range(len(ds)), key=_px, reverse=True)[:max(BATCHES)]
    print(f"worst-case probe: {max(BATCHES)} largest frames", flush=True)
else:
    probe = list(range(max(BATCHES)))

# Sweep descending batch sizes on one model load: the largest that fits is the
# per-device batch to train with. Gradients are only inspected for the one that
# fits, since an OOM aborts the backward before any grad exists.
out = None
for BATCH in BATCHES:
    model.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    inputs = collate([ds[i] for i in probe[:BATCH]])
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    try:
        o = model(**inputs)
        o.loss.backward()
    except torch.OutOfMemoryError:
        print(f"batch={BATCH:<4} OOM", flush=True)
        del inputs
        continue
    print(f"batch={BATCH:<4} seq_len={inputs['input_ids'].shape[1]:<6} "
          f"vis_tokens={int(inputs['mm_token_type_ids'].sum()):<7} "
          f"peak={torch.cuda.max_memory_allocated()/2**30:.1f} GiB  FITS", flush=True)
    out = o
    break

if out is None:
    print(f"\nRESULT: FAIL - OOM at every batch size tried ({BATCHES})")
    sys.exit(1)
print(f"\n>>> use --batch-size {BATCH}", flush=True)

# Only lora_B is informative on the first step: B is zero-initialised, so dL/dA
# is proportional to it and every lora_A legitimately reads zero here.
grads = {}
for n, p in model.named_parameters():
    if not p.requires_grad or "lora_" not in n:
        continue
    if "visual" in n:
        side = "VISION"
    elif any(m in n for m in LINEAR_ATTN_TARGET_MODULES):
        side = "LINEAR_ATTN"
    else:
        side = "LM"
    mat = "B" if "lora_B" in n else "A"
    g = 0.0 if p.grad is None else p.grad.float().norm().item()
    grads.setdefault((side, mat), []).append(g)

mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")
print(f"\nloss = {out.loss.item():.4f}")
for side in ["LM", "LINEAR_ATTN", "VISION"]:
    for mat in ["A", "B"]:
        gs = grads.get((side, mat), [])
        if not gs:
            continue
        print(f"{side:<12} lora_{mat}  tensors={len(gs):<4} "
              f"mean|grad|={mean(gs):.3e}  zero-grad={sum(g == 0.0 for g in gs)}")
print(f"peak GPU memory = {torch.cuda.max_memory_allocated()/2**30:.1f} GiB")

# Gate on the groups this run actually added: a silently-untrained new group
# would look like a null result rather than a bug.
required = ["LM"] + (["LINEAR_ATTN"] if WANT_LINEAR_ATTN else []) \
                  + (["VISION"] if WANT_VISION else [])
problems = []
for side in required:
    gs = grads.get((side, "B"), [])
    n_zero = sum(g == 0.0 for g in gs)
    if not gs or n_zero:
        problems.append(f"{side}: {n_zero}/{len(gs)} lora_B zero-grad")
ok = not problems
print("\nRESULT:", f"OK - adapters training ({', '.join(required)})" if ok
      else "FAIL - " + "; ".join(problems))
sys.exit(0 if ok else 1)
