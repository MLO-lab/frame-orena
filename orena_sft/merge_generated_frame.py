"""Append our generated FRAME questions to the built SFT train.jsonl.

The generated rows (frame_data/derived_all.jsonl, interp_all.jsonl) already carry an
absolute `frame_path` built with the same convention the builder uses --
`<dataset>/frames/<video stem>/frame{idx:07d}.jpg` -- so no path re-derivation is needed;
we verify existence instead of trusting it.

A2 (`derived`)  labels are ENTAILED by gold: the knowledge base closed with zero
                contradictions over 15,212 frames, so these carry no new label error.
A3 (`interpolated`) labels are INFERRED between two agreeing annotations, and every row
                carries `label_confidence` from a leave-one-out bracket test. ~8% of the
                counting rows are expected wrong, concentrated at counts >= 4. Use
                --min-confidence to trade rows for cleanliness.

    python merge_generated_frame.py --export sft_export/combined_alldata_test
"""
from __future__ import annotations
import argparse, json, random
from collections import Counter
import os
from pathlib import Path

GEN_DIR = Path(os.environ.get("FRAME_GEN_DIR", "generated"))


def to_chat(r: dict, qid: str, proc: dict[str, str]) -> dict:
    vid = f"{r['dataset']}/{r['video']}"
    return {
        "qID": qid,
        "source_dataset": r["dataset"],
        "videoID": vid,
        "procedure_type": proc.get(vid, "unknown"),
        "primary_capability": r.get("group") or ("aggregation" if r["format"] in
                                                 ("number", "binary") else "object_recognition"),
        "secondary_capabilities": [],
        "format": r["format"],
        "generated_by": r.get("source", "generated"),      # provenance stays in the row
        "label_confidence": r.get("label_confidence", 1.0),
        "messages": [
            {"role": "user", "content": [{"type": "image", "image": r["frame_path"]},
                                         {"type": "text", "text": r["question"]}]},
            {"role": "assistant", "content": [{"type": "text", "text": str(r["answer"])}]},
        ],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", type=Path, required=True)
    ap.add_argument("--files", nargs="+", default=["derived_all.jsonl", "interp_all.jsonl"])
    ap.add_argument("--min-confidence", type=float, default=0.0,
                    help="drop generated rows below this label confidence (A3 only; A2 is 1.0)")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    train = a.export / "train.jsonl"
    base = [json.loads(l) for l in train.open()]
    proc = {r["videoID"]: r["procedure_type"] for r in base}
    seen_q = {(r["videoID"], r["messages"][0]["content"][1]["text"],
               r["messages"][0]["content"][0]["image"]) for r in base}

    added, stats = [], Counter()
    for fn in a.files:
        for i, line in enumerate((GEN_DIR / fn).open()):
            r = json.loads(line)
            if r.get("label_confidence", 1.0) < a.min_confidence:
                stats["dropped_low_confidence"] += 1
                continue
            if not Path(r["frame_path"]).exists():
                stats["dropped_missing_frame"] += 1
                continue
            key = (f"{r['dataset']}/{r['video']}", r["question"], r["frame_path"])
            if key in seen_q:                      # never duplicate a real gold question
                stats["dropped_duplicate_of_gold"] += 1
                continue
            seen_q.add(key)
            added.append(to_chat(r, f"gen-{fn[:3]}-{i:06d}", proc))
            stats[f"kept_{r.get('source','?')}"] += 1

    merged = base + added
    random.Random(a.seed).shuffle(merged)
    out = a.export / "train.jsonl"
    with out.open("w") as f:
        for r in merged:
            f.write(json.dumps(r) + "\n")

    print(f"base {len(base)} + generated {len(added)} = {len(merged)} rows -> {out}")
    for k, v in sorted(stats.items()):
        print(f"  {k:28s} {v}")
    print("  format mix:", dict(Counter(r["format"] for r in merged).most_common()))


if __name__ == "__main__":
    main()
