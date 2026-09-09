"""A3: temporal label interpolation, aimed at multi-object counting.

Between two consecutive annotations of the *same question on the same video* whose gold
answers **agree**, the frames in between inherit that answer. This is not assumed — it is
bracketed by two independent human annotations, and `--verify` measures how often the
bracket actually holds using held-out published labels.

Why it targets counting: derivation (A2) cannot produce multi-object counting data at all,
because 100% of frames with >=2 objects already carry the instance-count question — that is
how their count is known. Interpolation instead multiplies those frames into new *images*
with the same count, and high-count states happen to be annotated more densely, so the
output distribution comes out flatter than the published one.

    python -m frame_data.interpolate --verify
    python -m frame_data.interpolate --out frame_data/interp_train.jsonl --splits train
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
import os
from pathlib import Path

from focus import DatasetSplit, FocusDataset, Track
from focus.config import DATASET_BASE_FPS

DEFAULT_ROOT = os.environ.get("ORENA_DATA_ROOT", "/projects/datasets_ML/orena")

# Measured bracket accuracy for `number`, by the gold count being interpolated (from the
# leave-one-out triple test at a <=5s span, so conservative for the 3s default). Emitted per
# row as `label_confidence` so training can down-weight uncertain labels rather than trust
# them equally. This matters more than the raw rate suggests: the metric is EXACT match and
# the model's dominant failure is off-by-one (36% of counting questions), so our label noise
# sits in exactly the dimension being scored.
NUMBER_CONFIDENCE = {1: 0.959, 2: 0.966, 3: 0.929, 4: 0.851, 5: 0.889}
NUMBER_CONFIDENCE_HIGH = 0.792          # counts >= 6
BINARY_CONFIDENCE = 0.997               # measured on 285 bracket triples
# Agreement between endpoints decays as the count rises (0.58 at gold 1 -> 0.23 at gold 7),
# so `number` gets a tighter bracket than the recognition formats.
# Bracket span per format. Tighter brackets carry measurably less label noise --
# `number` accuracy is 0.952 at <=2s, 0.941 at <=3s, 0.930 at <=5s, 0.909 at <=10s -- so
# `number` defaults to 3s, roughly halving wrong labels versus 5s while keeping 59% of the
# multi-object examples. Override with --max-gap.
MAX_GAP = {"number": 3.0, "binary": 10.0, "fo_class": 10.0,
           "multiple_choice": 10.0, "open_ended": 10.0}


def load_rows(datasets=("heico", "lapchole"), splits=("train", "test")):
    rows = []
    for ds in datasets:
        for sp in splits:
            split = DatasetSplit.TRAIN if sp == "train" else DatasetSplit.TEST
            for req, ref in FocusDataset(ds, split, Track.FRAME):
                rows.append(dict(dataset=ds, split=sp, qID=str(req.qID), video=req.videoID,
                                 time=float(req.start_time), question=req.question.strip(),
                                 answer=ref.answer.strip(), format=ref._format,
                                 leaf=ref.primary.value, group=ref.primary.group.value))
    return rows


def _series(rows):
    """{(dataset, video, question): [rows sorted by time]}"""
    s = defaultdict(list)
    for r in rows:
        s[(r["dataset"], r["video"], r["question"])].append(r)
    for v in s.values():
        v.sort(key=lambda r: r["time"])
    return s


def verify(rows, formats=None):
    """Leave-one-out test of the bracketing assumption on real published labels.

    For every consecutive triple (a, b, c) of the same question whose *outer* answers agree
    within the gap budget, predict b from the bracket and compare against b's own gold.
    """
    stats = defaultdict(lambda: [0, 0])
    by_count = defaultdict(lambda: [0, 0])
    for (ds, _v, _q), seq in _series(rows).items():
        for i in range(1, len(seq) - 1):
            a, b, c = seq[i - 1], seq[i], seq[i + 1]
            fmt = b["format"]
            if formats and fmt not in formats:
                continue
            if a["answer"] != c["answer"]:
                continue
            if c["time"] - a["time"] > MAX_GAP.get(fmt, 10.0):
                continue
            ok = int(b["answer"] == a["answer"])
            stats[fmt][0] += ok
            stats[fmt][1] += 1
            if fmt == "number" and b["answer"].isdigit():
                g = min(int(b["answer"]), 6)
                by_count[g][0] += ok
                by_count[g][1] += 1
    return dict(stats), dict(by_count)


def frame_path(root: Path, dataset: str, video: str, t: float) -> Path:
    fps = DATASET_BASE_FPS[dataset]
    idx = int(round(t * fps))
    return root / dataset / "frames" / Path(video).stem / f"frame{idx:07d}.jpg"


def generate(rows, root: Path, fps_out=2.0, formats=None, min_number=1, check_files=True):
    out, missing = [], 0
    for (ds, video, question), seq in _series(rows).items():
        for a, b in zip(seq, seq[1:]):
            fmt = a["format"]
            if formats and fmt not in formats:
                continue
            if a["answer"] != b["answer"]:
                continue                       # endpoints disagree: something entered/left
            gap = b["time"] - a["time"]
            if gap <= 0 or gap > MAX_GAP.get(fmt, 10.0):
                continue
            if fmt == "number" and a["answer"].isdigit() and int(a["answer"]) < min_number:
                continue
            step = 1.0 / fps_out
            t = a["time"] + step
            while t < b["time"] - 1e-6:
                p = frame_path(root, ds, video, t)
                if check_files and not p.exists():
                    missing += 1
                else:
                    if fmt == "number" and a["answer"].isdigit():
                        conf = NUMBER_CONFIDENCE.get(int(a["answer"]), NUMBER_CONFIDENCE_HIGH)
                    elif fmt == "binary":
                        conf = BINARY_CONFIDENCE
                    else:
                        conf = None
                    out.append(dict(dataset=ds, video=video, time=round(t, 3),
                                    frame_path=str(p), question=question,
                                    answer=a["answer"], format=fmt, leaf=a["leaf"],
                                    group=a["group"], source="interpolated",
                                    label_confidence=conf,
                                    anchors=[a["qID"], b["qID"]], gap=round(gap, 2)))
                t += step
    return out, missing


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["train", "test"])
    ap.add_argument("--formats", nargs="+", default=["number", "binary"],
                    help="default targets the aggregation bucket")
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--min-number", type=int, default=1,
                    help="emit `number` items only when the gold count is >= this")
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--out")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--no-check-files", action="store_true")
    ap.add_argument("--binary-multi-only", action="store_true",
                    help="keep only the MULTI-OBJECT binary items: 'are all FOs of the same "
                         "class?'->no and 'do X and Y co-occur?'->yes. Both require perceiving "
                         "two or more objects, and both are where the model is weak -- binary "
                         "scores 0.968 on single-class frames but 0.518 on two-class frames. "
                         "The single-object remainder is near-free and only dilutes the bucket.")
    ap.add_argument("--max-gap", type=float,
                    help="override the bracket span for `number` (default 5s). Tighter "
                         "brackets carry less label noise: measured 0.952 correct at <=2s, "
                         "0.941 at <=3s, 0.930 at <=5s, 0.909 at <=10s")
    a = ap.parse_args()
    if a.max_gap:
        MAX_GAP["number"] = a.max_gap

    rows = load_rows(splits=tuple(a.splits))
    print(f"published FRAME questions: {len(rows):,}")

    if a.verify:
        st, byc = verify(rows)
        print("\n=== bracket test: predict a held-out middle annotation from its neighbours ===")
        for fmt, (ok, n) in sorted(st.items(), key=lambda kv: -kv[1][1]):
            if n:
                print(f"  {fmt:16s} n={n:6,}  correct {ok/n:.4f}")
        print("\n  `number`, by the held-out gold count:")
        for g in sorted(byc):
            ok, n = byc[g]
            print(f"    count {'>=6' if g == 6 else g:>3}  n={n:5,}  correct {ok/n:.4f}")
        return

    rec, missing = generate(rows, Path(a.root), fps_out=a.fps, formats=set(a.formats),
                            min_number=a.min_number, check_files=not a.no_check_files)
    if a.binary_multi_only:
        def _hard(r):
            if r["format"] != "binary":
                return True
            same = "same class" in r["question"].lower()
            ans = r["answer"].strip().lower()
            return (ans == "no") if same else (ans == "yes")
        before = sum(1 for r in rec if r["format"] == "binary")
        rec = [r for r in rec if _hard(r)]
        print(f"binary kept (multi-object only): "
              f"{sum(1 for r in rec if r['format']=='binary'):,} of {before:,}")
    print(f"\ninterpolated rows: {len(rec):,}   (missing frame files skipped: {missing:,})")
    print("by format:", dict(Counter(r["format"] for r in rec)))
    num = [int(r["answer"]) for r in rec if r["format"] == "number" and r["answer"].isdigit()]
    if num:
        c = Counter(num)
        tot = len(num)
        print("\n`number` answers:", {k: c[k] for k in sorted(c)})
        print("       as shares:", {k: round(c[k] / tot, 3) for k in sorted(c)})
        print(f"  gold >= 2 : {sum(v for k, v in c.items() if k >= 2):,} ({sum(v for k,v in c.items() if k>=2)/tot:.0%})")
        print(f"  gold >= 3 : {sum(v for k, v in c.items() if k >= 3):,}")
    print("distinct new frames:", f"{len({(r['dataset'], r['video'], r['time']) for r in rec}):,}")

    if a.out:
        with open(a.out, "w") as fh:
            for r in rec:
                fh.write(json.dumps(r) + "\n")
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
