"""Turn the hernia-video mesh annotations into FRAME-track questions -- only the ones the
annotation actually entails.

Mesh has zero examples in any published split, so these 12 hernia repairs are the only mesh
footage we have. The other-objects checklist is empty on every row, and the reviewer confirms that is a
genuine absence, not a skipped step: these frames contain no other FOCUS-registered object.
(Tacks and staples are not Clips, and graspers stay connected to the outside, so neither
counts.) The class set is therefore **complete and equal to {Mesh}**, which unlocks every
set-based FRAME format.

Two things shape what gets emitted:

* **Almost every answer is a constant**, because every annotated frame has exactly one mesh
  and nothing else. A constant-answer block is answerable without looking at the image, so
  each kind is capped and spread across videos rather than emitted in full.
* **Half the clicks landed mid-frame** -- mesh is a large sheet that fills the view. Gold
  offers four quadrants and no `center`, so a click near an axis is a coin flip; `--margin`
  drops those from the *spatial* question. Those frames are not wasted: they still answer
  every set-based question.

    python -m frame_data.mesh_questions --csv-dir <dir>
"""
from __future__ import annotations

import argparse, csv, json, random
from collections import Counter, defaultdict
import os
from pathlib import Path

FRAMES = Path(os.environ.get("MESH_FRAMES_DIR", "annotations/public/frames/annotated"))

# every other class in the registry -- used for negative co-occurrence, the only questions
# here whose answer is not a constant
OTHERS = ["Clip", "Sponge", "Specimen", "Specimen bag", "External drain",
          "Silicone loop", "Needle", "Gallstone"]

Q = {
 "list_all":  ("List all foreign objects that are visible in this video frame. Please provide "
               "the class names separated by commas.", "fo_class"),
 "combo":     ("Which combination of foreign object classes is visible in this frame? Please "
               "provide the class names separated by commas.", "fo_class"),
 "single":    ("There is one surgical foreign object visible in the frame. What surgical "
               "foreign object is visible?", "fo_class"),
 "n_classes": ("How many different foreign object classes appear in this frame? Please "
               "provide a number.", "number"),
 "same":      ("Are all visible foreign objects in this frame of the same class? Please "
               "answer with yes or no.", "binary"),
 "count":     ("How many Meshes appear in this frame? Please provide a number.", "number"),
 "position":  ("Where is the center of the Mesh located relative to the image center in this "
               "frame? Please select one answer: top/left; top/right; bottom/left; "
               "bottom/right", "multiple_choice"),
}
COOC = "Do Meshes and {other}s co-occur in this frame? Please answer with yes or no."

# caps per kind. Everything except `cooccur` and `position` has a constant answer, so these
# are deliberately modest -- the point is to teach what mesh looks like, not to inflate rows.
CAPS = {"list_all": 232, "combo": 120, "single": 120, "n_classes": 100,
        "same": 100, "count": 100, "position": 232, "cooccur": 464,
        "cooccur_in_domain": 10**6}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("frame_data/mesh_questions.jsonl"))
    ap.add_argument("--frames", type=Path, default=FRAMES)
    ap.add_argument("--margin", type=float, default=0.05)
    ap.add_argument("--cooccur-per-frame", type=int, default=2)
    ap.add_argument("--in-domain-negatives", type=int, default=600,
                    help="Mesh co-occurrence questions asked about OUR heico/lapchole frames, "
                         "answer 'no'. Without these the token 'Mesh' appears only in hernia "
                         "footage, so the model can answer everything mesh-related from source "
                         "cues alone. Labels are entailed: the KB class sets are complete and "
                         "never contain Mesh.")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    frames = []
    missing = 0
    for fp in sorted(a.csv_dir.glob("*.csv")):
        for r in csv.DictReader(fp.open()):
            img = a.frames / r["video"] / r["image"]
            if not img.exists():
                missing += 1
                continue
            if int(r["count"] or 0) <= 0:
                continue
            frames.append({"video": r["video"], "path": str(img), "time": r["timestamp"],
                           "n": int(r["count"]), "xy": r["points_xy"]})

    rng = random.Random(a.seed)
    rng.shuffle(frames)
    buckets: dict[str, list] = defaultdict(list)

    for i, fr in enumerate(frames):
        base = {"dataset": "hernia_yt", "video": fr["video"], "frame_path": fr["path"],
                "time": fr["time"], "source": "mesh_annotation"}
        add = lambda kind, ans: buckets[kind].append(
            {**base, "question": Q[kind][0], "answer": ans,
             "format": Q[kind][1], "kind": kind})
        add("list_all", "Mesh")
        add("combo", "Mesh")
        add("single", "Mesh")
        add("n_classes", "1")
        add("same", "yes")
        add("count", str(fr["n"]))
        if fr["n"] == 1 and fr["xy"]:
            x, y = (float(v) for v in fr["xy"].split(",")[:2])
            if abs(x - .5) > a.margin and abs(y - .5) > a.margin:
                add("position", ("top" if y < .5 else "bottom") + "/" +
                    ("left" if x < .5 else "right"))
        # negative co-occurrence: rotate the partner class so the question text varies
        for k in range(a.cooccur_per_frame):
            other = OTHERS[(i * a.cooccur_per_frame + k) % len(OTHERS)]
            buckets["cooccur"].append({**base, "question": COOC.format(other=other),
                                       "answer": "no", "format": "binary", "kind": "cooccur"})

    if a.in_domain_negatives:
        from frame_data.interpolate import DEFAULT_ROOT, frame_path
        from frame_data.kb import build_kb
        kb, _ = build_kb()
        root = Path(DEFAULT_ROOT)
        pool = [(k, f.classes) for k, f in kb.items() if f.classes]
        rng.shuffle(pool)
        by_video, made = Counter(), 0
        per_video_cap = max(1, a.in_domain_negatives // 40)
        for (dsname, video, t), classes in pool:
            if made >= a.in_domain_negatives:
                break
            if by_video[video] >= per_video_cap:
                continue
            p = frame_path(root, dsname, video, t)
            if not p.exists():
                continue
            other = sorted(classes)[made % len(classes)]      # a class that IS in the frame
            buckets["cooccur_in_domain"].append(
                {"dataset": dsname, "video": video, "frame_path": str(p), "time": t,
                 "source": "mesh_in_domain_negative",
                 "question": COOC.format(other=other), "answer": "no",
                 "format": "binary", "kind": "cooccur_in_domain"})
            by_video[video] += 1
            made += 1

    rows = []
    for kind, rs in buckets.items():
        rng.shuffle(rs)
        rows += rs[:CAPS.get(kind, len(rs))]
    rng.shuffle(rows)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    print(f"{len(frames)} annotated frames over {len({f['video'] for f in frames})} videos "
          f"-> {len(rows)} questions   (missing images: {missing})")
    print(f"{'kind':12s} {'emitted':>8s} {'available':>10s}  answer")
    ans = {"list_all": "Mesh", "combo": "Mesh", "single": "Mesh", "n_classes": "1",
           "same": "yes", "count": "1", "cooccur": "no", "position": "4-way (varies)",
           "cooccur_in_domain": "no  <- on OUR frames, mesh absent"}
    for kind in sorted(buckets):
        got = sum(1 for r in rows if r["kind"] == kind)
        print(f"{kind:12s} {got:8d} {len(buckets[kind]):10d}  {ans[kind]}")
    print("  formats:", dict(Counter(r["format"] for r in rows).most_common()))
    nd = sum(1 for r in rows if r["kind"] == "cooccur_in_domain")
    print(f"\n{nd} in-domain negatives put the token 'Mesh' on OUR frames with answer 'no',")
    print("so mesh questions are no longer answerable from source cues alone. Still absent:")
    print("a frame where Mesh is asked for and some OTHER class is the answer -- gold has no")
    print("empty answers, so that example cannot be built. Watch objrec OOD on the platform.")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
