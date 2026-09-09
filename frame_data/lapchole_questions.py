"""FRAME-track questions from the Job-1 lapchole pass (Gallstone, Absorbable Hemostatic Agent).

Why these two classes are worth a dedicated pass: across the whole gold FRAME train split
(heico + lapchole, 15,212 annotated frames) **Gallstone appears in 51 frames from 6 videos and
Absorbable Hemostatic Agent in none at all**. Only 29 gold questions mention either. So this
annotation is not an increment on existing supervision -- for AHA it *is* the supervision.

What the annotation entails, and what it does not
-------------------------------------------------
The reviewer swept every extracted frame of the video looking for **both** target classes, so
for a video whose CSV we received, the count of Gallstone and of AHA is known on every frame,
including the frames that are absent from the CSV (the CSV is positives-only). That gives us
counts, quadrants, and co-occurrence *between the two targets* in both directions.

It does **not** give us the complete class set. The other-objects checklist was optional in
Job 1 ("only fill it in for frames you have already stopped on"), so a tick proves presence
but a blank proves nothing. That rules out `fo_class` -- which is exact set equality, and the
single highest-value format on this track. If the annotators confirm they ticked the list
exhaustively on every frame they stopped on, `--trust-checklist` turns those on.

Two shortcuts are deliberately closed:

* **class-token shortcut.** These 70 videos are the only place "Gallstone"/"Absorbable
  Hemostatic Agent" would appear, so the class name alone would predict "yes". In-domain
  negatives ask the same co-occurrence questions about gold heico/lapchole frames, answer no.
* **video-identity shortcut.** Frames from the same 70 videos with no target present supply
  the negative half, so the video itself does not predict the answer either.

Geometry: the annotator viewed autocropped frames (38 of 70 videos, median 11% of pixels
removed) while gold frames are the raw video frame. Clicks are therefore mapped back through
`crop_boxes.json` into full-frame coordinates before the quadrant is computed, and the frames
written for training are re-extracted **uncropped** -- otherwise "was cropped" would itself be
a cue, and every quadrant near an axis would be wrong.

    python -m frame_data.lapchole_questions --csv-dir <dir> --extract
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
import os
from pathlib import Path

ANNOT = Path(os.environ.get("ORENA_ANNOT_DIR", "annotations/private/raw"))
VIDEOS = Path(os.environ.get("ORENA_DATA_ROOT", "/projects/datasets_ML/orena")) / "lapchole/videos"
OUT_FRAMES = Path(os.environ.get("ORENA_ANNOT_FRAMES", "annot_frames"))

BASE_FPS = 30.0                       # DATASET_BASE_FPS["lapchole"]; every source video checked
GALL, AHA = "Gallstone", "Absorbable Hemostatic Agent"
TARGETS = [GALL, AHA]

# surface forms exactly as gold writes them -- gold lowercases the second word ("Specimen
# bags", "External drains"), so matching that keeps our rows off a spurious style signal
PLURAL = {GALL: "Gallstones", AHA: "Absorbable Hemostatic Agents", "Clip": "Clips",
          "Sponge": "Sponges", "Specimen": "Specimens", "Specimen bag": "Specimen bags",
          "Specimen Bag": "Specimen bags", "External drain": "External drains",
          "External Drain": "External drains", "Needle": "Needles",
          "Silicone loop": "Silicone loops", "Silicone Loop": "Silicone loops",
          "Mesh": "Meshes"}

COUNT_Q = "How many {p} appear in this frame? Please provide a number."
POS_Q = ("Where is the center of the {s} located relative to the image center in this frame? "
         "Please select one answer: top/left; top/right; bottom/left; bottom/right")
COOC_Q = "Do {a} and {b} co-occur in this frame? Please answer with yes or no."
LIST_Q = ("List all foreign objects that are visible in this video frame. Please provide the "
          "class names or answer with none.")
SINGLE_Q = ("There is one surgical foreign object visible in the frame. What surgical foreign "
            "object is visible in this video frame? Please provide a class name.")
NCLS_Q = "How many different foreign object classes appear in this frame? Please provide a number."
SAME_Q = ("Are all visible foreign objects in this frame of the same class? Please answer with "
          "yes or no.")

# Everything except `count` and `position` has a low-entropy answer, so caps keep the mix from
# being dominated by questions that are answerable without looking.
# The negative kinds are budgeted rather than trimmed at random: `same` is a real
# multi-object judgement and is taken in full; the in-domain rows are what stop the class
# token alone predicting "yes", so they get a fixed floor; generic in-video negatives fill
# the remainder. `--max-no-ratio` is then only a safety net.
CAPS = {"count": 4000, "position": 1500, "cooc_targets_yes": 400, "cooc_targets_no": 150,
        "cooc_checklist": 1200, "cooc_in_domain": 200, "same": 130,
        "list_all": 800, "single": 400, "n_classes": 400}

# When binary has to be trimmed for balance, drop the cheapest negatives first. `same` is a
# genuine multi-object judgement; a co-occurrence "no" against a class the frame never had is
# much closer to a class-token lookup.
DROP_ORDER = ["cooc_in_domain", "cooc_targets_no", "cooc_checklist", "same"]

# Per-answer ceilings, set from what our rows do to the POOLED answer distribution rather
# than from what the annotation happens to contain. Two templates needed it:
#
#  * `count` -- 55% of our gallstone frames hold exactly one stone against gold's 24%, so
#    shipping them all pushes the pooled "1" rate to 31.7%. Counting is the weakest part of
#    this track; a stronger "answer 1" prior is the opposite of help. 145 holds the pool at
#    27%. Answers >= 3 are kept in full: 9-13 barely exist in gold and are the whole point.
#  * `same` -- every row we can entail is "no", which flips a template gold answers "yes"
#    58% of the time into one that is 59% "no". Capped to keep the pool near even.
#
# The frames are not lost: they still carry position, co-occurrence and same-class rows.
ANSWER_CAPS = {("count", "1"): 145}


def secs(ts: str) -> float:
    h, m, s = (float(x) for x in ts.replace("-", ":").split(":"))
    return h * 3600 + m * 60 + s


def load_csvs(csv_dir: Path):
    """-> {video: {image: {"marks": {cls: [(x, y), ...]}, "co": [str]}}}, one entry per
    reviewed video. A delivered CSV means the reviewer marked the video complete in the
    status sheet, which is what licenses treating unlisted frames as target-free."""
    vids: dict[str, dict] = defaultdict(dict)
    dupes = Counter()
    for fp in sorted(csv_dir.glob("*.csv")):
        vids[fp.stem]                      # a header-only CSV means "swept, found nothing" --
        for r in csv.DictReader(fp.open()):   # seed from the filename or it vanishes here
            rec = vids[r["video"]].setdefault(r["image"], {"marks": defaultdict(list), "co": []})
            for o in filter(None, (r.get("other_objects") or "").split(";")):
                if o not in rec["co"]:
                    rec["co"].append(o)
            cls, n = (r.get("target_class") or "").strip(), int(r.get("count") or 0)
            if not cls or n <= 0:
                continue
            pts = [tuple(float(v) for v in p.split(","))
                   for p in filter(None, (r.get("points_xy") or "").split(";"))]
            if cls in rec["marks"]:                 # two CSVs for one video, or a hand edit --
                dupes[r["video"]] += 1             # silently keeping the last would be wrong
            rec["marks"][cls] = pts if pts else [(None, None)] * n
    for v, n in dupes.items():
        print(f"  ! {v}: {n} duplicate (frame, class) rows -- is this video annotated twice?")
    return vids


def uncrop(xy, box):
    """Click in autocropped view -> position in the full video frame, both normalised."""
    if xy[0] is None or box is None:
        return None
    x0, y0, x1, y1 = box["crop_box"]
    w, h = box["source_wh"]
    return ((x0 + xy[0] * (x1 - x0)) / w, (y0 + xy[1] * (y1 - y0)) / h)


def extract(need: dict[str, set[float]], quality: int = 90) -> int:
    """Re-decode the frames we actually use, uncropped, named like the gold extraction."""
    import decord
    from PIL import Image

    written = 0
    for video, times in sorted(need.items()):
        src = VIDEOS / f"{video}.mp4"
        if not src.exists():
            print(f"  ! no source video for {video}")
            continue
        d = OUT_FRAMES / video
        d.mkdir(parents=True, exist_ok=True)
        vr = decord.VideoReader(str(src))
        fps = float(vr.get_avg_fps()) or BASE_FPS
        if abs(fps - BASE_FPS) > 1e-6:      # filenames encode BASE_FPS; a mismatch would put
            print(f"  ! {video}: {fps} fps, expected {BASE_FPS} -- skipping")   # the wrong
            continue                                          # image behind every question
        want = {}
        for t in times:
            i = min(len(vr) - 1, int(round(t * fps)))
            p = d / f"frame{i:07d}.jpg"
            if not p.exists():
                want[i] = p
        idx = sorted(want)
        for s0 in range(0, len(idx), 64):
            chunk = idx[s0:s0 + 64]
            for fr, i in zip(vr.get_batch(chunk).asnumpy(), chunk):
                Image.fromarray(fr).save(want[i], quality=quality)
                written += 1
        print(f"  {video[:40]:42s} +{len(idx):4d} frames ({len(times)} needed)")
    return written


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("frame_data/lapchole_questions.jsonl"))
    ap.add_argument("--annot", type=Path, default=ANNOT)
    ap.add_argument("--margin", type=float, default=0.05,
                    help="drop position questions whose click sits this close to an axis")
    ap.add_argument("--neg-per-video", type=int, default=30,
                    help="target-free frames sampled per video for the negative half")
    ap.add_argument("--in-domain-negatives", type=int, default=900)
    ap.add_argument("--trust-checklist", action="store_true",
                    help="treat the other-objects checklist as COMPLETE on annotated frames, "
                         "which unlocks fo_class. Only pass this if the annotators confirm "
                         "they ticked it exhaustively -- a blank list is otherwise ambiguous.")
    ap.add_argument("--max-no-ratio", type=float, default=2.5,
                    help="cap binary 'no' rows at this multiple of the 'yes' rows. Gold "
                         "lapchole binary is 233 yes / 229 no; the entailment here is "
                         "lopsided towards 'no', so unbalanced output would teach the prior "
                         "rather than the object.")
    ap.add_argument("--keep-empty-videos", action="store_true",
                    help="include videos whose CSV has no positive at all. Off by default: "
                         "such a CSV is indistinguishable from a review saved half-way, and "
                         "it contributes nothing but negatives.")
    ap.add_argument("--extract", action="store_true", help="write the uncropped frames too")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    boxes = json.loads((a.annot / "crop_boxes.json").read_text())
    vids = load_csvs(a.csv_dir)
    if not vids:
        raise SystemExit(f"no CSVs in {a.csv_dir}")

    rng = random.Random(a.seed)
    buckets: dict[str, list] = defaultdict(list)
    pair_yes: Counter = Counter()      # (target, other) -> how often we answer "yes"
    empty_pool: list = []
    need: dict[str, set[float]] = defaultdict(set)
    stats = Counter()

    for video, rows in sorted(vids.items()):
        box = boxes.get(video)
        all_frames = sorted(p.name for p in (a.annot / video).glob("*.jpg"))
        if not all_frames:
            print(f"  ! no extracted frames for {video}, skipping")
            continue
        if not any(rows.get(f, {}).get("marks") for f in all_frames) and not a.keep_empty_videos:
            # a CSV with no positive is either a genuinely empty video or a review that was
            # saved early. We cannot tell them apart, and the two differ only in the
            # negatives -- exactly what an unfinished review would poison. Drop it.
            print(f"  - {video}: no positives, skipped (--keep-empty-videos to include)")
            stats["videos_empty"] += 1
            continue
        stats["videos"] += 1

        # `vid` is bound per call, not captured: the negatives are emitted after this loop
        # ends, and a late-bound closure would file every one of them under the last video.
        def emit(kind, image, question, answer, fmt, extra=None, vid=None):
            vid = vid or video
            t = secs(image.rsplit("__", 1)[-1][:-4])
            need[vid].add(t)
            buckets[kind].append({"dataset": "lapchole", "video": f"{vid}.mp4",
                                  "frame_path": None, "video_stem": vid, "time": t,
                                  "source": "lapchole_annotation", "question": question,
                                  "answer": answer, "format": fmt, "kind": kind, **(extra or {})})

        for image in all_frames:
            rec = rows.get(image)
            marks = rec["marks"] if rec else {}
            present = [c for c in TARGETS if marks.get(c)]
            stats["frames_reviewed"] += 1

            if not present:
                # entailed: the sweep covered both targets, so this frame has neither
                stats["frames_negative"] += 1
                continue
            stats["frames_positive"] += 1

            for c in present:
                pts = marks[c]
                stats[f"marks_{c}"] += len(pts)
                emit("count", image, COUNT_Q.format(p=PLURAL[c]), str(len(pts)), "number")
                if len(pts) == 1:
                    p = uncrop(pts[0], box)
                    if p and abs(p[0] - .5) > a.margin and abs(p[1] - .5) > a.margin:
                        emit("position", image, POS_Q.format(s=c),
                             ("top" if p[1] < .5 else "bottom") + "/" +
                             ("left" if p[0] < .5 else "right"), "multiple_choice")

            # both targets were swept, so this one is entailed either way
            emit("cooc_targets_yes" if len(present) == 2 else "cooc_targets_no", image,
                 COOC_Q.format(a=PLURAL[GALL], b=PLURAL[AHA]),
                 "yes" if len(present) == 2 else "no", "binary")

            # a tick proves presence; a blank proves nothing, so only "yes" here
            ticks = [o for o in (rec["co"] if rec else []) if o in PLURAL]
            for c in present:
                for o in ticks:
                    emit("cooc_checklist", image, COOC_Q.format(a=PLURAL[c], b=PLURAL[o]),
                         "yes", "binary")
                    pair_yes[(c, o)] += 1
            for i2 in range(len(ticks)):                  # pairs among the ticks themselves --
                for o2 in ticks[i2 + 1:]:                 # both observed, so also entailed
                    emit("cooc_checklist", image,
                         COOC_Q.format(a=PLURAL[ticks[i2]], b=PLURAL[o2]), "yes", "binary")

            # Two observed classes settle this outright: whatever else may be in the frame
            # unobserved can only add classes, never make them all the same.
            if len(set(present) | set(ticks)) >= 2:
                emit("same", image, SAME_Q, "no", "binary")

            if a.trust_checklist and rec is not None:
                classes = sorted({*present, *(o for o in rec["co"] if o in PLURAL)})
                emit("list_all", image, LIST_Q, ", ".join(classes), "fo_class")
                emit("n_classes", image, NCLS_Q, str(len(classes)), "number")
                if len(classes) == 1:
                    emit("single", image, SINGLE_Q, classes[0], "fo_class")

        # negative half: target-free frames from the same videos, so video identity alone
        # cannot answer a gallstone/AHA question. The pairs are chosen after every video is
        # in, so they can mirror the pairs we assert "yes" for.
        empty = [f for f in all_frames if not any(rows.get(f, {}).get("marks", {}).get(c)
                                                  for c in TARGETS)]
        rng.shuffle(empty)
        empty_pool += [(video, f) for f in empty[:a.neg_per_video]]
        emit_fn = emit                       # same function object every iteration; `vid`
                                             # is what makes it safe to call later

    # Gold asks the SAME pair both ways -- "Do Clips and Sponges co-occur?" is 75 yes and
    # 69 no -- so the pair alone never settles it. Our positives are one-sided by
    # construction (196 of the 285 are Gallstone x Clip), which would make the pair itself
    # predictive. A frame where the target is absent answers "no" for ANY partner, since
    # co-occurrence needs both, and target absence IS entailed by the sweep. So the same
    # pairs get their negatives.
    if empty_pool:
        want = list(pair_yes.elements()) or [(GALL, "Clip")]
        rng.shuffle(want)
        rng.shuffle(empty_pool)
        for i, (vid, image) in enumerate(empty_pool):
            c, o = want[i % len(want)]
            emit_fn("cooc_targets_no", image, COOC_Q.format(a=PLURAL[c], b=PLURAL[o]),
                    "no", "binary", vid=vid)

    # in-domain negatives: the class tokens on gold frames, answer no
    if a.in_domain_negatives:
        from frame_data.interpolate import DEFAULT_ROOT, frame_path
        from frame_data.kb import build_kb
        kb, _ = build_kb()
        root = Path(DEFAULT_ROOT)
        pool = [(k, f.classes) for k, f in kb.items()
                if f.classes and not ({GALL, AHA} & set(f.classes))]
        rng.shuffle(pool)
        per_video_cap = max(1, a.in_domain_negatives // 40)
        by_video, made = Counter(), 0
        for (dsname, video, t), classes in pool:
            if made >= a.in_domain_negatives:
                break
            if by_video[video] >= per_video_cap:
                continue
            p = frame_path(root, dsname, video, t)
            if not p.exists():
                continue
            tgt = TARGETS[made % 2]
            other = sorted(classes)[made % len(classes)]      # a class that IS in the frame
            buckets["cooc_in_domain"].append(
                {"dataset": dsname, "video": video, "frame_path": str(p), "time": t,
                 "source": "lapchole_in_domain_negative",
                 "question": COOC_Q.format(a=PLURAL[tgt], b=PLURAL.get(other, other + "s")),
                 "answer": "no", "format": "binary", "kind": "cooc_in_domain"})
            by_video[video] += 1
            made += 1

    if a.extract:
        print(f"extracting uncropped frames -> {OUT_FRAMES}")
        n = extract(need)
        print(f"  wrote {n} new frames")

    for rs in buckets.values():
        for r in rs:
            if r["frame_path"] is None:
                stem = r.pop("video_stem")
                r["frame_path"] = str(OUT_FRAMES / stem /
                                      f"frame{int(round(r['time'] * BASE_FPS)):07d}.jpg")
            r.pop("video_stem", None)

    rows = []
    for kind, rs in buckets.items():
        rng.shuffle(rs)
        if any(k == kind for k, _ in ANSWER_CAPS):
            kept, per = [], Counter()
            for r in rs:
                lim = ANSWER_CAPS.get((kind, str(r["answer"])))
                if lim is not None and per[r["answer"]] >= lim:
                    continue
                per[r["answer"]] += 1
                kept.append(r)
            rs = kept
        rows += rs[:CAPS.get(kind, len(rs))]

    dropped = 0
    if a.max_no_ratio:
        yes = sum(1 for r in rows if r["format"] == "binary" and r["answer"] == "yes")
        keep = max(200, int(a.max_no_ratio * yes))          # a floor, so a checklist-free
        no = [r for r in rows if r["format"] == "binary" and r["answer"] == "no"]  # delivery
        rng.shuffle(no)                                     # still closes the shortcut
        no.sort(key=lambda r: -(DROP_ORDER.index(r["kind"]) if r["kind"] in DROP_ORDER else 99))
        drop = set(map(id, no[keep:]))
        dropped = len(drop)
        rows = [r for r in rows if id(r) not in drop]
    missing = sum(1 for r in rows if not Path(r["frame_path"]).exists())
    rng.shuffle(rows)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    print(f"\n{stats['videos']} videos ({stats['videos_empty']} dropped as empty), "
          f"{stats['frames_reviewed']:,} frames swept "
          f"({stats['frames_positive']} with a target, {stats['frames_negative']} without)")
    print(f"  instances: " + ", ".join(f"{c} {stats['marks_' + c]}" for c in TARGETS))
    print(f"\n{'kind':20s} {'emitted':>8s} {'available':>10s}")
    for kind in sorted(buckets):
        got = sum(1 for r in rows if r["kind"] == kind)
        print(f"{kind:20s} {got:8d} {len(buckets[kind]):10d}")
    print(f"{'TOTAL':20s} {len(rows):8d}")
    print("  formats:", dict(Counter(r["format"] for r in rows).most_common()))
    print("  binary answers:", dict(Counter(r["answer"] for r in rows if r["format"] == "binary")),
          f"({dropped} 'no' rows dropped to balance)" if dropped else "")
    if missing:
        print(f"\n  ! {missing} rows point at a frame that does not exist -- rerun with --extract")
    if not a.trust_checklist:
        print("\nno fo_class rows: the Job-1 checklist was optional, so a blank list does not")
        print("entail absence. If the annotators ticked it exhaustively, pass --trust-checklist.")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
