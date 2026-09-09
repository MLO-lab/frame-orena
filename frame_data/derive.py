"""Emit FRAME questions whose answers the knowledge base *entails*.

Nothing here is inferred, guessed or model-generated: every emitted answer follows
logically from a published gold answer about the same frame. Frames carry ~1.31 questions
each while 6,614 of them have their full class set known, so most of the entailment is
simply unused.

Two rules keep the output safe to train on:

* **verbatim wording** — question strings are lifted from the published data (exemplars
  collected at runtime), never hand-written, so phrasing and answer-format routing match
  what the evaluator sees;
* **verbatim answer surface form** — class names use the casing observed in golds
  (`External drain`, not the registry's `External Drain`) and sets are alphabetically
  sorted, which every one of the 8,971 published `fo_class` golds already is.

    python -m frame_data.derive --report
    python -m frame_data.derive --out frame_data/derived_train.jsonl --splits train
"""
from __future__ import annotations

import argparse
import itertools
import json
import random
import re
from pathlib import Path
from collections import Counter, defaultdict

from focus import DatasetSplit, FocusDataset, Track

from frame_data.interpolate import DEFAULT_ROOT, frame_path
from frame_data.kb import (QUADRANTS, T_CENTRE_OF, T_CLASS_AT_Q, T_COMBINATION, T_COOCCUR,
                           T_LIST_ALL, T_N_CLASSES, T_N_INSTANCES, T_N_OF_CLASS,
                           T_SAME_CLASS, build_kb, canon, _norm_q)

# `number` golds on FRAME are never 0 (verified: min is 1), so a derived count of 0 would
# be off-distribution. Emitting them is opt-in.
ALLOW_ZERO_COUNTS = False
# One co-occurrence per frame, not two. These questions follow mechanically from the class
# set, so they are cheap to generate *because* they carry little information a model that
# can already list classes does not have -- and `binary` is already the second-strongest
# format (0.771). Ungated, the rule produced 15,097 binary items against 1,954 published,
# i.e. 40% of the whole mixture on the least informative question type.
MAX_COOCCUR_PER_FRAME = 1
# Binary is capped as a SHARE of the derived output, not an absolute count, so it scales
# when `--only-classes` shrinks the pool. Published binary is 9.8% of the FRAME data; 0.15
# leaves it modestly over-represented, which is as far as a redundant format deserves.
DEFAULT_BINARY_SHARE = 0.15

# Cap on the derived `number` block. At >=2 these are ~99% "how many different CLASSES" --
# the BEST-performing counting sub-type (0.670, vs 0.348 for instance counts and 0.406 for
# per-class counts), 97.7% within +/-1, and worth only +0.025 SCORE if perfected against
# +0.096 and +0.087 for the other two. Uncapped it takes ~35% of the counting budget for
# ~12% of the recoverable score; 1250 holds it near its published ~20-25% share.
DEFAULT_NUMBER_CAP = 1250


def collect_exemplars(datasets=("heico", "lapchole"), splits=("train", "test")):
    """Harvest verbatim question strings and gold class casing from the published data."""
    ex: dict = {"n_of_class": {}, "centre_of": {}, "class_at_q": {}, "cooccur": {}}
    plain = defaultdict(Counter)          # template name -> Counter of exact question text
    display: dict[str, Counter] = defaultdict(Counter)   # canonical class -> observed casing

    for ds in datasets:
        for sp in splits:
            split = DatasetSplit.TRAIN if sp == "train" else DatasetSplit.TEST
            for req, ref in FocusDataset(ds, split, Track.FRAME):
                q = req.question.strip()
                if ref._format == "fo_class":
                    for part in (p.strip() for p in ref.answer.split(",")):
                        if (c := canon(part)):
                            display[c][part] += 1
                if (m := T_N_OF_CLASS.search(q)):
                    if (c := canon(m.group(1))):
                        ex["n_of_class"].setdefault(c, q)
                elif (m := T_CENTRE_OF.search(q)):
                    if (c := canon(m.group(1))):
                        ex["centre_of"].setdefault(c, q)
                elif (m := T_CLASS_AT_Q.search(q)):
                    ex["class_at_q"].setdefault(f"{m.group(1).lower()}/{m.group(2).lower()}", q)
                elif (m := T_COOCCUR.search(q)):
                    x, y = canon(m.group(1)), canon(m.group(2))
                    if x and y:
                        ex["cooccur"].setdefault(tuple(sorted((x, y))), q)
                elif T_N_CLASSES.search(q):
                    plain["n_classes"][q] += 1
                elif T_N_INSTANCES.search(q):
                    plain["n_instances"][q] += 1
                elif T_SAME_CLASS.search(q):
                    plain["same_class"][q] += 1
                elif T_LIST_ALL.search(q):
                    plain["list_all"][q] += 1
                elif T_COMBINATION.search(q):
                    plain["combination"][q] += 1

    ex["plain"] = {k: c.most_common(1)[0][0] for k, c in plain.items()}
    ex["display"] = {c: cnt.most_common(1)[0][0] for c, cnt in display.items()}
    return ex


def _cooccur_q(ex, a, b):
    """Verbatim co-occurrence question for a class pair, or a templated fallback."""
    key = tuple(sorted((a, b)))
    if key in ex["cooccur"]:
        return ex["cooccur"][key]
    proto = next(iter(ex["cooccur"].values()), None)
    if proto is None:
        return None
    m = T_COOCCUR.search(proto)
    x, y = m.group(1), m.group(2)
    da, db = ex["display"].get(key[0], key[0]), ex["display"].get(key[1], key[1])
    return proto[:m.start(1)] + da + proto[m.end(1):m.start(2)] + db + proto[m.end(2):]


def derive(kb, ex, seed: int = 42, root=None):
    """Yield derived records. Never emits a question already present on that frame."""
    rng = random.Random(seed)
    all_classes = sorted(ex["display"])
    out = []

    def emit(f, question, answer, fmt, rule):
        if question is None or _norm_q(question) in f.seen_questions:
            return
        f.seen_questions.add(_norm_q(question))          # no duplicates within a frame
        out.append(dict(dataset=f.dataset, video=f.video, time=f.time,
                        frame_path=str(frame_path(Path(root or DEFAULT_ROOT),
                                                  f.dataset, f.video, f.time)),
                        question=question, answer=answer, format=fmt,
                        rule=rule, source="derived", provenance=f.sources))

    for f in kb.values():
        disp = lambda c: ex["display"].get(c, c)

        if f.classes is not None:
            s = sorted(f.classes, key=str.lower)
            answer = ", ".join(disp(c) for c in s) if s else "none"
            emit(f, ex["plain"].get("list_all"), answer, "fo_class", "class_set")
            emit(f, ex["plain"].get("combination"), answer, "fo_class", "class_set")
            if s or ALLOW_ZERO_COUNTS:
                emit(f, ex["plain"].get("n_classes"), str(len(s)), "number", "n_classes")
            if s:
                emit(f, ex["plain"].get("same_class"),
                     "yes" if len(s) == 1 else "no", "binary", "same_class")

            # co-occurrence, balanced: one true pair and one false pair per frame
            cands = []
            if len(s) >= 2:
                cands += [(p, "yes") for p in itertools.combinations(s, 2)]
            absent = [c for c in all_classes if c not in f.classes]
            if s and absent:
                cands += [((rng.choice(s), rng.choice(absent)), "no")]
            rng.shuffle(cands)
            for (a, b), ans in cands[:MAX_COOCCUR_PER_FRAME]:
                emit(f, _cooccur_q(ex, a, b), ans, "binary", "cooccur")

        if f.total is not None and (f.total > 0 or ALLOW_ZERO_COUNTS):
            emit(f, ex["plain"].get("n_instances"), str(f.total), "number", "total_instances")

        for c, n in f.counts.items():
            if n > 0 or ALLOW_ZERO_COUNTS:
                emit(f, ex["n_of_class"].get(c), str(n), "number", "per_class_count")

        for q, c in f.quadrant_class.items():
            emit(f, ex["class_at_q"].get(q), disp(c), "fo_class", "class_at_quadrant")

        for c, q in f.centre_of.items():
            emit(f, ex["centre_of"].get(c), q, "multiple_choice", "centre_of_class")

    return out


def cap_number(rec, target, seed=42):
    """Subsample the derived `number` block to `target`, stratified by answer value."""
    num = [r for r in rec if r["format"] == "number"]
    if target is None or len(num) <= target:
        return rec
    rng = random.Random(seed)
    strata = defaultdict(list)
    for r in num:
        strata[r["answer"]].append(r)
    for v in strata.values():
        rng.shuffle(v)
    keep, share = [], target / len(num)
    for v in strata.values():                       # preserve the 2/3/4 proportions
        keep.extend(v[:max(1, round(len(v) * share))])
    return [r for r in rec if r["format"] != "number"] + keep[:target]


def cap_binary(rec, target, seed=42):
    """Subsample the binary block to `target`, balanced over (rule x yes/no).

    Without this the derived data is 40% binary against 10% in the published mixture, and
    6,401 of those are a single repeated sentence.
    """
    binary = [r for r in rec if r["format"] == "binary"]
    if target is None or len(binary) <= target:
        return rec
    others = [r for r in rec if r["format"] != "binary"]
    rng = random.Random(seed)
    strata: dict[tuple, list] = defaultdict(list)
    for r in binary:
        strata[(r["rule"], r["answer"])].append(r)
    for v in strata.values():
        rng.shuffle(v)
    keep, quota = [], target // len(strata)
    for v in strata.values():
        keep.extend(v[:quota])
    leftover = [r for v in strata.values() for r in v[quota:]]   # top up from short strata
    rng.shuffle(leftover)
    keep.extend(leftover[:target - len(keep)])
    return others + keep


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["train", "test"])
    ap.add_argument("--out")
    ap.add_argument("--report", action="store_true")
    # `number` is EXCLUDED by default. The frames whose facts we can complete are
    # overwhelmingly single-object, so derived counts are 80% "1" against 31% in the
    # published data -- adding them would reinforce the count-1 prior the model already
    # over-uses. Recognition formats are unaffected. Re-enable only behind a
    # macro-balanced sampler.
    ap.add_argument("--formats", nargs="+",
                    default=["fo_class", "binary", "multiple_choice"],
                    help="answer formats to emit (default excludes `number`)")
    ap.add_argument("--only-classes", nargs="*", default=None,
                    help="emit only for frames whose class set intersects these. Unrestricted, "
                         "A2 spends most of its output on `clip` (3,167 frames, recall 0.820) "
                         "and `sponge` (2,027, 0.755), which have the least headroom. Note A2 "
                         "adds questions, never pixels, so a very narrow filter concentrates "
                         "repetition on few videos rather than teaching a rare class.")
    ap.add_argument("--min-classes", type=int, default=None,
                    help="keep frames with at least this many distinct classes. Targets the "
                         "composition failure (fo_class accuracy 0.707 -> 0.525 -> 0.193 by "
                         "gold set size). Combined with --only-classes as a UNION, since the "
                         "two target different weaknesses: rare-class recall needs rare-class "
                         "frames even when single-class, composition needs multi-class frames "
                         "even when the classes are common.")
    ap.add_argument("--min-number", type=int, default=None,
                    help="when `number` is in --formats, emit only answers >= this. Derived "
                         "counts are 73-80%% the answer '1' (the frames whose labels can be "
                         "fully reconstructed are the simple ones), so unfiltered they deepen "
                         "the count-1 bias. At >=2 they are ~99%% 'how many different CLASSES', "
                         "where the model scores 0.473 / 0.278 / 0.000 at 2 / 3 / 4 classes -- "
                         "weak, and requiring the multi-object perception we are targeting.")
    ap.add_argument("--number-cap", type=int, default=DEFAULT_NUMBER_CAP,
                    help="cap the derived `number` block, stratified by answer value "
                         "(0 disables). See DEFAULT_NUMBER_CAP for why.")
    ap.add_argument("--root", default=DEFAULT_ROOT,
                    help="data root used to build `frame_path` (rebase if the frame store "
                         "lives elsewhere)")
    ap.add_argument("--binary-share", type=float, default=DEFAULT_BINARY_SHARE,
                    help="binary as a share of the derived output, balanced over "
                         "rule x yes/no (0 disables the cap). Published share is 0.098")
    a = ap.parse_args()

    kb, st = build_kb(splits=tuple(a.splits))
    ex = collect_exemplars(splits=tuple(a.splits))
    if a.only_classes or a.min_classes:
        want = set(a.only_classes or ())
        keep = lambda f: bool(f.classes) and (
            (want and bool(f.classes & want)) or
            (a.min_classes is not None and len(f.classes) >= a.min_classes))
        kb = {k: v for k, v in kb.items() if keep(v)}
        crit = []
        if want:
            crit.append(f"contains one of {sorted(want)}")
        if a.min_classes:
            crit.append(f">={a.min_classes} classes")
        print(f"restricted to {len(kb):,} frames ({' OR '.join(crit)})")
    rec = [r for r in derive(kb, ex, root=a.root) if r["format"] in set(a.formats)]
    if a.min_number is not None:
        before = sum(1 for r in rec if r["format"] == "number")
        rec = [r for r in rec if r["format"] != "number"
               or (r["answer"].isdigit() and int(r["answer"]) >= a.min_number)]
        print(f"`number` kept (answer >= {a.min_number}): "
              f"{sum(1 for r in rec if r['format']=='number'):,} of {before:,}")
    rec = cap_number(rec, a.number_cap or None)
    if a.binary_share:
        n_other = sum(1 for r in rec if r["format"] != "binary")
        rec = cap_binary(rec, round(a.binary_share / (1 - a.binary_share) * n_other))

    print(f"published questions : {st['questions']:,}")
    print(f"labelled frames     : {st['frames']:,}")
    print(f"DERIVED questions   : {len(rec):,}   (x{len(rec)/st['questions']:.2f} the published set)")
    print("\nby rule:")
    for k, v in Counter(r["rule"] for r in rec).most_common():
        print(f"  {k:22s} {v:7,}")
    print("\nby answer format:")
    for k, v in Counter(r["format"] for r in rec).most_common():
        print(f"  {k:22s} {v:7,}")
    print("\nframes touched:", f"{len({(r['dataset'],r['video'],r['time']) for r in rec}):,}")
    n = Counter(r["answer"] for r in rec if r["format"] == "number")
    print("derived `number` answers:", dict(sorted(n.items(), key=lambda kv: int(kv[0]))[:10]))
    b = Counter(r["answer"] for r in rec if r["format"] == "binary")
    print("derived `binary` balance:", dict(b))

    if a.out:
        with open(a.out, "w") as fh:
            for r in rec:
                fh.write(json.dumps(r) + "\n")
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
