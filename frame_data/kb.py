"""Frame knowledge base: parse published FRAME QA into per-frame facts.

Each labelled frame carries only ~1.31 questions, but a single answer often determines
the answer to several others. This module extracts what each question *establishes* about
its frame so `derive.py` can emit the entailed questions.

Twelve templates cover 97% of the 20,000 FRAME questions; only those are parsed. Anything
unrecognised is ignored (counted in `stats["unparsed"]`) — silence here is safe because a
missing fact only costs us derived questions, whereas a wrong fact would poison training.

    python -m frame_data.kb --report
"""
from __future__ import annotations

import argparse
import re
from collections import defaultdict
from dataclasses import dataclass, field

from focus import DatasetSplit, FocusDataset, Track
from focus.foreign_objects import FOType

CLASSES: tuple[str, ...] = tuple(FOType.names())
_CANON = {c.lower(): c for c in CLASSES}
QUADRANTS = ("top/left", "top/right", "bottom/left", "bottom/right")

# Longest first so "Specimen Bag" is matched before "Specimen".
_CLASS_RE = "|".join(re.escape(c) for c in sorted(CLASSES, key=len, reverse=True))


def canon(name: str) -> str | None:
    return _CANON.get(name.strip().lower())


def parse_class_set(answer: str) -> frozenset[str] | None:
    """`fo_class` answer -> canonical class set. `none` -> empty set."""
    parts = [p.strip() for p in answer.split(",") if p.strip()]
    if not parts:
        return None
    if len(parts) == 1 and parts[0].lower() == "none":
        return frozenset()
    out = set()
    for p in parts:
        c = canon(p)
        if c is None:
            return None                      # unknown token: establish nothing
        out.add(c)
    return frozenset(out)


# ── templates that establish a fact ───────────────────────────────────────────
T_LIST_ALL    = re.compile(r"list all foreign objects that are visible", re.I)
T_COMBINATION = re.compile(r"which combination of foreign object classes is visible", re.I)
T_SINGLE_ID   = re.compile(r"there is one surgical foreign object visible in the frame", re.I)
T_N_INSTANCES = re.compile(r"how many different foreign object instances", re.I)
T_N_CLASSES   = re.compile(r"how many different foreign object classes", re.I)
T_N_OF_CLASS  = re.compile(rf"how many ({_CLASS_RE})(?:s|es)? appear in this frame", re.I)
T_SAME_CLASS  = re.compile(r"are all visible foreign objects in this frame of the same class", re.I)
T_COOCCUR     = re.compile(rf"do ({_CLASS_RE})(?:s|es)? and ({_CLASS_RE})(?:s|es)? co-occur", re.I)
T_CLASS_AT_Q  = re.compile(r"what class is the foreign object located in the "
                           r"(top|bottom)/(left|right) relative to the image cent", re.I)
T_CENTRE_OF   = re.compile(rf"where is the cent(?:er|re) of the ({_CLASS_RE})(?:s|es)? located", re.I)
T_POSITIONS   = re.compile(r"please provide all relative central positions of foreign objects", re.I)

_LISTED_OBJ = re.compile(rf"\d+\.\s*({_CLASS_RE})\s*:\s*(top|bottom)/(left|right)", re.I)


@dataclass
class FrameFacts:
    """Everything the published QA establishes about one frame."""

    dataset: str
    video: str
    time: float

    classes: frozenset[str] | None = None          # the full set of classes present
    counts: dict[str, int] = field(default_factory=dict)   # per-class instance counts
    total: int | None = None                       # total instances
    n_classes: int | None = None                   # |classes|
    objects: list[tuple[str, str]] | None = None   # [(class, quadrant)], the richest fact
    quadrant_class: dict[str, str] = field(default_factory=dict)
    same_class: bool | None = None
    cooccur: dict[tuple[str, str], bool] = field(default_factory=dict)
    centre_of: dict[str, str] = field(default_factory=dict)  # class -> quadrant of its centre

    seen_questions: set[str] = field(default_factory=set)   # normalised, for dedup
    sources: list[str] = field(default_factory=list)        # qIDs
    conflicts: list[str] = field(default_factory=list)

    @property
    def key(self):
        return (self.dataset, self.video, round(self.time, 2))

    # -- derived-fact closure -------------------------------------------------
    def close(self) -> None:
        """Propagate everything that follows from what is already known."""
        if self.objects is not None:
            cls = [c for c, _ in self.objects]
            self._set("classes", frozenset(cls))
            self._set("total", len(cls))
            for c in set(cls):
                self._set_count(c, cls.count(c))
            by_q: dict[str, list[str]] = defaultdict(list)
            for c, q in self.objects:
                by_q[q].append(c)
            for q, cs in by_q.items():
                if len(cs) == 1:
                    self.quadrant_class.setdefault(q, cs[0])
            for c in set(cls):
                qs = {q for cc, q in self.objects if cc == c}
                if len(qs) == 1 and cls.count(c) == 1:
                    self.centre_of.setdefault(c, next(iter(qs)))

        if self.classes is not None:
            self._set("n_classes", len(self.classes))
            if len(self.classes) > 0:
                self._set("same_class", len(self.classes) == 1)
            for c in self.classes:
                self.cooccur.setdefault((c, c), True)

        # ---- presence / absence propagation (completes multi-object frames) ----
        present: set[str] = set(self.counts) | set(self.quadrant_class.values()) | set(self.centre_of)
        if self.classes is not None:
            present |= set(self.classes)
        absent: set[str] = set()
        for (x, y), co in self.cooccur.items():
            if co:
                present |= {x, y}
            elif x != y:
                # NOT(x and y): if one is known present, the other must be absent
                if x in present:
                    absent.add(y)
                if y in present:
                    absent.add(x)
        if self.same_class is True and len(present) == 1:
            self._set("classes", frozenset(present))
        # counts that sum to a known total account for every instance -> no other class
        if self.total is not None and self.counts and sum(self.counts.values()) == self.total:
            self._set("classes", frozenset(self.counts))
        # as many known-present classes as there are classes -> the set is complete
        if self.n_classes is not None and len(present) == self.n_classes:
            self._set("classes", frozenset(present))
        if self.classes is not None:
            present |= set(self.classes)

        # a known total over a single-class frame -> that class's count
        if self.classes is not None and len(self.classes) == 1 and self.total is not None:
            self._set_count(next(iter(self.classes)), self.total)

        # counts complete over a known class set -> total
        if self.classes is not None and self.classes and self.total is None:
            if set(self.counts) == set(self.classes):
                self._set("total", sum(self.counts.values()))
        # total plus all-but-one per-class count -> the missing one
        if self.classes is not None and self.total is not None:
            missing = set(self.classes) - set(self.counts)
            if len(missing) == 1:
                m = missing.pop()
                v = self.total - sum(self.counts.values())
                if v >= 1:
                    self._set_count(m, v)
        # a single class present with a known total -> that class's count
        if self.classes is not None and len(self.classes) == 1 and self.total is not None:
            self._set_count(next(iter(self.classes)), self.total)

    def _set(self, attr: str, value) -> None:
        cur = getattr(self, attr)
        if cur is None:
            setattr(self, attr, value)
        elif cur != value:
            self.conflicts.append(f"{attr}: {cur!r} vs {value!r}")

    def _set_count(self, cls: str, n: int) -> None:
        if cls in self.counts and self.counts[cls] != n:
            self.conflicts.append(f"count[{cls}]: {self.counts[cls]} vs {n}")
        else:
            self.counts[cls] = n


def _norm_q(q: str) -> str:
    return re.sub(r"\s+", " ", q.strip().lower())


def build_kb(datasets=("heico", "lapchole"), splits=("train", "test")):
    """Return ``{frame_key: FrameFacts}`` plus a stats dict."""
    kb: dict[tuple, FrameFacts] = {}
    stats = defaultdict(int)

    for ds in datasets:
        for sp in splits:
            split = DatasetSplit.TRAIN if sp == "train" else DatasetSplit.TEST
            for req, ref in FocusDataset(ds, split, Track.FRAME):
                key = (ds, req.videoID, round(req.start_time, 2))
                f = kb.setdefault(key, FrameFacts(ds, req.videoID, req.start_time))
                f.seen_questions.add(_norm_q(req.question))
                f.sources.append(str(req.qID))
                q, a = req.question, ref.answer.strip()
                stats["questions"] += 1

                if T_POSITIONS.search(q):
                    objs = [(canon(c), f"{v.lower()}/{h.lower()}")
                            for c, v, h in _LISTED_OBJ.findall(a)]
                    if objs and all(c for c, _ in objs):
                        f.objects = objs
                        stats["fact_objects"] += 1
                    continue

                if T_LIST_ALL.search(q) or T_COMBINATION.search(q):
                    s = parse_class_set(a)
                    if s is not None:
                        f._set("classes", s)
                        stats["fact_class_set"] += 1
                    continue

                if T_SINGLE_ID.search(q):
                    s = parse_class_set(a)
                    if s is not None and len(s) == 1:
                        f._set("classes", s)
                        f._set("total", 1)
                        stats["fact_single"] += 1
                    continue

                if T_N_INSTANCES.search(q):
                    if a.isdigit():
                        f._set("total", int(a))
                        stats["fact_total"] += 1
                    continue

                if T_N_CLASSES.search(q):
                    if a.isdigit():
                        f._set("n_classes", int(a))
                        stats["fact_n_classes"] += 1
                    continue

                if (m := T_N_OF_CLASS.search(q)):
                    c = canon(m.group(1))
                    if c and a.isdigit():
                        f._set_count(c, int(a))
                        stats["fact_per_class_count"] += 1
                    continue

                if T_SAME_CLASS.search(q):
                    if a.lower().strip(".") in ("yes", "no"):
                        f._set("same_class", a.lower().strip(".") == "yes")
                        stats["fact_same_class"] += 1
                    continue

                if (m := T_COOCCUR.search(q)):
                    x, y = canon(m.group(1)), canon(m.group(2))
                    if x and y and a.lower().strip(".") in ("yes", "no"):
                        f.cooccur[tuple(sorted((x, y)))] = a.lower().strip(".") == "yes"
                        stats["fact_cooccur"] += 1
                    continue

                if (m := T_CLASS_AT_Q.search(q)):
                    c = parse_class_set(a)
                    if c and len(c) == 1:
                        f.quadrant_class[f"{m.group(1).lower()}/{m.group(2).lower()}"] = next(iter(c))
                        stats["fact_quadrant_class"] += 1
                    continue

                if (m := T_CENTRE_OF.search(q)):
                    c = canon(m.group(1))
                    if c and a.strip().lower() in QUADRANTS:
                        f.centre_of[c] = a.strip().lower()
                        stats["fact_centre_of"] += 1
                    continue

                stats["unparsed"] += 1

    for f in kb.values():
        f.close()
        f.close()                      # second pass: new facts feed the later rules
        if f.conflicts:
            stats["frames_with_conflicts"] += 1
    stats["frames"] = len(kb)
    return kb, dict(stats)


def _report() -> None:
    kb, st = build_kb()
    print("=== frame knowledge base ===")
    for k in sorted(st):
        print(f"  {k:24s} {st[k]:,}")
    have = lambda p: sum(1 for f in kb.values() if p(f))
    print("\n=== what is known per frame ===")
    print(f"  class set known          {have(lambda f: f.classes is not None):,}")
    print(f"  total instances known    {have(lambda f: f.total is not None):,}")
    print(f"  >=1 per-class count      {have(lambda f: bool(f.counts)):,}")
    print(f"  full object list         {have(lambda f: f.objects is not None):,}")
    print(f"  any quadrant known       {have(lambda f: bool(f.quadrant_class) or bool(f.centre_of)):,}")
    print(f"  nothing usable           {have(lambda f: f.classes is None and f.total is None and not f.counts):,}")
    bad = [f for f in kb.values() if f.conflicts]
    print(f"\n=== consistency: {len(bad):,} frames with contradictory facts ===")
    for f in bad[:8]:
        print(f"  {f.video[:34]} t={f.time}  {f.conflicts}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true")
    ap.parse_args()
    _report()
