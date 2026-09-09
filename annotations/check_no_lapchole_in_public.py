"""Fail if any LapChole-FOCUS-derived content appears under annotations/public/.

The data usage agreement prohibits publishing LapChole-FOCUS or its derivatives until the
organisers release it. The generators mix datasets -- in-domain negatives borrow clips from
the pooled export -- so the public/private split is per ROW, not per file. This asserts it.

    python annotations/check_no_lapchole_in_public.py
"""
import json, sys
from pathlib import Path

root = Path(__file__).resolve().parent
bad = []

for p in sorted((root / "public").rglob("*.jsonl")):
    for i, line in enumerate(p.open(), 1):
        r = json.loads(line)
        if r.get("source_dataset") == "lapchole":
            bad.append(f"{p.relative_to(root)}:{i} source_dataset=lapchole")
        if "lapchole" in str(r.get("videoID", "")).lower():
            bad.append(f"{p.relative_to(root)}:{i} videoID={r['videoID']}")

for p in sorted((root / "public").rglob("*.csv")):
    head = p.open().read(4096).lower()
    if "cholecystectomy" in head or "lapchole" in head:
        bad.append(f"{p.relative_to(root)} mentions cholecystectomy")

if bad:
    print("FAIL: LapChole-derived content found under public/")
    for b in bad[:20]:
        print("  ", b)
    sys.exit(1)

# Hernia frames: FRAME questions are single images, and every shipped frame must be one the
# annotation sheets actually mark up. The agreement with the clinical collaborator permits
# the annotated frames only.
frames = root / "public/frames"
if frames.exists():
    import csv
    marked = set()
    for f in sorted((root / "public/raw/hernia_mesh_annotations").glob("*.csv")):
        for r in csv.DictReader(f.open()):
            marked.add((r["video"], r["image"]))
    n = 0
    for vd in sorted(frames.iterdir()):
        if not vd.is_dir():
            continue
        for f in vd.glob("*.jpg"):
            n += 1
            if (vd.name, f.name) not in marked:
                bad.append(f"public/frames/{vd.name}/{f.name} is not in any annotation sheet")
    if bad:
        print("FAIL: frames that no annotation sheet marks up")
        for b in bad[:20]:
            print("  ", b)
        sys.exit(1)
    print(f"OK: {n} hernia frames, all marked up by an annotation sheet")

pub = sum(1 for p in (root / "public").rglob("*.jsonl") for _ in p.open())
prv = sum(1 for p in (root / "private").rglob("*.jsonl") for _ in p.open())
print(f"OK: public/ holds {pub} rows, none LapChole-derived; private/ holds {prv} rows")
