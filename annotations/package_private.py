"""Package the LapChole half as a single archive for direct transfer to the organisers.

The private bundle is ~1.2 MB, so hosting it is unnecessary. This produces one .tar.gz plus
a SHA-256, with a MANIFEST listing exactly what is inside, so the recipient can verify they
got what was sent and we have a record of what left.

    python annotations/package_private.py --out orena-segment-lapchole-annotations.tar.gz

Contents are covered by the LapChole-FOCUS data usage agreement: not to be redistributed or
published until the organisers release LapChole-FOCUS.
"""
import argparse, hashlib, json, tarfile
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path,
                    default=HERE.parent / f"orena-frame-lapchole-annotations_{date.today()}.tar.gz")
    a = ap.parse_args()

    src = HERE / "private"
    if not src.is_dir():
        raise SystemExit(f"{src} not found -- run annotations/fetch.py --private")

    files = sorted(p for p in src.rglob("*") if p.is_file())
    vqa = src / "vqa/lapchole.jsonl"
    n_rows = sum(1 for _ in vqa.open()) if vqa.exists() else 0
    n_sheets = len(list((src / "raw/lapchole_annotations").glob("*.csv")))

    manifest = {
        "bundle": "ORena FOCUS 2026 -- FRAME track, LapChole-FOCUS supplementary annotations",
        "team": "MLO-Lab",
        "packaged": str(date.today()),
        "vqa_pairs": n_rows,
        "annotation_sheets": n_sheets,
        "files": [str(p.relative_to(src)) for p in files],
        "licence": ("Covered by the LapChole-FOCUS data usage agreement. Not to be "
                    "redistributed or published until the organisers release "
                    "LapChole-FOCUS, at which point these annotations become public."),
        "notes": [
            "vqa/lapchole.jsonl -- VQA pairs; see annotations/README.md for the schema.",
            "160 rows carry explicit frames_indices and requires_extraction=true: their "
            "windows are absent from the official export, so their clips cannot be "
            "inherited and the frames must be extracted from the source videos.",
            "raw/lapchole_annotations/ -- the frame-level sheets the questions derive from.",
            "raw/crop_boxes.json -- geometry of the frames shown to annotators; required to "
            "map points_xy back to full-frame coordinates on auto-cropped videos.",
            "No LapChole video or frame is included.",
        ],
    }
    mpath = src / "MANIFEST.json"
    mpath.write_text(json.dumps(manifest, indent=2))

    with tarfile.open(a.out, "w:gz") as tar:
        tar.add(src, arcname="lapchole_annotations")
    mpath.unlink()

    h = hashlib.sha256(a.out.read_bytes()).hexdigest()
    Path(str(a.out) + ".sha256").write_text(f"{h}  {a.out.name}\n")
    print(f"{a.out}  ({a.out.stat().st_size/1e6:.2f} MB)")
    print(f"  {n_rows} VQA pairs, {n_sheets} annotation sheets, {len(files)} files")
    print(f"  sha256 {h}")


if __name__ == "__main__":
    main()
