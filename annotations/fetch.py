"""Download the released annotations from Hugging Face into annotations/.

Two dataset repos back this directory:

  PUBLIC   heico_derived.jsonl, hernia_mesh.jsonl, the hernia annotation sheets, and the
           hernia frames (290 MB). Open access.
  PRIVATE  lapchole.jsonl, the LapChole annotation sheets and crop_boxes.json. Gated: the
           LapChole-FOCUS data usage agreement prohibits publication until the organisers
           release that dataset.

Both are optional. Whatever is present is used; the pipeline degrades cleanly:

    python annotations/fetch.py                 # public only
    python annotations/fetch.py --private       # both (needs access + `hf auth login`)

Rows rebuilt, out of 23,454: 21,440 with the public half alone (the official 20,000 plus
our 1,440 HeiCo and hernia questions), 23,294 adding the private half, and all 23,454 once
the 160 LapChole questions whose windows are absent from the official export also have
their frames extracted -- see --extra-frames on integrate_released_annotations.py.
"""
import argparse, os, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PUBLIC_REPO  = os.environ.get("ORENA_ANNOT_REPO_PUBLIC",
                              "Machine-Learning-Oncology/orena-frame-annotations")
# The LapChole half is deliberately NOT on the Hub -- its data usage agreement forbids
# publishing any derivative until the organisers release LapChole-FOCUS. It is supplied
# directly as an archive. Set this only if a gated repo is ever created for it.
PRIVATE_REPO = os.environ.get("ORENA_ANNOT_REPO_PRIVATE", "")


def pull(repo: str, dest: Path, label: str) -> bool:
    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError
    try:
        snapshot_download(repo_id=repo, repo_type="dataset", local_dir=str(dest))
    except (GatedRepoError, RepositoryNotFoundError, OSError) as e:
        print(f"  {label}: UNAVAILABLE ({type(e).__name__})")
        print(f"    repo {repo}")
        if label == "private":
            print("    this is expected without access; the public half still works")
        return False
    n = sum(1 for _ in dest.rglob("*") if _.is_file())
    try:                                  # --dest may point outside the repo
        shown = dest.relative_to(HERE.parent)
    except ValueError:
        shown = dest
    print(f"  {label}: {n} files -> {shown}")
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--private", action="store_true",
                    help="also expect the LapChole half (supplied on request, not hosted)")
    ap.add_argument("--dest", type=Path, default=HERE)
    a = ap.parse_args()

    print("fetching released annotations")
    ok_pub = pull(PUBLIC_REPO, a.dest / "public", "public")
    ok_prv = None
    if a.private:
        if (a.dest / "private/vqa/lapchole.jsonl").exists():
            print("  private: already present")
            ok_prv = True
        elif PRIVATE_REPO:
            ok_prv = pull(PRIVATE_REPO, a.dest / "private", "private")
        else:
            print("  private: not hosted -- request the archive, then")
            print(f"    tar xzf orena-frame-lapchole-annotations_*.tar.gz "
                  f"-C {a.dest/'private'} --strip-components=1")
            ok_prv = False

    print()
    if not ok_pub:
        sys.exit("FATAL: the public annotations are required")
    if a.private and not ok_prv:
        print("continuing with the public half only (21,440 of 23,454 rows)")
    elif a.private:
        print("both halves present (23,294 rows; 23,454 with --extra-frames)")
    else:
        print("public only (21,440 of 23,454 rows); pass --private for the rest")


if __name__ == "__main__":
    main()
