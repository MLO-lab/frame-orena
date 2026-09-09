"""Publish the annotations to the two Hugging Face dataset repos.

    python annotations/upload.py --public          # open access
    python annotations/upload.py --private         # gated, LapChole only

Keeps the two halves in separate repos so access can be granted independently: the LapChole
material must stay unpublished until the organisers release LapChole-FOCUS, while everything
else can be linked openly as the challenge requires.

Runs check_no_lapchole_in_public.py first and refuses to upload if it fails.
"""
import argparse, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PUBLIC_REPO  = "Machine-Learning-Oncology/orena-frame-annotations"
PRIVATE_REPO = "Machine-Learning-Oncology/orena-frame-annotations-lapchole"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--public", action="store_true")
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--stage-public", action="store_true",
                    help="create the PUBLIC-half repo as private for now (review first)")
    a = ap.parse_args()
    if not (a.public or a.private):
        sys.exit("pick --public and/or --private")

    guard = subprocess.run([sys.executable, str(HERE / "check_no_lapchole_in_public.py")])
    if guard.returncode:
        sys.exit("refusing to upload: the public/private split failed its check")

    from huggingface_hub import HfApi
    api = HfApi()
    # --stage-public keeps the "public" half private on the Hub while it is being reviewed;
    # flip it with `huggingface-cli repo visibility <repo> --type dataset public` or the
    # repo Settings page when you are ready to publish the link.
    for flag, repo, folder, private in [
            (a.public, PUBLIC_REPO, HERE / "public", a.stage_public),
            (a.private, PRIVATE_REPO, HERE / "private", True)]:
        if not flag:
            continue
        if not folder.is_dir():
            sys.exit(f"{folder} does not exist")
        api.create_repo(repo, repo_type="dataset", private=private, exist_ok=True)
        print(f"uploading {folder.name}/ -> {repo} (private={private})")
        api.upload_folder(folder_path=str(folder), repo_id=repo, repo_type="dataset")
        print(f"  https://huggingface.co/datasets/{repo}")


if __name__ == "__main__":
    main()
