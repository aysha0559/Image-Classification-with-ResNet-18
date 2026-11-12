"""Publish the demo to Hugging Face (Phase 11, DECISIONS D10). Only with the human's go-ahead.

    python -m scripts.publish_hf --model-repo USER/NAME --space USER/NAME [--dry-run]
        [--checkpoints dist/checkpoints] [--skip-model] [--skip-space]

1. Checks `dist/checkpoints/` (from `scripts.export_checkpoints`) against its `models.json`.
2. Uploads it to the model repo (created if missing, public).
3. Stages the Space from files tracked by git at HEAD (refusing if any of them has uncommitted
   changes): Dockerfile, requirements, LICENSE, app/, src/, configs/, splits/, results/, with
   `app/README.md` (the Spaces YAML header) as the Space's `README.md`. Creates the Space
   (Docker SDK) if missing, sets its variable `HF_MODEL_REPO`, and uploads.
4. Downloads every checkpoint back from the Hub and checks its sha256 against `models.json`.

The token is read from the environment variable `HF_TOKEN` only. `--dry-run` needs no token and
no network: it checks the checkpoints and lists what would be uploaded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from src.utils import REPO_ROOT

DEFAULT_CHECKPOINTS = REPO_ROOT / "dist" / "checkpoints"
SPACE_PATHS = (
    "Dockerfile",
    ".dockerignore",
    "requirements.txt",
    "requirements-local.txt",
    "LICENSE",
    "app",
    "src",
    "configs",
    "splits",
    "results",
)
SPACE_README = "app/README.md"
# Remote files under these folders that are not in the new upload are deleted, so the Space
# matches HEAD. README.md, Dockerfile and the rest are overwritten in place.
SPACE_DELETE = ["app/**", "src/**", "configs/**", "splits/**", "results/**"]


def sha256_file(path: Path) -> str:
    """Return the hex sha256 of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def check_checkpoints(folder: Path) -> list[dict]:
    """Return the models.json entries after checking every file's sha256; raise on a mismatch."""
    entries = json.loads((folder / "models.json").read_text())["models"]
    if len(entries) != 6:
        raise ValueError(f"{folder}/models.json lists {len(entries)} models, expected 6")
    for e in entries:
        got = sha256_file(folder / e["file"])
        if got != e["sha256"]:
            raise ValueError(f"{e['file']}: sha256 {got} does not match models.json")
    return entries


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, check=True, capture_output=True, text=True
    ).stdout


def space_files() -> list[str]:
    """Return the tracked files that make up the Space; refuse if any has uncommitted changes."""
    dirty = _git("status", "--porcelain", "--", *SPACE_PATHS).strip()
    if dirty:
        raise RuntimeError(f"commit these first, the Space is built from HEAD:\n{dirty}")
    files = _git("ls-files", "--", *SPACE_PATHS).split()
    if SPACE_README not in files:
        raise FileNotFoundError(f"{SPACE_README} is not tracked")
    return files


def stage_space(files: list[str], dest: Path) -> None:
    """Copy `files` into `dest`, with app/README.md also copied to README.md."""
    for name in files:
        target = dest / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO_ROOT / name, target)
    shutil.copy2(REPO_ROOT / SPACE_README, dest / "README.md")


def verify_hub(model_repo: str, entries: list[dict], token: str) -> None:
    """Download each checkpoint from the Hub into a fresh cache and compare sha256s."""
    from huggingface_hub import hf_hub_download

    with tempfile.TemporaryDirectory() as cache:
        for e in entries:
            path = Path(hf_hub_download(model_repo, e["file"], cache_dir=cache, token=token))
            got = sha256_file(path)
            status = "ok" if got == e["sha256"] else "MISMATCH"
            print(f"hub {e['file']}: {got} {status}")
            if got != e["sha256"]:
                raise ValueError(f"{e['file']} on the Hub does not match models.json")


def publish(args: argparse.Namespace) -> None:
    """Run the steps of the module docstring."""
    folder = Path(args.checkpoints)
    entries = check_checkpoints(folder) if not args.skip_model else []
    for e in entries:
        print(f"checked {e['file']} {e['sha256']}")
    files = space_files() if not args.skip_space else []
    commit = _git("rev-parse", "--short", "HEAD").strip()

    if args.dry_run:
        print(f"[dry run] model repo {args.model_repo}: {sorted(p.name for p in folder.iterdir())}")
        print(f"[dry run] Space {args.space} from commit {commit}: {len(files)} files + README.md")
        print(f"[dry run] Space variable HF_MODEL_REPO={args.model_repo}")
        return

    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        raise SystemExit("set HF_TOKEN in the environment (it is never read from a file or flag)")
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    print(f"logged in as {api.whoami()['name']}")
    if not args.skip_model:
        api.create_repo(args.model_repo, repo_type="model", exist_ok=True)
        api.upload_folder(
            repo_id=args.model_repo,
            folder_path=folder,
            repo_type="model",
            commit_message=f"Seed-0 checkpoints, study commit {commit}",
        )
        print(f"uploaded {folder} to https://huggingface.co/{args.model_repo}")
    if not args.skip_space:
        api.create_repo(args.space, repo_type="space", space_sdk="docker", exist_ok=True)
        api.add_space_variable(args.space, "HF_MODEL_REPO", args.model_repo)
        with tempfile.TemporaryDirectory() as tmp:
            stage_space(files, Path(tmp))
            api.upload_folder(
                repo_id=args.space,
                folder_path=tmp,
                repo_type="space",
                commit_message=f"App from study commit {commit}",
                delete_patterns=SPACE_DELETE,
            )
        print(f"uploaded the app to https://huggingface.co/spaces/{args.space}")
    if not args.skip_model:
        verify_hub(args.model_repo, entries, token)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description="Publish checkpoints and the app to Hugging Face.")
    parser.add_argument("--model-repo", required=True, help="model repo id, USER/NAME")
    parser.add_argument("--space", required=True, help="Space id, USER/NAME")
    parser.add_argument("--checkpoints", default=str(DEFAULT_CHECKPOINTS))
    parser.add_argument("--dry-run", action="store_true", help="check and list only; no network")
    parser.add_argument("--skip-model", action="store_true", help="do not upload checkpoints")
    parser.add_argument("--skip-space", action="store_true", help="do not upload the app")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Command-line entry point."""
    publish(parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
