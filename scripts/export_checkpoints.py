"""Export the 6 seed-0 final models for the demo (Phase 11, DECISIONS D10).

    python -m scripts.export_checkpoints [--runs-dir DIR] [--out dist/checkpoints] [--seed 0]

For each (model, strategy) of the main matrix, reads `$RUNS_DIR/{run_id}/model_final.pt`
(CONTRACTS C6), checks it is that run's finished model and loads into `build_model`, and saves
only its state dict (no optimizer, no config) as `{run_id}.pt`. Writes `models.json` with
run_id, model, strategy, seed, file, clean accuracy (from `$RESULTS_DIR/raw/{run_id}.csv`) and the
sha256 of each file, and a model card `README.md` for the Hugging Face model repo.

Refuses, writing nothing, if any of the 6 runs is missing or inconsistent.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

import torch

from src.constants import CLASS_NAMES
from src.models import build_model
from src.utils import REPO_ROOT, count_parameters, env_path

DEFAULT_OUT = REPO_ROOT / "dist" / "checkpoints"
MODELS = (("simple_cnn", "cnn"), ("resnet18", "resnet18"))
STRATEGIES = ("standard", "augmix", "corruption_aware")
MODELS_JSON = "models.json"


def run_ids(seed: int = 0) -> list[tuple[str, str, str]]:
    """Return (run_id, model, strategy) for the 6 main configurations at `seed`."""
    return [
        (f"{short}-{strategy}-s{seed}", model, strategy)
        for model, short in MODELS
        for strategy in STRATEGIES
    ]


def clean_accuracy(results_dir: Path, run_id: str) -> float:
    """Return the clean-test accuracy of `run_id` from its C7 results CSV."""
    path = results_dir / "raw" / f"{run_id}.csv"
    with path.open(newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["corruption"] == "clean"]
    if len(rows) != 1:
        raise ValueError(f"{path}: expected 1 clean row, found {len(rows)}")
    return float(rows[0]["accuracy"])


def load_state_dict(run_dir: Path, run_id: str, model: str, strategy: str, seed: int) -> dict:
    """Return the checked state dict of a finished run's `model_final.pt`."""
    meta = json.loads((run_dir / "run_meta.json").read_text())
    if meta["status"] != "finished":
        raise ValueError(f"{run_dir}: status is {meta['status']!r}, not finished")
    final = torch.load(run_dir / "model_final.pt", map_location="cpu", weights_only=True)
    cfg = final["config"]
    found = (final["run_id"], cfg["model"], cfg["strategy"], cfg["seed"])
    if found != (run_id, model, strategy, seed) or meta["run_id"] != run_id:
        raise ValueError(f"{run_dir}: model_final.pt holds {found}, expected {run_id}")
    net = build_model(model)
    net.load_state_dict(final["state_dict"])  # strict: every key and shape must match
    return final["state_dict"]


def sha256_file(path: Path) -> str:
    """Return the hex sha256 of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def model_card(entries: list[dict]) -> str:
    """Return the model repo's README.md: Hub metadata header and a short description."""
    rows = "\n".join(
        f"| `{e['file']}` | {e['model']} | {e['strategy']} | {100 * e['clean_accuracy']:.2f}% |"
        for e in entries
    )
    return f"""---
license: mit
library_name: pytorch
tags:
- image-classification
- cifar10
- robustness
---

# CIFAR-10 robustness study: seed-0 final models

The final-epoch models (seed 0) of the 6 configurations of the CIFAR-10 robustness study:
a Simple CNN and a CIFAR ResNet-18, each trained with standard augmentation, AugMix, or
corruption-aware training. They are used by the study's demo app.

| File | Model | Strategy | Clean test accuracy |
|---|---|---|---|
{rows}

Each `.pt` file is a PyTorch state dict for `src.models.build_model(model)` of the study's code.
Inputs are 32x32 RGB images, converted with `ToTensor` and normalised with the training-set
mean and std in `splits/cifar10_stats.json`. Classes, in output order: {", ".join(CLASS_NAMES)}.
`models.json` lists every file with its sha256.

Trained on CIFAR-10 (Krizhevsky, 2009, *Learning Multiple Layers of Features from Tiny Images*)
and evaluated on CIFAR-10-C (Hendrycks & Dietterich, 2019, ICLR; CC BY 4.0). The code is under
the MIT licence; see the study's repository and report for the training setup and results.
"""


def export(runs_dir: Path, results_dir: Path, out: Path, seed: int = 0) -> list[dict]:
    """Write the 6 state dicts, models.json and README.md to `out`; return the entries."""
    with tempfile.TemporaryDirectory(dir=out.parent if out.parent.exists() else None) as tmp:
        staging = Path(tmp)
        entries = []
        for run_id, model, strategy in run_ids(seed):
            run_dir = runs_dir / run_id
            if not (run_dir / "model_final.pt").is_file():
                raise FileNotFoundError(f"{run_dir / 'model_final.pt'} not found")
            state = load_state_dict(run_dir, run_id, model, strategy, seed)
            file = f"{run_id}.pt"
            torch.save(state, staging / file)
            entries.append(
                {
                    "run_id": run_id,
                    "model": model,
                    "strategy": strategy,
                    "seed": seed,
                    "file": file,
                    "n_params": count_parameters(build_model(model)),
                    "clean_accuracy": clean_accuracy(results_dir, run_id),
                    "sha256": sha256_file(staging / file),
                    "bytes": (staging / file).stat().st_size,
                }
            )
        payload = {"format": "state_dict for src.models.build_model(model)", "models": entries}
        (staging / MODELS_JSON).write_text(json.dumps(payload, indent=2) + "\n")
        (staging / "README.md").write_text(model_card(entries))
        if out.exists():
            if any(out.iterdir()) and not (out / MODELS_JSON).is_file():
                raise FileExistsError(
                    f"{out} exists and is not an earlier export; not replacing it"
                )
            shutil.rmtree(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(staging, out)
        out.chmod(0o755)  # the temporary folder was private; others (e.g. Docker) must read it
    return entries


def main(argv: list[str] | None = None) -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Export the seed-0 final models for the demo.")
    parser.add_argument("--runs-dir", default=None, help="default: $RUNS_DIR or ./runs")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="default: dist/checkpoints")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    runs_dir = Path(args.runs_dir) if args.runs_dir else env_path("RUNS_DIR")
    entries = export(runs_dir, env_path("RESULTS_DIR"), Path(args.out), args.seed)
    for e in entries:
        print(f"{e['file']}: clean {e['clean_accuracy']:.4f}, {e['bytes']} bytes, {e['sha256']}")
    print(f"wrote {len(entries)} checkpoints, {MODELS_JSON} and README.md to {args.out}")


if __name__ == "__main__":
    main()
