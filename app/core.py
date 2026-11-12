"""Logic of the results viewer and demo (Phase 11), kept free of Streamlit so tests can import it.

The Results page reads only committed files under `results/`. The Try it page reads the bundled
samples in `app/assets/` and the 6 seed-0 checkpoints, either from a local folder
(`LOCAL_CHECKPOINT_DIR`) or from a Hugging Face model repo (`HF_MODEL_REPO`, DECISIONS D10).

Uploaded images are decoded from bytes in memory and never written anywhere.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from torch import nn

from src.constants import CLASS_NAMES, SEEN_CORRUPTIONS
from src.data.corruptions import apply_corruption
from src.data.transforms import build_eval_transform
from src.models import build_model

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"  # the committed results, not $RESULTS_DIR
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"
ASSETS_DIR = Path(__file__).resolve().parent / "assets"
SAMPLES_FILE = ASSETS_DIR / "samples.npz"
SAMPLES_C_FILE = ASSETS_DIR / "samples_c.npz"

MODELS_JSON = "models.json"
CLEAN = "clean"
IMAGE_SIZE = 32
UPLOAD_NOISE_SEED = (
    0  # noise for uploaded images is seeded, so the same choice gives the same image
)
TOP_K = 3

MODEL_LABELS = {"simple_cnn": "Simple CNN", "resnet18": "ResNet-18"}
STRATEGY_LABELS = {
    "none": "none",
    "standard": "standard",
    "augmix": "AugMix",
    "corruption_aware": "corruption-aware",
}
# Display order of the 6 seed-0 models: CNN first, then ResNet-18; standard, AugMix, corr.-aware.
MODEL_ORDER = ("simple_cnn", "resnet18")
STRATEGY_ORDER = ("standard", "augmix", "corruption_aware")

# (file, title, caption). Captions follow report/report.md.
FIGURES: tuple[tuple[str, str, str], ...] = (
    (
        "fig1_main_table.png",
        "Figure 1. Main results",
        "The main results table as an image (mean ± std over 3 seeds).",
    ),
    (
        "fig2_acc_vs_severity.png",
        "Figure 2. Accuracy against severity",
        "Mean accuracy over the 15 corruptions at each severity (severity 0 is the clean test "
        "set), mean over seeds.",
    ),
    (
        "fig3_per_corruption_heatmap.png",
        "Figure 3. Accuracy per corruption",
        "Accuracy per corruption (mean over severities and seeds) for each configuration.",
    ),
    (
        "fig4_per_family.png",
        "Figure 4. Drop per corruption family",
        "Mean absolute accuracy drop per corruption family, per strategy and model.",
    ),
    (
        "fig5_seen_unseen.png",
        "Figure 5. Seen and unseen corruptions",
        "Accuracy on seen corruptions, unseen corruptions from the same families, and unseen "
        "corruptions from other families.",
    ),
    (
        "fig6_confusion.png",
        "Figure 6. Confusion matrices",
        "Confusion matrices of ResNet-18 with standard training (seed 0), in % of each true class.",
    ),
    (
        "fig7_sample_grid.png",
        "Figure 7. The corruptions on one test image",
        "One CIFAR-10 test image under all 15 corruptions at severities 1-5.",
    ),
    (
        "fig8_training_curves.png",
        "Figure 8. Training curves",
        "Training loss and validation accuracy per epoch, mean over seeds.",
    ),
)


# ---------------------------------------------------------------- results tables


def _pm(mean: float, std: float) -> str:
    """Format a fraction pair as 'xx.x% ± y.y%'."""
    return f"{100 * mean:.1f}% ± {100 * std:.1f}%"


def load_main_results() -> pd.DataFrame:
    """Return `results/tables/main_results.csv` as written by `src.analysis`."""
    return pd.read_csv(TABLES_DIR / "main_results.csv")


def format_main_results(df: pd.DataFrame) -> pd.DataFrame:
    """Return the main table for display: percentages as 'mean ± std' over seeds."""
    return pd.DataFrame(
        {
            "Model": df["model"].map(MODEL_LABELS),
            "Strategy": df["strategy"].map(STRATEGY_LABELS),
            "Seeds": df["n_seeds"],
            "Clean accuracy": [
                _pm(m, s) for m, s in zip(df["clean_acc_mean"], df["clean_acc_std"], strict=True)
            ],
            "Mean corrupted accuracy": [
                _pm(m, s)
                for m, s in zip(
                    df["mean_corrupted_acc_mean"], df["mean_corrupted_acc_std"], strict=True
                )
            ],
            "Drop (pp)": [f"{100 * m:.1f}" for m in df["mean_abs_degradation_mean"]],
            "mCE": [_pm(m, s) for m, s in zip(df["mCE_mean"], df["mCE_std"], strict=True)],
            "Relative mCE": [
                _pm(m, s)
                for m, s in zip(df["relative_mCE_mean"], df["relative_mCE_std"], strict=True)
            ],
        }
    )


def load_per_corruption() -> pd.DataFrame:
    """Return `results/tables/per_corruption.csv` formatted for display."""
    df = pd.read_csv(TABLES_DIR / "per_corruption.csv")
    return pd.DataFrame(
        {
            "model": df["model"],
            "strategy": df["strategy"],
            "Model": df["model"].map(MODEL_LABELS),
            "Strategy": df["strategy"].map(STRATEGY_LABELS),
            "Family": df["family"],
            "Corruption": df["corruption"],
            "Accuracy": [
                _pm(m, s) for m, s in zip(df["accuracy_mean"], df["accuracy_std"], strict=True)
            ],
            "CE": [_pm(m, s) for m, s in zip(df["ce_mean"], df["ce_std"], strict=True)],
        }
    )


# ---------------------------------------------------------------- bundled samples


@dataclass(frozen=True)
class Samples:
    """The bundled CIFAR-10 test images and, if present, their CIFAR-10-C versions."""

    images: np.ndarray  # uint8 (N, 32, 32, 3)
    labels: np.ndarray  # int64 (N,)
    test_indices: np.ndarray  # int64 (N,), index in the CIFAR-10 test set
    corrupted: np.ndarray | None  # uint8 (C, S, N, 32, 32, 3)
    corruptions: tuple[str, ...]
    severities: tuple[int, ...]


def load_samples(
    samples_file: Path = SAMPLES_FILE, samples_c_file: Path = SAMPLES_C_FILE
) -> Samples:
    """Load `samples.npz` and, if it exists, `samples_c.npz`; check that they belong together."""
    with np.load(samples_file) as f:
        images, labels, indices = f["images"], f["labels"], f["test_indices"]
    corrupted, corruptions, severities = None, (), ()
    if samples_c_file.is_file():
        with np.load(samples_c_file) as f:
            corrupted = f["images"]
            corruptions = tuple(str(c) for c in f["corruptions"])
            severities = tuple(int(s) for s in f["severities"])
            if not np.array_equal(f["test_indices"], indices):
                raise ValueError(f"{samples_c_file} and {samples_file} hold different images")
        expected = (len(corruptions), len(severities), *images.shape)
        if corrupted.shape != expected:
            raise ValueError(f"{samples_c_file}: shape {corrupted.shape}, expected {expected}")
    return Samples(images, labels, indices, corrupted, corruptions, severities)


def sample_image(samples: Samples, i: int, corruption: str, severity: int) -> np.ndarray:
    """Return bundled sample `i`, clean or as its precomputed CIFAR-10-C version."""
    if corruption == CLEAN:
        return samples.images[i]
    if samples.corrupted is None:
        raise ValueError("the CIFAR-10-C versions of the samples are not bundled")
    c = samples.corruptions.index(corruption)
    s = samples.severities.index(severity)
    return samples.corrupted[c, s, i]


# ---------------------------------------------------------------- uploads


def prepare_upload(data: bytes) -> np.ndarray:
    """Decode an uploaded image in memory: centre square crop, resize to 32x32, uint8 RGB."""
    with Image.open(BytesIO(data)) as img:
        img = ImageOps.exif_transpose(img).convert("RGB")
        side = min(img.size)
        img = ImageOps.fit(img, (side, side), method=Image.Resampling.BICUBIC)
        img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS)
        return np.asarray(img, dtype=np.uint8).copy()


def corrupt_upload(img: np.ndarray, corruption: str, severity: int) -> np.ndarray:
    """Apply one of our 4 seen corruptions (src/data/corruptions.py) to an uploaded image."""
    if corruption == CLEAN:
        return img
    if corruption not in SEEN_CORRUPTIONS:
        raise ValueError(f"uploads support only {', '.join(SEEN_CORRUPTIONS)}, not {corruption}")
    return apply_corruption(img, corruption, severity, np.random.default_rng(UPLOAD_NOISE_SEED))


# ---------------------------------------------------------------- checkpoints


def sha256_file(path: Path) -> str:
    """Return the hex sha256 of a file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_source() -> tuple[str, str] | None:
    """Return ('local', dir) from LOCAL_CHECKPOINT_DIR, else ('hub', repo) from HF_MODEL_REPO."""
    local = os.environ.get("LOCAL_CHECKPOINT_DIR", "").strip()
    if local:
        return "local", local
    repo = os.environ.get("HF_MODEL_REPO", "").strip()
    if repo:
        return "hub", repo
    return None


def fetch_checkpoints(kind: str, where: str) -> tuple[list[dict], dict[str, Path]]:
    """Return (models.json entries, {run_id: local .pt path}), each file checked by sha256.

    `kind` 'local' reads the folder `where`; 'hub' downloads from the model repo `where` with
    `huggingface_hub.hf_hub_download`, which caches the files.
    """
    if kind == "local":
        folder = Path(where)

        def get(name: str) -> Path:
            path = folder / name
            if not path.is_file():
                raise FileNotFoundError(f"{path} not found")
            return path
    elif kind == "hub":
        from huggingface_hub import hf_hub_download

        def get(name: str) -> Path:
            return Path(hf_hub_download(repo_id=where, filename=name))
    else:
        raise ValueError(f"unknown checkpoint source {kind!r}")

    entries = json.loads(get(MODELS_JSON).read_text())["models"]
    paths = {}
    for entry in entries:
        path = get(entry["file"])
        got = sha256_file(path)
        if got != entry["sha256"]:
            raise ValueError(f"{entry['file']}: sha256 {got} does not match {MODELS_JSON}")
        paths[entry["run_id"]] = path
    return entries, paths


def load_model(entry: dict, path: Path) -> nn.Module:
    """Build `entry['model']`, load the exported state dict on the CPU, set eval mode."""
    model = build_model(entry["model"])
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    return model.eval()


def sort_entries(entries: list[dict]) -> list[dict]:
    """Order models.json entries for display: CNN then ResNet-18; standard, AugMix, corr.-aware."""

    def key(e: dict) -> tuple[int, int]:
        m = MODEL_ORDER.index(e["model"]) if e["model"] in MODEL_ORDER else len(MODEL_ORDER)
        s = STRATEGY_ORDER.index(e["strategy"]) if e["strategy"] in STRATEGY_ORDER else 9
        return m, s

    return sorted(entries, key=key)


# ---------------------------------------------------------------- prediction

_EVAL_TRANSFORM = None


def eval_transform():
    """The evaluation transform of `src.evaluate` (ToTensor + Normalize, DECISIONS D2)."""
    global _EVAL_TRANSFORM
    if _EVAL_TRANSFORM is None:
        _EVAL_TRANSFORM = build_eval_transform()
    return _EVAL_TRANSFORM


@torch.no_grad()
def predict_probs(model: nn.Module, img: np.ndarray) -> np.ndarray:
    """Return the softmax over the 10 classes for one uint8 HWC image, as `src.evaluate` sees it."""
    x = eval_transform()(Image.fromarray(np.asarray(img, dtype=np.uint8))).unsqueeze(0)
    return torch.softmax(model(x).float(), dim=1)[0].numpy()


def top_k(probs: np.ndarray, k: int = TOP_K) -> list[tuple[str, float]]:
    """Return the k most probable (class name, probability), most probable first."""
    order = np.argsort(-probs, kind="stable")[:k]
    return [(CLASS_NAMES[i], float(probs[i])) for i in order]
