"""Figure 7: one CIFAR-10 test image under every corruption and severity.

    python -m scripts.make_sample_grid [--index 23] [--out results/figures/fig7_sample_grid.png]
    python -m scripts.make_sample_grid --ours [--index 23]

Rows are the 15 standard corruptions (grouped by family), columns are the clean image and
severities 1-5. Writes the PNG (200 dpi) and a PDF with the same stem.

`--ours` instead compares our 4 seen corruptions with their CIFAR-10-C versions on the same
image, one row each, and writes results/figures/extra_our_corruptions.png. This is for a person
to look at; nothing from it reaches training.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.constants import (  # noqa: E402
    CLASS_NAMES,
    CORRUPTION_TO_FAMILY,
    CORRUPTIONS,
    SEEN_CORRUPTIONS,
    SEVERITIES,
)
from src.data.cifar10 import get_test_dataset  # noqa: E402
from src.data.cifar10c import load_cifar10c  # noqa: E402
from src.data.corruptions import apply_corruption  # noqa: E402
from src.utils import env_path  # noqa: E402

DEFAULT_INDEX = 23  # a red fire truck: large, centred, high contrast (chosen by eye)
DEFAULT_OUT = "results/figures/fig7_sample_grid.png"
DEFAULT_OURS_OUT = "results/figures/extra_our_corruptions.png"
OURS_SEED = 0  # noise generator for the --ours grid


def collect_grid(data_dir: str, index: int) -> tuple[np.ndarray, np.ndarray, int]:
    """Return (clean image, (15, 5, 32, 32, 3) corrupted images, label) for test `index`."""
    test = get_test_dataset(data_dir)
    clean = np.asarray(test.data[index])
    label = int(test.targets[index])
    grid = np.empty((len(CORRUPTIONS), len(SEVERITIES), 32, 32, 3), dtype=np.uint8)
    for i, corruption in enumerate(CORRUPTIONS):
        for j, severity in enumerate(SEVERITIES):
            images, labels = load_cifar10c(data_dir, corruption, severity)
            if int(labels[index]) != label:
                raise ValueError(f"{corruption} s{severity}: label {labels[index]} != {label}")
            grid[i, j] = images[index]
    return clean, grid, label


def plot_grid(clean: np.ndarray, grid: np.ndarray, label: int, index: int) -> plt.Figure:
    """Draw the labelled grid: rows = corruptions, columns = clean + severities 1-5."""
    n_rows, n_cols = grid.shape[0], grid.shape[1] + 1
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.25 + 1.6, n_rows * 1.25 + 0.6))
    for i, corruption in enumerate(CORRUPTIONS):
        for j in range(n_cols):
            ax = axes[i, j]
            ax.imshow(clean if j == 0 else grid[i, j - 1], interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_title("clean" if j == 0 else f"severity {j}", fontsize=9)
        family = CORRUPTION_TO_FAMILY[corruption]
        axes[i, 0].set_ylabel(
            f"{corruption}\n({family})", rotation=0, ha="right", va="center", fontsize=8
        )
    fig.suptitle(f"CIFAR-10-C, test image {index} (class: {CLASS_NAMES[label]})", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.975))  # leave room for the suptitle
    return fig


def collect_comparison(data_dir: str, index: int) -> tuple[np.ndarray, np.ndarray, list[str], int]:
    """Return (clean, (8, 5, 32, 32, 3) images, row labels, label): ours and theirs, alternating."""
    test = get_test_dataset(data_dir)
    clean = np.asarray(test.data[index])
    label = int(test.targets[index])
    rng = np.random.default_rng(OURS_SEED)
    rows: list[list[np.ndarray]] = []
    row_labels: list[str] = []
    for corruption in SEEN_CORRUPTIONS:
        rows.append([apply_corruption(clean, corruption, s, rng) for s in SEVERITIES])
        rows.append([load_cifar10c(data_dir, corruption, s)[0][index] for s in SEVERITIES])
        row_labels += [f"{corruption}\n(ours)", f"{corruption}\n(CIFAR-10-C)"]
    return clean, np.array(rows, dtype=np.uint8), row_labels, label


def plot_comparison(
    clean: np.ndarray, grid: np.ndarray, row_labels: list[str], label: int, index: int
) -> plt.Figure:
    """Draw ours and CIFAR-10-C rows: columns = clean + severities 1-5."""
    n_rows, n_cols = grid.shape[0], grid.shape[1] + 1
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.25 + 1.6, n_rows * 1.25 + 0.6))
    for i, row_label in enumerate(row_labels):
        for j in range(n_cols):
            ax = axes[i, j]
            ax.imshow(clean if j == 0 else grid[i, j - 1], interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_title("clean" if j == 0 else f"severity {j}", fontsize=9)
        axes[i, 0].set_ylabel(row_label, rotation=0, ha="right", va="center", fontsize=8)
    fig.suptitle(
        f"Ours vs CIFAR-10-C, test image {index} (class: {CLASS_NAMES[label]})", fontsize=11
    )
    fig.tight_layout(rect=(0, 0, 1, 0.965))  # leave room for the suptitle
    return fig


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Make the CIFAR-10-C sample grid (figure 7).")
    parser.add_argument("--index", type=int, default=DEFAULT_INDEX, help="CIFAR-10 test index")
    parser.add_argument("--out", help=f"PNG path (default {DEFAULT_OUT}, or {DEFAULT_OURS_OUT})")
    parser.add_argument(
        "--ours", action="store_true", help="compare our 4 seen corruptions with CIFAR-10-C"
    )
    args = parser.parse_args(argv)
    data_dir = str(env_path("DATA_DIR"))
    if args.ours:
        clean, grid, row_labels, label = collect_comparison(data_dir, args.index)
        out = Path(args.out or DEFAULT_OURS_OUT)
        out.parent.mkdir(parents=True, exist_ok=True)
        plot_comparison(clean, grid, row_labels, label, args.index).savefig(out, dpi=200)
        print(f"wrote {out} (index {args.index}, {CLASS_NAMES[label]})")
        return
    clean, grid, label = collect_grid(data_dir, args.index)
    fig = plot_grid(clean, grid, label, args.index)
    out = Path(args.out or DEFAULT_OUT)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    fig.savefig(out.with_suffix(".pdf"))
    print(f"wrote {out} and {out.with_suffix('.pdf')} (index {args.index}, {CLASS_NAMES[label]})")


if __name__ == "__main__":
    main()
