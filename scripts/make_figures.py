"""Build all figures from result CSVs (python -m scripts.make_figures [--allow-partial]).

Reads only `$RESULTS_DIR/tables/` and `$RESULTS_DIR/raw/`, and writes the C9 figures as PNG
(200 dpi) and PDF to `$RESULTS_DIR/figures/`. Run `python -m src.analysis` first.

Like the analysis, it refuses to run while runs are missing, unless `--allow-partial`. With the
flag, every figure has `PARTIAL` in its title and `_PARTIAL` in its file name. Figure 7 shows no
results, so it is never stamped; it is regenerated only if missing, which needs CIFAR-10-C.

Encoding, the same in every figure: colour = strategy (Okabe-Ito blue, vermilion, bluish green,
which passes the colour-vision-deficiency check of the palette validator), line style = model,
marker = strategy. Accuracy and error are shown in %, differences in percentage points.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from src.analysis import (  # noqa: E402
    ANALYSED_STRATEGIES,
    MAIN_FAMILIES,
    PARTIAL_SUFFIX,
    expected_run_ids,
    missing_runs,
    output_name,
    run_id_of,
)
from src.constants import ANALYSIS_GROUPS, CLASS_NAMES, MODELS, SEEDS, SEVERITIES  # noqa: E402
from src.utils import env_path  # noqa: E402

A4_WIDTH_IN = 7.2
DPI = 200
STRATEGY_COLOURS = {"standard": "#0072B2", "augmix": "#D55E00", "corruption_aware": "#009E73"}
STRATEGY_MARKERS = {"standard": "o", "augmix": "s", "corruption_aware": "^"}
STRATEGY_LABELS = {
    "standard": "S1 standard",
    "augmix": "S2 AugMix",
    "corruption_aware": "S3 corruption-aware",
}
MODEL_STYLES = {"simple_cnn": "--", "resnet18": "-"}
MODEL_LABELS = {"simple_cnn": "Simple CNN", "resnet18": "ResNet-18"}
GROUP_LABELS = {
    "seen": "Seen",
    "unseen_same_family": "Unseen,\nsame family",
    "unseen_other_family": "Unseen,\nother family",
}
INK, MUTED, GRID = "#262625", "#6b6a66", "#e4e3df"
CONFUSION_RUN = "resnet18-standard-s0"
CONFUSION_PANELS = (
    ("clean__s0", "Clean"),
    ("fog__s5", "Fog, s5"),
    ("gaussian_noise__s5", "Gaussian noise, s5"),
)
SAMPLE_GRID = "fig7_sample_grid"


@dataclass(frozen=True)
class Context:
    """Where a figure reads from and how its title is stamped."""

    results_dir: Path
    partial: bool
    n_runs: int

    def table(self, stem: str) -> pd.DataFrame:
        """Return one table of `results/tables/` (stamped if partial)."""
        return pd.read_csv(self.results_dir / "tables" / output_name(stem, "csv", self.partial))

    def title(self, text: str) -> str:
        """Return a figure title, stamped when partial."""
        total = len(expected_run_ids())
        return f"PARTIAL ({self.n_runs} of {total} runs): {text}" if self.partial else text


def _set_style() -> None:
    """Set the shared look: small type for A4 width, recessive grid and axes."""
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.titlesize": 9,
            "figure.titlesize": 10,
            "axes.edgecolor": MUTED,
            "axes.labelcolor": INK,
            "text.color": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "legend.frameon": False,
            "lines.linewidth": 1.6,
            "pdf.fonttype": 42,
        }
    )


def _configs(table: pd.DataFrame) -> list[tuple[str, str]]:
    """Return the (model, strategy) pairs in `table`, in fixed order."""
    present = set(zip(table["model"], table["strategy"], strict=True))
    return [(m, s) for m in MODELS for s in ANALYSED_STRATEGIES if (m, s) in present]


def _config_rows(table: pd.DataFrame, model: str, strategy: str) -> pd.DataFrame:
    """Return the rows of one configuration."""
    return table[(table["model"] == model) & (table["strategy"] == strategy)]


def _line_kwargs(model: str, strategy: str) -> dict:
    """Return the line encoding of one configuration."""
    return {
        "color": STRATEGY_COLOURS[strategy],
        "linestyle": MODEL_STYLES[model],
        "label": f"{MODEL_LABELS[model]}, {STRATEGY_LABELS[strategy]}",
    }


def _mean_std_cell(mean: float, std: float) -> str:
    """Format one main-table cell in %; a single seed has no std."""
    return f"{100 * mean:.1f}" if pd.isna(std) else f"{100 * mean:.1f} ± {100 * std:.1f}"


def main_table_figure(ctx: Context) -> Figure:
    """Figure 1: the main results table as an image."""
    table = ctx.table("main_results")
    metrics = {
        "clean_acc": "Clean\nacc.",
        "clean_f1": "Clean\nmacro-F1",
        "mean_corrupted_acc": "Corrupted\nacc.",
        "mean_corrupted_f1": "Corrupted\nmacro-F1",
        "mean_abs_degradation": "Abs.\ndegr. (pp)",
        "mean_rel_degradation": "Rel.\ndegr.",
        "mCE": "mCE",
        "relative_mCE": "Relative\nmCE",
    }
    rows = [
        [
            MODEL_LABELS[row["model"]],
            STRATEGY_LABELS[row["strategy"]],
            str(row["n_seeds"]),
            *[_mean_std_cell(row[f"{m}_mean"], row[f"{m}_std"]) for m in metrics],
        ]
        for _, row in table.iterrows()
    ]
    fig, ax = plt.subplots(figsize=(A4_WIDTH_IN, 2.6), layout="constrained")
    ax.axis("off")
    ax.grid(False)
    header = ["Model", "Strategy", "Seeds", *metrics.values()]
    widths = [0.095, 0.15, 0.05, *[0.088] * len(metrics)]
    grid = ax.table(
        cellText=rows, colLabels=header, colWidths=widths, loc="center", cellLoc="center"
    )
    grid.auto_set_font_size(False)
    grid.set_fontsize(6.5)
    for (row, _), cell in grid.get_celld().items():
        cell.set_edgecolor(GRID)
        cell.set_height(0.2 if row == 0 else 0.115)
        if row == 0:
            cell.set_text_props(fontweight="bold")
            cell.set_facecolor("#f1f0ec")
    fig.suptitle(ctx.title("Main results, % (mean ± std over seeds, ddof=1)"))
    return fig


def severity_figure(ctx: Context) -> Figure:
    """Figure 2: accuracy against severity 0-5 with a ±std band."""
    table = ctx.table("per_severity")
    fig, ax = plt.subplots(figsize=(A4_WIDTH_IN, 3.6), layout="constrained")
    for model, strategy in _configs(table):
        rows = _config_rows(table, model, strategy).sort_values("severity")
        mean = 100 * rows["accuracy_mean"].to_numpy()
        std = 100 * rows["accuracy_std"].to_numpy()
        marker = STRATEGY_MARKERS[strategy]
        ax.plot(rows["severity"], mean, marker=marker, ms=4, **_line_kwargs(model, strategy))
        colour = STRATEGY_COLOURS[strategy]
        ax.fill_between(rows["severity"], mean - std, mean + std, color=colour, alpha=0.15, lw=0)
    ax.set_xticks([0, *SEVERITIES])
    ax.set_xlabel("Corruption severity (0 = clean)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 100)
    ax.legend(loc="lower left", ncols=2)
    fig.suptitle(ctx.title("Accuracy against severity (mean, band = ±std over seeds)"))
    return fig


def corruption_heatmap_figure(ctx: Context) -> Figure:
    """Figure 3: accuracy per corruption and configuration, with numbers in the cells."""
    table = ctx.table("per_corruption")
    configs = _configs(table)
    corruptions = list(table["corruption"].drop_duplicates())
    values = np.array(
        [
            _config_rows(table, *config).set_index("corruption").loc[corruptions, "accuracy_mean"]
            for config in configs
        ]
    ).T
    fig, ax = plt.subplots(figsize=(A4_WIDTH_IN, 5.2), layout="constrained")
    image = ax.imshow(100 * values, cmap="Blues", vmin=0, vmax=100, aspect="auto")
    ax.grid(False)
    ax.set_xticks(
        range(len(configs)),
        [f"{MODEL_LABELS[m]}\n{STRATEGY_LABELS[s]}" for m, s in configs],
        fontsize=7,
    )
    ax.set_yticks(range(len(corruptions)), [c.replace("_", " ") for c in corruptions])
    for (row, column), value in np.ndenumerate(100 * values):
        colour = "white" if value > 60 else INK
        ax.text(column, row, f"{value:.1f}", ha="center", va="center", color=colour, fontsize=7.5)
    families = table.drop_duplicates("corruption")["family"].to_numpy()
    for boundary in np.flatnonzero(families[1:] != families[:-1]):
        ax.axhline(boundary + 0.5, color="white", lw=2)
    fig.colorbar(image, ax=ax, label="Accuracy (%)", shrink=0.7)
    fig.suptitle(ctx.title("Accuracy per corruption, mean over severities and seeds"))
    return fig


def _grouped_bars(
    ax: plt.Axes, table: pd.DataFrame, model: str, column: str, value: str, groups: dict[str, str]
) -> None:
    """Draw one bar per strategy in each group; the std is an error bar; label each bar."""
    width = 0.26
    for offset, strategy in enumerate(ANALYSED_STRATEGIES):
        rows = _config_rows(table, model, strategy).set_index(column)
        if rows.empty:
            continue
        means = 100 * rows.loc[list(groups), f"{value}_mean"].to_numpy()
        stds = 100 * rows.loc[list(groups), f"{value}_std"].to_numpy()
        positions = np.arange(len(groups)) + (offset - 1) * width
        bars = ax.bar(
            positions,
            means,
            width * 0.92,
            yerr=np.nan_to_num(stds),
            error_kw={"ecolor": INK, "elinewidth": 0.8},
            color=STRATEGY_COLOURS[strategy],
            label=STRATEGY_LABELS[strategy],
        )
        ax.bar_label(bars, fmt="%.1f", padding=2, fontsize=6)
    ax.set_xticks(range(len(groups)), list(groups.values()))
    ax.grid(axis="x", visible=False)
    ax.set_title(MODEL_LABELS[model])


def _two_model_bars(
    ctx: Context, table: pd.DataFrame, column: str, value: str, groups: dict[str, str], y: str
) -> tuple[Figure, np.ndarray]:
    """Return a two-panel bar figure, one panel per model."""
    fig, axes = plt.subplots(
        1, len(MODELS), figsize=(A4_WIDTH_IN, 3.2), layout="constrained", sharey=True
    )
    for ax, model in zip(axes, MODELS, strict=True):
        _grouped_bars(ax, table, model, column, value, groups)
    axes[0].set_ylabel(y)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=len(ANALYSED_STRATEGIES))
    return fig, axes


def family_figure(ctx: Context) -> Figure:
    """Figure 4: absolute degradation per family, by strategy, one panel per model."""
    table = ctx.table("per_family")
    groups = {family: family.capitalize() for family in MAIN_FAMILIES}
    y_label = "Absolute degradation (percentage points)"
    fig, _ = _two_model_bars(ctx, table, "family", "abs_degradation", groups, y_label)
    fig.suptitle(ctx.title("Accuracy lost per corruption family (mean, bar = ±std)"))
    return fig


def seen_unseen_figure(ctx: Context) -> Figure:
    """Figure 5: accuracy on the three D6 groups, by strategy, one panel per model."""
    table = ctx.table("seen_unseen")
    groups = {group: GROUP_LABELS[group] for group in ANALYSIS_GROUPS}
    fig, axes = _two_model_bars(ctx, table, "analysis_group", "accuracy", groups, "Accuracy (%)")
    axes[0].set_ylim(0, 100)
    fig.suptitle(ctx.title("Accuracy on seen and unseen corruptions (mean, bar = ±std)"))
    return fig


def _row_normalised(counts: np.ndarray) -> np.ndarray:
    """Return a confusion matrix in % of each true class."""
    return 100 * counts / counts.sum(axis=1, keepdims=True)


def confusion_figure(ctx: Context) -> Figure:
    """Figure 6: row-normalised confusion matrices of ResNet-18 S1 seed 0."""
    confusion = np.load(ctx.results_dir / "raw" / f"{CONFUSION_RUN}_confusion.npz")
    fig, axes = plt.subplots(
        1, len(CONFUSION_PANELS), figsize=(A4_WIDTH_IN, 3.1), layout="constrained", sharey=True
    )
    ticks = range(len(CLASS_NAMES))
    for ax, (key, name) in zip(axes, CONFUSION_PANELS, strict=True):
        share = _row_normalised(confusion[key])
        image = ax.imshow(share, cmap="Blues", vmin=0, vmax=100)
        ax.grid(False)
        ax.set_title(name)
        ax.set_xticks(ticks, CLASS_NAMES, rotation=90, fontsize=6.5)
        ax.set_yticks(ticks, CLASS_NAMES, fontsize=6.5)
        ax.set_xlabel("Predicted class")
        for (row, column), value in np.ndenumerate(share):
            if value >= 10:  # selective labels: the cells that carry the collapse
                colour = "white" if value > 60 else INK
                ax.text(
                    column, row, f"{value:.0f}", ha="center", va="center", color=colour, fontsize=5
                )
    axes[0].set_ylabel("True class")
    fig.colorbar(image, ax=axes, label="% of the true class", shrink=0.8)
    fig.suptitle(ctx.title("Confusion matrices, ResNet-18, S1 standard, seed 0"))
    return fig


def _training_logs(ctx: Context) -> pd.DataFrame:
    """Return all train logs of the matrix that exist, with the configuration of each run."""
    frames = []
    for model in MODELS:
        for strategy in ANALYSED_STRATEGIES:
            for seed in SEEDS:
                path = ctx.results_dir / "raw" / f"{run_id_of(model, strategy, seed)}_train_log.csv"
                if path.is_file():
                    frames.append(pd.read_csv(path).assign(model=model, strategy=strategy))
    return pd.concat(frames, ignore_index=True)


def training_curves_figure(ctx: Context) -> Figure:
    """Figure 8: train loss and validation accuracy per epoch, mean over seeds."""
    logs = _training_logs(ctx)
    means = logs.groupby(["model", "strategy", "epoch"])[["train_loss", "val_acc"]].mean()
    means = means.reset_index()
    fig, (loss_ax, acc_ax) = plt.subplots(1, 2, figsize=(A4_WIDTH_IN, 3.4), layout="constrained")
    for model, strategy in _configs(means):
        rows = _config_rows(means, model, strategy).sort_values("epoch")
        kwargs = _line_kwargs(model, strategy)
        loss_ax.plot(rows["epoch"], rows["train_loss"], **kwargs)
        acc_ax.plot(rows["epoch"], 100 * rows["val_acc"], **kwargs)
    loss_ax.set(xlabel="Epoch", ylabel="Train loss (cross-entropy)", title="Train loss")
    acc_ax.set(xlabel="Epoch", ylabel="Accuracy (%)", title="Validation accuracy")
    handles, labels = loss_ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=2)
    fig.suptitle(ctx.title("Training curves, mean over seeds"))
    return fig


FIGURES: dict[str, Callable[[Context], Figure]] = {
    "fig1_main_table": main_table_figure,
    "fig2_acc_vs_severity": severity_figure,
    "fig3_per_corruption_heatmap": corruption_heatmap_figure,
    "fig4_per_family": family_figure,
    "fig5_seen_unseen": seen_unseen_figure,
    "fig6_confusion": confusion_figure,
    "fig8_training_curves": training_curves_figure,
}


def save_figure(fig: Figure, figures_dir: Path, stem: str, partial: bool) -> list[Path]:
    """Save one figure as PNG and PDF; return the paths."""
    paths = [figures_dir / output_name(stem, ext, partial) for ext in ("png", "pdf")]
    for path in paths:
        metadata = {"CreationDate": None} if path.suffix == ".pdf" else None
        fig.savefig(path, dpi=DPI, metadata=metadata)
    plt.close(fig)
    return paths


def ensure_sample_grid(figures_dir: Path) -> None:
    """Regenerate figure 7 only if it is missing (needs CIFAR-10-C in $DATA_DIR)."""
    if not (figures_dir / f"{SAMPLE_GRID}.png").is_file():
        out = str(figures_dir / f"{SAMPLE_GRID}.png")
        subprocess.run([sys.executable, "-m", "scripts.make_sample_grid", "--out", out], check=True)


def make_figures(results_dir: Path, allow_partial: bool = False) -> list[Path]:
    """Build every figure into `results_dir/figures`; return the stamped or final paths."""
    missing = missing_runs(results_dir / "raw")
    if missing and not allow_partial:
        raise SystemExit(
            f"refusing to run: {len(missing)} runs are missing; use --allow-partial for PARTIAL "
            f"figures. First missing: {missing[0]}"
        )
    partial = bool(missing)
    ctx = Context(results_dir, partial, len(expected_run_ids()) - len(missing))
    figures_dir = results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    _set_style()
    written = []
    for stem, build in FIGURES.items():
        written += save_figure(build(ctx), figures_dir, stem, partial)
    ensure_sample_grid(figures_dir)
    return written


def main() -> None:
    """Command-line entry point (CONTRACTS C5)."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="run with runs missing; every result figure is stamped PARTIAL",
    )
    args = parser.parse_args()
    written = make_figures(env_path("RESULTS_DIR"), args.allow_partial)
    for path in written:
        print(path)
    if any(PARTIAL_SUFFIX in path.name for path in written):
        print("PARTIAL figures: not for the report or README.", file=sys.stderr)


if __name__ == "__main__":
    main()
