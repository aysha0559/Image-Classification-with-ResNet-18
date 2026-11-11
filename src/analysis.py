"""Aggregate result CSVs into tables (python -m src.analysis [--allow-partial]).

Reads only `$RESULTS_DIR/raw/{run_id}.csv` (CONTRACTS C7) and writes every C9 table to
`$RESULTS_DIR/tables/`. It needs no run folder, checkpoint, GPU or network.

Without `--allow-partial` it refuses to run unless all 18 runs of the matrix are present. With
the flag, every output has `_PARTIAL` in its file name, a `data_status` column (CSV), a `PARTIAL`
title and banner (Markdown), or `meta.status = "PARTIAL"` (JSON). Partial numbers must not go
into the report or the README.

Definitions (DECISIONS D8):
- Baseline: Simple CNN + standard; the error of each condition is the mean over its seeds.
- Every run gets one value per metric; those values are then summarised per (model, strategy).
- `*_mean` and `*_std` are the mean and the sample standard deviation (ddof=1) over seeds. With
  a single seed the std is undefined: empty in CSV, null in JSON, `n/a` in Markdown.
- Degradation columns are based on accuracy, not macro-F1.
- Comparisons between configurations are paired by seed and use only the seeds both have. The
  rule is applied to the mean paired difference: "within seed noise" when it is smaller than
  the larger of the two configurations' stds.

Analysis groups for RQ4 (DECISIONS D6), read from `src.constants.ANALYSIS_GROUPS`:
- `seen`: gaussian_noise, defocus_blur, contrast, jpeg_compression
- `unseen_same_family`: shot_noise, impulse_noise, glass_blur, motion_blur, zoom_blur, brightness,
  elastic_transform, pixelate
- `unseen_other_family`: snow, frost, fog
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src import metrics
from src.constants import (
    ANALYSIS_GROUPS,
    CORRUPTION_TO_FAMILY,
    CORRUPTIONS,
    FAMILIES,
    MODEL_SHORT_NAMES,
    MODELS,
    SEEDS,
    SEVERITIES,
)
from src.utils import env_path

ANALYSED_STRATEGIES: tuple[str, ...] = ("standard", "augmix", "corruption_aware")  # no optional S0
ALTERNATIVE_STRATEGIES: tuple[str, ...] = ("augmix", "corruption_aware")
MAIN_FAMILIES: tuple[str, ...] = tuple(f for f in FAMILIES if f != "extra")  # D7
BASELINE: tuple[str, str] = ("simple_cnn", "standard")
EXPECTED_N_IMAGES = 10000
SEED_STD_DDOF = 1

PARTIAL_SUFFIX = "_PARTIAL"
WITHIN_NOISE = "within seed noise"
BEYOND_NOISE = "beyond seed noise"
NOT_ASSESSABLE = "not assessable (needs 2+ seeds on both sides)"

RUN_COLUMNS = ["run_id", "model", "strategy", "seed"]
CONFIG_COLUMNS = ["model", "strategy"]
BREAKDOWN_METRICS = ["accuracy", "macro_f1", "abs_degradation", "rel_degradation"]
MAIN_METRICS = [
    "clean_acc",
    "clean_f1",
    "mean_corrupted_acc",
    "mean_corrupted_f1",
    "mean_abs_degradation",
    "mean_rel_degradation",
    "mCE",
    "relative_mCE",
]
COMPARED_METRICS = [
    "mean_corrupted_acc",
    "mean_abs_degradation",
    "mean_rel_degradation",
    "mCE",
    "relative_mCE",
]
CORRUPTION_TO_GROUP = {c: g for g, cs in ANALYSIS_GROUPS.items() for c in cs}
CONDITIONS = [("clean", 0)] + [(c, s) for c in CORRUPTIONS for s in SEVERITIES]


def output_name(stem: str, extension: str, partial: bool) -> str:
    """Return the file name of one output, with `_PARTIAL` before the extension if partial."""
    return f"{stem}{PARTIAL_SUFFIX if partial else ''}.{extension}"


def run_id_of(model: str, strategy: str, seed: int) -> str:
    """Return the C3 run id of one run."""
    return f"{MODEL_SHORT_NAMES[model]}-{strategy}-s{seed}"


def expected_run_ids() -> list[str]:
    """Return the 18 run ids of the matrix, in table order."""
    return [
        run_id_of(model, strategy, seed)
        for model in MODELS
        for strategy in ANALYSED_STRATEGIES
        for seed in SEEDS
    ]


def missing_runs(raw_dir: Path) -> list[str]:
    """Return the expected run ids that have no results CSV in `raw_dir`."""
    return [run_id for run_id in expected_run_ids() if not (raw_dir / f"{run_id}.csv").is_file()]


def _check_run(df: pd.DataFrame, run_id: str) -> pd.DataFrame:
    """Return one run's C7 rows in canonical condition order, without extras; raise if wrong."""
    df = df[df["corruption"].isin(["clean", *CORRUPTIONS])]
    found = list(zip(df["corruption"], df["severity"], strict=True))
    if len(found) != len(CONDITIONS) or set(found) != set(CONDITIONS):
        raise ValueError(f"{run_id}: expected the {len(CONDITIONS)} conditions once each")
    if set(df["run_id"]) != {run_id}:
        raise ValueError(f"{run_id}: the run_id column says {sorted(set(df['run_id']))}")
    if set(df["n_images"]) != {EXPECTED_N_IMAGES}:
        raise ValueError(f"{run_id}: n_images must be {EXPECTED_N_IMAGES} on every row")
    configs = df[["model", "strategy", "seed"]].drop_duplicates()
    if len(configs) != 1 or run_id_of(*configs.iloc[0]) != run_id:
        raise ValueError(f"{run_id}: model, strategy and seed columns do not match the file name")
    indexed = df.set_index(["corruption", "severity"])
    return indexed.loc[CONDITIONS].reset_index()


def load_raw(raw_dir: Path) -> pd.DataFrame:
    """Return the checked rows of all expected runs found in `raw_dir`, in table order."""
    run_ids = [r for r in expected_run_ids() if (raw_dir / f"{r}.csv").is_file()]
    if not run_ids:
        raise ValueError(f"no result CSV of the matrix in {raw_dir}")
    raw = pd.concat(
        [_check_run(pd.read_csv(raw_dir / f"{r}.csv"), r) for r in run_ids], ignore_index=True
    )
    raw["family"] = raw["corruption"].map(CORRUPTION_TO_FAMILY).fillna("clean")
    raw["analysis_group"] = raw["corruption"].map(CORRUPTION_TO_GROUP).fillna("clean")
    return raw


def baseline_table(raw: pd.DataFrame) -> pd.DataFrame:
    """Return the D8 baseline: seed-mean accuracy of Simple CNN + standard per condition."""
    model, strategy = BASELINE
    rows = raw[(raw["model"] == model) & (raw["strategy"] == strategy)]
    if rows.empty:
        raise ValueError("mCE needs at least one run of the baseline (Simple CNN + standard)")
    means = rows.groupby(["corruption", "severity"], sort=False)["accuracy"].mean()
    return means.reset_index()


def run_metrics(raw: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    """Return one row per run with the 8 main metrics (D8), as fractions."""
    rows = []
    for keys, run in raw.groupby(RUN_COLUMNS, sort=False):
        clean = run[run["corruption"] == "clean"].iloc[0]
        corrupted = run[run["corruption"] != "clean"]
        clean_acc, corrupted_acc = float(clean["accuracy"]), float(corrupted["accuracy"].mean())
        rows.append(
            {
                **dict(zip(RUN_COLUMNS, keys, strict=True)),
                "clean_acc": clean_acc,
                "clean_f1": float(clean["macro_f1"]),
                "mean_corrupted_acc": corrupted_acc,
                "mean_corrupted_f1": float(corrupted["macro_f1"].mean()),
                "mean_abs_degradation": metrics.absolute_degradation(clean_acc, corrupted_acc),
                "mean_rel_degradation": metrics.relative_degradation(clean_acc, corrupted_acc),
                "mCE": metrics.mce(run, baseline),
                "relative_mCE": metrics.relative_mce(run, baseline),
            }
        )
    return pd.DataFrame(rows)


def breakdown(raw: pd.DataFrame, column: str, include_clean: bool = False) -> pd.DataFrame:
    """Return one row per (run, `column` value): accuracy, macro-F1 and accuracy degradation."""
    clean = raw[raw["corruption"] == "clean"].set_index("run_id")["accuracy"]
    rows = raw if include_clean else raw[raw["corruption"] != "clean"]
    grouped = rows.groupby([*RUN_COLUMNS, column], sort=False)
    table = grouped[["accuracy", "macro_f1"]].mean().reset_index()
    run_clean = table["run_id"].map(clean)
    table["abs_degradation"] = run_clean - table["accuracy"]
    table["rel_degradation"] = table["abs_degradation"] / run_clean
    return table


def corruption_breakdown(raw: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    """Return `breakdown` per corruption, plus the family and the CE of each run."""
    table = breakdown(raw, "corruption")
    errors = {
        run_id: metrics.corruption_errors(run, baseline) for run_id, run in raw.groupby("run_id")
    }
    table["family"] = table["corruption"].map(CORRUPTION_TO_FAMILY)
    table["ce"] = [errors[r][c] for r, c in zip(table["run_id"], table["corruption"], strict=True)]
    return table


def summarise(
    table: pd.DataFrame, values: list[str], extra_keys: Iterable[str] = ()
) -> pd.DataFrame:
    """Return mean, std (ddof=1) and the seed count of `values` per config and `extra_keys`."""
    keys = [*CONFIG_COLUMNS, *extra_keys]
    grouped = table.groupby(keys, sort=False)
    columns: dict[str, pd.Series] = {"n_seeds": grouped.size()}
    for value in values:
        columns[f"{value}_mean"] = grouped[value].mean()
        columns[f"{value}_std"] = grouped[value].std(ddof=SEED_STD_DDOF)
    return pd.DataFrame(columns).reset_index()


@dataclass(frozen=True)
class Comparison:
    """Two configurations (or two conditions) compared by seed: a minus b."""

    a: pd.Series  # one value per seed
    b: pd.Series

    @property
    def seed_diffs(self) -> pd.Series:
        """Return a - b for each seed both sides have."""
        return (self.a - self.b).dropna()

    @property
    def diff(self) -> float:
        """Return the mean of the paired differences (NaN if no seed is shared)."""
        return float(self.seed_diffs.mean()) if len(self.seed_diffs) else float("nan")

    @property
    def stds(self) -> tuple[float, float]:
        """Return the seed std of each side (NaN with fewer than 2 seeds)."""
        return float(self.a.std(ddof=SEED_STD_DDOF)), float(self.b.std(ddof=SEED_STD_DDOF))

    @property
    def verdict(self) -> str:
        """Return the BUILD seed-noise label for the mean difference."""
        return seed_noise_label(self.diff, *self.stds)


def seed_noise_label(diff: float, std_a: float, std_b: float) -> str:
    """Return "within seed noise" if |diff| < the larger std; not assessable if a std is NaN."""
    if any(np.isnan(value) for value in (diff, std_a, std_b)):
        return NOT_ASSESSABLE
    return WITHIN_NOISE if abs(diff) < max(std_a, std_b) else BEYOND_NOISE


def by_seed(table: pd.DataFrame, value: str, **filters: Any) -> pd.Series:
    """Return `value` per seed for the rows matching `filters` (column=value)."""
    rows = table
    for column, wanted in filters.items():
        rows = rows[rows[column] == wanted]
    return rows.set_index("seed")[value]


@dataclass
class Numbers:
    """The flat `report_numbers` map, with the documentation of every key."""

    values: dict[str, Any]
    docs: dict[str, tuple[str, str, str]]

    def add(self, key: str, value: Any, meaning: str, formula: str, source: str) -> None:
        """Record one number; NaN becomes None and NumPy scalars plain Python values."""
        if key in self.values:
            raise ValueError(f"report number {key} added twice")
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, float) and np.isnan(value):
            value = None
        self.values[key] = value
        self.docs[key] = (meaning, formula, source)

    def add_comparison(self, key: str, c: Comparison, meaning: str, source: str) -> None:
        """Record a mean paired difference, its per-seed values and its seed-noise label."""
        self.add(key, c.diff, meaning, "mean over shared seeds of (a - b)", source)
        for seed, value in c.seed_diffs.items():
            self.add(f"{key}.seed{seed}", value, f"{meaning}, seed {seed}", "a - b", source)
        rule = "|mean difference| < larger of the two seed stds"
        self.add(f"{key}.seed_noise", c.verdict, f"Seed-noise label: {meaning}", rule, source)


def pct(value: float) -> str:
    """Format a fraction as a percentage with 1 decimal; `n/a` if undefined."""
    return "n/a" if pd.isna(value) else f"{100 * value:.1f}%"


def pp(value: float) -> str:
    """Format a fraction difference in signed percentage points; `n/a` if undefined."""
    return "n/a" if pd.isna(value) else f"{100 * value:+.1f} pp"


def pct_std(mean: float, std: float) -> str:
    """Format mean ± std as percentages."""
    return f"{pct(mean)} ± {pct(std)}"


def markdown_table(header: list[str], rows: list[list[Any]]) -> list[str]:
    """Return the lines of a pipe table."""
    lines = ["| " + " | ".join(header) + " |", "|" + " --- |" * len(header)]
    return lines + ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]


def _short(model: str) -> str:
    """Return the short model name used in report keys."""
    return MODEL_SHORT_NAMES[model]


def _configs(summary: pd.DataFrame) -> list[tuple[str, str]]:
    """Return the (model, strategy) pairs present in `summary`, in table order."""
    return list(summary[CONFIG_COLUMNS].drop_duplicates().itertuples(index=False, name=None))


def _seed_cell(c: Comparison) -> str:
    """Format the per-seed differences of a comparison."""
    return ", ".join(f"s{seed}: {pp(v)}" for seed, v in c.seed_diffs.items()) or "n/a"


def add_main_numbers(numbers: Numbers, summary: pd.DataFrame) -> None:
    """Record `main.*`: mean, std and seed count of the 8 metrics of each configuration."""
    source = "main_results.csv"
    for row in summary.itertuples(index=False):
        prefix = f"main.{_short(row.model)}.{row.strategy}"
        config = f"{row.model} + {row.strategy}"
        numbers.add(f"{prefix}.n_seeds", row.n_seeds, f"Seeds of {config}", "count", source)
        for metric in MAIN_METRICS:
            for kind in ("mean", "std"):
                formula = f"{kind} over seeds of {metric}" + (", ddof=1" if kind == "std" else "")
                value = getattr(row, f"{metric}_{kind}")
                key = f"{prefix}.{metric.lower()}.{kind}"
                numbers.add(key, value, f"{metric} of {config} ({kind})", formula, source)


def _record_rq1_drops(
    numbers: Numbers, prefix: str, config_name: str, config: pd.DataFrame
) -> list[float]:
    """Record the absolute and relative drop per severity of one config; return the drops."""
    source = "per_severity.csv"
    drops = []
    for severity in SEVERITIES:
        drop = float(config.loc[severity, "abs_degradation_mean"])
        relative = float(config.loc[severity, "rel_degradation_mean"])
        drops.append(drop)
        meaning = f"Absolute accuracy drop at severity {severity}, {config_name}"
        formula = "seed mean of (clean - mean accuracy at the severity)"
        numbers.add(f"{prefix}.abs_drop_sev{severity}", drop, meaning, formula, source)
        meaning = f"Relative accuracy drop at severity {severity}, {config_name}"
        formula = "seed mean of (clean - mean accuracy at the severity) / clean"
        numbers.add(f"{prefix}.rel_drop_sev{severity}", relative, meaning, formula, source)
    return drops


def answer_rq1(numbers: Numbers, per_severity: pd.DataFrame, summary: pd.DataFrame) -> list[str]:
    """RQ1: clean to corrupted drop per configuration and severity."""
    header = ["Model", "Strategy", "Clean", "Corrupted", "Abs. drop", "Rel. drop"]
    header += [f"Drop s{s}" for s in SEVERITIES] + ["Drop grows with severity"]
    rows = []
    for row in summary.itertuples(index=False):
        prefix = f"rq1.{_short(row.model)}.{row.strategy}"
        name = f"{row.model} + {row.strategy}"
        match = (per_severity["model"] == row.model) & (per_severity["strategy"] == row.strategy)
        drops = _record_rq1_drops(numbers, prefix, name, per_severity[match].set_index("severity"))
        grows = all(later > earlier for earlier, later in zip(drops, drops[1:], strict=False))
        meaning = f"Mean drop rises at every severity step, {name}"
        key = f"{prefix}.drop_grows_with_severity"
        numbers.add(key, grows, meaning, "drop at s < drop at s+1, s = 1..4", "per_severity.csv")
        rows.append(
            [
                row.model,
                row.strategy,
                pct_std(row.clean_acc_mean, row.clean_acc_std),
                pct_std(row.mean_corrupted_acc_mean, row.mean_corrupted_acc_std),
                pp(row.mean_abs_degradation_mean),
                pct(row.mean_rel_degradation_mean),
                *[pp(drop) for drop in drops],
                "yes" if grows else "no",
            ]
        )
    return ["## RQ1: degradation, clean to corrupted", "", *markdown_table(header, rows)]


def answer_rq2(numbers: Numbers, run_table: pd.DataFrame) -> list[str]:
    """RQ2: ResNet-18 minus Simple CNN per strategy, on clean, corrupted and relative metrics."""
    header = ["Strategy", "Metric", "Simple CNN", "ResNet-18", "ResNet-18 - CNN"]
    header += ["Per seed", "Seed noise"]
    rows = []
    for strategy in ANALYSED_STRATEGIES:
        for metric in ["clean_acc", *COMPARED_METRICS]:
            cnn = by_seed(run_table, metric, model="simple_cnn", strategy=strategy)
            resnet = by_seed(run_table, metric, model="resnet18", strategy=strategy)
            if cnn.empty or resnet.empty:
                continue
            c = Comparison(resnet, cnn)
            meaning = f"ResNet-18 minus Simple CNN, {metric}, {strategy}"
            key = f"rq2.{strategy}.{metric.lower()}.resnet18_minus_cnn"
            numbers.add_comparison(key, c, meaning, "main_results.csv")
            cells = [strategy, metric, pct(cnn.mean()), pct(resnet.mean()), pp(c.diff)]
            rows.append([*cells, _seed_cell(c), c.verdict])
    return ["## RQ2: architecture", "", *markdown_table(header, rows)]


def _rq3_metric_rows(
    numbers: Numbers, run_table: pd.DataFrame, model: str, strategy: str
) -> list[list[Any]]:
    """Record each main metric of a strategy against standard; return the table rows."""
    rows = []
    for metric in COMPARED_METRICS:
        base = by_seed(run_table, metric, model=model, strategy="standard")
        alt = by_seed(run_table, metric, model=model, strategy=strategy)
        if base.empty or alt.empty:
            continue
        c = Comparison(alt, base)
        meaning = f"{strategy} minus standard, {metric}, {model}"
        key = f"rq3.{_short(model)}.{strategy}.{metric.lower()}.diff_vs_standard"
        numbers.add_comparison(key, c, meaning, "main_results.csv")
        cells = [model, strategy, metric, pct(base.mean()), pct(alt.mean()), pp(c.diff)]
        rows.append([*cells, _seed_cell(c), c.verdict])
    return rows


def _rq3_family_rows(
    numbers: Numbers, families: pd.DataFrame, model: str, strategy: str
) -> list[list[Any]]:
    """Record the accuracy gain on each family against standard; return the table rows."""
    rows = []
    for family in MAIN_FAMILIES:
        where = {"model": model, "family": family}
        base = by_seed(families, "accuracy", strategy="standard", **where)
        alt = by_seed(families, "accuracy", strategy=strategy, **where)
        if base.empty or alt.empty:
            continue
        c = Comparison(alt, base)
        meaning = f"{strategy} minus standard, accuracy on {family}, {model}"
        key = f"rq3.{_short(model)}.{strategy}.{family}_acc_gain_vs_standard"
        numbers.add_comparison(key, c, meaning, "per_family.csv")
        rows.append([model, strategy, family, pp(c.diff), _seed_cell(c), c.verdict])
    return rows


def answer_rq3(numbers: Numbers, run_table: pd.DataFrame, families: pd.DataFrame) -> list[str]:
    """RQ3: AugMix and corruption-aware training against standard, overall and per family."""
    rows, family_rows = [], []
    for model in MODELS:
        for strategy in ALTERNATIVE_STRATEGIES:
            rows += _rq3_metric_rows(numbers, run_table, model, strategy)
            family_rows += _rq3_family_rows(numbers, families, model, strategy)
    header = ["Model", "Strategy", "Metric", "Standard", "Strategy value", "Strategy - standard"]
    family_header = ["Model", "Strategy", "Family", "Accuracy gain vs standard", "Per seed"]
    return [
        "## RQ3: strategies against standard training",
        "",
        *markdown_table([*header, "Per seed", "Seed noise"], rows),
        "",
        "Accuracy gain per family:",
        "",
        *markdown_table([*family_header, "Seed noise"], family_rows),
    ]


def answer_rq4(numbers: Numbers, groups: pd.DataFrame) -> list[str]:
    """RQ4: accuracy gain against standard on seen and unseen corruption groups."""
    header = ["Model", "Strategy", "Group", "Standard", "Strategy value", "Gain vs standard"]
    header += ["Per seed", "Seed noise"]
    rows = []
    for model in MODELS:
        for strategy in ALTERNATIVE_STRATEGIES:
            for group in ANALYSIS_GROUPS:
                where = {"model": model, "analysis_group": group}
                base = by_seed(groups, "accuracy", strategy="standard", **where)
                alt = by_seed(groups, "accuracy", strategy=strategy, **where)
                if base.empty or alt.empty:
                    continue
                c = Comparison(alt, base)
                meaning = f"{strategy} minus standard, accuracy on {group}, {model}"
                key = f"rq4.{_short(model)}.{strategy}.{group}_gain_vs_standard"
                numbers.add_comparison(key, c, meaning, "seen_unseen.csv")
                cells = [model, strategy, group, pct(base.mean()), pct(alt.mean()), pp(c.diff)]
                rows.append([*cells, _seed_cell(c), c.verdict])
    return ["## RQ4: seen and unseen corruptions", "", *markdown_table(header, rows)]


def _ranked(summary: pd.DataFrame, model: str, strategy: str, column: str) -> pd.Series:
    """Return mean absolute degradation per `column` value, largest first (stable on ties)."""
    rows = summary[(summary["model"] == model) & (summary["strategy"] == strategy)]
    values = rows.set_index(column)["abs_degradation_mean"]
    return values.sort_values(ascending=False, kind="stable")


def _record_rq5_ranks(
    numbers: Numbers, model: str, strategy: str, ranked: dict[str, pd.Series]
) -> None:
    """Record every family's and corruption's drop and the worst and mildest of each."""
    short, config = _short(model), f"{model} + {strategy}"
    formula = "seed mean of (clean - mean accuracy over its conditions)"
    for kind, source in (("family", "per_family.csv"), ("corruption", "per_corruption.csv")):
        values = ranked[kind]
        for name, value in values.items():
            key = f"rq5.{short}.{strategy}.{name}.abs_degradation"
            numbers.add(key, value, f"Absolute drop on {name}, {config}", formula, source)
        for label, name, how in (
            ("worst", values.index[0], "argmax"),
            ("mildest", values.index[-1], "argmin"),
        ):
            key = f"rq5.{label}_{kind}.{short}.{strategy}"
            meaning = f"The {label} {kind} by mean drop, {config}"
            numbers.add(key, name, meaning, f"{how} of the mean absolute degradation", source)


def answer_rq5(
    numbers: Numbers,
    per_family: pd.DataFrame,
    per_corruption: pd.DataFrame,
    families: pd.DataFrame,
) -> list[str]:
    """RQ5: drop per family, the worst and mildest family and corruption, and the worst gap."""
    header = ["Model", "Strategy", *MAIN_FAMILIES, "Worst family", "Mildest family"]
    header += ["Worst corruption", "Mildest corruption", "Worst - runner-up", "Seed noise"]
    rows = []
    for model, strategy in _configs(per_family):
        ranked = {
            "family": _ranked(per_family, model, strategy, "family"),
            "corruption": _ranked(per_corruption, model, strategy, "corruption"),
        }
        _record_rq5_ranks(numbers, model, strategy, ranked)
        order = ranked["family"].index
        drops = [
            by_seed(families, "abs_degradation", model=model, strategy=strategy, family=name)
            for name in order[:2]
        ]
        c = Comparison(*drops)
        meaning = f"Worst family minus runner-up ({order[0]} - {order[1]}), {model} + {strategy}"
        key = f"rq5.{_short(model)}.{strategy}.worst_family_gap"
        numbers.add_comparison(key, c, meaning, "per_family.csv")
        rows.append(
            [
                model,
                strategy,
                *[pp(ranked["family"][family]) for family in MAIN_FAMILIES],
                order[0],
                order[-1],
                ranked["corruption"].index[0],
                ranked["corruption"].index[-1],
                pp(c.diff),
                c.verdict,
            ]
        )
    return ["## RQ5: corruption families (mean absolute drop)", "", *markdown_table(header, rows)]


@dataclass(frozen=True)
class Stamp:
    """How complete the data behind an output is."""

    n_runs: int
    missing: tuple[str, ...]
    baseline_seeds: int

    @property
    def partial(self) -> bool:
        """Return True if any expected run is missing."""
        return bool(self.missing)

    @property
    def status(self) -> str:
        """Return `PARTIAL` or `FULL`."""
        return "PARTIAL" if self.partial else "FULL"

    def heading(self, title: str) -> str:
        """Return a Markdown title, stamped when partial."""
        return f"# PARTIAL: {title}" if self.partial else f"# {title}"

    def banner(self) -> list[str]:
        """Return the notice under a Markdown title (empty for full data)."""
        if not self.partial:
            return []
        total = self.n_runs + len(self.missing)
        return [
            "",
            f"**PARTIAL: {self.n_runs} of {total} runs. Not for the report or README.** "
            f"The mCE baseline has {self.baseline_seeds} of {len(SEEDS)} seeds, so mCE will "
            "change.",
            "",
            f"Missing: {', '.join(self.missing)}.",
        ]


STD_NOTE = "Mean ± std over seeds; std is the sample standard deviation (ddof=1)."
COMPARISON_NOTE = (
    "Differences are paired by seed and use only the seeds both sides have. `within seed noise`: "
    "the mean difference is smaller than the larger of the two seed stds."
)


def main_results_markdown(summary: pd.DataFrame, stamp: Stamp) -> str:
    """Return `main_results.md`: percentages with 1 decimal."""
    header = ["Model", "Strategy", "Seeds", *MAIN_METRICS]
    rows = [
        [
            row["model"],
            row["strategy"],
            row["n_seeds"],
            *[pct_std(row[f"{m}_mean"], row[f"{m}_std"]) for m in MAIN_METRICS],
        ]
        for _, row in summary.iterrows()
    ]
    lines = [stamp.heading("Main results"), *stamp.banner(), "", STD_NOTE, ""]
    return "\n".join([*lines, *markdown_table(header, rows)]) + "\n"


def seen_unseen_markdown(table: pd.DataFrame, stamp: Stamp) -> str:
    """Return `seen_unseen.md`: percentages with 1 decimal."""
    header = ["Model", "Strategy", "Group", "Seeds", "Accuracy", "Abs. degradation"]
    rows = [
        [
            row["model"],
            row["strategy"],
            row["analysis_group"],
            row["n_seeds"],
            pct_std(row["accuracy_mean"], row["accuracy_std"]),
            pct_std(row["abs_degradation_mean"], row["abs_degradation_std"]),
        ]
        for _, row in table.iterrows()
    ]
    lines = [stamp.heading("Seen and unseen corruptions"), *stamp.banner(), "", STD_NOTE, ""]
    return "\n".join([*lines, *markdown_table(header, rows)]) + "\n"


def report_numbers_markdown(numbers: Numbers, stamp: Stamp) -> str:
    """Return `report_numbers.md`: key, meaning, formula and source table of every number."""
    rows = [
        [f"`{key}`", *(cell.replace("|", "\\|") for cell in numbers.docs[key])]
        for key in sorted(numbers.values)
    ]
    header = ["Key", "Meaning", "Formula", "Source"]
    lines = [stamp.heading("Report numbers"), *stamp.banner(), "", STD_NOTE, ""]
    return "\n".join([*lines, *markdown_table(header, rows)]) + "\n"


def add_meta_numbers(numbers: Numbers, stamp: Stamp) -> None:
    """Record the status of the data behind the numbers."""
    source = "results/raw/"
    numbers.add("meta.status", stamp.status, "FULL or PARTIAL data", "all 18 runs present?", source)
    numbers.add("meta.n_runs", stamp.n_runs, "Runs analysed", "count of result CSVs", source)
    numbers.add(
        "meta.baseline_seeds", stamp.baseline_seeds, "Seeds in the mCE baseline", "count", source
    )
    numbers.add("meta.std_ddof", SEED_STD_DDOF, "ddof of every std", "constant", "src/analysis.py")


def build_tables(raw: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Return the per-run tables and their summaries, keyed by name."""
    baseline = baseline_table(raw)
    runs = run_metrics(raw, baseline)
    corruptions = corruption_breakdown(raw, baseline)
    severities = breakdown(raw, "severity", include_clean=True)
    families = breakdown(raw, "family")
    groups = breakdown(raw, "analysis_group")
    return {
        "runs": runs,
        "families": families,
        "groups": groups,
        "main_results": summarise(runs, MAIN_METRICS),
        "per_corruption": summarise(
            corruptions, [*BREAKDOWN_METRICS, "ce"], ["family", "corruption"]
        ),
        "per_severity": summarise(severities, BREAKDOWN_METRICS, ["severity"]),
        "per_family": summarise(families, BREAKDOWN_METRICS, ["family"]),
        "seen_unseen": summarise(groups, BREAKDOWN_METRICS, ["analysis_group"]),
    }


def answers_markdown(numbers: Numbers, tables: dict[str, pd.DataFrame], stamp: Stamp) -> str:
    """Return `rq_answers.md` and record the `rq*` numbers."""
    runs, families = tables["runs"], tables["families"]
    sections = [
        answer_rq1(numbers, tables["per_severity"], tables["main_results"]),
        answer_rq2(numbers, runs),
        answer_rq3(numbers, runs, families),
        answer_rq4(numbers, tables["groups"]),
        answer_rq5(numbers, tables["per_family"], tables["per_corruption"], families),
    ]
    lines = [stamp.heading("Data for RQ1-RQ5"), *stamp.banner(), "", STD_NOTE, COMPARISON_NOTE]
    for section in sections:
        lines += ["", *section]
    return "\n".join(lines) + "\n"


def write_csv(table: pd.DataFrame, path: Path, stamp: Stamp) -> None:
    """Write one table, with a leading `data_status` column when partial."""
    if stamp.partial:
        table = table.assign(data_status=stamp.status)
        table = table[["data_status", *table.columns[:-1]]]
    table.to_csv(path, index=False, lineterminator="\n")


def write_outputs(tables: dict[str, pd.DataFrame], tables_dir: Path, stamp: Stamp) -> list[Path]:
    """Write every C9 table to `tables_dir`; return the paths written."""
    tables_dir.mkdir(parents=True, exist_ok=True)
    numbers = Numbers({}, {})
    add_meta_numbers(numbers, stamp)
    add_main_numbers(numbers, tables["main_results"])
    answers = answers_markdown(numbers, tables, stamp)  # also records the rq* numbers
    number_map = json.dumps(numbers.values, indent=2, sort_keys=True, allow_nan=False) + "\n"
    texts = {
        ("rq_answers", "md"): answers,
        ("main_results", "md"): main_results_markdown(tables["main_results"], stamp),
        ("seen_unseen", "md"): seen_unseen_markdown(tables["seen_unseen"], stamp),
        ("report_numbers", "md"): report_numbers_markdown(numbers, stamp),
        ("report_numbers", "json"): number_map,
    }
    written = []
    for stem in ("main_results", "per_corruption", "per_severity", "per_family", "seen_unseen"):
        path = tables_dir / output_name(stem, "csv", stamp.partial)
        write_csv(tables[stem], path, stamp)
        written.append(path)
    for (stem, extension), text in texts.items():
        path = tables_dir / output_name(stem, extension, stamp.partial)
        path.write_text(text, encoding="utf-8")
        written.append(path)
    return written


def analyse(results_dir: Path, allow_partial: bool = False) -> list[Path]:
    """Build every table from `results_dir/raw` into `results_dir/tables`; return the paths."""
    raw_dir = results_dir / "raw"
    missing = missing_runs(raw_dir)
    if missing and not allow_partial:
        raise SystemExit(
            f"refusing to run: {len(missing)} of {len(expected_run_ids())} runs are missing "
            f"from {raw_dir}:\n  " + "\n  ".join(missing) + "\nUse --allow-partial for PARTIAL "
            "outputs."
        )
    raw = load_raw(raw_dir)
    model, strategy = BASELINE
    base_rows = raw[(raw["model"] == model) & (raw["strategy"] == strategy)]
    stamp = Stamp(raw["run_id"].nunique(), tuple(missing), base_rows["seed"].nunique())
    return write_outputs(build_tables(raw), results_dir / "tables", stamp)


def main() -> None:
    """Command-line entry point (CONTRACTS C5)."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="run with runs missing; every output is stamped PARTIAL",
    )
    args = parser.parse_args()
    written = analyse(env_path("RESULTS_DIR"), args.allow_partial)
    for path in written:
        print(path)
    if any(PARTIAL_SUFFIX in path.name for path in written):
        print("PARTIAL outputs: not for the report or README.", file=sys.stderr)


if __name__ == "__main__":
    main()
