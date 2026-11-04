"""Experiment matrix status and remaining GPU hours (Phase 08).

Usage:
    python -m scripts.plan                        # the run table and the remaining GPU hours
    python -m scripts.plan --run-ids              # run ids in execution order, one per line
    python -m scripts.plan --next-step RUN_ID [--remaining-s S]   # what run_all.sh does next

`--models`, `--strategies` and `--seeds` narrow the first two forms (defaults: both models,
standard augmix corruption_aware, seeds 0 1 2).

Status comes from the run folders in $RUNS_DIR, then `$RESULTS_DIR/runs.csv` for a run with no
folder here, and the result CSVs in `$RESULTS_DIR/raw/`:

    done      finished and evaluated
    evaluate  finished, not evaluated yet
    running   started, not finished (it may be paused until the next Colab session)
    failed    the last session of the run stopped with an error
    missing   not started
    no-meta   results exist but no run metadata does; a human must look at it

This module imports no torch, because run_all.sh calls it before every step.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from src.constants import MODEL_SHORT_NAMES, SEEDS, STRATEGIES

REPO_ROOT = Path(__file__).resolve().parent.parent
PILOT_REPORT = REPO_ROOT / "results" / "pilot_report.md"

SHORT_MODELS = tuple(MODEL_SHORT_NAMES.values())
MATRIX_STRATEGIES = ("standard", "augmix", "corruption_aware")
RUN_ID = re.compile(r"(?P<short>[a-z0-9]+)-(?P<strategy>[a-z_]+)-s(?P<seed>\d+)")
# Rows of the pilot report's two timing tables.
PILOT_EPOCH_ROW = re.compile(r"^\| (cnn|resnet18) (\w+) \|.*\| ([\d.]+)[^|]*\| [^|]*\|$")
PILOT_EVAL_ROW = re.compile(r"^\| (simple_cnn|resnet18) \| (\d+) \|")


@dataclass(frozen=True)
class RunState:
    """Where one run stands."""

    run_id: str
    status: str
    epochs_done: int
    epochs_total: int
    commit: str = ""
    clean_accuracy: str = ""


def env_dir(name: str, default: str) -> Path:
    """Return the folder in env var `name` (CONTRACTS C2); src.utils.env_path would import torch."""
    return Path(os.environ.get(name) or default)


def parse_run_id(run_id: str) -> tuple[str, str, int]:
    """Return (short model, strategy, seed) of a real run id; raise ValueError for anything else."""
    match = RUN_ID.fullmatch(run_id)
    if (
        match is None
        or match["short"] not in SHORT_MODELS
        or match["strategy"] not in STRATEGIES
        or int(match["seed"]) not in SEEDS
    ):
        raise ValueError(f"not a matrix run id: {run_id!r}")
    return match["short"], match["strategy"], int(match["seed"])


def config_epochs(short: str, strategy: str) -> int:
    """Return the epoch count in a run's config, which D4 freezes."""
    path = REPO_ROOT / "configs" / f"{short}_{strategy}.yaml"
    return int(yaml.safe_load(path.read_text())["epochs"])


def read_pilot_times(report: Path = PILOT_REPORT) -> tuple[dict, dict]:
    """Return the pilot's ({(short, strategy): s per epoch}, {short: s per evaluation})."""
    epoch_s: dict[tuple[str, str], float] = {}
    eval_s: dict[str, float] = {}
    for line in report.read_text().splitlines():
        if match := PILOT_EPOCH_ROW.match(line):
            epoch_s[(match[1], match[2])] = float(match[3])
        elif match := PILOT_EVAL_ROW.match(line):
            eval_s[MODEL_SHORT_NAMES[match[1]]] = float(match[2])
    expected = {(m, s) for m in SHORT_MODELS for s in MATRIX_STRATEGIES}
    if set(epoch_s) != expected or set(eval_s) != set(SHORT_MODELS):
        raise ValueError(f"{report}: timing tables not as expected")
    return epoch_s, eval_s


def pilot_epoch_seconds(pilot: dict, short: str, strategy: str) -> float:
    """Return the pilot's time per epoch. S0 was not timed, so it uses S1's, its nearest config."""
    return pilot.get((short, strategy), pilot[(short, "standard")])


def matrix_run_ids(
    models: list[str], strategies: list[str], seeds: list[int], pilot: dict
) -> list[str]:
    """Return run ids in execution order: lowest seed first, cheapest first within a seed.

    Cost comes from the pilot only, so that the order does not change between sessions.
    """

    def cost(run: tuple[int, str, str]) -> float:
        seed, short, strategy = run
        return pilot_epoch_seconds(pilot, short, strategy) * config_epochs(short, strategy)

    runs = [(seed, m, s) for seed in seeds for m in models for s in strategies]
    return [f"{m}-{s}-s{seed}" for seed, m, s in sorted(runs, key=lambda r: (r[0], cost(r)))]


def read_log_rows(path: Path) -> list[dict[str, str]]:
    """Return the complete rows of a train log; a row cut off by a crash is left out."""
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return [row for row in csv.DictReader(f) if None not in row.values()]


def read_registry(results_dir: Path) -> dict[str, dict[str, str]]:
    """Return `results_dir`/runs.csv keyed by run id; empty if the file does not exist."""
    path = results_dir / "runs.csv"
    if not path.exists():
        return {}
    with path.open(newline="") as f:
        return {row["run_id"]: row for row in csv.DictReader(f)}


def run_state(run_id: str, runs_dir: Path, results_dir: Path, registry: dict) -> RunState:
    """Return where `run_id` stands. A run folder wins over the registry, being fresher."""
    short, strategy, _ = parse_run_id(run_id)
    total = config_epochs(short, strategy)
    row = registry.get(run_id, {})
    meta_path = runs_dir / run_id / "run_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else row
    has_results = (results_dir / "raw" / f"{run_id}.csv").exists()
    meta_status = meta.get("status")
    done = len(read_log_rows(runs_dir / run_id / "train_log.csv"))
    if not meta:
        status = "no-meta" if has_results else "missing"
    elif meta_status == "finished":
        status, done = ("done" if has_results else "evaluate"), total
    elif has_results:
        status = "no-meta"  # results from a run that says it never finished: never train over it
    else:
        status = str(meta_status)
    return RunState(
        run_id,
        status,
        done,
        total,
        str(meta.get("git_commit") or "")[:7],
        row.get("clean_accuracy", ""),
    )


def seconds_per_epoch(short: str, strategy: str, results_dir: Path, pilot: dict) -> float:
    """Return the mean epoch time of this config's finished runs, else the pilot's.

    `$RESULTS_DIR/raw/{run_id}_train_log.csv` is written only when a run finishes.
    """
    logs = sorted((results_dir / "raw").glob(f"{short}-{strategy}-s*_train_log.csv"))
    times = [float(row["epoch_time_s"]) for log in logs for row in read_log_rows(log)]
    return sum(times) / len(times) if times else pilot_epoch_seconds(pilot, short, strategy)


def next_epoch_seconds(run_id: str, runs_dir: Path, results_dir: Path, pilot: dict) -> float:
    """Return the expected time of the run's next epoch: its last epoch, if it has one."""
    rows = read_log_rows(runs_dir / run_id / "train_log.csv")
    if rows:
        return float(rows[-1]["epoch_time_s"])
    short, strategy, _ = parse_run_id(run_id)
    return seconds_per_epoch(short, strategy, results_dir, pilot)


def next_step(
    state: RunState,
    next_epoch_s: float,
    evaluate_s: float,
    remaining_s: float | None,
) -> str:
    """Return the line run_all.sh acts on.

    `skip`; `evaluate`; `train STOP DONE`, where STOP is the epoch to stop after, or `-` for the
    end of the run, and DONE is the epochs logged so far; `stop REASON`, the budget is spent;
    `refuse REASON`, a human must look.
    """
    if state.status == "done":
        return "skip"
    if state.status == "no-meta":
        return f"refuse {state.run_id}: has results but no finished run metadata"
    if state.status == "evaluate":
        if remaining_s is not None and remaining_s < evaluate_s:
            return f"stop evaluating needs about {evaluate_s:.0f} s, {remaining_s:.0f} s left"
        return "evaluate"
    if remaining_s is None:
        return f"train - {state.epochs_done}"
    fits = int(remaining_s // next_epoch_s)
    if fits < 1:
        return f"stop the next epoch needs about {next_epoch_s:.0f} s, {remaining_s:.0f} s left"
    stop_after = state.epochs_done + fits
    return f"train {'-' if stop_after >= state.epochs_total else stop_after} {state.epochs_done}"


def remaining_seconds(
    state: RunState, results_dir: Path, pilot: dict, eval_pilot: dict
) -> tuple[float, float]:
    """Return the (training, evaluation) seconds a run still needs."""
    short, strategy, _ = parse_run_id(state.run_id)
    epochs_left = state.epochs_total - state.epochs_done
    training = epochs_left * seconds_per_epoch(short, strategy, results_dir, pilot)
    return training, 0.0 if state.status == "done" else eval_pilot[short]


def print_table(states: list[RunState], results_dir: Path, pilot: dict, eval_pilot: dict) -> None:
    """Print one line per run, the status counts and the remaining GPU hours."""
    print(f"{'run_id':<30} {'status':<9} {'epochs':>6} {'clean_acc':>9}  commit")
    for s in states:
        epochs = f"{s.epochs_done}/{s.epochs_total}"
        print(f"{s.run_id:<30} {s.status:<9} {epochs:>6} {s.clean_accuracy:>9}  {s.commit}")
    counts = {}
    for s in states:
        counts[s.status] = counts.get(s.status, 0) + 1
    print(", ".join(f"{name} {n}" for name, n in sorted(counts.items())), f"of {len(states)} runs")
    left = [remaining_seconds(s, results_dir, pilot, eval_pilot) for s in states]
    training, evaluation = (sum(values) / 3600 for values in zip(*left, strict=True))
    print(
        f"remaining GPU hours: {training + evaluation:.2f} "
        f"(training {training:.2f}, evaluation {evaluation:.2f}; session setup not included)"
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description="Experiment matrix status.")
    parser.add_argument("--models", nargs="+", choices=SHORT_MODELS, default=list(SHORT_MODELS))
    parser.add_argument(
        "--strategies", nargs="+", choices=STRATEGIES, default=list(MATRIX_STRATEGIES)
    )
    parser.add_argument("--seeds", nargs="+", type=int, choices=SEEDS, default=list(SEEDS))
    parser.add_argument("--run-ids", action="store_true", help="print run ids in execution order")
    parser.add_argument("--next-step", metavar="RUN_ID", help="print run_all.sh's next step")
    parser.add_argument("--remaining-s", type=float, help="seconds left in the session budget")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Command-line entry point."""
    args = parse_args(argv)
    runs_dir, results_dir = env_dir("RUNS_DIR", "./runs"), env_dir("RESULTS_DIR", "./results")
    pilot, eval_pilot = read_pilot_times()
    registry = read_registry(results_dir)
    if args.next_step:
        short = parse_run_id(args.next_step)[0]
        state = run_state(args.next_step, runs_dir, results_dir, registry)
        epoch_s = next_epoch_seconds(args.next_step, runs_dir, results_dir, pilot)
        print(next_step(state, epoch_s, eval_pilot[short], args.remaining_s))
        return
    run_ids = matrix_run_ids(args.models, args.strategies, args.seeds, pilot)
    if args.run_ids:
        print("\n".join(run_ids))
        return
    states = [run_state(r, runs_dir, results_dir, registry) for r in run_ids]
    print_table(states, results_dir, pilot, eval_pilot)


if __name__ == "__main__":
    main()
