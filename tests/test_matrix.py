"""Phase 08 orchestration on fake runs: order, skipping, budget, resume, sync, import checks."""

import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from scripts import collect_results, plan
from scripts.collect_results import collect
from src.constants import CORRUPTIONS, SEVERITIES
from src.evaluate import CSV_COLUMNS, family_of, group_of
from src.train import LOG_COLUMNS

REPO = Path(__file__).resolve().parent.parent
ALL_RUNS = 18
PILOT_RUN = "cnn-standard-s0"
EPOCH_SECONDS = 60.0

# Stands in for `python`: real for scripts.plan, a fake for training, evaluation and the registry.
FAKE_PYTHON = """#!/usr/bin/env bash
if [[ "$1 $2" == "-m src.train" || "$1 $2" == "-m src.evaluate" ]]; then
    exec {real} {fakes} "$2" "${{@:3}}"
elif [[ "$2" == scripts.registry ]]; then
    echo "registry rebuilt" >>"$FAKE_CALLS"
    exit 0
fi
exec {real} "$@"
"""

# Stands in for `date`, so that a fake epoch uses up budget without taking any time.
FAKE_DATE = """#!/usr/bin/env bash
cat "$FAKE_CLOCK"
"""

FAKES = """
import json, os, sys
from pathlib import Path

which, args = sys.argv[1], sys.argv[2:]
log = Path(os.environ["FAKE_CALLS"])
runs = Path(os.environ["RUNS_DIR"])
results = Path(os.environ["RESULTS_DIR"]) / "raw"
results.mkdir(parents=True, exist_ok=True)
clock = Path(os.environ["FAKE_CLOCK"])

def note(text):
    with log.open("a") as f:
        f.write(text + "\\n")

if which == "src.evaluate":
    run_id = Path(args[args.index("--run-dir") + 1]).name
    note(f"evaluate {run_id} {' '.join(args[2:])}".strip())
    clock.write_text(str(int(clock.read_text()) + 200))
    if os.environ.get("FAKE_FAIL") == f"evaluate {run_id}":
        sys.exit(3)
    (results / f"{run_id}_confusion.npz").write_text("npz")
    (results / f"{run_id}.csv").write_text("csv")
    sys.exit(0)

config = Path(args[args.index("--config") + 1]).stem
seed = args[args.index("--seed") + 1]
short, strategy = config.split("_", 1)
run_id = f"{short}-{strategy}-s{seed}"
total = 40 if short == "resnet18" else 30
folder = runs / run_id
folder.mkdir(parents=True, exist_ok=True)
log_path = folder / "train_log.csv"
done = len(log_path.read_text().splitlines()) - 1 if log_path.exists() else 0
stop = os.environ.get("STOP_AFTER_EPOCHS")
last = min(total, int(stop)) if stop else total
note(f"train {run_id} from {done} to {last}")
if os.environ.get("FAKE_FAIL") == f"train {run_id}":
    (folder / "run_meta.json").write_text(json.dumps({"status": "failed", "git_commit": "abc"}))
    sys.exit(3)
if not log_path.exists():
    log_path.write_text("epoch,epoch_time_s\\n")
if os.environ.get("FAKE_STUCK") != run_id:
    with log_path.open("a") as f:
        for epoch in range(done + 1, last + 1):
            f.write(f"{epoch},{float(os.environ['FAKE_EPOCH_S'])}\\n")
    spent = (last - done) * float(os.environ["FAKE_EPOCH_S"])
    clock.write_text(str(int(clock.read_text()) + int(spent)))
status = "finished" if last == total else "running"
(folder / "run_meta.json").write_text(json.dumps({"status": status, "git_commit": "abc"}))
(folder / "config_resolved.yaml").write_text("a: 1\\n")
if status == "finished":
    (results / f"{run_id}_train_log.csv").write_text(log_path.read_text())
"""


class Matrix:
    """A folder of fake runs and one way to call the real run_all.sh on them."""

    def __init__(self, tmp_path: Path) -> None:
        self.runs = tmp_path / "runs"
        self.results = tmp_path / "results"
        self.calls = tmp_path / "calls.log"
        self.clock = tmp_path / "clock"
        self.sync = tmp_path / "drive"
        self.bin = tmp_path / "bin"
        self.bin.mkdir()
        self.runs.mkdir()
        self.calls.touch()
        self.clock.write_text("1000000")
        fakes = tmp_path / "fakes.py"
        fakes.write_text(FAKES)
        python = self.bin / "fake-python"
        python.write_text(FAKE_PYTHON.format(real=sys.executable, fakes=fakes))
        python.chmod(0o755)
        date = self.bin / "date"
        date.write_text(FAKE_DATE)
        date.chmod(0o755)

    def run(self, *args: str, **env: str) -> subprocess.CompletedProcess:
        """Run scripts/run_all.sh with the fakes in place."""
        return subprocess.run(
            ["bash", str(REPO / "scripts" / "run_all.sh"), *args],
            capture_output=True,
            text=True,
            timeout=30,  # a loop that never ends must fail the test, not hang it
            env={
                "PATH": f"{self.bin}:{os.environ['PATH']}",
                "PYTHON": str(self.bin / "fake-python"),
                "RUNS_DIR": str(self.runs),
                "RESULTS_DIR": str(self.results),
                "FAKE_CALLS": str(self.calls),
                "FAKE_CLOCK": str(self.clock),
                "FAKE_EPOCH_S": str(EPOCH_SECONDS),
                **env,
            },
        )

    def trained(self) -> list[str]:
        """Return the training calls so far, as lines of the call log."""
        return [c for c in self.calls.read_text().splitlines() if c.startswith("train ")]

    def evaluated(self) -> list[str]:
        """Return the evaluation calls so far, as lines of the call log."""
        return [c for c in self.calls.read_text().splitlines() if c.startswith("evaluate ")]

    def add_finished(self, run_id: str, evaluated: bool = True) -> None:
        """Create a finished run, as the pilot left it."""
        folder = self.runs / run_id
        folder.mkdir()
        (folder / "run_meta.json").write_text(
            json.dumps({"status": "finished", "git_commit": "pilot"})
        )
        if evaluated:
            (self.results / "raw").mkdir(parents=True, exist_ok=True)
            (self.results / "raw" / f"{run_id}.csv").write_text("csv")


@pytest.fixture
def matrix(tmp_path: Path) -> Matrix:
    return Matrix(tmp_path)


def test_default_order_is_by_seed_then_cheapest():
    pilot, _ = plan.read_pilot_times()
    ids = plan.matrix_run_ids(["cnn", "resnet18"], list(plan.MATRIX_STRATEGIES), [0, 1, 2], pilot)
    seed_0 = [
        "cnn-standard-s0",
        "cnn-corruption_aware-s0",
        "resnet18-standard-s0",
        "resnet18-corruption_aware-s0",
        "cnn-augmix-s0",
        "resnet18-augmix-s0",
    ]
    assert len(set(ids)) == ALL_RUNS
    assert ids[:6] == seed_0
    assert ids[6:12] == [r.replace("-s0", "-s1") for r in seed_0]
    assert ids[12:] == [r.replace("-s0", "-s2") for r in seed_0]


def test_full_run_trains_each_id_once_and_keeps_the_pilot(matrix):
    matrix.add_finished(PILOT_RUN)
    result = matrix.run()
    assert result.returncode == 0, result.stderr
    assert len(matrix.trained()) == ALL_RUNS - 1
    assert len(matrix.evaluated()) == ALL_RUNS - 1
    assert not any(PILOT_RUN in call for call in matrix.calls.read_text().splitlines())
    assert f"{PILOT_RUN}: already evaluated, skipping" in result.stdout
    assert "done 18 of 18 runs" in result.stdout

    # A second pass finds nothing to do.
    before = matrix.calls.read_text().count("train ")
    assert matrix.run().returncode == 0
    assert matrix.calls.read_text().count("train ") == before


def test_filters_narrow_the_matrix(matrix):
    result = matrix.run("--models", "cnn", "--strategies", "standard", "--seeds", "1", "2")
    assert result.returncode == 0, result.stderr
    assert matrix.trained() == [
        "train cnn-standard-s1 from 0 to 30",
        "train cnn-standard-s2 from 0 to 30",
    ]


def test_finished_unevaluated_run_is_only_evaluated(matrix):
    matrix.add_finished(PILOT_RUN, evaluated=False)
    args = ("--models", "cnn", "--strategies", "standard", "--seeds", "0")
    assert matrix.run(*args).returncode == 0
    assert matrix.trained() == []
    assert matrix.evaluated() == [f"evaluate {PILOT_RUN}"]


def test_leftover_confusion_file_is_replaced_but_results_never_are(matrix):
    matrix.add_finished(PILOT_RUN, evaluated=False)
    raw = matrix.results / "raw"
    raw.mkdir(parents=True)
    (raw / f"{PILOT_RUN}_confusion.npz").write_text("cut short")
    args = ("--models", "cnn", "--strategies", "standard", "--seeds", "0")
    assert matrix.run(*args).returncode == 0
    assert matrix.evaluated() == [f"evaluate {PILOT_RUN} --force"]


def test_budget_stops_at_the_epoch_that_fits_and_the_next_session_resumes(matrix):
    args = ("--models", "cnn", "--strategies", "standard", "--seeds", "1")
    # No log yet: the pilot's 20.667 s per epoch, so 2 minutes fit 5 epochs of the fake's 60 s.
    result = matrix.run(*args, SESSION_BUDGET_MIN="2")
    assert result.returncode == 0, result.stderr
    assert "session budget reached before cnn-standard-s1" in result.stdout
    assert matrix.trained() == ["train cnn-standard-s1 from 0 to 5"]
    meta = json.loads((matrix.runs / "cnn-standard-s1" / "run_meta.json").read_text())
    assert meta["status"] == "running"

    # Now the run's own last epoch (60 s) is the estimate: 5 minutes fit 5 more.
    assert matrix.run(*args, SESSION_BUDGET_MIN="5").returncode == 0
    assert matrix.trained()[1:] == ["train cnn-standard-s1 from 5 to 10"]

    # Enough for the rest and the evaluation: no stop epoch at all.
    assert matrix.run(*args, SESSION_BUDGET_MIN="30").returncode == 0
    assert matrix.trained()[2:] == ["train cnn-standard-s1 from 10 to 30"]
    assert matrix.evaluated() == ["evaluate cnn-standard-s1"]


def test_zero_budget_trains_nothing(matrix):
    result = matrix.run(SESSION_BUDGET_MIN="0")
    assert result.returncode == 0
    assert matrix.trained() == []
    assert "session budget reached" in result.stdout


def test_no_time_for_the_evaluation_stops_cleanly(matrix):
    matrix.add_finished(PILOT_RUN, evaluated=False)
    args = ("--models", "cnn", "--strategies", "standard", "--seeds", "0")
    result = matrix.run(*args, SESSION_BUDGET_MIN="1")
    assert result.returncode == 0
    assert matrix.evaluated() == []
    assert "evaluating needs about 182 s" in result.stdout


def test_bad_budget_is_refused(matrix):
    result = matrix.run(SESSION_BUDGET_MIN="two hours")
    assert result.returncode == 2
    assert matrix.trained() == []


@pytest.mark.parametrize("step", ["train", "evaluate"])
def test_a_failed_step_stops_the_matrix(matrix, step):
    result = matrix.run(FAKE_FAIL=f"{step} cnn-standard-s0")
    assert result.returncode == 1
    assert f"ERROR: {'training' if step == 'train' else 'evaluation'} of cnn-standard-s0" in (
        result.stderr
    )
    assert len(matrix.trained()) == 1
    assert "remaining GPU hours" in result.stdout


def test_failed_run_resumes_in_the_next_session(matrix):
    assert matrix.run(FAKE_FAIL="train cnn-standard-s0").returncode == 1
    args = ("--models", "cnn", "--strategies", "standard", "--seeds", "0")
    assert matrix.run(*args).returncode == 0
    assert len(matrix.trained()) == 2
    assert matrix.evaluated() == ["evaluate cnn-standard-s0"]


def test_a_run_that_makes_no_progress_is_not_retried_forever(matrix):
    result = matrix.run(SESSION_BUDGET_MIN="5", FAKE_STUCK="cnn-standard-s0")
    assert result.returncode == 1
    assert "made no progress" in result.stderr
    assert len(matrix.trained()) == 1


def test_results_without_metadata_are_never_trained_over(matrix):
    (matrix.results / "raw").mkdir(parents=True)
    (matrix.results / "raw" / f"{PILOT_RUN}.csv").write_text("csv")
    result = matrix.run()
    assert result.returncode == 1
    assert "has results but no finished run metadata" in result.stderr
    assert matrix.trained() == []


def test_results_reach_the_sync_folder_after_each_evaluated_run(matrix):
    args = ("--models", "cnn", "--strategies", "standard", "--seeds", "0")
    (matrix.results).mkdir()
    (matrix.results / "runs.csv").write_text("registry")
    assert matrix.run(*args, RESULTS_SYNC_DIR=str(matrix.sync)).returncode == 0
    names = sorted(p.name for p in (matrix.sync / "raw").iterdir())
    assert names == [f"{PILOT_RUN}.csv", f"{PILOT_RUN}_confusion.npz", f"{PILOT_RUN}_train_log.csv"]
    folder = matrix.sync / "run_folders" / PILOT_RUN
    assert sorted(p.name for p in folder.iterdir()) == [
        "config_resolved.yaml",
        "run_meta.json",
        "train_log.csv",
    ]
    assert (matrix.sync / "runs.csv").read_text() == "registry"


def test_a_session_that_stops_early_still_saves_the_paused_run(matrix):
    args = ("--models", "cnn", "--strategies", "standard", "--seeds", "1")
    result = matrix.run(*args, SESSION_BUDGET_MIN="2", RESULTS_SYNC_DIR=str(matrix.sync))
    assert result.returncode == 0
    meta = matrix.sync / "run_folders" / "cnn-standard-s1" / "run_meta.json"
    assert json.loads(meta.read_text())["status"] == "running"


def test_status_table_and_remaining_hours_on_the_real_pilot_report(tmp_path, capsys, monkeypatch):
    runs, results = tmp_path / "runs", tmp_path / "results"
    (runs / "cnn-standard-s1").mkdir(parents=True)
    (runs / "cnn-standard-s1" / "run_meta.json").write_text(
        json.dumps({"status": "failed", "git_commit": "abcdef012345"})
    )
    (runs / "cnn-standard-s1" / "train_log.csv").write_text("epoch,epoch_time_s\n1,20\n2,20\n")
    monkeypatch.setenv("RUNS_DIR", str(runs))
    monkeypatch.setenv("RESULTS_DIR", str(results))
    plan.main([])
    out = capsys.readouterr().out
    assert "failed 1, missing 17 of 18 runs" in out
    assert "2/30" in out
    assert "abcdef0" in out
    # The report's own total, 11.23 h, less the 2 epochs of 20.667 s already done.
    assert "remaining GPU hours: 11.22" in out


def test_measured_times_replace_the_pilot(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "cnn-augmix-s0_train_log.csv").write_text("epoch,epoch_time_s\n1,10\n2,30\n")
    pilot, _ = plan.read_pilot_times()
    assert plan.seconds_per_epoch("cnn", "augmix", tmp_path, pilot) == 20.0
    assert plan.seconds_per_epoch("resnet18", "augmix", tmp_path, pilot) == 122.687


def test_registry_row_is_used_when_there_is_no_run_folder(tmp_path):
    row = {"status": "finished", "git_commit": "abcdef012345", "clean_accuracy": "0.9"}
    state = plan.run_state("cnn-standard-s1", tmp_path, tmp_path, {"cnn-standard-s1": row})
    assert (state.status, state.clean_accuracy, state.commit) == ("evaluate", "0.9", "abcdef0")


@pytest.mark.parametrize(
    "run_id", ["smoke-cnn-standard-s0", "timing-cnn_augmix", "cnn-standard-s3", "cnn-other-s0"]
)
def test_only_real_run_ids_parse(run_id):
    with pytest.raises(ValueError, match="not a matrix run id"):
        plan.parse_run_id(run_id)


# --- collect_results ---


def write_result_files(raw: Path, run_id: str, accuracy: float = 0.8) -> None:
    """Write the 3 C7 files of a run; every condition has `accuracy` on 10,000 images."""
    short, strategy, seed = plan.parse_run_id(run_id)
    raw.mkdir(parents=True, exist_ok=True)
    rows = [("clean", 0)] + [(c, s) for c in CORRUPTIONS for s in SEVERITIES]
    matrices = {}
    with (raw / f"{run_id}.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for corruption, severity in rows:
            writer.writerow(
                {
                    "run_id": run_id,
                    "model": collect_results.LONG_MODEL[short],
                    "strategy": strategy,
                    "seed": seed,
                    "corruption": corruption,
                    "family": family_of(corruption),
                    "group": group_of(corruption),
                    "severity": severity,
                    "n_images": 10000,
                    "accuracy": accuracy,
                    "macro_f1": 0.79,
                    "loss": 0.5,
                    "git_commit": "abc",
                    "evaluated_at": "2026-11-05T00:00:00+00:00",
                }
            )
            matrix = np.zeros((10, 10), dtype=np.int64)
            np.fill_diagonal(matrix, int(1000 * accuracy))
            matrix[0, 1] = 10000 - matrix.sum()
            matrices[f"{corruption}__s{severity}"] = matrix
    np.savez(raw / f"{run_id}_confusion.npz", **matrices)
    epochs = plan.config_epochs(short, strategy)
    lines = [",".join(LOG_COLUMNS)] + [
        f"{e},0.1,1.0,0.5,1.0,0.5,20.0" for e in range(1, epochs + 1)
    ]
    (raw / f"{run_id}_train_log.csv").write_text("\n".join(lines) + "\n")


def write_download(source: Path, run_id: str, status: str = "finished") -> None:
    """Write one evaluated run the way a session leaves it in the Drive results folder."""
    short, strategy, seed = plan.parse_run_id(run_id)
    write_result_files(source / "raw", run_id)
    folder = source / "run_folders" / run_id
    folder.mkdir(parents=True)
    meta = {
        "run_id": run_id,
        "model": collect_results.LONG_MODEL[short],
        "strategy": strategy,
        "seed": seed,
        "epochs": plan.config_epochs(short, strategy),
        "git_commit": "abcdef012345",
        "git_dirty": False,
        "device_name": "Tesla T4",
        "status": status,
    }
    (folder / "run_meta.json").write_text(json.dumps(meta))
    (folder / "config_resolved.yaml").write_text("a: 1\n")


@pytest.fixture
def dirs(tmp_path: Path) -> tuple[Path, Path, Path]:
    source, results, runs = tmp_path / "download", tmp_path / "results", tmp_path / "runs"
    runs.mkdir()
    return source, results, runs


def registry_rows(results: Path) -> list[dict[str, str]]:
    with (results / "runs.csv").open(newline="") as f:
        return list(csv.DictReader(f))


def test_c7_columns_are_the_evaluators():
    assert collect_results.C7_COLUMNS == CSV_COLUMNS
    assert collect_results.C6_LOG_COLUMNS == LOG_COLUMNS


def test_collect_imports_a_good_run_and_rebuilds_the_registry(dirs):
    source, results, runs = dirs
    write_download(source, "resnet18-augmix-s2")
    found = collect(source, results, runs)
    assert found.problems == []
    assert sorted(p.name for p in (results / "raw").iterdir()) == [
        "resnet18-augmix-s2.csv",
        "resnet18-augmix-s2_confusion.npz",
        "resnet18-augmix-s2_train_log.csv",
    ]
    assert (runs / "resnet18-augmix-s2" / "run_meta.json").exists()
    rows = registry_rows(results)
    assert [r["run_id"] for r in rows] == ["resnet18-augmix-s2"]
    assert rows[0]["has_results"] == "True"
    assert rows[0]["clean_accuracy"] == "0.8"


def test_collect_finds_the_results_folder_inside_a_download(dirs):
    source, results, runs = dirs
    write_download(source / "results", "cnn-augmix-s1")
    assert collect(source, results, runs).evaluated == ["cnn-augmix-s1"]


def test_collect_again_copies_nothing(dirs):
    source, results, runs = dirs
    write_download(source, "cnn-augmix-s1")
    collect(source, results, runs)
    found = collect(source, results, runs)
    assert found.problems == []
    assert found.copies == []


def test_collect_refuses_a_duplicate_with_different_content_and_copies_nothing(dirs):
    source, results, runs = dirs
    write_download(source, "cnn-augmix-s1")
    collect(source, results, runs)
    original = (results / "raw" / "cnn-augmix-s1.csv").read_bytes()
    write_result_files(source / "raw", "cnn-augmix-s1", accuracy=0.7)
    write_download(source, "cnn-augmix-s2")
    found = collect(source, results, runs)
    assert any("differs from" in p for p in found.problems)
    assert (results / "raw" / "cnn-augmix-s1.csv").read_bytes() == original
    assert not (results / "raw" / "cnn-augmix-s2.csv").exists()


def edit_csv(path: Path, old: str, new: str) -> None:
    text = path.read_text()
    assert old in text
    path.write_text(text.replace(old, new, 1))


@pytest.mark.parametrize(
    ("change", "expected"),
    [
        (lambda p: edit_csv(p, ",10000,", ",9000,"), "n_images"),
        (lambda p: edit_csv(p, ",0.8,", ",1.8,"), "accuracy is"),
        (lambda p: edit_csv(p, ",0.8,", ",nan,"), "accuracy is"),
        (lambda p: edit_csv(p, ",0.5,", ",inf,"), "loss is"),
        (
            lambda p: edit_csv(p, "gaussian_noise,noise,seen", "gaussian_noise,noise,unseen"),
            "group",
        ),
        (lambda p: edit_csv(p, "accuracy,macro_f1", "macro_f1,accuracy"), "columns"),
        (lambda p: edit_csv(p, ",clean,clean,0,", ",fog,weather,unseen,1,"), "fields"),
        (lambda p: edit_csv(p, "clean,clean,clean,0", "fog,weather,unseen,1"), "expected 76"),
        (lambda p: p.write_text("".join(p.read_text().splitlines(True)[:-1])), "75 rows"),
        (lambda p: edit_csv(p, "resnet18,", "simple_cnn,"), "model is"),
        (lambda p: edit_csv(p, "abc,", ","), "empty"),
        (lambda p: edit_csv(p, ",0.5,", ",lots,"), "does not parse"),
    ],
)
def test_collect_rejects_a_bad_results_csv(dirs, change, expected):
    source, results, runs = dirs
    write_download(source, "resnet18-standard-s0")
    change(source / "raw" / "resnet18-standard-s0.csv")
    found = collect(source, results, runs)
    assert any(expected in p for p in found.problems), found.problems
    assert not (results / "raw").exists()
    assert not (results / "runs.csv").exists()


@pytest.mark.parametrize("key", ["clean__s0", "fog__s3"])
def test_collect_rejects_a_confusion_matrix_that_disagrees_with_the_csv(dirs, key):
    source, results, runs = dirs
    write_download(source, "cnn-augmix-s0")
    path = source / "raw" / "cnn-augmix-s0_confusion.npz"
    with np.load(path) as data:
        matrices = {k: data[k] for k in data.files}
    matrices[key][0, 0] -= 5
    matrices[key][0, 2] += 5
    np.savez(path, **matrices)
    assert any("accuracy differs" in p for p in collect(source, results, runs).problems)


@pytest.mark.parametrize(
    ("lines", "expected"),
    [
        (lambda rows: rows[:-1], "rows do not count"),
        (lambda rows: rows + rows[-1:], "rows do not count"),
        (lambda rows: rows[:1] + [rows[1].replace("1,0.1", "1,nan")] + rows[2:], "NaN"),
    ],
)
def test_collect_rejects_a_bad_train_log(dirs, lines, expected):
    source, results, runs = dirs
    write_download(source, "cnn-augmix-s0")
    path = source / "raw" / "cnn-augmix-s0_train_log.csv"
    path.write_text("".join(lines(path.read_text().splitlines(True))))
    assert any(expected in p for p in collect(source, results, runs).problems)


def test_collect_rejects_an_unfinished_or_shortened_run(dirs):
    source, results, runs = dirs
    write_download(source, "cnn-augmix-s0", status="running")
    assert any("status is 'running'" in p for p in collect(source, results, runs).problems)
    meta_path = source / "run_folders" / "cnn-augmix-s0" / "run_meta.json"
    meta = json.loads(meta_path.read_text())
    meta_path.write_text(json.dumps({**meta, "status": "finished", "epochs": 29}))
    assert any("epochs is 29" in p for p in collect(source, results, runs).problems)


def test_collect_rejects_a_missing_result_file_or_metadata(dirs):
    source, results, runs = dirs
    write_download(source, "cnn-augmix-s0")
    (source / "raw" / "cnn-augmix-s0_confusion.npz").unlink()
    assert any("missing" in p for p in collect(source, results, runs).problems)
    write_result_files(source / "raw", "cnn-augmix-s0")
    shutil.rmtree(source / "run_folders")
    assert any("no run_meta.json" in p for p in collect(source, results, runs).problems)


def test_collect_leaves_smoke_timing_and_unevaluated_runs_out(dirs):
    source, results, runs = dirs
    write_download(source, "cnn-augmix-s0")
    write_result_files(source / "raw", "cnn-augmix-s1")
    (source / "raw" / "cnn-augmix-s1.csv").unlink()
    (source / "raw" / "cnn-augmix-s1_confusion.npz").unlink()
    (source / "raw" / "smoke-cnn-standard-s0.csv").write_text("not a result")
    (source / "raw" / ".cnn-augmix-s2.csv.partial").write_text("cut short")
    (source / "run_folders" / "timing-cnn_augmix").mkdir()
    found = collect(source, results, runs)
    assert found.problems == []
    assert found.evaluated == ["cnn-augmix-s0"]
    assert sorted(p.name for p in (results / "raw").iterdir()) == [
        "cnn-augmix-s0.csv",
        "cnn-augmix-s0_confusion.npz",
        "cnn-augmix-s0_train_log.csv",
    ]
    assert any("smoke-cnn-standard-s0" in n for n in found.notes)
    assert any("trained, not evaluated yet" in n for n in found.notes)
    assert any("timing-cnn_augmix" in n for n in found.notes)


def test_paused_run_updates_its_folder_but_a_finished_one_is_protected(dirs):
    source, results, runs = dirs
    folder = source / "run_folders" / "cnn-augmix-s0"
    folder.mkdir(parents=True)
    (source / "raw").mkdir()
    (folder / "run_meta.json").write_text(json.dumps({"status": "running", "run_id": "r"}))
    assert collect(source, results, runs).problems == []
    assert registry_rows(results)[0]["status"] == "running"

    (folder / "run_meta.json").write_text(json.dumps({"status": "finished", "run_id": "r"}))
    assert collect(source, results, runs).problems == []
    (folder / "run_meta.json").write_text(json.dumps({"status": "running", "run_id": "r"}))
    assert any("differs from" in p for p in collect(source, results, runs).problems)


def test_collect_cli_exits_1_on_problems(dirs, monkeypatch, capsys):
    source, results, runs = dirs
    write_download(source, "cnn-augmix-s0")
    edit_csv(source / "raw" / "cnn-augmix-s0.csv", ",10000,", ",9000,")
    monkeypatch.setenv("RUNS_DIR", str(runs))
    monkeypatch.setenv("RESULTS_DIR", str(results))
    with pytest.raises(SystemExit) as stopped:
        collect_results.main(["--from", str(source)])
    assert stopped.value.code == 1
    assert "nothing imported" in capsys.readouterr().err


def test_collect_cli_reports_each_run(dirs, monkeypatch, capsys):
    source, results, runs = dirs
    write_download(source, "cnn-augmix-s0")
    monkeypatch.setenv("RUNS_DIR", str(runs))
    monkeypatch.setenv("RESULTS_DIR", str(results))
    collect_results.main(["--from", str(source)])
    out = capsys.readouterr().out
    assert "cnn-augmix-s0: finished, 30 epochs, commit abcdef0, dirty False, Tesla T4" in out


def test_real_pilot_files_pass_the_c7_check():
    run_id = PILOT_RUN
    raw = REPO / "results" / "raw"
    rows, problems = collect_results.check_results_csv(raw / f"{run_id}.csv", run_id)
    assert problems == []
    assert collect_results.check_confusion(raw / f"{run_id}_confusion.npz", rows) == []
    assert collect_results.check_train_log(raw / f"{run_id}_train_log.csv", 30) == []
