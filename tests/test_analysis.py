"""Analysis and figures on a tiny synthetic results set with known answers (Phase 09)."""

import json
import re
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd
import pytest

from scripts import make_figures as figures
from src import analysis
from src.constants import (
    ANALYSIS_GROUPS,
    CORRUPTION_TO_FAMILY,
    CORRUPTIONS,
    MODEL_SHORT_NAMES,
    NUM_CLASSES,
    SEEDS,
    SEVERITIES,
)

REPO = Path(__file__).resolve().parent.parent

# (model, strategy): clean accuracy and accuracy on every corrupted condition, at seed 1.
CONFIG_ACCURACY = {
    ("simple_cnn", "standard"): (0.80, 0.50),
    ("simple_cnn", "augmix"): (0.80, 0.60),
    ("simple_cnn", "corruption_aware"): (0.78, 0.65),
    ("resnet18", "standard"): (0.90, 0.60),
    ("resnet18", "augmix"): (0.90, 0.75),
    ("resnet18", "corruption_aware"): (0.88, 0.80),
}
SEED_SHIFT = 0.01  # corrupted accuracy moves by this per seed, around seed 1


def _rows(model, strategy, seed):
    """Return the 76 C7 rows of one synthetic run."""
    run_id = f"{MODEL_SHORT_NAMES[model]}-{strategy}-s{seed}"
    clean, corrupted = CONFIG_ACCURACY[(model, strategy)]
    corrupted += SEED_SHIFT * (seed - 1)
    conditions = [("clean", 0, clean)] + [
        (c, s, corrupted) for c in CORRUPTIONS for s in SEVERITIES
    ]
    return pd.DataFrame(
        [
            {
                "run_id": run_id,
                "model": model,
                "strategy": strategy,
                "seed": seed,
                "corruption": corruption,
                "family": CORRUPTION_TO_FAMILY.get(corruption, "clean"),
                "group": "x",
                "severity": severity,
                "n_images": 10000,
                "accuracy": accuracy,
                "macro_f1": accuracy - 0.01,
                "loss": 1.0,
                "git_commit": "abc",
                "evaluated_at": "2026-11-05T00:00:00+00:00",
            }
            for corruption, severity, accuracy in conditions
        ]
    )


def _write_run(results_dir, model, strategy, seed):
    """Write the raw files of one synthetic run."""
    raw_dir = results_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    run_id = analysis.run_id_of(model, strategy, seed)
    _rows(model, strategy, seed).to_csv(raw_dir / f"{run_id}.csv", index=False)
    log = {"epoch": [1, 2], "train_loss": [2.0, 1.0], "val_acc": [0.3, 0.5]}
    pd.DataFrame(log).to_csv(raw_dir / f"{run_id}_train_log.csv", index=False)
    matrix = np.eye(NUM_CLASSES, dtype=np.int64) * 100
    keys = ("clean__s0", "fog__s5", "gaussian_noise__s5")
    np.savez(raw_dir / f"{run_id}_confusion.npz", **dict.fromkeys(keys, matrix))


@pytest.fixture
def results(tmp_path):
    """Return a results folder holding all 18 synthetic runs."""
    for model, strategy in CONFIG_ACCURACY:
        for seed in SEEDS:
            _write_run(tmp_path, model, strategy, seed)
    return tmp_path


@pytest.fixture
def partial_results(results):
    """Return the folder without seeds 1 and 2 of ResNet-18 + AugMix and seed 2 of the baseline."""
    for run_id in ("resnet18-augmix-s1", "resnet18-augmix-s2", "cnn-standard-s2"):
        (results / "raw" / f"{run_id}.csv").unlink()
        (results / "raw" / f"{run_id}_train_log.csv").unlink()
    return results


def _main_row(results_dir, model, strategy, partial=False):
    """Return one row of the written main table."""
    name = analysis.output_name("main_results", "csv", partial)
    table = pd.read_csv(results_dir / "tables" / name)
    return table[(table["model"] == model) & (table["strategy"] == strategy)].iloc[0]


def test_expected_runs_are_the_18_of_the_matrix():
    run_ids = analysis.expected_run_ids()
    assert len(set(run_ids)) == 18
    assert "cnn-standard-s0" in run_ids and not any("-none-" in r for r in run_ids)


def test_refuses_and_names_the_missing_run(results):
    (results / "raw" / "resnet18-augmix-s2.csv").unlink()
    with pytest.raises(SystemExit, match="resnet18-augmix-s2"):
        analysis.analyse(results)
    assert not (results / "tables").exists()


def test_figures_refuse_missing_runs(partial_results):
    with pytest.raises(SystemExit, match="missing"):
        figures.make_figures(partial_results)


def test_full_run_writes_every_c9_table_unstamped(results):
    written = {path.name for path in analysis.analyse(results)}
    assert written == {
        "main_results.csv",
        "main_results.md",
        "per_corruption.csv",
        "per_severity.csv",
        "per_family.csv",
        "seen_unseen.csv",
        "seen_unseen.md",
        "report_numbers.json",
        "report_numbers.md",
        "rq_answers.md",
    }
    assert "data_status" not in pd.read_csv(results / "tables" / "main_results.csv").columns
    assert json.loads((results / "tables" / "report_numbers.json").read_text())["meta.status"] == (
        "FULL"
    )


def test_baseline_mce_is_exactly_100_percent(results):
    analysis.analyse(results)
    row = _main_row(results, "simple_cnn", "standard")
    assert row["mCE_mean"] == pytest.approx(1.0, abs=1e-12)
    assert row["relative_mCE_mean"] == pytest.approx(1.0, abs=1e-12)


def test_known_main_values(results):
    analysis.analyse(results)
    row = _main_row(results, "resnet18", "augmix")
    assert row["clean_acc_mean"] == pytest.approx(0.90)
    assert row["clean_f1_mean"] == pytest.approx(0.89)
    assert row["mean_corrupted_acc_mean"] == pytest.approx(0.75)
    assert row["mean_abs_degradation_mean"] == pytest.approx(0.15)
    assert row["mean_rel_degradation_mean"] == pytest.approx(0.15 / 0.90)
    assert row["mCE_mean"] == pytest.approx(0.25 / 0.50)
    assert row["relative_mCE_mean"] == pytest.approx((0.25 - 0.10) / (0.50 - 0.20))


def test_std_is_the_sample_std_over_seeds(results):
    analysis.analyse(results)
    row = _main_row(results, "resnet18", "augmix")
    assert row["mean_corrupted_acc_std"] == pytest.approx(SEED_SHIFT)  # ddof=0 would be 0.00816


def test_single_seed_has_no_std_and_is_null_in_json(partial_results):
    analysis.analyse(partial_results, allow_partial=True)
    row = _main_row(partial_results, "resnet18", "augmix", partial=True)
    assert row["n_seeds"] == 1 and np.isnan(row["clean_acc_std"])
    numbers = analysis.output_name("report_numbers", "json", True)
    values = json.loads((partial_results / "tables" / numbers).read_text())
    assert values["main.resnet18.augmix.clean_acc.std"] is None
    assert values["main.resnet18.augmix.n_seeds"] == 1


def test_partial_baseline_uses_the_seeds_it_has(partial_results):
    analysis.analyse(partial_results, allow_partial=True)
    row = _main_row(partial_results, "simple_cnn", "standard", partial=True)
    assert row["n_seeds"] == 2
    assert row["mCE_mean"] == pytest.approx(1.0, abs=1e-12)
    baseline_error = 1 - np.mean([0.49, 0.50])  # seeds 0 and 1 only
    row = _main_row(partial_results, "resnet18", "augmix", partial=True)
    assert row["mCE_mean"] == pytest.approx((1 - 0.74) / baseline_error)  # seed 0 only


def test_every_partial_output_is_stamped(partial_results):
    written = analysis.analyse(partial_results, allow_partial=True)
    assert len(written) == 10
    assert all("_PARTIAL." in path.name for path in written)
    for path in written:
        text = path.read_text()
        if path.suffix == ".csv":
            assert pd.read_csv(path)["data_status"].eq("PARTIAL").all()
        elif path.suffix == ".md":
            assert text.startswith("# PARTIAL: ") and "Not for the report or README" in text
        else:
            assert json.loads(text)["meta.status"] == "PARTIAL"


def test_figures_partial_are_stamped_full_are_not(partial_results):
    analysis.analyse(partial_results, allow_partial=True)
    figures_dir = partial_results / "figures"
    figures_dir.mkdir()
    (figures_dir / f"{figures.SAMPLE_GRID}.png").write_bytes(b"exists")
    written = figures.make_figures(partial_results, allow_partial=True)
    assert len(written) == 2 * len(figures.FIGURES)
    assert all("_PARTIAL." in path.name and path.stat().st_size > 0 for path in written)
    assert {path.suffix for path in written} == {".png", ".pdf"}
    context = figures.Context(partial_results, True, 15)
    assert context.title("x").startswith("PARTIAL (15 of 18 runs)")
    assert figures.Context(partial_results, False, 18).title("x") == "x"


def test_seen_unseen_uses_the_d6_groups(results):
    analysis.analyse(results)
    table = pd.read_csv(results / "tables" / "seen_unseen.csv")
    assert set(table["analysis_group"]) == set(ANALYSIS_GROUPS)
    assert len(table) == 3 * len(CONFIG_ACCURACY)


def test_group_means_follow_the_conditions(results):
    analysis.analyse(results)  # a known value per group: lift `fog` only
    raw = pd.read_csv(results / "raw" / "cnn-standard-s0.csv")
    raw.loc[raw["corruption"] == "fog", "accuracy"] += 0.3
    raw.to_csv(results / "raw" / "cnn-standard-s0.csv", index=False)
    analysis.analyse(results)
    table = pd.read_csv(results / "tables" / "seen_unseen.csv")
    row = table[
        (table["model"] == "simple_cnn")
        & (table["strategy"] == "standard")
        & (table["analysis_group"] == "unseen_other_family")
    ].iloc[0]
    assert row["accuracy_mean"] == pytest.approx(0.5 + 0.3 / 3 / 3)


def test_per_tables_have_the_expected_shape_and_ce(results):
    analysis.analyse(results)
    tables = results / "tables"
    corruption = pd.read_csv(tables / "per_corruption.csv")
    assert len(corruption) == 6 * 15
    assert len(pd.read_csv(tables / "per_severity.csv")) == 6 * 6
    family = pd.read_csv(tables / "per_family.csv")
    assert len(family) == 6 * 4 and "extra" not in set(family["family"])
    row = corruption[(corruption["model"] == "resnet18") & (corruption["strategy"] == "augmix")]
    assert row["ce_mean"].to_numpy() == pytest.approx([0.5] * 15)
    assert set(row["family"]) == {"noise", "blur", "weather", "digital"}


def test_severity_zero_is_clean_with_no_degradation(results):
    analysis.analyse(results)
    table = pd.read_csv(results / "tables" / "per_severity.csv")
    rows = table[table["severity"] == 0]
    assert rows["abs_degradation_mean"].to_numpy() == pytest.approx([0.0] * 6)


def test_extra_corruptions_never_enter_the_averages(results):
    analysis.analyse(results)
    before = (results / "tables" / "main_results.csv").read_text()
    path = results / "raw" / "resnet18-augmix-s0.csv"
    raw = pd.read_csv(path)
    extra = raw[raw["corruption"] == "fog"].assign(corruption="spatter", family="extra", accuracy=0)
    pd.concat([raw, extra]).to_csv(path, index=False)
    analysis.analyse(results)
    assert (results / "tables" / "main_results.csv").read_text() == before


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (lambda df: df.iloc[:-1], "76 conditions"),
        (lambda df: df.assign(n_images=100), "n_images"),
        (lambda df: df.assign(seed=2), "do not match"),
        (lambda df: df.assign(run_id="other"), "run_id column"),
    ],
)
def test_a_wrong_run_file_is_rejected(results, edit, message):
    path = results / "raw" / "cnn-augmix-s0.csv"
    edit(pd.read_csv(path)).to_csv(path, index=False)
    with pytest.raises(ValueError, match=message):
        analysis.analyse(results)


def test_needs_the_baseline_even_when_partial(results):
    for seed in SEEDS:
        (results / "raw" / f"cnn-standard-s{seed}.csv").unlink()
    with pytest.raises(ValueError, match="baseline"):
        analysis.analyse(results, allow_partial=True)


def test_seed_noise_rule():
    assert analysis.seed_noise_label(0.1, 0.2, 0.05) == analysis.WITHIN_NOISE
    assert analysis.seed_noise_label(-0.1, 0.2, 0.05) == analysis.WITHIN_NOISE
    assert analysis.seed_noise_label(0.3, 0.2, 0.05) == analysis.BEYOND_NOISE
    assert analysis.seed_noise_label(0.3, float("nan"), 0.05) == analysis.NOT_ASSESSABLE


def test_comparison_pairs_only_shared_seeds():
    a = pd.Series({0: 0.9})
    b = pd.Series({0: 0.5, 1: 0.1})
    c = analysis.Comparison(a, b)
    assert c.diff == pytest.approx(0.4)  # the means of all seeds would give 0.6
    assert list(c.seed_diffs.index) == [0]
    assert c.verdict == analysis.NOT_ASSESSABLE


def test_rq_answers_carry_the_known_gains(results):
    analysis.analyse(results)
    values = json.loads((results / "tables" / "report_numbers.json").read_text())
    assert values["rq4.resnet18.corruption_aware.seen_gain_vs_standard"] == pytest.approx(0.20)
    assert values["rq4.resnet18.corruption_aware.seen_gain_vs_standard.seed2"] == pytest.approx(0.2)
    key = "rq4.resnet18.corruption_aware.seen_gain_vs_standard.seed_noise"
    assert values[key] == analysis.BEYOND_NOISE
    assert values["rq1.cnn.standard.abs_drop_sev5"] == pytest.approx(0.30)
    assert values["rq1.cnn.standard.drop_grows_with_severity"] is False  # all severities equal
    assert values["rq2.standard.clean_acc.resnet18_minus_cnn"] == pytest.approx(0.10)
    assert values["rq3.cnn.augmix.mce.diff_vs_standard"] == pytest.approx(-0.2)
    assert "rq5.worst_family.resnet18.standard" in values


def test_every_report_number_is_documented_and_vice_versa(results):
    analysis.analyse(results)
    tables = results / "tables"
    keys = set(json.loads((tables / "report_numbers.json").read_text()))
    documented = re.findall(r"^\| `([^`]+)` \|", (tables / "report_numbers.md").read_text(), re.M)
    assert len(documented) == len(set(documented))
    assert set(documented) == keys


def test_rebuild_is_byte_identical(results):
    first = {p.name: p.read_bytes() for p in analysis.analyse(results)}
    second = {p.name: p.read_bytes() for p in analysis.analyse(results)}
    assert first == second


def test_notebook_is_valid_and_holds_no_results():
    notebook = nbformat.read(REPO / "notebooks" / "02_analysis.ipynb", as_version=4)
    nbformat.validate(notebook)
    code = [cell for cell in notebook.cells if cell.cell_type == "code"]
    assert code and all(not cell.outputs for cell in code)
    text = "\n".join(cell.source for cell in code)
    assert "src.analysis" in text and "scripts.make_figures" in text


def _write_registry(results_dir):
    """Write a synthetic `runs.csv` for the 18 runs; the Kaggle run trains faster."""
    rows = []
    for run_id in analysis.expected_run_ids():
        model = "simple_cnn" if run_id.startswith("cnn-") else "resnet18"
        strategy = run_id.split("-")[1]
        seconds = {"standard": 600.0, "augmix": 3000.0, "corruption_aware": 900.0}[strategy]
        rows.append(
            {
                "run_id": run_id,
                "model": model,
                "strategy": strategy,
                "n_params": 100 if model == "simple_cnn" else 200,
                "device_name": "Tesla T4",
                "torch_version": "2.14.1" if run_id.endswith("s2") else "2.11.0",
                "torchvision_version": "0.29.1" if run_id.endswith("s2") else "0.26.0",
                "python_version": "3.13.16",
                "git_commit": "7d334eb220c19c79",
                "git_dirty": False,
                "train_time_s": 60.0 if run_id in analysis.KAGGLE_RUNS else seconds,
            }
        )
    pd.DataFrame(rows).to_csv(results_dir / "runs.csv", index=False)


def test_rq3_records_the_clean_accuracy_cost(results):
    analysis.analyse(results)
    values = json.loads((results / "tables" / "report_numbers.json").read_text())
    key = "rq3.resnet18.corruption_aware.clean_acc.diff_vs_standard"
    assert values[key] == pytest.approx(-0.02)
    assert values["rq3.cnn.augmix.clean_acc.diff_vs_standard"] == pytest.approx(0.0)


def test_rq5_rank_gaps_and_names(results):
    analysis.analyse(results)
    values = json.loads((results / "tables" / "report_numbers.json").read_text())
    for kind in ("family", "corruption"):
        for gap in ("second_gap", "mildest_gap"):
            assert values[f"rq5.cnn.standard.{kind}_{gap}"] == pytest.approx(0.0)
    assert values["rq5.cnn.standard.corruption_worst_gap.seed_noise"] == analysis.WITHIN_NOISE
    assert "rq5.cnn.standard.family_worst_gap" not in values  # worst_family_gap covers it
    assert values["rq5.runner_up_family.cnn.standard"] in analysis.MAIN_FAMILIES
    assert "Other rank gaps" in (results / "tables" / "rq_answers.md").read_text()


def test_setup_and_compute_numbers_come_from_the_registry(results):
    analysis.analyse(results)
    values = json.loads((results / "tables" / "report_numbers.json").read_text())
    assert not any(k.startswith(("setup.", "compute.")) for k in values)  # no runs.csv yet
    _write_registry(results)
    analysis.analyse(results)
    values = json.loads((results / "tables" / "report_numbers.json").read_text())
    assert values["setup.n_params.cnn"] == 100
    assert values["setup.n_params.resnet18"] == 200
    assert values["setup.torch_versions"] == "2.11.0 (12 runs), 2.14.1 (6 runs)"
    assert values["setup.git_commits"] == "7d334eb (18 runs)"
    assert values["setup.n_dirty_runs"] == 0
    assert values["compute.cnn.augmix.time_ratio_vs_standard"] == pytest.approx(5.0)
    assert values["compute.resnet18.augmix.colab_train_minutes"] == pytest.approx(50.0)
    assert values["compute.resnet18.augmix.train_minutes"] == pytest.approx((3000 * 2 + 60) / 180)
    assert values["compute.kaggle_train_minutes"] == pytest.approx(1.0)
    hours = (3 * 2 * (600 + 900) + 5 * 3000 + 60) / 3600
    assert values["compute.train_hours_total"] == pytest.approx(hours)


def test_registry_must_agree_on_parameters(results):
    _write_registry(results)
    registry = pd.read_csv(results / "runs.csv")
    registry.loc[0, "n_params"] = 999
    registry.to_csv(results / "runs.csv", index=False)
    with pytest.raises(ValueError, match="disagree on n_params"):
        analysis.analyse(results)


def test_confusion_numbers_name_the_most_predicted_classes(results):
    matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    matrix[:, 6] = 70  # every true class: 70 predicted frog, 30 predicted bird
    matrix[:, 2] = 30
    for seed in SEEDS:
        path = results / "raw" / f"resnet18-standard-s{seed}_confusion.npz"
        with np.load(path) as old:
            arrays = dict(old)
        arrays["gaussian_noise__s5"] = matrix
        np.savez(path, **arrays)
    analysis.analyse(results)
    values = json.loads((results / "tables" / "report_numbers.json").read_text())
    prefix = "confusion.resnet18.standard.gaussian_noise__s5"
    assert values[f"{prefix}.top_class"] == "frog"
    assert values[f"{prefix}.top_share"] == pytest.approx(0.7)
    assert values[f"{prefix}.second_class"] == "bird"
    assert values[f"{prefix}.second_share"] == pytest.approx(0.3)
    assert values[f"{prefix}.top_class_every_seed"] is True
    assert values["confusion.resnet18.standard.clean__s0.top_share"] == pytest.approx(0.1)


def test_group_numbers_and_seen_minus_unseen_gain(results):
    analysis.analyse(results)
    values = json.loads((results / "tables" / "report_numbers.json").read_text())
    key = "groups.resnet18.corruption_aware.seen.accuracy.mean"
    assert values[key] == pytest.approx(0.80)
    assert values["groups.cnn.standard.unseen_other_family.abs_degradation.mean"] == pytest.approx(
        0.30
    )
    gap = "rq4.resnet18.corruption_aware.seen_minus_unseen_other_family_gain"
    assert values[gap] == pytest.approx(0.0)  # synthetic gains are the same on every group
    assert values[f"{gap}.seed1"] == pytest.approx(0.0)
