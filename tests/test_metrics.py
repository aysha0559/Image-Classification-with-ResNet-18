"""Metrics against answers computed by hand, and macro-F1 against scikit-learn."""

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score

from src import metrics
from src.constants import CORRUPTIONS, EXTRA_CORRUPTIONS, SEVERITIES
from src.metrics import (
    absolute_degradation,
    accuracy,
    confusion_matrix,
    corruption_errors,
    macro_f1,
    mce,
    relative_degradation,
    relative_mce,
)


def table(errors: dict[tuple[str, int], float]) -> pd.DataFrame:
    """Build a C7-style table from {(corruption, severity): error}."""
    rows = [{"corruption": c, "severity": s, "accuracy": 1.0 - e} for (c, s), e in errors.items()]
    return pd.DataFrame(rows)


def full_table(offset: float = 0.0) -> pd.DataFrame:
    """All 76 conditions; errors differ per condition so that a wrong pairing shows."""
    errors = {("clean", 0): 0.1 + offset}
    for i, c in enumerate(CORRUPTIONS):
        for s in SEVERITIES:
            errors[(c, s)] = 0.2 + 0.01 * i + 0.05 * s + offset
    return table(errors)


@pytest.fixture
def toy_grid(monkeypatch):
    """Shrink the required grid to 2 corruptions x 2 severities for the hand-computed tables."""
    monkeypatch.setattr(metrics, "CORRUPTIONS", ("a", "b"))
    monkeypatch.setattr(metrics, "SEVERITIES", (1, 2))


# Hand-computed toy tables, errors:
#            clean   a,s1   a,s2   b,s1   b,s2
# run        0.10    0.20   0.40   0.30   0.30
# baseline   0.20    0.40   0.60   0.50   0.70
#
# CE_a = (0.20 + 0.40) / (0.40 + 0.60) = 0.6 / 1.0 = 0.6
# CE_b = (0.30 + 0.30) / (0.50 + 0.70) = 0.6 / 1.2 = 0.5
# mCE  = (0.6 + 0.5) / 2 = 0.55
#
# rCE_a = (0.10 + 0.30) / (0.20 + 0.40) = 0.4 / 0.6 = 2/3
# rCE_b = (0.20 + 0.20) / (0.30 + 0.50) = 0.4 / 0.8 = 0.5
# relative mCE = (2/3 + 1/2) / 2 = 7/12
TOY_RUN = {("clean", 0): 0.1, ("a", 1): 0.2, ("a", 2): 0.4, ("b", 1): 0.3, ("b", 2): 0.3}
TOY_BASE = {("clean", 0): 0.2, ("a", 1): 0.4, ("a", 2): 0.6, ("b", 1): 0.5, ("b", 2): 0.7}


def test_accuracy_by_hand():
    assert accuracy([0, 0, 1, 1], [0, 1, 1, 1]) == 0.75


def test_macro_f1_by_hand():
    # Class 0: TP 1, FP 0, FN 1 -> F1 = 2/3. Class 1: TP 2, FP 1, FN 0 -> F1 = 4/5.
    assert macro_f1([0, 0, 1, 1], [0, 1, 1, 1], labels=[0, 1]) == pytest.approx(11 / 15)


def test_macro_f1_defaults_to_all_ten_classes():
    # The 8 absent classes count as F1 0: (2/3 + 4/5) / 10.
    assert macro_f1([0, 0, 1, 1], [0, 1, 1, 1]) == pytest.approx(22 / 150)


def test_macro_f1_class_never_predicted_scores_zero():
    # Class 0 never predicted: F1 0. Class 1: TP 1, FP 1, FN 0 -> 2/3.
    assert macro_f1([0, 1], [1, 1], labels=[0, 1]) == pytest.approx(1 / 3)


def test_macro_f1_matches_sklearn_on_random_predictions():
    rng = np.random.default_rng(0)
    for trial in range(20):
        y_true = rng.integers(0, 10, 1000)
        y_pred = np.where(rng.random(1000) < 0.4, y_true, rng.integers(0, 10, 1000))
        if trial == 0:
            y_pred[y_pred == 3] = 4  # a class that is never predicted
        expected = f1_score(y_true, y_pred, average="macro", labels=range(10), zero_division=0)
        assert macro_f1(y_true, y_pred) == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize("fn", [accuracy, macro_f1, confusion_matrix])
@pytest.mark.parametrize("y_true, y_pred", [([], []), ([0, 1], [0]), ([[0]], [[0]])])
def test_bad_label_arrays_raise(fn, y_true, y_pred):
    with pytest.raises(ValueError, match="1-D"):
        fn(y_true, y_pred)


def test_confusion_matrix_rows_are_true_classes():
    matrix = confusion_matrix([0, 0, 1, 1], [0, 1, 1, 1], num_classes=2)
    assert matrix.tolist() == [[1, 1], [0, 2]]
    assert matrix.dtype == np.int64


def test_degradation_by_hand():
    assert absolute_degradation(0.9, 0.6) == pytest.approx(0.3)
    assert relative_degradation(0.9, 0.6) == pytest.approx(1 / 3)
    assert relative_degradation(0.8, 0.8) == 0.0


def test_relative_degradation_of_zero_clean_raises():
    with pytest.raises(ValueError, match="positive"):
        relative_degradation(0.0, 0.0)


def test_corruption_errors_by_hand(toy_grid):
    ce = corruption_errors(table(TOY_RUN), table(TOY_BASE))
    assert ce == pytest.approx({"a": 0.6, "b": 0.5})


def test_mce_by_hand(toy_grid):
    assert mce(table(TOY_RUN), table(TOY_BASE)) == pytest.approx(0.55)


def test_relative_mce_by_hand(toy_grid):
    assert relative_mce(table(TOY_RUN), table(TOY_BASE)) == pytest.approx(7 / 12)


def test_baseline_against_itself_is_exactly_one():
    base = full_table()
    assert mce(base, base) == 1.0
    assert relative_mce(base, base) == 1.0
    assert set(corruption_errors(base, base).values()) == {1.0}


def test_mce_is_the_mean_of_ce_over_15_corruptions():
    run, base = full_table(offset=0.1), full_table()
    ce = corruption_errors(run, base)
    assert list(ce) == list(CORRUPTIONS)
    assert mce(run, base) == pytest.approx(sum(ce.values()) / 15)


def test_row_order_does_not_change_the_answer():
    run, base = full_table(offset=0.1), full_table()
    shuffled = run.sample(frac=1.0, random_state=0)
    assert mce(shuffled, base) == mce(run, base)


def test_clean_and_extra_rows_never_enter_mce():
    run, base = full_table(offset=0.1), full_table()
    expected = mce(run, base)
    extras = pd.DataFrame({"corruption": list(EXTRA_CORRUPTIONS), "severity": 1, "accuracy": 0.0})
    assert mce(pd.concat([run, extras]), base) == expected
    changed_clean = run.copy()
    changed_clean.loc[changed_clean["corruption"] == "clean", "accuracy"] = 0.0
    assert mce(changed_clean, base) == expected


@pytest.mark.parametrize("fn", [corruption_errors, mce, relative_mce])
@pytest.mark.parametrize("side", ["run", "baseline"])
@pytest.mark.parametrize(
    "dropped",
    [
        lambda df: df["corruption"] != "fog",  # a whole corruption
        lambda df: df["severity"] != 5,  # a whole severity
        lambda df: ~((df["corruption"] == "snow") & (df["severity"] == 3)),  # one condition
    ],
)
def test_missing_conditions_raise(fn, side, dropped):
    full = full_table()
    short = full[dropped(full)]
    run, base = (short, full) if side == "run" else (full, short)
    with pytest.raises(ValueError, match="missing condition"):
        fn(run, base)


def test_relative_mce_needs_the_clean_row():
    full = full_table()
    no_clean = full[full["corruption"] != "clean"]
    assert mce(no_clean, full) == pytest.approx(1.0)  # CE does not use clean
    with pytest.raises(ValueError, match="clean row"):
        relative_mce(no_clean, full)
    with pytest.raises(ValueError, match="clean row"):
        relative_mce(full, no_clean)


def test_repeated_condition_raises():
    """Passing several seeds at once must not silently sum them."""
    full = full_table()
    with pytest.raises(ValueError, match="repeated condition"):
        mce(pd.concat([full, full]), full)
    with pytest.raises(ValueError, match="repeated condition"):
        mce(full, pd.concat([full, full]))


@pytest.mark.parametrize("bad", [float("nan"), 92.5, -0.1])
def test_accuracy_that_is_not_a_fraction_raises(bad):
    full = full_table()
    broken = full.copy()
    broken.loc[broken.index[3], "accuracy"] = bad
    with pytest.raises(ValueError, match="expected a fraction"):
        mce(broken, full)


def test_missing_column_raises():
    full = full_table()
    with pytest.raises(ValueError, match="missing column"):
        mce(full.rename(columns={"accuracy": "acc"}), full)


def test_zero_baseline_denominator_raises(toy_grid):
    perfect = table({("clean", 0): 0.0, ("a", 1): 0.0, ("a", 2): 0.0, ("b", 1): 0.1, ("b", 2): 0.1})
    with pytest.raises(ValueError, match="denominator is 0 for a"):
        mce(table(TOY_RUN), perfect)
