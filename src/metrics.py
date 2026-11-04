"""Accuracy, macro-F1, degradation and corruption error (DECISIONS D8, CONTRACTS C10).

Pure functions on NumPy arrays and pandas DataFrames. All values are fractions: accuracy 0.75,
error 0.25, mCE 1.0 for "the same as the baseline". Percentages belong in tables and figures.

The table functions take rows with the C7 columns `corruption`, `severity` and `accuracy`:
`df_run` holds one run, and `df_baseline` is already the seed-mean table of Simple CNN + S1
(one row per condition). Rows for `clean` and the 4 extra corruptions are ignored by the CE
sums, so extras can never enter mCE (D7). A missing, repeated or empty condition raises.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd

from src.constants import CORRUPTIONS, NUM_CLASSES, SEVERITIES


def _as_label_arrays(y_true: Sequence[int], y_pred: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
    """Return both label sequences as arrays; raise unless they are 1-D, equal and non-empty."""
    true, pred = np.asarray(y_true), np.asarray(y_pred)
    if true.ndim != 1 or true.shape != pred.shape or true.size == 0:
        raise ValueError(f"expected two equal non-empty 1-D arrays: {true.shape} {pred.shape}")
    return true, pred


def accuracy(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    """Return the fraction of predictions equal to the true label."""
    true, pred = _as_label_arrays(y_true, y_pred)
    return float(np.mean(true == pred))


def macro_f1(
    y_true: Sequence[int], y_pred: Sequence[int], labels: Iterable[int] = range(NUM_CLASSES)
) -> float:
    """Return the unweighted mean of per-class F1 over `labels` (default: the 10 classes).

    A class with no true and no predicted images, or never predicted correctly, has F1 0. This
    matches `sklearn.metrics.f1_score(average="macro", labels=labels, zero_division=0)`.
    """
    true, pred = _as_label_arrays(y_true, y_pred)
    scores = []
    for label in labels:
        true_pos = np.sum((true == label) & (pred == label))
        support = np.sum(true == label) + np.sum(pred == label)
        scores.append(2 * true_pos / support if support else 0.0)
    if not scores:
        raise ValueError("macro_f1 needs at least one label")
    return float(np.mean(scores))


def confusion_matrix(
    y_true: Sequence[int], y_pred: Sequence[int], num_classes: int = NUM_CLASSES
) -> np.ndarray:
    """Return the int64 count matrix: rows = true class, columns = predicted class (C7)."""
    true, pred = _as_label_arrays(y_true, y_pred)
    pairs = num_classes * true.astype(np.int64) + pred.astype(np.int64)
    counts = np.bincount(pairs, minlength=num_classes**2)
    return counts.reshape(num_classes, num_classes).astype(np.int64)


def absolute_degradation(clean: float, corrupted: float) -> float:
    """Return clean - corrupted (a fraction; shown as percentage points)."""
    return clean - corrupted


def relative_degradation(clean: float, corrupted: float) -> float:
    """Return (clean - corrupted) / clean. Raise if `clean` is not positive."""
    if clean <= 0:
        raise ValueError(f"relative degradation needs a positive clean value, got {clean}")
    return (clean - corrupted) / clean


def _lookup(df: pd.DataFrame, what: str) -> dict[tuple[str, int], float]:
    """Return {(corruption, severity): error}; raise on repeats, or accuracy outside [0, 1]."""
    missing = {"corruption", "severity", "accuracy"} - set(df.columns)
    if missing:
        raise ValueError(f"{what}: missing column(s) {sorted(missing)}")
    errors: dict[tuple[str, int], float] = {}
    rows = zip(df["corruption"], df["severity"], df["accuracy"], strict=True)
    for corruption, severity, value in rows:
        key = (corruption, int(severity))
        if key in errors:
            raise ValueError(f"{what}: repeated condition {key}; expected one row per condition")
        if not 0.0 <= value <= 1.0:  # also False for NaN
            raise ValueError(f"{what}: accuracy of {key} is {value}, expected a fraction")
        errors[key] = 1.0 - float(value)
    return errors


def _corrupted_errors(df: pd.DataFrame, what: str) -> dict[str, np.ndarray]:
    """Return {corruption: errors over all severities}; raise if any condition is missing."""
    errors = _lookup(df, what)
    needed = [(c, s) for c in CORRUPTIONS for s in SEVERITIES]
    missing = [key for key in needed if key not in errors]
    if missing:
        raise ValueError(f"{what}: {len(missing)} missing condition(s), first {missing[0]}")
    return {c: np.array([errors[(c, s)] for s in SEVERITIES]) for c in CORRUPTIONS}


def _clean_error(df: pd.DataFrame, what: str) -> float:
    """Return the error of the single (`clean`, 0) row; raise if there is none."""
    errors = _lookup(df, what)
    if ("clean", 0) not in errors:
        raise ValueError(f"{what}: missing the clean row (corruption 'clean', severity 0)")
    return errors[("clean", 0)]


def _safe_ratio(numerator: float, denominator: float, corruption: str) -> float:
    """Return numerator / denominator; raise on a zero denominator."""
    if denominator == 0:
        raise ValueError(f"baseline denominator is 0 for {corruption}")
    return float(numerator / denominator)


def corruption_errors(df_run: pd.DataFrame, df_baseline: pd.DataFrame) -> dict[str, float]:
    """Return CE per corruption: sum_s E(run,c,s) / sum_s E(base,c,s) (D8)."""
    run = _corrupted_errors(df_run, "df_run")
    base = _corrupted_errors(df_baseline, "df_baseline")
    return {c: _safe_ratio(run[c].sum(), base[c].sum(), c) for c in CORRUPTIONS}


def mce(df_run: pd.DataFrame, df_baseline: pd.DataFrame) -> float:
    """Return mCE: the mean of CE over the 15 corruptions (1.0 = same error as the baseline)."""
    return float(np.mean(list(corruption_errors(df_run, df_baseline).values())))


def relative_mce(df_run: pd.DataFrame, df_baseline: pd.DataFrame) -> float:
    """Return relative mCE (D8): the mean over corruptions of
    sum_s(E_c,s - E_clean) / sum_s(E_base_c,s - E_base_clean).
    """
    run = _corrupted_errors(df_run, "df_run")
    base = _corrupted_errors(df_baseline, "df_baseline")
    run_clean = _clean_error(df_run, "df_run")
    base_clean = _clean_error(df_baseline, "df_baseline")
    ratios = [
        _safe_ratio((run[c] - run_clean).sum(), (base[c] - base_clean).sum(), c)
        for c in CORRUPTIONS
    ]
    return float(np.mean(ratios))
