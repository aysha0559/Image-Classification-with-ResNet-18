# Validation: Phase 06 evaluation

- Date: 2026-11-04
- Branch / commit checked: phase-06-evaluation @ cb0a12b74cffc72e1e65afa0187a72725d99ea87
- Working tree clean before checking: yes
- Environment: Python 3.12.16, torch 2.14.1+cpu (torchvision 0.29.2+cpu), device cpu
- Verdict: PASS

## Summary
All 17 checks (16 Must, 1 Should) were re-run from scratch and passed. CE, mCE and relative mCE match my own plain-Python D8 formulas on 15 x 5 toy tables, and accuracy and macro-F1 match scikit-learn. The evaluator's CSV agrees with its confusion matrices and with my own forward pass, on both the 1-epoch smoke model and a slightly better model trained so that a label misalignment would show. The CUDA paths (AMP under `torch.autocast`, `pin_memory`) could not run on this CPU-only machine and stay unverified until Phase 07. All accuracies below are smoke numbers.

## Checks
| ID | Check | Must/Should | Result | Evidence |
|---|---|---|---|---|
| V1 | Branch, clean tree, commit | Must | PASS | `git rev-parse --abbrev-ref HEAD` -> `phase-06-evaluation`; `git rev-parse HEAD` -> `cb0a12b...`; `git status --short` -> empty; `git merge-base --is-ancestor main HEAD` -> true; `git diff --stat main HEAD` -> 5 files: `PROGRESS.md`, `src/evaluate.py`, `src/metrics.py`, `tests/test_evaluate_smoke.py`, `tests/test_metrics.py` |
| V2 | Lint, format, tests | Must | PASS | `ruff format --check .` -> `78 files already formatted`; `ruff check .` -> `All checks passed!`; `time pytest -q` -> `264 passed in 23.29s` (real 0m24.716s, CPU) |
| V3 | Accuracy and macro-F1 vs scikit-learn | Must | PASS | Own script, 20 random pairs of length 1,000, including one with class 7 never predicted, one with 3 classes never predicted, one with a class absent from both arrays: max difference accuracy `0.0`, macro-F1 `0.0` (limit 1e-12) |
| V4 | Degradation formulas | Must | PASS | `absolute_degradation(0.9, 0.6)` -> `0.30000000000000004`; `relative_degradation(0.9, 0.6)` -> `0.33333333333333337`. Both equal the expected value within 1e-12 (see Notes on the tolerance) |
| V5 | CE / mCE by hand | Must | PASS | Toy tables for all 15 x 5, shuffled row order, with clean and extra rows that must be ignored. My own double loop: max CE difference `1.67e-16`; mCE `0.5935986844916801` both ways, difference `0.0`. Closed form (run error = k_c x base error): `0.6399999999999999` vs expected `0.64` |
| V6 | Relative mCE by hand | Must | PASS | Same tables, my own D8 rCE loop: `0.8259425377997446` vs `0.8259425377997449`, difference `3.3e-16`. Changing the run's clean error changes the result, so the clean term is really used |
| V7 | Baseline self-mCE = 1 | Must | PASS | `mce(base, base)` -> `1.0`, `== 1.0` True; `relative_mce(base, base) == 1.0` True. Three toy seeds against their seed-mean table: `0.9464 0.9821 1.0715`, mean `1.0` within 1e-12 (D8) |
| V8 | Missing condition raises | Must | PASS | Each of the 75 corrupted rows dropped, from either table, for `mce`, `relative_mce` and `corruption_errors`: `ValueError` in `450/450` cases, for example `df_run: 1 missing condition(s), first ('pixelate', 3)`. Whole corruption, NaN accuracy and a repeated table also raise. Dropping the clean row: `relative_mce` raises, `mce` does not, and the mCE formula does not use it |
| V9 | Evaluator end to end (smoke) | Must | PASS | `RUNS_DIR=/tmp/val06/runs RESULTS_DIR=/tmp/val06/results`; `python -m src.train --config configs/smoke.yaml --seed 0` exit 0; `python -m src.evaluate --run-dir $RUNS_DIR/smoke-cnn-standard-s0 --subset 500` exit 0 -> `76 conditions, 500 images each`. CSV 76 rows, columns exactly C7 in order. npz: 76 keys, the expected key set, all `int64`, `(10, 10)`, every matrix sums to 500 |
| V10 | CSV numbers agree with the confusion matrices | Must | PASS | Every row: max difference accuracy vs trace/sum `0.0`; macro-F1 rebuilt from the matrix through scikit-learn `0.0`. Same result for the second model (2,000 images) and the full pass (10,000 images) |
| V11 | Independent re-evaluation | Must | PASS | Own loop on `model_final.pt`, no `src.evaluate`. `load_cifar10c("fog", 5)[:500]` -> `0.108`, CSV `0.108`, equal True, with the eval transform and with normalisation written by hand. Table below |
| V12 | Eval mode and no grad | Must | PASS | `src/evaluate.py:129` `@torch.no_grad()`, `:134` `model.eval()`, `:140` `torch.autocast(...)`; no `.train()` in the file. V11 and a batch-size change (below) confirm it: BatchNorm and dropout are not active |
| V13 | Group and severity labels | Must | PASS | Checked against D6 and C3 lists typed in my script, not `src.constants`: `0` bad rows; `clean` 1, `seen` 20, `unseen` 55; clean row `clean`, `clean`, severity 0 |
| V14 | Refusals | Must | PASS | Second evaluate -> `FileExistsError: refusing to overwrite [...]; use --force`, exit 1. Copy of the run with `status: running` -> `ValueError: refusing to evaluate ...: status is 'running'`, exit 1, no results folder created; `failed` is refused the same way. Only the npz present -> also refused. `--force` -> exit 0 |
| V15 | Extras off by default | Must | PASS | Default: no extras in the CSV (`[]`), 76 rows. With `--subset 50 --include-extra`: 96 rows, 96 npz keys, the 4 extras with `family = extra`, summary still `(over 75 conditions)` |
| V16 | Smoke isolation | Must | PASS | After all runs, `git status --short` in the repo -> empty; `ls -A results/raw` -> `.gitkeep` only; `results/smoke` does not exist in the repo. A copy renamed to the real id `cnn-standard-s0` with `--subset 50` wrote only to `smoke/smoke-cnn-standard-s0.*`; no `raw/` folder was created |
| V17 | Eval timing recorded | Should | PASS | Every `PROGRESS.md` value is within 8% of my run. Table below |

### V11, own forward pass vs CSV [smoke]
Exact equality on accuracy for every row shown. "Second model" is `cnn_standard.yaml --epochs 5 --subset 8000 --run-id smoke-signal`, evaluated with `--subset 2000`, because the 1-epoch model predicts only 4-5 classes at about 10% accuracy.

| Condition | Smoke model, 500 images: mine / CSV | Second model, 2,000 images: mine / CSV | Second model, loss: mine / CSV |
|---|---|---|---|
| clean | 0.096 / 0.096 | 0.2435 / 0.2435 | not compared |
| fog s5 | 0.108 / 0.108 | 0.164 / 0.164 | 2.256157 / 2.256158 |
| gaussian_noise s3 | 0.086 / 0.086 | 0.246 / 0.246 | 1.915818 / 1.915818 |
| contrast s1 | 0.108 / 0.108 | 0.243 / 0.243 | 1.981314 / 1.981314 |

### V17, evaluation time on CPU [smoke]
This machine, 4 torch threads, batch size 512, untrained models, fog severity 3, 10,000 images, `score_condition`, 3 repeats, nothing else running. These do not predict Colab time.

| Model | `num_workers` | Mine, s | Median | `PROGRESS.md` median | Difference | 76 conditions, mine | 76 conditions, `PROGRESS.md` |
|---|---|---|---|---|---|---|---|
| simple_cnn | 2 | 11.20 11.00 10.68 | 11.00 | 10.23 | +7.5% | estimated 836 s | estimated 777 s |
| simple_cnn | 0 | 11.48 11.20 11.33 | 11.33 | 12.00 | -5.6% | measured 842 s (`real 14m2.337s`) | measured 860 s |
| resnet18 | 2 | 108.25 106.95 107.14 | 107.14 | 106.30 | +0.8% | estimated 8,143 s | estimated 8,079 s |

### Extra checks (not in the list)
- Full pass, no `--subset`: `76 conditions, 10000 images each`, 77 CSV lines, clean 0.1182, mean corrupted 0.1142, same as `PROGRESS.md`. It also passes the V9, V10 and V13 scripts. [smoke]
- Batch size 64 instead of 512, `--force`: accuracy and macro-F1 identical on all 76 rows; max loss difference 4.4e-07.
- `python -m scripts.registry` -> `has_results` True, `clean_accuracy` 0.096 for `smoke-cnn-standard-s0`.
- A `model_final.pt` from another run is refused: `model_final.pt is from 'smoke-cnn-standard-s0', run_meta.json from 'cnn-standard-s0'`, exit 1.
- Leakage: only `src/evaluate.py` and `scripts/make_sample_grid.py` import `cifar10c`; `grep -nE "train=False|cifar10c" src/train.py` -> no matches (exit 1).

## Failures (if any)
None.

## Notes (non-blocking)
- **CUDA paths did not run:** AMP under `torch.autocast`, `pin_memory` and the T4 time are unverified. Check them in Phase 07.
- **V4's second example has no tolerance in its text:** `relative_degradation(0.9, 0.6) == 1/3` is False bit for bit (`0.33333333333333337`), which is ordinary rounding: 0.9 - 0.6 is not exactly 0.3. I applied the "float tolerance" written in the same line to both values.
- **The extras' `group` is not defined anywhere:** C7 allows only `clean`, `seen` and `unseen`, D6 does not list the extras, and the code labels them `unseen`. This needs a decision only if `--include-extra` results are ever used, and the human may want to record it in C7. `PROGRESS.md` already lists it.
- **`git_commit` does not show uncommitted changes:** C7 has no dirty flag, so evaluate real runs from a clean tree, as `PROGRESS.md` says.
- **`num_workers` gave almost no speed-up here:** 11.00 s with 2 workers against 11.33 s with none, on this machine. `PROGRESS.md` shows a larger gap (10.23 against 12.00), so the two measurements differ on the size of the gain. It has no effect on results.
- **C10 additions:** `macro_f1` has an optional `labels` argument and `metrics.py` has a `confusion_matrix` function that C10 does not list. C10's two-argument call works unchanged; `PROGRESS.md` discloses both.

## Claims in PROGRESS.md that could not be confirmed
- The 20 mutation checks ("the tests catch broken code") -> not re-run; this phase's VALIDATE does not ask for them.
- The CSV's `git_commit` of `5c96ebd...` -> my CSV shows `cb0a12b...`, as expected: the build ran before its commit.
- Everything else re-run agrees with `PROGRESS.md`, including the step 2 summary (0.0960, 0.0969, the same 3 worst conditions), `fog s5` 0.108, and the full-pass numbers.
