# Phase 06 BUILD: Metrics and the 76-condition evaluator

## Read first
1. `docs/agent-prompts/RULES.md`
2. `docs/agent-prompts/DECISIONS.md` D3, D6, D7, D8
3. `docs/agent-prompts/CONTRACTS.md` C5, C7, C10
4. `docs/PROJECT_OVERVIEW.md` sections 9, 10.1
5. `PROGRESS.md`; `validation/phase-05.md` must be PASS or PASS WITH NOTES

Branch: `phase-06-evaluation` from `main`.

## Goal
Correct, verified metric functions, and `python -m src.evaluate` that scores any finished run on the clean test set and all 75 (corruption, severity) pairs, writing exactly the C7 files.

## In scope
- `src/metrics.py` (C10), all pure functions on NumPy arrays or pandas DataFrames:
  - `accuracy`, `macro_f1` (equal weight over the 10 classes; a class never predicted gets F1 0, matching `sklearn.metrics.f1_score(average="macro", zero_division=0)`), `absolute_degradation`, `relative_degradation`.
  - `corruption_errors(df_run, df_baseline)`: for each of the 15 corruptions, `CE_c = sum_s E(run,c,s) / sum_s E(base,c,s)`, where `df_baseline` is already the **seed-mean** baseline table (D8).
  - `mce` = mean of CE over the 15 corruptions; `relative_mce` per D8.
  - Each function raises on missing conditions (never silently averages over fewer than 15 corruptions or 5 severities).
- `src/evaluate.py` (C5):
  - Loads `model_final.pt` (refuses if `run_meta.json` status is not `finished`). Puts the model in `eval()` mode, `torch.no_grad()`, AMP on CUDA.
  - Clean test set + every corruption in `CORRUPTIONS_15` x severity 1-5 (+ extras only with `--include-extra`, labelled `family = extra`).
  - Writes `results/raw/{run_id}.csv` with exactly the C7 columns and `results/raw/{run_id}_confusion.npz`. Refuses to overwrite unless `--force`.
  - `--subset N`: first N images of each condition, `smoke-` naming, output to `results/smoke/`.
  - Prints a short summary: clean accuracy, mean corrupted accuracy, worst 3 conditions.
  - Batch size default 512 for eval.
- `tests/test_metrics.py` with **hand-computed** cases: e.g. y_true `[0,0,1,1]`, y_pred `[0,1,1,1]` -> accuracy 0.75, macro-F1 (over the classes present in the test, with labels arg = `[0,1]`) = mean(0.6667, 0.8) = 0.7333; degradation examples; a 2-corruption x 2-severity toy CE/mCE table whose answer you compute by hand in the test comment; baseline vs itself gives mCE exactly 1.0; missing condition raises. Also cross-check `macro_f1` against `sklearn.metrics.f1_score` on 1,000 random predictions.
- `tests/test_evaluate_smoke.py`: tiny fake model + fake CIFAR-10-C folder (from the Phase 02 test helpers) -> CSV has 76 rows and C7 columns.

## Out of scope
Aggregation over seeds and tables/figures (Phase 09), real full runs (Phase 07).

## Steps
1. Implement and unit-test metrics first, then the evaluator.
2. Run a smoke train (`configs/smoke.yaml`) then `python -m src.evaluate --run-dir $RUNS_DIR/smoke-cnn-standard-s0 --subset 500`; show the summary and the first 10 CSV rows.
3. Time a full evaluation pass for one condition (10,000 images) on CPU, and estimate the time for 76 conditions on CPU; record. (On a T4 the overview estimates 3-10 minutes per model.)
4. Lint, format, tests. Progress entry. Commit `phase 06: metrics and CIFAR-10-C evaluator`.

## Done when
- [ ] Metric tests with hand-computed answers pass; `macro_f1` equals scikit-learn's on random data.
- [ ] mCE of the baseline against its own table is exactly 1.0.
- [ ] Missing conditions raise instead of averaging fewer.
- [ ] Evaluator writes 76 rows with exactly the C7 columns and the confusion `.npz` with 76 keys.
- [ ] `group` labels follow D6; `severity` is 0 for clean.
- [ ] Refuses unfinished runs and existing results without `--force`.
- [ ] Smoke output goes only to `results/smoke/`.
- [ ] Lint, format, tests pass.
