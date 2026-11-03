# Phase 06 VALIDATE: Metrics and evaluator

You are the independent validator for Phase 06. Metric bugs silently ruin every later result, so compute things yourself. Do not edit repo files except creating `validation/phase-06.md`. Use `RUNS_DIR=/tmp/val06/runs`, `RESULTS_DIR=/tmp/val06/results`.

## Read first
`docs/agent-prompts/RULES.md`, `CONTRACTS.md` (C5, C7, C10), `DECISIONS.md` (D6, D7, D8), `docs/PROJECT_OVERVIEW.md` section 9, `docs/agent-prompts/phase-06-evaluation/BUILD.md`, `docs/agent-prompts/templates/VALIDATION-report.md`.

## Checks
| ID | Check | Must/Should | How |
|---|---|---|---|
| V1 | Branch, clean tree, commit | Must | git |
| V2 | Lint, format, tests | Must | `ruff format --check . && ruff check . && pytest -q` |
| V3 | Accuracy and macro-F1 vs scikit-learn | Must | your own script: 20 random (y_true, y_pred) pairs of length 1,000 including one where a class is never predicted; compare to `sklearn.metrics.accuracy_score` and `f1_score(average="macro", labels=range(10), zero_division=0)` within 1e-12 |
| V4 | Degradation formulas | Must | check `absolute_degradation(0.9, 0.6) == 0.3` (float tolerance) and `relative_degradation(0.9, 0.6) == 1/3` |
| V5 | CE / mCE by hand | Must | build a toy run table and baseline table for all 15 corruptions x 5 severities with numbers you choose; compute CE and mCE in your own code from the D8 formula; compare |
| V6 | Relative mCE by hand | Must | same toy tables; your own D8 rCE formula |
| V7 | Baseline self-mCE = 1 | Must | `mce(base, base) == 1.0` |
| V8 | Missing condition raises | Must | drop one row from the toy table; `mce` must raise |
| V9 | Evaluator end to end (smoke) | Must | train `configs/smoke.yaml` into `/tmp/val06/runs`, evaluate with `--subset 500`; CSV has 76 rows, columns exactly C7 in order; npz has 76 keys each (10,10) int, each summing to 500 |
| V10 | CSV numbers agree with the confusion matrices | Must | for every row, accuracy == trace/sum of the matching confusion matrix; macro_f1 recomputed from the matrix matches |
| V11 | Independent re-evaluation | Must | write your own 20-line loop that loads `model_final.pt`, runs the model on `load_cifar10c("fog", 5)[:500]` with the eval transform, and computes accuracy; must equal the CSV row exactly |
| V12 | Eval mode and no grad | Must | read code: `model.eval()`, `torch.no_grad()` (or `inference_mode`) |
| V13 | Group and severity labels | Must | `group` matches D6 for every corruption; clean row is `clean`, severity 0 |
| V14 | Refusals | Must | evaluating twice without `--force` refuses; evaluating a run whose `run_meta.json` status is `running` refuses |
| V15 | Extras off by default | Must | without `--include-extra` no extra corruptions appear |
| V16 | Smoke isolation | Must | repo `results/raw/` unchanged (`git status`) |
| V17 | Eval timing recorded | Should | compare `PROGRESS.md` with your run |

Write `validation/phase-06.md`; give me the one-line verdict.
