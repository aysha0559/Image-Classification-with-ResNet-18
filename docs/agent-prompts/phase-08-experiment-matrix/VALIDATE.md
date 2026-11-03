# Phase 08 VALIDATE: Full experiment matrix

You are the independent validator for Phase 08. You confirm the 18 runs are complete, comparable and genuine. Do not edit repo files except creating `validation/phase-08.md`.

## Read first
`docs/agent-prompts/RULES.md`, `CONTRACTS.md` (C6, C7, C8), `DECISIONS.md` (all, including the change log), `docs/agent-prompts/phase-08-experiment-matrix/BUILD.md`, `docs/agent-prompts/templates/VALIDATION-report.md`, `results/RUNLOG.md`.

Ask the human for the Drive `runs/` folder (at least each run's `config_resolved.yaml`, `run_meta.json`, and the `model_final.pt` of 2 runs you pick at random) if they are not in `$RUNS_DIR`.

## Checks
| ID | Check | Must/Should | How |
|---|---|---|---|
| V1 | Branch, clean tree, commit | Must | git |
| V2 | Lint, format, tests | Must | `ruff format --check . && ruff check . && pytest -q` |
| V3 | Exactly the 18 expected run ids | Must | compare `results/runs.csv` with the product {cnn, resnet18} x {standard, augmix, corruption_aware} x {s0, s1, s2}; no extras, no missing |
| V4 | Every result file passes C7 | Must | your own script: 76 rows, exact columns, `n_images` 10000, 15 corruptions x 5 severities + clean, no duplicates, no NaN, accuracies in [0, 1]; total 1,368 rows |
| V5 | Confusion matrices agree with CSVs | Must | for every run and condition, accuracy == trace/sum |
| V6 | Identical settings across runs | Must | diff every `config_resolved.yaml` against the one for the same model; only `strategy`, `seed` and the strategy's own params may differ; epochs equal D4 per model |
| V7 | Code version consistency | Must | list distinct `git_commit` values; for each pair, `git diff <a> <b> -- src/ configs/` must not change training or evaluation behaviour; any `git_dirty: true` run is a FAIL unless explained and approved |
| V8 | Training logs complete | Must | each `_train_log.csv` has epochs 1..E exactly once, no NaN |
| V9 | Spot re-evaluation | Must | for 2 random runs, re-evaluate on CPU with `--subset 1000` into `/tmp` and compare 3 random conditions with your own loop exactly; full-set CSV within about 2 points of the subset values |
| V10 | Results look like the expected science (sanity, not targets) | Should | ResNet-18 clean > CNN clean for every strategy; corrupted < clean for all runs; report the S3 vs S1 difference on seen corruptions (expected positive). Unexpected findings are not failures, but must be listed |
| V11 | Seed spread | Should | clean accuracy std across seeds per config; flag any > 1.5 points |
| V12 | No smoke, timing or invalid runs mixed in | Must | `results/raw/` contains only the 18 ids |
| V13 | No code or config changes during the matrix | Must | `git log` between the Phase 07 merge and now: any commit touching `src/` or `configs/` needs a `DECISIONS.md` change-log row and evidence that affected runs were redone |
| V14 | RUNLOG and PROGRESS agree with files | Should | compare |

Write `validation/phase-08.md`; give me the one-line verdict.
