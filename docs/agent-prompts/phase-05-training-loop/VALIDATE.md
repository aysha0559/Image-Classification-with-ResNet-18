# Phase 05 VALIDATE: Training loop

You are the independent validator for Phase 05. Re-run everything on CPU. Do not edit repo files except creating `validation/phase-05.md`. Use a scratch `RUNS_DIR=/tmp/val05/runs` and `RESULTS_DIR=/tmp/val05/results` so you do not touch the repo's results.

## Read first
`docs/agent-prompts/RULES.md`, `CONTRACTS.md` (C4, C5, C6, C8), `DECISIONS.md` (D3, D4), `docs/agent-prompts/phase-05-training-loop/BUILD.md`, `docs/agent-prompts/templates/VALIDATION-report.md`.

## Checks
| ID | Check | Must/Should | How |
|---|---|---|---|
| V1 | Branch, clean tree, commit | Must | git |
| V2 | Lint, format, tests | Must | `ruff format --check . && ruff check . && pytest -q` |
| V3 | Smoke run | Must | `python -m src.train --config configs/smoke.yaml --seed 0`; exits 0; prints one line per epoch |
| V4 | Run folder contract | Must | list `/tmp/val05/runs/smoke-cnn-standard-s0/`; compare file names, `train_log.csv` columns and `run_meta.json` keys with C6 exactly |
| V5 | All 8 combinations train | Must | for each main config and the 2 S0 configs: `--subset 128 --epochs 1`; all exit 0 |
| V6 | Resume equivalence | Must | run A: smoke config `--epochs 3 --run-id smoke-a`; run B: `STOP_AFTER_EPOCHS=2 ... --epochs 3 --run-id smoke-b`, then the same command without the variable; compare logs (1e-6) and `model_final.pt` weights (`allclose atol 1e-6`) |
| V7 | Resume really resumed | Must | run B's second invocation starts at epoch 3 (log shows no epoch 1-2 re-run) |
| V8 | Changed config refused on resume | Must | after `STOP_AFTER_EPOCHS=1`, resume with `--epochs 5`: must refuse and name `epochs` |
| V9 | Finished run is skipped | Must | re-run V3's command; exits 0 quickly without training |
| V10 | Optimiser and schedule per D4 | Must | read code: SGD momentum 0.9 nesterov, wd 5e-4, lr 0.1, cosine `T_max=epochs`, stepped per epoch; check `lr` column falls from 0.1 toward 0 |
| V11 | Final-epoch model saved (D3) | Must | no "best" checkpoint logic; `model_final.pt` equals the last epoch's weights |
| V12 | Atomic checkpoint | Should | read code: temp file then `os.replace` |
| V13 | NaN guard | Should | read code; optionally run with `lr: 1e6` in a temp config and confirm `status: failed` |
| V14 | Failure status | Should | interrupt a run with an exception (for example a bad `DATA_DIR`); `run_meta.json` says `failed` |
| V15 | Smoke results isolated | Must | `ls results/raw/` in the repo has no `smoke-` files; smoke logs went to `$RESULTS_DIR/smoke/` |
| V16 | No test or CIFAR-10-C use in training | Must | grep `src/train.py` for `train=False`, `get_test_dataset`, `cifar10c`: empty |
| V17 | Registry | Should | `RUNS_DIR=/tmp/val05/runs python -m scripts.registry` writes a CSV with the C8 columns (point `RESULTS_DIR` at `/tmp`) |
| V18 | Per-epoch CPU times in `PROGRESS.md` are plausible | Should | compare with your V5 times |
| V19 | Resume does not replay the random stream | Must | read code: the DataLoader generator state is restored before the first iterator is created. Run `corruption_aware`, `num_workers: 2`, subset 256, `--epochs 3` with `STOP_AFTER_EPOCHS=2`, then resume; the resumed epoch's shuffle order and worker base seed differ from the first session's |

Write `validation/phase-05.md`; give me the one-line verdict.
