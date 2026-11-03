# Phase 07 VALIDATE: Pilot run

You are the independent validator for Phase 07. You check that the pilot is real, correct and correctly reported. You do not need a GPU: you check files, and re-evaluate on a subset on CPU. Do not edit repo files except creating `validation/phase-07.md`.

## Read first
`docs/agent-prompts/RULES.md`, `CONTRACTS.md` (C6, C7, C8), `DECISIONS.md` (D3, D4, D9 and the change log), `docs/PROJECT_OVERVIEW.md` sections 7, 10.2, `docs/agent-prompts/phase-07-pilot/BUILD.md`, `docs/agent-prompts/templates/VALIDATION-report.md`, `results/pilot_report.md`.

Ask the human for `model_final.pt` of `cnn-standard-s0` (from Drive) if it is not in `$RUNS_DIR`; it is needed for V8.

## Checks
| ID | Check | Must/Should | How |
|---|---|---|---|
| V1 | Branch, clean tree, commit | Must | git |
| V2 | Lint, format, tests | Must | `ruff format --check . && ruff check . && pytest -q` |
| V3 | Results files complete | Must | `results/raw/cnn-standard-s0.csv` has 76 rows, C7 columns, `n_images` 10000 on every row; npz has 76 matrices each summing to 10000 |
| V4 | Run was a real full run | Must | `run_meta.json` / `runs.csv`: status `finished`, epochs 30, CUDA device name present, `git_commit` exists in `git log` |
| V5 | Training log sane | Must | `results/raw/cnn-standard-s0_train_log.csv`: 30 rows, epochs 1..30 once each, lr follows cosine from 0.1, no NaN, val accuracy at epoch 30 > epoch 1 |
| V6 | Resume evidence | Should | the log or `PROGRESS.md` shows the disconnect and resume; no duplicated epochs in the log |
| V7 | Clean accuracy within gate | Must | clean row accuracy in [0.78, 0.90], or a written human decision in `PROGRESS.md` |
| V8 | Independent re-evaluation on CPU | Must | with `model_final.pt`, your own loop on the first 2,000 clean test images and the first 2,000 of `gaussian_noise` severity 5; compare with a `--subset 2000` run of `src.evaluate` (to `/tmp`) exactly; and the full-set CSV values should be within about 2 points of your subset values |
| V9 | Corruption sanity | Must | mean corrupted accuracy < clean by more than 10 points; mean accuracy by severity decreases from 1 to 5 |
| V10 | Pilot report numbers trace to files | Must | for every number in `results/pilot_report.md`, recompute it from the CSV/log and list mismatches |
| V11 | Timing projection arithmetic | Must | recompute total GPU hours from the stated per-epoch times and epochs |
| V12 | Epochs frozen correctly | Must | D4 ResNet epochs `CONFIRMED` with a change-log row; all ResNet configs use that value; CNN configs 30; nothing else in configs changed vs Phase 06 (`git diff main -- configs/`) |
| V13 | No timing or smoke runs in results | Must | `results/raw/` has only `cnn-standard-s0*` files |
| V14 | No tuning on test data | Must | `git diff main -- src/ configs/`: no hyperparameter change other than ResNet epochs; if any, FAIL unless the human approved it in writing |
| V15 | Versions recorded | Should | Colab torch/torchvision in `run_meta.json` vs `requirements-local.txt`; note differences |
| V16 | Notebook has no logic | Should | read `notebooks/run_on_colab.ipynb` |

Write `validation/phase-07.md`; give me the one-line verdict.
