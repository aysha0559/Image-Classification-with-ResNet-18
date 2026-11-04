# Validation: Phase 07 pilot

- Date: 2026-11-04
- Branch / commit checked: phase-07-pilot @ e80d814e6ce68152cbd64c31b486bf43672cfa91
- Working tree clean before checking: yes
- Environment: Python 3.12.16, torch 2.11.0+cpu, torchvision 0.26.0+cpu, device cpu
- Verdict: PASS

## Summary
All 16 checks pass. The pilot is a real 30-epoch T4 run: `model_final.pt` reproduces the logged epoch-30 validation accuracy and, through a separate loop that shares no data code with `src.evaluate`, the CSV's accuracy on full clean and `gaussian_noise` severity 5. Every number in `results/pilot_report.md` recomputes. The one limit is provenance: the timing and console logs are not in the repo, so V6, V10 and V11 used the copy the build session downloaded, not the Drive originals.

## Checks
| ID | Check | Must/Should | Result | Evidence |
|---|---|---|---|---|
| V1 | Branch, clean tree, commit | Must | PASS | `git branch --show-current` -> `phase-07-pilot`; `git status --short \| wc -l` -> `0`, also after all checks; `main` is an ancestor of HEAD |
| V2 | Lint, format, tests | Must | PASS | `ruff format --check .` -> `81 files already formatted`; `ruff check .` -> `All checks passed!`; `time pytest -q` -> `265 passed in 33.22s` (real 0m34.577s, CPU) |
| V3 | Results files complete | Must | PASS | CSV `(76, 14)`, columns equal C7 in order, `n_images` only `[10000]`, no NaN, no repeated condition, conditions exactly clean + 15 x 5; family and group match C3 and D6. npz: 76 keys, all 10x10 int64, each sums to 10000, each row to 1000. Max \|CSV accuracy - matrix\| 0; max \|CSV macro-F1 - scikit-learn on the matrix\| 5.6e-17 |
| V4 | Real full run | Must | PASS | `run_meta.json`: `status` finished, `epochs` 30, `device_name` Tesla T4, `git_dirty` false, `git_commit` 004edc3c..., and `git cat-file -t` -> `commit`, an ancestor of HEAD. `git diff --stat 004edc3 e80d814 -- src scripts configs notebooks splits requirements.txt` -> only 4 ResNet config comment lines. `runs.csv` agrees. Config in `model_final.pt` equals `configs/cnn_standard.yaml` apart from the added `run_id`; `subset` is None |
| V5 | Training log sane | Must | PASS | 30 rows, epochs 1..30 once each, no NaN or inf. Against `0.05 * (1 + cos(pi * (e - 1) / 30))`: max difference 1.1e-16, `allclose` True, lr 0.1 to 0.000274. Val accuracy 0.2584 at epoch 1, 0.8474 at epoch 30 |
| V6 | Resume evidence | Should | PASS | Console log: `Traceback` at epoch 12, then `cnn-standard-s0: resuming after epoch 11/30`, then `epoch 12/30`; no `WARNING`. No duplicated epochs (V5). Independent of that log: `finished_at - started_at` = 829 s against `train_time_s` 620.0 s, a 209 s gap. Mean epoch 22.5 s for epochs 1-11, 19.6 s for epochs 12-30. `train_time_s` equals the sum of `epoch_time_s` (difference 0.0) |
| V7 | Clean accuracy within gate | Must | PASS | Clean row accuracy 0.8404, inside [0.78, 0.90] |
| V8 | Independent re-evaluation on CPU | Must | PASS | Own loop: raw CIFAR-10 pickle and `gaussian_noise.npy` read directly, normalised from `splits/cifar10_stats.json`. See the V8 table below. `src.evaluate --run-dir runs/cnn-standard-s0 --subset 2000` -> `76 conditions, 2000 images each`, written to a scratch `RESULTS_DIR` under `/tmp` |
| V9 | Corruption sanity | Must | PASS | Mean corrupted 0.597372, 24.30 points below clean. Severity 1-5: 0.741800, 0.667427, 0.602327, 0.532767, 0.442540, strictly decreasing |
| V10 | Report numbers trace to files | Must | PASS | No mismatches. Gates, results, 5 worst conditions, severity table and training curve from `results/raw/`; timings from the logs named in the report (provenance note below) |
| V11 | Timing projection arithmetic | Must | PASS | Recomputed from the per-epoch means: 30 epochs 4.175 + 4.557 + 0.980 = 9.713; 40 epochs 4.175 + 6.077 + 0.980 = 11.232; 50 epochs 4.175 + 7.596 + 0.980 = 12.751. Report: 9.71, 11.23, 12.75. Same from the rounded values in the report |
| V12 | Epochs frozen correctly | Must | PASS | D4 `**40.** ... CONFIRMED`; change-log row dated 2026-11-04. `grep '^epochs' configs/*.yaml`: all 4 `resnet18_*` 40, all 4 `cnn_*` 30, `smoke` 1. `git diff main -- configs/` -> one comment line in each of 4 ResNet configs, nothing else |
| V13 | No timing or smoke runs in results | Must | PASS | `ls -A results/raw` -> `.gitkeep` and the 3 `cnn-standard-s0*` files; `results/smoke/` does not exist; sha256 of the 3 files recorded after V8, and `git status` still clean |
| V14 | No tuning on test data | Must | PASS | `git diff main -- src/ configs/` -> `src/train.py` skips an empty scaler state on resume, plus the V12 comments. No hyperparameter changed |
| V15 | Versions recorded | Should | PASS | `run_meta.json`: torch 2.11.0+cu130, torchvision 0.26.0+cu130, Python 3.13.16. `requirements-local.txt`: 2.11.0, 0.26.0, the same major.minor (D9). The local Python is 3.12.16 |
| V16 | Notebook has no logic | Should | PASS | All 22 cells read: parameters, shell calls, copies and step checks. No model, training or metric logic; no stored outputs |

### V8 detail

| Condition | Own loop, 2,000 | `--subset 2000` | Macro-F1 difference | Loss difference | Full CSV | Points from subset | Own loop, 10,000 |
|---|---|---|---|---|---|---|---|
| clean | 0.8405 | 0.8405 | 0.0 | 2.3e-08 | 0.8404 | 0.01 | 0.8404 |
| gaussian_noise s5 | 0.1870 | 0.1870 | 0.0 | 9.2e-08 | 0.1791 | 0.79 | 0.1791 |

- Accuracy and macro-F1 are exact on 2,000 images. Loss differs only in the last digits of float32 sums, because the batch sizes differ (256 and 512).
- Full-set accuracy on CPU equals the GPU CSV. Macro-F1 differs by at most 5.6e-06 and loss by at most 0.00195, as expected between CPU and GPU with AMP.
- Extra check, 5,000 validation images, own loop: accuracy 0.8474, equal to epoch 30 of the log; loss 0.43683 against 0.43688. This shows `model_final.pt` is the final-epoch model (D3).

## Failures (if any)
None.

## Notes (non-blocking)
- **Provenance of the timing logs.** `cnn-standard-s0.console.log`, `timing.console.log` and the `timing-*` `train_log.csv` and `run_meta.json` files are not in the repo. They were read from the build session's scratch download. Searching the connected Google Drive found no project files, so that copy was not compared with Drive. Supporting evidence: its `results/raw/` files, `runs.csv` and `run_meta.json` are byte-identical to the repo's; all 5 timing runs show Tesla T4, commit 004edc3, not dirty, status `running`; their `started_at` times are in the order the notebook runs them and fit the logged epoch times. The report writes the timing values out in full, as it says it must, because those runs were deleted.
- The console log has 2 tracebacks at the disconnect, and the report describes 1. The first is `_append_log` (`src/train.py`, line 192 at the run's commit). The second comes from `_write_meta` while recording the failure, also because Drive had gone. So `status` stayed `running`, not `failed`, and the resume was unaffected.
- The timeline supports PROGRESS's reading. Epoch 11 ended at about 13:17:30, so epoch 12 ended at about 13:17:51, and PROGRESS gives the next session as 13:17:55. That margin is small: a re-run started sooner could overlap a dying process on the same Drive files. Worth keeping in mind for Phase 08.
- Mean accuracy falls with severity, but 2 corruptions are not monotonic: glass_blur 0.3263, 0.3421, 0.3840, 0.2833, 0.3080 and snow 0.7572, 0.5673, 0.6462, 0.6318, 0.5506. V9 asks only about the mean, and Phase 02 recorded glass_blur's pixel difference as non-monotonic too.
- The first 2 epochs may overstate ResNet-18 time: the pilot's epoch 1 took 26.84 s against a 20.45 s mean for the rest. Session speed also varied about 13% (22.5 s against 19.6 s per epoch). The 11.23 h projection leaves plenty of room under the 20 h budget, so neither changes the decision.
- The report says 20 hours would allow "about 97 epochs"; the value is 97.7. Correct as an approximation.
- `requirements.txt` was installed on Python 3.13 on Colab, which Phase 00's install check did not cover. PROGRESS already says so.
- `git grep` for the human's name finds nothing in tracked files.

## Claims in PROGRESS.md that could not be confirmed
- "Page output stopped at epoch 6" and the `Connecting` / `Runtime disconnected` page states -> seen by the human on Colab; no file records them. Epochs 7-11 are in the console log.
- `=== session 2026-11-04T13:17:55Z` -> absent from the console log, as PROGRESS says. The timeline above is consistent with it.
- "Session setup about 4 minutes" -> no file gives this directly. The resume started at about 13:20:59 by the log's epoch times, about 3 minutes after the 13:17:55 banner, and the cells before that banner are not timed. Plausible.
- Timing folders moved to Drive Trash (`6 items moved to trash`) -> not visible through the connected Drive. Only `cnn-standard-s0` is in the local `runs/`.
- "Results are unaffected" by the interrupted first T4 session, and `all 20 files match` in every session -> the console log does not hold the verification output. The pilot's results themselves were validated above.
