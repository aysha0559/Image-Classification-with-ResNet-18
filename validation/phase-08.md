# Validation: Phase 08 experiment-matrix

- Date: 2026-11-10
- Branch / commit checked: phase-08-experiment-matrix @ 58a0f0ab07d241799cba4b1fd7784d93501ce1b9
- Working tree clean before checking: yes (`git status --short` empty)
- Environment: Python 3.12.16 (`.venv`), torch 2.11.0+cpu / torchvision 0.26.0+cpu, device cpu
- Verdict: PASS WITH NOTES

## Summary
All 18 runs are present, finished, evaluated on all 76 conditions (1,368 rows), and consistent with their confusion matrices, configs, training logs and code version. Every Must check passes. A spot re-evaluation of 2 randomly picked runs, using checkpoints downloaded from Drive, reproduced the evaluator exactly with an independent loop. One Should check (V14) has a minor finding: PROGRESS.md says the Kaggle run's torch version is "the same as the Colab runs", but the Colab runs used two torch versions (2.11.0 on 2026-11-04, 2.14.1 on 2026-11-09), and neither PROGRESS.md nor RUNLOG.md records the change.

## Checks
| ID | Check | Must/Should | Result | Evidence |
|---|---|---|---|---|
| V1 | Branch, clean tree, commit | Must | PASS | `git branch --show-current` -> `phase-08-experiment-matrix`; `git status --short` -> empty; HEAD `58a0f0a phase 08: full experiment matrix results` |
| V2 | Lint, format, tests | Must | PASS | `ruff format --check .` -> `85 files already formatted`; `ruff check .` -> `All checks passed!`; `pytest -q` -> `317 passed in 43.72s` |
| V3 | Exactly the 18 expected run ids | Must | PASS | scratch script on `results/runs.csv`: `n rows 18 dup 0 missing set() extra set()`; `status {'finished': 18} has_results {True: 18}` |
| V4 | Every result file passes C7 | Must | PASS | own script: exact 14 columns in C7 order, 76 rows each, `n_images` 10000, conditions = clean + 15x5, no duplicates, no NaN, accuracy/macro_f1 in [0,1], id/model/strategy/seed fields match file name, family and group labels match C3/D6 -> `total rows 1368 V4 failures []` |
| V5 | Confusion matrices agree with CSVs | Must | PASS | every npz has exactly the 76 keys, 10x10 int, sum 10000; `V5 failures [] count 0 max |acc - trace/sum| 0` |
| V6 | Identical settings across runs | Must | PASS | flattened `runs/*/config_resolved.yaml` compared per model: only `strategy`, `seed` (and `run_id`) differ; `epochs simple_cnn {30}`, `epochs resnet18 {40}` = D4; optimiser sgd/0.1/0.9/nesterov/5e-4, cosine, batch 128, amp, split_seed 42, val_size 5000, subset null in all |
| V7 | Code version consistency | Must | PASS | distinct `git_commit`: `7d334eb` (17 runs), `004edc3` (pilot `cnn-standard-s0` only); both ancestors of `main`; `git diff --stat 004edc3 7d334eb -- src/ configs/` -> 4 ResNet config files, 1 comment line each (`# Proposed D4 value (40)...` -> `# D4: 40 epochs, confirmed...`), no change under `src/`; `git_dirty` False for all 18 |
| V8 | Training logs complete | Must | PASS | each `results/raw/*_train_log.csv`: epochs exactly 1..30 (CNN) / 1..40 (ResNet), C6 columns, no NaN, identical to `runs/<id>/train_log.csv`; final lr 2.74e-4 (CNN) / 1.54e-4 (ResNet) in every run |
| V9 | Spot re-evaluation | Must | PASS | random pick (`random.sample` with a fresh seed) -> `cnn-standard-s2`, `cnn-corruption_aware-s0`; their `model_final.pt` downloaded from Drive `runs/<id>/` (run_id inside checked). `python -m src.evaluate --run-dir runs/<id> --subset 1000 --force` with `RESULTS_DIR` in the scratch dir -> rc 0, 76 rows, n_images 1000. Own loop (raw `.npy` / `test_batch` pickle, normalise with `splits/cifar10_stats.json`, repo model) on 3 random conditions each, all exact: cnn-standard-s2 elastic_transform s3 0.7290 (full 0.7372), impulse_noise s4 0.2650 (full 0.2761), fog s5 0.5020 (full 0.5097); cnn-corruption_aware-s0 shot_noise s2 0.8030 (full 0.7987), glass_blur s5 0.5610 (full 0.5561), zoom_blur s4 0.7960 (full 0.7780). Full minus subset over all 76 conditions: mean +0.26 / -0.41 pp, max abs 2.26 / 3.42 pp |
| V10 | Expected science (sanity) | Should | PASS | clean mean ResNet-18 vs CNN: standard 93.69 vs 84.46, augmix 93.69 vs 83.39, corruption_aware 92.46 vs 82.31; mean corrupted < clean in every run (`True`). S3 - S1 on seen corruptions: CNN +19.94 pp, ResNet-18 +20.49 pp (positive as expected). Unexpected findings under Notes |
| V11 | Seed spread | Should | PASS | clean accuracy std over seeds (pp): cnn standard 0.616, augmix 0.585, corruption_aware 0.371; resnet18 standard 0.056, augmix 0.025, corruption_aware 0.206; max-min at most 1.13 pp; none > 1.5 |
| V12 | No smoke, timing or invalid runs mixed in | Must | PASS | `results/raw/`: 54 files = the 18 ids x {csv, confusion, train log}, plus `.gitkeep`; no `smoke-` or other ids |
| V13 | No code or config changes during the matrix | Must | PASS | `git log --oneline e4481c9..HEAD -- src/ configs/` -> empty; `git log 7d334eb..HEAD -- src/ configs/` -> empty. Changes after `7d334eb` are docs, notebook `run_on_kaggle.ipynb`, results and PROGRESS/RUNLOG; the Kaggle change has a DECISIONS change-log row (2026-11-10) |
| V14 | RUNLOG and PROGRESS agree with files | Should | FAIL | Registry table, seed table, commits, epochs, 1,368 rows, interrupted runs and the Kaggle move agree with the files. Disagreement: `run_meta.json` torch versions are `2.11.0+cu130` for 11 runs (Colab 2026-11-04 and the Kaggle run) and `2.14.1+cu130` / torchvision `0.29.1+cu130` for 7 runs (Colab 2026-11-09: cnn-augmix-s1/s2, cnn-corruption_aware-s2, cnn-standard-s2, resnet18-augmix-s1, resnet18-corruption_aware-s2, resnet18-standard-s2). PROGRESS says the Kaggle versions are "the same as the Colab runs" and "Same commit, GPU model and torch version"; RUNLOG does not mention the change |

## Failures (if any)
### V14: torch version change not recorded (Should)
- Expected: PROGRESS.md and RUNLOG.md agree with the run files (Phase 08 VALIDATE V14).
- Found: two torch/torchvision versions inside the Colab runs (above). PROGRESS.md, Phase 08 results entry, "Verified" and "Known issues", says the Kaggle run matches the Colab runs; it matches only the 2026-11-04 Colab runs.
- Likely cause: Colab's preinstalled torch changed between the 2026-11-04 and 2026-11-09 sessions. DECISIONS D9 allows this (use the platform's version, record it in metadata, which was done), so it is not a rule violation. Only the narrative is incomplete.
- Suggested fix: add a line to PROGRESS.md Known issues and the 2026-11-09 RUNLOG rows: torch 2.14.1 / torchvision 0.29.1 for the 7 runs listed; disclose it in the report's limitations next to the Kaggle exception.

## Notes (non-blocking)
- 21 of 1,350 corrupted conditions score at or slightly above the run's clean accuracy, all at severity 1-2 (defocus_blur, fog, brightness, contrast s1); largest gap +0.52 pp (cnn-augmix-s1 defocus_blur s1). Within sampling noise; per-run mean corrupted accuracy is always below clean.
- Mean accuracy over corruptions falls with severity in every run (as PROGRESS says), but per corruption 63 of 270 (run, corruption) curves are not monotone (glass_blur 18, snow 16, elastic_transform 13, motion_blur 8, zoom_blur 5, frost 3). This is a known property of CIFAR-10-C severity levels, not a defect.
- S3 (corruption_aware) lowers clean accuracy against S1: CNN -2.15 pp, ResNet-18 -1.23 pp (seed means). S3 also beats AugMix on unseen corruptions for both models (mean unseen accuracy S3 vs AugMix vs S1: CNN 73.67 / 67.67 / 60.34, ResNet-18 86.63 / 84.04 / 72.30), mostly from the same-family noise and blur corruptions. Brightness is the only corruption where S3 is below S1 for both models (-1.18 / -0.77 pp). Worth discussing in Phase 09/10; not failures.
- V9 subset vs full set: the 6 sampled conditions are all within 1.8 pp; over all 76 conditions 4 of 152 differ by more than 2 pp (max 3.42 pp, snow s5). With 1,000 images the standard error at 50-80% accuracy is about 1.3-1.6 pp, so this is expected sampling noise.
- `resnet18-augmix-s2` (Kaggle) trained in 2,502 s against about 4,600-4,700 s for the other ResNet AugMix runs; PROGRESS explains it (CPU-bound AugMix, more CPU on Kaggle). Its clean accuracy 0.9369 is in line with seeds 0 and 1 (0.9372, 0.9367).
- `cnn-augmix-s0` and `resnet18-standard-s2` were resumed after interruptions; their clean accuracies are within the seed spread of their configs.
- The partial Colab `runs/resnet18-augmix-s2` (8 epochs) is still on Drive without the `_invalid-moved-to-kaggle` suffix, as PROGRESS records; it is not in `results/`.
- Checkpoints used for V9 were copied into the git-ignored `runs/cnn-standard-s2/` and `runs/cnn-corruption_aware-s0/`; smoke outputs went to the session scratch directory, not the repo.

## Claims in PROGRESS.md that could not be confirmed
- "Kaggle run versions: ... torch `2.11.0+cu130`, torchvision `0.26.0+cu130` ...: the same as the Colab runs" and "Same commit, GPU model and torch version" -> true for the 2026-11-04 Colab runs only; 7 Colab runs from 2026-11-09 used torch `2.14.1+cu130`, torchvision `0.29.1+cu130` (see V14).
- All other checked claims (registry table, seed-consistency table, 1,368 rows, commits, epochs, `git_dirty` False, config identity, lint/test results) were confirmed.
