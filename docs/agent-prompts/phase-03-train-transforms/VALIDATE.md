# Phase 03 VALIDATE: Training transforms for S0-S3

You are the independent validator for Phase 03. This phase carries the highest leakage risk; be strict. Do not edit repo files except creating `validation/phase-03.md`.

## Read first
`docs/agent-prompts/RULES.md`, `CONTRACTS.md` (C3, C4, C10), `DECISIONS.md` (D5, D6), `docs/agent-prompts/phase-03-train-transforms/BUILD.md`, `docs/agent-prompts/templates/VALIDATION-report.md`.

## Checks
| ID | Check | Must/Should | How |
|---|---|---|---|
| V1 | Branch, clean tree, commit | Must | git |
| V2 | Lint, format, tests | Must | `ruff format --check . && ruff check . && pytest -q` |
| V3 | Leakage: no CIFAR-10-C in training code | Must | `grep -rniE "cifar10c\|CIFAR-10-C\|\.npy" src/data/transforms.py src/data/corruptions.py src/data/cifar10.py src/train.py src/models/` is empty |
| V4 | Leakage: import graph | Must | `python -c "import sys, src.data.transforms, src.data.corruptions; print([m for m in sys.modules if 'cifar10c' in m])"` prints `[]` |
| V5 | Only seen corruptions implemented | Must | read `corruptions.py`; `apply_corruption(img, "fog", 1, rng)` raises; the supported set equals D6 seen list |
| V6 | Parameters cited and match the reference | Must | open the cited reference file in `github.com/hendrycks/robustness` yourself; compare the 5 severity parameters for each of the 4 corruptions with the code. List any mismatch |
| V7 | Each strategy pipeline is exactly D5 | Must | print `build_train_transform(s, params)` for each strategy; compare ops and arguments, including AugMix `severity=3, mixture_width=3, chain_depth=-1, alpha=1.0, all_ops=False` |
| V8 | Normalisation uses `splits/cifar10_stats.json` | Must | inspect the Normalize op's mean/std |
| V9 | Output contract | Must | each strategy on a real training PIL image -> float tensor (3,32,32) |
| V10 | Determinism | Must | `apply_corruption` with same seed twice -> identical arrays; `none` and eval transforms identical across calls |
| V11 | Severity ordering | Must | for each of the 4, mean abs difference on 200 train images increases from severity 1 to 5 |
| V12 | Probability `p` | Must | with `p=0.5` on 2,000 calls, fraction changed in 0.45-0.55; `p=0` -> 0 changed; `p=1` -> all changed |
| V13 | Per-worker randomness | Should | a DataLoader with `num_workers=2` does not give identical corruption choices in both workers (read the code and/or test) |
| V14 | Comparison grid looks right | Should | open `results/figures/extra_our_corruptions.png`; describe whether ours and CIFAR-10-C's look similar in kind and strength |
| V15 | Strength table in `PROGRESS.md` matches a re-run | Should | recompute 2 rows |
| V16 | Datasets wired to transforms | Must | `get_train_val_datasets(cfg)` for `corruption_aware` config uses the S3 transform for train and the eval transform for val |

Write `validation/phase-03.md`; give me the one-line verdict.
