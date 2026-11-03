# Phase 01 VALIDATE: Clean CIFAR-10 data

You are the independent validator for Phase 01. Do not trust `PROGRESS.md`; re-run every check. Do not edit repo files except creating `validation/phase-01.md`.

## Read first
`docs/agent-prompts/RULES.md`, `CONTRACTS.md` (C2, C10), `DECISIONS.md` (D1, D2), `docs/agent-prompts/phase-01-data-clean/BUILD.md`, `docs/agent-prompts/templates/VALIDATION-report.md`.

## Checks
Write small check scripts in `/tmp/val01/` (outside the repo) and run them from the repo root.

| ID | Check | Must/Should | How |
|---|---|---|---|
| V1 | Branch, clean tree, commit present | Must | `git status --short`, `git log --oneline -5` |
| V2 | Lint, format, tests | Must | `ruff format --check . && ruff check . && pytest -q` |
| V3 | Split sizes 45,000 / 5,000 / 10,000 | Must | load via the C10 functions |
| V4 | Stratified: 500 per class in val, 4,500 in train | Must | count labels by index from `torchvision.datasets.CIFAR10(train=True).targets` |
| V5 | Disjoint and complete | Must | `set(train) & set(val) == set()`, union == `range(50000)` |
| V6 | Same split in a new process | Must | run the loader twice in separate `python -c` calls; compare hashes of the index lists |
| V7 | Split independent of training seed | Must | build datasets with configs seed 0 and seed 2; indices identical |
| V8 | Mismatch is caught | Must | call `get_split_indices` with `split_seed=7`; must raise, must not overwrite the JSON (check file hash before/after) |
| V9 | Stats computed on train only | Must | recompute mean/std yourself over the 45,000 train indices (pixels /255) and compare to `splits/cifar10_stats.json` within 1e-4; also compute over all 50,000 and confirm the file matches the 45,000 value, not the 50,000 one (they differ in the 4th decimal or so; if identical to 6 decimals, note it) |
| V10 | Eval transform is deterministic | Must | apply `build_eval_transform()` to the same PIL image twice; tensors equal; normalised values use the file's stats |
| V11 | Test set is not used anywhere for training or selection | Must | `grep -rn "train=False" src/` only appears in `get_test_dataset` |
| V12 | Subset works | Must | config with `subset: 64` gives 64 train and 64 val items |
| V13 | DataLoader determinism | Should | two train loaders with the same seed produce the same first-batch label order |
| V14 | No out-of-scope work | Should | no CIFAR-10-C, model or training code added |
| V15 | Notebook calls repo code only | Should | read cells |
| V16 | `PROGRESS.md` claims match | Should | compare numbers in the entry with your own output |

Write `validation/phase-01.md` and give me the one-line verdict.
