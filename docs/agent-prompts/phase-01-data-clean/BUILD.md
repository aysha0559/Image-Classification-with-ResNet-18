# Phase 01 BUILD: Clean CIFAR-10 data

## Read first
1. `docs/agent-prompts/RULES.md`
2. `docs/agent-prompts/DECISIONS.md` D1, D2 (stop if `CONFIRM`)
3. `docs/agent-prompts/CONTRACTS.md` C1, C2, C4, C10
4. `docs/PROJECT_OVERVIEW.md` sections 6.1, 18
5. `PROGRESS.md` (Phase 00 entry) and `validation/phase-00.md` (must say PASS or PASS WITH NOTES; if not, stop)

Start from `main` after Phase 00 is merged: `git checkout -b phase-01-data-clean`.

## Goal
Reliable loaders for clean CIFAR-10: a fixed, stratified 45,000 / 5,000 train/validation split that is identical for every run, normalisation statistics computed from the 45,000 training images, and the test set.

## In scope
- `src/data/cifar10.py` with the C10 functions:
  - Uses `torchvision.datasets.CIFAR10(root=DATA_DIR, download=True)`.
  - `get_split_indices`: stratified, 500 per class in validation, deterministic from `split_seed` only. On first call it writes `splits/cifar10_val_indices.json` (`{"split_seed", "val_size", "val_indices": [...], "train_indices": [...]}`); later calls **read** the file and raise if `split_seed`/`val_size` differ from the file.
  - `get_normalization_stats`: reads `splits/cifar10_stats.json`. A function `compute_normalization_stats(data_dir)` computes per-channel mean and std over the 45,000 training images (pixels scaled to [0,1]) and writes the file. Run it once and commit the file.
  - `get_train_val_datasets(cfg)`: returns `Subset`-style datasets. The training subset gets the transform from `build_train_transform` (Phase 03). Until Phase 03 exists, accept a `train_transform` argument and default to ToTensor + Normalize; leave a clearly named hook, not a stub that silently differs from the contract.
  - Validation and test always use `build_eval_transform()` = ToTensor + Normalize (put this function in `src/data/transforms.py` now; Phase 03 adds the training transforms there).
  - Honors `cfg["subset"]` (first N of each split, after the fixed split) for smoke runs.
- A `make_loaders(cfg)` helper returning train and val `DataLoader`s (shuffle train only; seeded generator and `worker_init_fn` from `src/utils.py`; `pin_memory` when CUDA).
- Commit `splits/cifar10_val_indices.json` and `splits/cifar10_stats.json`.
- `tests/test_cifar10_split.py` using a **fake** small labels array (no download): stratified, disjoint, union complete, deterministic, file round-trip, mismatch raises.
- `notebooks/01_explore_data.ipynb`: a first section showing 10 images per class and the class counts of each split, calling repo code only.

## Out of scope
CIFAR-10-C, training augmentations (S1-S3), models, training.

## Steps
1. Implement, then run `compute_normalization_stats` once on the real data and commit the JSON.
2. Print and record in `PROGRESS.md`: split sizes, per-class counts in each split, mean and std values (from the file).
3. `ruff format --check . && ruff check . && pytest -q`.
4. Progress entry + commit `phase 01: clean CIFAR-10 split, stats, loaders`.

## Done when
- [ ] Train 45,000 / val 5,000 / test 10,000; val has exactly 500 per class; train has 4,500 per class.
- [ ] Train and val indices are disjoint and cover 0..49,999.
- [ ] Re-running from a new Python process returns the identical split (from the file).
- [ ] The split does not depend on the training `seed`.
- [ ] Stats JSON exists, computed on the 45,000 training images only; mean values are each between 0.4 and 0.5 and std between 0.24 and 0.27 (sanity range for CIFAR-10).
- [ ] Validation and test transforms contain no randomness.
- [ ] Lint, format, tests pass.

## Stop and ask me if
The CIFAR-10 download fails or the dataset sizes differ from 50,000/10,000.
