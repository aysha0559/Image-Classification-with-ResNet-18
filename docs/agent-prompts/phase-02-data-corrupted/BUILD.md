# Phase 02 BUILD: CIFAR-10-C download, verification and loader

## Read first
1. `docs/agent-prompts/RULES.md` (section 3 on leakage especially)
2. `docs/agent-prompts/DECISIONS.md` D7
3. `docs/agent-prompts/CONTRACTS.md` C1, C2, C3, C5, C10
4. `docs/PROJECT_OVERVIEW.md` sections 6.2, 12 (Colab practicalities), 18
5. `PROGRESS.md` and `validation/phase-01.md` (must be PASS or PASS WITH NOTES)

Branch: `phase-02-data-corrupted` from `main`.

## Goal
Download CIFAR-10-C once, prove the files are complete and correct, and load any (corruption, severity) pair as 10,000 images with labels. Produce the sample image grid (figure 7).

## Facts you must verify, not assume
- The dataset is on Zenodo record 2535967 (`https://zenodo.org/records/2535967`). Get the archive's file name, size and **checksum from Zenodo's own record metadata** (the record page or `https://zenodo.org/api/records/2535967`). Do not type a checksum from memory.
- Expected after extraction: a `CIFAR-10-C/` folder with 19 corruption `.npy` files and `labels.npy`. Each corruption file is uint8 with shape (50000, 32, 32, 3): severity 1 is rows 0-9,999, severity 2 is rows 10,000-19,999, and so on. `labels.npy` has shape (50000,).
- If anything you observe differs from this, stop and tell me.

## In scope
- `scripts/download_cifar10c.py` (+ `scripts/download_cifar10c.sh` wrapper):
  - Downloads to `DATA_DIR` with resume support and a progress bar; skips download if the archive already verifies.
  - Verifies the archive checksum against the value fetched from Zenodo, extracts, then computes the **sha256 of every extracted `.npy`** and writes `splits/cifar10c_checksums.json` (`{"source": "zenodo 2535967", "archive": {"name", "size", "checksum_type", "checksum"}, "files": {"gaussian_noise.npy": "<sha256>", ...}}`). Commit this file.
  - On later runs, `--verify-only` re-checks the extracted files against the JSON (used on Colab after copying from Drive).
- `src/data/cifar10c.py` with C10 `load_cifar10c` and `CIFAR10C`:
  - Validates the corruption name against `src/constants.py` and severity in 1..5.
  - Uses `np.load(..., mmap_mode="r")` and slices the 10,000 rows for the severity (do not load all 50,000 into memory).
  - `CIFAR10C` applies `build_eval_transform()` by default.
- A module-level note in `cifar10c.py`'s docstring: "Test-only. Never import from training code." Plus a tiny guard test (below).
- `scripts/make_sample_grid.py`: one test image (default index chosen so the object is clear; record which) under 15 corruptions x 5 severities, plus the clean image, as a labelled grid -> `results/figures/fig7_sample_grid.png` and `.pdf`.
- `notebooks/00_colab_setup.ipynb`: add cells to download CIFAR-10-C to Drive once (`MyDrive/cifar10-robustness/data/`), and on later sessions copy it to `/content/data` and run `--verify-only`.
- `notebooks/01_explore_data.ipynb`: add a section displaying the grid.
- Tests (no real download): `tests/test_cifar10c.py` builds a fake `CIFAR-10-C/` folder in a temp dir with tiny arrays (for example 5 images per severity, by monkeypatching the per-severity count) and checks slicing per severity, name/severity validation, label dtype. `tests/test_no_leakage.py`: fails if any file under `src/train.py`, `src/data/transforms.py`, `src/data/corruptions.py` or `src/models/` contains `cifar10c` or `CIFAR-10-C`.

## Out of scope
Training transforms, our own corruption functions, models, evaluation loop.

## Steps
1. Implement, download for real (about 2.9 GB), verify, write checksums JSON.
2. Check alignment yourself: `labels.npy[s*10000:(s+1)*10000]` equals the CIFAR-10 test labels for every severity s. Record the result.
3. Make the sample grid and look at it (open the PNG). Each column should be visibly stronger than the last.
4. Lint, format, tests. Progress entry. Commit `phase 02: CIFAR-10-C download, verify, loader, sample grid`.

## Done when
- [ ] Archive checksum matches the value from Zenodo's metadata (show both).
- [ ] 19 corruption files + `labels.npy` present; every corruption file is uint8 (50000, 32, 32, 3).
- [ ] `splits/cifar10c_checksums.json` committed with 20 file hashes.
- [ ] For every severity, CIFAR-10-C labels equal the CIFAR-10 test labels in order.
- [ ] `load_cifar10c("fog", 3)` returns (10000, 32, 32, 3) uint8 and (10000,) labels.
- [ ] Bad corruption name or severity 0/6 raises `ValueError`.
- [ ] Sample grid exists and is labelled with corruption names and severities.
- [ ] Leakage test exists and passes.
- [ ] Lint, format, tests pass.

## Stop and ask me if
The download is blocked, the checksum fails twice, or any shape, count or label alignment differs from the facts above.
