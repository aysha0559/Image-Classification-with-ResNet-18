# Validation: Phase 01 data-clean

- Date: 2026-11-03
- Branch / commit checked: phase-01-data-clean @ 7544bc68f28565b12c704031aedc65c4b4ed7b29   (from `git rev-parse HEAD`)
- Working tree clean before checking: yes
- Environment: Python 3.12.16, torch 2.14.1+cpu (torchvision 0.29.2+cpu), device cpu (repo `.venv`; real CIFAR-10 in `./data`)
- Verdict: PASS

## Summary
All 16 checks were re-run on the real CIFAR-10 data with check scripts kept outside the repo. The split is 45,000 / 5,000 / 10,000, stratified (4,500 / 500 / 1,000 per class), disjoint and complete. It is the same in separate processes and for training seeds 0 and 2, and it is protected against a mismatched `split_seed`/`val_size` (raises, and the file is unchanged byte for byte and by mtime). The stats file matches my own 45,000-image recomputation exactly (max diff 0.0) and differs from the 50,000-image values by 4.4e-4, so the stats were computed on the training split only. Nothing failed; the notes below are minor.

## Checks
| ID | Check | Must/Should | Result | Evidence |
|---|---|---|---|---|
| V1 | Branch, clean tree, commit present | Must | PASS | `git branch --show-current` -> `phase-01-data-clean`; `git status --short` -> (empty); `git log --oneline -5` -> `7544bc6 phase 01: clean CIFAR-10 split, stats, loaders`, `9f61f71 Merge phase-00-setup ... (validated PASS)`, `23c0f26`, `5d41c93`, `9e94075` |
| V2 | Lint, format, tests | Must | PASS | `ruff format --check .` -> `65 files already formatted`; `ruff check .` -> `All checks passed!`; `time pytest -q` -> `43 passed in 3.70s`, `real 0m4.967s` |
| V3 | Split sizes 45,000 / 5,000 / 10,000 | Must | PASS | `get_split_indices('./data', 5000, 42)`, `get_test_dataset('./data')` -> `V3 sizes 45000 5000 10000` |
| V4 | Stratified: 500 per class in val, 4,500 in train | Must | PASS | `np.bincount` of `CIFAR10(train=True).targets` at the indices -> train `[4500]*10`, val `[500]*10`; test `[1000]*10` |
| V5 | Disjoint and complete | Must | PASS | `set(train) & set(val) == set()` -> `True`; `sorted(set(train) \| set(val)) == list(range(50000))` -> `True`; no duplicates -> `True` |
| V6 | Same split in a new process | Must | PASS | two separate `python -c` calls, sha256[:16] of `json.dumps([train, val])` / train / val -> `d287d84919e6905f 02f9e6d5f3defa57 ef1c409e25946018` both times |
| V7 | Split independent of training seed | Must | PASS | `get_train_val_datasets` with `load_config('configs/cnn_standard.yaml', seed=0)` vs `seed=2` -> `same train True same val True` |
| V8 | Mismatch is caught | Must | PASS | `get_split_indices(D, split_seed=7)` -> `ValueError: .../splits/cifar10_val_indices.json holds split_seed=42, val_size=5000, but ...`; `val_size=4000` -> same `ValueError`; sha256 before/after equal (`c29472f6eba0b349`), mtime equal -> `V8 file unchanged True True` |
| V9 | Stats computed on train only | Must | PASS | file mean `[0.49183993285335986, 0.4825411911934913, 0.44681958341843653]`, std `[0.2470507307231891, 0.24354212224302502, 0.2616922130832278]`, `n_images 45000`. My 45k recompute (/255, ddof=0): identical, `maxdiff file-45k 0.0`. 50k recompute: mean `[0.49140, 0.48216, 0.44653]`, std `[0.24703, 0.24349, 0.26159]`, `maxdiff file-50k 0.00044` -> file matches the 45k values, not the 50k ones. Sanity range: means in (0.4, 0.5), stds in (0.24, 0.27) -> `True True` |
| V10 | Eval transform is deterministic | Must | PASS | `build_eval_transform()` twice on train image 0 -> `torch.equal True`; the transform is `Compose(ToTensor(), Normalize(mean=(0.49183993285335986, ...), std=(0.2470507307231891, ...)))` = file values; manual `(x/255 - mean)/std` vs output -> max diff `2.2e-07` (float32); val item and test item fetched twice -> equal |
| V11 | Test set not used for training or selection | Must | PASS | `grep -rn "train=False" src/` -> only `src/data/cifar10.py:244` in `get_test_dataset`; no other file in `src/` references `get_test_dataset` |
| V12 | Subset works | Must | PASS | `load_config('configs/cnn_standard.yaml', seed=0, subset=64)` -> run id `smoke-cnn-standard-s0`, `64 64` items, item shape `(3, 32, 32)`, indices = first 64 of each fixed split -> `True True` |
| V13 | DataLoader determinism | Should | PASS | two `make_loaders(smoke cfg seed 0)` -> first-batch labels equal `True` (first 10: `[7, 8, 8, 9, 4, 0, 8, 1, 4, 3]`); seed 1 -> differs `True` |
| V14 | No out-of-scope work | Should | PASS | `git diff --name-only 9f61f71 HEAD` -> only `.gitignore, PROGRESS.md, notebooks/01_explore_data.ipynb, splits/*.json, src/data/{__init__,cifar10,cifar10c,corruptions,transforms}.py, src/utils.py, tests/test_cifar10_split.py, tests/test_utils.py`. `cifar10c.py` / `corruptions.py` / `__init__.py` are the 1-line Phase 00 docstring stubs (new to git only because of the `.gitignore` fix). No models, training, CIFAR-10-C or augmentation code added |
| V15 | Notebook calls repo code only | Should | PASS | 8 cells, no saved outputs; code cells only import and call `examples_per_class`, `split_class_counts`, `get_normalization_stats`, `env_path`, then plot / build a DataFrame. Executed headless from the repo root: counts table totals `45000 5000 10000`, images `(10, 10, 32, 32, 3) uint8` |
| V16 | `PROGRESS.md` claims match | Should | PASS | sizes, per-class counts, split hash `d287d84919e6905f`, seed-independence, mean/std values (all digits), 50k comparison values, subset `64 64`, loader determinism, `grep` line 244, lint counts (`65 files`) and `43 passed` all reproduced. `.gitignore` claim checked: `git check-ignore -v` -> `/data/`, `/runs/`, `/dist/` ignore `data/x runs/x dist/x`; `src/data/*.py` is now tracked (`git ls-files src/data` -> 5 files) |

## Failures (if any)
None.

## Notes (non-blocking)
- Check scripts were kept in the session scratchpad (outside the repo) rather than `/tmp/val01/`; same intent (outside the repo). They needed `PYTHONPATH=.` because the script lives outside the repo root.
- V13 was checked with the smoke config (`num_workers: 0`). The `num_workers > 0` path (`seed_worker`, `persistent_workers`) was not exercised; worth a quick check in the training phase on Colab.
- `load_or_create_split` writes the split JSON with a plain `write_text`, not `atomic_save`. Only matters if the file is missing (it is committed), so harmless now.
- The `.gitignore` change from Phase 00 (`data/` -> `/data/`) is a real fix: before it, `src/data/` was never committed. It is recorded under "Changed from earlier phases" as RULES section 2 asks.
- `make_loaders` uses `drop_last=False`, as PROGRESS notes; BUILD does not specify this.

## Claims in PROGRESS.md that could not be confirmed
- None. (The ~35 min download time was not re-measured: the data was already present.)
