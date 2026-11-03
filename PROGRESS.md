# Progress log

One entry per phase, appended at the end of each build phase
(template: `docs/agent-prompts/templates/PROGRESS-entry.md`). Validators re-check every claim.

## Phase 00: setup  (2026-11-03, branch phase-00-setup, commit: the one titled "phase 00: project skeleton, config, seeding, guardrails")

### Built
- Git repo: `main` holds the pre-existing `docs/` as a baseline commit; this phase is on `phase-00-setup`.
- Folder layout of CONTRACTS C1: every later-phase module exists as a one-line docstring naming the phase that fills it; empty result/report/app/validation/splits folders hold a `.gitkeep`.
- `.gitignore`, `pyproject.toml` (ruff line length 100, pytest settings), `requirements.txt`, `requirements-local.txt`.
- `src/constants.py`: class names, 15 corruptions + 4 extras, families, seen/unseen lists, three analysis groups (C3, D6).
- `src/config.py`: `load_config(path, seed, epochs, subset, run_id)`, `validate_config(cfg)`, `build_run_id(...)`. Rejects unknown/missing keys, wrong types and invalid choices with a `ValueError` naming the file and key. A `subset` adds the `smoke-` prefix to the run id.
- `src/utils.py`: `set_seed`, `get_device`, `get_git_commit`, `is_git_dirty`, `atomic_save`, `env_path` (C2 defaults), `Timer`.
- `configs/`: 6 main + 2 optional S0 (`*_none.yaml`) + `smoke.yaml`; values from C4/D4/D5; ResNet epochs = 40 (proposed D4 value).
- `tests/test_constants.py`, `tests/test_config.py`, `tests/test_utils.py`.
- `notebooks/00_colab_setup.ipynb`: mount Drive, clone/pull `REPO_URL` (set to the GitHub repo) at `BRANCH`, `pip install -r requirements.txt`, set `DATA_DIR`/`RUNS_DIR`, print python/torch/torchvision/GPU.
- `CLAUDE.md`, `README.md` (placeholder), `LICENSE` (MIT, holder "cifar10-robustness-study contributors": the human asked to stay anonymous), this file.
- GitHub: public repo https://github.com/aysha0559/Image-Classification-with-ResNet-18 created at the human's request; `main` and `phase-00-setup` pushed. Not merged (merge waits for validation).

### How to run it
    # Python 3.12 was used (this machine only had 3.14; 3.12.16 was fetched with `uv python install 3.12`)
    python3.12 -m venv .venv
    .venv/bin/pip install -r requirements-local.txt -r requirements.txt
    .venv/bin/ruff format --check . && .venv/bin/ruff check .
    .venv/bin/pytest -q
    .venv/bin/python -c "from src.config import load_config; c = load_config('configs/resnet18_augmix.yaml', seed=2); print(c['run_id'])"

### Verified (with output)
- Lint: `ruff format --check .` -> `61 files already formatted`; `ruff check .` -> `All checks passed!`
- Tests: `time pytest -q` -> `33 passed in 1.57s` (real 0m2.362s, CPU)
- run_id: the step-4 command -> `resnet18-augmix-s2`; all 6 main configs x seeds 0,1,2 print `{cnn|resnet18}-{strategy}-s{0,1,2}`; `smoke.yaml` -> `smoke-cnn-standard-s0`
- All 9 configs load (`load_config` over `configs/*.yaml`).
- Bad config, `epoch:` typo -> `ValueError: .../bad.yaml: unknown config key(s) ['epoch']; allowed here: amp, batch_size, ...`; `lr: "fast"` -> `ValueError: .../bad2.yaml: config key 'optimizer.lr' must be float, got str ('fast')`
- Ignore rules: `git check-ignore data/x runs/x a.pt .env` -> prints all four.
- No machine paths: `grep -rnE "/home/|/content/|C:\\\\" src scripts configs` -> no matches.
- Requirements resolve as binary wheels for Python 3.10, 3.11 and 3.12 on manylinux x86_64 (`pip download --only-binary=:all: --python-version 3.1x ...`), and install in `.venv` on Python 3.12.16. Pillow 13.0.0 was dropped because it has no Python 3.10 wheel; pinned 12.3.0.
- Environment: Python 3.12.16; torch 2.14.1+cpu, torchvision 0.29.2+cpu (from `pip freeze`).

### Numbers produced
- none

### Changed from earlier phases
- none (first phase)

### Not done / deferred
- Colab's current torch/torchvision versions: UNKNOWN (not checked with the human). `requirements-local.txt` pins the latest stable pair available on the PyTorch CPU index today (torch 2.14.1 / torchvision 0.29.2; torch 2.15.0 exists but its torchvision is not yet published). **The human must compare these with Colab's `torch.__version__` in Phase 07** and adjust the local pins to the same major.minor (D9).
- The notebook's `BRANCH` is `main`, which holds only `docs/` until `phase-00-setup` is merged; until then set `BRANCH = "phase-00-setup"` on Colab.
- ResNet-18 epochs (D4) stay open until Phase 07.

### Known issues
- `pip install -r requirements.txt` on Colab may replace some preinstalled packages (for example numpy is pinned to 2.2.6, the last release with Python 3.10 wheels). If Colab's torch needs a newer numpy, the install will show a conflict; check in Phase 07 and loosen the pin if needed.
- `requirements-local.txt` uses `--extra-index-url https://download.pytorch.org/whl/cpu` so local installs get the small CPU build; on a CUDA machine install torch separately.

### Decisions used
- D4: SGD momentum 0.9 Nesterov, wd 5e-4, lr 0.1 cosine, batch 128, CNN 30 epochs, ResNet 40 epochs (proposed), seeds 0/1/2, cudnn deterministic, benchmark off.
- D5: AugMix severity 3, mixture_width 3, chain_depth -1, alpha 1.0, all_ops false; S3 p 0.5 over the 4 seen corruptions, severities 1-5.
- D6: seen/unseen lists and the three analysis groups.
- D1: val_size 5000, split_seed 42 (config values only).
- D9: torch not installed on Colab; local pins as above.

### Ideas for later
- none

## Phase 01: data-clean  (2026-11-03, branch phase-01-data-clean, commit: the one titled "phase 01: clean CIFAR-10 split, stats, loaders")

### Built
- `src/data/cifar10.py`: the C10 functions plus helpers.
  - `make_stratified_split(labels, val_size, split_seed)`: pure, 500 per class in val, uses only `np.random.default_rng(split_seed)`.
  - `load_or_create_split(path, labels_fn, ...)`: reads the JSON if present (raises `ValueError` without touching the file if `split_seed`/`val_size` differ), otherwise builds and writes it.
  - `get_split_indices(data_dir, val_size, split_seed)`: the above on `splits/cifar10_val_indices.json`.
  - `compute_normalization_stats(data_dir)`: per-channel mean and std (population std, ddof=0) over the 45,000 train-split images /255; writes `splits/cifar10_stats.json`. `get_normalization_stats()` reads it.
  - `get_train_val_datasets(cfg, train_transform=None, data_dir=None)`: `TransformedSubset`s over one CIFAR-10 train set; honours `cfg["subset"]` (first N of each split after the fixed split). `get_test_dataset(data_dir)`. Both check sizes are 50,000 / 10,000.
  - `make_loaders(cfg, ...)`: train shuffled with `make_generator(cfg["seed"])`, val not shuffled; `worker_init_fn=seed_worker`; `pin_memory` when the device is CUDA.
  - `split_class_counts`, `examples_per_class`: used by the notebook.
  - **Phase 03 hook:** `phase01_default_train_transform()` (= ToTensor + Normalize) is used when no `train_transform` is passed. Phase 03 should make `get_train_val_datasets` default to `build_train_transform(cfg["strategy"], ...)` instead.
- `src/data/transforms.py`: `build_eval_transform()` = ToTensor + Normalize(stats file). `build_train_transform` is not there yet (Phase 03).
- `splits/cifar10_val_indices.json`, `splits/cifar10_stats.json`: computed on the real data, committed.
- `tests/test_cifar10_split.py`: 8 tests on a fake 600-label array (no download): stratified, disjoint and complete, deterministic, unaffected by the global NumPy seed, non-divisible val_size raises, file round-trip without reloading labels, mismatch (seed or size) raises and leaves the file byte-identical.
- `notebooks/01_explore_data.ipynb`: 10 train-split images per class, class counts per split, the stats table; calls repo code only.

### How to run it
    .venv/bin/python -c "from src.data.cifar10 import compute_normalization_stats as c; print(c('./data'))"   # only if splits/*.json are missing
    .venv/bin/ruff format --check . && .venv/bin/ruff check . && .venv/bin/pytest -q
    MPLBACKEND=Agg .venv/bin/python -c "import nbformat; g={}; [exec(c.source, g) for c in nbformat.read('notebooks/01_explore_data.ipynb', 4).cells if c.cell_type == 'code']; print(g['counts'])"

### Verified (with output)
- Download: `torchvision.datasets.CIFAR10(root='./data', download=True)` -> `train 50000 (50000, 32, 32, 3)`, `test 10000 (10000, 32, 32, 3)`.
- Sizes: `get_split_indices` / `get_test_dataset` -> `sizes train/val/test: 45000 5000 10000`.
- Disjoint and complete: `disjoint: True | union == range(50000): True`.
- Per-class counts (`split_class_counts`): train `[4500]*10`, val `[500]*10`, test `[1000]*10`.
- Same split in a new process: sha256 of `[train, val]` JSON -> `d287d84919e6905f` in the check script and `d287d84919e6905f` in a separate `python -c`.
- Independent of training seed: `get_train_val_datasets` with `cnn_standard.yaml` seed 0 vs seed 2 -> `same train indices: True`.
- Eval transform deterministic: applied twice to one test image -> `True`; it is `Compose(ToTensor(), Normalize(mean=(0.4918..., 0.4825..., 0.4468...), std=(0.2470..., 0.2435..., 0.2616...)))`, i.e. the file's values.
- Subset: smoke config with `subset: 64` -> `64 64`, item shape `(3, 32, 32)`.
- Loader determinism: two `make_loaders(smoke cfg)` -> `same first batch labels: True`.
- Test set use: `grep -rn "train=False" src/` -> only `src/data/cifar10.py:244` in `get_test_dataset`.
- Notebook: all code cells executed headless; counts table totals `45000 5000 10000`; image array `(10, 10, 32, 32, 3)`.
- Lint/tests: `ruff format --check .` -> `65 files already formatted`; `ruff check .` -> `All checks passed!`; `time pytest -q` -> `43 passed in 3.02s` (real 0m4.027s, CPU).

### Numbers produced
- Normalisation mean (R, G, B): 0.49183993285335986, 0.4825411911934913, 0.44681958341843653 (source: `splits/cifar10_stats.json`) [real]
- Normalisation std (R, G, B): 0.2470507307231891, 0.24354212224302502, 0.2616922130832278 (source: `splits/cifar10_stats.json`) [real]
- For comparison only (not stored): the same over all 50,000 train images is mean `[0.4914, 0.482158, 0.446531]`, std `[0.247032, 0.243485, 0.261588]` (check script output), so the file differs from the 50k values in the 4th decimal, as expected for 45k-only stats.

### Changed from earlier phases
- `.gitignore`: `data/`, `runs/`, `dist/` -> `/data/`, `/runs/`, `/dist/` because the unanchored `data/` also ignored `src/data/`, so the `src/data/` package (including the Phase 00 stubs) had never been committed. A fresh clone of `main` has no `src/data/`. `git check-ignore` still ignores `data/x`, `runs/x`, `dist/x`.
- `src/utils.py`: added `seed_worker` (DataLoader `worker_init_fn`) and `make_generator(seed)` because BUILD asks for the loader's generator and worker init to come from utils. `tests/test_utils.py`: 2 tests for them.

### Not done / deferred
- Training transforms S0-S3 and `build_train_transform` -> Phase 03 (hook described above).
- `make_loaders` uses `drop_last=False` (BUILD does not specify it); revisit in the training phase if needed.

### Known issues
- The CIFAR-10 download from cs.toronto.edu ran at about 75 kB/s on this machine (~35 min). One-time cost; on Colab, `DATA_DIR=/content/data` needs it per session unless the tarball is copied from Drive.
- `docs/PROJECT_OVERVIEW.md` 6.1 says the validation split is "for model selection"; DECISIONS D3 (confirmed) says monitoring only, final-epoch model evaluated. D3 ranks higher; no effect on this phase's code.

### Decisions used
- D1: 45,000 / 5,000 stratified (500 per class), split_seed 42 independent of the training seed, stored in `splits/cifar10_val_indices.json`.
- D2: per-channel mean/std of the 45,000 training images, stored in `splits/cifar10_stats.json`, used for val/test (and later CIFAR-10-C).

### Ideas for later
- none
