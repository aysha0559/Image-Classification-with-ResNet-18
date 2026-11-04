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
- GitHub: a public repo (since replaced, see below) was created at the human's request; `main` and `phase-00-setup` pushed. Not merged (merge waits for validation).
- GitHub (changed 2026-11-04, at the human's request): the remote is now https://github.com/aysha0559/Image-Classification-with-ResNet-18; `REPO_URL` in `notebooks/00_colab_setup.ipynb` points there.

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

## Phase 02: data-corrupted  (2026-11-03, branch phase-02-data-corrupted, commit: the one titled "phase 02: CIFAR-10-C download, verify, loader, sample grid")

### Built
- Before branching, at the human's request: committed `validation/phase-01.md` (PASS) on `phase-01-data-clean` and merged that branch into `main` (`--no-ff`, not pushed). `phase-02-data-corrupted` branches from that merge.
- `scripts/download_cifar10c.py` (+ `scripts/download_cifar10c.sh` wrapper, passes arguments on; `PYTHON` env var picks the interpreter):
  - Reads the archive's name, size and md5 from `https://zenodo.org/api/records/2535967` at run time (nothing typed from memory).
  - Downloads to `DATA_DIR/CIFAR-10-C.tar` via `CIFAR-10-C.tar.part` with HTTP Range resume and a tqdm bar; skips the download if the archive already verifies, and skips everything if the extracted folder already matches the checksums JSON.
  - md5 check against Zenodo (prints both); on failure deletes and re-downloads once, a second failure stops with an error.
  - Extracts (only paths under `CIFAR-10-C/` allowed; tar `data` filter when available), checks exactly 20 `.npy` files with the expected shapes/dtypes, sha256s each file.
  - Writes `splits/cifar10c_checksums.json` if absent; if present, compares against it instead (so Colab checks against the committed hashes) and also checks Zenodo's metadata still matches the JSON's `archive` block.
  - `--verify-only`: no network; structure + sha256 against the JSON. Exit code 1 with `ERROR: ...` on any failure.
- `splits/cifar10c_checksums.json`: source, archive {name, size, checksum_type, checksum} and 20 file sha256s. Committed.
- `src/data/cifar10c.py`: docstring "Test-only. Never import from training code."; `validate_condition` (name in the 15 + 4 extras, severity an int in 1..5, else `ValueError`); `load_cifar10c(data_dir, corruption, severity)` memory-maps the file (`mmap_mode="r"`) and slices the severity's rows; labels returned as int64. `N_PER_SEVERITY = 10_000` is a module constant so tests can shrink it. `CIFAR10C(corruption, severity, transform=None, data_dir=None)` defaults to `build_eval_transform()` and `DATA_DIR`.
- `scripts/make_sample_grid.py`: rows = 15 corruptions (labelled with name and family), columns = clean + severity 1-5; PNG at 200 dpi + PDF. Default `--index 23` (a red fire truck, chosen by eye from test images 0-39 as large, centred and high-contrast). Also checks the CIFAR-10-C label of that image equals the clean label for every condition.
- `results/figures/fig7_sample_grid.png` and `.pdf`.
- `notebooks/00_colab_setup.ipynb`: new cell. If `MyDrive/cifar10-robustness/data/CIFAR-10-C` exists: copy to `/content/data`, run `--verify-only`. Otherwise: run the download script into `/content/data` (raises on failure), copy the verified folder to Drive, delete the local archive.
- `notebooks/01_explore_data.ipynb`: section 2 shows the saved grid PNG (matplotlib only, no IPython dependency).
- `tests/test_cifar10c.py` (22 tests, fake 5-images-per-severity folder): rows per severity, label dtype int64, bad names, severities 0/6/-1/2.0/"3"/True, wrong row count, missing file, Dataset default transform normalises, custom transform.
- `tests/test_no_leakage.py`: (a) `src/train.py`, `src/data/transforms.py`, `src/data/corruptions.py`, `src/models/*.py` never contain `cifar10c` / `cifar-10-c` (case-insensitive); (b) AST check that no module under `src/` or `scripts/` except `src/evaluate.py` and `scripts/make_sample_grid.py` imports `cifar10c`.

### How to run it
    .venv/bin/python -m scripts.download_cifar10c                 # or: bash scripts/download_cifar10c.sh [--data-dir DIR]
    .venv/bin/python -m scripts.download_cifar10c --verify-only
    .venv/bin/python -m scripts.make_sample_grid                  # [--index 23] [--out results/figures/fig7_sample_grid.png]
    .venv/bin/ruff format --check . && .venv/bin/ruff check . && .venv/bin/pytest -q

### Verified (with output)
- Zenodo metadata (`curl https://zenodo.org/api/records/2535967`): `CIFAR-10-C.tar 2918471680 md5:56bf5dcef84df0e2308c6dcbcbbd8499`.
- Download path: the script resumed a partial `.part` file over HTTP Range (`CIFAR-10-C.tar: 2%| 59.1M/2.92G` at start, server honoured the range). At about 400 kB/s this would take ~2 h, so **the human downloaded the same URL with a download manager** and the archive was moved into `data/`. The script then verified it like any other archive.
- Archive checksum, first full run: `archive md5: zenodo 56bf5dcef84df0e2308c6dcbcbbd8499  local 56bf5dcef84df0e2308c6dcbcbbd8499` -> `archive checksum matches Zenodo`; `extracting 21 entries to data/CIFAR-10-C` (folder + 20 files); `structure ok: 20 files, corruptions uint8 (50000, 32, 32, 3)`; `wrote splits/cifar10c_checksums.json (20 file hashes)` (real 0m17s).
- Idempotent: second run -> `all 20 files match splits/cifar10c_checksums.json` / `extracted files already verified; nothing to download`.
- `--verify-only` (also via the `.sh` wrapper) -> `all 20 files match ...`.
- Tamper: scratch folder with symlinks to the 19 real files + a copy of `fog.npy` with one bit flipped at byte 1000 -> `ERROR: sha256 mismatch against cifar10c_checksums.json: ['fog.npy']`, exit 1.
- Inventory (scratch script, `np.load(mmap_mode="r")`): 19 corruption files uint8 (50000, 32, 32, 3); `labels uint8 (50000,)`.
- Label alignment: `labels[s*10000:(s+1)*10000] == CIFAR10(train=False).targets` -> `[True, True, True, True, True]`; also equal for every (corruption, severity) loaded through `load_cifar10c`.
- `load_cifar10c("./data", "fog", 3)` -> `(10000, 32, 32, 3) uint8 (10000,) int64`.
- Severity ordering, mean |corrupted - clean| over the 10,000 images (pixel units 0-255), s1..s5:
  gaussian_noise 7.99 11.90 15.74 17.63 19.51 | shot_noise 5.91 8.31 13.03 14.98 18.21 | impulse_noise 1.28 2.55 3.82 6.38 8.92 | defocus_blur 1.38 3.32 5.01 6.53 9.32 | **glass_blur 11.17 11.31 11.12 18.18 17.44** | **motion_blur 6.97 10.43 13.10 13.08 15.29** | zoom_blur 8.81 10.38 12.41 14.50 16.50 | snow 10.70 23.75 25.38 37.15 51.80 | **frost 30.42 44.75 49.42 44.82 43.08** | fog 10.80 21.31 27.61 32.42 39.07 | brightness 10.02 20.28 30.09 39.37 56.00 | contrast 10.65 21.30 25.56 29.82 36.21 | **elastic_transform 21.87 20.00 17.95 16.54 13.67** | pixelate 1.79 3.40 4.14 6.24 8.66 | jpeg_compression 5.40 6.65 7.07 7.53 8.14 -> strictly increasing for **11/15**.
- Sample grid opened and inspected: title, column headers clean / severity 1-5, 15 labelled rows. Noise, defocus/zoom blur, snow, fog, brightness, contrast, pixelate, jpeg visibly get stronger left to right. glass_blur s1-s3 look alike; frost s5 shows more frost texture but less blue tint than s3-s4; elastic_transform s1 looks the most distorted and later severities look milder (matches the decreasing pixel difference).
- Leakage guard: in a scratch copy of the repo, appending `from src.data.cifar10c import load_cifar10c` to `src/train.py` and `import src.data.cifar10c` to `src/analysis.py` -> both tests fail (`CIFAR-10-C referenced in training code: ['src/train.py: cifar10c']`; `... may import cifar10c: ['src/analysis.py', 'src/train.py']`). In the repo both pass.
- Notebook 01 executed headless (code cells via nbformat): counts totals `[45000, 5000, 10000]`, grid image `(3870, 1820, 4)`. Notebook 00 cells are Colab-only (Drive mount); not executed here.
- Lint/tests: `ruff format --check .` -> `68 files already formatted`; `ruff check .` -> `All checks passed!`; `time pytest -q` -> `67 passed in 4.25s` (real 0m5.4s, CPU).

### Numbers produced
- Archive: CIFAR-10-C.tar, 2918471680 bytes, md5 56bf5dcef84df0e2308c6dcbcbbd8499 (source: Zenodo API, stored in `splits/cifar10c_checksums.json`) [real]
- Per-file sha256: `splits/cifar10c_checksums.json` [real]
- Per-severity mean pixel differences above (source: scratch check script output, not stored in the repo) [real, descriptive only]

### Changed from earlier phases
- `main`: phase 01 merged (see Built), at the human's request.
- `notebooks/00_colab_setup.ipynb`, `notebooks/01_explore_data.ipynb`: cells added (in scope for this phase).

### Not done / deferred
- Evaluation over CIFAR-10-C -> Phase 06.
- `--include-extra` handling -> Phase 06 (the loader already accepts the 4 extra names).

### Known issues
- **VALIDATE V8 originally expected mean pixel difference to grow from severity 1 to 5 for at least 13 of 15 corruptions; the real (checksum-verified) data gives 11/15. The human chose to lower the threshold to 11/15 (VALIDATE.md updated).** The exceptions are glass_blur, motion_blur (s3 13.10 vs s4 13.08), frost and elastic_transform. This is the published data, not the slicing: the 11 others increase cleanly, the labels align for every severity, and the grid shows the same pattern. Pixel difference is a weak severity proxy for frost (the image is blended with a frost texture at shifting weights) and elastic_transform (CIFAR-10-C's parameters for it do not make the pixel change monotonic).
- The local venv has no IPython, so notebook 01 shows the grid with matplotlib rather than `IPython.display`.
- On free Drive (15 GB), the extracted CIFAR-10-C takes about 2.9 GB; the Colab cell does not keep the archive on Drive.

### Decisions used
- D7: 15 standard corruptions x 5 severities in the grid; the 4 extras are accepted by the loader but not in the grid or the defaults.
- RULES section 3: CIFAR-10-C only through `src/data/cifar10c.py`; imported only by `scripts/make_sample_grid.py` so far (and `src/evaluate.py` later), guarded by `tests/test_no_leakage.py`.

### Ideas for later
- none

## Phase 03: train-transforms  (2026-11-04, branch phase-03-train-transforms, commit: the one titled "phase 03: training transforms S0-S3 and seen corruptions")

### Built
- `src/data/corruptions.py`: `apply_corruption(img, name, severity, rng)` for the 4 seen corruptions only (any other name raises `ValueError` listing the allowed ones; bad severity or non-uint8-HWC input also raises). `RandomSeenCorruption(p, corruptions, severities)` on PIL images. Its docstring cites `ImageNet-C/create_c/make_cifar_c.py` at commit `ca360085d57889288c4143fddd8a707bbcae01bd` of `github.com/hendrycks/robustness` (read in this session) and lists the parameters per severity.
- The generator is seeded from `torch.initial_seed()` and rebuilt when that seed changes, so workers get separate streams even if the object was used in the parent first. Names outside the seen list are rejected when the object is built, not at the first image. `cv2.setNumThreads(0)` is set at import, as a precaution for forked DataLoader workers.
- `src/data/transforms.py`: `build_train_transform(strategy, params)` for `none`, `standard`, `augmix`, `corruption_aware`; `params` is the config's `strategy_params` block. Unknown strategy raises `ValueError`.
- `src/data/cifar10.py`: `get_train_val_datasets` now defaults to `build_train_transform(cfg["strategy"], cfg["strategy_params"])`.
- `scripts/make_sample_grid.py --ours`: rows alternate ours / CIFAR-10-C for the 4 seen corruptions, columns clean + severities 1-5. Writes `results/figures/extra_our_corruptions.png`.
- `tests/test_corruptions.py`, `tests/test_transforms.py`: synthetic images only.

### How to run it
    .venv/bin/python -m scripts.make_sample_grid --ours            # [--index 23] [--out PATH]
    .venv/bin/ruff format --check . && .venv/bin/ruff check . && .venv/bin/pytest -q
    .venv/bin/python -c "from src.config import load_config as l; from src.data.transforms import build_train_transform as b; c = l('configs/cnn_augmix.yaml', seed=0); print(b(c['strategy'], c['strategy_params']))"

### Verified (with output)
- Lint/tests: `ruff format --check .` -> `71 files already formatted`; `ruff check .` -> `All checks passed!`; `time pytest -q` -> `150 passed in 5.22s` (real 0m6.46s, CPU).
- Leakage: `grep -rniE "cifar10c|CIFAR-10-C|\.npy" src/data/transforms.py src/data/corruptions.py src/data/cifar10.py src/train.py src/models/` -> no matches (exit 1). `python -c "import sys, src.data.transforms, src.data.corruptions; print([m for m in sys.modules if 'cifar10c' in m])"` -> `[]`.
- Installed torchvision 0.29.2 supports `all_ops` (`inspect.signature(AugMix.__init__)`).
- Pipelines (printed): augmix -> `RandomCrop(size=(32, 32), padding=4)`, `RandomHorizontalFlip(p=0.5)`, `AugMix(severity=3, mixture_width=3, chain_depth=-1, alpha=1.0, all_ops=False, interpolation=InterpolationMode.BILINEAR, fill=None)`, `ToTensor()`, `Normalize(...)`; corruption_aware -> crop, flip, `RandomSeenCorruption(p=0.5, corruptions=['gaussian_noise', 'defocus_blur', 'contrast', 'jpeg_compression'], severities=[1, 2, 3, 4, 5])`, `ToTensor()`, `Normalize(...)`. Normalize holds the values of `splits/cifar10_stats.json`.
- `apply_corruption(img, "fog", 1, rng)` -> `unsupported corruption 'fog'; allowed: gaussian_noise, defocus_blur, contrast, jpeg_compression`.
- Wiring on the real data, `configs/cnn_corruption_aware.yaml`: train `['RandomCrop', 'RandomHorizontalFlip', 'RandomSeenCorruption', 'ToTensor', 'Normalize']`, val `['ToTensor', 'Normalize']`, item `torch.Size([3, 32, 32]) torch.float32`.
- Real `make_loaders` with `num_workers=2`, S2 and S3 configs -> `torch.Size([128, 3, 32, 32])`, no hang.
- Severity order, mean abs pixel difference on 200 training images, s1..s5: gaussian_noise 8.01 11.90 15.76 17.66 19.56 | defocus_blur 1.42 3.45 5.21 6.78 9.69 | contrast 10.63 21.26 25.51 29.76 36.14 | jpeg_compression 5.39 6.72 7.16 7.63 8.25 -> strictly increasing for all 4.
- `p` on the first 2,000 training-split images, fraction changed: `p=0` -> 0.0, `p=0.5` -> 0.503, `p=1` -> 1.0.
- `make_sample_grid` default mode unchanged: `cmp` of a new fig7 PNG with the committed one -> same.
- The tests catch broken code (scratch copy, one change at a time): removing the seed-change check -> 2 failed; `p` ignored -> 2 failed; constant seed -> 4 failed; every severity using s1 noise -> 1 failed.
- Comparison grid opened and inspected. gaussian_noise: same kind and strength; the pixels differ because the noise is random. defocus_blur and contrast: look the same as CIFAR-10-C at every severity. jpeg_compression: same blockiness and colour bleeding with the same growth, but the CIFAR-10-C row has visibly stronger colour artefacts on this image, most at severity 1 (see Known issues).

### Numbers produced
Mean abs pixel difference (0-255), severities 1..5 [real]. Ours: first 1,000 images of `train_indices`. CIFAR-10-C: first 1,000 test images. Source: scratch script output, not stored in the repo; `default_rng(0)` for ours.

| Corruption | Ours (train) | CIFAR-10-C (test) |
|---|---|---|
| gaussian_noise | 7.99 11.89 15.74 17.64 19.52 | 8.00 11.91 15.75 17.64 19.52 |
| defocus_blur | 1.41 3.41 5.14 6.70 9.57 | 1.38 3.32 5.00 6.51 9.28 |
| contrast | 10.73 21.46 25.76 30.05 36.49 | 10.68 21.35 25.62 29.89 36.30 |
| jpeg_compression | 5.38 6.66 7.10 7.56 8.18 | 5.42 6.68 7.11 7.57 8.18 |

Fidelity check (scratch script, not asked for by BUILD): ours applied to the same 1,000 test images and compared pixel by pixel with CIFAR-10-C [real]. No parameter was chosen or changed from this.

| Corruption | Result across severities 1-5 |
|---|---|
| defocus_blur | identical images 1.0000 at every severity |
| contrast | identical images 0.9950, max difference 1 level |
| jpeg_compression | identical images 0.014-0.019, identical values 0.52-0.67, max difference 82-99 levels; mean strength ours 5.32 6.59 7.02 7.48 8.09 vs 5.42 6.68 7.11 7.57 8.18 |

Time for 1,000 training images, whole pipeline including ToTensor + Normalize, 1 CPU thread, median of 3, this machine (not Colab) [real]:

| Strategy | s / 1,000 images | ms / image |
|---|---|---|
| none | 0.137 | 0.137 |
| standard | 0.222 | 0.222 |
| augmix | 1.789 | 1.789 |
| corruption_aware | 0.330 | 0.330 |

`apply_corruption` alone at severity 3, s / 1,000 images: gaussian_noise 0.091, defocus_blur 0.286, contrast 0.042, jpeg_compression 0.288.

### Changed from earlier phases
- `src/data/cifar10.py`: removed `phase01_default_train_transform` and switched the default to `build_train_transform`, as the Phase 01 hook asked.
- `scripts/make_sample_grid.py`: added `--ours`; `--out` now defaults by mode (in scope for this phase).

### Not done / deferred
- Models, training loop, JSD loss: out of scope.
- `results/figures/extra_our_corruptions.png` is PNG only (BUILD names only the PNG).

### Known issues
- **Our jpeg_compression does not reproduce CIFAR-10-C pixel for pixel, although the code path matches the reference.** Mean strength agrees to about 0.1 pixel levels. On 1,000 test images at severity 1, Pillow's default (4:2:0) is the closest setting tried: mean difference 1.36, against 3.56 for 4:4:4 and 3.02 for 4:2:2; `optimize` and `progressive` change nothing. The gap is mostly colour: mean luma difference 0.18, mean RGB difference 1.36. Likely cause: different JPEG library versions (here Pillow uses libjpeg-turbo, `jpg` 6.2) handling chroma differently; this is NOT confirmed. Impact: S3's JPEG training images have somewhat milder colour artefacts than the benchmark's on some images, which could slightly understate S3 on `jpeg_compression`. Not tuned, because RULES section 3 forbids choosing parameters from CIFAR-10-C. The human accepted this (DECISIONS change log, 2026-11-04); Phase 10 discloses it.
- Contrast: 5 of the 1,000 test images differ from CIFAR-10-C by 1 level at most. Cause not investigated; too small to matter.
- Augmentation randomness runs in the DataLoader workers for every strategy with augmentation (crop, flip, AugMix, S3), not only S3, so it cannot go into `checkpoint_last.pt`'s `rng_states` from the main process. `RandomSeenCorruption`'s generator is also not checkpointed with `num_workers: 0`. The human accepted this (DECISIONS change log, 2026-11-04); Phase 05 BUILD now requires that a resumed run does not replay the first session's stream.
- `src/config.py` accepts any of the 15 standard names in `corruption_aware.corruptions`. A non-seen name fails when `RandomSeenCorruption` is built, not when the config loads.
- D5 does not state AugMix's interpolation; the torchvision default (bilinear) is used.

### Decisions used
- D5: S0-S3 pipelines; AugMix severity 3, mixture_width 3, chain_depth -1, alpha 1.0, all_ops False, no JSD; S3 p 0.5, uniform corruption and severity 1-5.
- D6: seen list gaussian_noise, defocus_blur, contrast, jpeg_compression; formulas and severity parameters from the Hendrycks & Dietterich reference code.
- D2: Normalize from `splits/cifar10_stats.json`.

### Ideas for later
- none

## Phase 04: models  (2026-11-04, branch phase-04-models, commit: the one titled "phase 04: simple CNN and CIFAR ResNet-18")

### Built
- `src/models/simple_cnn.py`: `SimpleCNN(num_classes=10)`, the layer list of overview 7.1; padding and dropout from D11 are named constants and stated in the docstring.
- `src/models/resnet.py`: `BasicBlock` and `ResNet18(num_classes=10)`, written by hand, no torchvision model. 3x3 stride-1 stem, no max-pool, blocks [2, 2, 2, 2], widths 64/128/256/512, stride 2 at stages 2-4 with a 1x1 projection shortcut, global average pool, Linear(512 -> 10). Kaiming normal on convs, BatchNorm weight 1 / bias 0.
- `src/models/__init__.py`: `build_model(name, num_classes=10)`; an unknown name raises `ValueError` listing the allowed names.
- `src/utils.py`: `count_parameters(model)` (trainable parameters).
- `tests/test_models.py`: 17 test cases: shapes, finite outputs, gradients on every parameter, `num_classes`, unknown name, parameter ranges, same seed same init, CNN layer order, D11 values, ResNet stem, stages, shortcuts and init.

### How to run it
    .venv/bin/ruff format --check . && .venv/bin/ruff check . && .venv/bin/pytest -q
    .venv/bin/python -c "from src.models import build_model as b; from src.utils import count_parameters as c; [print(b(n), n, c(b(n))) for n in ('simple_cnn', 'resnet18')]"

### Verified (with output)
- Lint/tests: `ruff format --check .` -> `73 files already formatted`; `ruff check .` -> `All checks passed!`; `time pytest -q` -> `167 passed in 5.55s` (real 0m6.810s, CPU).
- Simple CNN `print(model)`: `Conv2d(3, 32)`, BN, ReLU, `Conv2d(32, 32)`, BN, ReLU, `MaxPool2d(2)`, `Dropout(0.25)`; `Conv2d(32, 64)`, BN, ReLU, `Conv2d(64, 64)`, BN, ReLU, `MaxPool2d(2)`, `Dropout(0.25)`; `Conv2d(64, 128)`, BN, ReLU, `MaxPool2d(2)`, `Dropout(0.25)`; `Flatten`, `Linear(2048, 256)`, ReLU, `Dropout(0.5)`, `Linear(256, 10)`. Every conv is 3x3 with `padding=(1, 1)`. This matches overview 7.1 and D11 layer by layer.
- ResNet `print(model)`: stem `Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))`, no MaxPool in the module; 4 stages of 2 `BasicBlock`s; the first block of stages 2-4 has `stride=(2, 2)` and a `Conv2d(.., kernel_size=(1, 1), stride=(2, 2))` + BN shortcut; `AdaptiveAvgPool2d(1)`; `Linear(512, 10)`.
- No torchvision model: `grep -rn "torchvision.models\|from torchvision" src/models/` -> no matches.
- Overfit sanity (scratch script, the same recipe as VALIDATE V10): seeded 64 real training-split images, eval transform, Adam lr 1e-3, CPU. First step at 100% on the batch, train mode / eval mode: simple_cnn 32 / 40, resnet18 4 / 32. Both within 200 steps. [smoke]
- The tests catch broken code (scratch copy, one change at a time, `pytest -q tests/test_models.py`): block dropout 0.3 -> 1 failed; conv padding 0 -> 4 failed; ResNet blocks (3, 2, 2, 2) -> 5 failed; stem stride 2 -> 5 failed; Kaiming `fan_in` instead of `fan_out` -> 4 failed.

### Numbers produced
- simple_cnn parameters: 666,858 (source: `count_parameters`, same as `sum(p.numel() ...)`) [real]. Inside 0.5-1.0 M; D11's estimate was about 0.67 M.
- resnet18 parameters: 11,173,962 (same source) [real]
- One forward + backward on a batch of 128, random data, CPU, 4 torch threads, this machine, not Colab, torch 2.14.1+cpu; median of 5 after discarding the first (scratch script output, not stored) [smoke]:

| Model | Run 1 (s) | Run 2 (s) |
|---|---|---|
| simple_cnn | not captured (output cut off) | 0.323 |
| resnet18 | 2.711 | 4.330 |

- ResNet-18 varied by about 60% between the two runs, so treat these as rough. They do not predict Colab time; Phase 07 measures that.
- GPU timing: UNKNOWN; this machine has no CUDA (`torch.cuda.is_available()` -> False). Phase 07 measures it on Colab.

### Changed from earlier phases
- `src/utils.py`: added `count_parameters`, because BUILD asks for it there. Module docstring updated.

### Not done / deferred
- Training loop -> Phase 05. `n_params` in `run_meta.json` should use `count_parameters`.

### Known issues
- D11's "Expected parameters" row still reads "to be measured in Phase 04"; the measured count is recorded here and `DECISIONS.md` was left unchanged.

### Decisions used
- D11: padding 1 on every 3x3 conv; dropout 0.25 after each conv block; dropout 0.5 before the last Linear.
- Choices the sources leave open, none affects the layer lists: convs that feed BatchNorm have no bias in both models; `nn.Dropout`, not `Dropout2d`, in the CNN; Kaiming uses `fan_out` with relu; the last BatchNorm in each block is not zero-initialised; Linear layers and all of `SimpleCNN` use PyTorch's default init (BUILD asks for Kaiming only on the ResNet).

### Ideas for later
- none

## Phase 05: training-loop  (2026-11-04, branch phase-05-training-loop, commit: the one titled "phase 05: resumable config-driven training loop")

### Built
- `src/train.py`: `python -m src.train` per C5; `train(cfg, resume=True)` is the callable form. SGD from config, `CosineAnnealingLR(T_max=epochs)` stepped per epoch, cross-entropy, AMP only on CUDA with `amp: true`. Writes the C6 run folder. `lr` in `train_log.csv` is the rate used during that epoch. `epoch_time_s` covers training + validation, not the checkpoint write.
- Checkpoint after every epoch via `atomic_save`, with RNG states for Python, NumPy, torch CPU, torch CUDA (if available) and the DataLoader generator. It loads on the CPU, because RNG states must stay there.
- Resume: restores model, optimizer, scheduler, scaler, then the RNG states, before the first iterator exists. A changed training key is refused before anything in the run folder changes; the message names the key (nested keys as `optimizer.lr`). `run_id`, `device` and `num_workers` may change between sessions (`NON_TRAINING_KEYS`).
- Log rows after the checkpointed epoch are dropped on resume: a row is written before its checkpoint, so a crash between them would otherwise repeat an epoch. Rows are fsynced.
- `STOP_AFTER_EPOCHS`, documented in the module docstring: an absolute epoch number; exit 0, `status` stays `running`.
- `train_time_s` is the sum of `epoch_time_s` over the log, so it counts every session. A resumed run keeps its first `started_at`.
- Divergence: the loss is checked on every batch; NaN or inf raises `DivergenceError`, `status: failed`.
- A finished run is skipped, also with `--no-resume`. `--no-resume` on an unfinished run deletes its checkpoint and empties its log.
- `src/utils.py`: `get_rng_states`, `set_rng_states`.
- `scripts/registry.py`: `build_registry`; writes `$RESULTS_DIR/runs.csv` (C8), sorted by run id. Reads results from `raw/`, or `smoke/` for `smoke-` runs.
- `tests/test_train_smoke.py`: 21 test cases on fake data, no download.

### How to run it
    .venv/bin/python -m src.train --config configs/smoke.yaml --seed 0
    STOP_AFTER_EPOCHS=2 .venv/bin/python -m src.train --config configs/smoke.yaml --epochs 3 --run-id smoke-b
    .venv/bin/python -m src.train --config configs/smoke.yaml --epochs 3 --run-id smoke-b    # resumes
    .venv/bin/python -m scripts.registry
    .venv/bin/ruff format --check . && .venv/bin/ruff check . && .venv/bin/pytest -q

All checks below used scratch `RUNS_DIR` and `RESULTS_DIR` folders outside the repo.

### Verified (with output)
All on CPU, real CIFAR-10 unless stated. All numbers are smoke numbers.
- Lint/tests: `ruff format --check .` -> `75 files already formatted`; `ruff check .` -> `All checks passed!`; `time pytest -q` -> `190 passed in 19.21s` (real 0m20.437s, CPU).
- Step 1, `python -m src.train --config configs/smoke.yaml --seed 0` -> `epoch 1/1 lr 0.10000 train_loss 2.6624 train_acc 0.1113 val_loss 2.3188 val_acc 0.1172 time 1.4s`, exit 0. Folder: `checkpoint_last.pt config_resolved.yaml model_final.pt run_meta.json train_log.csv`; log at `$RESULTS_DIR/smoke/smoke-cnn-standard-s0_train_log.csv`. `run_meta.json` holds exactly the 16 C6 keys (also asserted in tests).
- Finished run skipped: same command -> `already finished, nothing to do`, exit 0.
- Step 2, resume equivalence, `--epochs 3`, runs `smoke-a` and `smoke-b` (`STOP_AFTER_EPOCHS=2`, then resumed). First session: exit 0, `"status": "running"`, no `model_final.pt`. Second session: `resuming after epoch 2/3`, then only `epoch 3/3`. Max absolute log difference: lr 0.0, train_loss 0.0, train_acc 0.0, val_loss 0.0, val_acc 0.0. Weights: `allclose True identical True` over 34 tensors.

| Epoch | lr A | lr B | train_loss A | train_loss B | val_loss A | val_loss B | val_acc A | val_acc B |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.1 | 0.1 | 2.6623692512512207 | 2.6623692512512207 | 2.318750739097595 | 2.318750739097595 | 0.1171875 | 0.1171875 |
| 2 | 0.07500000000000001 | 0.07500000000000001 | 2.3245916962623596 | 2.3245916962623596 | 2.2590643167495728 | 2.2590643167495728 | 0.177734375 | 0.177734375 |
| 3 | 0.025000000000000015 | 0.025000000000000015 | 2.154705047607422 | 2.154705047607422 | 2.190445840358734 | 2.190445840358734 | 0.203125 | 0.203125 |

- Resume does not replay: `cnn_corruption_aware.yaml`, `num_workers: 2`, `--subset 256 --epochs 3`, traced by a scratch wrapper around `src.train.main` (`train_log.csv` columns unchanged).

| Session | base seed | epoch | shuffle sha256 prefix | first 8 indices |
|---|---|---|---|---|
| 1 | 900450186894289455 | 1 | fc5d2c78dc09 | 117 136 193 28 227 75 253 205 |
| 1 | same iterator | 2 | e23b0780d963 | 94 144 250 40 248 126 96 202 |
| 2 (resumed) | 6428247394855981926 | 3 | f8d5a7b54eed | 20 49 50 11 252 191 239 13 |

  Control: with the generator restore disabled in a scratch copy, session 2 replays session 1 exactly (base seed 900450186894289455, shuffle fc5d2c78dc09), so this check can fail.
- Refusal: `STOP_AFTER_EPOCHS=1 ... --epochs 3 --run-id smoke-c`, then `--epochs 5` -> `ValueError: refusing to resume smoke-c: config differs from the checkpoint in epochs: 3 -> 5. Restore the old value or use another --run-id.`, exit 1, `status` still `running`.
- Step 3: all 8 (model, strategy) combinations exit 0 and end `finished` (table below).
- NaN guard: `lr: 1000000.0` in a scratch config -> `DivergenceError: training loss is nan at batch 4`, `"status": "failed"`.
- Failure status: unusable `DATA_DIR` -> `PermissionError`, `"status": "failed"` with the message in `error`.
- Smoke isolation: `ls results/raw | wc -l` -> `0`.
- Registry: `python -m scripts.registry` -> `wrote .../runs.csv (4 runs)`; header is the 16 meta columns + `has_results,clean_accuracy`.
- Shortened run: `python -m src.train --config configs/cnn_standard.yaml --seed 0 --epochs 1`. Before the C5 change -> `cnn-standard-s0`, log in `raw/`. After it -> `smoke-cnn-standard-s0`, log in `smoke/`, no `raw/` folder. `load_config` still gives `cnn-standard-s0` with no override, and 30 or 31 epochs keep that id (tests).
- Same run id, other settings: finished `smoke-x`, then `--subset 128 --run-id smoke-x` -> `WARNING: smoke-x was NOT retrained`, `subset: 512 -> 128`, exit 0.
- `error` cleared: failed `smoke-y` has 1 `"error"` line in `run_meta.json`; after a successful re-run it has 0.
- Leakage: `grep -nE "train=False|get_test_dataset|cifar10c" src/train.py` -> no matches (exit 1); importing `src.train` loads no `cifar10c` module -> `[]`.
- The tests catch broken code (scratch copy, one change at a time, `pytest -q tests/test_train_smoke.py`): no RNG restore -> 1 failed; generator not restored -> 1 failed; optimizer not restored -> 1 failed; scheduler not restored -> 1 failed; log not trimmed -> 2 failed; smoke logs sent to raw -> 3 failed; no NaN guard -> 1 failed; no `scheduler.step()` -> 1 failed; failed status not set -> 2 failed. The scheduler change first passed all tests, so the resume test now runs 4 epochs with the stop at 2.

### Numbers produced
Time per epoch, `--subset 256 --epochs 1`, 256 training + 256 validation images, `num_workers: 2`, CPU, this machine, seed 0, 1 run each (source: scratch `run_meta.json` `train_time_s`) [smoke]. These do not predict Colab time; Phase 07 measures that.

| Strategy | simple_cnn (s) | resnet18 (s) |
|---|---|---|
| none | 0.88 | 10.26 |
| standard | 0.93 | 10.99 |
| augmix | 1.01 | 11.16 |
| corruption_aware | 0.93 | 10.97 |

Run folder size, resnet18 (`du -sh`) [smoke]: `checkpoint_last.pt` 86M, `model_final.pt` 43M.

### Changed from earlier phases
- `src/utils.py`: added `get_rng_states` and `set_rng_states`, because C1 places checkpoint io there. Module docstring updated.
- `docs/agent-prompts/CONTRACTS.md` (human approved, 2026-11-04): C6 adds `error` to `run_meta.json`, failed runs only, because BUILD asks for the error message and C6 had no key for it. C5 makes `--epochs` below the config's value a smoke run, because a shortened run with a real run id would count as finished and a later full run would be skipped.
- `src/config.py`: `load_config` applies that C5 rule. `tests/test_config.py`: 2 new tests.
- `docs/agent-prompts/phase-05-training-loop/VALIDATE.md` (human approved): V5 uses its own `RUNS_DIR` and checks that each run really trained.
- `scripts/registry.py`: was a Phase 08 stub docstring; filled now because this BUILD asks for it.

### Not done / deferred
- CUDA paths never ran: this machine has no CUDA. AMP, `GradScaler`, CUDA RNG restore and `device_name` on a GPU are unverified -> check in Phase 07.
- Real training -> Phase 07. Multi-run orchestration -> Phase 08.
- From `validation/phase-05.md`, to do at the start of Phase 07 (agreed with the human before the merge):
  - `src/train.py` `scaler.load_state_dict(checkpoint["scaler"])`: skip an empty scaler state, and add a test. A checkpoint saved on CPU has one, and loading it into an enabled scaler raises `RuntimeError`, so a run started without a GPU would probably fail on every resume on a GPU.
  - Confirm `device_name` in `run_meta.json` shows a GPU before a real run continues; `device: auto` falls back to the CPU silently.
  - CONTRACTS C8: say that `error` is not a registry column.
  - `train_time_s` counts only finished epochs, so do not read it as total GPU time for an interrupted run.

### Known issues
- **A smoke run can share its run id with an earlier one.** `configs/smoke.yaml` and `configs/cnn_standard.yaml --subset N` both give `smoke-cnn-standard-s0` at seed 0. If the first is finished, the second exits 0 without training, as BUILD requires, and prints `WARNING: ... was NOT retrained`. VALIDATE V5 now uses its own `RUNS_DIR` for this reason.
- An unfinished run with different settings is refused (exit 1); only finished runs warn and exit 0.
- With `num_workers > 0`, a resumed epoch's shuffle differs from the uninterrupted run's (`f8d5a7b54eed` vs `5b2957f464c8` for epoch 3), because the new session draws a base seed from the generator first. This is within the human's accepted difference (DECISIONS change log, 2026-11-04). A pilot or real run that gets interrupted must be noted in the run log.
- A resume across commits prints a warning, and `run_meta.json` records only the last session's `git_commit`.
- A process killed during `atomic_save` leaves a hidden `.checkpoint_last.pt.*.tmp` of up to 86M for ResNet-18; nothing removes it. Delete it by hand if Drive space runs short.
- A row missing from `train_log.csv` for a checkpointed epoch (possible if Drive syncs the files out of order) prints a warning and training continues.

### Decisions used
- D3: final-epoch model saved; no best-checkpoint logic; validation is monitoring only.
- D4: SGD momentum 0.9, Nesterov, weight decay 5e-4, lr 0.1, cosine to 0 over all epochs, no warm-up, AMP on CUDA, batch size from config, seeding through `set_seed`.
- DECISIONS change log 2026-11-04: resumed runs need not match bit for bit with `num_workers > 0` or `corruption_aware`, but must not replay.

### Ideas for later
- none

## Phase 06: evaluation  (2026-11-04, branch phase-06-evaluation, commit: the one titled "phase 06: metrics and CIFAR-10-C evaluator")

### Built
- `src/metrics.py`: the C10 functions as pure functions on arrays and DataFrames, all in fractions (mCE 1.0 = same error as the baseline). `macro_f1` is written by hand, so the scikit-learn comparison is a real cross-check.
  - The table functions read the C7 columns `corruption`, `severity`, `accuracy`. `df_run` is one run; `df_baseline` is already the seed-mean table.
  - A missing condition, a repeated condition (for example several seeds passed at once), a missing column, an accuracy that is NaN or outside [0, 1], or a zero baseline denominator raises `ValueError`. `clean` and extra rows are ignored by the CE sums, so extras cannot enter mCE (D7). `relative_mce` also needs the clean row.
  - Also there: `confusion_matrix(y_true, y_pred, num_classes=10)`, used by the evaluator.
- `src/evaluate.py`: `python -m src.evaluate` per C5; `evaluate(run_dir, subset, include_extra, batch_size, force)` is the callable form.
  - Refuses, before any evaluation: a run that is not `finished`, a folder with no metadata, a `model_final.pt` from another run, and existing results (either file) without `--force`.
  - Writes all three C7 files at the end of the pass. The confusion file and log copy come first and the CSV last, each CSV and npz via a temp file and rename, so a CSV means the evaluation finished.
  - `--subset N` adds `smoke-` once; any `smoke-` run id goes to `$RESULTS_DIR/smoke/`, with or without `--subset`.
  - The DataLoader uses the run's `num_workers` from its stored config. Device and AMP also come from there (AMP only on CUDA).
- `tests/test_metrics.py`, `tests/test_evaluate_smoke.py`: 74 test cases on hand-computed tables, a tiny fake model and a fake CIFAR-10-C folder. No download.

### How to run it
    .venv/bin/python -m src.train --config configs/smoke.yaml --seed 0
    .venv/bin/python -m src.evaluate --run-dir ./runs/smoke-cnn-standard-s0 --subset 500
    .venv/bin/ruff format --check . && .venv/bin/ruff check . && .venv/bin/pytest -q

All checks below used scratch `RUNS_DIR` and `RESULTS_DIR` folders outside the repo.

### Verified (with output)
All on CPU. All accuracies are smoke numbers: the model had 1 epoch on 512 images.
- Lint/tests: `ruff format --check .` -> `78 files already formatted`; `ruff check .` -> `All checks passed!`; `time pytest -q` -> `264 passed in 27.36s` (real 0m28.690s, CPU).
- Hand-computed cases (written out in `tests/test_metrics.py`): accuracy 0.75; macro-F1 11/15 over labels [0, 1] and 22/150 over the 10 classes; degradation 0.3 and 1/3; toy tables CE {a: 0.6, b: 0.5}, mCE 0.55, relative mCE 7/12.
- `mce(base, base) == 1.0` and `relative_mce(base, base) == 1.0`, exact equality, on all 76 conditions.
- Missing conditions: a whole corruption, a whole severity and one condition, on either table, for all 3 functions -> `ValueError`.
- `macro_f1` vs `sklearn.metrics.f1_score(average="macro", labels=range(10), zero_division=0)`: 20 random pairs of length 1,000, one with a class never predicted -> equal within 1e-12.
- Step 2, `python -m src.train --config configs/smoke.yaml --seed 0`, then `python -m src.evaluate --run-dir $RUNS_DIR/smoke-cnn-standard-s0 --subset 500`:

      smoke-cnn-standard-s0: 76 conditions, 500 images each
      clean accuracy          0.0960
      mean corrupted accuracy 0.0969 (over 75 conditions)
      worst conditions:
        frost s3: 0.0820
        brightness s5: 0.0820
        snow s5: 0.0840

- First rows of that CSV (`head`, last two columns left out; `git_commit` = `5c96ebd...`, the commit before this phase, because the code was uncommitted):

      run_id,model,strategy,seed,corruption,family,group,severity,n_images,accuracy,macro_f1,loss
      smoke-cnn-standard-s0,simple_cnn,standard,0,clean,clean,clean,0,500,0.096,0.041743941327731154,2.33381494140625
      smoke-cnn-standard-s0,simple_cnn,standard,0,gaussian_noise,noise,seen,1,500,0.092,0.03734136960539118,2.33472802734375
      smoke-cnn-standard-s0,simple_cnn,standard,0,gaussian_noise,noise,seen,2,500,0.092,0.03816108449297106,2.3356103515625
      smoke-cnn-standard-s0,simple_cnn,standard,0,gaussian_noise,noise,seen,3,500,0.086,0.03227378293384894,2.336406494140625
      smoke-cnn-standard-s0,simple_cnn,standard,0,gaussian_noise,noise,seen,4,500,0.09,0.036592229364566055,2.337005126953125
      smoke-cnn-standard-s0,simple_cnn,standard,0,gaussian_noise,noise,seen,5,500,0.09,0.03401585090459174,2.337259033203125
      smoke-cnn-standard-s0,simple_cnn,standard,0,shot_noise,noise,unseen,1,500,0.094,0.03847356333837883,2.33453515625
      smoke-cnn-standard-s0,simple_cnn,standard,0,shot_noise,noise,unseen,2,500,0.092,0.03728077710605791,2.33511181640625
      smoke-cnn-standard-s0,simple_cnn,standard,0,shot_noise,noise,unseen,3,500,0.088,0.03580323935843721,2.336005126953125
      smoke-cnn-standard-s0,simple_cnn,standard,0,shot_noise,noise,unseen,4,500,0.094,0.03890588032864455,2.33648388671875

- Files: `smoke/` holds the `.csv` (77 lines: header + 76), `_confusion.npz` and `_train_log.csv`. `ls results/raw | wc -l` -> `0` in the repo.
- Independent recomputation (scratch script, real data): `rows 76 keys 76 all sum 500 True`; `max |acc csv - matrix| 0.0`; macro-F1 from each matrix through scikit-learn differs by at most 9.7e-17; a direct forward pass over `load_cifar10c("fog", 5)[:500]` -> `fog s5 direct 0.108 csv 0.108 equal True`.
- Refusals: a second evaluate -> `FileExistsError: refusing to overwrite [...]; use --force`; `status: running` -> `ValueError: refusing to evaluate ...: status is 'running'`.
- Extras: `--subset 50 --include-extra` -> `96 conditions`, summary still `(over 75 conditions)`; the 4 extras carry `family = extra`, `group = unseen`. Without the flag: no extras (test).
- Full pass, no `--subset`: `76 conditions, 10000 images each`, CSV 77 lines (timing below).
- Registry: `python -m scripts.registry` -> `has_results` True, `clean_accuracy` 0.096 for the smoke run.
- Leakage: `src/train.py` has no `train=False` or `cifar10c`; `tests/test_no_leakage.py` passes with `src/evaluate.py` as an allowed importer.
- The tests catch broken code (scratch copy, one change at a time, the two new test files; unchanged baseline passes first): 20 changes, all fail at least 1 test. They cover `model.eval()` and `no_grad` removed, clean severity 1, extras on by default, wrong groups, smoke to raw, `--force` and status checks off, subset ignored or taking the last images, every condition reading clean data, transposed confusion matrix, wrong macro-F1, CE as a mean of ratios, mCE as a sum of sums, rCE without the clean error, missing and repeated conditions not raised, error = accuracy, wrong relative degradation. On the first pass, `no_grad` removed and the transposed matrix survived, so 2 tests were added.

### Numbers produced
Evaluation time on CPU, this machine (no CUDA), 4 torch threads, batch size 512, not Colab [smoke]. Source: scratch timing script calling `score_condition`, 3 repeats on fog severity 3 with 10,000 images; the `time` output for the full pass; neither is stored in the repo.

| Model | `num_workers` | One condition, s (3 repeats) | Median | 76 conditions |
|---|---|---|---|---|
| simple_cnn | 2 | 10.23 10.11 10.87 | 10.23 | estimated 777 s (about 13 min) |
| simple_cnn | 0 | 12.14 11.33 12.00 | 12.00 | measured 860 s (14m20s; estimate 912 s) |
| resnet18 | 2 | 105.71 106.30 109.64 | 106.30 | estimated 8,079 s (about 2.2 h) |

- T4 time: UNKNOWN; this machine has no CUDA. The overview estimates 3-10 minutes per model. Phase 07 measures it.
- Full 10,000-image smoke pass: clean 0.1182, mean corrupted 0.1142 [smoke].

### Changed from earlier phases
- `src/metrics.py`, `src/evaluate.py`: were Phase 00 stubs. No other earlier file changed.

### Not done / deferred
- CUDA paths never ran here: AMP under `torch.autocast`, `pin_memory` -> check in Phase 07.
- Seed aggregation and building the seed-mean baseline table -> Phase 09. Average `accuracy` per (corruption, severity, including clean) over the 3 Simple CNN + S1 seeds, one row per condition.
- Mean corrupted accuracy (D8) is computed only for the printed summary here; the stored table comes in Phase 09.

### Known issues
- **`macro_f1` has an optional `labels` argument and the metrics module a `confusion_matrix` function that C10 does not list.** BUILD's example passes `labels=[0, 1]`, and the default stays the 10 classes, so C10's two-argument call works unchanged.
- **BUILD asks for a 2 x 2 toy CE table, and also says the functions must raise on fewer than 15 corruptions and 5 severities.** C10's signatures are kept, so the toy tests shrink the module's `CORRUPTIONS` and `SEVERITIES` with `monkeypatch`, as `tests/test_cifar10c.py` does with `N_PER_SEVERITY`. Validators building toy tables outside a test need all 15 x 5.
- **C7 has no `group` for the extras.** D6 does not list them, and C7 allows only `clean`, `seen`, `unseen`. They are labelled `unseen` because training never uses them. This matters only with `--include-extra`, which is off by default.
- `git_commit` in the CSV is HEAD and does not show uncommitted changes. Evaluate real runs from a clean tree.
- An interrupted evaluation keeps nothing: results are written only after all conditions. A leftover hidden `.tmp` file in the results folder can be deleted and is replaced by the next run.
- A run whose stored config says `device: cuda` cannot be evaluated without a GPU. The configs use `auto`, so this applies only to a hand-edited config.
- `--subset` on a real run writes to `smoke/` under the same `smoke-` id a smoke training run may use, so `--force` is needed if both are evaluated (same kind of id sharing as the Phase 05 known issue).
- The train log copy is overwritten on every evaluation; it is a copy of the run folder's log, so this loses nothing.

### Decisions used
- D3: `model_final.pt` (the final-epoch model) is what gets evaluated.
- D2: the eval transform, for clean and CIFAR-10-C.
- D6: `group` `seen` for the 4 seen corruptions, `unseen` for the other 11.
- D7: clean + 15 x 5 = 76 conditions; extras only with `--include-extra`, never in mCE or the main averages.
- D8: fractions; error = 1 - accuracy; degradation, CE, mCE and relative mCE formulas; seed-mean baseline; mean corrupted accuracy over the 75 conditions.

### Ideas for later
- none
