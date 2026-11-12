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

## Phase 07: pilot  (2026-11-04, branch phase-07-pilot, commit: the one titled "phase 07: pilot run cnn-standard-s0 and frozen epochs"; the run itself used 004edc3)

### Built
- `notebooks/run_on_colab.ipynb`: parameters cell (`MODEL`, `STRATEGY`, `SEED`, `STAGE`, plus `RUN_TIMING`), then setup, data, train, evaluate, save to Drive, optional timing, flush. It stops on any failed step and if there is no GPU. Console output is kept on Drive under `results/logs/`. Calls only.
  - The setup cells are copied from `00_colab_setup.ipynb` rather than `%run`, because the repo does not exist yet when the notebook opens.
  - `results/raw/` is restored from Drive at the start, and evaluation is skipped if the run's CSV is already there.
  - `RUN_TIMING` also evaluates a throwaway 1-epoch ResNet-18 (`smoke-timing-resnet18-eval`), because BUILD's timing runs stop unfinished and cannot be evaluated. Its output goes to `results/smoke/`; only the time is used.
- `results/raw/cnn-standard-s0.csv`, `_confusion.npz`, `_train_log.csv`, `results/runs.csv`, `results/pilot_report.md`.

### How to run it
    # Colab: open the notebook from this branch, GPU runtime, set the parameters, Runtime > Run all
    https://colab.research.google.com/github/aysha0559/Image-Classification-with-ResNet-18/blob/phase-07-pilot/notebooks/run_on_colab.ipynb
    # Locally, with the run's run_meta.json, train_log.csv and model_final.pt in runs/cnn-standard-s0/
    .venv/bin/python -m scripts.registry
    RESULTS_DIR=/some/scratch .venv/bin/python -m src.evaluate --run-dir runs/cnn-standard-s0 --subset 2000

### Verified (with output)
- GPU: `device_name` in `run_meta.json` -> `Tesla T4`, for the pilot and all 5 timing runs. `git_commit` -> `004edc3c3324ebd5174b4a1819d9bb6858885511`, `git_dirty` -> false, `status` -> `finished`, `epochs` -> 30.
- AMP, `GradScaler`, CUDA RNG restore and resume on a GPU ran for the first time here: no error came from them.
- Real disconnect: Runtime > Disconnect and delete runtime during training. The next session printed `cnn-standard-s0: resuming after epoch 11/30`, then `epoch 12/30`, with no `WARNING`. The log has 30 rows, epochs 1..30 once each, no NaN; `lr` matches the cosine formula (`numpy.allclose` -> True). `started_at` kept the first session's value.
- Evaluation output on Colab: `76 conditions, 10000 images each`, `clean accuracy 0.8404`, `mean corrupted accuracy 0.5974 (over 75 conditions)`.
- Files after download: CSV `(76, 14)`, columns match C7, `n_images` only `[10000]`; npz 76 matrices, each summing to 10000; `train_log.csv` in the run folder identical to `results/raw/`.
- Own forward pass on CPU, `model_final.pt`, all 10,000 clean test images, torch 2.14.1 (before the pin change) -> `8404 10000 0.8404`, CSV `0.8404`.
- `src.evaluate --run-dir runs/cnn-standard-s0 --subset 2000` with a scratch `RESULTS_DIR`, torch 2.11.0 -> `76 conditions, 2000 images each`, clean 0.8405 [smoke]. `results/raw` unchanged.
- `python -m scripts.registry` -> `wrote results/runs.csv (1 runs)`; row has `has_results` True, `clean_accuracy` 0.8404.
- Scaler fix: the new test passes, and fails with the fix removed (`1 failed`).
- Notebook dry run under IPython in a scratch folder, CPU, data shrunk, fake Drive: stopped at epoch 3 and re-run -> `resuming after epoch 3/30`; third run -> `already finished` and `results already exist`; a missing config -> `FAILED at cell 13`. One timing-cell line changed after that and was checked only with `var_expand`; it then ran on Colab.
- Lint/tests with torch 2.11.0: `ruff format --check .` -> `81 files already formatted`; `ruff check .` -> `All checks passed!`; `time pytest -q` -> `265 passed in 56.01s` (real 0m59.137s, CPU).
- Timing runs: `results/raw` holds only `cnn-standard-s0*`. The 5 `timing-*` folders and `smoke-timing-resnet18-eval` were moved to Drive Trash after the human confirmed (Drive: `6 items moved to trash`). Only `cnn-standard-s0` remains in `runs/`.

### Numbers produced
All in `results/pilot_report.md`, with sources [real, except the ResNet-18 evaluation model, which is smoke and used only for its time].
- Clean accuracy 0.8404, macro-F1 0.8391; mean corrupted accuracy 0.5974 (source: `results/raw/cnn-standard-s0.csv`). Gate 0.78-0.90 passed.
- Train time 620.0 s, 30 epochs (source: `run_meta.json`).
- Evaluation wall time on 76 conditions: simple_cnn 182 s, resnet18 210 s (source: console logs).
- Mean s per epoch on the T4: cnn standard 20.667, augmix 118.171, corruption_aware 28.180; resnet18 standard 24.340, augmix 122.687, corruption_aware 35.268 (source: train logs; the timing values are written out in the report).
- Projected GPU hours for the 18-run matrix: 9.71 at 30 ResNet epochs, 11.23 at 40, 12.75 at 50.

### Changed from earlier phases
- `src/train.py`: skips an empty scaler state on resume, because a CPU checkpoint would otherwise fail on a GPU (`validation/phase-05.md` note, agreed before that merge). `tests/test_train_smoke.py`: 1 new test.
- `docs/agent-prompts/CONTRACTS.md` C8: `error` is not a registry column (same agreement).
- `docs/agent-prompts/DECISIONS.md`: D4 ResNet-18 epochs `CONFIRMED` at 40; 2 change-log rows.
- `configs/resnet18_*.yaml`: comment only; `epochs` was already 40.
- `requirements-local.txt`: torch 2.11.0, torchvision 0.26.0 (were 2.14.1 and 0.29.2), the human's choice, to meet D9. `.venv` reinstalled.

### Not done / deferred
- `BRANCH` in `run_on_colab.ipynb` is `phase-07-pilot`; set it to `main` after the merge -> Phase 08.
- "Run the matrix" mode and syncing after each run -> Phase 08.
- The timing folders are in Drive Trash, not permanently deleted; Trash was not emptied.
- `model_final.pt` is at `runs/cnn-standard-s0/` on this machine and on Drive, not committed.

### Known issues
- **The pilot differs from BUILD's single session.** A first T4 session (12:00 UTC) was stopped during the download, which would have taken about 2 hours of GPU time. The human chose to prepare the data on a CPU runtime instead. Training then ran in two T4 sessions, and the timing runs in the second, as BUILD asks. Results are unaffected: CIFAR-10-C passed `all 20 files match` in every session.
- **Disconnect and delete does not stop the process at once.** Page output stopped at epoch 6, but the process ran on through epoch 11 and died writing the log for epoch 12 with `FileNotFoundError`. That traceback is in `results/logs/cnn-standard-s0.console.log` on Drive. Session 2 did not start until the old process had stopped writing, judging by the log having no gaps.
- The console log on Drive has no `=== session 2026-11-04T13:17:55Z` line, although the notebook showed it. Likely cause: two runtimes wrote the same Drive file. This is not confirmed. No result reads this file.
- Colab's page can show `Connecting`, `Runtime disconnected` or `Resuming execution` while the runtime keeps working; it twice recovered with all output intact. Read the outputs before treating one as a failure.
- `pip install -r requirements.txt` on Colab reports `google-colab 1.0.0 requires pandas==2.2.3, but you have pandas 2.3.3`. Nothing failed from it here.
- Colab runs Python 3.13.16; the local venv is 3.12.16. Phase 00's install check covered Python 3.10-3.12 only, though the install worked on Colab.
- AugMix is about 5 times slower than standard on ResNet-18, above the overview's 1.5-3 times. The extra cost looks like CPU work in the 2 DataLoader workers: the CNN and ResNet-18 take about the same time per epoch with it. Not tuned; it fits the budget.
- Projected hours leave out session setup, about 4 minutes per session here.

### Decisions used
- D4: CNN 30 epochs, seed 0, AMP on CUDA, seeds and determinism as stated; ResNet-18 40 epochs frozen from measured time.
- D3: `model_final.pt` evaluated.
- D7: 76 conditions, no extras.
- D9: Colab's preinstalled torch used; local pins now match its major.minor; data and runs on Drive.

### Ideas for later
- none

### Phase 07 addendum (2026-11-04, after validation PASS)
- `results/pilot_logs/`: added at the human's approval, with C1 updated first. It holds the 2 console logs, `run_meta.json` of the pilot and the 5 timing runs, and the timing runs' train logs, so `results/pilot_report.md` traces to files in the repo. They are the files downloaded from Drive, unchanged.
- `results/pilot_report.md`: sources point there, and the disconnect section now says 2 tracebacks, as `validation/phase-07.md` noted. No number changed.

## Phase 08: experiment matrix, orchestration only  (2026-11-04, branch phase-08-experiment-matrix, commit: the one titled "phase 08: matrix orchestration")

**The matrix has not run.** This entry covers the tooling. The human starts the Colab sessions, and the final entry with the registry table comes after them. None of the "Done when" items that need results is met.

### Built
- `scripts/run_all.sh`: C5's loop. Skips evaluated runs, resumes unfinished ones, evaluates finished ones, stops on budget, prints the registry and plan at the end. Stops the whole matrix on any failure (exit 1), because the same fault would hit the next runs.
- `scripts/plan.py`: the run table and remaining GPU hours; also what `run_all.sh` asks before each step (`--run-ids`, `--next-step`). Reads epoch times from finished runs' logs, else from the tables in `results/pilot_report.md`. It imports no torch, so a call takes about 0.1 s.
- `scripts/collect_results.py`: `--from PATH`. Checks every evaluated run against C7, the confusion matrices against the CSV, the train log and metadata against the configs (D4). Any problem or differing duplicate lists all problems and copies nothing.
- `notebooks/run_on_colab.ipynb`: `STAGE = "matrix"` (now the default), `SESSION_MINUTES = 180`, `BRANCH = "main"`. The single-run stages still work the same way.
- `results/RUNLOG.md`: header only, no sessions yet.
- `tests/test_matrix.py`: 52 cases. Runs the real `run_all.sh` with a fake interpreter and clock; the C7 check also runs on the real pilot files.

### How to run it
    # Colab, after this branch is merged to main and pushed; repeat until done
    https://colab.research.google.com/github/aysha0559/Image-Classification-with-ResNet-18/blob/main/notebooks/run_on_colab.ipynb
    # Here
    .venv/bin/python -m scripts.plan
    .venv/bin/python -m scripts.collect_results --from <downloaded results folder>

### Verified (with output)
- Lint/tests: `ruff format --check .` -> `84 files already formatted`; `ruff check .` -> `All checks passed!`; `time pytest -q` -> `317 passed in 56.14s` (real 0m57.518s, CPU).
- Nothing that affects training or evaluation changed: `git diff --stat main -- src configs requirements.txt` -> empty. `git diff --stat 004edc3 -- src configs` -> `4 files changed, 4 insertions(+), 4 deletions(-)`, the 4 ResNet config comment lines that V4 of Phase 07 already recorded.
- `python -m scripts.plan` -> `done 1, missing 17 of 18 runs`, `remaining GPU hours: 11.01 (training 10.08, evaluation 0.93; session setup not included)`. That is the report's 11.23 less the pilot's own training and evaluation. Execution order within a seed: cnn standard, cnn corruption_aware, resnet18 standard, resnet18 corruption_aware, cnn augmix, resnet18 augmix.
- Notebook dry run under IPython: the real notebook, patched only for paths, repo URL, `SESSION_MINUTES`, `--models cnn`, and no pip or GPU check, in a scratch folder on CPU with a fake Drive and a scratch `main` whose CNN configs had 2 epochs. Two sessions, each from a fresh `/content` [smoke]:
  - Session 1: `cnn-standard-s0: already evaluated, skipping`; `cnn-corruption_aware-s0` trained and evaluated; `session budget reached before cnn-augmix-s0: the next epoch needs about 118 s, -449 s left`; exit 0.
  - Session 2: both evaluated runs skipped after restoring from the fake Drive; `cnn-augmix-s0` trained and evaluated; `session budget reached before cnn-standard-s1`; exit 0.
  - Both new runs: `git_dirty: False`, one commit. The repo copy stayed clean (`git status --short | wc -l` -> `0`). Fake Drive: 9 result files, `runs.csv` with 3 rows, CSVs of 77 lines, no `.partial` file.
- `collect_results` on that fake Drive, real files [smoke]: against the real configs it refused, `2 rows do not count epochs 1..30 once each` and `run_meta epochs is 2`, as it should. With scratch configs matching the runs: `3 evaluated runs checked, 12 files copied`; run again: `0 files copied`.
- The tests catch broken code (scratch copy, one change at a time, baseline `52 passed`): 12 changes, all fail at least 1 test: order, skip logic, epochs that fit, evaluation budget check, last-epoch estimate, no-progress guard, `--force` rule, final sync, import after problems, duplicate refusal, train log check, confusion check. The no-progress change hung until killed by hand, so `Matrix.run` now has a 30 s timeout.

### Numbers produced
None for the matrix. All accuracies above are [smoke].

### Changed from earlier phases
- `notebooks/run_on_colab.ipynb`: see Built; `BRANCH` was deferred here by Phase 07.
- `docs/agent-prompts/CONTRACTS.md` C2: adds `SESSION_BUDGET_MIN`, `RESULTS_SYNC_DIR` and the Colab matrix `RESULTS_DIR`, approved by the human; `DECISIONS.md` change log has the row.
- `src/`, `configs/`, `requirements*.txt`: nothing. The budget uses the existing `STOP_AFTER_EPOCHS`, because V7 and V13 treat any change to `src/` after the pilot as a change the pilot would have to answer for.

### Not done / deferred
- The 17 remaining runs -> the human, on Colab, about 11 GPU hours.
- Agent steps after each batch, RUNLOG entries, the registry table, commit `phase 08: full experiment matrix results` -> after the sessions.
- `BUILD` asks that new runs use a commit on `main`, so this orchestration was merged to `main` and pushed before the matrix, with the human's go-ahead. The phase branch stays open for the results.
- CPU only here: `run_all.sh` has not run on a GPU or on the real Drive.

### Known issues
- **`RESULTS_DIR` is `/content/results` in matrix mode.** `results/runs.csv` is tracked, so rebuilding it in the clone would mark every later run `git_dirty`, a V7 failure. The notebook also refuses to start if tracked files have changed.
- **`results/run_folders/{run_id}/` on Drive holds `run_meta.json`, `config_resolved.yaml`, `train_log.csv`.** BUILD does not list it, but `scripts.registry` builds from run folders, so a local rebuild would otherwise drop every run whose folder is only on Drive. `collect_results` copies these to `$RUNS_DIR`.
- **The budget is an estimate made when a process starts, not after each epoch.** `src.train` cannot be told to stop later without changing `src/`. An overrun is the estimate error times the epochs in that call; `run_all.sh` re-estimates and continues if more fits, which counts as a resume. `epoch_time_s` leaves out checkpoint writes. Evaluation can also overrun: the dry-run overruns (449 s and 617 s) came from CPU evaluation against the 182 s T4 estimate.
- A budget stop leaves `status: running`, so the next session resumes it. Note those resumes in the run log (Phase 05 known issue).
- `SESSION_MINUTES = 180` is a setting, not a measured limit. If Colab ends a session earlier, at most one epoch is lost, but the last files may not reach Drive; the next session redoes them.
- Drive space is unmeasured. Each ResNet-18 epoch replaces a checkpoint of up to 86 MB (Phase 05). Whether replaced files pile up in Drive Trash is UNKNOWN; the notebook prints `df -h` at the start and end of each session, so check it after the first ResNet-18 run.
- A manual disconnect can leave the old process writing for a while (Phase 07), so the notebook says to wait before starting again. Nothing enforces it.
- **`results/logs/matrix.console.log` on Drive cannot show that a disconnected process has stopped.** `tee` keeps it open, so Drive uploads it only now and then: during session 2 it stayed at 18:40 for 3 hours while the run was healthy. In session 1 the page said "Runtime disconnected" at 18:30 and the log's time was 18:26, yet the process logged up to epoch 20, about 18:36. Session 2 was started at about 18:36 on that wrong reading. The log shows epochs 1-20 once each, no traceback and no epoch 21 from the old process, and session 2 began training at about 18:43. Checked on import: the run's train log has epochs 1-30 once each, `lr` follows the cosine schedule, and its first 20 rows match the console log exactly, so nothing shows that the two processes overlapped.
- **After a disconnect, judge by files that are written and closed**: `run_meta.json`, `checkpoint_last.pt`, the results CSV. Wait at least 15 minutes before starting a new session.
- **Colab's free GPU limit was reached on 2026-11-04 at 21:59 UTC**, after about 5 hours of GPU time that day. The reset time is UNKNOWN: a retry on 2026-11-05 at 06:34 UTC, over 8 hours later, was refused with the same message. On 2026-11-09 at 12:43 UTC the reply changed to "No backend with GPU available", a capacity message rather than the usage limit.
- **After a disconnect, `drive.mount` failed once with `ValueError: mount failed`** when the prompt had stayed open during an extension reconnect (18:38). Run all on the same runtime mounted it. No result was affected.
- If the log is behind the checkpoint by more than the epochs that fit, `run_all.sh` stops with `made no progress` rather than looping.

- **`resnet18-corruption_aware-s1` shows as `missing` here but is trained.** Its folder files reach Drive's `run_folders/` only when a session syncs it, and session 2 was cut off first. `run_meta.json` in `runs/` on Drive says finished (21:33 UTC), and the next session only evaluates it. So `scripts.plan`'s 6.27 hours here overstates the time left by that run's training.
- **The last run moves to Kaggle (2026-11-10, approved by the human).** Colab refused a GPU from 2026-11-09 23:29 PKT onwards. `notebooks/run_on_kaggle.ipynb` clones commit `7d334eb`, the same as every Colab run, downloads and verifies the data, and runs `scripts/run_all.sh --models resnet18 --strategies augmix --seeds 2` from scratch, exporting in the Drive `results/` layout. Recorded in DECISIONS D9 and its change log, and in CONTRACTS C1 and C2. No change to `src/`, `configs/` or `main`. The partial Colab run (8 of 40 epochs) is not used.
- **Second batch imported (2026-11-09):** 17 runs, 1,292 rows, all passing C7, all Tesla T4, `git_dirty` False, 16 at `7d334eb`. Configs identical per model; train logs hold epochs 1..E with the cosine `lr`; accuracy falls with severity in every run; clean accuracy in range. Clean-accuracy spread over seeds, in points: cnn standard 1.13, cnn augmix 1.06, cnn corruption_aware 0.71, resnet18 standard 0.11, resnet18 corruption_aware 0.37, resnet18 augmix 0.05 (2 seeds). None above 1.5. `resnet18-augmix-s2` has 8 of 40 epochs on Drive.
- **First batch imported (2026-11-05):** 9 runs, 684 rows, all passing C7. All on Tesla T4, `git_dirty` False; 8 at `7d334eb`, the pilot at `004edc3` (4 comment lines differ under `src/` and `configs/`). Configs identical per model apart from strategy and seed. Clean accuracy in the overview's range, no NaN, accuracy falls with severity in every run. Seed spread so far, in points: cnn standard 0.14, cnn corruption_aware 0.17, resnet18 standard 0.04.

### Decisions used
- D3: `src.evaluate` scores the final-epoch model. D4: epochs from the configs, 30 and 40. D5, D6: default strategies, no S0. D7: 76 conditions. D9: data and runs on Drive.

### Ideas for later
- none

## Phase 08: full experiment matrix results  (2026-11-10, branch phase-08-experiment-matrix, commit: the one titled "phase 08: full experiment matrix results")

All 18 runs are trained, evaluated, imported and checked. The orchestration entry above describes the tools; this entry covers the results. Not yet validated: `validation/phase-08.md` comes from a separate session.

### Built
- `notebooks/run_on_kaggle.ipynb`: runs `scripts/run_all.sh` on a Kaggle GPU at commit `7d334eb` and exports in the Drive `results/` layout. Used for `resnet18-augmix-s2` only (DECISIONS D9 exception).
- `results/raw/`: 54 files (CSV, confusion, train log) for the 18 runs. `results/runs.csv`: rebuilt by `scripts.collect_results`.
- `results/RUNLOG.md`: one row per session, 4 Colab and 1 Kaggle.

### How to run it
    .venv/bin/python -m scripts.plan
    .venv/bin/python -m scripts.collect_results --from <downloaded results or export folder>

### Verified (with output)
- `python -m scripts.plan` -> `done 18 of 18 runs`, `remaining GPU hours: 0.00`.
- `collect_results` on each download: `exit 0`; last batch `1 evaluated runs checked, 6 files copied`, `resnet18-augmix-s2: finished, 40 epochs, commit 7d334eb, dirty False, Tesla T4, clean 0.9369`.
- Scratch check script on `results/raw/` and the run folders: `18 runs; 1368 rows`; `epochs per model: {'simple_cnn': 30, 'resnet18': 40}`; `commits: {'7d334eb': 17, '004edc3': 1}`, all `Tesla T4`, `git_dirty` False, `finished`; every train log holds epochs 1..E once with the cosine `lr`; config_resolved.yaml identical per model apart from strategy and seed; no `CONFIG DIFF`; accuracy falls with severity in every run; every clean accuracy inside the overview range.
- `git diff --stat 004edc3 7d334eb -- src configs` -> 4 comment lines in the ResNet configs, as recorded in Phase 07; the pilot is the only run at `004edc3`.
- Kaggle run versions: `run_meta.json` -> torch `2.11.0+cu130`, torchvision `0.26.0+cu130`, Python `3.13.16`, `Tesla T4`: the same as the 2026-11-04 Colab runs (the 2026-11-09 Colab runs used torch 2.14.1; see Known issues).
- Kaggle notebook dry run on CPU first (scratch clone, 1-epoch config): `done 1 of 1 runs`, export holds `raw/` (3 files), `run_folders/` (3 files), `runs.csv`; `git_dirty` False.
- Lint/tests: `ruff format --check .` -> `85 files already formatted`; `ruff check .` -> `All checks passed!`; `pytest -q` -> `317 passed in 39.53s`.

### Registry (`results/runs.csv`)
| run_id | status | epochs | clean_accuracy | train_time_s | device | git_commit |
|---|---|---|---|---|---|---|
| cnn-standard-s0 | finished | 30 | 0.8404 | 620 | Tesla T4 | 004edc3 |
| cnn-standard-s1 | finished | 30 | 0.8418 | 532 | Tesla T4 | 7d334eb |
| cnn-standard-s2 | finished | 30 | 0.8517 | 576 | Tesla T4 | 7d334eb |
| cnn-augmix-s0 | finished | 30 | 0.8368 | 3281 | Tesla T4 | 7d334eb |
| cnn-augmix-s1 | finished | 30 | 0.8272 | 3343 | Tesla T4 | 7d334eb |
| cnn-augmix-s2 | finished | 30 | 0.8378 | 3602 | Tesla T4 | 7d334eb |
| cnn-corruption_aware-s0 | finished | 30 | 0.8202 | 816 | Tesla T4 | 7d334eb |
| cnn-corruption_aware-s1 | finished | 30 | 0.8219 | 763 | Tesla T4 | 7d334eb |
| cnn-corruption_aware-s2 | finished | 30 | 0.8273 | 869 | Tesla T4 | 7d334eb |
| resnet18-standard-s0 | finished | 40 | 0.9364 | 992 | Tesla T4 | 7d334eb |
| resnet18-standard-s1 | finished | 40 | 0.9368 | 954 | Tesla T4 | 7d334eb |
| resnet18-standard-s2 | finished | 40 | 0.9375 | 1047 | Tesla T4 | 7d334eb |
| resnet18-augmix-s0 | finished | 40 | 0.9372 | 4602 | Tesla T4 | 7d334eb |
| resnet18-augmix-s1 | finished | 40 | 0.9367 | 4691 | Tesla T4 | 7d334eb |
| resnet18-augmix-s2 | finished | 40 | 0.9369 | 2502 | Tesla T4 | 7d334eb |
| resnet18-corruption_aware-s0 | finished | 40 | 0.9236 | 1343 | Tesla T4 | 7d334eb |
| resnet18-corruption_aware-s1 | finished | 40 | 0.927 | 1288 | Tesla T4 | 7d334eb |
| resnet18-corruption_aware-s2 | finished | 40 | 0.9233 | 1475 | Tesla T4 | 7d334eb |

### Seed consistency (clean accuracy, points)
| Config | max - min | sample std |
|---|---|---|
| cnn standard | 1.13 | 0.62 |
| cnn augmix | 1.06 | 0.59 |
| cnn corruption_aware | 0.71 | 0.37 |
| resnet18 standard | 0.11 | 0.06 |
| resnet18 augmix | 0.05 | 0.03 |
| resnet18 corruption_aware | 0.37 | 0.21 |

None above 1.5 points; nothing flagged.

### Numbers produced
- Clean accuracy per run: table above (source: `results/runs.csv`) [real]. Analysis numbers belong to Phase 09.

### Changed from earlier phases
- `docs/agent-prompts/DECISIONS.md` D9 and change log, `CONTRACTS.md` C1 and C2: the Kaggle exception, approved by the human on 2026-11-10.

### Not done / deferred
- `validation/phase-08.md` -> a separate validation session (VALIDATE V9 needs 2 `model_final.pt` files: Drive `runs/` for the Colab runs; the Kaggle run's is in its Kaggle output and in `~/Downloads/results.zip`).
- Merge to `main` and push -> after validation, with the human's go-ahead.
- Rename the unused partial Colab run on Drive (`runs/resnet18-augmix-s2`, 8 epochs) with an `_invalid-moved-to-kaggle` suffix -> when Drive is tidied; the human asked to leave Drive alone for now.

### Known issues
- **`resnet18-augmix-s2` ran on Kaggle, the other 17 on Colab.** Same commit and GPU model; same torch version as the 2026-11-04 Colab runs. Its epochs took about 63 s against about 120 s on Colab, because AugMix is limited by CPU and Kaggle gave more CPU; `train_time_s` and timing comparisons across platforms are not like for like. Disclose in the report (DECISIONS change log, 2026-11-10).
- **Two torch versions in the matrix (found by `validation/phase-08.md`, V14).** Colab's preinstalled torch changed between sessions: torch `2.14.1+cu130` / torchvision `0.29.1+cu130` for the 7 runs of 2026-11-09 (`cnn-augmix-s1`, `cnn-augmix-s2`, `cnn-corruption_aware-s2`, `cnn-standard-s2`, `resnet18-augmix-s1`, `resnet18-corruption_aware-s2`, `resnet18-standard-s2`); torch `2.11.0+cu130` / torchvision `0.26.0+cu130` for the other 11 (Colab 2026-11-04 and Kaggle). Allowed by DECISIONS D9 (platform versions, recorded in `run_meta.json`). Disclose in the report's limitations.
- **Interrupted runs:** `cnn-augmix-s0` (Colab session 1 cut off after epoch 20) and `resnet18-standard-s2` (budget stop after epoch 2) were resumed, so they need not match uninterrupted runs bit for bit (DECISIONS change log, 2026-11-04). `resnet18-corruption_aware-s1` was evaluated in a later session after its first evaluation was cut off.
- **Colab gave about 5 GPU hours a day and then refused**; one session ended at its maximum duration. Each case is in `results/RUNLOG.md`.

### Decisions used
- D3 final-epoch model; D4 epochs 30 and 40, seeds 0-2; D5 strategies S1-S3; D7 76 conditions; D9 with the Kaggle exception.

### Ideas for later
- none

## Phase 09: analysis  (2026-11-11, branch phase-09-analysis, commit: the one titled "phase 09: analysis tables and figures")

Built first on 9 of 18 runs (2026-11-05, in the worktree `../cifar10-phase09`, uncommitted, PARTIAL outputs never committed), with the human's approval while Colab waited for a GPU. On 2026-11-11 the branch was fast-forwarded to `main` (`2117ca0`, all 18 runs, `validation/phase-08.md` PASS WITH NOTES), the PARTIAL outputs were deleted, and every BUILD step was redone on the full data. All numbers below are from the 18 runs.

### Built
- `src/analysis.py`: `python -m src.analysis`. Reads only `$RESULTS_DIR/raw/*.csv`; refuses and lists the missing runs unless all 18 are present or `--allow-partial` is given (then every output is stamped PARTIAL in its name and title). Writes every C9 table, `report_numbers.json`, `report_numbers.md` (key, meaning, formula, source for every key) and `rq_answers.md` (RQ1-RQ5 data tables, paired-by-seed differences with per-seed values and a seed-noise label).
- `scripts/make_figures.py`: `python -m scripts.make_figures`. Reads only `results/tables/` and `results/raw/`. Writes fig1-fig6 and fig8 as PNG 200 dpi + PDF; fig7 is kept and regenerated only if missing. Same refusal guard.
- `notebooks/02_analysis.ipynb`: calls the two commands and displays the tables and figures. Saved without outputs.
- `tests/test_analysis.py`: 27 tests on a synthetic 18-run results set with known answers.
- `.gitignore`: ignores `results/tables/*_PARTIAL*` and `results/figures/*_PARTIAL*`.
- `results/tables/` (10 files) and `results/figures/` (fig1-6, fig8; PNG + PDF): the full outputs, committed.

### How to run it
    python -m src.analysis
    python -m scripts.make_figures

### Verified (with output)
- Full run: `python -m src.analysis` and `python -m scripts.make_figures` -> exit 0, 10 tables and 14 figure files written. `report_numbers.json`: 670 keys, 0 null; `meta.status = "FULL"`, `meta.n_runs = 18`, `meta.baseline_seeds = 3`.
- Baseline mCE: `main.cnn.standard.mce.mean` = `1.0` -> 100.0% (`main_results.md`: `100.0% ± 0.7%`).
- Guard: a copy of `results/raw` without `resnet18-augmix-s2.csv` -> `refusing to run: 1 of 18 runs are missing ...` exit 1 (`src.analysis`); `refusing to run: 1 runs are missing; use --allow-partial for PARTIAL figures. First missing: resnet18-augmix-s2` exit 1 (`scripts.make_figures`).
- Hand check: a separate script using only the `csv` and `statistics` modules (no pandas, no project code) recomputes from `results/raw/*.csv`:

      hand script                                             main_results.csv
      cnn.augmix.mean_corrupted_acc           0.6701591111111111   0.670159111111111
      resnet18.standard.mCE                   0.6848805918524928   0.6848805918524928
      resnet18.corruption_aware.relative_mCE  0.25712654792060935  0.25712654792060935
      cnn.standard.mCE                        1.0                  1.0

- Rebuild: copied all 24 outputs aside, deleted `results/tables/*` and every figure except fig7 (and `extra_our_corruptions.png`), re-ran both commands -> `identical=24 different=0` (`cmp`, CSV/JSON/MD and also PNG/PDF).
- Figures: fig1-6 and fig8 opened and read by eye. Titles, axis labels with units (%, percentage points), legends and cell numbers readable; same colour per strategy (S1 blue, S2 orange, S3 green) and line style per model (CNN dashed, ResNet-18 solid). No change needed.
- Notebook: all code cells executed in order from the repo root with a stand-in `IPython.display` (IPython is not in `.venv`) -> `18 of 18 runs present`, `Missing: none`, 14 displays, every file found; outputs afterwards still byte-identical (24/24). **Not run in a real Jupyter kernel.**
- Lint: `ruff format --check .` -> `92 files already formatted`; `ruff check .` -> `All checks passed!`
- Tests: `DATA_DIR=<repo>/data pytest -q` -> `344 passed in 59.10s`; `pytest -q tests/test_analysis.py` -> `27 passed in 19.51s`.

### Numbers produced
All [real], 3 seeds each, mean ± std (ddof=1). Source: `results/tables/main_results.csv` / `.md`.

| Model | Strategy | Clean acc | Mean corrupted acc | mCE | Relative mCE |
|---|---|---|---|---|---|
| simple_cnn | standard | 84.5% ± 0.6% | 59.7% ± 0.0% | 100.0% ± 0.7% | 100.0% ± 2.3% |
| simple_cnn | augmix | 83.4% ± 0.6% | 67.0% ± 0.8% | 82.9% ± 1.7% | 63.6% ± 0.9% |
| simple_cnn | corruption_aware | 82.3% ± 0.4% | 74.8% ± 0.6% | 68.5% ± 1.6% | 36.1% ± 1.3% |
| resnet18 | standard | 93.7% ± 0.1% | 71.6% ± 0.3% | 68.5% ± 0.9% | 90.7% ± 1.8% |
| resnet18 | augmix | 93.7% ± 0.0% | 83.7% ± 0.5% | 40.7% ± 0.9% | 42.7% ± 1.6% |
| resnet18 | corruption_aware | 92.5% ± 0.2% | 87.6% ± 0.1% | 33.8% ± 0.3% | 25.7% ± 1.0% |

- Seed-noise labels in `report_numbers.json`: 72 comparisons, all `beyond seed noise`, none `within seed noise` or `not assessable`.
- Worst family by mean absolute drop: `noise` for both models with standard training (`rq5.worst_family.*.standard`); worst corruption `gaussian_noise` (`rq5.worst_corruption.*.standard`).

### Changed from earlier phases
- none. No change to training, evaluation, configs or Phase 08 scripts.

### Not done / deferred
- Prose conclusions for RQ1-RQ5 -> Phase 10.
- Notebook run in a real Jupyter/Colab kernel -> Phase 10 or 11 (IPython is not installed locally).

### Known issues
- `--allow-partial` on `scripts.make_figures` is not listed in CONTRACTS C5 (BUILD names it for `src.analysis` only), and the `.gitignore` PARTIAL rule is not in C1's "Not committed" list. Neither affects full runs; the human may want them recorded in CONTRACTS.
- `src.analysis` imports `src.utils` for `env_path`, which imports torch, so it needs the torch install although it uses no torch.
- `simple_cnn + standard` mean corrupted accuracy shows `± 0.0%`: the std rounds to 0.0 at one decimal (`main_results.csv` has the full value).

### Decisions used
- D6: the three analysis groups, from `src/constants.py`. D7: extras dropped, 75 corrupted conditions. D8: baseline = seed-mean of Simple CNN + standard, CE, mCE, relative mCE, degradation on accuracy, fractions in CSV/JSON.
- std is the sample std over seeds (ddof=1) everywhere (`meta.std_ddof`). Per-corruption mean and std: each seed is averaged over severities first, then summarised over seeds. No p-values.

### Ideas for later
- none

## Phase 10: report  (2026-11-11, branch phase-10-report, commit: the one titled "phase 10: report and README")

### Built
- `scripts/build_report.py`: `python -m scripts.build_report [--quiet]`. Fills `report/report.template.md` -> `report/report.md` and `README.template.md` -> `README.md` from `$RESULTS_DIR/tables/report_numbers.json`. Formats `pct1`, `pp1`, `negpp1` (the negated difference, for prose that states the direction in words; fails if the value is positive), `f1`, `f3`, `int`, and no format for text and yes/no values only. Fails on: an unknown key or format, a null value, PARTIAL data, a malformed placeholder, and any digit outside a placeholder that no `ALLOW_LIST` entry covers. Nothing is written unless both templates succeed. Every allow-list hit is printed with its reason.
- The allow-list has 39 entries, each bound to its context (for example `40 epochs`, `batch size 128`, `probability 0.5`, `Section 5.6`, `[3]`, run ids, `ResNet-18`), not bare numbers. Reference entries are exempt only line by line ("Surname, X. ... (year).").
- `report/report.template.md`: the report in the BUILD order (Abstract; 1 Introduction; 2 Related work; 3 Methodology; 4 Experimental setup; 5 Results, RQ1-RQ5 with a verdict each, plus 5.6 on class collapse; 6 Discussion; 7 Limitations; 8 Conclusion and future work; References). All 8 figures embedded and cited.
- `README.template.md`: title and one-line summary (overview 17), fig2 and fig7, the 6-row results table, 5 findings, reproduce steps (view, re-run analysis, one Colab run, everything), layout, citations, licence, "Live demo" placeholder line for Phase 11.
- `report/report.md`, `README.md`: generated.
- `src/analysis.py`: new report numbers (see Changed from earlier phases). `results/tables/report_numbers.json` now has 1105 keys (was 670), 0 null.
- `tests/test_build_report.py`: 28 test cases (formats, failures, the guard with context-bound allow-list, nothing written on failure, committed outputs equal a rebuild, all 8 figures cited). `tests/test_analysis.py`: 6 new tests for the new keys.

### How to run it
    .venv/bin/python -m src.analysis
    .venv/bin/python -m scripts.make_figures
    .venv/bin/python -m scripts.build_report
    # optional PDF, pandoc is not installed here (not run):
    cd report && pandoc report.md -o report.pdf --resource-path=.

### Verified (with output)
- `python -m scripts.build_report` -> exit 0, `wrote report/report.md`, `wrote README.md`, the allow-list hits printed; every allow-list entry is used at least once (checked with `check_digits` over both templates -> `unused: []`).
- Guard, in a scratch copy of the repo: `The model has an accuracy of 91.2% on fog.` appended to the template -> `ERROR: digits outside placeholders ...` `report/report.template.md:255`, exit 1; the same sentence in the abstract -> exit 1; `{{no.such.key}}` -> `ERROR: report/report.template.md: unknown report number key(s): no.such.key`, exit 1. A first version exempted the whole References section, so text appended at the end passed; that is why reference entries are now exempt line by line, with a test.
- Rebuild: `python -m scripts.build_report` twice -> identical outputs; `test_committed_outputs_match_a_rebuild` passes.
- Analysis outputs: after the changes, `main_results.csv/.md`, `per_corruption.csv`, `per_severity.csv`, `per_family.csv`, `seen_unseen.csv/.md` are byte-identical to `main` (`cmp`); none of the 670 earlier keys changed value (`changed []`). `python -m scripts.make_figures` -> no figure file changed (`git status`).
- Citations: all 7 checked in this session against publisher or proceedings pages (CVF and Crossref, arXiv API, ICLR virtual site and ML Anthology for the OpenReview ids, NeurIPS proceedings, PMLR) and the Zenodo record of CIFAR-10-C; full author lists and pages filled in for [5] and [7]. OpenReview and dblp blocked automated access, so the ICLR entries were checked through arXiv and ML Anthology.
- RQ verdicts against `rq_answers.md` and the seed-noise rule: RQ1 supported, RQ2 partly supported, RQ3 supported, RQ4 supported, RQ5 supported for standard training. Each verdict sentence quotes the seed-noise labels it rests on by placeholder.
- Every DECISIONS change-log entry is disclosed in Limitations: initial choices fixed before the runs; JPEG mismatch; resumed runs; C5/C6 (shortened runs labelled as tests, run metadata); ResNet-18 epochs at 40 from measured time; D9 local pins; C1 pilot logs; C2 environment variables; the Kaggle run.
- Length: `sed '/^## References/,$d' report/report.md | wc -w` -> `5401` (tables included).
- Lint/tests: `ruff format --check .` -> `97 files already formatted`; `ruff check .` -> `All checks passed!`; `DATA_DIR=<repo>/data pytest -q` -> `378 passed in 91.17s`.

### Numbers produced
All in `results/tables/report_numbers.json` [real]; new ones written by `src/analysis.py` (documented in `report_numbers.md`). Highlights used in the text:
- Training time of all 18 runs: 9.2 GPU hours (`compute.train_hours_total`, from `results/runs.csv`).
- Clean-accuracy change against standard: Simple CNN AugMix -1.1 pp, Simple CNN corruption-aware -2.2 pp, ResNet-18 corruption-aware -1.2 pp (all beyond seed noise), ResNet-18 AugMix +0.0 pp (within seed noise) (`rq3.*.clean_acc.diff_vs_standard`).
- RQ5 ties found: worst corruption under standard training, Gaussian noise vs glass blur, within seed noise for both models; mildest family for ResNet-18 standard (digital vs weather) and ResNet-18 AugMix (weather vs digital), within seed noise.
- ResNet-18 standard, Gaussian noise severity 5: 50.9% of predictions "frog", 30.2% "bird" (`confusion.*`).

### Changed from earlier phases
- `src/analysis.py`, as BUILD asks for numbers the report needs ("add it to src/analysis.py"). Earlier outputs unchanged (see Verified). New keys:
  - `rq3.*.clean_acc.diff_vs_standard` (clean-accuracy cost of each strategy);
  - `rq4.*.seen_minus_{group}_gain` (gain on seen minus gain on an unseen group, paired by seed, with seed-noise label), and a second RQ4 table in `rq_answers.md`;
  - `rq5.*.{family,corruption}_{second,mildest}_gap`, `rq5.*.corruption_worst_gap`, and `rq5.{runner_up,third,second_mildest}_{family,corruption}.*`, with a second RQ5 table in `rq_answers.md`;
  - `groups.*` (accuracy and drop per D6 analysis group);
  - `setup.*` (parameter counts, GPU, Python/torch/torchvision versions, commits, dirty runs) and `compute.*` (training time in total and per configuration, Colab-only time ratios against standard), read from `results/runs.csv`; left out with a message if that file is missing;
  - `confusion.*` (two most predicted classes and their share of predictions, seed mean, for the fig 6 conditions), read from `raw/{run_id}_confusion.npz`.
  The module docstring now says it also reads the registry and the confusion files.
- `README.md`: the Phase 00 placeholder is replaced by the generated README.

### Not done / deferred
- `report/report.pdf`: pandoc is not installed on this machine; command above, for the human.
- `validation/phase-10.md` -> a separate validation session.
- Live demo link in the README -> Phase 11.
- Not pushed or merged.

### Known issues
- `--quiet` on `scripts.build_report` is not in CONTRACTS C5 (C5 lists the command without flags; the default behaviour matches it).
- `compute.resnet18.augmix.train_minutes` mixes platforms (the Kaggle run trained faster); the report says so and uses the Colab-only value for time ratios. `KAGGLE_RUNS` in `src/analysis.py` lists the D9 exception by hand.
- README step 3 for a local GPU (`src.train`/`src.evaluate`) and step 4 were not run in this phase; they are the Phase 05-08 commands. Steps 1 and 2 need no GPU.
- The report is about 5,400 words including its tables, near the top of BUILD's range.

### Decisions used
- D1-D9, D11 as stated in report Sections 3-4 and 7; the full DECISIONS change log is disclosed in Limitations.
- D8: verdicts use the seed-noise rule of Phase 09 (mean paired difference vs the larger seed std).

### Ideas for later
- none

## Phase 10: fixes after validation  (2026-11-12, branch phase-10-report, commit: the one titled "phase 10: fix reference [5] pages and validation notes")

### Built
- Fixes for `validation/phase-10.md` (verdict FAIL on V11, plus non-blocking notes), at the human's request ("fix it and then revalidate again").
- `report/report.template.md`:
  - Reference [5] (Geirhos et al. 2018) now gives pp. 7549–7561, from the University of Tübingen bibliography (handle 10900/89433). The NeurIPS proceedings BibTeX gives no pages. The earlier 7538–7550 had no source (V11).
  - Section 3.4, Section 5.5 and Limitations now say that our family assignment differs from [3]. [3] counts brightness as weather (its Table 1, checked this session in the arXiv PDF 1903.12261, page 6); we count it as digital, following PROJECT_OVERVIEW and CONTRACTS.
  - Section 4: the Kaggle sentence now gives the Kaggle run's own training time and says that CPU counts were not recorded. Before, it stated "the Kaggle machine had more CPU" as fact and compared the mean of all three runs, Kaggle included, with the Colab mean.
  - Discussion: "operations unrelated to" became "operations chosen not to overlap with". "Corruption robustness matters … driver assistance, medical imaging …" became "may matter …", without naming domains, because BUILD allows such domains only with "may suggest" and a reference.
  - Limitations: "discarded, not used" became "not used" (DECISIONS 2026-11-10: the partial run was not deleted).
- `README.template.md`: step 2 now says that the sample-image grid is not rebuilt there (it needs CIFAR-10-C).
- `src/analysis.py`: new key `compute.kaggle_train_minutes`, the mean training time of the Kaggle runs, documented in `report_numbers.md`. `tests/test_analysis.py` checks it.

### How to run it
    .venv/bin/python -m src.analysis
    .venv/bin/python -m scripts.make_figures
    .venv/bin/python -m scripts.build_report

### Verified (with output)
- `python -m src.analysis`; `python -m scripts.make_figures`; `python -m scripts.build_report --quiet` -> `wrote report/report.md`, `wrote README.md`. `git status` shows only the edited files plus `report_numbers.json/.md` (one key added); all other tables and figures unchanged.
- `ruff format --check .` -> `98 files already formatted`; `ruff check .` -> `All checks passed!`; `pytest -q` -> `378 passed in 70.82s`.

### Numbers produced
- `compute.kaggle_train_minutes` = 41.7 minutes (source: results/runs.csv, `train_time_s` 2502.27 s of `resnet18-augmix-s2`) [real]

### Changed from earlier phases
- `src/analysis.py`: one new `compute.*` key, which BUILD requires for any number the report needs. No existing key changed.

### Not done / deferred
- Re-validation of Phase 10 -> run after this commit by an independent validator.

### Known issues
- none new

### Decisions used
- D6 and CONTRACTS families (brightness in digital) are unchanged; the report now discloses the difference from [3].

### Ideas for later
- none

## Phase 11: deploy  (2026-11-12, branch phase-11-deploy, commit: the one titled "phase 11: packaging, Docker, Colab badge, results viewer")

**Not published.** The human chose "Not yet" for Hugging Face; nothing was uploaded, no Space or model repo exists, and the README says the viewer is not online yet. Everything else in BUILD is built and tested locally.

### Built
- `Makefile`: `help setup data smoke pilot train-all analyze report app docker-build docker-app check`, each a thin wrapper over the C5 commands. `make help` (the default) lists them with one-line descriptions.
- `Dockerfile` (CPU app): `python:3.11-slim`; torch/torchvision from `requirements-local.txt` (PyTorch CPU index; the build asserts a `+cpu` torch), `requirements.txt` + `app/requirements.txt`; copies `src/ configs/ splits/ results/ app/` and `LICENSE`; user `user` (uid 1000, as Hugging Face Spaces require); `EXPOSE 7860`; health check on `/_stcore/health`; `CMD streamlit run app/streamlit_app.py` on port 7860 with a 5 MB upload limit.
- `Dockerfile.gpu` (optional): `pytorch/pytorch:2.11.0-cuda13.0-cudnn9-runtime` (tag checked on Docker Hub; torch 2.11 + CUDA 13.0 like the Colab runs of 2026-11-04), adds git and `requirements.txt`; the repo is mounted at `/repo` so runs record their commit; `DATA_DIR=/data`, `RUNS_DIR=/runs`, `RESULTS_DIR=/results` as volumes; entry point `bash scripts/run_all.sh`.
- `.dockerignore`: `.git`, `.venv`, `data`, `runs`, `dist`, `results/smoke`, caches, `*.pt`, `.env`, `validation`.
- `app/streamlit_app.py`: two pages (`st.navigation`). **Results**: main table from `results/tables/main_results.csv` with model/strategy filters, the 8 figures with captions taken from the report, the per-corruption table with a family filter, links to the report and repo. **Try it**: a bundled test image (class, then one of 10 images; clean or any of the 15 CIFAR-10-C corruptions at severity 1-5, precomputed) or an upload (centre-cropped, resized to 32x32 in memory; clean or our 4 seen corruptions, noise seeded); the 6 seed-0 models' top-3 with probabilities, a tick or cross against the true class for bundled images; both datasets cited on the page. If no checkpoints are configured, or loading fails, Try it shows a message and Results still works.
- `app/core.py`: the app's logic without Streamlit (tables, samples, upload decoding, checkpoint fetching with sha256 check, prediction through `build_eval_transform` as in `src.evaluate`). Checkpoints come from `LOCAL_CHECKPOINT_DIR`, else `HF_MODEL_REPO` via `hf_hub_download`; loaded once per process (`st.cache_resource`).
- `app/build_assets.py` (`python -m app.build_assets`): the demo asset builder RULES section 3 names. `app/assets/samples.npz`: the first 10 test images of each class (chosen by index only), 0.28 MB. `app/assets/samples_c.npz`: their CIFAR-10-C images, 15 corruptions x 5 severities x 100, 19.86 MB compressed (under the 25 MB limit, so all 5 severities are kept; the builder falls back to severities 1, 3, 5 above it).
- `app/requirements.txt` (`streamlit==1.67.0`, `huggingface_hub==2.4.0`), `app/README.md` (Spaces YAML header: `title`, `emoji`, `colorFrom`, `colorTo`, `sdk: docker`, `app_port: 7860`, `short_description`, `license: mit`; fields checked this session in the Hub docs "Spaces Configuration Reference" and "Licenses"), `app/__init__.py`.
- `scripts/export_checkpoints.py`: the 6 seed-0 `model_final.pt` -> bare state dicts `dist/checkpoints/{run_id}.pt`, `models.json` (run_id, model, strategy, seed, file, n_params, clean accuracy from `results/raw/{run_id}.csv`, sha256, bytes) and a model card `README.md`. Checks status, run id, model, strategy and seed, and a strict `load_state_dict`; writes nothing if any run is missing or wrong.
- `scripts/publish_hf.py`: checks `dist/checkpoints/` against `models.json`; uploads it to the model repo; stages the Space from git-tracked files at HEAD (refuses uncommitted changes), with `app/README.md` as the Space `README.md`; creates the Space (Docker SDK), sets the variable `HF_MODEL_REPO`; downloads every checkpoint back and compares sha256. Token from `HF_TOKEN` only. `--dry-run` needs no token or network. **Not run against the Hub.**
- `notebooks/run_on_colab.ipynb`: first cell has the Open in Colab badge and a table of the 4 modes `smoke`, `pilot`, `matrix`, `analyze` with expected times; new `STAGE` values `smoke` (CPU, run kept in `/content/runs`, results in `results/smoke/`), `pilot` (`cnn-standard-s0`, evaluated into `/content/pilot_results` and printed next to the published numbers) and `analyze` (no Drive, data or GPU). `train`, `evaluate`, `both`, `matrix` are unchanged.
- `docs/CLOUD.md`: paths (a)-(e).
- `README.template.md` -> `README.md`: Open in Colab badge, "not published online yet" line pointing to `make app`, Docker and `docs/CLOUD.md`, a `make` shortcuts paragraph, `app/` in the layout.
- `tests/test_app.py`: 22 tests (export, sha256 checks, local fetch, prediction equals softmax of `model(build_eval_transform(img))`, uploads, no file writes in the app, samples, results tables, publish checks and token refusal, Space staging, `make help`, Dockerfile and `.dockerignore`).

### How to run it
    make setup && make check
    make app                                           # http://localhost:7860
    .venv/bin/python -m scripts.export_checkpoints     # needs the 6 seed-0 model_final.pt in $RUNS_DIR
    LOCAL_CHECKPOINT_DIR=dist/checkpoints make app
    make docker-build && make docker-app CHECKPOINTS=dist/checkpoints
    DATA_DIR=./data .venv/bin/python -m app.build_assets        # rebuild app/assets (needs CIFAR-10-C)
    .venv/bin/python -m scripts.publish_hf --model-repo <user>/<repo> --space <user>/<space> --dry-run

### Verified (with output)
- `make help` -> 12 targets with descriptions (help, setup, data, smoke, pilot, train-all, analyze, report, app, docker-build, docker-app, check).
- `make check` -> ruff format and check clean, `398 passed, 12 warnings in 78.43s` (CPU).
- Clean clone (a snapshot of this working tree, committed in a scratch repo and cloned; `git clone` of GitHub needs the push): `make setup` exit 0 (2m41s; Python 3.12.16, torch `2.11.0+cpu`), `make check` -> `398 passed ... in 87.62s`, `DATA_DIR=<existing data> make smoke` exit 0: `all 20 files match splits/cifar10c_checksums.json`, `extracted files already verified; nothing to download`, `epoch 1/1 ... val_acc 0.1172`, `wrote results/smoke/smoke-cnn-standard-s0.csv` [smoke]. The 2.9 GB download itself was not repeated.
- `make docker-build` -> exit 0 in 8m18s; image `cifar10-robustness-app` **2.75 GB** (`docker image inspect` -> `2753738918 bytes`); inside: Python 3.11.18, torch `2.11.0+cpu`, torchvision `0.26.0+cpu`, streamlit 1.67.0, huggingface_hub 2.4.0, uid 1000 (`user`); `/app` holds `LICENSE app configs requirements*.txt results splits src`; `find /` for `*.pt`, `.git`, CIFAR files, `.env` -> nothing.
- `make docker-app CHECKPOINTS=dist/checkpoints` -> `/_stcore/health` `ok`, `GET /` 200; `/checkpoints` mounted with the 6 files.
- Browser (Chrome, http://localhost:7860, container with the 6 checkpoints): Results shows the filters, the main table (values as in `main_results.md`, e.g. ResNet-18 corruption-aware `92.5% ± 0.2%`, mCE `33.8% ± 0.3%`), all 8 figures (JS check: 8 images decoded, 8 figure headings, 2 tables); Try it shows the 6 models' top-3, e.g. test image 3 (airplane), CIFAR-10-C Gaussian noise severity 1: Simple CNN standard `ship: 89.6%` (wrong), ResNet-18 corruption-aware `airplane: 99.5%`.
- No network (`docker run --network none`), checked with Streamlit's AppTest inside the container: Results `exceptions 0 | tables 2 | figures 8`; Try it without `HF_MODEL_REPO` -> the "not configured" info message; with an unreachable `HF_MODEL_REPO` -> after the Hub client's retries, the warning "could not be loaded ... The Results page still works"; no exception.
- Step 2, predictions (`dist/checkpoints` via the app's code against `model_final.pt` from `runs/` with `build_eval_transform`, `src.evaluate` style): 5 bundled clean samples (test images 3, 35, 77, 29, 76) and 2 CIFAR-10-C samples (Gaussian noise s5 on image 52, fog s3 on image 12) for each of the 6 models: same top-1 everywhere, `max |prob difference| ... 0.00e+00`.
- Exported checkpoints on the full clean test set (CPU, fp32, the app's model, `src.evaluate.predict`): 0.8404, 0.8368, 0.8201, 0.9364, 0.9372, 0.9236 against `models.json` 0.8404, 0.8368, 0.8202, 0.9364, 0.9372, 0.9236: equal except `cnn-corruption_aware-s0`, 1 image of 10,000 apart (the study evaluated on a T4 with AMP).
- `models.json` clean accuracies equal `results/runs.csv` / the raw CSVs for all 6 runs.
- Colab notebook dry run on this machine under IPython (scratch clone, fake `google.colab`, paths moved off `/content`, no pip): `STAGE = "analyze"` -> `ALL CELLS OK` in 19 s, main table printed, `git status --short` empty; `STAGE = "smoke"` -> `ALL CELLS OK` in 78 s with the data on a fake Drive (`all 20 files match`, `wrote results/smoke/smoke-cnn-standard-s0.csv`). **Not run on Colab**: Colab clones from GitHub `main`, which does not have this notebook until it is pushed.
- `RESULTS_DIR=<copy of results/raw + runs.csv> python -m src.analysis` and `scripts.make_figures` -> tables and figures written into that folder; `main_results.csv` byte-identical to the committed one (backs CLOUD.md (d)).
- `python -m scripts.publish_hf --model-repo USER/cifar10-robustness-models --space USER/cifar10-robustness --dry-run` on the committed tree -> `checked` with the 6 sha256 values of the table below, `model repo ...: ['README.md', 'cnn-augmix-s0.pt', ..., 'resnet18-standard-s0.pt']` (8 files), `Space ... : 142 files + README.md`, `Space variable HF_MODEL_REPO=...`. Nothing was sent.
- Leakage guard: `tests/test_no_leakage.py` now also scans `app/`; only `app/build_assets.py` there imports `src.data.cifar10c`.
- Secrets: `git grep -nIE "hf_[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{20,}"` -> empty. Largest new tracked file `app/assets/samples_c.npz` 19,857,685 bytes; no `.pt` or `dist/` tracked.

### Numbers produced
- Docker image size 2.75 GB (2,753,738,918 bytes) (source: `docker image inspect cifar10-robustness-app`) [real].
- Seed-0 checkpoint sha256 and sizes (source: `dist/checkpoints/models.json`, not committed) [real]:

  | file | bytes | sha256 |
  |---|---|---|
  | cnn-standard-s0.pt | 2680765 | 5568342ee72f4e6af987148850f8856b3327cf3f8db0e588f511074571e5ac36 |
  | cnn-augmix-s0.pt | 2680685 | 0b945d76cfb6bbdc6a0f4c2866a7265bf549f450f3d25cb3cc1c7517794368c9 |
  | cnn-corruption_aware-s0.pt | 2681213 | 403c004a477dbd8a24af170f3fb4bc509a52a74f94e32752df4fcacb37162d59 |
  | resnet18-standard-s0.pt | 44772491 | 037a0860bfe9d71d5bad49d84889a777cfda3149fff3a808ae76cf8f0619da27 |
  | resnet18-augmix-s0.pt | 44772235 | 59c33811ecce98db7306cb840dda4ac00e3b8dffcd8bd6baae36f4aeffede130 |
  | resnet18-corruption_aware-s0.pt | 44773579 | 9a5f92bda4b627b4d186308eeeff38146356d9f583f746bc851de04f96415137 |

- No result changed; no training.

### Changed from earlier phases
- `notebooks/run_on_colab.ipynb` (Phase 08): new stages and the first cell, because BUILD section 4 asks for the 4 modes; the old stages behave as before.
- `tests/test_no_leakage.py` (Phase 00/02): `app/build_assets.py` added to the allowed importers and `app/` scanned, as RULES section 3 already allows the demo asset builder.
- `README.template.md` / `README.md` (Phase 10): badge, demo line, `make` paragraph, `app/` in the layout. `report/report.md` unchanged by the rebuild.
- `.venv` (not in git): `app/requirements.txt` installed.

### Not done / deferred
- Publishing (model repo, Space, URLs in README, Hub sha256 check) -> when the human gives the Hugging Face ids and says go; they set `HF_TOKEN` in their own terminal. Then: `scripts.publish_hf` (dry run first), open the Space, check both pages and the upload path on the Space, put the URL into `README.template.md` and rebuild.
- `make docker-app` with `HF_MODEL_REPO` (a real Hub download) -> after publishing.
- CLOUD.md (b) on real Colab -> after this branch is merged and pushed (the notebook clones `main` from GitHub).
- `Dockerfile.gpu` not built or run (no NVIDIA GPU here); CLOUD.md (d) says it is untested.
- Uploads tested through code and unit tests, not through the browser (the browser tool can upload only files shared with the session).
- GitHub (BUILD section 6), suggestions only, nothing changed on GitHub: description "How CIFAR-10 classifiers fail under common corruptions (CIFAR-10-C) and how AugMix and corruption-aware training help: PyTorch code, results, report and demo"; topics `cifar10`, `cifar10-c`, `robustness`, `corruption-robustness`, `augmix`, `data-augmentation`, `pytorch`, `resnet`, `streamlit`, `reproducible-research`.

### Known issues
- **CIFAR-10 has no stated licence.** Its page asks only for a citation of Krizhevsky (2009); CIFAR-10-C is CC BY 4.0 (Zenodo 2535967). The human approved bundling the 100 test images and their CIFAR-10-C versions with both cited on the page and in the model card ("Yes, bundle with citations").
- The image is 2.75 GB because `requirements.txt` (BUILD: install it) brings scipy, scikit-image, scikit-learn, opencv, matplotlib, seaborn, pytest and ruff, most of which the app does not use. A slimmer app-only requirements file would be a change to C1; not done.
- With an unreachable `HF_MODEL_REPO`, Try it waits for the Hub client's retries (about 20 s) before the warning, and again on each page rerun, because failures are not cached.
- `LOCAL_CHECKPOINT_DIR`, `HF_MODEL_REPO` and `HF_TOKEN` (BUILD) and the Makefile variables `PYTHON`, `IMAGE`, `PORT`, `CHECKPOINTS` are not in CONTRACTS C2; `app/core.py`, `app/build_assets.py`, `app/__init__.py` are not listed under `app/` in C1. The human may want them recorded there.
- `make pilot` and `make train-all` in a clone need a new `RESULTS_DIR` (the committed results would make `src.evaluate` refuse and `run_all.sh` skip everything); the help text and CLOUD.md say so.
- Streamlit's file uploads on Hugging Face Spaces sometimes need `--server.enableXsrfProtection=false`; not added, to be checked on the Space.
- Four checkpoints were downloaded from Drive on 2026-11-12 into `runs/{run_id}/model_final.pt` (`cnn-augmix-s0`, `resnet18-standard-s0`, `resnet18-corruption_aware-s0`, `resnet18-augmix-s0`); each one's `run_id` was checked before use. Drive was not changed.

### Decisions used
- D10: Streamlit app, Docker + Makefile, seed-0 checkpoints for a Hugging Face model repo (not yet uploaded), one Colab notebook for any stage.
- D2: the app predicts through `build_eval_transform` (training-set mean/std). D3: final-epoch models. D6/D7: the 15 corruptions for bundled samples, the 4 seen ones for uploads. D9: torch 2.11 CPU in the image, the CUDA base image for the GPU path.

### Ideas for later
- A slimmer `app/requirements-runtime.txt` for the image (needs a C1 change).
- Cache a failed model load for a few minutes so offline reruns of Try it do not wait for the Hub again.
