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
- **VALIDATE V8 expects mean pixel difference to grow from severity 1 to 5 for at least 13 of 15 corruptions; the real (checksum-verified) data gives 11/15.** The exceptions are glass_blur, motion_blur (s3 13.10 vs s4 13.08), frost and elastic_transform. This is the published data, not the slicing: the 11 others increase cleanly, the labels align for every severity, and the grid shows the same pattern. Pixel difference is a weak severity proxy for frost (the image is blended with a frost texture at shifting weights) and elastic_transform (CIFAR-10-C's parameters for it do not make the pixel change monotonic). The human should decide whether V8's threshold should be 11/15 or the check should be reworded (for example "severity 1 and 5 differ, and ordering is monotonic for the noise, defocus/zoom blur, snow, fog, brightness, contrast, pixelate and jpeg rows").
- The local venv has no IPython, so notebook 01 shows the grid with matplotlib rather than `IPython.display`.
- On free Drive (15 GB), the extracted CIFAR-10-C takes about 2.9 GB; the Colab cell does not keep the archive on Drive.

### Decisions used
- D7: 15 standard corruptions x 5 severities in the grid; the 4 extras are accepted by the loader but not in the grid or the defaults.
- RULES section 3: CIFAR-10-C only through `src/data/cifar10c.py`; imported only by `scripts/make_sample_grid.py` so far (and `src/evaluate.py` later), guarded by `tests/test_no_leakage.py`.

### Ideas for later
- none
