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
