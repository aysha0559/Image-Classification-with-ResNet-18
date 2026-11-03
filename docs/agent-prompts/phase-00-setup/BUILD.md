# Phase 00 BUILD: Project setup and guardrails

You are a coding agent building Phase 00 of a research codebase. Work carefully and only inside this phase's scope.

## Read first (in this order)
1. `docs/agent-prompts/RULES.md` (all of it)
2. `docs/agent-prompts/DECISIONS.md` (check D1-D4, D9 are no longer marked `CONFIRM`, except the ResNet-18 epochs row, which stays open until Phase 07; if anything else is, stop and ask me)
3. `docs/agent-prompts/CONTRACTS.md` (C1, C2, C3, C4, C5)
4. `docs/PROJECT_OVERVIEW.md` sections 1, 5, 12, 13

## Goal
A clean, runnable repository skeleton with the configuration system, seeding, device selection, constants, and the guardrails (git, ignore rules, lint, sanity tests, Colab setup notebook) that every later phase builds on. No data loading, models or training yet.

## In scope
- `git init` (if not already a repo), default branch `main`, then branch `phase-00-setup`.
- The folder layout of CONTRACTS C1 (create folders; empty modules may contain only a docstring saying which phase fills them).
- `.gitignore` covering `data/ runs/ dist/ *.pt *.pth .env __pycache__/ .ipynb_checkpoints/ .venv/ results/smoke/ .pytest_cache/ .ruff_cache/`.
- `pyproject.toml` with ruff (line length 100) and pytest settings.
- `requirements.txt` (pinned: numpy, pandas, scikit-learn, matplotlib, seaborn, pyyaml, scipy, scikit-image, opencv-python-headless, Pillow, tqdm, pytest, ruff, nbformat). Do not add `wand`/ImageMagick: none of our training corruptions need it. Use versions that install on Python 3.10-3.12. Pin exact versions you actually install and verify.
- `requirements-local.txt` with torch and torchvision pins (D9). If you cannot learn Colab's current versions, pin the latest stable pair you can install, and write in `PROGRESS.md` that the human must compare with Colab in Phase 07.
- `src/constants.py`: class names, the 15 corruptions, 4 extras, families, seen and unseen lists and the three analysis groups (D6), exactly as C3 and D6.
- `src/config.py`: load YAML, validate against C4 (unknown key or wrong type -> clear `ValueError`), apply CLI overrides (`seed`, `epochs`, `subset`, `run_id`), build `run_id` per C3.
- `src/utils.py`: `set_seed(seed)` (python, numpy, torch, cuda, cudnn flags per D4), `get_device(pref)`, `get_git_commit()` and `is_git_dirty()`, `atomic_save(obj, path)`, `env_path(name, default)` for C2 variables, a simple `Timer`.
- `configs/`: the 6 main configs + 2 optional S0 configs + `smoke.yaml` (simple_cnn, standard, epochs 1, subset 512, num_workers 0, device cpu), all valid per C4 with values from D4/D5. ResNet epochs = the D4 value.
- `tests/test_config.py`, `tests/test_utils.py`, `tests/test_constants.py`: valid configs load; a bad key fails; run_id format; same seed gives same random numbers (python, numpy, torch); constants have 15 + 4 corruptions, 10 classes, every corruption in exactly one family, seen and unseen disjoint and together equal the 15.
- `notebooks/00_colab_setup.ipynb`: mount Drive, clone or pull the repo (repo URL as a variable the human fills in), `pip install -r requirements.txt` (not torch), set `DATA_DIR`/`RUNS_DIR`, print torch, torchvision and GPU info. No logic beyond calling repo code and shell commands.
- `CLAUDE.md` at the root: 5-10 lines telling any agent to read `docs/agent-prompts/RULES.md`, `DECISIONS.md`, `CONTRACTS.md` and `PROGRESS.md` before working, and to work one phase at a time.
- `PROGRESS.md` with a header and your Phase 00 entry.
- `README.md`: a short placeholder (title, one paragraph, "Status: in development", link to `docs/PROJECT_OVERVIEW.md`). The full README is Phase 10.
- `LICENSE` (MIT, copyright holder left as `<name>` for the human to fill; ask me if I want a name).

## Out of scope
Data downloading or loading, transforms, models, training, evaluation, analysis, Docker, the app.

## Steps
1. Create a virtual environment (`python -m venv .venv`), install `requirements-local.txt` then `requirements.txt`. Record exact Python and package versions.
2. Build the files above.
3. Run `ruff format --check .`, `ruff check .`, `pytest -q`.
4. Run `python -c "from src.config import load_config; c = load_config('configs/resnet18_augmix.yaml', seed=2); print(c['run_id'])"` and confirm it prints `resnet18-augmix-s2`.
5. Append your Phase 00 entry to `PROGRESS.md` (template in `docs/agent-prompts/templates/PROGRESS-entry.md`). Commit `phase 00: project skeleton, config, seeding, guardrails`.

## Done when
- [ ] Every folder in C1 exists; empty modules say which phase fills them.
- [ ] `ruff format --check .` and `ruff check .` pass.
- [ ] `pytest -q` passes on CPU in under 2 minutes.
- [ ] All 9 configs load and validate; a config with an unknown key fails with a readable message.
- [ ] `run_id` matches C3 for all 6 main configs and 3 seeds.
- [ ] `src/constants.py` matches C3 and D6 exactly.
- [ ] `.gitignore` excludes data, runs and checkpoints (`git check-ignore data/x runs/x a.pt` prints all three).
- [ ] The Colab notebook contains no training, model or metric logic.
- [ ] `PROGRESS.md` entry written; work committed on `phase-00-setup`.

## Stop and ask me if
- Any decision you need is still `CONFIRM`.
- A package will not install on Python 3.10-3.12.
