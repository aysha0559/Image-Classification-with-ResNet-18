# Validation: Phase 00 setup

- Date: 2026-11-03
- Branch / commit checked: phase-00-setup @ 5d41c930e6d4a50551b7a3518805dec73c2d0cc3   (from `git rev-parse HEAD`)
- Working tree clean before checking: yes
- Environment: Python 3.12.16, torch 2.14.1+cpu (torchvision 0.29.2+cpu), device cpu (fresh venv outside the repo, in the session scratchpad)
- Verdict: PASS

## Summary
All 18 checks were re-run in a fresh virtual environment created outside the repo. Requirements install from scratch; lint, format and the 33 tests pass on CPU in about 2 s; all 9 configs validate; the two bad configs are rejected with readable `ValueError`s; run ids, constants and config values match CONTRACTS C3/C4 and DECISIONS D4-D6 item by item; seeding repeats exactly. Nothing failed. The notes below are non-blocking (mainly the Colab torch version, which is still UNKNOWN and must be checked in Phase 07).

## Checks
| ID | Check | Must/Should | Result | Evidence |
|---|---|---|---|---|
| V1 | On branch `phase-00-setup`, tree clean | Must | PASS | `git status --short` -> (empty); `git branch --show-current` -> `phase-00-setup`; `git log --oneline -5` -> `5d41c93 phase 00: anonymous license holder, repo URL in Colab notebook`, `dbbfda2 phase 00: project skeleton, config, seeding, guardrails`, `9e94075 docs: research plan and agent prompts` |
| V2 | Requirements install from scratch | Must | PASS | `python3.12 -m venv <scratchpad>/val-venv && pip install -r requirements-local.txt -r requirements.txt` -> exit 0; `pip freeze` -> `torch==2.14.1+cpu`, `torchvision==0.29.2+cpu`, `numpy==2.2.6`, `pillow==12.3.0`, `pytest==9.2.0`, `ruff==0.17.3` |
| V3 | Every package pinned with `==` | Must | PASS | `requirements.txt`: 14 packages, all `==` (matplotlib, nbformat, numpy, opencv-python-headless, pandas, Pillow, pytest, PyYAML, ruff, scikit-image, scikit-learn, scipy, seaborn, tqdm). `requirements-local.txt`: `torch==2.14.1`, `torchvision==0.29.2`, plus an `--extra-index-url` line (an option, not a package) |
| V4 | Folder layout matches C1 | Must | PASS | `find . -path ./.git -prune -o -path ./.venv -prune -o -print \| sort`: every C1 folder exists (`app/ configs/ docs/agent-prompts/ notebooks/ splits/ src/data/ src/models/ scripts/ results/{raw,tables,figures}/ report/ validation/ tests/`); every C1 `src/` and `scripts/` module exists; all 9 configs present. Files C1 assigns to later phases (Makefile, Dockerfiles, README.template.md, docs/CLOUD.md, splits/*.json, other notebooks, results/runs.csv, report template) are absent, as expected |
| V5 | Lint and format | Must | PASS | `ruff format --check . && ruff check .` -> `62 files already formatted`, `All checks passed!` |
| V6 | Tests pass on CPU under 2 min | Must | PASS | `time pytest -q` -> `33 passed in 1.54s`, `real 0m2.256s` |
| V7 | All 9 configs validate | Must | PASS | `load_config` over `configs/*.yaml` -> all 9 `OK`; run ids `cnn-augmix-s0 ... resnet18-standard-s0`, `smoke.yaml` -> `smoke-cnn-standard-s0` |
| V8 | Bad config rejected clearly | Must | PASS | temp YAML (outside repo) with `epoch:` -> `ValueError: .../bad_epoch.yaml: unknown config key(s) ['epoch']; allowed here: amp, batch_size, device, epochs, ...`; with `lr: "fast"` -> `ValueError: .../bad_lr.yaml: config key 'optimizer.lr' must be float, got str ('fast')` |
| V9 | run_id format | Must | PASS | 6 main configs x seeds 0,1,2 -> `cnn-standard-s{0,1,2}`, `cnn-augmix-s{0,1,2}`, `cnn-corruption_aware-s{0,1,2}`, `resnet18-standard-s{0,1,2}`, `resnet18-augmix-s{0,1,2}`, `resnet18-corruption_aware-s{0,1,2}`; matches C3 `{short_model}-{strategy}-s{seed}` |
| V10 | Constants exact | Must | PASS | script asserting `CORRUPTIONS` (15, C3 order), `EXTRA_CORRUPTIONS` (4), `CLASS_NAMES` (10, C3 order), `FAMILIES` (all 5), `MODELS`, `STRATEGIES`, `MODEL_SHORT_NAMES`, `SEEN_CORRUPTIONS`, `UNSEEN_CORRUPTIONS`, `ANALYSIS_GROUPS` against C3/D6 -> `constants match C3/D6; seen&unseen disjoint; union=15; groups ok` |
| V11 | Config values match decisions | Must | PASS | every config: optimizer `{sgd, lr 0.1, momentum 0.9, nesterov true, wd 0.0005}`, scheduler cosine, batch 128, amp true, val_size 5000, split_seed 42, AugMix `{3, 3, -1, 1.0, false}`, S3 `{p 0.5, 4 seen corruptions, severities 1-5}` -> `params ok` for all 9; epochs CNN 30, ResNet 40 (proposed D4), smoke 1. `smoke.yaml` also has subset 512, num_workers 0, device cpu per BUILD |
| V12 | Seeding works | Must | PASS | `set_seed(0)` then `random.random()`, `np.random.rand(3)`, `torch.rand(3)`, twice -> `identical: True`; `cudnn.deterministic True`, `cudnn.benchmark False` (D4) |
| V13 | Ignore rules | Must | PASS | `git check-ignore -v data/x runs/x a.pt .env` -> `.gitignore:1:data/`, `.gitignore:2:runs/`, `.gitignore:4:*.pt`, `.gitignore:6:.env`. `.gitignore` also covers every pattern BUILD lists |
| V14 | No hard-coded machine paths | Must | PASS | `grep -rnE "/home/\|/content/\|C:\\\\" src scripts configs` -> no output, exit 1 |
| V15 | Colab notebook has no logic | Must | PASS | 7 cells, no saved outputs: markdown intro; `REPO_URL`/`BRANCH`/`REPO_DIR`; `drive.mount`; git clone or fetch/checkout/pull; `pip install -q -r requirements.txt` (no torch); set and create `DATA_DIR`/`RUNS_DIR`; print python/torch/torchvision/CUDA/GPU + `nvidia-smi` |
| V16 | `CLAUDE.md` points to the 4 agent files | Should | PASS | lists `RULES.md`, `DECISIONS.md`, `CONTRACTS.md`, `PROGRESS.md` in order, plus one-phase-at-a-time instruction |
| V17 | Nothing out of scope built | Should | PASS | every later-phase module in `src/` and `scripts/*.py` is a 1-line docstring naming its phase; `download_cifar10c.sh` and `run_all.sh` are 5-line stubs that print "Not implemented"; no data loading, model or training code |
| V18 | `PROGRESS.md` entry complete and true | Should | PASS | all claims checked (see Notes for one small count difference) |

## Failures (if any)
None.

## Notes (non-blocking)
- Colab's torch/torchvision versions are still UNKNOWN (PROGRESS says so honestly). D9 requires the local pins to match Colab's major.minor; this must be checked in Phase 07, together with the possible numpy 2.2.6 conflict on Colab.
- The notebook's `BRANCH = "main"`, and `main` holds only `docs/` until `phase-00-setup` is merged (noted in PROGRESS). After the merge this is correct.
- `load_config` also accepts an optional top-level `run_id` key in the YAML (str or null), which is not in the C4 schema. Harmless and used for the `--run-id` override, but it is one key beyond C4.
- `ruff format --check .` reports 62 files here, not the 61 in PROGRESS; this ruff version also counts the Markdown files under `docs/`. Either way the result is the same: all formatted.
- Wheels for Python 3.10 and 3.11 (manylinux x86_64) also resolve: `pip download --only-binary=:all: --python-version 3.10|3.11 ... -r requirements.txt -r requirements-local.txt` -> exit 0 for both.
- The tests are 16 test functions, parametrized to 33 cases.
- `origin/main` and `origin/phase-00-setup` match the local branches (`git branch -vv -a`); not merged, as PROGRESS says.

## Claims in PROGRESS.md that could not be confirmed
- `ruff format --check .` -> `61 files already formatted` -> saw `62 files already formatted` at HEAD 5d41c93 (likely counted before a later edit or with a different file set; not a problem).
