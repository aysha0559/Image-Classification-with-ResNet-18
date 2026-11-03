# Phase 11 BUILD: Packaging, cloud reproduction, and a public results viewer

## Read first
1. `docs/agent-prompts/RULES.md` (section 5: accounts, tokens and anything public need the human)
2. `docs/agent-prompts/DECISIONS.md` D10 (stop if `CONFIRM`)
3. `docs/agent-prompts/CONTRACTS.md` C1, C2, C5
4. `README.md`, `PROGRESS.md`; `validation/phase-10.md` must be PASS or PASS WITH NOTES

Branch: `phase-11-deploy` from `main`.

## Goal
Three easy ways for anyone to see or reproduce the results, from least to most effort:

| Who | Path | Effort |
|---|---|---|
| A reviewer who just wants to look | Hugging Face Space: results tables, figures, and a live demo | Click a link |
| Someone who wants to re-run on free GPU | "Open in Colab" badge -> `notebooks/run_on_colab.ipynb` | A Google account |
| Someone on their own machine or any cloud VM | `make` targets, or Docker | Docker installed |

This is a showcase of a research study, not a product: no user accounts, no database, no uploads stored.

## In scope
### 1. `Makefile` (root)
Targets, each a thin wrapper over C5 commands: `setup` (venv + install), `data` (download + verify CIFAR-10-C), `smoke` (smoke train + eval, CPU), `pilot`, `train-all` (`run_all.sh`), `analyze` (analysis + figures), `report`, `app` (run the viewer locally), `docker-build`, `docker-app`, `check` (ruff + pytest). `make help` lists them with one-line descriptions.

### 2. Docker
- `Dockerfile`: `python:3.11-slim` (match the version you verified works), CPU-only torch from the PyTorch CPU wheel index, installs `requirements.txt` + `app/requirements.txt`, copies `src/ app/ results/ configs/ splits/`, non-root user, `EXPOSE 7860`, default command runs the Streamlit app on port 7860. No data, no checkpoints baked in.
- `Dockerfile.gpu` (optional, for training on a cloud GPU VM): based on an official `pytorch/pytorch` CUDA runtime image; entry point is `bash scripts/run_all.sh`; `DATA_DIR` and `RUNS_DIR` as mounted volumes. Document `docker run --gpus all -v ...` in `docs/CLOUD.md`.
- `.dockerignore` (data, runs, .venv, .git, notebooks checkpoints).

### 3. Results viewer and demo: `app/`
- `app/streamlit_app.py` + `app/requirements.txt` (pinned `streamlit`, `huggingface_hub`).
- Page "Results": the main table (from `results/tables/main_results.csv`), filters by model/strategy, the 8 figures with captions, per-corruption table, links to the report and repo. Reads only committed files under `results/`.
- Page "Try it": choose a CIFAR-10 test image from a small bundled set (100 test images saved as `app/assets/samples.npz`, 10 per class; check the CIFAR-10 and CIFAR-10-C terms of use before bundling, tell the human what they say, and cite both on the page) **or** upload an image (resized to 32x32, never stored). Pick a corruption and severity; the app shows the corrupted image and the predictions (top-3 with probabilities) of all 6 seed-0 models side by side.
  - Where the corrupted image comes from: for bundled samples, the real CIFAR-10-C version of that test image, precomputed (precomputed into `app/assets/samples_c.npz` for 15 corruptions x 5 severities x the 100 samples, about 23 MB uint8; check size and use 3 severities if over 25 MB). For uploaded images, only our 4 corruptions from `src/data/corruptions.py`. Say this on the page.
  - Models load on CPU from a Hugging Face model repo (D10) with `huggingface_hub.hf_hub_download`, cached; repo id from the env var `HF_MODEL_REPO`. If unset or unreachable, the Results page still works and Try it shows a clear message.
- `scripts/export_checkpoints.py`: collects the 6 seed-0 `model_final.pt` files, saves state dicts only (`.pt`, no optimizer) plus `models.json` (run_id, model, strategy, clean accuracy from the CSV, sha256 of each file) into `dist/checkpoints/`.
- `scripts/publish_hf.py`: uploads `dist/checkpoints/` to the model repo and the app to the Space. Reads the token from `HF_TOKEN` only. **Runs only when the human says so**, after they have created the account/token and named the repo and Space ids.
- Space metadata: `app/README.md` with the Hugging Face Spaces YAML header (`sdk: docker`, `app_port: 7860`, title, short description, licence). Check the current Hugging Face Spaces docs for the exact header fields; do not guess.

### 4. One-click Colab
- An "Open in Colab" badge in `README.md` pointing to `notebooks/run_on_colab.ipynb` on GitHub (needs the repo URL from the human).
- The notebook's first markdown cell explains the 4 modes (`smoke`, `pilot`, `matrix`, `analyze`) and expected times from `results/pilot_report.md`.

### 5. `docs/CLOUD.md`
Short, step-by-step: (a) view the Space; (b) run on Colab; (c) run locally with `make`; (d) train on any cloud GPU VM with `Dockerfile.gpu` (generic: needs an NVIDIA GPU, Docker and the NVIDIA container toolkit); (e) host the viewer elsewhere (any service that runs a Docker image on port 7860). No provider-specific prices or claims you have not checked.

### 6. GitHub (only with the human's go-ahead)
The human creates the public repo (or tells you to with `gh`). Add repo topics and a short description suggestion in `PROGRESS.md`. Large files: none over 50 MB in git; checkpoints live on Hugging Face, not in git.

## Out of scope
Training anything new, changing results, user accounts, analytics, storing uploads, paid hosting.

## Steps
1. Build Makefile, Docker, app, scripts, docs. Run `make check`, `make docker-build`, `make docker-app` and open `http://localhost:7860`; check both pages; with `HF_MODEL_REPO` unset, Results works and Try it degrades cleanly.
2. Test Try it locally by pointing the app at a local folder of exported checkpoints (env var `LOCAL_CHECKPOINT_DIR`) before anything is uploaded. Confirm the app's prediction for 5 bundled clean samples equals `src.evaluate`-style predictions from the same checkpoint.
3. Ask the human for: GitHub repo URL; Hugging Face username, model repo id, Space id; whether to publish now. Publish only after a yes.
4. After publishing: open the Space URL, check both pages, record the URLs in `README.md` (via the template) and `PROGRESS.md`.
5. Progress entry; commit `phase 11: packaging, Docker, Colab badge, results viewer`.

## Done when
- [ ] `make help` lists all targets; `make check` and `make smoke` pass on a clean clone (CPU).
- [ ] `docker build` succeeds; the container serves the app on 7860; image size recorded.
- [ ] Results page shows tables and all 8 figures from committed files only.
- [ ] Try it shows 6 models' predictions; matches offline predictions on 5 samples.
- [ ] App works (Results) with no network and no checkpoints.
- [ ] No token or secret in git (`git grep -nIE "hf_[A-Za-z0-9]{20,}"` empty).
- [ ] If published (human said yes): Space live, URLs in README; checkpoints' sha256 on the Hub match `models.json`.
- [ ] `docs/CLOUD.md` paths (a)-(e) written and (b), (c) tested.
