# Running the study in the cloud or on your own machine

Five ways to see or reproduce the results, from least to most effort. Commands run from the
repository root unless said otherwise.

| Path | For | Needs |
|---|---|---|
| (a) View the results viewer | looking at the results and the demo | a browser (or `make app`) |
| (b) Run on Google Colab | re-running a stage on a free GPU | a Google account |
| (c) Run locally with `make` | everything on your own machine | Python 3.10+ (3.12 used), `make`; Docker for the image |
| (d) Train on a cloud GPU machine | the full matrix without Colab's limits | an NVIDIA GPU, Docker, the NVIDIA container toolkit |
| (e) Host the viewer elsewhere | your own copy of the viewer | any service that runs a Docker image |

## (a) View the results viewer

The viewer has two pages:

- **Results**: the main table (with model and strategy filters), the 8 figures of the report with
  captions, and the per-corruption table, read from the committed files in `results/`.
- **Try it**: choose one of 100 bundled CIFAR-10 test images (10 per class) or upload your own
  image, pick a corruption and a severity, and compare the top-3 predictions of the 6 seed-0
  models. Bundled images use their real CIFAR-10-C versions (all 15 corruptions, severities
  1-5). Uploaded images are cropped to a square, shrunk to 32x32 and can only use our own versions
  of the 4 corruptions seen in corruption-aware training (`src/data/corruptions.py`). Uploads are
  kept in memory only and never saved.

The hosted version (Streamlit Community Cloud, DECISIONS D10) is not online yet. Until then, run
it yourself with (c); `make app` downloads the 6 models from the public model repo
[x-striker-x/cifar10-robustness-models](https://huggingface.co/x-striker-x/cifar10-robustness-models).

## (b) Run on Google Colab

1. Open [`notebooks/run_on_colab.ipynb` in Colab](https://colab.research.google.com/github/aysha0559/Image-Classification-with-ResNet-18/blob/main/notebooks/run_on_colab.ipynb)
   (or use the "Open in Colab" badge in the README).
2. Runtime > Change runtime type > T4 GPU (not needed for `smoke` and `analyze`).
3. In the first code cell set `STAGE`:
   - `smoke`: a 1-epoch Simple CNN on 512 images on the CPU, evaluated on 500 images per
     condition. Checks that everything runs; the numbers mean nothing.
   - `pilot`: trains and evaluates `cnn-standard-s0` and prints its accuracy next to the
     published one.
   - `matrix`: all 18 runs; run it again in new sessions until all are done.
   - `analyze`: rebuilds the tables, figures, report and README from the committed results.
4. Runtime > Run all. Approve the Google Drive prompt (every stage except `analyze` keeps data
   and checkpoints in `MyDrive/cifar10-robustness/`).

The notebook's first cell lists the expected time of each stage. The first session downloads
CIFAR-10 and CIFAR-10-C (about 2.9 GB) and keeps a copy on Drive for later sessions.

## (c) Run locally with `make`

`make help` lists every target. On Linux or macOS:

```bash
git clone https://github.com/aysha0559/Image-Classification-with-ResNet-18.git
cd Image-Classification-with-ResNet-18
make setup      # .venv with the pinned packages and CPU torch (PYTHON=python3.12 by default)
make check      # ruff + pytest, CPU, no download
make app        # the viewer on http://localhost:7860 (Results works with no network)
make analyze    # rebuild tables and figures from results/raw
make report     # rebuild report/report.md and README.md
make smoke      # downloads CIFAR-10 and CIFAR-10-C (about 2.9 GB) first, then a tiny CPU run
```

`make setup` uses `python3.12`; pass another interpreter with `make setup PYTHON=python3.11`.
`make data` (run by `smoke`, `pilot` and `train-all`) downloads into `DATA_DIR` (default
`./data`) and verifies every file against `splits/cifar10c_checksums.json`; it skips files that
are already there and verified.

The Try it page needs the 6 checkpoints. `make app` and `make docker-app` download them from the
public model repo (`HF_MODEL_REPO`, default `x-striker-x/cifar10-robustness-models`, about
140 MB once; `HF_MODEL_REPO=` for Results only). With access to the run folders (`$RUNS_DIR`,
default `./runs`), you can export them and use the local files instead:

```bash
.venv/bin/python -m scripts.export_checkpoints          # writes dist/checkpoints/
make app CHECKPOINTS=dist/checkpoints
```

**Docker** (the same image as (e)):

```bash
make docker-build                               # docker build -t cifar10-robustness-app .
make docker-app                                 # http://localhost:7860, models from the Hub
make docker-app CHECKPOINTS=dist/checkpoints    # mounts local files for Try it instead
make docker-app HF_MODEL_REPO=                  # Results only
```

The image is CPU only (`python:3.11-slim`, torch from the PyTorch CPU wheel index), runs as a
non-root user, and holds no data, runs or checkpoints.

**GPU on your own machine**: `make pilot` and `make train-all` call `src.train`, `src.evaluate`
and `scripts/run_all.sh`. In a clone, `results/raw/` already holds the published results, so
set `RESULTS_DIR` to a new folder, for example `RESULTS_DIR=./results-rerun make train-all`;
otherwise `src.evaluate` refuses to overwrite them and `run_all.sh` skips every run as done.

## (d) Train on a cloud GPU machine with `Dockerfile.gpu`

Any Linux machine with an NVIDIA GPU, Docker and the NVIDIA container toolkit. The image is
the environment only (`pytorch/pytorch:2.11.0-cuda13.0-cudnn9-runtime` plus `requirements.txt`);
the repository is mounted into it, so every run records its git commit.

```bash
git clone https://github.com/aysha0559/Image-Classification-with-ResNet-18.git
cd Image-Classification-with-ResNet-18
docker build -f Dockerfile.gpu -t cifar10-robustness-train .
mkdir -p data runs results-gpu
docker run --rm --gpus all --user "$(id -u):$(id -g)" \
    -v "$PWD:/repo" -v "$PWD/data:/data" -v "$PWD/runs:/runs" -v "$PWD/results-gpu:/results" \
    --entrypoint python cifar10-robustness-train -m scripts.download_cifar10c
docker run --rm --gpus all --user "$(id -u):$(id -g)" \
    -v "$PWD:/repo" -v "$PWD/data:/data" -v "$PWD/runs:/runs" -v "$PWD/results-gpu:/results" \
    cifar10-robustness-train                    # all 18 runs; add --models/--strategies/--seeds
```

The default command is `bash scripts/run_all.sh`: it skips evaluated runs and resumes
unfinished ones, so it can be started again after an interruption. Results land in
`results-gpu/raw/` and `results-gpu/runs.csv`. Analyse them there, next to the published ones
rather than over them: `RESULTS_DIR=results-gpu make analyze` writes `results-gpu/tables/` and
`results-gpu/figures/` (it refuses until all 18 runs are present).

Not tested in this project: the study's runs were made on Colab and Kaggle T4 GPUs
(`results/RUNLOG.md`). Check the GPU first with
`docker run --rm --gpus all --entrypoint nvidia-smi cifar10-robustness-train`.

## (e) Host the viewer elsewhere

**Streamlit Community Cloud** (free; how this project hosts it, DECISIONS D10). You need a
Streamlit Community Cloud account signed in with GitHub and **admin** rights on the repository
(their requirement). At share.streamlit.io: Create app, choose the repository, branch `main` and
the file `app/streamlit_app.py`; under Advanced settings choose Python 3.12 if a version is offered
(it is the version this project was developed with), and put this into
Secrets:

```toml
HF_MODEL_REPO = "x-striker-x/cifar10-robustness-models"
```

Community Cloud installs `app/requirements.txt`, which is complete on its own (CPU torch, the pins
of `requirements.txt`, Streamlit), and runs from the repository root. The app copies
`HF_MODEL_REPO` from the secrets into its environment.

**Any Docker host.**

The `Dockerfile` builds a self-contained viewer that listens on port 7860. Any service that runs
a Docker image and routes HTTP to a container port can host it: build and push the image to a
registry the service can read, run it with port 7860 exposed, and set `HF_MODEL_REPO` in its
environment for the Try it page. No secret is needed to read a public model repo. Streamlit's
health check is at `/_stcore/health`.

**Hugging Face Space.** Docker Spaces on the free CPU need a Hugging Face PRO subscription (the
Hub refused to create one with `402 Payment Required` on 2026-11-12). With one,
`scripts/publish_hf.py` uploads the files the Dockerfile
needs plus `app/README.md` as the Space's `README.md` (its YAML header sets `sdk: docker` and
`app_port: 7860`), and sets the Space variable `HF_MODEL_REPO`:

```bash
.venv/bin/python -m scripts.export_checkpoints
.venv/bin/python -m scripts.publish_hf --model-repo <user>/<model repo> --space <user>/<space> --dry-run
HF_TOKEN=... .venv/bin/python -m scripts.publish_hf --model-repo <user>/<model repo> --space <user>/<space>
```

`--skip-space` uploads only the checkpoints, as was done for
`x-striker-x/cifar10-robustness-models` (all 6 sha256 values on the Hub match `models.json`).

The token is read from the environment only; never commit it.
