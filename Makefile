# Thin wrappers over the commands of CONTRACTS C5 (Phase 11). `make help` lists the targets.
# Environment variables of CONTRACTS C2 (DATA_DIR, RUNS_DIR, RESULTS_DIR) pass through unchanged.

PYTHON ?= python3.12
VENV ?= .venv
PY := $(VENV)/bin/python
IMAGE ?= cifar10-robustness-app
PORT ?= 7860
# docker-app: a folder from scripts.export_checkpoints to mount for the Try it page (optional)
CHECKPOINTS ?=
SMOKE_RUN := $(or $(RUNS_DIR),./runs)/smoke-cnn-standard-s0

.DEFAULT_GOAL := help
.PHONY: help setup data smoke pilot train-all analyze report app docker-build docker-app check

help: ## List the targets
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-13s %s\n", $$1, $$2}'

setup: ## Create .venv (PYTHON=python3.12) and install the pinned packages, CPU torch, app packages
	$(PYTHON) -m venv $(VENV)
	$(PY) -m pip install -r requirements-local.txt -r requirements.txt -r app/requirements.txt

data: ## Download CIFAR-10 and CIFAR-10-C (about 2.9 GB) into DATA_DIR and verify the checksums
	$(PY) -c "from src.data.cifar10 import get_test_dataset; from src.utils import env_path; get_test_dataset(str(env_path('DATA_DIR')))"
	$(PY) -m scripts.download_cifar10c

smoke: data ## Smoke test on CPU: 1-epoch Simple CNN on 512 images, evaluated on 500 images per condition
	$(PY) -m src.train --config configs/smoke.yaml --seed 0
	$(PY) -m src.evaluate --run-dir $(SMOKE_RUN) --subset 500 --force

pilot: data ## Train and evaluate cnn-standard-s0 (GPU); use a new RESULTS_DIR in a clone
	$(PY) -m src.train --config configs/cnn_standard.yaml --seed 0
	$(PY) -m src.evaluate --run-dir $(or $(RUNS_DIR),./runs)/cnn-standard-s0

train-all: data ## Train and evaluate all 18 runs with scripts/run_all.sh (GPU); use a new RESULTS_DIR
	PYTHON=$(PY) bash scripts/run_all.sh

analyze: ## Rebuild results/tables and figures from results/raw (CPU, no data needed)
	$(PY) -m src.analysis
	$(PY) -m scripts.make_figures

report: ## Rebuild report/report.md and README.md from report_numbers.json
	$(PY) -m scripts.build_report

app: ## Run the results viewer locally on http://localhost:7860 (PORT=...)
	$(PY) -m streamlit run app/streamlit_app.py --server.port=$(PORT) --server.headless=true \
		--server.maxUploadSize=5 --browser.gatherUsageStats=false

docker-build: ## Build the CPU app image (IMAGE=cifar10-robustness-app)
	docker build -t $(IMAGE) .

docker-app: ## Run the app image on port 7860; Try it needs HF_MODEL_REPO=... or CHECKPOINTS=dir
	docker run --rm -p $(PORT):7860 \
		$(if $(HF_MODEL_REPO),-e HF_MODEL_REPO=$(HF_MODEL_REPO)) \
		$(if $(CHECKPOINTS),-v $(abspath $(CHECKPOINTS)):/checkpoints:ro -e LOCAL_CHECKPOINT_DIR=/checkpoints) \
		$(IMAGE)

check: ## Lint (ruff format --check, ruff check) and run the tests (pytest -q)
	$(VENV)/bin/ruff format --check .
	$(VENV)/bin/ruff check .
	$(VENV)/bin/pytest -q
