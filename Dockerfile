# Results viewer and demo (Phase 11): CPU only, serves the Streamlit app on port 7860.
#   docker build -t cifar10-robustness-app .
#   docker run --rm -p 7860:7860 [-e HF_MODEL_REPO=USER/NAME] cifar10-robustness-app
# No data, runs or checkpoints are copied in (.dockerignore); the models are downloaded at start
# from HF_MODEL_REPO, or read from a mounted folder named by LOCAL_CHECKPOINT_DIR.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Hugging Face Spaces run the container as user id 1000.
RUN useradd -m -u 1000 user

WORKDIR /app

# torch/torchvision pins from requirements-local.txt, whose extra index is PyTorch's CPU wheels.
COPY requirements-local.txt requirements.txt ./
COPY app/requirements.txt app/requirements.txt
RUN pip install -r requirements-local.txt -r requirements.txt -r app/requirements.txt \
    && python -c "import torch; assert torch.__version__.endswith('+cpu'), torch.__version__"

COPY --chown=user src/ src/
COPY --chown=user configs/ configs/
COPY --chown=user splits/ splits/
COPY --chown=user results/ results/
COPY --chown=user app/ app/
COPY --chown=user LICENSE ./

USER user
ENV HOME=/home/user
EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/_stcore/health', timeout=4)"

CMD ["streamlit", "run", "app/streamlit_app.py", \
     "--server.port=7860", "--server.address=0.0.0.0", "--server.headless=true", \
     "--server.maxUploadSize=5", "--browser.gatherUsageStats=false"]
