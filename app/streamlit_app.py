"""Results viewer and live demo (Phase 11, DECISIONS D10).

    streamlit run app/streamlit_app.py --server.port 7860

Results: tables and figures from the committed files under `results/`; works with no network and
no checkpoints. Try it: the 6 seed-0 models on a bundled CIFAR-10 test image (with its real
CIFAR-10-C versions) or on an uploaded image (with our 4 seen corruptions). Checkpoints come from
`LOCAL_CHECKPOINT_DIR` or the Hugging Face model repo in `HF_MODEL_REPO`.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:  # `streamlit run` puts only app/ on the path
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import streamlit as st  # noqa: E402
from PIL import Image  # noqa: E402

from app import core  # noqa: E402
from src.constants import CLASS_NAMES, CORRUPTION_TO_FAMILY, SEEN_CORRUPTIONS  # noqa: E402

REPO_URL = "https://github.com/aysha0559/Image-Classification-with-ResNet-18"
REPORT_URL = f"{REPO_URL}/blob/main/report/report.md"
SHOWN_SIZE = 160  # pixels; the 32x32 images are enlarged without smoothing

CITATIONS = """
**Data.** The bundled images are 100 images of the CIFAR-10 test set (10 per class) and their
CIFAR-10-C versions.

- Krizhevsky, A. (2009). *Learning Multiple Layers of Features from Tiny Images*. Technical report,
  University of Toronto. (CIFAR-10, https://www.cs.toronto.edu/~kriz/cifar.html)
- Hendrycks, D., & Dietterich, T. (2019). Benchmarking Neural Network Robustness to Common
  Corruptions and Perturbations. *ICLR*. (CIFAR-10-C, https://doi.org/10.5281/zenodo.2535967,
  licence CC BY 4.0)
"""


def pretty(name: str) -> str:
    """'gaussian_noise' -> 'Gaussian noise'."""
    return name.replace("_", " ").capitalize()


def enlarge(img: np.ndarray) -> Image.Image:
    """Return the 32x32 image enlarged with nearest-neighbour, so pixels stay visible."""
    return Image.fromarray(img).resize((SHOWN_SIZE, SHOWN_SIZE), Image.Resampling.NEAREST)


# ---------------------------------------------------------------- Results page


def results_page() -> None:
    """Tables and figures from the committed result files."""
    st.title("CIFAR-10 robustness study: results")
    st.markdown(
        "A Simple CNN and ResNet-18 trained on CIFAR-10 with three strategies (standard, AugMix, "
        "corruption-aware), 3 seeds each, tested on clean CIFAR-10 and on CIFAR-10-C "
        "(15 corruptions x 5 severities). "
        f"Full write-up: [report]({REPORT_URL}). Code: [repository]({REPO_URL})."
    )

    main = core.load_main_results()
    models = st.multiselect(
        "Models",
        list(core.MODEL_LABELS),
        default=list(core.MODEL_LABELS),
        format_func=core.MODEL_LABELS.get,
    )
    strategies = [s for s in core.STRATEGY_ORDER if s in set(main["strategy"])]
    strategies = st.multiselect(
        "Strategies", strategies, default=strategies, format_func=core.STRATEGY_LABELS.get
    )

    st.header("Main results")
    st.caption(
        "Mean ± std over 3 seeds (sample std). mCE (lower is better) uses the Simple CNN with "
        "standard training as the 100% reference. Drop = clean minus mean corrupted accuracy, "
        "in percentage points."
    )
    shown = main[main["model"].isin(models) & main["strategy"].isin(strategies)]
    st.dataframe(core.format_main_results(shown), hide_index=True, width="stretch")

    st.header("Figures")
    for file, title, caption in core.FIGURES:
        st.subheader(title)
        st.image(str(core.FIGURES_DIR / file), caption=caption, width="stretch")

    st.header("Per-corruption results")
    st.caption(
        "Accuracy and corruption error (CE), mean over severities and seeds, ± std over seeds."
    )
    per = core.load_per_corruption()
    families = list(dict.fromkeys(per["Family"]))
    chosen = st.multiselect("Families", families, default=families)
    per = per[
        per["model"].isin(models) & per["strategy"].isin(strategies) & per["Family"].isin(chosen)
    ]
    st.dataframe(per.drop(columns=["model", "strategy"]), hide_index=True, width="stretch")


# ---------------------------------------------------------------- Try it page


def secrets_to_env(names: tuple[str, ...] = ("HF_MODEL_REPO",)) -> None:
    """Copy settings from Streamlit secrets into the environment if not already set there.

    Streamlit Community Cloud takes app settings as secrets; Docker and local runs use
    environment variables and usually have no secrets file at all.
    """
    for name in names:
        if os.environ.get(name, "").strip():
            continue
        try:
            value = st.secrets.get(name)
        except Exception:  # no secrets file: nothing to copy
            continue
        if value:
            os.environ[name] = str(value).strip()


@st.cache_resource(show_spinner="Loading the 6 models ...")
def load_models(kind: str, where: str) -> list[tuple[dict, object]]:
    """Fetch, check and load the seed-0 checkpoints once per process."""
    entries, paths = core.fetch_checkpoints(kind, where)
    return [(e, core.load_model(e, paths[e["run_id"]])) for e in core.sort_entries(entries)]


@st.cache_resource
def load_failures() -> core.FailureMemo:
    """One failure memo per process, shared by all sessions."""
    return core.FailureMemo()


@st.cache_resource
def load_samples() -> core.Samples | None:
    """The bundled samples, or None if they are missing."""
    if not core.SAMPLES_FILE.is_file():
        return None
    return core.load_samples()


def choose_bundled(samples: core.Samples) -> tuple[np.ndarray, int, str, int]:
    """Widgets for a bundled sample; return (shown image, true label, corruption, severity)."""
    cls = st.selectbox("Class", range(len(CLASS_NAMES)), format_func=lambda i: CLASS_NAMES[i])
    options = [i for i in range(len(samples.labels)) if samples.labels[i] == cls]
    i = st.selectbox(
        "Test image",
        options,
        format_func=lambda j: f"CIFAR-10 test image {samples.test_indices[j]}",
    )
    corruptions = [core.CLEAN]
    if samples.corrupted is not None:
        corruptions += list(samples.corruptions)
    corruption = st.selectbox(
        "Corruption (real CIFAR-10-C images)",
        corruptions,
        format_func=lambda c: (
            "none (clean)" if c == core.CLEAN else f"{pretty(c)} ({CORRUPTION_TO_FAMILY[c]})"
        ),
    )
    severity = 0
    if corruption != core.CLEAN:
        severity = st.select_slider("Severity", options=list(samples.severities))
    return (
        core.sample_image(samples, i, corruption, severity),
        int(samples.labels[i]),
        corruption,
        severity,
    )


def choose_upload() -> tuple[np.ndarray, None, str, int] | None:
    """Widgets for an uploaded image; return (shown image, None, corruption, severity) or None."""
    uploaded = st.file_uploader(
        "Image (PNG or JPEG). It is cropped to a square, shrunk to 32x32 and kept in memory only; "
        "nothing is saved.",
        type=["png", "jpg", "jpeg"],
    )
    if uploaded is None:
        return None
    try:
        img = core.prepare_upload(uploaded.getvalue())
    except Exception as err:  # any decoding error: tell the user, do not crash the page
        st.error(f"Could not read that image: {err}")
        return None
    corruption = st.selectbox(
        "Corruption (our own code, src/data/corruptions.py)",
        [core.CLEAN, *SEEN_CORRUPTIONS],
        format_func=lambda c: "none (clean)" if c == core.CLEAN else pretty(c),
    )
    severity = 0
    if corruption != core.CLEAN:
        severity = st.select_slider("Severity", options=[1, 2, 3, 4, 5])
    return core.corrupt_upload(img, corruption, severity), None, corruption, severity


def show_predictions(models: list[tuple[dict, object]], img: np.ndarray, label: int | None) -> None:
    """Top-3 predictions of each model, in a 3-column grid."""
    for row in range(0, len(models), 3):
        for col, (entry, model) in zip(st.columns(3), models[row : row + 3], strict=False):
            with col:
                model_name = core.MODEL_LABELS[entry["model"]]
                name = f"{model_name}, {core.STRATEGY_LABELS[entry['strategy']]}"
                st.markdown(f"**{name}**")
                st.caption(f"clean test accuracy {100 * entry['clean_accuracy']:.1f}% (seed 0)")
                top = core.top_k(core.predict_probs(model, img))
                for rank, (cls, p) in enumerate(top):
                    mark = ""
                    if label is not None and rank == 0:
                        mark = " ✅" if cls == CLASS_NAMES[label] else " ❌"
                    st.write(f"{cls}: {100 * p:.1f}%{mark}")
                    st.progress(min(max(p, 0.0), 1.0))


def try_it_page() -> None:
    """Corrupt an image and compare the 6 seed-0 models."""
    st.title("Try it: corrupt an image, watch the predictions")
    st.markdown(
        "Pick a CIFAR-10 test image or upload your own, choose a corruption and a severity, and "
        "see the top-3 predictions of the 6 seed-0 models. **Bundled images** use the real "
        "CIFAR-10-C version of that test image (precomputed, all 15 corruptions). **Uploaded "
        "images** can only use our own versions of the 4 corruptions seen by corruption-aware "
        "training (Gaussian noise, defocus blur, contrast, JPEG), from `src/data/corruptions.py`; "
        "noise uses a fixed seed."
    )

    secrets_to_env()
    source = core.checkpoint_source()
    if source is None:
        st.info(
            "The demo models are not configured here: set `HF_MODEL_REPO` (a Hugging Face model "
            "repo) or `LOCAL_CHECKPOINT_DIR` (a folder made by `python -m "
            "scripts.export_checkpoints`). The Results page works without them."
        )
        st.markdown(CITATIONS)
        return
    failures = load_failures()
    message = failures.recent(source, time.monotonic())
    if message is None:
        try:
            models = load_models(*source)
            failures.clear(source)
        except Exception as err:  # network down, missing file, bad checksum: degrade, no crash
            reason = str(err).strip().splitlines()[0][:200] if str(err).strip() else ""
            message = f"{type(err).__name__}: {reason}"
            failures.record(source, time.monotonic(), message)
    if message is not None:
        minutes = round(failures.retry_after_s / 60)
        st.warning(
            f"The demo models could not be loaded from {source[1]} ({message}). The Results "
            f"page still works. Loading is tried again {minutes} minutes after the failure."
        )
        st.markdown(CITATIONS)
        return

    samples = load_samples()
    choices = ["Bundled CIFAR-10 test image", "Upload an image"]
    if samples is None:
        st.warning("The bundled samples (app/assets/samples.npz) are missing; only uploads work.")
        choices = choices[1:]
    mode = st.radio("Image", choices, horizontal=True)
    picked = choose_bundled(samples) if mode == choices[0] and samples else choose_upload()
    if picked is None:
        st.markdown(CITATIONS)
        return
    img, label, corruption, severity = picked

    left, right = st.columns([1, 3])
    with left:
        what = "clean" if corruption == core.CLEAN else f"{pretty(corruption)}, severity {severity}"
        st.image(enlarge(img), caption=f"{what} (32x32, enlarged)")
        if label is not None:
            st.caption(f"true class: **{CLASS_NAMES[label]}**")
    with right:
        show_predictions(models, img, label)
    st.markdown(CITATIONS)


def main() -> None:
    """Page setup and navigation."""
    st.set_page_config(page_title="CIFAR-10 robustness study", layout="wide")
    pages = [
        st.Page(results_page, title="Results", url_path="results", default=True),
        st.Page(try_it_page, title="Try it", url_path="try-it"),
    ]
    st.navigation(pages).run()


if __name__ == "__main__":  # streamlit runs the script as __main__; tests import it
    main()
