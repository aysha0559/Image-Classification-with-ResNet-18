"""Phase 11: the demo's logic, checkpoint export, publish dry run, Makefile and Docker files.

No network, no CIFAR-10-C, no real checkpoints: models are untrained `build_model` instances.
"""

import json
import re
import subprocess
from io import BytesIO
from pathlib import Path

import numpy as np
import PIL
import pytest
import torch
import torch.nn.functional as F
from PIL import Image

from app import core
from scripts import export_checkpoints, publish_hf
from src.config import load_config
from src.constants import CLASS_NAMES, CORRUPTIONS, SEEN_CORRUPTIONS
from src.data.transforms import build_eval_transform
from src.models import build_model

REPO = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------- fixtures


def write_finished_run(runs_dir: Path, run_id: str, model: str, strategy: str, seed: int = 0):
    """A finished run folder with an untrained model (CONTRACTS C6 layout, minimal fields)."""
    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True)
    short = "cnn" if model == "simple_cnn" else "resnet18"
    cfg = load_config(f"configs/{short}_{strategy}.yaml", seed=seed)
    torch.manual_seed(len(run_id))
    net = build_model(model)
    torch.save(
        {"state_dict": net.state_dict(), "config": cfg, "run_id": run_id},
        run_dir / "model_final.pt",
    )
    (run_dir / "run_meta.json").write_text(json.dumps({"run_id": run_id, "status": "finished"}))
    return net


def write_clean_csv(results_dir: Path, run_id: str, acc: float):
    raw = results_dir / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    (raw / f"{run_id}.csv").write_text(
        f"run_id,corruption,severity,accuracy\n{run_id},clean,0,{acc}\n{run_id},fog,1,0.5\n"
    )


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    """Export 6 fake runs; return (out folder, {run_id: model}, accuracies)."""
    root = tmp_path_factory.mktemp("export")
    nets, accs = {}, {}
    for i, (run_id, model, strategy) in enumerate(export_checkpoints.run_ids(0)):
        nets[run_id] = write_finished_run(root / "runs", run_id, model, strategy)
        accs[run_id] = 0.8 + i / 100
        write_clean_csv(root / "results", run_id, accs[run_id])
    out = root / "dist" / "checkpoints"
    export_checkpoints.export(root / "runs", root / "results", out)
    return out, nets, accs


# ---------------------------------------------------------------- export


def test_export_writes_state_dicts_and_models_json(exported):
    out, nets, accs = exported
    meta = json.loads((out / "models.json").read_text())
    entries = meta["models"]
    assert [e["run_id"] for e in entries] == [r for r, _, _ in export_checkpoints.run_ids(0)]
    assert sorted(p.name for p in out.iterdir()) == sorted(
        ["models.json", "README.md", *[e["file"] for e in entries]]
    )
    for e in entries:
        state = torch.load(out / e["file"], map_location="cpu", weights_only=True)
        assert set(state) == set(nets[e["run_id"]].state_dict())  # a bare state dict, no optimizer
        assert e["sha256"] == core.sha256_file(out / e["file"])
        assert e["clean_accuracy"] == pytest.approx(accs[e["run_id"]])
    assert "license: mit" in (out / "README.md").read_text()


def test_export_refuses_a_wrong_run(tmp_path):
    for run_id, model, strategy in export_checkpoints.run_ids(0):
        # cnn-augmix-s0's folder holds the standard model: must be refused
        write_finished_run(
            tmp_path / "runs", run_id, model, "standard" if run_id == "cnn-augmix-s0" else strategy
        )
        write_clean_csv(tmp_path / "results", run_id, 0.8)
    with pytest.raises(ValueError, match="cnn-augmix-s0"):
        export_checkpoints.export(tmp_path / "runs", tmp_path / "results", tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_export_refuses_a_missing_run(tmp_path):
    with pytest.raises(FileNotFoundError, match="cnn-standard-s0"):
        export_checkpoints.export(tmp_path / "runs", tmp_path / "results", tmp_path / "out")


# ---------------------------------------------------------------- checkpoints in the app


def test_fetch_local_checks_sha256_and_loads(exported, tmp_path):
    out, nets, _ = exported
    entries, paths = core.fetch_checkpoints("local", str(out))
    assert len(entries) == 6
    for e in entries:
        model = core.load_model(e, paths[e["run_id"]])
        assert not model.training
        ref = nets[e["run_id"]].state_dict()
        assert all(torch.equal(v, ref[k]) for k, v in model.state_dict().items())

    bad = tmp_path / "bad"
    bad.mkdir()
    for p in out.iterdir():
        (bad / p.name).write_bytes(p.read_bytes())
    first = json.loads((out / "models.json").read_text())["models"][0]["file"]
    (bad / first).write_bytes(b"not a checkpoint")
    with pytest.raises(ValueError, match="sha256"):
        core.fetch_checkpoints("local", str(bad))


def test_checkpoint_source_prefers_local(monkeypatch):
    monkeypatch.delenv("LOCAL_CHECKPOINT_DIR", raising=False)
    monkeypatch.delenv("HF_MODEL_REPO", raising=False)
    assert core.checkpoint_source() is None
    monkeypatch.setenv("HF_MODEL_REPO", "user/repo")
    assert core.checkpoint_source() == ("hub", "user/repo")
    monkeypatch.setenv("LOCAL_CHECKPOINT_DIR", "/ckpt")
    assert core.checkpoint_source() == ("local", "/ckpt")


def test_sort_entries_order():
    entries = [
        {"model": m, "strategy": s}
        for m in ("resnet18", "simple_cnn")
        for s in ("corruption_aware", "standard", "augmix")
    ]
    got = [(e["model"], e["strategy"]) for e in core.sort_entries(entries)]
    assert got == [(m, s) for m in core.MODEL_ORDER for s in core.STRATEGY_ORDER]


# ---------------------------------------------------------------- prediction


def test_predict_matches_the_evaluate_path():
    """Same transform and model as src.evaluate: softmax of model(eval_transform(PIL image))."""
    torch.manual_seed(0)
    model = build_model("simple_cnn").eval()
    img = np.random.default_rng(0).integers(0, 256, (32, 32, 3), dtype=np.uint8)
    probs = core.predict_probs(model, img)
    with torch.no_grad():
        logits = model(build_eval_transform()(Image.fromarray(img)).unsqueeze(0))
    np.testing.assert_allclose(probs, F.softmax(logits, dim=1)[0].numpy(), atol=1e-6)
    assert probs.shape == (10,) and probs.sum() == pytest.approx(1.0, abs=1e-5)


def test_top_k():
    probs = np.array([0.05, 0.4, 0.05, 0.3, 0, 0, 0, 0, 0.2, 0])
    assert core.top_k(probs) == [
        (CLASS_NAMES[1], pytest.approx(0.4)),
        (CLASS_NAMES[3], pytest.approx(0.3)),
        (CLASS_NAMES[8], pytest.approx(0.2)),
    ]


# ---------------------------------------------------------------- uploads


def _png(width, height, color=(200, 30, 30)) -> bytes:
    buf = BytesIO()
    Image.new("RGB", (width, height), color).save(buf, format="PNG")
    return buf.getvalue()


def test_prepare_upload_any_size_to_32x32():
    for w, h in ((640, 480), (31, 90), (32, 32)):
        img = core.prepare_upload(_png(w, h))
        assert img.shape == (32, 32, 3) and img.dtype == np.uint8
        assert tuple(img[16, 16]) == (200, 30, 30)
    gray = BytesIO()
    Image.new("L", (50, 50), 128).save(gray, format="PNG")
    assert core.prepare_upload(gray.getvalue()).shape == (32, 32, 3)


def test_prepare_upload_rejects_non_images():
    with pytest.raises(PIL.UnidentifiedImageError):
        core.prepare_upload(b"this is not an image")


def test_corrupt_upload_only_seen_and_seeded():
    img = core.prepare_upload(_png(100, 100))
    assert core.corrupt_upload(img, core.CLEAN, 0) is img
    for name in SEEN_CORRUPTIONS:
        a = core.corrupt_upload(img, name, 3)
        assert a.shape == img.shape and a.dtype == np.uint8
        np.testing.assert_array_equal(a, core.corrupt_upload(img, name, 3))
    with pytest.raises(ValueError, match="uploads support only"):
        core.corrupt_upload(img, "fog", 3)


def test_app_never_writes_uploads():
    """RULES/BUILD: uploads stay in memory. No file writing in the app's code paths."""
    text = (REPO / "app" / "core.py").read_text() + (REPO / "app" / "streamlit_app.py").read_text()
    assert not re.search(
        r"open\([^)]*['\"][wa]b?['\"]|\.write_bytes|\.write_text|\.save\(|tofile", text
    )


# ---------------------------------------------------------------- bundled samples


def test_load_samples_and_pick(tmp_path):
    n = 4
    images = np.arange(n * 32 * 32 * 3, dtype=np.uint64).reshape(n, 32, 32, 3).astype(np.uint8)
    idx = np.array([3, 5, 7, 9])
    np.savez(tmp_path / "s.npz", images=images, labels=np.arange(n), test_indices=idx)
    corrupted = np.stack([np.stack([images + c + s for s in range(2)]) for c in range(3)])
    np.savez(
        tmp_path / "c.npz",
        images=corrupted,
        corruptions=np.array(["fog", "snow", "frost"]),
        severities=np.array([1, 5]),
        test_indices=idx,
    )
    s = core.load_samples(tmp_path / "s.npz", tmp_path / "c.npz")
    np.testing.assert_array_equal(core.sample_image(s, 2, core.CLEAN, 0), images[2])
    np.testing.assert_array_equal(core.sample_image(s, 2, "snow", 5), corrupted[1, 1, 2])
    np.savez(
        tmp_path / "c2.npz",
        images=corrupted,
        corruptions=np.array(["fog", "snow", "frost"]),
        severities=np.array([1, 5]),
        test_indices=idx + 1,
    )
    with pytest.raises(ValueError, match="different images"):
        core.load_samples(tmp_path / "s.npz", tmp_path / "c2.npz")
    only_clean = core.load_samples(tmp_path / "s.npz", tmp_path / "missing.npz")
    assert only_clean.corrupted is None


def test_bundled_assets_if_present():
    if not core.SAMPLES_FILE.is_file():
        pytest.skip("app/assets not built")
    s = core.load_samples()
    assert s.images.shape == (100, 32, 32, 3)
    assert np.bincount(s.labels, minlength=10).tolist() == [10] * 10
    if s.corrupted is not None:
        assert s.corruptions == CORRUPTIONS
        assert s.corrupted.shape[:3] == (15, len(s.severities), 100)


# ---------------------------------------------------------------- results page data


def test_results_tables_format():
    main = core.load_main_results()
    shown = core.format_main_results(main)
    assert len(shown) == 6
    row = shown[(shown["Model"] == "ResNet-18") & (shown["Strategy"] == "standard")].iloc[0]
    src = main[(main["model"] == "resnet18") & (main["strategy"] == "standard")].iloc[0]
    assert (
        row["Clean accuracy"]
        == f"{100 * src['clean_acc_mean']:.1f}% ± {100 * src['clean_acc_std']:.1f}%"
    )
    per = core.load_per_corruption()
    assert len(per) == 6 * 15
    assert all((core.FIGURES_DIR / f).is_file() for f, _, _ in core.FIGURES)
    assert len(core.FIGURES) == 8


# ---------------------------------------------------------------- publish dry run


def test_publish_check_checkpoints(exported, tmp_path):
    out, _, _ = exported
    assert len(publish_hf.check_checkpoints(out)) == 6
    bad = tmp_path / "bad"
    bad.mkdir()
    for p in out.iterdir():
        (bad / p.name).write_bytes(p.read_bytes())
    meta = json.loads((bad / "models.json").read_text())
    meta["models"][0]["sha256"] = "0" * 64
    (bad / "models.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="sha256"):
        publish_hf.check_checkpoints(bad)


def test_publish_needs_token(exported, monkeypatch):
    out, _, _ = exported
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr(publish_hf, "space_files", lambda: ["app/README.md"])
    args = publish_hf.parse_args(
        ["--model-repo", "u/m", "--space", "u/s", "--checkpoints", str(out)]
    )
    with pytest.raises(SystemExit, match="HF_TOKEN"):
        publish_hf.publish(args)


def test_stage_space_puts_header_readme_at_root(tmp_path):
    publish_hf.stage_space(["app/README.md", "src/constants.py"], tmp_path)
    readme = (tmp_path / "README.md").read_text()
    assert readme.startswith("---\n") and "sdk: docker" in readme and "app_port: 7860" in readme
    assert (tmp_path / "src" / "constants.py").is_file()


# ---------------------------------------------------------------- Makefile and Docker


def test_make_help_lists_every_target():
    out = subprocess.run(
        ["make", "-s", "help"], cwd=REPO, capture_output=True, text=True, check=True
    ).stdout
    listed = {line.split()[0] for line in out.splitlines() if line.strip()}
    wanted = {
        "setup",
        "data",
        "smoke",
        "pilot",
        "train-all",
        "analyze",
        "report",
        "app",
        "docker-build",
        "docker-app",
        "check",
        "help",
    }
    assert wanted <= listed


def test_dockerfile_and_ignore():
    dockerfile = (REPO / "Dockerfile").read_text()
    assert dockerfile.startswith("#") and "FROM python:3.11-slim" in dockerfile
    assert "EXPOSE 7860" in dockerfile and "USER user" in dockerfile
    for folder in ("src/", "app/", "results/", "configs/", "splits/"):
        assert f"COPY --chown=user {folder}" in dockerfile
    ignored = (REPO / ".dockerignore").read_text().split()
    for entry in ("data", "runs", ".venv", ".git", "dist", "*.pt", ".env"):
        assert entry in ignored
    header = (REPO / "app" / "README.md").read_text().split("---")[1]
    assert "sdk: docker" in header and "app_port: 7860" in header


def test_failure_memo_remembers_then_expires():
    memo = core.FailureMemo(retry_after_s=300)
    key = ("hub", "user/repo")
    assert memo.recent(key, now=0.0) is None
    memo.record(key, now=10.0, message="OSError: offline")
    assert memo.recent(key, now=200.0) == "OSError: offline"
    assert memo.recent(("local", "/x"), now=200.0) is None
    assert memo.recent(key, now=310.0) is None
    memo.clear(key)
    assert memo.recent(key, now=11.0) is None


def test_app_requirements_pin_the_same_torch_as_requirements_local():
    """app/requirements.txt (complete, for Streamlit Community Cloud) must match the local pins."""

    def pins(path):
        out = {}
        for line in (REPO / path).read_text().splitlines():
            line = line.split("#")[0].strip()
            if "==" in line and not line.startswith("-"):
                name, version = line.split("==")
                out[name.lower()] = version.split("+")[0]
        return out

    app, local = pins("app/requirements.txt"), pins("requirements-local.txt")
    assert app["torch"] == local["torch"] and app["torchvision"] == local["torchvision"]
    assert "-r ../requirements.txt" in (REPO / "app" / "requirements.txt").read_text()
