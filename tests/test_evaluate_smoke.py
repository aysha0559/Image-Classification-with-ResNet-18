"""Evaluator on a tiny fake model and a fake CIFAR-10-C folder (no download)."""

import csv
import json

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import TensorDataset

import src.evaluate
from src.config import load_config
from src.constants import CORRUPTIONS, EXTRA_CORRUPTIONS, SEEN_CORRUPTIONS, SEVERITIES
from src.data import cifar10c
from src.data.cifar10c import load_cifar10c
from src.data.transforms import build_eval_transform
from src.evaluate import CSV_COLUMNS, conditions, evaluate, group_of, main

N = 20  # images per condition in the fake data
RUN_ID = "cnn-standard-s0"


def tiny_model(name="simple_cnn"):
    """A small stand-in for `build_model`. Dropout makes train mode visibly non-deterministic."""
    torch.manual_seed(0)
    return nn.Sequential(
        nn.AvgPool2d(8), nn.Flatten(), nn.Linear(48, 32), nn.Dropout(0.5), nn.Linear(32, 10)
    )


def write_run(run_dir, run_id=RUN_ID, status="finished"):
    """Create a finished-run folder holding the untrained tiny model."""
    run_dir.mkdir(parents=True)
    cfg = load_config("configs/cnn_standard.yaml", seed=0)
    cfg.update(device="cpu", num_workers=0, run_id=run_id)
    torch.save(
        {"state_dict": tiny_model().state_dict(), "config": cfg, "run_id": run_id},
        run_dir / "model_final.pt",
    )
    meta = {
        "run_id": run_id,
        "model": "simple_cnn",
        "strategy": "standard",
        "seed": 0,
        "status": status,
    }
    (run_dir / "run_meta.json").write_text(json.dumps(meta))
    (run_dir / "train_log.csv").write_text("epoch,lr\n1,0.1\n")


@pytest.fixture
def fake_env(tmp_path, monkeypatch):
    """Fake CIFAR-10-C (every severity differs), fake clean test set, scratch env folders."""
    monkeypatch.setattr(cifar10c, "N_PER_SEVERITY", N)
    folder = tmp_path / "data" / "CIFAR-10-C"
    folder.mkdir(parents=True)
    rng = np.random.default_rng(0)
    for name in CORRUPTIONS + EXTRA_CORRUPTIONS:
        images = rng.integers(0, 256, (5 * N, 32, 32, 3), dtype=np.uint8)
        np.save(folder / f"{name}.npy", images)
    np.save(folder / "labels.npy", (np.arange(5 * N) % 10).astype(np.uint8))

    clean = TensorDataset(torch.randn(N, 3, 32, 32), torch.arange(N) % 10)
    monkeypatch.setattr(src.evaluate, "get_test_dataset", lambda data_dir: clean)
    monkeypatch.setattr(src.evaluate, "build_model", tiny_model)
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("RESULTS_DIR", str(tmp_path / "results"))
    run_dir = tmp_path / "runs" / RUN_ID
    write_run(run_dir)
    return tmp_path, run_dir


def test_conditions_default_to_76():
    all_conditions = list(conditions())
    assert len(all_conditions) == 76 == len(set(all_conditions))
    assert all_conditions[0] == ("clean", 0)


def test_writes_exactly_the_c7_files(fake_env):
    tmp_path, run_dir = fake_env
    csv_path = evaluate(run_dir)
    raw = tmp_path / "results" / "raw"
    assert csv_path == raw / f"{RUN_ID}.csv"
    assert sorted(p.name for p in raw.iterdir()) == [
        f"{RUN_ID}.csv",
        f"{RUN_ID}_confusion.npz",
        f"{RUN_ID}_train_log.csv",
    ]
    assert not (tmp_path / "results" / "smoke").exists()
    assert (raw / f"{RUN_ID}_train_log.csv").read_text() == "epoch,lr\n1,0.1\n"

    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert tuple(reader.fieldnames) == CSV_COLUMNS
    assert len(rows) == 76


def test_columns_match_contract_c7(fake_env):
    """Written out here too, so a change to CSV_COLUMNS alone cannot pass."""
    _, run_dir = fake_env
    header = evaluate(run_dir).read_text().splitlines()[0]
    assert header == (
        "run_id,model,strategy,seed,corruption,family,group,severity,n_images,"
        "accuracy,macro_f1,loss,git_commit,evaluated_at"
    )


def test_labels_follow_d6(fake_env):
    _, run_dir = fake_env
    df = pd.read_csv(evaluate(run_dir))
    clean = df[df["corruption"] == "clean"]
    assert len(clean) == 1
    assert clean.iloc[0][["family", "group", "severity"]].tolist() == ["clean", "clean", 0]
    corrupted = df[df["corruption"] != "clean"]
    assert set(corrupted["severity"]) == set(SEVERITIES)
    assert set(corrupted["corruption"]) == set(CORRUPTIONS)
    assert set(corrupted[corrupted["group"] == "seen"]["corruption"]) == set(SEEN_CORRUPTIONS)
    assert set(corrupted["group"]) == {"seen", "unseen"}
    assert df[df["corruption"] == "fog"]["family"].unique().tolist() == ["weather"]
    assert (df["n_images"] == N).all()
    assert df[["run_id", "model", "strategy", "seed"]].drop_duplicates().values.tolist() == [
        [RUN_ID, "simple_cnn", "standard", 0]
    ]


def test_confusion_npz_agrees_with_the_csv(fake_env):
    tmp_path, run_dir = fake_env
    df = pd.read_csv(evaluate(run_dir))
    with np.load(tmp_path / "results" / "raw" / f"{RUN_ID}_confusion.npz") as npz:
        assert len(npz.files) == 76
        for row in df.itertuples():
            matrix = npz[f"{row.corruption}__s{row.severity}"]
            assert matrix.shape == (10, 10)
            assert np.issubdtype(matrix.dtype, np.integer)
            assert matrix.sum() == N
            assert np.trace(matrix) / matrix.sum() == pytest.approx(row.accuracy)


def test_confusion_rows_are_true_classes(fake_env):
    """Fake labels are balanced and an untrained model's predictions are not (C7)."""
    tmp_path, run_dir = fake_env
    evaluate(run_dir)
    with np.load(tmp_path / "results" / "raw" / f"{RUN_ID}_confusion.npz") as npz:
        assert all((npz[key].sum(axis=1) == N // 10).all() for key in npz.files)


def test_forward_pass_runs_in_eval_mode_without_gradients(fake_env, monkeypatch):
    _, run_dir = fake_env
    seen = []

    def recording_model(name="simple_cnn"):
        model = tiny_model(name)
        model.register_forward_hook(
            lambda module, *_: seen.append((module.training, torch.is_grad_enabled()))
        )
        return model

    monkeypatch.setattr(src.evaluate, "build_model", recording_model)
    evaluate(run_dir)
    assert len(seen) >= 76
    assert set(seen) == {(False, False)}


def test_conditions_are_really_different_images(fake_env):
    """Guards against every row being scored on the same data."""
    tmp_path, run_dir = fake_env
    df = pd.read_csv(evaluate(run_dir))
    assert df["loss"].nunique() == 76


def test_model_is_in_eval_mode_and_gives_the_same_scores_twice(fake_env):
    """Dropout left on would change the loss between two identical passes."""
    tmp_path, run_dir = fake_env
    first = pd.read_csv(evaluate(run_dir))
    second = pd.read_csv(evaluate(run_dir, force=True))
    assert first["loss"].tolist() == second["loss"].tolist()


def test_batch_size_does_not_change_scores(fake_env):
    _, run_dir = fake_env
    first = pd.read_csv(evaluate(run_dir, batch_size=7))
    second = pd.read_csv(evaluate(run_dir, batch_size=64, force=True))
    assert first["accuracy"].tolist() == second["accuracy"].tolist()
    assert np.allclose(first["loss"], second["loss"])


def test_refuses_to_overwrite_without_force(fake_env):
    tmp_path, run_dir = fake_env
    csv_path = evaluate(run_dir)
    before = csv_path.read_bytes()
    with pytest.raises(FileExistsError, match="--force"):
        evaluate(run_dir)
    assert csv_path.read_bytes() == before
    evaluate(run_dir, force=True)


def test_refuses_if_only_the_confusion_file_exists(fake_env):
    tmp_path, run_dir = fake_env
    csv_path = evaluate(run_dir)
    csv_path.unlink()
    with pytest.raises(FileExistsError, match="confusion"):
        evaluate(run_dir)


@pytest.mark.parametrize("status", ["running", "failed"])
def test_refuses_unfinished_run(fake_env, status):
    tmp_path, _ = fake_env
    run_dir = tmp_path / "runs" / "other"
    write_run(run_dir, "other", status=status)
    with pytest.raises(ValueError, match=f"status is '{status}'"):
        evaluate(run_dir)
    assert not (tmp_path / "results").exists()


def test_refuses_a_folder_without_metadata(fake_env, tmp_path):
    with pytest.raises(FileNotFoundError, match="run_meta.json"):
        evaluate(tmp_path)


def test_refuses_a_model_from_another_run(fake_env):
    _, run_dir = fake_env
    meta = json.loads((run_dir / "run_meta.json").read_text())
    meta["run_id"] = "resnet18-augmix-s1"
    (run_dir / "run_meta.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="model_final.pt is from"):
        evaluate(run_dir)


def test_subset_goes_only_to_smoke(fake_env):
    tmp_path, run_dir = fake_env
    csv_path = evaluate(run_dir, subset=6)
    assert csv_path == tmp_path / "results" / "smoke" / f"smoke-{RUN_ID}.csv"
    assert not (tmp_path / "results" / "raw").exists()
    df = pd.read_csv(csv_path)
    assert len(df) == 76
    assert (df["n_images"] == 6).all()
    assert df["run_id"].unique().tolist() == [f"smoke-{RUN_ID}"]
    with np.load(csv_path.with_name(f"smoke-{RUN_ID}_confusion.npz")) as npz:
        assert all(npz[key].sum() == 6 for key in npz.files)


def test_subset_uses_the_first_images(fake_env):
    """Recompute one row with a direct forward pass over the first images of the condition."""
    tmp_path, run_dir = fake_env
    df = pd.read_csv(evaluate(run_dir, subset=6))
    row = df[(df["corruption"] == "fog") & (df["severity"] == 3)].iloc[0]

    images, labels = load_cifar10c(str(tmp_path / "data"), "fog", 3)
    transform = build_eval_transform()
    batch = torch.stack([transform(Image.fromarray(image)) for image in images[:6]])
    model = tiny_model()
    model.load_state_dict(torch.load(run_dir / "model_final.pt")["state_dict"])
    with torch.no_grad():
        loss = F.cross_entropy(model.eval()(batch), torch.from_numpy(labels[:6]))
    assert row["loss"] == pytest.approx(loss.item())


def test_smoke_run_stays_in_smoke_without_subset_and_is_not_prefixed_twice(fake_env):
    tmp_path, _ = fake_env
    run_dir = tmp_path / "runs" / "smoke-x"
    write_run(run_dir, "smoke-x")
    assert evaluate(run_dir).parent.name == "smoke"
    assert evaluate(run_dir, subset=4, force=True).name == "smoke-x.csv"


@pytest.mark.parametrize("subset", [0, -5])
def test_bad_subset_raises(fake_env, subset):
    _, run_dir = fake_env
    with pytest.raises(ValueError, match="--subset"):
        evaluate(run_dir, subset=subset)


def test_extras_only_with_the_flag(fake_env):
    _, run_dir = fake_env
    df = pd.read_csv(evaluate(run_dir))
    assert not set(df["corruption"]) & set(EXTRA_CORRUPTIONS)

    df = pd.read_csv(evaluate(run_dir, include_extra=True, force=True))
    assert len(df) == 96
    extras = df[df["corruption"].isin(EXTRA_CORRUPTIONS)]
    assert len(extras) == 20
    assert extras["family"].unique().tolist() == ["extra"]
    assert extras["group"].unique().tolist() == ["unseen"]


def test_summary_ignores_extras(fake_env, capsys):
    _, run_dir = fake_env
    evaluate(run_dir, include_extra=True)
    out = capsys.readouterr().out
    assert "96 conditions" in out
    assert "(over 75 conditions)" in out
    assert "worst conditions:" in out


def test_group_of_every_corruption():
    assert group_of("clean") == "clean"
    assert sum(group_of(c) == "seen" for c in CORRUPTIONS) == 4
    assert sum(group_of(c) == "unseen" for c in CORRUPTIONS) == 11


def test_cli(fake_env, capsys):
    tmp_path, run_dir = fake_env
    main(["--run-dir", str(run_dir), "--subset", "5", "--batch-size", "4"])
    assert (tmp_path / "results" / "smoke" / f"smoke-{RUN_ID}.csv").exists()
    assert "clean accuracy" in capsys.readouterr().out
