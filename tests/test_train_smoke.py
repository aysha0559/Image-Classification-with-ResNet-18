"""Training loop on fake data: run folder contract, resume, refusal, skip, failure, registry."""

import csv
import json
import random
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

import src.train
from scripts import registry
from src.config import load_config
from src.train import LOG_COLUMNS, DivergenceError, train
from src.utils import make_generator

META_KEYS = {
    "run_id",
    "model",
    "strategy",
    "seed",
    "epochs",
    "git_commit",
    "git_dirty",
    "torch_version",
    "torchvision_version",
    "python_version",
    "device_name",
    "n_params",
    "started_at",
    "finished_at",
    "train_time_s",
    "status",
}


class FakeImages(Dataset):
    """64 random images. Training items draw from every global RNG, as augmentation would."""

    def __init__(self, seed: int, augment: bool, bad: bool = False) -> None:
        g = torch.Generator().manual_seed(seed)
        self.images = torch.randn(64, 3, 32, 32, generator=g)
        self.labels = torch.randint(0, 10, (64,), generator=g).tolist()
        self.augment = augment
        self.bad = bad

    def __len__(self) -> int:
        return 64

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        image = self.images[i].clone()
        if self.bad:
            image[:] = float("nan")
        if self.augment:
            image += 0.1 * torch.rand(1) + 0.1 * np.random.rand() + 0.1 * random.random()
        return image, self.labels[i]


def fake_loaders(cfg: dict, bad: bool = False) -> tuple[DataLoader, DataLoader]:
    """Stand-in for `make_loaders`: same shuffling generator, no download."""
    train_loader = DataLoader(
        FakeImages(0, augment=True, bad=bad),
        batch_size=16,
        shuffle=True,
        generator=make_generator(cfg["seed"]),
    )
    return train_loader, DataLoader(FakeImages(1, augment=False), batch_size=16)


@pytest.fixture
def dirs(tmp_path, monkeypatch):
    """Point RUNS_DIR and RESULTS_DIR at a temp folder and use the fake loaders."""
    monkeypatch.setenv("RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.delenv("STOP_AFTER_EPOCHS", raising=False)
    monkeypatch.setattr(src.train, "make_loaders", fake_loaders)
    return tmp_path


def smoke_cfg(run_id: str = "smoke-t", epochs: int = 2, **changes) -> dict:
    """The smoke config with `epochs` and `changes` applied."""
    cfg = load_config("configs/smoke.yaml", epochs=epochs)
    cfg.update(run_id=run_id, **changes)
    return cfg


def read_log(run_dir) -> list[dict[str, str]]:
    with (run_dir / "train_log.csv").open(newline="") as f:
        return list(csv.DictReader(f))


def meta(run_dir) -> dict:
    return json.loads((run_dir / "run_meta.json").read_text())


def final_weights(run_dir) -> dict[str, torch.Tensor]:
    return torch.load(run_dir / "model_final.pt", weights_only=False)["state_dict"]


def interrupted_run(monkeypatch, run_id: str, epochs: int, stop_after: int) -> Path:
    """Run until `stop_after`, as if Colab disconnected there; clear the variable."""
    monkeypatch.setenv("STOP_AFTER_EPOCHS", str(stop_after))
    run_dir = train(smoke_cfg(run_id, epochs))
    monkeypatch.delenv("STOP_AFTER_EPOCHS")
    return run_dir


def test_run_folder_matches_contract(dirs):
    run_dir = train(smoke_cfg())
    assert sorted(p.name for p in run_dir.iterdir()) == [
        "checkpoint_last.pt",
        "config_resolved.yaml",
        "model_final.pt",
        "run_meta.json",
        "train_log.csv",
    ]
    rows = read_log(run_dir)
    assert (
        tuple(rows[0])
        == LOG_COLUMNS
        == (
            "epoch",
            "lr",
            "train_loss",
            "train_acc",
            "val_loss",
            "val_acc",
            "epoch_time_s",
        )
    )
    assert [int(r["epoch"]) for r in rows] == [1, 2]
    assert 0 <= float(rows[0]["train_acc"]) <= 1 and 0 <= float(rows[0]["val_acc"]) <= 1

    info = meta(run_dir)
    assert set(info) == META_KEYS
    assert info["status"] == "finished"
    assert info["n_params"] == 666858
    assert info["device_name"] == "cpu"
    assert info["train_time_s"] == pytest.approx(sum(float(r["epoch_time_s"]) for r in rows))

    checkpoint = torch.load(run_dir / "checkpoint_last.pt", weights_only=False)
    assert set(checkpoint) == {
        "model",
        "optimizer",
        "scheduler",
        "scaler",
        "epoch",
        "rng_states",
        "config",
    }
    assert checkpoint["epoch"] == 2
    assert set(checkpoint["rng_states"]) == {"python", "numpy", "torch", "cuda", "dataloader"}
    final = torch.load(run_dir / "model_final.pt", weights_only=False)
    assert set(final) == {"state_dict", "config", "run_id"}
    for name, tensor in final["state_dict"].items():
        assert torch.equal(tensor, checkpoint["model"][name])
    resolved = yaml.safe_load((run_dir / "config_resolved.yaml").read_text())
    assert resolved == smoke_cfg()


def test_optimizer_and_cosine_schedule(dirs, monkeypatch):
    seen = {}
    original = torch.optim.lr_scheduler.CosineAnnealingLR

    def spy(optimizer, T_max):
        seen.update(optimizer.param_groups[0], T_max=T_max)
        return original(optimizer, T_max=T_max)

    monkeypatch.setattr(torch.optim.lr_scheduler, "CosineAnnealingLR", spy)
    rows = read_log(train(smoke_cfg(epochs=4)))
    assert seen["T_max"] == 4
    assert (seen["momentum"], seen["nesterov"], seen["weight_decay"]) == (0.9, True, 0.0005)
    expected = [0.05 * (1 + np.cos(np.pi * e / 4)) for e in range(4)]
    assert [float(r["lr"]) for r in rows] == pytest.approx(expected)


def test_resume_matches_uninterrupted_run(dirs, monkeypatch, capsys):
    # Two epochs after the stop: the last learning rate depends on the restored scheduler.
    run_a = train(smoke_cfg("smoke-a", epochs=4))
    run_b = interrupted_run(monkeypatch, "smoke-b", epochs=4, stop_after=2)
    assert meta(run_b)["status"] == "running"
    assert not (run_b / "model_final.pt").exists()
    assert len(read_log(run_b)) == 2

    capsys.readouterr()
    train(smoke_cfg("smoke-b", epochs=4))
    epochs_run = [
        x.split()[1] for x in capsys.readouterr().out.splitlines() if x.startswith("epoch")
    ]
    assert epochs_run == ["3/4", "4/4"]

    rows_a, rows_b = read_log(run_a), read_log(run_b)
    assert [r["epoch"] for r in rows_b] == ["1", "2", "3", "4"]
    for a, b in zip(rows_a, rows_b, strict=True):
        for key in ("lr", "train_loss", "train_acc", "val_loss", "val_acc"):
            assert float(a[key]) == pytest.approx(float(b[key]), abs=1e-6)
    weights_a, weights_b = final_weights(run_a), final_weights(run_b)
    for name in weights_a:
        assert torch.allclose(weights_a[name].float(), weights_b[name].float(), atol=1e-6, rtol=0.0)
    info_b = meta(run_b)
    assert info_b["status"] == "finished"
    assert info_b["train_time_s"] == pytest.approx(sum(float(r["epoch_time_s"]) for r in rows_b))


def test_resume_keeps_first_started_at(dirs, monkeypatch):
    run_dir = interrupted_run(monkeypatch, "smoke-t", epochs=2, stop_after=1)
    info = meta(run_dir)
    info["started_at"] = "2000-01-01T00:00:00+00:00"
    (run_dir / "run_meta.json").write_text(json.dumps(info))
    train(smoke_cfg("smoke-t", epochs=2))
    assert meta(run_dir)["started_at"] == "2000-01-01T00:00:00+00:00"


@pytest.mark.parametrize(
    ("key", "value", "name"),
    [
        ("epochs", 5, "epochs"),
        ("seed", 3, "seed"),
        ("optimizer", {"lr": 0.2}, "optimizer.lr"),
    ],
)
def test_changed_training_key_is_refused(dirs, monkeypatch, key, value, name):
    run_dir = interrupted_run(monkeypatch, "smoke-t", epochs=3, stop_after=1)
    before = (run_dir / "config_resolved.yaml").read_text(), meta(run_dir)
    cfg = smoke_cfg("smoke-t", epochs=3)
    if isinstance(value, dict):
        cfg[key].update(value)
    else:
        cfg[key] = value
    with pytest.raises(ValueError, match=rf"refusing to resume.*{name}"):
        train(cfg)
    assert ((run_dir / "config_resolved.yaml").read_text(), meta(run_dir)) == before


def test_non_training_keys_may_change_on_resume(dirs, monkeypatch):
    run_dir = interrupted_run(monkeypatch, "smoke-t", epochs=2, stop_after=1)
    train(smoke_cfg("smoke-t", epochs=2, num_workers=2, device="auto"))
    assert meta(run_dir)["status"] == "finished"


def test_finished_run_is_skipped(dirs, monkeypatch, capsys):
    run_dir = train(smoke_cfg())
    before = (run_dir / "train_log.csv").read_bytes(), meta(run_dir)

    def must_not_run(*args):
        raise AssertionError("training started on a finished run")

    monkeypatch.setattr(src.train, "_run_epochs", must_not_run)
    train(smoke_cfg())
    assert "already finished" in capsys.readouterr().out
    train(smoke_cfg(), resume=False)
    assert ((run_dir / "train_log.csv").read_bytes(), meta(run_dir)) == before


def test_finished_run_with_other_settings_warns(dirs, capsys):
    train(smoke_cfg(epochs=1))
    capsys.readouterr()
    train(smoke_cfg(epochs=2))
    out = capsys.readouterr().out
    assert "was NOT retrained" in out and "epochs: 1 -> 2" in out


def test_no_resume_discards_an_unfinished_run(dirs, monkeypatch):
    run_dir = interrupted_run(monkeypatch, "smoke-t", epochs=2, stop_after=1)
    train(smoke_cfg("smoke-t", epochs=2), resume=False)
    assert [r["epoch"] for r in read_log(run_dir)] == ["1", "2"]
    baseline = train(smoke_cfg("smoke-base", epochs=2))
    assert read_log(run_dir)[0]["train_loss"] == read_log(baseline)[0]["train_loss"]


@pytest.mark.parametrize("cut_off", ["whole row", "partial row"])
def test_row_written_after_the_checkpoint_is_dropped(dirs, monkeypatch, cut_off):
    """A crash between the log row and the checkpoint must not leave a repeated epoch."""
    run_dir = interrupted_run(monkeypatch, "smoke-t", epochs=3, stop_after=1)
    extra = "2,0.075,9,0,9,0,1\n" if cut_off == "whole row" else "2,0.07"
    with (run_dir / "train_log.csv").open("a", newline="") as f:
        f.write(extra)
    train(smoke_cfg("smoke-t", epochs=3))
    rows = read_log(run_dir)
    assert [r["epoch"] for r in rows] == ["1", "2", "3"]
    assert rows[1]["train_loss"] != "9"


def test_crash_after_last_checkpoint_still_finishes(dirs):
    run_dir = train(smoke_cfg())
    info = meta(run_dir)
    info.update(status="running", finished_at=None, train_time_s=None)
    (run_dir / "run_meta.json").write_text(json.dumps(info))
    (run_dir / "model_final.pt").unlink()
    train(smoke_cfg())
    assert meta(run_dir)["status"] == "finished"
    assert len(read_log(run_dir)) == 2
    assert (run_dir / "model_final.pt").exists()


def test_divergence_stops_with_failed_status(dirs, monkeypatch):
    monkeypatch.setattr(src.train, "make_loaders", lambda cfg: fake_loaders(cfg, bad=True))
    with pytest.raises(DivergenceError, match="training loss is nan"):
        train(smoke_cfg())
    info = meta(dirs / "runs" / "smoke-t")
    assert info["status"] == "failed"
    assert "training loss is nan" in info["error"]
    assert not (dirs / "runs" / "smoke-t" / "checkpoint_last.pt").exists()


def test_error_sets_failed_status_and_success_clears_it(dirs, monkeypatch):
    def broken(cfg):
        raise OSError("no data here")

    monkeypatch.setattr(src.train, "make_loaders", broken)
    with pytest.raises(OSError, match="no data here"):
        train(smoke_cfg())
    run_dir = dirs / "runs" / "smoke-t"
    assert meta(run_dir)["status"] == "failed"
    assert "no data here" in meta(run_dir)["error"]

    monkeypatch.setattr(src.train, "make_loaders", fake_loaders)
    train(smoke_cfg())
    assert set(meta(run_dir)) == META_KEYS


def test_smoke_log_goes_to_smoke_folder(dirs):
    train(smoke_cfg("smoke-t"))
    assert (dirs / "results" / "smoke" / "smoke-t_train_log.csv").exists()
    assert not (dirs / "results" / "raw").exists()


def test_smoke_named_run_without_subset_stays_out_of_raw(dirs):
    train(smoke_cfg("smoke-t", subset=None))
    assert not (dirs / "results" / "raw").exists()


def test_real_log_goes_to_raw_folder(dirs):
    train(smoke_cfg("cnn-standard-s0", subset=None))
    assert (dirs / "results" / "raw" / "cnn-standard-s0_train_log.csv").exists()
    assert not (dirs / "results" / "smoke").exists()


def test_subset_on_the_command_line_forces_smoke_prefix():
    cfg = load_config("configs/cnn_standard.yaml", seed=0, subset=8)
    assert cfg["run_id"] == "smoke-cnn-standard-s0"


def test_registry_has_contract_columns(dirs):
    train(smoke_cfg("smoke-t"))
    run_dir = dirs / "runs" / "smoke-bad"
    run_dir.mkdir()
    info = {**meta(dirs / "runs" / "smoke-t"), "run_id": "smoke-bad", "status": "failed"}
    (run_dir / "run_meta.json").write_text(json.dumps({**info, "error": "boom"}))
    results_csv = dirs / "results" / "smoke" / "smoke-t.csv"
    results_csv.write_text("corruption,severity,accuracy\nfog,1,0.5\nclean,0,0.875\n")

    out = registry.build_registry(dirs / "runs", dirs / "results")
    with out.open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert out == dirs / "results" / "runs.csv"
    assert list(rows[0]) == sorted(META_KEYS, key=registry.META_COLUMNS.index) + [
        "has_results",
        "clean_accuracy",
    ]
    assert [r["run_id"] for r in rows] == ["smoke-bad", "smoke-t"]
    assert (rows[0]["has_results"], rows[0]["clean_accuracy"]) == ("False", "")
    assert (rows[1]["has_results"], rows[1]["clean_accuracy"]) == ("True", "0.875")
