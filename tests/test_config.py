"""Config loading, validation and run_id format (CONTRACTS C3, C4)."""

from pathlib import Path

import pytest
import yaml

from src.config import load_config

CONFIGS = Path(__file__).resolve().parent.parent / "configs"
MAIN = {
    "cnn_standard.yaml": "cnn-standard",
    "cnn_augmix.yaml": "cnn-augmix",
    "cnn_corruption_aware.yaml": "cnn-corruption_aware",
    "resnet18_standard.yaml": "resnet18-standard",
    "resnet18_augmix.yaml": "resnet18-augmix",
    "resnet18_corruption_aware.yaml": "resnet18-corruption_aware",
}


def test_all_nine_configs_load():
    files = sorted(CONFIGS.glob("*.yaml"))
    assert len(files) == 9
    for f in files:
        load_config(f)


@pytest.mark.parametrize("name,prefix", MAIN.items())
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_run_id(name, prefix, seed):
    assert load_config(CONFIGS / name, seed=seed)["run_id"] == f"{prefix}-s{seed}"


def test_smoke_run_id_and_overrides():
    cfg = load_config(CONFIGS / "smoke.yaml")
    assert cfg["run_id"] == "smoke-cnn-standard-s0"
    cfg = load_config(CONFIGS / "cnn_standard.yaml", seed=1, epochs=2, subset=64)
    assert (cfg["seed"], cfg["epochs"], cfg["subset"]) == (1, 2, 64)
    assert cfg["run_id"] == "smoke-cnn-standard-s1"
    assert load_config(CONFIGS / "cnn_standard.yaml", run_id="mine")["run_id"] == "mine"


def test_fewer_epochs_than_the_config_make_a_smoke_run():
    cfg = load_config(CONFIGS / "cnn_standard.yaml", seed=0, epochs=1)
    assert cfg["run_id"] == "smoke-cnn-standard-s0"
    cfg = load_config(CONFIGS / "cnn_standard.yaml", seed=0, epochs=1, run_id="mine")
    assert cfg["run_id"] == "smoke-mine"


def test_config_epochs_or_more_keep_the_real_run_id():
    for epochs in (None, 30, 31):
        cfg = load_config(CONFIGS / "cnn_standard.yaml", seed=0, epochs=epochs)
        assert cfg["run_id"] == "cnn-standard-s0"


def test_main_configs_differ_only_in_model_strategy_epochs():
    base = yaml.safe_load((CONFIGS / "cnn_standard.yaml").read_text())
    for name in MAIN:
        cfg = yaml.safe_load((CONFIGS / name).read_text())
        diff = {k for k in base if base[k] != cfg[k]}
        assert diff <= {"model", "strategy", "epochs"}, (name, diff)


def _write_bad(tmp_path, edit):
    cfg = yaml.safe_load((CONFIGS / "cnn_standard.yaml").read_text())
    edit(cfg)
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


def test_unknown_key_rejected(tmp_path):
    path = _write_bad(tmp_path, lambda c: c.update(epoch=3))
    with pytest.raises(ValueError, match="unknown config key.*epoch"):
        load_config(path)


def test_wrong_type_rejected(tmp_path):
    path = _write_bad(tmp_path, lambda c: c["optimizer"].update(lr="fast"))
    with pytest.raises(ValueError, match="optimizer.lr"):
        load_config(path)


def test_bad_choice_rejected(tmp_path):
    path = _write_bad(tmp_path, lambda c: c.update(strategy="mixup"))
    with pytest.raises(ValueError, match="strategy"):
        load_config(path)
