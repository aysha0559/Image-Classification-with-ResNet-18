"""Seeding, device selection, atomic save, env paths, git helpers."""

import random

import numpy as np
import torch

from src.utils import (
    Timer,
    atomic_save,
    env_path,
    get_device,
    get_git_commit,
    is_git_dirty,
    set_seed,
)


def _draw():
    return random.random(), np.random.rand(3).tolist(), torch.rand(3).tolist()


def test_same_seed_same_numbers():
    set_seed(0)
    a = _draw()
    set_seed(0)
    b = _draw()
    set_seed(1)
    c = _draw()
    assert a == b
    assert a != c
    assert torch.backends.cudnn.deterministic and not torch.backends.cudnn.benchmark


def test_get_device():
    assert get_device("cpu").type == "cpu"
    assert get_device("auto").type in ("cpu", "cuda")


def test_atomic_save(tmp_path):
    path = tmp_path / "sub" / "x.pt"
    atomic_save({"a": torch.ones(2)}, path)
    assert torch.equal(torch.load(path)["a"], torch.ones(2))
    assert [p.name for p in path.parent.iterdir()] == ["x.pt"]


def test_env_path(monkeypatch):
    monkeypatch.delenv("DATA_DIR", raising=False)
    assert str(env_path("DATA_DIR")) == "data"
    monkeypatch.setenv("DATA_DIR", "/tmp/somewhere")
    assert str(env_path("DATA_DIR")) == "/tmp/somewhere"


def test_git_helpers_and_timer():
    assert isinstance(get_git_commit(), str)
    assert isinstance(is_git_dirty(), bool)
    with Timer() as t:
        sum(range(1000))
    assert t.elapsed >= 0


def test_make_generator_repeats():
    from src.utils import make_generator

    a = torch.randperm(20, generator=make_generator(0))
    b = torch.randperm(20, generator=make_generator(0))
    c = torch.randperm(20, generator=make_generator(1))
    assert torch.equal(a, b) and not torch.equal(a, c)


def test_seed_worker_seeds_numpy():
    from src.utils import seed_worker

    torch.manual_seed(5)
    seed_worker(0)
    a = np.random.rand(3)
    torch.manual_seed(5)
    seed_worker(0)
    assert np.array_equal(a, np.random.rand(3))
