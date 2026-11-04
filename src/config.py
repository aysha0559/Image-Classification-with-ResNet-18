"""Load a YAML config, validate it against CONTRACTS C4, apply CLI overrides, build the run_id."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from src.constants import CORRUPTIONS, MODEL_SHORT_NAMES, MODELS, SEVERITIES, STRATEGIES

# Expected type for every key in C4. Nested dicts describe nested sections.
# A tuple of types means "any of these"; NoneType allows null.
_INT = int
_FLOAT = (int, float)
SCHEMA: dict[str, Any] = {
    "model": str,
    "strategy": str,
    "seed": _INT,
    "epochs": _INT,
    "batch_size": _INT,
    "optimizer": {
        "name": str,
        "lr": _FLOAT,
        "momentum": _FLOAT,
        "nesterov": bool,
        "weight_decay": _FLOAT,
    },
    "scheduler": {"name": str},
    "amp": bool,
    "val_size": _INT,
    "split_seed": _INT,
    "num_workers": _INT,
    "device": str,
    "subset": (int, type(None)),
    "strategy_params": {
        "augmix": {
            "severity": _INT,
            "mixture_width": _INT,
            "chain_depth": _INT,
            "alpha": _FLOAT,
            "all_ops": bool,
        },
        "corruption_aware": {
            "p": _FLOAT,
            "corruptions": list,
            "severities": list,
        },
    },
}

OVERRIDABLE = ("seed", "epochs", "subset", "run_id")


def _type_name(expected: Any) -> str:
    """Readable name for a type or tuple of types."""
    types = expected if isinstance(expected, tuple) else (expected,)
    names = {t.__name__ for t in types}
    if "int" in names and "float" in names:
        names.discard("int")
    return " or ".join(sorted("null" if n == "NoneType" else n for n in names))


def _check_type(value: Any, expected: Any, where: str) -> None:
    """Raise ValueError if `value` is not of `expected` type (bool never counts as a number)."""
    types = expected if isinstance(expected, tuple) else (expected,)
    if isinstance(value, bool) and bool not in types:
        ok = False
    else:
        ok = isinstance(value, types)
    if not ok:
        raise ValueError(
            f"config key '{where}' must be {_type_name(expected)}, "
            f"got {type(value).__name__} ({value!r})"
        )


def _validate_section(cfg: dict, schema: dict, prefix: str = "") -> None:
    """Recursively check that `cfg` has exactly the keys of `schema` with the right types."""
    unknown = sorted(set(cfg) - set(schema))
    if unknown:
        allowed = ", ".join(sorted(schema))
        raise ValueError(
            f"unknown config key(s) {[prefix + k for k in unknown]}; allowed here: {allowed}"
        )
    missing = sorted(set(schema) - set(cfg))
    if missing:
        raise ValueError(f"missing config key(s) {[prefix + k for k in missing]}")
    for key, expected in schema.items():
        where = prefix + key
        if isinstance(expected, dict):
            if not isinstance(cfg[key], dict):
                raise ValueError(f"config key '{where}' must be a mapping")
            _validate_section(cfg[key], expected, where + ".")
        else:
            _check_type(cfg[key], expected, where)


def _check_choice(value: Any, choices: tuple, where: str) -> None:
    """Raise ValueError if `value` is not one of `choices`."""
    if value not in choices:
        raise ValueError(f"config key '{where}' must be one of {list(choices)}, got {value!r}")


def validate_config(cfg: dict) -> None:
    """Validate a config dict against CONTRACTS C4; raise ValueError with a clear message."""
    if not isinstance(cfg, dict):
        raise ValueError("config must be a YAML mapping")
    body = {k: v for k, v in cfg.items() if k != "run_id"}
    if "run_id" in cfg and not (cfg["run_id"] is None or isinstance(cfg["run_id"], str)):
        raise ValueError("config key 'run_id' must be str or null")
    _validate_section(body, SCHEMA)
    _check_choice(cfg["model"], MODELS, "model")
    _check_choice(cfg["strategy"], STRATEGIES, "strategy")
    _check_choice(cfg["device"], ("auto", "cpu", "cuda"), "device")
    _check_choice(cfg["optimizer"]["name"], ("sgd",), "optimizer.name")
    _check_choice(cfg["scheduler"]["name"], ("cosine",), "scheduler.name")
    for key in ("epochs", "batch_size", "val_size"):
        if cfg[key] < 1:
            raise ValueError(f"config key '{key}' must be >= 1, got {cfg[key]}")
    if cfg["seed"] < 0 or cfg["num_workers"] < 0:
        raise ValueError("config keys 'seed' and 'num_workers' must be >= 0")
    if cfg["subset"] is not None and cfg["subset"] < 1:
        raise ValueError(f"config key 'subset' must be null or >= 1, got {cfg['subset']}")
    ca = cfg["strategy_params"]["corruption_aware"]
    if not 0.0 <= ca["p"] <= 1.0:
        raise ValueError("config key 'strategy_params.corruption_aware.p' must be in [0, 1]")
    for c in ca["corruptions"]:
        _check_choice(c, CORRUPTIONS, "strategy_params.corruption_aware.corruptions")
    for s in ca["severities"]:
        _check_choice(s, SEVERITIES, "strategy_params.corruption_aware.severities")


def build_run_id(model: str, strategy: str, seed: int, smoke: bool = False) -> str:
    """Return the run id per CONTRACTS C3: {short_model}-{strategy}-s{seed}, with smoke- prefix."""
    run_id = f"{MODEL_SHORT_NAMES[model]}-{strategy}-s{seed}"
    return f"smoke-{run_id}" if smoke else run_id


def load_config(
    path: str | Path,
    seed: int | None = None,
    epochs: int | None = None,
    subset: int | None = None,
    run_id: str | None = None,
) -> dict:
    """Load and validate a YAML config, apply CLI overrides, and set cfg["run_id"].

    A `subset` (from the YAML or the override), or `epochs` below the YAML's value, marks the run
    as smoke, so the run id gets the `smoke-` prefix (CONTRACTS C5). An explicit `run_id` is used
    as given, plus the smoke prefix if needed.
    """
    path = Path(path)
    try:
        cfg = yaml.safe_load(path.read_text())
    except yaml.YAMLError as e:
        raise ValueError(f"{path}: invalid YAML: {e}") from e
    try:
        validate_config(cfg)
        config_epochs = cfg["epochs"]
        overrides = {"seed": seed, "epochs": epochs, "subset": subset}
        for key, value in overrides.items():
            if value is not None:
                cfg[key] = value
        if run_id is not None:
            cfg["run_id"] = run_id
        validate_config(cfg)
    except ValueError as e:
        raise ValueError(f"{path}: {e}") from e

    smoke = cfg["subset"] is not None or cfg["epochs"] < config_epochs
    given = cfg.get("run_id")
    if given:
        cfg["run_id"] = given if not smoke or given.startswith("smoke-") else f"smoke-{given}"
    else:
        cfg["run_id"] = build_run_id(cfg["model"], cfg["strategy"], cfg["seed"], smoke)
    return cfg
