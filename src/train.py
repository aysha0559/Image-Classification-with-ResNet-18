"""Training entry point: python -m src.train --config FILE [--seed N] [--epochs N] [--subset N]
[--run-id ID] [--no-resume].

Trains one (model, strategy, seed) from a config and writes the run folder of CONTRACTS C6.
A checkpoint is saved after every epoch, and a re-run of the same command resumes from it.
The model evaluated later is the final-epoch model (DECISIONS D3); validation is monitoring only.

Environment variables:
    DATA_DIR, RUNS_DIR, RESULTS_DIR: see CONTRACTS C2.
    STOP_AFTER_EPOCHS: exit cleanly once this epoch number (counted from 1 over the whole run) is
        checkpointed, leaving the run folder as a Colab disconnect would (`status: running`).
        It is for resume checks and changes nothing except when the process exits.

Resume: on CPU with `num_workers: 0` and the `standard` strategy, a resumed run matches an
uninterrupted one. With `num_workers > 0`, or with `corruption_aware`, augmentation randomness is
not checkpointed, so a resumed run differs; it continues the random stream and does not replay it
(DECISIONS change log, 2026-11-04).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torchvision
import yaml
from torch import nn
from torch.utils.data import DataLoader

from src.config import load_config
from src.data.cifar10 import make_loaders
from src.models import build_model
from src.utils import (
    Timer,
    atomic_save,
    count_parameters,
    env_path,
    get_device,
    get_git_commit,
    get_rng_states,
    is_git_dirty,
    set_rng_states,
    set_seed,
)

LOG_COLUMNS = ("epoch", "lr", "train_loss", "train_acc", "val_loss", "val_acc", "epoch_time_s")
# Config keys that may change between sessions: they do not define the experiment.
NON_TRAINING_KEYS = ("run_id", "device", "num_workers")


class DivergenceError(RuntimeError):
    """The training loss became NaN or inf."""


def _now() -> str:
    """Return the current UTC time in ISO 8601."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _is_smoke(cfg: dict) -> bool:
    """A subset run, or a run named as smoke, must never reach results/raw/."""
    return cfg["subset"] is not None or cfg["run_id"].startswith("smoke-")


def _stop_after_epochs() -> int | None:
    """Return STOP_AFTER_EPOCHS as an int, or None if it is not set."""
    value = os.environ.get("STOP_AFTER_EPOCHS")
    return int(value) if value else None


def _config_differences(old: dict, new: dict, prefix: str = "") -> list[str]:
    """Return one "key: old -> new" line per training-relevant key that differs."""
    diffs: list[str] = []
    for key in sorted(set(old) | set(new)):
        if not prefix and key in NON_TRAINING_KEYS:
            continue
        a, b = old.get(key), new.get(key)
        if isinstance(a, dict) and isinstance(b, dict):
            diffs += _config_differences(a, b, f"{prefix}{key}.")
        elif a != b:
            diffs.append(f"{prefix}{key}: {a!r} -> {b!r}")
    return diffs


def _read_meta(run_dir: Path) -> dict | None:
    """Return run_meta.json as a dict, or None if it does not exist."""
    path = run_dir / "run_meta.json"
    return json.loads(path.read_text()) if path.exists() else None


def _write_meta(run_dir: Path, meta: dict) -> None:
    """Write run_meta.json atomically."""
    tmp = run_dir / ".run_meta.json.tmp"
    tmp.write_text(json.dumps(meta, indent=2) + "\n")
    os.replace(tmp, run_dir / "run_meta.json")


def _report_finished(run_dir: Path, cfg: dict) -> None:
    """Say the run is already finished, and warn if this command asked for something else."""
    print(f"{cfg['run_id']}: already finished, nothing to do ({run_dir})")
    stored = yaml.safe_load((run_dir / "config_resolved.yaml").read_text())
    diffs = _config_differences(stored, cfg)
    if diffs:
        print(f"WARNING: {cfg['run_id']} was NOT retrained; it was finished with other settings:")
        for line in diffs:
            print(f"  {line}")
        print("Use another --run-id or RUNS_DIR to train these settings.")


def _load_checkpoint(run_dir: Path, cfg: dict) -> dict | None:
    """Return the run's checkpoint, or None if there is none.

    Raises ValueError, before anything in the run folder changes, if a training-relevant key
    differs from the stored config. Tensors load on the CPU because RNG states must stay there;
    `load_state_dict` moves the rest to the right device.
    """
    path = run_dir / "checkpoint_last.pt"
    if not path.exists():
        return None
    # weights_only=False: rng_states holds Python and NumPy objects. This is our own file.
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    diffs = _config_differences(checkpoint["config"], cfg)
    if diffs:
        raise ValueError(
            f"refusing to resume {cfg['run_id']}: config differs from the checkpoint in "
            f"{'; '.join(diffs)}. Restore the old value or use another --run-id."
        )
    return checkpoint


def _start_meta(cfg: dict, previous: dict | None) -> dict:
    """Build run_meta.json for this session; a resumed run keeps its first `started_at`."""
    commit = get_git_commit()
    if previous and previous["git_commit"] != commit:
        print(f"WARNING: resuming at commit {commit}; the run began at {previous['git_commit']}")
    return {
        "run_id": cfg["run_id"],
        "model": cfg["model"],
        "strategy": cfg["strategy"],
        "seed": cfg["seed"],
        "epochs": cfg["epochs"],
        "git_commit": commit,
        "git_dirty": is_git_dirty(),
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "python_version": platform.python_version(),
        "device_name": None,
        "n_params": None,
        "started_at": previous["started_at"] if previous else _now(),
        "finished_at": None,
        "train_time_s": None,
        "status": "running",
    }


def _read_log(path: Path) -> list[dict[str, str]]:
    """Return the complete rows of train_log.csv (a row cut off by a crash is left out)."""
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return [row for row in csv.DictReader(f) if None not in row.values()]


def _prepare_log(path: Path, last_epoch: int) -> None:
    """Rewrite train_log.csv with the header and only the rows up to `last_epoch`.

    A row is written before its checkpoint, so a crash between the two leaves a row for an epoch
    that will run again.
    """
    rows = [row for row in _read_log(path) if int(row["epoch"]) <= last_epoch]
    if len(rows) != last_epoch:
        print(f"WARNING: {path} has {len(rows)} rows for {last_epoch} checkpointed epochs")
    tmp = path.with_name(f".{path.name}.tmp")
    with tmp.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def _append_log(path: Path, row: dict[str, float]) -> None:
    """Append one row to train_log.csv and push it to disk."""
    with path.open("a", newline="") as f:
        csv.DictWriter(f, fieldnames=LOG_COLUMNS).writerow(row)
        f.flush()
        os.fsync(f.fileno())


def _train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    use_amp: bool,
) -> tuple[float, float]:
    """Train for one epoch; return (mean loss, accuracy). Raise DivergenceError on NaN or inf."""
    model.train()
    criterion = nn.CrossEntropyLoss()
    total_loss, correct, seen = 0.0, 0, 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, labels)
        value = loss.item()
        if not math.isfinite(value):
            raise DivergenceError(f"training loss is {value} at batch {seen // len(labels) + 1}")
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += value * len(labels)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        seen += len(labels)
    return total_loss / seen, correct / seen


@torch.no_grad()
def _validate(
    model: nn.Module, loader: DataLoader, device: torch.device, use_amp: bool
) -> tuple[float, float]:
    """Return (mean loss, accuracy) on the validation split."""
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    total_loss, correct, seen = 0.0, 0, 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            outputs = model(images)
            total_loss += criterion(outputs, labels).item()
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        seen += len(labels)
    return total_loss / seen, correct / seen


def _finish_run(cfg: dict, run_dir: Path, model: nn.Module, meta: dict) -> None:
    """Save the final-epoch model, copy the log to the results folder, mark the run finished."""
    state_dict = {name: tensor.cpu() for name, tensor in model.state_dict().items()}
    atomic_save(
        {"state_dict": state_dict, "config": cfg, "run_id": cfg["run_id"]},
        run_dir / "model_final.pt",
    )
    log_path = run_dir / "train_log.csv"
    folder = env_path("RESULTS_DIR") / ("smoke" if _is_smoke(cfg) else "raw")
    folder.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(log_path, folder / f"{cfg['run_id']}_train_log.csv")
    meta["status"] = "finished"
    meta["finished_at"] = _now()
    # Summed from the log so that time from every session counts, including disconnected ones.
    meta["train_time_s"] = sum(float(row["epoch_time_s"]) for row in _read_log(log_path))
    _write_meta(run_dir, meta)


def _run_epochs(
    cfg: dict, run_dir: Path, meta: dict, checkpoint: dict | None, stop_after: int | None
) -> None:
    """Build everything, restore `checkpoint` if given, and train until done or `stop_after`."""
    set_seed(cfg["seed"])
    device = get_device(cfg["device"])
    meta["device_name"] = torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
    model = build_model(cfg["model"]).to(device)
    meta["n_params"] = count_parameters(model)
    _write_meta(run_dir, meta)

    train_loader, val_loader = make_loaders(cfg)
    opt = cfg["optimizer"]
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=opt["lr"],
        momentum=opt["momentum"],
        nesterov=opt["nesterov"],
        weight_decay=opt["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])
    use_amp = cfg["amp"] and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    done = 0
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        # A checkpoint from a CPU session holds an empty state, which an enabled scaler refuses.
        if checkpoint["scaler"]:
            scaler.load_state_dict(checkpoint["scaler"])
        done = checkpoint["epoch"]
        # Last, and before the first iterator: it draws the shuffle order and the worker base
        # seed from the DataLoader generator, which must not restart from its seed.
        set_rng_states(checkpoint["rng_states"], train_loader.generator)
        print(f"{cfg['run_id']}: resuming after epoch {done}/{cfg['epochs']}", flush=True)

    log_path = run_dir / "train_log.csv"
    _prepare_log(log_path, done)
    last = cfg["epochs"] if stop_after is None else min(cfg["epochs"], stop_after)
    for epoch in range(done + 1, last + 1):
        lr = optimizer.param_groups[0]["lr"]
        with Timer() as timer:
            train_loss, train_acc = _train_one_epoch(
                model, train_loader, optimizer, scaler, device, use_amp
            )
            val_loss, val_acc = _validate(model, val_loader, device, use_amp)
        scheduler.step()
        _append_log(
            log_path,
            {
                "epoch": epoch,
                "lr": lr,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "epoch_time_s": timer.elapsed,
            },
        )
        atomic_save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "epoch": epoch,
                "rng_states": get_rng_states(train_loader.generator),
                "config": cfg,
            },
            run_dir / "checkpoint_last.pt",
        )
        print(
            f"epoch {epoch}/{cfg['epochs']} lr {lr:.5f} "
            f"train_loss {train_loss:.4f} train_acc {train_acc:.4f} "
            f"val_loss {val_loss:.4f} val_acc {val_acc:.4f} time {timer.elapsed:.1f}s",
            flush=True,
        )

    if max(done, last) < cfg["epochs"]:
        print(f"{cfg['run_id']}: stopped after epoch {max(done, last)} (STOP_AFTER_EPOCHS)")
        return
    _finish_run(cfg, run_dir, model, meta)
    print(f"{cfg['run_id']}: finished, {run_dir}")


def train(cfg: dict[str, Any], resume: bool = True) -> Path:
    """Train the run described by `cfg`, resuming if allowed; return its run folder.

    A finished run is left alone, even with `resume=False`. With `resume=False` an unfinished
    run's checkpoint and log are discarded. An error sets `status: failed` and is re-raised.
    """
    stop_after = _stop_after_epochs()
    run_dir = env_path("RUNS_DIR") / cfg["run_id"]
    previous = _read_meta(run_dir)
    if previous and previous["status"] == "finished":
        _report_finished(run_dir, cfg)
        return run_dir

    checkpoint = _load_checkpoint(run_dir, cfg) if resume else None
    run_dir.mkdir(parents=True, exist_ok=True)
    if not resume:
        (run_dir / "checkpoint_last.pt").unlink(missing_ok=True)
    meta = _start_meta(cfg, previous if checkpoint else None)
    (run_dir / "config_resolved.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    _write_meta(run_dir, meta)
    try:
        _run_epochs(cfg, run_dir, meta, checkpoint, stop_after)
    except (Exception, KeyboardInterrupt) as e:
        meta["status"] = "failed"
        meta["error"] = f"{type(e).__name__}: {e}"
        _write_meta(run_dir, meta)
        raise
    return run_dir


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line of CONTRACTS C5."""
    parser = argparse.ArgumentParser(description="Train one model from a config.")
    parser.add_argument("--config", required=True, help="YAML config (CONTRACTS C4)")
    parser.add_argument("--seed", type=int, help="override the config's seed")
    parser.add_argument("--epochs", type=int, help="override the config's epochs")
    parser.add_argument("--subset", type=int, help="first N images per split; makes a smoke run")
    parser.add_argument("--run-id", help="override the run id")
    parser.add_argument("--no-resume", action="store_true", help="discard an unfinished run")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Command-line entry point."""
    args = parse_args(argv)
    cfg = load_config(
        args.config, seed=args.seed, epochs=args.epochs, subset=args.subset, run_id=args.run_id
    )
    train(cfg, resume=not args.no_resume)


if __name__ == "__main__":
    main()
