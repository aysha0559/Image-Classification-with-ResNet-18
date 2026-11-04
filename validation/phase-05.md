# Validation: Phase 05 training-loop

- Date: 2026-11-04
- Branch / commit checked: phase-05-training-loop @ 0f04df78ea964f8419f719c9b1968822735150a4
- Working tree clean before checking: yes
- Environment: Python 3.12.16, torch 2.14.1+cpu (torchvision 0.29.2+cpu), device cpu
- Verdict: PASS

## Summary
All 19 checks (14 Must, 5 Should) were re-run from scratch and passed. All 8 (model, strategy) combinations train, the run folder matches C6 exactly, and resume equivalence holds bit for bit on real CIFAR-10, both after `STOP_AFTER_EPOCHS` and after a real `SIGKILL` that landed during a checkpoint write. A resumed `corruption_aware` run does not replay the first session's stream, and a control shows that check can fail. The CUDA paths (AMP, `GradScaler`, CUDA RNG restore) could not run on this CPU-only machine and stay unverified until Phase 07. The first note below describes a resume failure that matters only if a run moves from CPU to GPU.

## Checks
| ID | Check | Must/Should | Result | Evidence |
|---|---|---|---|---|
| V1 | Branch, clean tree, commit | Must | PASS | `git rev-parse --abbrev-ref HEAD` -> `phase-05-training-loop`; `git rev-parse HEAD` -> `0f04df7...`; `git status --short` -> empty; `git merge-base --is-ancestor main HEAD` -> true; `git diff --stat main HEAD` -> 9 files |
| V2 | Lint, format, tests | Must | PASS | `ruff format --check .` -> `75 files already formatted`; `ruff check .` -> `All checks passed!`; `time pytest -q` -> `190 passed in 14.01s` (real 0m15.204s, CPU) |
| V3 | Smoke run | Must | PASS | `python -m src.train --config configs/smoke.yaml --seed 0` -> `epoch 1/1 lr 0.10000 train_loss 2.6624 train_acc 0.1113 val_loss 2.3188 val_acc 0.1172 time 1.5s`, one line per epoch, exit 0 |
| V4 | Run folder contract | Must | PASS | Folder: `checkpoint_last.pt config_resolved.yaml model_final.pt run_meta.json train_log.csv`. Log header `epoch,lr,train_loss,train_acc,val_loss,val_acc,epoch_time_s`. `run_meta.json` holds exactly the 16 C6 keys (no `error`, as the run did not fail), `status` `finished`. Checkpoint keys `config epoch model optimizer rng_states scaler scheduler`; `rng_states` keys `cuda dataloader numpy python torch`; `model_final.pt` keys `config run_id state_dict` |
| V5 | All 8 combinations train | Must | PASS | `RUNS_DIR=/tmp/val05/runs_v5`, `--seed 0 --subset 128 --epochs 1` for all 8 configs: every exit 0, every `run_meta.json` `finished`, every `train_log.csv` 1 row, each folder's `strategy` and `model` match its config. Table below |
| V6 | Resume equivalence | Must | PASS | Runs `smoke-a`, `smoke-b`: max absolute log difference lr 0.0, train_loss 0.0, train_acc 0.0, val_loss 0.0, val_acc 0.0; weights `allclose True identical True` over 34 tensors. Table below |
| V7 | Resume really resumed | Must | PASS | Session 1: exit 0, `"status": "running"`, no `model_final.pt`, 2 log rows. Session 2 printed `smoke-b: resuming after epoch 2/3`, then only `epoch 3/3`; log epochs `['1', '2', '3']` |
| V8 | Changed config refused on resume | Must | PASS | `--epochs 5` -> `ValueError: refusing to resume smoke-c: config differs from the checkpoint in epochs: 3 -> 5. Restore the old value or use another --run-id.`, exit 1; `status` still `running`, log unchanged. Also refused: `seed: 0 -> 1`, nested `optimizer.lr: 0.1 -> 0.2` |
| V9 | Finished run is skipped | Must | PASS | Re-run of V3 -> `smoke-cnn-standard-s0: already finished, nothing to do (...)`, exit 0, real 0m3.432s, no epoch line |
| V10 | Optimiser and schedule per D4 | Must | PASS | `src/train.py` builds `SGD` from config and `CosineAnnealingLR(optimizer, T_max=cfg["epochs"])`; `scheduler.step()` is called once per epoch, after validation. All 9 configs: `lr 0.1, momentum 0.9, nesterov true, weight_decay 0.0005, cosine`. Checkpoint `param_groups`: momentum 0.9, nesterov True, weight_decay 0.0005, initial_lr 0.1; `T_max 3, eta_min 0.0`. `lr` column equals `0.05(1+cos(pi t/T))`: 0.1, 0.075, 0.025 (3 epochs); 0.1, 0.0933, 0.075, 0.05, 0.025, 0.0067 (6 epochs); 0.0 after the last step |
| V11 | Final-epoch model saved (D3) | Must | PASS | `grep -nEi "best\|early.?stop" src/train.py` -> no matches (exit 1). `model_final.pt` `state_dict` is tensor-equal to `checkpoint_last.pt` `model` at epoch 3 -> `True`; loads into `build_model("simple_cnn")` with `<All keys matched successfully>` |
| V12 | Atomic checkpoint | Should | PASS | `src/utils.py:101` `tempfile.mkstemp(dir=path.parent, ...)`, `:104` `torch.save`, `:105` `os.replace(tmp, path)`. A `SIGKILL` during a write (see extra check) left a stray temp file and an intact `checkpoint_last.pt` at epoch 2 |
| V13 | NaN guard | Should | PASS | Read: `math.isfinite` on every batch loss. Ran `lr: 1000000.0` in a scratch config -> `DivergenceError: training loss is nan at batch 4`, exit 1, `"status": "failed"`, `error` set |
| V14 | Failure status | Should | PASS | `DATA_DIR=/tmp/val05/afile/data` (parent is a file) -> `NotADirectoryError`, exit 1, `"status": "failed"`, `error` holds the message. Re-run with good data -> `finished`, `grep -c '"error"'` -> 0 |
| V15 | Smoke results isolated | Must | PASS | `ls -A results/raw` in the repo -> `.gitkeep` only; `grep -c smoke-` -> 0. `git status --short` empty after all runs. Smoke logs are in `$RESULTS_DIR/smoke/`; no `raw/` folder was created in any scratch results folder |
| V16 | No test or CIFAR-10-C use in training | Must | PASS | `grep -nE "train=False\|get_test_dataset\|cifar10c" src/train.py` -> no matches (exit 1); same grep on `scripts/registry.py src/utils.py src/config.py` -> no matches; modules loaded by `import src.train` containing `cifar10c` -> `[]`; `pytest -q tests/test_no_leakage.py` -> `2 passed` |
| V17 | Registry | Should | PASS | `RUNS_DIR=/tmp/val05/runs RESULTS_DIR=/tmp/val05/results python -m scripts.registry` -> `wrote /tmp/val05/results/runs.csv (8 runs)`, exit 0; columns are exactly the 16 meta columns + `has_results,clean_accuracy`; 8 rows for 8 run folders, including one `running` and one `failed`. With a scratch results CSV for `smoke-a`: `has_results True`, `clean_accuracy 0.2031`; all others `False` and empty |
| V18 | Per-epoch CPU times in `PROGRESS.md` plausible | Should | PASS | V5 at 128 images: simple_cnn 0.50-0.94 s, resnet18 3.88-5.58 s. Like-for-like repeat at 256 images: within 12% of every `PROGRESS.md` value. Table below |
| V19 | Resume does not replay the random stream | Must | PASS | Read: `src/train.py` calls `set_rng_states(checkpoint["rng_states"], train_loader.generator)` before the first iterator is created. Traced run: resumed epoch's shuffle and base seed differ from session 1's; control with the generator restore removed replays session 1. Table below |

### V5 and V18, time per epoch (s) [smoke]
One run each, seed 0, `num_workers: 2`, CPU, this machine. Source: `run_meta.json` `train_time_s`. These do not predict Colab time.

| Model | Strategy | 128 images (V5) | 256 images (this check) | 256 images (`PROGRESS.md`) |
|---|---|---|---|---|
| simple_cnn | none | 0.50 | 0.93 | 0.88 |
| simple_cnn | standard | 0.54 | 0.82 | 0.93 |
| simple_cnn | augmix | 0.94 | 1.07 | 1.01 |
| simple_cnn | corruption_aware | 0.68 | 0.92 | 0.93 |
| resnet18 | none | 5.51 | 9.64 | 10.26 |
| resnet18 | standard | 3.88 | 10.31 | 10.99 |
| resnet18 | augmix | 4.89 | 10.64 | 11.16 |
| resnet18 | corruption_aware | 5.58 | 10.58 | 10.97 |

### V6, resume equivalence (real CIFAR-10, subset 512) [smoke]

| Epoch | lr | train_loss A | train_loss B | val_loss A | val_loss B | val_acc A | val_acc B |
|---|---|---|---|---|---|---|---|
| 1 | 0.1 | 2.6623692512512207 | 2.6623692512512207 | 2.318750739097595 | 2.318750739097595 | 0.1171875 | 0.1171875 |
| 2 | 0.07500000000000001 | 2.3245916962623596 | 2.3245916962623596 | 2.2590643167495728 | 2.2590643167495728 | 0.177734375 | 0.177734375 |
| 3 | 0.025000000000000015 | 2.154705047607422 | 2.154705047607422 | 2.190445840358734 | 2.190445840358734 | 0.203125 | 0.203125 |

### V19, shuffle order and worker base seed
`configs/cnn_corruption_aware.yaml --subset 256 --epochs 3 --seed 0`, `num_workers: 2`. Traced by `/tmp/val05/trace.py`, a scratch wrapper around `src.train.main`; the base seed is the DataLoader iterator's `_base_seed`, the hash is sha256 of the JSON index list, first 12 characters.

| Run | Session | Base seed | Epoch | Shuffle hash | First 8 indices |
|---|---|---|---|---|---|
| Interrupted | 1 | 900450186894289455 | 1 | fc5d2c78dc09 | 117 136 193 28 227 75 253 205 |
| Interrupted | 1 | same iterator | 2 | e23b0780d963 | 94 144 250 40 248 126 96 202 |
| Interrupted | 2 (resumed) | 6428247394855981926 | 3 | f8d5a7b54eed | 20 49 50 11 252 191 239 13 |
| Uninterrupted | 1 | 900450186894289455 | 3 | 5b2957f464c8 | 242 54 248 205 226 53 246 113 |
| Control, generator restore removed | 2 (resumed) | 900450186894289455 | 3 | fc5d2c78dc09 | 117 136 193 28 227 75 253 205 |

The resumed epoch's shuffle differs from epochs 1 and 2, and its base seed differs from session 1's. The control's resumed session repeats session 1's base seed and epoch 1's shuffle, so this check can detect a replay. The interrupted run differs from the uninterrupted one at epoch 3 (train_loss 2.32059645652771 vs 2.3263050317764282), which is the difference the human accepted (DECISIONS change log, 2026-11-04).

### Extra checks, not asked for by VALIDATE.md

**Resume after a real kill.** `STOP_AFTER_EPOCHS` is a clean exit and a Colab disconnect is not, so `smoke-k1` (`--epochs 6`) was killed with `SIGKILL` once its log held 3 rows. State after the kill: 3 log rows, checkpoint at epoch 2, a stray `.checkpoint_last.pt.ft_izsmm.tmp`, `status` `running`. So the kill landed during the checkpoint write for epoch 3, the case the log trim and atomic save exist for. The re-run printed `resuming after epoch 2/6` and trained epochs 3-6. Against an uninterrupted `smoke-k0`: log epochs 1-6 with no duplicates, max absolute difference 0.0 in lr, train_loss, train_acc, val_loss and val_acc; weights `allclose True identical True` over 34 tensors.

**Run ids.** From `load_config` on `configs/cnn_standard.yaml`, seed 0: no override -> `cnn-standard-s0`; `epochs=30` -> `cnn-standard-s0`; `epochs=29` -> `smoke-cnn-standard-s0`; `epochs=1, run_id="cnn-standard-s0"` -> `smoke-cnn-standard-s0`; `subset=64, run_id="foo"` -> `smoke-foo`; `subset=64, run_id="smoke-a"` -> `smoke-a`. A shortened run cannot take a real id (C5).

**Mutation table from `PROGRESS.md`.** In a scratch copy from `git archive HEAD`, one change at a time, `pytest -q tests/test_train_smoke.py`. All 9 failure counts equal `PROGRESS.md`:

| Mutation | Result |
|---|---|
| No RNG restore | 1 failed, 20 passed |
| Generator not restored | 1 failed, 20 passed |
| Optimizer not restored | 1 failed, 20 passed |
| Scheduler not restored | 1 failed, 20 passed |
| Log not trimmed | 2 failed, 19 passed |
| Smoke logs sent to raw | 3 failed, 18 passed |
| No NaN guard | 1 failed, 20 passed |
| No `scheduler.step()` | 1 failed, 20 passed |
| Failed status not set | 2 failed, 19 passed |

## Failures (if any)
None.

## Notes (non-blocking)
- **A checkpoint saved on CPU probably cannot be resumed on a GPU with `amp: true`.** `NON_TRAINING_KEYS` lets `device` change between sessions, and the module comment treats that as safe. On CPU the scaler is disabled and `scaler.state_dict()` returns `{}`. Loading `{}` into an enabled scaler raises `RuntimeError: The source state dict is empty, possibly because it was saved from a disabled instance of GradScaler.` This was reproduced with `torch.amp.GradScaler("cpu", enabled=True)`, because this machine has no CUDA; `src.train` itself was not run this way. The likely path is `scaler.load_state_dict(checkpoint["scaler"])` in `_run_epochs`, so with `device: auto` a run started on a Colab session without a GPU would then fail on every re-run under a GPU, with `status: failed`. Real runs are planned on GPU only, so this does not block the merge. Suggested fix, for the human to decide: load the scaler only if the saved state is non-empty.
- **CUDA paths never ran here.** AMP, `GradScaler`, CUDA RNG restore and `device_name` on a GPU are unverified, as `PROGRESS.md` says. Phase 07 needs to check them, including one interrupted and resumed run on the GPU.
- **C6 and C8 differ in wording for failed runs.** C8 says "all `run_meta.json` fields", and C6 now gives failed runs an `error` field, but `scripts/registry.py` drops it on purpose (`extrasaction="ignore"`). The columns match C8's fixed list and V17 passes. If the human wants `error` left out, C8 could say "all fields except `error`".
- `train_time_s` is summed from the log (`_finish_run`), and `_prepare_log` drops the row of an epoch lost to a crash, so time spent on lost epochs is not counted (read from the code, not measured). The code comment "including disconnected ones" holds only for completed epochs. Phase 07 should keep this in mind if it estimates hours from `train_time_s`.
- The `PROGRESS.md` known issue about stray temp files is real: the `SIGKILL` above left one. It was not removed by the later resume. A ResNet-18 temp file can reach 86M on Drive.
- The `PROGRESS.md` known issue about shared smoke run ids is real, and it is why V5 needs its own `RUNS_DIR`.
- `/tmp/val05` is used as VALIDATE.md specifies. It holds this validation's scratch runs, scripts and the mutated copy, and nothing there is needed after the merge.

## Claims in PROGRESS.md that could not be confirmed
- "human approved" for the `CONTRACTS.md` C5 and C6 changes and the VALIDATE.md V5 change -> the edits exist on this branch, and the code and VALIDATE follow them, but the repo holds no record of the approval. The human should confirm.
- `wrote .../runs.csv (4 runs)` -> depends on what the build's scratch folder held. This validation got 8 runs from 8 folders, which is the same behaviour.
- `Failure status: unusable DATA_DIR -> PermissionError` -> not reproduced with that error, because a different bad path was used. It gave `NotADirectoryError` with the same `failed` status and `error` message.
- The "before the C5 change" output and "the scheduler change first passed all tests" -> history of earlier code, not repeatable at this commit. The current behaviour was checked above.
