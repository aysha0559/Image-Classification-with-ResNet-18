# Phase 05 BUILD: Training loop (config-driven, resumable)

## Read first
1. `docs/agent-prompts/RULES.md`
2. `docs/agent-prompts/DECISIONS.md` D3, D4
3. `docs/agent-prompts/CONTRACTS.md` C2, C4, C5, C6
4. `docs/PROJECT_OVERVIEW.md` sections 10.2, 12 (Colab practicalities), 18
5. `PROGRESS.md`; `validation/phase-04.md` must be PASS or PASS WITH NOTES

Branch: `phase-05-training-loop` from `main`.

## Goal
One training entry point, `python -m src.train`, that trains any (model, strategy, seed) from a config, survives Colab disconnects by checkpointing every epoch and resuming exactly, and writes the run folder of C6. All checks in this phase use tiny CPU runs; real training is Phase 07.

## In scope
- `src/train.py` implementing C5's `src.train` command:
  - Load and validate config, apply CLI overrides, compute `run_id`, create `$RUNS_DIR/{run_id}/`, write `config_resolved.yaml` and `run_meta.json` (`status: running`).
  - `set_seed(seed)`; device from `get_device`; model from `build_model`; loaders from `make_loaders`.
  - SGD (momentum, nesterov, weight decay from config), `CosineAnnealingLR(T_max=epochs)` stepped once per epoch, cross-entropy loss, AMP (`torch.autocast` + `GradScaler`) only when on CUDA and `amp: true`.
  - After each epoch: evaluate on the validation split (no augmentation), append a row to `train_log.csv` (C6 columns), save `checkpoint_last.pt` atomically including model, optimizer, scheduler, scaler, epoch and **all RNG states** (python, numpy, torch CPU, torch CUDA if used, and the DataLoader generator).
  - Resume: if `checkpoint_last.pt` exists and `--no-resume` is not given, restore everything and continue from the next epoch. If the stored config differs from the new one in any training-relevant key, refuse to resume and say which key differs.
  - At the end: save `model_final.pt` (C6), copy `train_log.csv` to `results/raw/{run_id}_train_log.csv` (smoke runs: `results/smoke/`), set `run_meta.json` `status: finished`, `finished_at`, `train_time_s`, `n_params`. On an exception: `status: failed` with the error message, then re-raise.
  - If the run is already `finished`, print that and exit 0 without training.
  - Detect divergence: if the training loss is NaN or inf, stop with `status: failed` and a clear message.
  - `--subset N` forces the `smoke-` run_id prefix (C5).
  - Progress output: one line per epoch (epoch, lr, train loss/acc, val loss/acc, time). `tqdm` bar optional.
- `scripts/registry.py` (C8): scans `$RUNS_DIR/*/run_meta.json` and results CSVs, writes `results/runs.csv`.
- `tests/test_train_smoke.py`: uses a fake tiny dataset (monkeypatch the loaders to 64 random images) so it needs no download; trains 2 epochs on CPU; checks run folder files exist with the right columns; then checks resume (below) on the fake data.

## Resume equivalence (the key check)
Run A: 3 epochs uninterrupted. Run B: same config and seed (different `--run-id`), started with the environment variable `STOP_AFTER_EPOCHS=2` so it exits cleanly after epoch 2 (as a Colab disconnect would leave it), then started again without the variable so it resumes and finishes epoch 3. On CPU with `num_workers: 0`, the `train_log.csv` loss/accuracy columns and the final weights of A and B must match (losses within 1e-6; weights `torch.allclose(atol=1e-6)`). Document `STOP_AFTER_EPOCHS` in the module docstring; it must change nothing except when the process exits. Note: `--epochs 2` is not a valid way to simulate this, because it changes the cosine schedule.

## Out of scope
Evaluation on the test set or CIFAR-10-C (Phase 06), real long runs (Phase 07), multi-run orchestration (Phase 08).

## Steps
1. Implement. Run `python -m src.train --config configs/smoke.yaml --seed 0` with real CIFAR-10 (subset 512, 1 epoch, CPU); show the log.
2. Run the resume equivalence check with `configs/smoke.yaml` and `--epochs 3` (real data, subset); show both logs side by side.
3. Run a smoke for each of the 4 strategies and both models (subset 256, 1 epoch) to prove every combination trains. Record time per epoch.
4. Lint, format, tests. Progress entry. Commit `phase 05: resumable config-driven training loop`.

## Done when
- [ ] All 8 (model, strategy) combinations train 1 smoke epoch on CPU without error.
- [ ] Run folder matches C6 exactly (file names, `train_log.csv` columns, `run_meta.json` keys).
- [ ] Resume equivalence holds on CPU.
- [ ] Resuming with a changed training key is refused with a clear message.
- [ ] A finished run is skipped on re-run.
- [ ] Checkpoint writes are atomic (temp file + rename).
- [ ] Smoke results never land in `results/raw/`.
- [ ] `python -m scripts.registry` writes `results/runs.csv` with the C8 columns.
- [ ] The test set and CIFAR-10-C are not touched by training code (leakage test still passes).
- [ ] Lint, format, tests pass.
