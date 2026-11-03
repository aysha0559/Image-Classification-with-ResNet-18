# Phase 07 BUILD: Pilot run, end to end on real GPU (CNN + S1, seed 0)

The overview's advice: build the full pipeline end to end for one configuration before running the others. This phase does that, measures real time, and freezes the epoch counts.

## Read first
1. `docs/agent-prompts/RULES.md`
2. `docs/agent-prompts/DECISIONS.md` D4, D9
3. `docs/agent-prompts/CONTRACTS.md` C2, C5, C6, C7, C8
4. `docs/PROJECT_OVERVIEW.md` sections 7, 10.2, 12, 15 (tip)
5. `PROGRESS.md`; `validation/phase-06.md` must be PASS or PASS WITH NOTES

Branch: `phase-07-pilot` from `main`.

## Who does what
You (the agent) prepare everything and analyse the outputs. The human runs Colab, because you cannot click in Colab. If `torch.cuda.is_available()` is true on this machine, you may run the GPU steps locally instead; say so and record the GPU name.

## In scope
1. **Push readiness.** The Colab notebook clones the repo. Ask the human for the GitHub repo URL (or whether to upload a zip to Drive instead). Do not create a GitHub repo or push without the human's go-ahead.
2. **`notebooks/run_on_colab.ipynb`**: a notebook with a parameters cell (`MODEL`, `STRATEGY`, `SEED`, `STAGE` in {`train`, `evaluate`, `both`}) and cells that: run the setup from `00_colab_setup.ipynb` (or `%run` it), copy CIFAR-10-C from Drive to `/content/data` and verify it, run `python -m src.train ...` with `RUNS_DIR` on Drive, run `python -m src.evaluate ...`, then copy `results/raw/{run_id}*` and the registry to Drive under `MyDrive/cifar10-robustness/results/`. Only calls; no logic.
3. **Human step, pilot A (speed):** CNN + standard, seed 0, 30 epochs, then evaluate. Also, in the same session, time **2 epochs** each of `resnet18-standard`, `resnet18-augmix`, `resnet18-corruption_aware`, `cnn-augmix`, `cnn-corruption_aware` with `--run-id timing-...` and `STOP_AFTER_EPOCHS=2` (these timing runs are thrown away, and their run folders deleted after the human confirms).
4. **Human step, resume test on GPU:** while the CNN pilot is training, the human disconnects the runtime (Runtime > Disconnect and delete runtime) once after a few epochs, then re-runs the notebook. Training must resume from the last epoch on Drive.
5. **Bring back results:** the human copies `results/raw/cnn-standard-s0*` and the run's `run_meta.json`/`train_log.csv` into the repo (or you pull them from a Drive download folder the human names).
6. **Analyse the pilot** and write `results/pilot_report.md`:
   - Clean test accuracy, macro-F1; mean corrupted accuracy; the 5 worst conditions; accuracy by severity (0-5). All from the CSV.
   - Training curve sanity (from `train_log.csv`): train loss falls, val accuracy rises, no NaNs.
   - Time per epoch for each timed config on the actual GPU; projected total GPU hours for the 18-run matrix plus 18 evaluations, using the proposed epochs (CNN 30, ResNet 40) and one alternative (ResNet 30 and 50).
   - Torch / torchvision / Python versions Colab used vs `requirements-local.txt`.
7. **Freeze epochs.** Propose the ResNet-18 epoch count that keeps the full matrix within about 20 GPU hours (overview estimate) and give the projection. **Ask the human to confirm**, then update `DECISIONS.md` D4 to `CONFIRMED` with the value and a change-log row, and update the 3 ResNet configs (and S0 config).
8. Run `python -m scripts.registry` on the collected run.

## Expected ranges (from the overview; used as gates, not targets)
- Simple CNN + S1, 30 epochs, 45k training images: clean accuracy about 80-87%. Below 78% or above 90%: stop and investigate with the human before Phase 08 (do not tune hyperparameters on the test set; check for bugs such as wrong normalisation, eval in train mode, augmentation applied to test images, wrong labels).
- Mean corrupted accuracy must be clearly below clean (expect a drop of more than 10 points), and mean accuracy should fall as severity rises from 1 to 5.

## Out of scope
Running the other 17 runs (Phase 08). Any hyperparameter tuning.

## Done when
- [ ] `cnn-standard-s0` trained 30 epochs on GPU and evaluated; `results/raw/cnn-standard-s0.csv` (76 rows), `_confusion.npz`, `_train_log.csv` committed.
- [ ] Resume after a real Colab disconnect worked (log shows it; describe it in `PROGRESS.md`).
- [ ] `results/pilot_report.md` written, every number traceable to a file.
- [ ] Clean accuracy within the gate, or the human has agreed in writing how to proceed.
- [ ] D4 ResNet epochs confirmed by the human and applied to configs.
- [ ] `results/runs.csv` has the pilot row.
- [ ] Timing runs removed from Drive (after the human confirms) and not in `results/raw/`.
- [ ] Progress entry; commit `phase 07: pilot run cnn-standard-s0 and frozen epochs`.
