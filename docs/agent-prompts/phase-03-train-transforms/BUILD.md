# Phase 03 BUILD: Training transforms for S0-S3 (including our own corruptions)

## Read first
1. `docs/agent-prompts/RULES.md` (section 3 on leakage especially)
2. `docs/agent-prompts/DECISIONS.md` D5, D6 (stop if `CONFIRM`)
3. `docs/agent-prompts/CONTRACTS.md` C3, C4, C10
4. `docs/PROJECT_OVERVIEW.md` section 8 (all), section 18
5. `PROGRESS.md`; `validation/phase-02.md` must be PASS or PASS WITH NOTES

Branch: `phase-03-train-transforms` from `main`.

## Goal
`build_train_transform(strategy, params)` returns the exact training pipeline for each strategy, and `src/data/corruptions.py` holds our own implementations of the 4 **seen** corruptions used by S3. This is where data leakage is most likely, so the code must be independent of CIFAR-10-C.

## In scope
- `src/data/transforms.py`:
  - `none`: ToTensor + Normalize.
  - `standard`: RandomCrop(32, padding=4) + RandomHorizontalFlip + ToTensor + Normalize.
  - `augmix`: crop + flip, then `torchvision.transforms.AugMix` with the D5 parameters (check the installed torchvision supports `all_ops`; if not, stop and tell me), then ToTensor + Normalize.
  - `corruption_aware`: crop + flip, then `RandomSeenCorruption(p, corruptions, severities)`, then ToTensor + Normalize.
  - Unknown strategy raises `ValueError`. Wire `get_train_val_datasets(cfg)` (Phase 01) to use `build_train_transform(cfg["strategy"], cfg["strategy_params"])`.
- `src/data/corruptions.py`:
  - `apply_corruption(img, name, severity, rng)` for `gaussian_noise`, `defocus_blur`, `contrast`, `jpeg_compression` only. Any other name raises `ValueError` naming the allowed ones.
  - Follow the formulas and the 5 CIFAR severity parameters described in Hendrycks & Dietterich (2019) and its public reference code (`github.com/hendrycks/robustness`, the CIFAR corruption script) for these 4 corruptions. Read the source; do not reproduce parameter values from memory. In the module docstring cite the file you followed and list the parameter values per severity.
  - Pure NumPy / OpenCV / Pillow; deterministic given `rng`. Input and output uint8 HWC (32x32x3), values clipped to [0, 255].
  - `RandomSeenCorruption`: a callable on PIL images; with probability `p` picks one corruption uniformly and a severity uniformly from `severities`. Uses a NumPy `Generator` seeded per worker so runs are reproducible (seed from `torch.initial_seed()` inside the worker).
- `scripts/make_sample_grid.py --ours`: a second grid of **our** 4 corruptions x 5 severities on the same image, saved to `results/figures/extra_our_corruptions.png`, next to the CIFAR-10-C versions of the same 4 corruptions, so a person can compare them by eye. (This script may import `cifar10c` for display only; it is not training code.)
- `tests/test_transforms.py`: each strategy returns a float tensor (3, 32, 32); `none` and eval transforms are deterministic; `standard` changes the image across calls with different seeds; unknown strategy raises.
- `tests/test_corruptions.py`: output shape/dtype/range; same rng seed gives same output; severity 5 changes the image more than severity 1 (mean abs difference) for each of the 4; `p=0` never changes the image and `p=1` always does; unsupported name raises.

## Out of scope
Models, training loop, any unseen corruption implementation, JSD loss.

## Steps
1. Implement. Look at the comparison grid PNG yourself. Our corruptions should look similar in kind and strength to the CIFAR-10-C ones at the same severity; record your honest description.
2. Measure, on 1,000 training images, the mean absolute pixel difference our version makes at each severity, and the same for CIFAR-10-C's version of the same corruption on 1,000 test images. Put both tables in `PROGRESS.md` (they should be the same order of magnitude; they will not be identical because the images differ and noise is random).
3. Time each transform on 1,000 images on CPU and record (AugMix and S3 cost matters for Colab budget).
4. Lint, format, tests. Progress entry. Commit `phase 03: training transforms S0-S3 and seen corruptions`.

## Done when
- [ ] All 4 strategies build and produce correct tensors; unknown raises.
- [ ] AugMix uses exactly the D5 parameters, including `all_ops=False`.
- [ ] Our 4 corruptions exist, are cited, deterministic with a seed, and get stronger with severity.
- [ ] No unseen corruption is implemented in `corruptions.py`.
- [ ] `src/data/corruptions.py` and `src/data/transforms.py` do not import or read CIFAR-10-C (leakage test passes).
- [ ] The comparison grid and the strength table are in place.
- [ ] Lint, format, tests pass.
