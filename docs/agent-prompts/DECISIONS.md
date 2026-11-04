# Frozen decisions

The overview leaves these choices open or gives a range. Agents must not re-decide them.
Items marked **CONFIRM** need the human's answer before the phase that uses them. To answer, change `CONFIRM` to `CONFIRMED`, or edit the value and write `CONFIRMED (changed)`.

Changing a decision after Phase 07 means re-running affected experiments. Record every change in the log at the bottom.

## D1. Data split

| Item | Value | Status |
|---|---|---|
| Training set | 45,000 images from the CIFAR-10 train set | CONFIRMED |
| Validation set | 5,000 images from the CIFAR-10 train set, **stratified** (500 per class) | CONFIRMED |
| Split seed | `split_seed = 42`, the same for every run and every seed. The training seed does not change the split. | CONFIRMED |
| Split storage | The split indices are saved once to `splits/cifar10_val_indices.json` and committed, so every run uses the identical split. | CONFIRMED |
| Test set | The 10,000 CIFAR-10 test images, used only for final evaluation. | Fixed by overview |

## D2. Normalisation

Per-channel mean and std of the **45,000 training images** (computed once in Phase 01, saved to `splits/cifar10_stats.json`, committed). The same values are used for validation, test and CIFAR-10-C. Status: CONFIRMED

## D3. Which checkpoint is evaluated

The **final-epoch** model. No early stopping, no best-validation checkpoint selection (that would be a per-run choice and makes strategies less comparable). Validation accuracy is logged for monitoring only. Status: CONFIRMED

## D4. Training settings (identical across strategies)

| Setting | Value | Status |
|---|---|---|
| Optimiser | SGD, momentum 0.9, Nesterov, weight decay 5e-4 | Fixed by overview |
| Learning rate | 0.1, cosine annealing to 0 over all epochs, no warm-up | CONFIRMED |
| Batch size | 128 | Fixed by overview |
| Mixed precision | On when CUDA is available | Fixed by overview |
| Epochs, Simple CNN | 30 | Fixed by overview |
| Epochs, ResNet-18 | **40.** Frozen at the end of Phase 07 using measured Colab time (`results/pilot_report.md`). Must be the same for all ResNet strategies. | CONFIRMED |
| Seeds | 0, 1, 2 | Fixed by overview |
| Determinism | Seeds set for Python, NumPy, PyTorch (CPU and CUDA); `cudnn.benchmark = False`; `cudnn.deterministic = True`. Exact bit-for-bit repeats on GPU are not promised; same-seed runs should agree within about 0.5 points. | CONFIRMED |

## D5. Strategies

| ID | Name in code | Transforms on the training image | Status |
|---|---|---|---|
| S0 | `none` | ToTensor + Normalize only (optional ablation, run only if time allows after Phase 08) | CONFIRMED |
| S1 | `standard` | RandomCrop(32, padding=4) + RandomHorizontalFlip + ToTensor + Normalize | Fixed by overview |
| S2 | `augmix` | S1 crop and flip, then `torchvision.transforms.AugMix(severity=3, mixture_width=3, chain_depth=-1, alpha=1.0, all_ops=False)`, then ToTensor + Normalize. `all_ops=False` keeps AugMix's operations from overlapping with CIFAR-10-C corruptions, as in the AugMix paper. **No JSD loss** (compute budget). | CONFIRMED |
| S3 | `corruption_aware` | S1 crop and flip, then with probability **p = 0.5** apply one **seen** corruption chosen uniformly at random at a severity chosen uniformly from 1-5, using our own code; then ToTensor + Normalize. | CONFIRMED |

## D6. Seen and unseen corruptions (for S3 and RQ4)

| Group | Corruptions | Status |
|---|---|---|
| **Seen** (S3 trains on our own versions of these) | `gaussian_noise`, `defocus_blur`, `contrast`, `jpeg_compression` | CONFIRMED |
| **Unseen** (never used in training) | `shot_noise`, `impulse_noise`, `glass_blur`, `motion_blur`, `zoom_blur`, `snow`, `frost`, `fog`, `brightness`, `elastic_transform`, `pixelate` | CONFIRMED |

Unseen corruptions are split further for RQ4, because some are close relatives of a seen one (`shot_noise` and `impulse_noise` of `gaussian_noise`; the other blurs of `defocus_blur`):

| Analysis group | Corruptions | Status |
|---|---|---|
| `seen` | gaussian_noise, defocus_blur, contrast, jpeg_compression | CONFIRMED |
| `unseen_same_family` (a seen corruption is in the same family) | shot_noise, impulse_noise, glass_blur, motion_blur, zoom_blur, brightness, elastic_transform, pixelate | CONFIRMED |
| `unseen_other_family` (no seen corruption in the family) | snow, frost, fog | CONFIRMED |

The digital family is mixed (brightness and elastic transform are not much like contrast or JPEG), so the report also shows per-corruption results, not only these groups.

Our S3 corruption functions follow the published formulas and severity parameters of Hendrycks & Dietterich for those 4 corruptions, re-implemented in our own code. They never read CIFAR-10-C files.

## D7. Evaluation conditions

- Main evaluation: clean test set + the **15 standard corruptions x 5 severities** = 76 conditions per model.
- The 4 extra corruptions (`speckle_noise`, `gaussian_blur`, `spatter`, `saturate`) are **not** evaluated by default. Optional flag `--include-extra`. They are never part of mCE or the main averages. Status: CONFIRMED

## D8. Metric definitions

- Accuracy and macro-F1 are stored as fractions in [0, 1]. Shown as percentages only in tables and figures.
- Error = 1 - accuracy.
- Absolute degradation = clean - corrupted (fraction; shown as percentage points).
- Relative degradation = (clean - corrupted) / clean.
- **mCE baseline** = Simple CNN + S1, with the baseline error for each (corruption, severity) being the **mean over its 3 seeds**. Each run's CE is computed against that one baseline table, then mCE is averaged over seeds. So the baseline's own mCE is about 100% (exactly 100% for its seed-mean). Status: CONFIRMED
- Relative mCE: `rCE_c = sum_s(E_c,s - E_clean) / sum_s(E_base_c,s - E_base_clean)`, averaged over the 15 corruptions. Status: CONFIRMED
- "Mean corrupted accuracy" = unweighted mean over the 75 (corruption, severity) conditions.

## D9. Environment

| Item | Value | Status |
|---|---|---|
| Compute for real runs | Google Colab, free T4 (or better if available). | Fixed by overview |
| torch / torchvision on Colab | Use the versions Colab preinstalls; do not reinstall torch there. Record the versions in every run's metadata. | CONFIRMED |
| torch / torchvision locally | Pinned in `requirements-local.txt` to the same major.minor as Colab at Phase 00 time (the agent checks Colab's current versions with the human or records `UNKNOWN`). | CONFIRMED |
| Other packages | Pinned in `requirements.txt`. | Fixed |
| Storage on Colab | Data and runs on Google Drive under `MyDrive/cifar10-robustness/`; CIFAR-10-C copied to `/content/data` at the start of each session. | Fixed by overview |

## D10. Showcase and deployment (Phase 11)

| Item | Value | Status |
|---|---|---|
| Reproduce in the cloud | One "Open in Colab" notebook that runs any stage. | CONFIRMED |
| Results viewer + demo | A Streamlit app (results tables, figures, and a live "corrupt an image, watch predictions" demo) hosted on a free Hugging Face Space. | CONFIRMED |
| Checkpoint hosting | Final checkpoints for seed 0 (6 models) on a Hugging Face model repo, downloaded by the app at start. | CONFIRMED |
| Local / any-cloud | `Dockerfile` + `Makefile`; `make app` runs the viewer anywhere Docker runs. | CONFIRMED |

## D11. Simple CNN details the overview leaves open

| Item | Value | Status |
|---|---|---|
| Conv padding | 1 on every 3x3 conv | CONFIRMED |
| Dropout after each conv block | 0.25 | CONFIRMED |
| Dropout before the last Linear | 0.5 | CONFIRMED |
| Expected parameters | about 0.67 M with these choices (inside the overview's 0.5-1 M) | to be measured in Phase 04 |

## Change log

| Date | Decision | Change | Why | Runs affected |
|---|---|---|---|---|
| 2026-11-03 | all | Initial proposal | Written from docs/PROJECT_OVERVIEW.md | none |
| 2026-11-03 | all except D4 ResNet epochs | All proposed defaults accepted by the human | Human reply "Accept all defaults" | none |
| 2026-11-04 | D5, D6 | No change to any value. The human accepted that our `jpeg_compression` does not reproduce CIFAR-10-C pixel for pixel: the call and quality values follow the reference and mean strength agrees to about 0.1 pixel levels, but the published images show stronger colour artefacts on some images (PROGRESS Phase 03, Known issues). Do not tune it against CIFAR-10-C. Disclose it in the report's limitations. | Phase 03 comparison; RULES section 3 forbids choosing parameters from CIFAR-10-C | none |
| 2026-11-04 | D4 | No change to any value. The human accepted that resumed runs are not bit-for-bit identical to uninterrupted ones when `num_workers > 0` or the strategy is `corruption_aware`, because that augmentation state is not checkpointed. A resumed run must not replay the original shuffle order or worker seeds (Phase 05 BUILD). | Same-seed agreement within about 0.5 points is already the D4 standard | none |
| 2026-11-04 | none (CONTRACTS C5, C6; Phase 05 VALIDATE V5) | No change to any value here. The human approved three edits in the Phase 05 build session ("Go ahead with your recommendations"): C6 adds `error` to `run_meta.json` for failed runs only; C5 makes `--epochs` below the config's value a smoke run; V5 uses its own `RUNS_DIR` and checks that each run really trained. | Records the approval, which the Phase 05 validator could not find in the repo | none |
| 2026-11-04 | D4 | ResNet-18 epochs frozen at 40, the proposed value, confirmed by the human. Projection: 11.23 GPU hours for the 18-run matrix (`results/pilot_report.md`). | Chosen from measured T4 time only, not from test or CIFAR-10-C results | none; no ResNet run exists yet |
| 2026-11-04 | D9 | No change to the rule. `requirements-local.txt` now pins torch 2.11.0 and torchvision 0.26.0, the versions Colab used in the pilot, at the human's choice. It pinned 2.14.1 and 0.29.2 before. | D9 asks for the same major.minor as Colab; Phase 00 could not check Colab's versions | none |
| 2026-11-04 | none (CONTRACTS C1) | No change to any value here. The human approved adding `results/pilot_logs/` to C1 (reply "Go ahead with your recommendations"), so that the timing and console logs behind `results/pilot_report.md` stay in the repo. | `validation/phase-07.md` noted that these logs were not in the repo, and the timing runs were thrown away | none |
