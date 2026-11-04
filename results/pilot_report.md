# Pilot report: `cnn-standard-s0` on GPU (Phase 07)

Simple CNN, standard augmentation (S1), seed 0, 30 epochs, on a Colab Tesla T4, at commit `004edc3`. The numbers below are real, not smoke, except where marked.

Sources:
- `results/raw/cnn-standard-s0.csv`, `results/raw/cnn-standard-s0_train_log.csv`, `results/runs.csv`.
- `run_meta.json` of each run, and the console logs `cnn-standard-s0.console.log` and `timing.console.log`, downloaded from `MyDrive/cifar10-robustness/results/`.

The timing runs are thrown away after this phase, so their raw epoch times are written out in full in the table below.

## Gates

| Gate (Phase 07 BUILD) | Required | Measured | Result |
|---|---|---|---|
| Clean test accuracy | 0.78 to 0.90 | 0.8404 | pass |
| Drop from clean to mean corrupted | more than 10 points | 24.30 points | pass |
| Mean accuracy falls from severity 1 to 5 | decreasing | 0.7418, 0.6674, 0.6023, 0.5328, 0.4425 | pass |

## Results

Over the 76 conditions of 10,000 images each:

| Metric | Value |
|---|---|
| Clean accuracy | 0.8404 |
| Clean macro-F1 | 0.8391 |
| Mean corrupted accuracy (75 conditions) | 0.5974 |
| Mean corrupted macro-F1 (75 conditions) | 0.5921 |
| Absolute degradation | 0.2430 |
| Relative degradation | 0.2892 |

The 5 worst conditions:

| Corruption | Severity | Accuracy | Macro-F1 |
|---|---|---|---|
| impulse_noise | 5 | 0.1718 | 0.1353 |
| gaussian_noise | 5 | 0.1791 | 0.1458 |
| contrast | 5 | 0.2097 | 0.1408 |
| gaussian_noise | 4 | 0.2134 | 0.1930 |
| shot_noise | 5 | 0.2157 | 0.1997 |

Accuracy by severity, as the mean over the conditions at that severity:

| Severity | Conditions | Accuracy | Macro-F1 |
|---|---|---|---|
| 0 (clean) | 1 | 0.8404 | 0.8391 |
| 1 | 15 | 0.7418 | 0.7404 |
| 2 | 15 | 0.6674 | 0.6673 |
| 3 | 15 | 0.6023 | 0.6012 |
| 4 | 15 | 0.5328 | 0.5264 |
| 5 | 15 | 0.4425 | 0.4254 |

## Training curve

From `results/raw/cnn-standard-s0_train_log.csv`:

- 30 rows, epochs 1 to 30 once each, no NaN values.
- Train loss falls from 2.1748 at epoch 1 to 0.5675 at epoch 30.
- Validation accuracy rises from 0.2584 at epoch 1 to 0.8474 at epoch 30. It is noisy at high learning rates, for example 0.5656 at epoch 8, then settles as the learning rate falls.
- The learning rate matches cosine annealing from 0.1 to 0 at every epoch (`numpy.allclose`).

## Resume after a real disconnect

- Runtime > Disconnect and delete runtime was used during training.
- The output on the page stopped at epoch 6, but the process kept running after that. Epochs 7 to 11 are in the console log.
- It died at epoch 12 with `FileNotFoundError` when writing to Drive, because Drive had gone with the runtime. That traceback is in the console log. It is the old session ending, not a fault in the resumed run.
- The next session printed `cnn-standard-s0: resuming after epoch 11/30`, with no warning, and ran epochs 12 to 30.
- `started_at` stayed at the first session's value.
- `train_time_s` (620.0 s) counts completed epochs only, so the partial epoch 12 of the first session is not in it.

## Time on the actual GPU

All on a Tesla T4, `num_workers: 2`, batch size 128, with AMP. The training times below include validation.

| Config | Epoch 1 (s) | Epoch 2 (s) | Mean per epoch (s) | Source |
|---|---|---|---|---|
| cnn standard | see the log | see the log | 20.667 (all 30 epochs) | `cnn-standard-s0_train_log.csv` |
| cnn augmix | 121.069 | 115.274 | 118.171 | `timing-cnn_augmix` log |
| cnn corruption_aware | 28.943 | 27.416 | 28.180 | `timing-cnn_corruption_aware` log |
| resnet18 standard | 25.558 | 23.123 | 24.340 | `timing-resnet18_standard` log |
| resnet18 augmix | 123.767 | 121.606 | 122.687 | `timing-resnet18_augmix` log |
| resnet18 corruption_aware | 36.078 | 34.459 | 35.268 | `timing-resnet18_corruption_aware` log |

Evaluation on all 76 conditions:

| Model | Wall time (s) | Source |
|---|---|---|
| simple_cnn | 182 | `cnn-standard-s0.console.log` |
| resnet18 | 210 | `timing.console.log`, a 1-epoch `smoke-` model: only the time is used [smoke] |

AugMix is about 5 times slower than standard on ResNet-18 (122.7 s against 24.3 s) and about the same absolute time on the CNN (118.2 s), so its cost is CPU time in the 2 DataLoader workers, not the GPU. This is above the overview's estimate of 1.5 to 3 times.

## Projection for the 18-run matrix

Each of the 6 configs runs 3 times. Hours are mean epoch time x epochs x 3 seeds, using the means above, plus 18 evaluations (9 x 182 s + 9 x 210 s = 0.98 h).

| ResNet-18 epochs | CNN training (30 epochs, h) | ResNet-18 training (h) | Evaluations (h) | Total GPU hours |
|---|---|---|---|---|
| 30 | 4.18 | 4.56 | 0.98 | 9.71 |
| **40 (proposed)** | 4.18 | 6.08 | 0.98 | **11.23** |
| 50 | 4.18 | 7.60 | 0.98 | 12.75 |

- All three are within the overview's estimate of about 20 GPU hours.
- At these rates 20 hours would allow about 97 epochs, far outside the overview's range of 30 to 50.
- The projection leaves out session setup. A second-session setup with the data copy took about 4 minutes in this pilot.
- The ResNet-18 times come from 2 epochs each, and each evaluation time is one measurement.

## Versions

| Package | Colab (`run_meta.json`) | `requirements-local.txt` before | `requirements-local.txt` now |
|---|---|---|---|
| torch | 2.11.0+cu130 | 2.14.1 | 2.11.0 |
| torchvision | 0.26.0+cu130 | 0.29.2 | 0.26.0 |
| Python | 3.13.16 | not pinned there; the local venv is 3.12.16 | same |

The local pins were changed after the pilot to match Colab (D9), at the human's choice.

`pip install -r requirements.txt` on Colab reported one conflict: `google-colab 1.0.0 requires pandas==2.2.3, but you have pandas 2.3.3`. Nothing failed because of it in this pilot.

## Frozen epochs

ResNet-18: **40 epochs**, confirmed by the human and recorded in D4. The projection is 11.23 GPU hours, within the budget of about 20. The choice uses timing only, not any test or CIFAR-10-C result.
