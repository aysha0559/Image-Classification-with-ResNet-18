# Validation: Phase 04 models

- Date: 2026-11-04
- Branch / commit checked: phase-04-models @ 1492e0dbf685ce229875913a7188e35c09073467
- Working tree clean before checking: yes
- Environment: Python 3.12.16, torch 2.14.1+cpu (torchvision 0.29.2+cpu), device cpu
- Verdict: PASS

## Summary
All 14 checks (12 Must, 2 Should) were re-run from scratch and passed. Both models match their specifications layer by layer, the parameter counts are in range and equal the `PROGRESS.md` figures (simple_cnn 666,858; resnet18 11,173,962), and both overfit a fixed batch of 64 real training images within 200 steps on three independent seeds. The notes below do not block the merge. The one gap is in `PROGRESS.md`: one CPU timing value was not captured.

## Checks
| ID | Check | Must/Should | Result | Evidence |
|---|---|---|---|---|
| V1 | Branch, clean tree, commit | Must | PASS | `git rev-parse --abbrev-ref HEAD` -> `phase-04-models`; `git rev-parse HEAD` -> `1492e0d...`; `git status --short` -> empty; `git merge-base --is-ancestor main HEAD` -> true; `git diff --stat main HEAD` -> 6 files (`PROGRESS.md`, `src/models/__init__.py`, `src/models/resnet.py`, `src/models/simple_cnn.py`, `src/utils.py`, `tests/test_models.py`) |
| V2 | Lint, format, tests | Must | PASS | `ruff format --check .` -> `73 files already formatted`; `ruff check .` -> `All checks passed!`; `time pytest -q` -> `167 passed in 5.56s` (real 0m6.746s, CPU) |
| V3 | Simple CNN layer order matches overview 7.1 | Must | PASS | `print(build_model("simple_cnn"))`; 26 leaf modules equal the expected list exactly (`order matches overview+D11: True 26`). Comparison below |
| V4 | Simple CNN parameters in 0.5-1.0 M | Must | PASS | `sum(p.numel() for p in m.parameters())` -> `666858` |
| V5 | ResNet stem | Must | PASS | `conv1`: kernel `(3, 3)`, stride `(1, 1)`, padding `(1, 1)`, 3 -> 64 channels. The only pool module in the whole model is `avgpool` (`AdaptiveAvgPool2d(1)`, after stage 4). Feature shape after `conv1` and stage 1 is still `(2, 64, 32, 32)` |
| V6 | ResNet blocks | Must | PASS | 4 stages, each `n=2` `BasicBlock`; widths 64/128/256/512; first block of stages 2-4 has stride `(2, 2)` and a shortcut of 1x1 stride 2 conv + BatchNorm; all other shortcuts `Identity`. Shapes: 32x32, 16x16, 8x8, 4x4 at stages 1-4, then `Linear(512, 10)` |
| V7 | ResNet parameter count in 11.1-11.3 M | Must | PASS | `sum(p.numel() for p in m.parameters())` -> `11173962` |
| V8 | Not the ImageNet stem | Must | PASS | `grep -rn "torchvision.models\|from torchvision import models\|pretrained\|weights=" src scripts` -> no matches (exit 1); `grep -rn torchvision src/models/` finds no import; `src/models/resnet.py` imports only `torch` and `torch.nn` |
| V9 | Shapes and gradients | Must | PASS | forward (8,3,32,32) -> `(8, 10)` for both; after `loss.backward()`: simple_cnn 19 parameters, resnet18 62 parameters; `grad None: []`, `all-zero grads: []`, `finite: True` for both |
| V10 | Tiny overfit check | Must | PASS | `/tmp/val04/overfit.py`. Table below: all 6 runs reach 100% within 200 steps, in train mode and in eval mode |
| V11 | Unknown name raises | Must | PASS | `build_model("vgg")` -> `ValueError unknown model 'vgg'; allowed: simple_cnn, resnet18` |
| V12 | D11 confirmed and used | Must | PASS | `DECISIONS.md` D11: padding 1, dropout 0.25, dropout 0.5 are all `CONFIRMED`. Code: all 5 convs 3x3 stride 1 padding 1 (`all conv 3x3 stride1 pad1: True 5`); `Dropout(p=0.25)` x3 after conv blocks; `Dropout(p=0.5)` before the last Linear |
| V13 | Determinism of init | Should | PASS | `set_seed(0)`, build twice -> `identical: True` for both models; `set_seed(1)` -> `seed1 differs: True` for both, so the check can detect a difference |
| V14 | `PROGRESS.md` counts match | Should | PASS | `PROGRESS.md` 666,858 and 11,173,962 equal V4 and V7; `count_parameters` returns the same numbers. Lint and test counts also match (73 files, 167 tests) |

### V3 comparison, Simple CNN

| Block | Overview 7.1 | `print(model)` |
|---|---|---|
| 1 | Conv3x3 32, BN, ReLU | `Conv2d(3, 32)`, `BatchNorm2d(32)`, `ReLU` |
| 1 | Conv3x3 32, BN, ReLU, MaxPool, Dropout | `Conv2d(32, 32)`, `BatchNorm2d(32)`, `ReLU`, `MaxPool2d(2)`, `Dropout(0.25)` |
| 2 | Conv3x3 64, BN, ReLU | `Conv2d(32, 64)`, `BatchNorm2d(64)`, `ReLU` |
| 2 | Conv3x3 64, BN, ReLU, MaxPool, Dropout | `Conv2d(64, 64)`, `BatchNorm2d(64)`, `ReLU`, `MaxPool2d(2)`, `Dropout(0.25)` |
| 3 | Conv3x3 128, BN, ReLU, MaxPool, Dropout | `Conv2d(64, 128)`, `BatchNorm2d(128)`, `ReLU`, `MaxPool2d(2)`, `Dropout(0.25)` |
| Head | Flatten, Linear(->256), ReLU, Dropout, Linear(->10) | `Flatten`, `Linear(2048, 256)`, `ReLU`, `Dropout(0.5)`, `Linear(256, 10)` |

### V10 results [smoke]

Recipe: 64 images drawn with `default_rng(1000 + seed)` from `train_indices` only, eval transform (no augmentation), `set_seed(seed)`, Adam lr 1e-3, CPU, at most 200 steps. Train-mode accuracy is read from the logits of the step's training forward pass; eval-mode accuracy is measured after the update. The loop stopped once both had reached 100%.

| Seed | simple_cnn: first 100%, train / eval mode | resnet18: first 100%, train / eval mode |
|---|---|---|
| 0 | 39 / 34 | 4 / 26 |
| 1 | 40 / 44 | 5 / 31 |
| 2 | 36 / 33 | 4 / 33 |

## Failures (if any)
None.

## Notes (non-blocking)
- Choices the sources leave open, all disclosed in `PROGRESS.md`: convs before BatchNorm have no bias, the CNN uses `nn.Dropout` rather than `Dropout2d`, and Kaiming init uses `fan_out`. None contradicts the overview, D11 or BUILD. Dropping the conv bias lowers the Simple CNN count by 320 parameters compared with biased convs, which does not affect V4.
- BUILD's ResNet init was also checked: all 20 BatchNorm layers have weight 1 and bias 0, and a 512-channel conv has std 0.02084 against 0.02083 expected for Kaiming `fan_out`.
- V10 records the first step at 100%; it does not show that accuracy stays there. The Simple CNN wobbles right after that point because of dropout and BatchNorm: seed 2 read 0.9375 in eval mode on the step train mode was 100%, and seed 1 read 0.984375 in train mode on the step eval mode was 100%. The check only requires reaching 100%.
- The D11 row "Expected parameters" still reads `to be measured in Phase 04`. It is not a `CONFIRM` item, and the measured 666,858 agrees with its "about 0.67 M". `PROGRESS.md` already lists this under Known issues. The human can update the row; validators do not edit `DECISIONS.md`.
- The validation prompt places V10's script in `/tmp/val04`; it and the other scratch checks stayed there, outside the repo.

## Claims in PROGRESS.md that could not be confirmed
- Simple CNN CPU timing, run 1: `PROGRESS.md` records it as `not captured (output cut off)`, so BUILD step 2's measurement is only half recorded for that run. Other timing was re-measured and is the same order of magnitude, but timing is not a validation check: mine (median of 5 after discarding the first, batch 128, 4 torch threads) gave simple_cnn 0.275 / 0.305 s and resnet18 3.901 / 4.156 s, against the recorded 0.323 s and 2.711 / 4.330 s. These are [smoke] numbers and do not predict Colab time.
- Exact overfit step counts (simple_cnn 32 / 40, resnet18 4 / 32): not reproducible, because the builder's scratch script and seed are not stored. My independent runs show the same pattern (see V10).

All five mutation claims were reproduced in a scratch copy (`pytest -q tests/test_models.py`; baseline 17 passed): block dropout 0.3 -> 1 failed; conv padding 0 -> 4 failed; blocks (3, 2, 2, 2) -> 5 failed; stem stride 2 -> 5 failed; `fan_in` -> 4 failed.
