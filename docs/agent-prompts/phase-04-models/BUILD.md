# Phase 04 BUILD: Models (Simple CNN and CIFAR ResNet-18)

## Read first
1. `docs/agent-prompts/RULES.md`
2. `docs/agent-prompts/DECISIONS.md` D11; `docs/agent-prompts/CONTRACTS.md` C3, C10
3. `docs/PROJECT_OVERVIEW.md` section 7
4. `PROGRESS.md`; `validation/phase-03.md` must be PASS or PASS WITH NOTES

Branch: `phase-04-models` from `main`.

## Goal
Two model definitions that match the overview exactly, built by name through `build_model`.

## In scope
- `src/models/simple_cnn.py`: exactly the architecture in overview 7.1:
  `[Conv3x3 32, BN, ReLU, Conv3x3 32, BN, ReLU, MaxPool2, Dropout] -> [Conv3x3 64, BN, ReLU, Conv3x3 64, BN, ReLU, MaxPool2, Dropout] -> [Conv3x3 128, BN, ReLU, MaxPool2, Dropout] -> Flatten -> Linear(->256) -> ReLU -> Dropout -> Linear(->10)`.
  Padding and dropout rates come from `DECISIONS.md` D11 (stop if still `CONFIRM`). State them in the docstring.
- `src/models/resnet.py`: ResNet-18 (BasicBlock, layers [2, 2, 2, 2], widths 64/128/256/512) adapted for 32x32: first layer Conv3x3 stride 1 padding 1, **no** initial max-pool, global average pool, Linear(512 -> 10). Write it yourself (do not wrap `torchvision.models.resnet18`, whose stem is the ImageNet one). Standard Kaiming init for convs, BN weight 1 / bias 0.
- `src/models/__init__.py`: `build_model(name, num_classes=10)`; unknown name raises `ValueError`.
- `count_parameters(model)` in `src/utils.py`.
- `tests/test_models.py`: forward pass on (4, 3, 32, 32) gives (4, 10) for both; no NaN; backward pass works; `build_model("vgg")` raises; parameter counts within the overview ranges.

## Out of scope
Training loop, pretrained weights, any other architecture.

## Steps
1. Implement. Print each model's parameter count and a layer summary (plain `print(model)` is fine). Record the exact counts in `PROGRESS.md`.
2. Time one forward + backward pass on a batch of 128 on CPU (and GPU if available) and record it.
3. Lint, format, tests. Progress entry. Commit `phase 04: simple CNN and CIFAR ResNet-18`.

## Done when
- [ ] Simple CNN layer order matches overview 7.1 and has between 0.5 and 1.0 million parameters.
- [ ] ResNet-18 has a 3x3 stride-1 stem, no max-pool, [2,2,2,2] BasicBlocks, about 11.2 million parameters (record the exact count).
- [ ] Both give (N, 10) logits; backward works.
- [ ] Padding and dropout equal D11.
- [ ] Lint, format, tests pass.

## Stop and ask me if
The Simple CNN as specified lands outside 0.5-1.0 M parameters (that would mean the overview's estimate and the layer list disagree; do not change the layers on your own).
