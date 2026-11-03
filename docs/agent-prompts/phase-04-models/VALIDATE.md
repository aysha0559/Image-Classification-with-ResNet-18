# Phase 04 VALIDATE: Models

You are the independent validator for Phase 04. Re-run everything. Do not edit repo files except creating `validation/phase-04.md`.

## Read first
`docs/agent-prompts/RULES.md`, `CONTRACTS.md` (C10), `DECISIONS.md` (D11), `docs/PROJECT_OVERVIEW.md` section 7, `docs/agent-prompts/phase-04-models/BUILD.md`, `docs/agent-prompts/templates/VALIDATION-report.md`.

## Checks
| ID | Check | Must/Should | How |
|---|---|---|---|
| V1 | Branch, clean tree, commit | Must | git |
| V2 | Lint, format, tests | Must | `ruff format --check . && ruff check . && pytest -q` |
| V3 | Simple CNN layer order matches overview 7.1 | Must | `print(build_model("simple_cnn"))`; compare layer by layer, write the comparison in the report |
| V4 | Simple CNN parameter count in 0.5-1.0 M | Must | count yourself: `sum(p.numel() for p in m.parameters())` |
| V5 | ResNet stem | Must | first conv is 3x3, stride 1, padding 1, 64 channels; no MaxPool anywhere in the stem |
| V6 | ResNet blocks | Must | 4 stages x 2 BasicBlocks; widths 64/128/256/512; downsampling stride 2 at stages 2-4 with 1x1 projection shortcut |
| V7 | ResNet parameter count | Must | between 11.1 M and 11.3 M (CIFAR ResNet-18 is about 11.17 M) |
| V8 | Not the ImageNet stem | Must | code does not call `torchvision.models.resnet18` |
| V9 | Shapes and gradients | Must | forward (8,3,32,32) -> (8,10); `loss.backward()` gives non-None grads on every parameter |
| V10 | Tiny overfit check | Must | each model reaches 100% training accuracy on a fixed batch of 64 random-but-seeded CIFAR training images within 200 steps of Adam lr 1e-3 on CPU (proves the model can learn; do it in a script in `/tmp/val04`) |
| V11 | Unknown name raises | Must | `build_model("vgg")` |
| V12 | D11 confirmed | Must | `DECISIONS.md` D11 is `CONFIRMED` and the code uses those values |
| V13 | Determinism of init | Should | `set_seed(0)` then build twice -> identical state dicts |
| V14 | `PROGRESS.md` counts match | Should | compare |

Write `validation/phase-04.md`; give me the one-line verdict.
