# Phase 02 VALIDATE: CIFAR-10-C download, verification and loader

You are the independent validator for Phase 02. Re-run every check; do not trust `PROGRESS.md`. Do not edit repo files except creating `validation/phase-02.md`.

## Read first
`docs/agent-prompts/RULES.md`, `CONTRACTS.md` (C3, C5, C10), `DECISIONS.md` (D7), `docs/agent-prompts/phase-02-data-corrupted/BUILD.md`, `docs/agent-prompts/templates/VALIDATION-report.md`.

The data should already be in `DATA_DIR`. If it is not, run the download script yourself (that is itself check V3).

## Checks
| ID | Check | Must/Should | How |
|---|---|---|---|
| V1 | Branch, clean tree, commit | Must | git commands |
| V2 | Lint, format, tests | Must | `ruff format --check . && ruff check . && pytest -q` |
| V3 | Download script works and is idempotent | Must | run `python -m scripts.download_cifar10c`; second run must skip the download and only verify |
| V4 | Checksum source is Zenodo, not memory | Must | fetch `https://zenodo.org/api/records/2535967` yourself; compare its checksum with `splits/cifar10c_checksums.json["archive"]` |
| V5 | Extracted files verify | Must | `python -m scripts.download_cifar10c --verify-only`; then corrupt a copy: copy one `.npy` to `/tmp`, point a temp `DATA_DIR` at a folder where one file is changed by one byte, run verify, it must fail loudly |
| V6 | File inventory | Must | 19 corruption files + labels; names equal C3 lists; each uint8 (50000,32,32,3) via `np.load(mmap_mode="r")` |
| V7 | Label alignment | Must | for s in 0..4: `labels[s*10000:(s+1)*10000] == CIFAR10(train=False).targets` |
| V8 | Severity slicing is correct | Must | `load_cifar10c("gaussian_noise", 1)` and `(…, 5)`: images differ; mean absolute difference from the clean test images grows from severity 1 to 5 for at least 11 of the 15 corruptions (compute it; the published data gives 11: glass_blur, motion_blur, frost and elastic_transform are not strictly increasing in this measure, see PROGRESS Phase 02) |
| V9 | Memory-mapped loading | Should | read the code: uses `mmap_mode="r"` and slices |
| V10 | Input validation | Must | bad name, severity 0 and 6 each raise `ValueError` |
| V11 | Leakage guard | Must | `grep -rniE "cifar10c\|CIFAR-10-C" src/train.py src/data/transforms.py src/data/corruptions.py src/models/` is empty; the leakage test fails if you add such an import to a temp copy (do it in `/tmp`, not in the repo) |
| V12 | Sample grid | Must | open `results/figures/fig7_sample_grid.png`; 1 clean + 15 rows x 5 severities, labelled; describe what you see |
| V13 | Colab cells | Should | the setup notebook copies data from Drive to `/content/data` and runs `--verify-only`; no logic beyond calls |
| V14 | Tests use no real CIFAR-10-C | Should | read `tests/test_cifar10c.py` |
| V15 | `PROGRESS.md` claims match | Should | compare |

Write `validation/phase-02.md`; give me the one-line verdict.
