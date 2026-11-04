# Contracts: the interfaces every phase must respect

Phases are built by different agent sessions. These definitions are how their pieces fit.
Do not rename anything here without the human's approval and an update to this file first.

## C1. Folder layout

```
cifar10-robustness-study/
├── README.md
├── PROGRESS.md                     # handoff log, one entry per phase
├── CLAUDE.md                       # points agents at docs/agent-prompts/RULES.md
├── requirements.txt                # pinned, everything except torch/torchvision
├── requirements-local.txt          # torch + torchvision pins for local runs
├── pyproject.toml                  # ruff + pytest settings, package metadata
├── Makefile                        # Phase 11
├── Dockerfile                      # Phase 11 (CPU, serves the app)
├── Dockerfile.gpu                  # Phase 11 (optional, training on a GPU VM)
├── .dockerignore                   # Phase 11
├── README.template.md              # Phase 10, filled by build_report
├── LICENSE
├── .gitignore
├── docs/
│   ├── PROJECT_OVERVIEW.md         # the research plan
│   ├── agent-prompts/              # this folder
│   └── CLOUD.md                    # Phase 11
├── configs/
│   ├── cnn_standard.yaml           cnn_augmix.yaml          cnn_corruption_aware.yaml
│   ├── resnet18_standard.yaml      resnet18_augmix.yaml     resnet18_corruption_aware.yaml
│   ├── cnn_none.yaml               resnet18_none.yaml       # optional S0
│   └── smoke.yaml                  # tiny CPU run used by validators
├── splits/
│   ├── cifar10_val_indices.json    # Phase 01, committed
│   ├── cifar10_stats.json          # Phase 01, committed
│   └── cifar10c_checksums.json     # Phase 02, committed
├── src/
│   ├── __init__.py
│   ├── config.py                   # load + validate YAML, apply CLI overrides
│   ├── utils.py                    # seeding, device, checkpoint io, git commit, timers
│   ├── constants.py                # class names, corruption lists, families, seen/unseen
│   ├── data/
│   │   ├── __init__.py
│   │   ├── cifar10.py              # clean loaders + split
│   │   ├── cifar10c.py             # CIFAR-10-C loader (test only)
│   │   ├── corruptions.py          # our own corruptions for S3
│   │   └── transforms.py           # build_train_transform(strategy), build_eval_transform()
│   ├── models/
│   │   ├── __init__.py             # build_model(name)
│   │   ├── simple_cnn.py
│   │   └── resnet.py
│   ├── train.py
│   ├── evaluate.py
│   ├── metrics.py
│   └── analysis.py
├── scripts/
│   ├── download_cifar10c.sh        # calls python -m scripts.download_cifar10c
│   ├── download_cifar10c.py
│   ├── make_sample_grid.py
│   ├── run_all.sh
│   ├── registry.py                 # build results/runs.csv from runs/
│   ├── plan.py                     # Phase 08: matrix status + remaining GPU hours
│   ├── collect_results.py          # Phase 08: import results from a Drive download
│   ├── export_checkpoints.py       # Phase 11
│   ├── publish_hf.py               # Phase 11, only with the human's go-ahead
│   ├── make_figures.py
│   └── build_report.py
├── notebooks/
│   ├── 00_colab_setup.ipynb
│   ├── 01_explore_data.ipynb
│   ├── 02_analysis.ipynb
│   └── run_on_colab.ipynb          # Phase 08 / 11: one-click stages
├── results/                        # committed (small files only)
│   ├── runs.csv
│   ├── pilot_report.md             # Phase 07
│   ├── RUNLOG.md                   # Phase 08, one entry per Colab session
│   ├── raw/{run_id}.csv
│   ├── raw/{run_id}_confusion.npz
│   ├── raw/{run_id}_train_log.csv
│   ├── tables/
│   └── figures/
├── report/
│   ├── report.template.md
│   ├── report.md                   # generated
│   └── report.pdf                  # optional
├── app/                            # Phase 11: streamlit_app.py, requirements.txt, README.md (Space header), assets/
├── validation/phase-NN.md          # validation reports
└── tests/
```

Not committed: `data/`, `runs/`, `dist/`, `results/smoke/`, `*.pt`, `.env`, `__pycache__/`, `.ipynb_checkpoints/`.

## C2. Environment variables

| Name | Meaning | Default |
|---|---|---|
| `DATA_DIR` | Where CIFAR-10 and `CIFAR-10-C/` live | `./data` |
| `RUNS_DIR` | Where run folders (checkpoints, logs) are written | `./runs` |
| `RESULTS_DIR` | Where result CSVs are written | `./results` |

On Colab: `DATA_DIR=/content/data`, `RUNS_DIR=/content/drive/MyDrive/cifar10-robustness/runs`.

## C3. Names

| Thing | Allowed values |
|---|---|
| `model` | `simple_cnn`, `resnet18` |
| `strategy` | `none`, `standard`, `augmix`, `corruption_aware` |
| short model in run id | `cnn`, `resnet18` |
| `run_id` | `{short_model}-{strategy}-s{seed}`, for example `cnn-standard-s0`, `resnet18-augmix-s2`. Smoke runs add a `smoke-` prefix: `smoke-cnn-standard-s0`. |
| corruption names | exactly the CIFAR-10-C file stems: `gaussian_noise shot_noise impulse_noise defocus_blur glass_blur motion_blur zoom_blur snow frost fog brightness contrast elastic_transform pixelate jpeg_compression` (+ extras `speckle_noise gaussian_blur spatter saturate`) |
| families | `noise`: gaussian_noise shot_noise impulse_noise; `blur`: defocus_blur glass_blur motion_blur zoom_blur; `weather`: snow frost fog; `digital`: brightness contrast elastic_transform pixelate jpeg_compression; `extra`: the 4 extras |
| class names (index order) | `airplane automobile bird cat deer dog frog horse ship truck` |

All of these live in `src/constants.py`. The seen/unseen lists and the three analysis groups come from `DECISIONS.md` D6 and also live there.

## C4. Config schema (YAML)

```yaml
model: resnet18                 # C3
strategy: augmix                # C3
seed: 0                         # overridden by --seed
epochs: 40
batch_size: 128
optimizer:
  name: sgd
  lr: 0.1
  momentum: 0.9
  nesterov: true
  weight_decay: 0.0005
scheduler:
  name: cosine
amp: true                       # ignored on CPU
val_size: 5000
split_seed: 42
num_workers: 2
device: auto                    # auto | cpu | cuda
subset: null                    # int: use only the first N training and N validation images (smoke only)
strategy_params:
  augmix:
    severity: 3
    mixture_width: 3
    chain_depth: -1
    alpha: 1.0
    all_ops: false
  corruption_aware:
    p: 0.5
    corruptions: [gaussian_noise, defocus_blur, contrast, jpeg_compression]
    severities: [1, 2, 3, 4, 5]
```

`src/config.py` rejects unknown keys and wrong types with a clear error. Base configs differ only in `model`, `strategy` and `epochs`.

## C5. Command-line entry points

```bash
python -m src.train --config configs/cnn_standard.yaml --seed 0 [--epochs N] [--subset N] [--run-id ID] [--no-resume]
python -m src.evaluate --run-dir $RUNS_DIR/cnn-standard-s0 [--subset N] [--include-extra] [--batch-size 512]
python -m scripts.download_cifar10c [--data-dir DIR]          # download + verify checksums + extract
python -m scripts.make_sample_grid [--index 7] [--out results/figures/fig7_sample_grid.png]
python -m scripts.registry                                     # rebuild results/runs.csv
python -m src.analysis                                         # write results/tables/*
python -m scripts.make_figures                                 # write results/figures/*
python -m scripts.build_report                                 # report.template.md -> report.md
bash scripts/run_all.sh [--models ...] [--strategies ...] [--seeds ...]   # skip finished runs
```

`src.train` resumes automatically from `checkpoint_last.pt` if the run folder exists, unless `--no-resume`.
`src.evaluate` refuses to overwrite an existing results CSV unless `--force`.
`--subset` always prefixes the run id with `smoke-` and results go to `$RESULTS_DIR/smoke/`, never to `results/raw/`. `--epochs` below the config's value does the same, so a shortened run can never take a real run id. Epoch counts that change for real are edited in the configs, not passed with `--epochs`.

## C6. Run folder: `$RUNS_DIR/{run_id}/`

| File | Content |
|---|---|
| `config_resolved.yaml` | The final config after CLI overrides |
| `train_log.csv` | Columns: `epoch, lr, train_loss, train_acc, val_loss, val_acc, epoch_time_s` (accuracies as fractions; `epoch` starts at 1) |
| `checkpoint_last.pt` | `{model, optimizer, scheduler, scaler, epoch, rng_states, config}`; written every epoch, atomically (write to temp then rename) |
| `model_final.pt` | `{state_dict, config, run_id}` after the last epoch |
| `run_meta.json` | `run_id, model, strategy, seed, epochs, git_commit, git_dirty, torch_version, torchvision_version, python_version, device_name, n_params, started_at, finished_at, train_time_s, status` (`status`: `running`, `finished`, `failed`). Failed runs only: `error`, the error message; removed when a later session of the run starts. |

## C7. Evaluation output

`results/raw/{run_id}.csv`, one row per condition (76 rows by default):

| Column | Type | Notes |
|---|---|---|
| `run_id` | str | |
| `model` | str | C3 |
| `strategy` | str | C3 |
| `seed` | int | |
| `corruption` | str | `clean` for the clean set |
| `family` | str | `clean`, `noise`, `blur`, `weather`, `digital`, `extra` |
| `group` | str | `clean`, `seen` or `unseen`, from DECISIONS D6. Every strategy gets the same labels, so all strategies can be split the same way. |
| `severity` | int | 0 for clean, 1-5 otherwise |
| `n_images` | int | 10000 (smaller only in smoke) |
| `accuracy` | float | fraction |
| `macro_f1` | float | fraction |
| `loss` | float | mean cross-entropy |
| `git_commit` | str | |
| `evaluated_at` | str | ISO 8601 UTC |

`results/raw/{run_id}_confusion.npz`: keys `{corruption}__s{severity}` (and `clean__s0`), each a 10x10 int array, rows = true class, columns = predicted.

`results/raw/{run_id}_train_log.csv`: a copy of the run's `train_log.csv` (so figures can be built without the run folders).

## C8. Run registry: `results/runs.csv`

One row per run: all `run_meta.json` fields + `has_results` (bool) + `clean_accuracy` (from the results CSV, empty if none). Rebuilt by `python -m scripts.registry`; never edited by hand.

## C9. Analysis outputs

`results/tables/`:
- `main_results.csv` and `.md`: per (model, strategy), mean and std over seeds of: clean_acc, clean_f1, mean_corrupted_acc, mean_corrupted_f1, mean_abs_degradation, mean_rel_degradation, mCE, relative_mCE.
- `per_corruption.csv`: per (model, strategy, corruption), mean/std accuracy over severities and seeds, plus CE.
- `per_severity.csv`: per (model, strategy, severity 0-5).
- `per_family.csv`: per (model, strategy, family).
- `seen_unseen.csv`: per (model, strategy, group) with groups `seen`, `unseen_same_family`, `unseen_other_family`.
- `report_numbers.json`: flat `{key: value}` map of every number the report uses (see Phase 10).

`results/figures/` (PNG, 200 dpi, plus PDF):
`fig1_main_table.png`, `fig2_acc_vs_severity.png`, `fig3_per_corruption_heatmap.png`, `fig4_per_family.png`, `fig5_seen_unseen.png`, `fig6_confusion.png`, `fig7_sample_grid.png`, `fig8_training_curves.png`.

## C10. Python interfaces

```python
# src/data/cifar10.py
def get_split_indices(data_dir: str, val_size: int = 5000, split_seed: int = 42) -> tuple[list[int], list[int]]
def get_train_val_datasets(cfg) -> tuple[Dataset, Dataset]
def get_test_dataset(data_dir: str) -> Dataset
def get_normalization_stats() -> tuple[tuple[float, float, float], tuple[float, float, float]]

# src/data/cifar10c.py
def load_cifar10c(data_dir: str, corruption: str, severity: int) -> tuple[np.ndarray, np.ndarray]   # uint8 (10000,32,32,3), int64 (10000,)
class CIFAR10C(Dataset): ...   # (corruption, severity, transform)

# src/data/corruptions.py
def apply_corruption(img: np.ndarray, name: str, severity: int, rng: np.random.Generator) -> np.ndarray   # uint8 HWC in, uint8 HWC out
class RandomSeenCorruption:   # torchvision-style transform on PIL images, uses p and the seen list

# src/data/transforms.py
def build_train_transform(strategy: str, params: dict) -> Callable
def build_eval_transform() -> Callable

# src/models/__init__.py
def build_model(name: str, num_classes: int = 10) -> nn.Module

# src/metrics.py
def accuracy(y_true, y_pred) -> float
def macro_f1(y_true, y_pred) -> float
def absolute_degradation(clean: float, corrupted: float) -> float
def relative_degradation(clean: float, corrupted: float) -> float
def corruption_errors(df_run, df_baseline) -> dict[str, float]     # CE per corruption
def mce(df_run, df_baseline) -> float
def relative_mce(df_run, df_baseline) -> float
```
