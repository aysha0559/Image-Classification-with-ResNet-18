# Phase 09 VALIDATE: Analysis

You are the independent validator for Phase 09. Recompute the key numbers yourself from `results/raw/` with your own code (pandas), and compare. Do not edit repo files except creating `validation/phase-09.md`.

## Read first
`docs/agent-prompts/RULES.md`, `CONTRACTS.md` (C7, C9), `DECISIONS.md` (D6, D8), `docs/PROJECT_OVERVIEW.md` sections 4, 9, 10.3, `docs/agent-prompts/phase-09-analysis/BUILD.md`, `docs/agent-prompts/templates/VALIDATION-report.md`.

## Checks
| ID | Check | Must/Should | How |
|---|---|---|---|
| V1 | Branch, clean tree, commit | Must | git |
| V2 | Lint, format, tests | Must | `ruff format --check . && ruff check . && pytest -q` |
| V3 | Rebuild from scratch | Must | copy the repo to `/tmp/val09/repo`, delete `results/tables` and `results/figures/fig1-6,8`, run `python -m src.analysis && python -m scripts.make_figures`; all C9 files appear; CSV/JSON identical to the committed ones (`diff`) |
| V4 | Analysis reads only raw CSVs | Must | read code: no access to `RUNS_DIR`, checkpoints, or the network |
| V5 | Main table recomputed independently | Must | your own pandas code from D8: for all 6 configs, clean acc, mean corrupted acc, mean abs degradation, mCE, relative mCE (mean and std over seeds); compare with `main_results.csv` within 1e-9 |
| V6 | Baseline mCE | Must | CNN standard seed-mean mCE = 1.0 exactly (100.0%) |
| V7 | Std definition | Must | check which std (`ddof=0` or `1`) is used; it must be stated in the table header and be the same everywhere |
| V8 | Seen/unseen grouping | Must | `seen_unseen.csv` groups equal D6 exactly; recompute 2 rows |
| V9 | `report_numbers.json` correct | Must | for 10 random keys, recompute the value from raw CSVs using the documented formula; every key in the JSON is documented in `report_numbers.md` and vice versa |
| V10 | Partial-data guard | Must | in `/tmp/val09/repo`, delete one raw CSV; `python -m src.analysis` must refuse and name it; with `--allow-partial`, outputs are stamped PARTIAL |
| V11 | Figures exist and are right | Must | open each of the 8 PNGs; for each: what it shows, whether axes, units, legend and title are present and readable, and whether 2 values you read off it agree with the tables |
| V12 | Visual consistency | Should | same colour per strategy and line style per model across figures |
| V13 | "Within seed noise" rule applied | Should | check 2 comparisons in `rq_answers.md` against the rule in BUILD |
| V14 | No prose conclusions or new metrics | Should | scope check |
| V15 | Hand-checks in `PROGRESS.md` | Should | re-run them |

Write `validation/phase-09.md`; give me the one-line verdict.
