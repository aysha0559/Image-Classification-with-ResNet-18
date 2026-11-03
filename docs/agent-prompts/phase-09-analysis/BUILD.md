# Phase 09 BUILD: Analysis (tables and the 8 figures)

## Read first
1. `docs/agent-prompts/RULES.md`
2. `docs/agent-prompts/DECISIONS.md` D6, D8
3. `docs/agent-prompts/CONTRACTS.md` C7, C9
4. `docs/PROJECT_OVERVIEW.md` sections 4 (RQ1-RQ5), 9, 10.3
5. `PROGRESS.md`; `validation/phase-08.md` must be PASS or PASS WITH NOTES

Branch: `phase-09-analysis` from `main`.

## Goal
Turn the 1,368 result rows into the tables and figures that answer RQ1-RQ5, rebuilt by two commands from committed CSVs only (no run folders, no GPU).

## In scope
- `src/analysis.py` (`python -m src.analysis`), reading only `results/raw/*.csv`:
  - Refuses to run if the 18 expected runs are not all present (prints which are missing), unless `--allow-partial`, in which case every output is stamped "PARTIAL" in its title and file name.
  - Builds the seed-mean baseline table (Simple CNN + standard) per D8, then for every run: clean acc/F1, mean corrupted acc/F1 (75 conditions), mean absolute and relative degradation, CE per corruption, mCE, relative mCE. Then mean and std over the 3 seeds per (model, strategy).
  - Writes every C9 table (`.csv` always; `.md` for `main_results` and `seen_unseen`). Percentages with 1 decimal in `.md`; full precision fractions in `.csv`.
  - `seen_unseen.csv` uses the three groups defined in `DECISIONS.md` D6 (`seen`, `unseen_same_family`, `unseen_other_family`), read from `src/constants.py`. Repeat the mapping in the module docstring.
  - Writes `results/tables/report_numbers.json`: a flat map of every number the report will cite, with keys like `main.resnet18.augmix.clean_acc.mean`, `main.cnn.standard.mce.mean`, `rq1.cnn.standard.abs_drop_sev5`, `rq4.resnet18.corruption_aware.seen_gain_vs_standard`, `rq5.worst_family.resnet18.standard`. Document each key in `results/tables/report_numbers.md` (key, meaning, formula, source table).
  - Answers in data, not prose: for each RQ, a small table in `results/tables/rq_answers.md` with the numbers that support or refute the hypothesis (the words are written in Phase 10).
  - Statistical care: report mean ± std over 3 seeds; for strategy comparisons, also report the per-seed differences (paired by seed), and say "within seed noise" when the mean difference is smaller than the larger of the two stds. No p-values from 3 seeds.
- `scripts/make_figures.py` (`python -m scripts.make_figures`) producing the 8 C9 figures (PNG 200 dpi + PDF), reading only `results/tables/` and `results/raw/`:
  1. Main results table as an image.
  2. Accuracy vs severity (0-5), one line per (model, strategy), mean with a ±std band.
  3. Per-corruption heatmap (15 rows x 6 configs), accuracy averaged over severities and seeds, numbers in cells.
  4. Per-family absolute degradation bars by strategy, one panel per model.
  5. Seen vs unseen: S1 vs S2 vs S3 accuracy on the 3 groups, per model.
  6. Confusion matrices for ResNet-18 S1 seed 0: clean vs fog s5 vs gaussian_noise s5 (row-normalised, class names).
  7. Sample grid (already made in Phase 02; regenerate only if missing).
  8. Training curves: train loss and val accuracy per epoch, mean over seeds, per config.
  - One consistent colour per strategy and line style per model across all figures; colour-blind-safe palette; readable at A4 width; axis labels with units (%).
- `notebooks/02_analysis.ipynb`: displays the tables and figures by calling the two commands.

## Out of scope
Writing prose conclusions (Phase 10). New metrics or runs.

## Steps
1. Implement, run both commands, look at every figure yourself and fix legibility problems.
2. Hand-check 3 numbers from `main_results.csv` by recomputing them from the raw CSVs in a separate snippet; show the snippet output in `PROGRESS.md`.
3. Delete `results/tables/` and `results/figures/` (except fig7), re-run both commands, confirm the files are byte-identical for CSV/JSON (figures may differ in metadata only).
4. Lint, format, tests (add `tests/test_analysis.py` on a tiny synthetic results set with known answers). Progress entry. Commit `phase 09: analysis tables and figures`.

## Done when
- [ ] All C9 tables and 8 figures exist and rebuild from committed CSVs with the two commands.
- [ ] `report_numbers.json` and its key documentation exist.
- [ ] Baseline mCE (CNN standard, seed mean) is 100.0%.
- [ ] Three hand-checked numbers match.
- [ ] Rebuild is deterministic.
- [ ] Every figure checked by eye; titles, labels, legends readable.
- [ ] Lint, format, tests pass.
