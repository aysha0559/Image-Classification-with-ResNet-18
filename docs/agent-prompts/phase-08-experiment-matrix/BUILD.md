# Phase 08 BUILD: Full experiment matrix (6 configs x 3 seeds)

## Read first
1. `docs/agent-prompts/RULES.md`
2. `docs/agent-prompts/DECISIONS.md` (all; D4 epochs must be `CONFIRMED`)
3. `docs/agent-prompts/CONTRACTS.md` C2, C5, C6, C7, C8
4. `docs/PROJECT_OVERVIEW.md` sections 10, 12, 18
5. `PROGRESS.md`, `results/pilot_report.md`; `validation/phase-07.md` must be PASS or PASS WITH NOTES

Branch: `phase-08-experiment-matrix` from `main`.

## Goal
All 18 runs trained with identical settings except model, strategy and seed, each evaluated on 76 conditions, with a registry that shows at a glance what is done, running, failed or missing. The pilot run `cnn-standard-s0` counts as one of the 18 and is **not** re-run.

## Who does what
You build the orchestration and check results. The human starts and restarts Colab sessions. Colab free sessions end after a few hours, so the plan must survive any number of interruptions with no lost or duplicated work.

## In scope
- `scripts/run_all.sh` (C5): loops over models x strategies x seeds (defaults: both models, `standard augmix corruption_aware`, seeds `0 1 2`), skips finished+evaluated runs, resumes unfinished ones, evaluates finished-but-unevaluated ones, stops cleanly when a time budget env var `SESSION_BUDGET_MIN` would be exceeded by the next epoch (estimate from the last epoch time), and prints the registry at the end. Order: all seed-0 runs first (so a complete single-seed picture exists early), then seed 1, then seed 2; within a seed, cheapest first.
- `scripts/plan.py`: prints the 18-run table with status from `results/runs.csv` / run folders and the estimated remaining GPU hours (using measured times from finished runs, falling back to the pilot report).
- `notebooks/run_on_colab.ipynb`: add a "run the matrix" mode that calls `run_all.sh` with the budget set from the session, and syncs `results/raw/` and `results/runs.csv` to Drive after **each** evaluated run.
- `scripts/collect_results.py`: copies result files from a Drive download folder (path given by the human) into `results/raw/`, checks each against C7 (76 rows, columns, `n_images`), refuses duplicates with different content, and rebuilds the registry.
- A `results/RUNLOG.md` the human or agent appends to each session: date, GPU type, runs advanced, any errors. (A plain, factual log.)

## Human steps (repeat until done)
1. Open `notebooks/run_on_colab.ipynb` in Colab (GPU runtime), set `STAGE = "matrix"`, run all.
2. When the session ends, open a new one and run again. It resumes.
3. Each time, or at the end, download `MyDrive/cifar10-robustness/results/` and tell the agent the folder path.

## Agent steps after each batch of results
1. `python -m scripts.collect_results --from <path>`.
2. Check every new run: status `finished`, epoch count equals D4, `git_commit` is a commit on `main` at or after the Phase 07 merge, the same commit for all runs if possible (if runs used different commits, list which and confirm with `git diff` that no file under `src/` or `configs/` affecting training or evaluation changed between them).
3. Sanity per run: clean accuracy in the overview range for its model (CNN about 0.80-0.87; ResNet-18 about 0.92-0.95 at full epochs; if fewer epochs were frozen, the pilot report's projection), no NaN, train log length equals epochs.
4. Seed consistency: for each config, clean accuracy across seeds within 1.5 points. If not, flag; do not drop or re-run a seed without the human's decision.
5. Update `PROGRESS.md` with the registry table.

## Rules specific to this phase
- Never change code or configs that affect training or evaluation while the matrix is running. If a bug is found, stop, tell the human, and decide together which runs must be redone; record it in `DECISIONS.md` change log.
- Never delete a run or result. A run to be redone is renamed (for example `_invalid-<reason>` suffix in the run folder) and the reason logged.
- S0 (`none`) runs only if the human asks after the 18 are done.

## Done when
- [ ] 18 run ids present in `results/runs.csv` with status `finished` and `has_results` true.
- [ ] `results/raw/` has 18 x 3 files (CSV, confusion, train log), each passing the C7 check: 18 x 76 = 1,368 result rows in total.
- [ ] Every run's epoch count matches D4; all runs share identical training settings except model/strategy/seed (checked from `config_resolved.yaml`).
- [ ] Seed consistency checked and reported.
- [ ] `results/RUNLOG.md` covers every session.
- [ ] Progress entry with the registry table; commit `phase 08: full experiment matrix results`.
