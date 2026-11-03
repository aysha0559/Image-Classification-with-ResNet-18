# Phase 10 BUILD: Research report and README

## Read first
1. `docs/agent-prompts/RULES.md` (section 4 on numbers especially)
2. `docs/agent-prompts/DECISIONS.md` (all, including the change log: every change must be disclosed in the report)
3. `docs/agent-prompts/CONTRACTS.md` C9
4. `docs/PROJECT_OVERVIEW.md` sections 1-4, 8, 9, 11, 14, 16, 17, 18, 21
5. `results/tables/*`, `results/tables/report_numbers.md`, `results/figures/*`, `results/pilot_report.md`, `results/RUNLOG.md`
6. `PROGRESS.md`; `validation/phase-09.md` must be PASS or PASS WITH NOTES

Branch: `phase-10-report` from `main`.

## Goal
A 6-10 page research report and a polished README in which **every number comes from `report_numbers.json` by script**, every claim matches the data, and limitations are stated honestly.

## How numbers get into the report (mandatory)
- Write `report/report.template.md` with placeholders like `{{main.resnet18.augmix.clean_acc.mean|pct1}}`. Supported formats: `pct1` (fraction -> `93.4%`), `pp1` (fraction difference -> `12.3 pp`), `f3` (3 decimals), `int`.
- `scripts/build_report.py` (C5) fills the placeholders from `results/tables/report_numbers.json` and writes `report/report.md`. It **fails** on any unknown key and on any digit sequence in the template that is not inside a placeholder, except in an allow-list you define for non-result numbers (years, section numbers, "CIFAR-10", "ResNet-18", "32x32", image counts, hyperparameters from D4, citation years). Print the allow-list hits so a reviewer can see them.
- The same mechanism fills the README results summary (`README.template.md` -> `README.md`).
- If you need a number that is not in `report_numbers.json`, do not compute it in the report script. Stop, add it to `src/analysis.py` (with its documentation line in `report_numbers.md`), re-run analysis, and note it in `PROGRESS.md`.

## Report structure (overview section 14)
Abstract; 1 Introduction and motivation; 2 Related work (only the references in overview section 21, plus any you add after verifying the citation exists; give full citations); 3 Methodology (data and split, models, strategies S1-S3 with exact parameters, seen/unseen design, metrics with formulas); 4 Experimental setup (hardware, epochs, seeds, software versions from `runs.csv`, compute used from `RUNLOG.md`); 5 Results, one subsection per RQ1-RQ5 citing figures/tables, stating for each hypothesis "supported", "partly supported" or "not supported" based on the `rq_answers.md` numbers and the "within seed noise" rule; 6 Discussion (real-world implications from overview 16, written cautiously: this is 32x32 CIFAR-10, not a deployed system); 7 Limitations (single dataset, small images, 3 seeds, final-epoch checkpoint, no JSD, our own S3 corruptions, digital family mixed, any decision change from the change log); 8 Conclusion and future work (overview 19); References.

## README (reviewers often read only this)
Title and one-line summary (overview 17); the key figure (fig2) and the sample grid (fig7); a 6-row results table (filled by script); 3-5 bullet findings (each tied to a number by placeholder); "Reproduce" with exact commands for: view results only, re-run analysis from committed CSVs, re-run one experiment on Colab, re-run everything; repo layout; citation of CIFAR-10, CIFAR-10-C, AugMix; licence. Leave a "Live demo" line with a placeholder link for Phase 11.

## Optional
`report/report.pdf` via pandoc (if installed) or a documented command for the human. Figures embedded with relative paths.

## Writing rules
- Plain, precise English. No hype ("groundbreaking", "state of the art"). Hedge only where the data is genuinely uncertain.
- A hypothesis is "supported" only if the numbers show it beyond seed noise.
- Report negative or surprising results as they are.
- Do not claim anything about ImageNet, real cameras, medicine or cars beyond "may suggest" and a reference.

## Done when
- [ ] `python -m scripts.build_report` succeeds and writes `report/report.md` and `README.md`.
- [ ] No literal result number in the templates outside placeholders (the script enforces it).
- [ ] Every RQ has a verdict consistent with `rq_answers.md`.
- [ ] Every `DECISIONS.md` change-log entry is disclosed.
- [ ] All 8 figures referenced and rendering in the Markdown.
- [ ] Every citation is real and complete (you checked each one in this session).
- [ ] Progress entry; commit `phase 10: report and README`.
