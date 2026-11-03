# Phase 10 VALIDATE: Report and README

You are the independent validator for Phase 10. Your job is to catch any number, claim or citation the data does not support. Do not edit repo files except creating `validation/phase-10.md`.

## Read first
`docs/agent-prompts/RULES.md`, `DECISIONS.md` (with change log), `docs/agent-prompts/phase-10-report/BUILD.md`, `docs/agent-prompts/templates/VALIDATION-report.md`, `results/tables/*`, `report/report.template.md`, `report/report.md`, `README.md`.

## Checks
| ID | Check | Must/Should | How |
|---|---|---|---|
| V1 | Branch, clean tree, commit | Must | git |
| V2 | Lint, format, tests | Must | `ruff format --check . && ruff check . && pytest -q` |
| V3 | Report rebuilds identically | Must | `python -m scripts.build_report`; `git diff --exit-code report/report.md README.md` |
| V4 | Guard works | Must | in a copy under `/tmp/val10`, add the literal text `accuracy of 91.2%` to the template; the build must fail. Add `{{no.such.key}}`; it must fail |
| V5 | Allow-list is honest | Must | read the allow-list; it may contain only non-result numbers (years, sizes, hyperparameters, section numbers); any accuracy-like value is a FAIL |
| V6 | Every number traces to data | Must | extract every number from `report/report.md` and `README.md` (regex); for each, find its placeholder or allow-list entry; for 15 random result numbers, recompute from `results/raw/` with your own code |
| V7 | RQ verdicts match the data | Must | for each RQ1-RQ5, read the verdict, look at `rq_answers.md`, apply the "within seed noise" rule yourself; list any disagreement |
| V8 | No overclaiming | Must | list sentences that claim more than the data shows (for example causal claims, real-world safety guarantees, "significant" without a test, generalising beyond CIFAR-10) |
| V9 | Limitations complete | Must | all items listed in BUILD plus every change-log entry appear |
| V10 | Method matches what was run | Must | compare the report's methodology numbers (split, epochs, lr, wd, batch, AugMix params, S3 p and severities, seen list) with `DECISIONS.md` and one `config_resolved.yaml` per model |
| V11 | Citations real and complete | Must | check each reference (authors, title, venue, year) against a real source in this session; list unverifiable ones |
| V12 | Figures referenced and render | Should | each `![...](...)` path exists; all 8 figures cited in the text |
| V13 | README reproduce steps work | Must | in a fresh clone in `/tmp/val10/clone`, run the "view results" and "re-run analysis from committed CSVs" commands exactly as written; they must work without a GPU |
| V14 | Length and structure | Should | report sections in the BUILD order; about 6-10 pages (roughly 3,000-5,500 words plus figures) |
| V15 | Plain language | Should | flag hype words and unexplained jargon |

Write `validation/phase-10.md`; give me the one-line verdict.
