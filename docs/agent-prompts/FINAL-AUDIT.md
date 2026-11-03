# Final audit: end to end

Run this in a fresh agent session after Phase 11 has passed. It checks the project as a whole, the way a scholarship reviewer or another researcher would. Do not edit repo files except creating `validation/final-audit.md`.

## Read first
`docs/agent-prompts/RULES.md`, `DECISIONS.md` (with change log), `CONTRACTS.md`, `docs/PROJECT_OVERVIEW.md` (all), `README.md`, `report/report.md`, every `validation/phase-NN.md`, `docs/agent-prompts/templates/VALIDATION-report.md` (use its format).

## Checks
| ID | Check | Must/Should | How |
|---|---|---|---|
| A1 | Every phase passed | Must | each `validation/phase-00..11.md` verdict is PASS or PASS WITH NOTES; each note is either fixed or listed in `PROGRESS.md` "Known issues"; git tags `phase-NN-pass` exist |
| A2 | Overview deliverables (section 14) present | Must | repo, 18 x 76 results, 8 figures, report, README; tick each |
| A3 | Research questions answered | Must | RQ1-RQ5 each has a verdict and evidence in the report |
| A4 | Scope respected (section 11) | Must | nothing out of scope was built except Phase 11's showcase; no adversarial attacks, no pretrained models, no ImageNet |
| A5 | Leakage, whole repo | Must | `grep -rniE "cifar10c\|CIFAR-10-C" src/` appears only in `src/data/cifar10c.py`, `src/evaluate.py`, `src/analysis.py` (names only) and `src/constants.py`; `src/train.py` and `src/data/transforms.py` never import it; the test set is used only by `src/evaluate.py` |
| A6 | Clean-clone reproduction of the analysis | Must | fresh clone in `/tmp/final/clone`; `make setup && make analyze && make report`; `git status` afterwards shows no changes (tables, figures, report identical; figure files may differ only in metadata, check with an image diff if they show as changed) |
| A7 | Spot reproduction of one experiment | Should | if a GPU or a Colab session is available (ask the human), retrain `cnn-standard-s1` from scratch to a scratch `RUNS_DIR` and evaluate; clean and mean corrupted accuracy within about 0.5-1 point of the committed CSV |
| A8 | Number trace, report | Must | 20 random result numbers from `report/report.md` recomputed from `results/raw/` with your own code |
| A9 | Decisions honoured | Must | for every D-item, find where it is implemented and confirm the value in one config/code location |
| A10 | Contracts honoured | Must | folder layout, run ids, CSV columns and CLI flags match `CONTRACTS.md`; list any drift |
| A11 | Honesty | Must | the limitations section covers every change-log entry and every "Known issues" item that affects results |
| A12 | Viewer and Colab paths | Should | the Space (if published) and the Colab badge work |
| A13 | Hygiene | Should | no secrets, no files over 50 MB, licence present, citations complete |
| A14 | Overview still accurate | Should | list places where `docs/PROJECT_OVERVIEW.md` now disagrees with what was built (for example epochs, Phase 11 scope) so the human can update it |

Write `validation/final-audit.md`. End with: the verdict, the top 3 strengths a reviewer would notice, and the top 3 weaknesses to address or disclose.
