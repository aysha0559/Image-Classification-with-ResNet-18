# Phase 00 VALIDATE: Project setup and guardrails

You are the independent validator for Phase 00. You did not build it. Do not trust `PROGRESS.md`; re-run every check.

## Read first
`docs/agent-prompts/RULES.md` (especially section 9), `docs/agent-prompts/CONTRACTS.md`, `docs/agent-prompts/DECISIONS.md`, `docs/agent-prompts/phase-00-setup/BUILD.md`, `docs/agent-prompts/templates/VALIDATION-report.md`.

## Rules for you
- Do not edit any repo file except creating `validation/phase-00.md`.
- Use a fresh virtual environment outside the repo's `.venv` if possible (for example in `/tmp/val-venv`) so you prove the requirements install from scratch.

## Checks
| ID | Check | Must/Should | How |
|---|---|---|---|
| V1 | On branch `phase-00-setup`, tree clean | Must | `git status --short`, `git branch --show-current`, `git log --oneline -5` |
| V2 | Requirements install from scratch | Must | new venv; `pip install -r requirements-local.txt -r requirements.txt` |
| V3 | Every package in `requirements*.txt` is pinned with `==` | Must | read the files |
| V4 | Folder layout matches CONTRACTS C1 | Must | `find . -path ./.git -prune -o -path ./.venv -prune -o -print \| sort` and compare |
| V5 | Lint and format | Must | `ruff format --check . && ruff check .` |
| V6 | Tests pass on CPU under 2 min | Must | `time pytest -q` |
| V7 | All 9 configs validate | Must | loop over `configs/*.yaml` with `src.config.load_config` |
| V8 | Bad config rejected clearly | Must | write a temp YAML outside the repo with key `epoch:` (typo) and one with `lr: "fast"`; both must raise a readable `ValueError` |
| V9 | run_id format | Must | for each of the 6 main configs and seeds 0,1,2, print run_id; compare with C3 |
| V10 | Constants exact | Must | compare `src/constants.py` lists to C3 and D6 item by item; seen and unseen disjoint, union = 15 |
| V11 | Config values match decisions | Must | for each main config compare optimizer, lr, wd, batch, epochs, AugMix and S3 params against D4/D5 |
| V12 | Seeding works | Must | run `set_seed(0)` then draw from python `random`, numpy, torch; repeat; values identical |
| V13 | Ignore rules | Must | `git check-ignore -v data/x runs/x a.pt .env` |
| V14 | No hard-coded machine paths | Must | `grep -rnE "/home/|/content/|C:\\\\" src scripts configs` must be empty (notebooks may use `/content`) |
| V15 | Colab notebook has no logic | Must | read cells: only mount, clone, pip, env vars, version prints |
| V16 | `CLAUDE.md` points to the 4 agent files | Should | read it |
| V17 | Nothing out of scope built | Should | no data loading, model, training code beyond docstrings |
| V18 | `PROGRESS.md` entry complete and true | Should | compare each claim with what you observed |

Write `validation/phase-00.md` using the template, then give me the verdict in one line.
