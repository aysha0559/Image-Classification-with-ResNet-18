# Agent Prompts: how to build this project phase by phase

This folder turns `docs/PROJECT_OVERVIEW.md` into a sequence of small, checkable phases.
Each phase has two prompts:

- `BUILD.md`: given to a coding agent to build that phase.
- `VALIDATE.md`: given to a **different, fresh** agent session to check the work independently.

A phase is done only when its validation report says **PASS**. Then you move to the next phase.

## Files in this folder

| File | What it is | Who reads it |
|---|---|---|
| `RULES.md` | Rules every agent must follow in every phase (scope, data leakage, no invented numbers, stop-and-ask). | Every build and validate agent, first. |
| `DECISIONS.md` | Frozen choices the overview left open (seen/unseen split, epochs, checkpoint choice, mCE baseline...). Items marked **CONFIRM** need your answer. | Every agent. You, before Phase 00. |
| `CONTRACTS.md` | The shared interfaces: folder layout, config schema, CLI commands, file names, CSV columns. This is what keeps phases aligned. | Every agent. |
| `templates/FIX.md` | Prompt to give the build agent when validation FAILs. | Build agent, after a FAIL. |
| `templates/VALIDATION-report.md` | The fixed format of every validation report and the verdict rules. | Validate agents. |
| `templates/PROGRESS-entry.md` | The handoff block each build agent appends to `PROGRESS.md`. | Build agents. |
| `phase-NN-*/BUILD.md` | The build prompt for a phase. | Build agent. |
| `phase-NN-*/VALIDATE.md` | The validation prompt for a phase. | Validate agent. |
| `FINAL-AUDIT.md` | End-to-end audit once every phase has passed. | Validate agent, at the end. |

## The phases

| # | Folder | Builds | Needs a GPU? | Gate to pass |
|---|---|---|---|---|
| 00 | `phase-00-setup` | Repo skeleton, environment, config loader, seeding, guardrails, Colab setup notebook | No | Skeleton runs; rules in place |
| 01 | `phase-01-data-clean` | CIFAR-10 loaders, fixed 45k/5k split, normalisation | No | Split is fixed, disjoint, balanced |
| 02 | `phase-02-data-corrupted` | CIFAR-10-C download + checksum, loader by (corruption, severity), sample grid | No | Labels line up with the CIFAR-10 test set |
| 03 | `phase-03-train-transforms` | S0/S1/S2 (AugMix)/S3 (our own corruptions) training transforms | No | S3 never touches CIFAR-10-C |
| 04 | `phase-04-models` | Simple CNN and CIFAR ResNet-18 | No | Shapes and parameter counts match the overview |
| 05 | `phase-05-training-loop` | Config-driven trainer, AMP, checkpoint every epoch, resume, logs | No (smoke on CPU) | Resume gives the same result as no interruption |
| 06 | `phase-06-evaluation` | Metrics (acc, macro-F1, degradation, CE, mCE) and the 76-condition evaluator | No (subset on CPU) | Metrics match scikit-learn and hand examples |
| 07 | `phase-07-pilot` | One full real run: CNN + S1, seed 0, trained and evaluated | **Yes** | Clean accuracy in the expected range; epoch counts frozen |
| 08 | `phase-08-experiment-matrix` | All 6 configs x 3 seeds, run registry, Colab orchestration | **Yes** (you run sessions) | 18 runs complete, 18 x 76 rows, no gaps |
| 09 | `phase-09-analysis` | Aggregated tables and the 8 figures | No | Every figure and table rebuilds from the CSVs |
| 10 | `phase-10-report` | Research report + README, numbers filled from tables by script | No | Every number in the report traces to a CSV |
| 11 | `phase-11-deploy` | Makefile, Docker, one-click Colab, results viewer + live demo on Hugging Face Spaces | No | A stranger can see results in minutes |

Phases 00-06 are code and can be built and validated on a laptop CPU using small subsets.
Phases 07-08 need real GPU time (Google Colab). The agent prepares everything; you start the Colab sessions.

## The loop for each phase

```
1. New branch:            git checkout -b phase-NN-<name>
2. Build session:         paste phase-NN-*/BUILD.md into a fresh agent session
3. Agent finishes:        it appends a block to PROGRESS.md and commits
4. Validate session:      paste phase-NN-*/VALIDATE.md into a DIFFERENT fresh session
5. Read the report:       validation/phase-NN.md
      PASS             -> merge to main, tag phase-NN-pass, next phase
      PASS WITH NOTES  -> read the notes; fix now or log them in PROGRESS.md "Known issues"; then merge
      FAIL             -> paste templates/FIX.md (with the report) into the build session, then validate again
```

Why a different session for validation: an agent checking its own work tends to confirm it.
A fresh session only sees the code, the contracts and the results, and re-runs things itself.

How to start a session with Claude Code (from the project folder):

```bash
cd ~/Desktop/cifar10-robustness-study
claude
# then paste:  Read docs/agent-prompts/phase-00-setup/BUILD.md and do what it says.
```

Any other coding agent works the same way: give it the prompt file's contents.

## Before you start Phase 00

1. Read `DECISIONS.md` and answer every item marked **CONFIRM** (edit the file: change `CONFIRM` to `CONFIRMED` or write your change).
2. Have a Google account with Google Drive space (about 4 GB for data, plus about 2 GB for checkpoints).
3. For Phase 11 only: a GitHub account and a Hugging Face account. The agent will ask for these when it needs them; it never invents accounts or tokens.

## What keeps the agents on track

- **One phase, one scope.** Each BUILD prompt lists what is in and out. Work outside it is refused and noted.
- **Shared contracts.** File names, CLI flags and CSV columns are fixed in `CONTRACTS.md`. Changing them requires updating that file first and saying so in `PROGRESS.md`.
- **Frozen decisions.** Scientific choices live in `DECISIONS.md`; agents do not re-decide them.
- **No invented numbers.** Results come only from files the code produced. The report fills numbers by script (Phase 10). Validators re-run commands instead of trusting claims.
- **Leakage guard.** CIFAR-10-C is test-only; validators grep for it in training code every phase from 03 onward.
- **Handoff log.** `PROGRESS.md` records what each phase did, the commands to re-run it, and known issues, so the next agent starts from facts.
- **Git tags per passed phase.** If a later phase goes wrong, you can return to `phase-NN-pass`.

## What I added beyond the overview

- `DECISIONS.md`: the overview leaves several choices open (exact seen/unseen split, which checkpoint is evaluated, epochs, AugMix JSD on/off, mCE baseline over seeds, Python/torch versions on Colab). Leaving them open is how agents drift. They are proposed there for you to confirm.
- `CONTRACTS.md`: one place that defines the folder layout, config keys, CLI flags, and CSV schema, so Phase 06 output is exactly what Phase 09 reads.
- Checksums and a label-alignment check for CIFAR-10-C (catches a broken or reordered download).
- A run registry (`results/runs.csv`) with git commit, versions, device and time for every run.
- Report numbers filled from a JSON by script, so the report cannot contain a number the experiments did not produce.
- A pilot gate (Phase 07) that measures real Colab time before the 18-run matrix and freezes epoch counts.
- Phase 11 (deployment): the overview lists "deploying an app" as out of scope and an interactive demo as an extension. You asked for an easy way to see results in the cloud, so Phase 11 adds a results viewer and demo as a showcase, not a product. Consider updating overview section 11 to match.
- `FINAL-AUDIT.md`: a clean-clone reproduction check and a number-by-number report check at the end.
