# Rules for every agent (build and validate)

Read this whole file before doing anything. These rules win over anything else in a phase prompt, except an explicit instruction from the human in the chat.

## 1. Sources of truth, in order

1. The human's explicit instruction in this session.
2. `docs/agent-prompts/DECISIONS.md` (frozen scientific choices).
3. `docs/agent-prompts/CONTRACTS.md` (folder layout, config keys, CLI, file names, CSV columns).
4. `docs/PROJECT_OVERVIEW.md` (the research plan).
5. The current phase's `BUILD.md` or `VALIDATE.md`.
6. `PROGRESS.md` (what earlier phases claim they did; verify, do not assume).

If two of these disagree, **stop and tell the human** which lines disagree. Do not pick one silently.

## 2. Scope

- Work only on the current phase. Each `BUILD.md` lists what is in and out of scope.
- Do not "improve" code from earlier phases unless the current phase needs it. If you must change it, say so in your `PROGRESS.md` entry under "Changed from earlier phases", with the reason.
- Do not add features, models, strategies, metrics, datasets or dependencies that are not in the overview, decisions or contracts. Put ideas under "Ideas for later" in `PROGRESS.md`.
- Do not change a name, flag, file path or CSV column defined in `CONTRACTS.md`. If one is truly wrong, stop and propose the change to the human; after approval, edit `CONTRACTS.md` first, then the code.

## 3. Data leakage (the most important scientific rule)

- **CIFAR-10-C is test-only.** It is read only by `src/data/cifar10c.py`, and that module is imported only by `src/evaluate.py`, `scripts/make_sample_grid.py` and the demo app's asset builder (Phase 11). Analysis reads result CSVs, not images. Never by training code, transforms, or model selection.
- The CIFAR-10 **test** set is used only for final evaluation. Model monitoring during training uses the 5,000-image validation split.
- Corruption-aware training (S3) uses **our own** corruption functions in `src/data/corruptions.py`, applied to training images. It never loads CIFAR-10-C files or copies its arrays.
- No hyperparameter may be chosen by looking at test or CIFAR-10-C results.

## 4. No invented facts or numbers

- Never write a result, accuracy, time, parameter count, checksum, version or URL you did not get from running code or reading an authoritative source in this session. If you do not know, write `UNKNOWN` and say how to find it.
- Every number in `PROGRESS.md`, the README or the report must come from a file the code produced (log, CSV, JSON) or a command whose output you show.
- Never claim a command passed without running it. Show the command and the important part of its output.
- If a run was cut short (subset, fewer epochs, CPU), label its numbers as **smoke** numbers. Smoke numbers never go into results tables or the report.

## 5. Stop and ask the human when

- A decision in `DECISIONS.md` is still marked `CONFIRM` and your phase depends on it.
- You need an account, token, password, payment, or anything outward-facing (pushing to GitHub, creating a Hugging Face Space, sharing a Drive folder).
- A download fails, a checksum does not match, or a dataset looks different from what the overview describes.
- A result is far outside the expected range in the overview (for example clean CNN accuracy under 70% after full training). Do not tune your way out of it silently.
- You would need to delete or overwrite results, checkpoints or data.

## 6. Code standards

- Python 3.10+. Logic lives in `.py` files under `src/` and `scripts/`. Notebooks only call that code; no model, training or metric logic inside notebooks.
- Every entry point is runnable with `python -m ...` from the repo root as defined in `CONTRACTS.md`.
- Paths come from config or environment variables (`DATA_DIR`, `RUNS_DIR`), never hard-coded to a laptop or Colab path.
- Every random choice is seeded through `src/utils.py`. No unseeded randomness.
- Device is chosen automatically (CUDA if available, else CPU) unless the config sets it.
- Small sanity tests go in `tests/` and run with `pytest -q`. They must run on CPU in under 2 minutes total and must not download CIFAR-10-C (use tiny synthetic arrays).
- Keep functions short and named for what they do. Type hints on public functions. A docstring on each public function.
- Formatting: `ruff format` and `ruff check` pass.

## 7. Git

- One branch per phase: `phase-NN-<name>` (for example `phase-03-train-transforms`).
- Commit when the phase's "Done when" list is met. Commit message: `phase NN: <what was built>`.
- Never commit datasets, checkpoints, `runs/`, `.env` files or tokens. `.gitignore` covers them.
- Do not push, merge or tag unless the human asks.

## 8. Handoff

At the end of every build phase, append an entry to `PROGRESS.md` using `docs/agent-prompts/templates/PROGRESS-entry.md`. Be factual: what exists, how to run it, what was verified (with command output), what is not done, known issues.

## 9. Validators

- You are independent. Do not trust `PROGRESS.md`; re-run the checks.
- Do not fix the code. You may only create `validation/phase-NN.md` (and scratch files outside the repo). Leave it uncommitted; the human commits it with the merge. If a check needs a code change to run, that is a FAIL with a clear reason.
- Report every check with: the command, the relevant output, PASS or FAIL.
- Verdict is one of `PASS`, `PASS WITH NOTES`, `FAIL`. Any failed "must" check means `FAIL`.
