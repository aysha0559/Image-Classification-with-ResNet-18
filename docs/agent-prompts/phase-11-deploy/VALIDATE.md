# Phase 11 VALIDATE: Packaging and deployment

You are the independent validator for Phase 11. Behave like a stranger who has only the README. Do not edit repo files except creating `validation/phase-11.md`.

## Read first
`docs/agent-prompts/RULES.md`, `DECISIONS.md` (D10), `docs/agent-prompts/phase-11-deploy/BUILD.md`, `docs/agent-prompts/templates/VALIDATION-report.md`, then only `README.md` and `docs/CLOUD.md` for instructions.

## Checks
| ID | Check | Must/Should | How |
|---|---|---|---|
| V1 | Branch, clean tree, commit | Must | git |
| V2 | Fresh clone works on CPU | Must | `git clone` the repo into `/tmp/val11/clone`; follow the README exactly: `make setup && make check && make smoke`; note every step that needed knowledge not in the README |
| V3 | `make help` | Should | all targets listed with descriptions |
| V4 | Docker build and run | Must | `make docker-build`; record image size; `make docker-app`; `curl -sf http://localhost:7860` (or the Streamlit health endpoint) returns OK |
| V5 | Results page | Must | open the app in a browser (or read the rendered page); main table values equal `results/tables/main_results.csv`; all 8 figures shown |
| V6 | Offline behaviour | Must | run the container with no network (`--network none`) and without `HF_MODEL_REPO`: Results works, Try it shows a clear message, no crash |
| V7 | Demo predictions correct | Must | with checkpoints available (local dir or Hub), pick 3 bundled clean samples and 2 corrupted ones; compute the predictions yourself with `build_model` + the checkpoint + `build_eval_transform`; the app's top-1 and probabilities agree to 1e-4 |
| V8 | Checkpoint integrity | Must | sha256 of each downloaded checkpoint equals `models.json`; `models.json` clean accuracies equal the raw CSVs |
| V9 | No secrets | Must | `git grep -nIE "hf_[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{20,}|password|secret" -- . ':!docs/agent-prompts'` shows nothing sensitive; `.env` not tracked |
| V10 | No large files in git | Must | `git ls-files -z \| xargs -0 du -b \| sort -n \| tail` shows nothing over 50 MB; no `.pt` files tracked |
| V11 | Uploads not stored | Must | read the app: uploaded images live only in memory |
| V12 | Space live (only if published) | Must if published | open the Space URL from the README; both pages work; the Space's commit matches the repo's `app/` |
| V13 | Colab badge | Should | the badge link opens `notebooks/run_on_colab.ipynb` from the public repo |
| V14 | `docs/CLOUD.md` accuracy | Should | each path's commands exist and match the Makefile/Dockerfiles; no unverified provider claims |
| V15 | Docker image contents | Should | `docker run --rm <image> ls -R /app` (or equivalent): no data, runs, checkpoints, `.git`; runs as non-root |

Write `validation/phase-11.md`; give me the one-line verdict and the list of friction points a stranger would hit.
