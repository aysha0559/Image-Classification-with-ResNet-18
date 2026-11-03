# Fix prompt (use after a FAIL)

Paste this into the build session for the phase (or a fresh session on the same branch), with the phase number filled in.

---

The independent validation of Phase NN failed. The report is in `validation/phase-NN.md`.

1. Read `docs/agent-prompts/RULES.md`, `docs/agent-prompts/CONTRACTS.md`, `docs/agent-prompts/DECISIONS.md` and `validation/phase-NN.md`.
2. For every FAIL item, find the root cause. Do not change the check to make it pass, and do not weaken a test, a range or a contract. If you believe a check itself is wrong, stop and explain to me why, quoting the check, before changing anything.
3. Fix only what the failures need. No new features, no refactoring outside the failing area.
4. Re-run each failed check yourself and show the command and output.
5. Also re-run the phase's full "Done when" list from `BUILD.md`, since a fix can break something else.
6. Add a "Fix round N" section to your Phase NN entry in `PROGRESS.md`: each failure, its cause, the fix, the evidence.
7. Commit as `phase NN: fix <short description>`.
8. Tell me it is ready for re-validation. Validation must run again in a fresh session.
