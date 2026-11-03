Validators write `validation/phase-NN.md` in exactly this format.

```markdown
# Validation: Phase NN <name>

- Date: <YYYY-MM-DD>
- Branch / commit checked: <branch> @ <full sha>   (from `git rev-parse HEAD`)
- Working tree clean before checking: yes | no (if no, list files from `git status --short`)
- Environment: <python version>, torch <version>, device <cpu|cuda name>
- Verdict: PASS | PASS WITH NOTES | FAIL

## Summary
<2-4 sentences: what was checked, what failed, what matters most>

## Checks
| ID | Check | Must/Should | Result | Evidence |
|---|---|---|---|---|
| V1 | <check> | Must | PASS | `<command>` -> `<key output>` |

## Failures (if any)
### V<id>: <title>
- Expected: <what the contract/prompt requires, with file and line or section>
- Found: <what happened, with output>
- Likely cause: <your reading of the code, file:line>
- Suggested fix: <short>

## Notes (non-blocking)
- <observation>

## Claims in PROGRESS.md that could not be confirmed
- <claim> -> <what you saw instead>
```

Verdict rules: any failed **Must** check is `FAIL`. Only failed **Should** checks is `PASS WITH NOTES`. All pass is `PASS`.
