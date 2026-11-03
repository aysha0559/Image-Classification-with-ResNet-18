Append this block to `PROGRESS.md` at the end of a build phase. Fill every heading; write "none" if empty. Facts only.

```markdown
## Phase NN: <name>  (<YYYY-MM-DD>, branch phase-NN-<name>, commit <short sha>)

### Built
- <file or module>: <what it does, one line>

### How to run it
    <exact commands, copy-pasteable from the repo root>

### Verified (with output)
- <check>: `<command>` -> <the relevant output line(s)>

### Numbers produced
- <metric>: <value> (source: <file path>) [smoke | real]

### Changed from earlier phases
- <file>: <change> because <reason>

### Not done / deferred
- <item> -> <which phase or "Ideas for later">

### Known issues
- <issue>, <impact>, <suggested fix>

### Decisions used
- <D-number>: <value used>

### Ideas for later
- <idea>
```
