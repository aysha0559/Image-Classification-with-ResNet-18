# Run log

One entry per Colab session of the Phase 08 matrix. Facts only: what the session's output or files show. Note every session that was cut off, because a resumed run may differ a little from an uninterrupted one (DECISIONS change log, 2026-11-04).

| Date (UTC) | GPU | Commit | Runs advanced | Errors or interruptions |
|---|---|---|---|---|
| 2026-11-04, from 16:43 | Tesla T4 | `7d334eb` | `cnn-corruption_aware-s0`, `resnet18-standard-s0`, `resnet18-corruption_aware-s0` trained and evaluated; `cnn-augmix-s0` epochs 1-20 | Cut off early. The page showed "Runtime disconnected", stamped 18:30, about 1h 45m in; cause UNKNOWN. The process kept going after that: `matrix.console.log` reaches epoch 20, about 18:36 by epoch times, with no traceback. `cnn-augmix-s0` was interrupted. |
| 2026-11-04, from 18:40 | Tesla T4 | `7d334eb` (from the 18:40:47 session line) | `cnn-augmix-s0` epochs 21-30 and evaluation; `resnet18-augmix-s0`, `cnn-standard-s1`, `cnn-corruption_aware-s1`, `resnet18-standard-s1` trained and evaluated; `resnet18-corruption_aware-s1` trained | First attempt failed at 18:38 with `ValueError: mount failed`; Run all on the same runtime mounted Drive. Cut off during the evaluation of `resnet18-corruption_aware-s1`: `run_meta.json` last changed 21:33, the page showed Reconnect by 21:42, no results CSV reached Drive. Cause UNKNOWN. At 21:59 Colab refused a new GPU: "You cannot currently connect to a GPU due to usage limits in Colab." |
