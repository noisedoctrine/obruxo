# Archived diagnostic sources

These are the exact scripts used for the existing-artifact analysis and train-only feature probe on 2026-10-15 (Malaysia time). They preserve the original source bytes, including the script hash in `../data/cached_head_probe_plan.json`.

They use the historical local layout: execute copies from `ARTIFACT_ROOT/tmp/`, with this code checkout at `ARTIFACT_ROOT/tmp/training-pipeline-pr/` and the ignored training outputs at `ARTIFACT_ROOT/research/training/outputs/`. They write their diagnostic JSON into the checkout. Reproduction requires the pinned local checkpoints, pools and completed evaluation artifacts, plus the Python dependencies documented in `../../TRAINER.md`. Do not overwrite the frozen results when reproducing them; use a separate checkout and artifact root.

The subsequent native intervention has a separate, portable runner, `../../release_probe.py`, which accepts `--artifact-root`, `--plan`, `--output` and `--resume`. See [the experiment report](../RELEASE_DIAGNOSTIC_RESULTS.md) for its exact invocation, frozen identities and interpretation.
