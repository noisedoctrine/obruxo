# Archived diagnostic sources

`run_routing_check.py` and `run_routing_training_check.py` are current portable checks for [ordered connections](../../ORDERED_ROUTING.md). They write new output directories and fixed plans before native work. The first checks the complete declared vocabulary and all slot positions; the second runs bounded original-audio training and five validation families. Neither uses the historical layout below.

`run_identity_screen.py` runs the current frozen paired-identity feasibility screen from the checkout. It writes a plan before rendering, shares a bounded feature bank across both arms and measures native reconstruction plus reserved-performance representations. See [PAIRED_IDENTITY.md](../../PAIRED_IDENTITY.md).

`qualify_alpha_imports.py` is the current portable factory-import check, run directly from the checkout. It compares three pinned original exports with the imported alpha assets using native audio. Its arguments and validation boundary are documented in [ALPHA.md](../../ALPHA.md); it does not use the historical directory layout below.

These are the exact scripts used for the existing-artifact analysis and train-only feature probe on 2026-10-15 (Malaysia time). They preserve the original source bytes, including the script hash in `../data/cached_head_probe_plan.json`.

They use the historical local layout: execute copies from `ARTIFACT_ROOT/tmp/`, with this code checkout at `ARTIFACT_ROOT/tmp/training-pipeline-pr/` and the ignored training outputs at `ARTIFACT_ROOT/research/training/outputs/`. They write their diagnostic JSON into the checkout. Reproduction requires the pinned local checkpoints, pools and completed evaluation artifacts, plus the Python dependencies documented in `../../TRAINER.md`. Do not overwrite the frozen results when reproducing them; use a separate checkout and artifact root.

The subsequent native intervention has a separate, portable runner, `../../release_probe.py`, which accepts `--artifact-root`, `--plan`, `--output` and `--resume`. See [the experiment report](../RELEASE_DIAGNOSTIC_RESULTS.md) for its exact invocation, frozen identities and interpretation.

## Sound-helper evidence

`summarize_sound_proxy.py --run /path/to/completed/screen --tests-passed N` checks the completed plan and source hashes, then publishes the sound-helper report, frozen plan, family summaries and a descriptive bootstrap over validation patches. Supply the actual completed local-test count.
