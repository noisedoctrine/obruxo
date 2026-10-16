# Reusable render-worker qualification

The reusable worker **passed all local correctness, timing and memory gates**. With oscillator phases fixed, all 33 reused-worker outputs matched fresh-process float32 audio, native state and provenance exactly, including the process restart after 32 jobs. All 66 native attempts passed QA.

Mean whole-call time was **2.461s fresh** versus **2.051s reused**, a **+16.7% reduction**. The paired check alternated fresh-first and reuse-first order for adjacent renders of the same patch. It includes cold starts and recycling, rather than comparing one early baseline with a long later batch. Sampled peak process-tree RSS was 640.3 MiB.

## Correctness controls

An initial uncontrolled check mixed pipeline reproducibility with oscillator phase randomization. That was an invalid correctness test: identical settings may legitimately generate different waveforms when random phases change. Its speed numbers are not the final comparison. The corrected gate disables oscillator randomization in diagnostic copies, preserves fixed phase knobs and the actual training pool, and requires exact audio equality on this pinned local runtime.

A separate causal probe accepted 15/15 native attempts. Turning off oscillator phase randomization alone produced identical audio and state for all ten reused renders across five families. The full lifetime check then reproduced this over 32 jobs and a restart. Random-phase variation remains a separate training/sampling concern; no tolerance was loosened to make the reproducibility gate pass.

The pilot and alpha codec correctness checks now use explicit phase-controlled diagnostic variants as well. They compare codec controls and nested state before applying phase controls, register the derived states under training lineage, and save a distinct check identity. Ordinary training patches and audio-quality evaluation requests retain their requested phase settings.

The full local renderer/training command produced **250 passed and one failed**. The affected recovery test then passed in isolation in 15.60s, covering **251 distinct checks** across the full run and targeted rerun. Its first run recorded a simulated render timeout after a 383s scheduling gap while Windows Modern Standby transitions were observed. That error correctly advanced the recipe cursor, so the test no longer had the same patch sequence as its no-error control. No production deadline or correctness tolerance was relaxed.

Dedicated checks cover fresh renderer instances, bounded process lifetimes, QA rejection, crashes, deadlines, cancellation, shared-memory cleanup, queue limits, cursor compatibility and exact optimizer/RNG checkpoint recovery with reuse. Existing one-job contracts and exports remain compatible. GitHub Actions stays disabled.

## Scope

This qualifies the pinned Windows Vital/DawDreamer path on five authored training families with a 32-job lifetime. It does not qualify Linux/Colab, unrestricted full-schema presets, random-phase bit reproducibility or finished-model quality. A sampled RSS maximum over this bounded run is not an unlimited-duration leak test. The previous one-job default remains unchanged; new runs opt in and freeze their worker policy.

[Usage](../RENDER_WORKER.md) · [Frozen plan](data/render_worker_plan.json) · [Aggregate evidence](data/render_worker_summary.json)
