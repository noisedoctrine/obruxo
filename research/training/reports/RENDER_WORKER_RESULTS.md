# Render-worker results — 31 October 2026

Mean whole-request time fell from **1.643s to 0.752s (54.2% less time)** in 33 adjacent pairs with alternating execution order. Cold starts and recycling are included. All 66 attempts passed QA, and all 33 pairs matched float32 audio, state readback and provenance exactly.

The check covers 6 basic unison/filter combinations, held notes and chords, A–B–A ordering, fixed oscillator/sampler phases, 32,768 frames of preroll and recycling after 32 jobs. Peak sampled process-tree memory was 606.6 MiB, with no sampling errors or surviving children.

The implementation reuses the Python process, renderer setup and immutable state template. Each clip gets a fresh engine/plugin and full state readback. Reusing the plugin itself failed exact reproducibility in a six-request diagnostic (largest absolute sample difference 8.74e-10); that implementation was removed. No tolerance was relaxed.

## Remaining cost

A separate instrumented run reproduced all 12 saved source clips exactly. Its means were 1.738s fresh and 0.891s reused. The table is time per reused request; the engine render is included in the audio-render total.

| Work | Seconds |
| --- | ---: |
| Create plugin | 0.365 |
| Save/load plugin state | 0.168 |
| Validate request | 0.091 |
| Convert state | 0.026 |
| Audio rendering, preroll and copying | 0.060 |
| Native engine render alone (included above) | 0.059 |

Plugin creation and state handling remain the main render costs. Actual audio synthesis is already faster than real time. The earlier saved-update profile measured about 0.36s of network work per update; this change does not alter the network or its objective. A full training run has not been repeated, so the paired render speedup is not a measured whole-training speedup.

## Integration and validation

New online streams and NativeSession collections/fits use one renderer per worker, recycled after 32 jobs by default. The duplicate one-shot worker implementation was removed; fresh correctness checks use the same worker with a one-job lifetime. 109 focused regression tests passed, covering rejection, crash/deadline/cancellation recovery, cleanup, backpressure and optimizer/RNG recovery. Static correctness checks passed. GitHub Actions remains disabled.

The new lifecycle is recorded in stream contracts and native receipts. Historical runs use their recorded source revision; they cannot silently resume under this policy. This qualifies the recorded local Windows basic-control cases, not other native versions/platforms, arbitrary patches or model quality. Random-phase training remains stochastic.

[Usage](../RENDER_WORKER.md) · [Plan](data/render_worker_plan.json) · [Measurements](data/render_worker_summary.json)
