# Basic Vital session results — 1 November 2026

The same 126-render training update took **38.34s in the saved baseline and 48.41s with 128 jobs per worker**. Candidate controls and acceptance decisions were preserved. The profiles were separate runs with different host conditions; no full-update speedup is claimed from this comparison.

The adjacent-pair benchmark provides the controlled render comparison: **3.097s fresh versus 0.223s reused**, 92.8% less time across 129 pairs, including cold starts and recycling at 128 jobs. All 258 attempts passed QA and all restart/memory gates passed.

## Changes and measured costs

| Work per 126-render update | Before | After |
| --- | ---: | ---: |
| Create Vital instances | 1.633s | 0.766s |
| Load/apply patch controls | 0.882s | 0.386s |
| Validate requests | 0.427s | 0.208s |
| Whole native worker calls | 23.181s | 28.414s |

The basic profile keeps one engine/plugin per worker, clears MIDI each request, and uses the host's normal render reset. It loads the complete preset initially and updates the eleven controls thereafter. Setter normalization explicitly matches the full loader's float32 arithmetic. Every render still saves and verifies native state, fingerprints the plugin and checks the audio.

Only the reviewed legacy basic domain, 44.1kHz, 128-sample buffers, 32,768-frame preroll, fixed oscillator/sampler phases and unchanged init assets/routes/effects qualify. The shared control definition keeps the codec and renderer bounds aligned. Template changes force revalidation/recreation; other presets use the full loading path in the same renderer.

## Numerical correctness

The fixed-phase paired check found maximum absolute sample error **2.24e-08**, with native readback/provenance identical. 9/129 audio buffers were bit-identical; the remaining pairs passed the explicit numeric limits recorded in the plan. This is numerical equivalence, not a bit-identical claim.

Warm worker median memory changed from 313.17 MiB to 313.78 MiB across the 128-job lifetime, below the predeclared 64 MiB growth limit.

For the complete update, maximum loss difference was **5.96e-08**, gradient coefficient difference **5.96e-07**, and model-weight update difference **2.33e-10**. All 126 requested candidate controls matched the original workload exactly; 7/126 waveform hashes matched exactly. The full trace differs in tiny measured numeric values. Original training artifacts were unchanged.

An initial double-precision setter conversion amplified tiny rounding differences in Adam and failed the update-stability guard. Matching the full loader's float32 arithmetic fixed that failure. The rejected setter and float32-readback-fallback implementations were removed; no optimizer or loss changes were introduced to accommodate them.

41 focused local regression checks passed, covering worker failure/recycling, backpressure, stream contracts and checkpoint recovery. Static correctness lint and whitespace checks passed. GitHub Actions remains disabled.

## Remaining cost and scope

Native worker calls now account for 58.7% of the update. Spectral feature extraction and loss comparisons together take 10.85s. All 120 gradient-probe renders and six acceptance renders per update are retained.

This qualifies the recorded basic Windows workload and numerical bounds. It does not establish finished-model quality or arbitrary-preset/cross-platform equivalence. The v3 lifecycle is frozen in new stream contracts and native receipts; historical runs use their recorded source revision.

[Usage](../RENDER_WORKER.md) · [Plan](data/render_worker_plan.json) · [Measurements](data/render_worker_summary.json)
