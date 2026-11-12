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

A follow-up kept the same 126-render update design, source audio, batch sizes, fixed render phases and loss, and changed the CPU calculation from two threads to four. Feature extraction fell from **5.32s to 3.00s**, and loss calculation from **4.71s to 2.91s**: **10.03s to 5.90s combined**. The full update took 33.88s versus 33.02s, while Vital worker time increased from 16.53s to 19.91s, so this one comparison does not establish a full-update speedup. A cached replay at four threads reproduced the trace and checkpoint exactly. The four-thread update stayed within the existing limits against the two-thread reference and peaked at 2.41 GiB, below the 4 GiB cap. New balanced-fit plans now record the thread count and default to four; `--cpu-threads` selects another value before the plan is prepared.

This qualifies the recorded basic Windows workload and numerical bounds. It does not establish finished-model quality or arbitrary-preset/cross-platform equivalence. The v3 lifecycle is frozen in new stream contracts and native receipts; historical runs use their recorded source revision.

## Parallel native rendering

The same 126 fixed-phase training requests were sent through isolated workers. Two workers reduced measured wall time for the native renders from **19.87s serial to 13.18s parallel** (about **1.51×**). All 126 were accepted, candidate controls matched exactly, and the renderer identity and request IDs matched. Against the existing two-thread frozen reference, maximum loss, gradient, relative-gradient and parameter-update differences stayed within their declared limits; an exact same-thread cached replay reproduced the trace and weights. The full two-worker update peaked at **3.07 GiB**, below its 4 GiB cap. The recorded 26.21s full-update time came from a separate runner, so it is not presented as a controlled end-to-end speedup.

Four workers completed the isolated render-only batch in **5.97s** (about **3.33×** versus the same serial worker-call sum), but used **3.24 GiB** before the training update. A full four-worker update stopped after 16 of 126 renders and produced no optimizer update. It therefore does not qualify for the 4 GiB free-first run; new balanced plans default to two workers and can freeze one to four with `--render-workers`.

The 126-request receipts split worker time from time inside `VitalRenderer.render()`. Serial worker calls summed to **19.87s**: **13.77s** inside the renderer and **6.10s** outside it. The first worker call accounted for **3.71s** of that outside time; after startup, median outside time was **18ms per request** (90th percentile **27ms**). With two workers, paired receipt timings estimate **13.18s** elapsed across the batch: **8.00s** inside the renderer and **5.17s** outside. The first pair accounted for **4.38s** outside time; the remaining 62 pairs together used **0.80s** outside the renderer, about **13ms per pair**. This says warm dispatch/IPC is already small. `VitalRenderer.render()` itself includes validation, control/state setup, synthesis, audio checks and result construction, so these receipts do not isolate synthesis time from those steps. This batch gave each worker only 63 jobs and did not measure its 128-job recycle cost during a long fit.

[Usage](../RENDER_WORKER.md) · [Plan](data/render_worker_plan.json) · [Measurements](data/render_worker_summary.json)
