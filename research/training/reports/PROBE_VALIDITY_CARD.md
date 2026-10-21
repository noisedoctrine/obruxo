# Invalid-trial handling for the stopped native training run

Declared 2026-10-21 before new rendering. The original [run stopped during update 35](NATIVE_INVERSE_FIT_RESULTS.md), after 34 verified matched updates, on a clipping-only categorical probe. Keep that failed run unchanged. Four unison voices remain supported. A proposed setting can fail audio validity even though its preset is legal; the optimizer needs a feasibility cost for that trial.

Preflight plan ID: `86c15c998a9d9962704a41e5a85882f0f0f8305225892cd479ee1a6af96e7c3d`. Three focused policy tests pass: exact unchanged valid loss, monotonic clipping cost, and fatal handling of other failures or changed native identity/state.

Valid training probes retain the exact existing A0 spectral objective. A probe rejected solely for clipping receives **1000 + 100 × (peak − 1)² + clipped_samples / (frames × channels)**. This is a declared feasibility cost, not a measured spectral loss. Preserve the rejected receipt and mark that measurement explicitly. Require finite samples, correct frame/channel geometry, verified native state/readback and unchanged renderer identity. Other failures remain fatal. Source targets and all final output audits keep strict existing QA.

First perform a **six-request native check**, with a hard cap of 12 requests, ten minutes, 8 GiB RAM and 32 MiB output. Freeze its exact patches before rendering:

- Render the original source twice; require exact equality and its historical audio hash.
- Render the exact rejected patch twice; require the same rejection and audio hash as the stopped run.
- Lower only its normalized oscillator level by 0.1 and render twice. Require accepted QA and sample-exact repeats.
- The safe patch's real spectral loss must be below the invalid-trial cost, with a positive level secant pointing descent toward the lower-level valid patch. No inverse updates or model-quality claim follow from this check.

A continuation is allowed only after this check passes. It must reuse the audited 34-update matched checkpoint, optimizer/RNG states, original data/order and cached targets. Preserve every previous charge, including the uncommitted update and rejected render. Finish the original total of 128 updates per arm; do not add another 128 updates. The original four-hour deadline starts at the first run, and the combined archive remains within 2 GiB. The combined native cap remains **3,328**.

Reallocate phase caps within that total to **2,880 training** and **448 audit** attempts. The 743 spent training requests plus at most 2,068 remaining gradient queries fit the training cap. The original 424 planned audit requests plus the at-most-12-request preflight fit the audit cap. The continuation records parent/preflight identities and maintains the complete combined ledgers. Its feasibility-cost policy is an explicit new contract; the original run is not relabelled successful.

The original fit and held-out comparison thresholds remain unchanged. All final outputs must pass strict QA. Report all penalized probes separately from accepted audio, fix exception progress reporting, replay the combined update sequence exactly, and retain the unchanged incumbent. No extra prediction heads, source-label anchor, test states, Colab or GitHub Actions changes.
