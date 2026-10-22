# Corrected-render 404-state initialization: passed

Completed 2026-10-22. The compact basic model has a fresh initialization trained on corrected audio from **all 404 training states**. Settings fit, the fixed eight-state native audit and exact replay all pass. This is parameter-based initialization; held-out sound-loss quality remains unproven and the model is not qualified for release.

The [card](SETTLED_FULL_BOOTSTRAP_CARD.md), [plan](data/settled_full_bootstrap_plan.json) and [results with replay evidence](data/settled_full_bootstrap_summary.json) retain the protocol. Plan ID: `4738aea656807b39269ad75a3d3eddfe6c963daf39f85b83d619119e79f46a56`; executed source: `d88f050ba56a0e1a32baaa926bcd9f727a41ecb0`.

## Fit and sound check

Fresh seed-zero weights, the same 14 outputs and 355,806 trainable parameters, frozen frontend, 16,000 updates, batch size 32 and learning rate 0.001. Every source was rendered with the qualified 32,768-frame lead and fixed oscillator 1–3/sampler random phase. No old features or model weights entered fitting; no held-out or test source was used.

| Whole training-set measure | Initial | Final |
|---|---:|---:|
| Source-active normalized control MAE | 0.247510 | **0.044567** |
| Category accuracy | 41.71% | **100%** |
| Label loss | 1.876596 | **0.004284** |

The predeclared native audit covers the first eight training states, not all 404. Each patch is inferred once from reference audio, then exported with hard category choices and played on both performances. Transfer notes come from the existing development bank; this is not fresh confirmation.

| Method | Reference mean MRSTFT | Transfer mean MRSTFT |
|---|---:|---:|
| Default patch | 2.707261 | 4.544684 |
| Refreshed model | **0.306164** | **0.338171** |
| Training-only retrieval | 0 | 0 |
| Literal source repeat | 0 | 0 |

Model error is 88.69% / 92.56% lower than the default patch on these training sounds. Retrieval includes the exact training sources, so zero retrieval error is expected. All 16 native source-repeat pairs match exactly, and the eight audit reference sources match their training audio hashes. MRSTFT is a spectral distance, not a calibrated listener score.

## Cost and reproducibility

**484/484 charged native requests accepted**: 404 source inputs and 80 audit requests, with no interruption, retry or replacement source. Every request has its matching outcome. Runtime: **3,157.39 seconds (52.62 minutes)**; peak process-tree RAM: **1.30 GiB**; archive before the final summary: **45.95 MiB**. All fit within the frozen two-hour / 4-GiB / 512-MiB limits. The retained feature replay is 15,618,640 bytes. No waveform archive was written.

The source inputs contain 1,616 seconds of output audio. Renderer calls took 635.39 seconds in total; complete worker requests took 2,093.35 seconds. Fresh-worker setup and other work outside the renderer account for much of the elapsed time. The checkpoint writer serialized 4.58 GB over the run, while retaining only bounded current/previous files; its measured checkpoint time was 160.69 seconds.

The independent audit reconstructs all **16,000 updates** from the original weights and retained features. Final model, Adam state, batch RNG, global Torch RNG and metrics match exactly. All 404 source receipts, eight saved hard exports and 80 native audit requests are verified. Cached-input versus fresh-audio predictions differ by at most **1.11e-16** in normalized controls after codec conversion; hard categories match exactly.

Replay ran alongside the native audit, added zero renders and completed in **457.45 seconds**, including its wait for final native evidence; training replay itself took 428.29 seconds. Peak audit RAM was 0.71 GiB. The passive watcher was replaced by this parallel verifier; the original fit/render process remained uninterrupted. Candidate waveforms were discarded, so saved spectral scores were not independently rescored.

## Next

Initialization uses the original contract recordings, whose notes lie near middle C. The later sound-loss training recordings cover a much wider pitch range. Check the same new model on both sets of already-rendered training inputs before another long sound fit. This can expose an input-distribution mismatch without adding audio or claiming a perceptual result. The fixed category-feedback check is prepared, but its constants must be qualified for the eventual training inputs and new weights.

Keep the 14-output scope and unchanged held-out sound-quality gates. A2, fresh confirmation, listening and release packaging remain incomplete.
