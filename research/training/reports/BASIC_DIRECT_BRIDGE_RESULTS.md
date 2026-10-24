# Direct basic-output and reference-rehearsal bridge passes

Completed 2026-10-24 under the [frozen card](BASIC_DIRECT_BRIDGE_CARD.md), plan `954f36e517ae405500c2cc3f80b16f2c4fcd0e668f1065c664623fbd3d8c31cb`. This qualifies the new output connection and rehearsal data, not useful reconstruction or model admission.

The [bound summary](data/basic_direct_bridge_summary.json), [input identities](data/basic_direct_bridge_identity.jsonl) and [reference manifest in the plan](data/basic_direct_bridge_plan.json) establish:

- All **1,616 scheduling entries / 1,212 distinct recordings / 404 training patches** retain exact initial continuous values, hard choices and preset bytes after cold restoration. Both categorical probability vectors match their prior calibrated values at temperatures 1.0 and 0.5. Raw categorical score units intentionally change.
- Every responding component at the **24 qualified native points** reaches its direct output: **188 continuous and 48 categorical connections**. The old predictor has no gradients; weights and Torch RNG remain unchanged by these checks.
- All **32 original reference recordings** match their source recipes, feature/audio hashes, retained ground-truth waveforms, accepted-initial predictions and native receipts. The baseline comes from the accepted initializer, never the rejected unison model. The round-robin schedule visits each reference three times in a 48-update sketch.
- Software probes are recomputed after continuous values change. Two simulated steps plus one replay reconstruct model, Adam and Python/NumPy/Torch state exactly. The later step reaches hidden, continuous, unison and filter correction groups. Cold simulated outputs remain bounded and produce identical presets across the full cache.

The correction adds **27,118 trainable weights** to the frozen 355,806-weight predictor: **382,924 total**. That is only 363 more weights than the unison-only correction. The final interface stays at **14 values: nine continuous, three unison scores and two filter scores**, giving the same 11 codec selections. No oscillator, effect, LFO, curve, route or seed coverage is added.

The nine corrections operate directly on normalized control values and clamp them to their legal interval. Category corrections operate on the calibrated scores. Tests explicitly check the real clamp derivative, including zero derivative outside its interval; this bridge does not establish that later training avoids saturation. Fixed phases and the qualified 32,768-frame lead survive export.

Qualification uses **86.05 seconds / 0.73 GiB** process-tree RAM, within five minutes / 4 GiB and 64 MiB evidence. [Preparation](data/basic_direct_bridge_preparation.json) takes 0.46 seconds / 0.47 GiB. The [execution service](data/basic_direct_bridge_service.json) exits successfully, records zero RSS sampling errors and releases its scoped active-work request. Fourteen local checks, Ruff and diff checks pass. Implementation `d5425a3` and the numerical plan were committed before execution.

There are **zero native requests and zero experimental training updates**. Software qualification counts **three simulated optimizer applications and 224 software cost evaluations** explicitly. Its lower synthetic loss is not an audible gain. The rehearsal excess penalty is tested on both sides of the accepted baseline; its effectiveness still requires actual training and preserved real sounds.

The untrained `model.pt` has file SHA-256 `0e1b88f7916a31cf4afd4af20aacda14a69086ea6c61ecc303ba249391dca735` and tensor SHA-256 `adc4a6fb7ebc783c8782bc39692129b141a75bdb5f63172bc78fa81bc9574702`. Use the explicit `restore_direct` wrapper. Simulator files are separate and are not fitting initializers; release CLI/bundle support is not claimed.

Next, implement and freeze the [small native direct-output fit](BASIC_DIRECT_FIT_CARD.md). Its gradients must be measured afresh once continuous controls change; the earlier fixed-control category table is no longer valid. Keep all full A2/A3 quality and confirmation requirements.
