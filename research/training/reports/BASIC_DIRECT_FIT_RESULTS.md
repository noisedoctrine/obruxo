# Direct sound training improves reconstruction and preserves earlier sounds

Completed 2026-10-25 under the [frozen card](BASIC_DIRECT_FIT_CARD.md), plan `6cb54820c938cf145e7139b08314ec1b69116f6dfeca363a5d501b42413acd0f`. Both arms complete all **48 updates** from the same qualified initializer. All **5,659 native requests** succeed: 5,451 training probes and 208 fresh final comparisons. The [independent audit](data/basic_direct_fit_verification.json) exactly replays both models, Adam and randomness, and rescores **312 retained initial/final waveforms**. No development/test states or additional experimental updates are used.

The sound-trained model improves every tested input/playback group by more than 10% against initialization and passes the original-32 preservation gate. It beats the matched parameter-trained model in four groups but trails it in the two held-input groups, so **the full matched comparison still fails and no model is admitted**. This demonstrates basic sound learning and preservation on these training cases, not fresh-family generalization or release quality.

| Input and playback | Combined sound error vs initialization | MRSTFT vs initialization | Combined error vs parameter control | Group passes |
|---|---:|---:|---:|---|
| Original input, own performance | -10.89% | -10.77% | -45.87% | Yes |
| Original input, transferred performance | -11.82% | -10.93% | -44.47% | Yes |
| Wider held input, own performance | -17.57% | -18.65% | +3.97% | No |
| Wider held input, transferred performance | -17.66% | -17.26% | +6.06% | No |
| Wider alternate input, own performance | -19.27% | -14.64% | -17.47% | Yes |
| Wider alternate input, transferred performance | -14.62% | -10.60% | -6.52% | Yes |

Negative changes mean lower error. Original transfers are averaged within patch before the group mean. The [104 final input/playback cases](data/basic_direct_fit_cases.json) retain each arm's hard controls, request/audio identities and scores; the [summary](data/basic_direct_fit_summary.json) contains exact means and decisions.

On the 32 original recordings, combined error falls **9.53%** (`0.121607 → 0.110013`), mean MRSTFT falls **9.27%** (`0.323321 → 0.293335`) and p90 MRSTFT falls **21.48%** (`0.543779 → 0.426979`). The parameter control instead regresses on this panel. These reference recordings are now explicit rehearsal training data; their improvement is not independent confirmation. The full A2 original-32 improvement requirement is not replaced by this smaller prerequisite's preservation gate.

The final interface remains **14 values: nine continuous controls, three unison scores and two filter scores**, yielding the same 11 codec selections. Of 382,924 weights, the original 355,806 remain frozen and 27,118 correction weights train. Both arms receive the same four observations per update; the candidate uses measured sound losses and the original-reference excess penalty, without parameter labels. All 32 references receive three visits. Gradients are measured again at changed controls; every correction group receives later gradients in both arms.

Clamping is observable rather than treated as learning success: the sound arm has 48 below-zero and 36 above-one proposals out of 1,728 control observations (4.86%). The parameter arm has 109 and 16 respectively (7.23%). Legal exported controls alone do not resolve saturation or explain the remaining comparison gap.

Native execution takes **10,566.64 seconds (2 h 56 min) / 1.50 GiB RAM**, below four hours / 4 GiB. The retained archive before the summary is **583,759,832 bytes (0.544 GiB)**, below both the calculated 0.883-GiB bound and the 2-GiB cap. The [execution service](data/basic_direct_fit_service_001.json), including input checks, takes 2 h 57 min, records zero sampling errors and releases its scoped active-work request. There is one native execution, no failures or recovery attempts, and 48 charged update attempts per arm.

The [first-four-update check](data/basic_direct_fit_prefix.json) passes before continuation. Final independent verification takes **307.60 seconds / 0.86 GiB**, with its [service](data/basic_direct_fit_service_002.json) also exiting successfully. It replays all 5,451 scalar probe costs and validates their native requests and source identities. Training probe waveforms were transient, so this is distinct from independent audio rescoring of the 312 retained comparisons. Including early and final audits, 56 updates per arm are replayed in software; those are not additional experimental training or native work.

Final tensor SHA-256 values are `06dc8d184b5878c01f7a5cfd31960e6ff52c2319bacb1a7ac408dc9d248f4dd7` (sound) and `b0a18a318e8a29adf8c0c6f77eec29df9898c0c77b65b3747e3b888a06b78d09` (parameter control). Both remain diagnostic exports, explicitly unqualified for release and loaded through `restore_direct`. Implementation `aeb2de0` and numerical-plan commit `c008ba7` precede execution; 24 local tests, Ruff and diff checks pass.

Next, inspect the two remaining matched-control gaps using saved predictions and native measurements before changing the objective or scheduling another fit. Do not add epochs, weaken gates or expand outputs on this result. Full A2/A3 confirmation, a second seed, listening and a reproducible basic bundle remain unfinished. Work stays local; Actions remains disabled and Colab deferred.
