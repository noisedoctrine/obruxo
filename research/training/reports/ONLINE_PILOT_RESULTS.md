# Online pilot results — 2026-10-05

**Superseded for native patch fidelity, 2026-10-05.** The first trainer discovered that native Vital silently rejected the authored `99999.9.9` version marker and retained its default patch. The QA/timing observations below describe that incorrect render path. The reconstruction comparisons therefore do not qualify the codec. See [FIRST_TRAINER_RESULTS.md](FIRST_TRAINER_RESULTS.md) for the corrected versioned renderer, native control readback, distinct patch audio and trained-model evidence.

The authored pilot, limited codec and bounded online render stream are working against the reviewed local Vital plugin. **100 of 100 benchmark renders passed audio QA**, and all six codec reconstruction cases passed the existing audio tolerances. Audio remained in memory throughout.

These are data-pipeline measurements, with **zero optimizer steps**. They do not establish inverse-model quality, full-corpus readiness or GPU training throughput. Reproduction commands and API semantics are in [ONLINE_PILOT.md](../ONLINE_PILOT.md); the [aggregate JSON](data/online_pilot_summary.json) contains exact identities and measurements.

## Pilot population

| Measure | Result |
|---|---:|
| Independently authored, statically eligible states | 256 |
| Training / validation / test states | 206 / 22 / 28 |
| Continuous targets / categorical targets | 9 / 2 |
| Stored pool size | 196,064 bytes |
| Retained audio files | 0 |

The pool varies one init oscillator, optional filter 1 and envelope 1. Categorical values cover one, two or four unison voices and filter off/on. Inactive controls have fixed values and supervision masks; unsupported state is rejected. The pool uses the reviewed init asset content and legacy schema. It does not change the [downloaded corpus's eligibility result](CONTRACT_READINESS.md).

All native work reported here used **training states**. Validation/test states and reserved transfer performances remain available for the next model evaluation. The 100-job benchmark covered 84 distinct training states with reference performances. Static eligibility of all 256 states does not imply that every state/performance combination has passed native QA.

## Local render/consumer measurement

One worker rendered fresh four-second reference clips into a CPU log-spectrum consumer on Windows, Python 3.12.13, NumPy 2.4.6, DawDreamer 0.8.3 and the reviewed Vital 1.6.4 binary. No GPU or paid compute was used. A fresh process was started for each render; its startup and cleanup are included in job latency.

| Measure | Observed |
|---|---:|
| Requested / accepted / rejected / errors | 100 / 100 / 0 / 0 |
| Elapsed time | 211.745 s |
| Accepted fresh clips per second | 0.4723 |
| Median / P95 job latency | 2.072 s / 2.626 s |
| CPU feature computation, total | 2.734 s |
| Consumer time waiting for renders | 204.865 s; 96.75% of elapsed time |
| Configured prefetch budget | 4 clips and 32 MiB |
| Peak reserved prefetch | 1 clip; 1,411,200 bytes (1.35 MiB) |
| Sampled peak process-tree RSS | 402,735,104 bytes (384.1 MiB) |
| RSS sampling errors | 0 |
| Audio files written / optimizer steps | 0 / 0 |

The byte and clip limits apply before native dispatch and include reserved in-flight output. RSS includes interpreter/native working memory and samples the summed resident sets of the parent and child processes every 50 ms; shared pages may be counted more than once. Queue PCM limits are not a promise about total application memory.

Rendering dominated this lightweight consumer. The result supports keeping audio generation bounded and measuring reuse or qualified worker reuse later. It does not justify a GPU purchase or predict performance with the future frozen frontends and trainable model. Profile that combined workload before selecting local XPU versus free Colab. This run covers a small synth vocabulary and one machine, not the full project's compute requirements.

## Codec reconstruction check

The check selected one training state for each of the **six** unison/filter combinations. Each state produced a source render, a repeated source render and an encode/decode reconstruction render, using the same reference performance.

- **18/18** native renders passed audio QA.
- **6/6** reconstructed targets matched the source controls at float32 precision.
- **6/6** reconstructed renders passed the existing RMS/peak/log-spectrum repeatability tolerances.
- **6/6** source-repeat pairs passed those same tolerances.

The unchanged reference tolerances are 0.01 relative RMS difference, 0.10 absolute peak difference and 0.35 log-spectral RMSE. This is a bounded reconstruction smoke across pilot categories, not a calibrated quality metric for the final inverse model. Individual measurements and source-repeat differences are retained in the aggregate JSON.

## Recovery and failure behavior

The native benchmark's saved cursor resumed at sequence **100**, successfully handled **100–103**, and persisted next sequence **104**. Recipe and render-request IDs were independently regenerated from the frozen pool and matched the resumed events.

Automated process tests additionally cover early closure with delivered-but-unacknowledged work, replay of the same pending recipe, worker crashes, bounded retries, deadlines, cancellation and worker reaping. They use a controlled fake renderer to exercise failure modes. The native resume check continued a completed bounded segment; it was not a forced crash of an optimizer run. Exact audio replay and full model/optimizer checkpoint recovery are not claimed.

The event log distinguishes render outcomes, accepted buffers consumed by the feature stage, and handled rejections/errors. No failure is silently replaced or removed from the denominator. A resume cursor advances only after the consumer acknowledges handling an outcome; pending prefetch can be regenerated.

## Validation and remaining work

The training suite has **65 passing tests** and passes Ruff. Coverage includes the original contracts, pilot reconstruction/scope rejection, deterministic pair coverage, pool integrity, byte backpressure, QA failures, process failures/timeouts, cancellation, acknowledgement order, cursor compatibility and report accounting. The prior corpus snapshot still passes verification with its original contract identity.

Next is the small audio-to-patch baseline: model/optimizer, auxiliary continuous/categorical losses with codec masks, frozen performance features, real-render held-out evaluation and a complete checkpoint/recovery test. The existing stream cursor must be saved consistently with those training states. Wider assets, modern schema support, LFO/modulation/effects coverage and Linux/Colab renderer qualification remain open.

| Frozen identity | SHA-256 |
|---|---|
| Pilot pool | `218093714bdb46b421120e9c3be75b32ead11e7bb46e4ef62e096697af0cc74c` |
| Pilot codec | `b44a06357b9a2202f8aaaae6b0f43b1749e5d42b2acb31af54095517e8fdf6c5` |
| Pilot split manifest | `ceda5da4dca951e779b0c753e67977556e0d2af6b076df8e71a1695da8bd5fd0` |
| Training contract | `75e64f048f3b2868fcb2bb704e2440650f9e4561f76bddbae44f2bb38f088983` |
