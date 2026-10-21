# Sound-loss updates pass on the settled renderer

Completed 2026-10-21 under the [frozen card](SETTLED_NATIVE_GRADIENT_CARD.md). The corrected renderer passes the original numerical-gradient gates: continuous updates improve native MRSTFT in **22/24** cases and joint continuous/category updates in **21/24**. Each reference/transfer group exceeds the required 60% improvement rate and has lower mean loss under both metrics. This qualifies the method for fresh basic inverse training, not a trained-model quality claim.

[Audited evidence](data/settled_native_gradient_summary.json) · Frozen source commit `aad1e91` · Plan `a23dee76b9dff187770e50680a601c530877d5915b8beae5db383dc6808a9b30`

| Performance | Update | Improved MRSTFT | Mean MRSTFT change | Mean primary-loss change |
|---|---|---:|---:|---:|
| Reference | Continuous | 12/12 | −0.19979 | −0.08633 |
| Reference | Continuous + categories | 11/12 | −0.55650 | −0.17539 |
| Transfer | Continuous | 10/12 | −0.14737 | −0.06707 |
| Transfer | Continuous + categories | 10/12 | −0.53928 | −0.16300 |

All **560 native requests** pass unchanged strict QA; all **24 fixed-phase source repeats** are sample-exact. Every request uses the explicit 32,768-frame lead, preserving the requested attack and tail. The cases, starts, estimator and quality thresholds match the historical cold-renderer qualification exactly; the source and candidate audio were generated anew. No cold target cache, clipping penalty, learned helper, source-parameter loss or inverse-model update is used.

The non-improvements remain in the results. Reference case six worsens MRSTFT after its joint update; transfer cases seven and nine worsen after continuous and joint updates. Some of these joint updates still lower the primary spectral objective, showing disagreement between the two metrics. Transfer case nine worsens under both objectives after the joint step. A useful local estimate does not guarantee every simultaneous control change will help. This remains development evidence from the existing 12 source states, not fresh confirmation or test evidence.

The run takes **1,070.05 seconds (17.83 minutes)**, peaks at **930,021,376 bytes (0.87 GiB)** process-tree RAM and retains **3,322,617 bytes (3.17 MiB)** of receipts/measurements. It passes the declared 624-attempt, 45-minute, 8-GiB and 32-MiB limits, with no retries or RSS sampling errors. The publication audit checks frozen source, reconstructs every native request and numerical update from the saved scalar losses, and recomputes the complete gates/resource accounting. Audio was discarded; the audit verifies receipts and recorded losses rather than independently rescoring waveforms.

Proceed with a separately frozen fresh trial using the [matched inverse protocol](NATIVE_INVERSE_FIT_CARD.md): same historical starting model, fresh optimizers and newly rendered settled targets. Retain the 14-output scope. The old 34-update prefix remains historical and cannot seed corrected training. Fresh confirmation, second-seed replication, actual listening and bundle qualification remain later requirements. GitHub Actions stays disabled and Colab remains deferred.
