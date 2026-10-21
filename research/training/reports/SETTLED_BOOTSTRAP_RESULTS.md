# Eight-patch corrected-render fit: passed

Completed 2026-10-22. The existing **14-output basic model can fit these eight training patches and reconstruct their sounds under the corrected renderer**. Mean native MRSTFT is 98.77% lower than the default patch on reference notes and 98.76% lower on different notes. This is a training-set check, not evidence of held-out quality or a released model.

The [card](SETTLED_BOOTSTRAP_CARD.md), [frozen plan](data/settled_bootstrap_plan.json) and [results with replay evidence](data/settled_bootstrap_summary.json) retain the protocol. Plan ID: `36b192e3304431f38bebffad19fe9b682243acf178e1ba9a7aebbcfe37b0c569`; executed source: `5dd7e94a6af58d8610b015bc5d8a20c9e3a8490c`.

## What was checked

Eight distinct existing training states, original reference schedule, fixed oscillator 1–3/sampler random phase and the qualified 32,768-frame lead. Fresh seed-zero initialization, frozen frontend, 355,806 trainable parameters, 800 parameter-label updates at learning rate 0.001. No held-out source, test source, old feature cache or old model weights entered fitting. Parameter fitting is initialization only; perceptual training remains required.

| Measure | Initial | Final |
|---|---:|---:|
| Source-active normalized control MAE | 0.241232 | 0.000910 |
| Categorical accuracy | 50% | 100% |
| Label loss | 1.862587 | 0.00005068 |

All eight patches were inferred once from their reference recording, then exported with hard category choices and rendered on reference and transfer notes. Lower MRSTFT is better; it is a spectral distance, not a calibrated listener score.

| Method | Reference mean | Transfer mean |
|---|---:|---:|
| Default patch | 2.707261 | 4.544684 |
| Trained model | **0.033287** | **0.056367** |
| Training-only retrieval | 0 | 0 |
| Literal source repeat | 0 | 0 |

Retrieval includes these exact training states; its zero error is expected and provides no generalization evidence. All 16 source-repeat pairs have identical audio hashes. The eight audit reference sources also match the original training audio hashes.

## Recovery, resources and limits

**88/88 native attempts accepted**: eight training inputs and 80 audit requests, no interruption or retry. All charged request identities match their outcomes. Runtime was **478.42 seconds (7.97 minutes)**, peak process-tree RAM **1.34 GiB**, archive before the final summary **13.84 MiB**. All remain within 30 minutes / 4 GiB / 256 MiB. Checkpoints were written after each source and every 50 updates. No waveform files were archived.

The separate zero-render audit reconstructs all 800 updates from the original seeded weights and retained features: final model, Adam state, batch RNG, global Torch RNG and metrics match exactly. It also reconstructs every native audit request from the actual saved hard exports. The successful audit took 19.11 seconds and 0.62 GiB RAM, adding no renders or training updates.

The auditor initially assumed that cached-input predictions and fresh-audio predictions would produce identical preset bytes. Their hard categories match; the largest normalized control difference is **2.3842e-7**, below the explicitly reported 1e-6 numerical audit tolerance. The executed evaluator used its historical four-thread setting while training used two. The initial failed byte assertion is retained privately. Future corrected-render evaluation now preserves the training CPU thread setting and records it in its plan. Actual native request reconstruction uses the saved preset bytes exactly. Candidate waveforms were discarded, so saved spectral scores were not independently rescored.

Profile-aware checkpoint recovery matches uninterrupted training in local tests. Evaluation recovery also now hashes single-model prediction files and rejects modified exports before reusing a completed case. These checks do not substitute for the later larger-model recovery and fresh-environment release checks.

## Next

Freeze a full corrected-render initialization refresh on the same 404 training states, preserving the compact vocabulary. Require this small result and its replay evidence before execution. Then requalify category scaling for the new initial weights and return to matched sound-loss training with the unchanged held-out gates. A2, listening and final confirmation remain incomplete; this model is explicitly unqualified for release.
