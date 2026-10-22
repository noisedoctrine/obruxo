# Wider notes improve, but earlier reconstruction is lost

Completed 2026-10-22. The 14-output model learns the new recordings: native sound error falls **46.7% on their reference performances and 45.1% on transfer**. However, error on the original recordings rises to **3.08–3.10 times** the starting model. The frozen preservation gate fails. **Do not admit these weights or advance category qualification.**

The [card](VARIED_BOOTSTRAP_CARD.md), [plan](data/varied_bootstrap_plan.json) and [audited results](data/varied_bootstrap_summary.json) retain the evidence. Plan ID: `70983e15790c468c817a8fe5813a743ed9d1cd1183bd8124195df0d96a04123f`. Executed training source: `559ca4dd07316e07ed5ca5f60e81509df6505d8f`. Both panels use the same eight training patches, the same initial/final weights, fixed phases and the qualified render profile. These are training-state spectral comparisons, not fresh held-out perceptual evidence.

| Input panel | Playback | Starting model | New model | New / starting |
|---|---|---:|---:|---:|
| Original recordings | Reference | 0.3062 | 0.9444 | **3.085 — fail** |
| Original recordings | Transfer | 0.3382 | 1.0475 | **3.098 — fail** |
| Wider-note recordings | Reference | 1.3885 | 0.7400 | **0.533 — pass** |
| Wider-note recordings | Transfer | 1.2617 | 0.6927 | **0.549 — pass** |

The wider-input gate requires ratios below 0.9; the original-input gate requires at most 1.1. All four candidate/template comparisons pass, but they cannot override the original-input regression. No evaluator-selected artifact becomes an admitted model. The previous initializer remains the retained reference.

Training starts from the verified previous weights, with a fresh optimizer, and fits all 808 new observations for 8,000 updates. Source-active settings error falls from 31.00% to 9.79%; category accuracy rises from 65.84% to 99.94%. These diagnostics do not rescue the failed acoustic gate. The new training replay contains neither original-bank recordings nor their cached features, so it does not rehearse the earlier input distribution. Joint replay of old and new observations is the next controlled change; this run does not isolate every cause of the regression.

All **1,000 native requests** are accepted: 808 source recordings and 192 comparisons. All **32 source repeat pairs** match exactly. The independent audit replays all 8,000 updates with exact model weights, Adam state, batch RNG, global Torch RNG and metrics. It checks all 808 source receipts and reconstructs all 192 comparison requests and 32 hard patch exports. Cached/fresh predictions differ by at most `1.11e-16` in normalized controls, within the predeclared `1e-6` tolerance; categories match exactly. The audit adds no renders. Discarded waveforms are not independently rescored.

Cost: **43.17 minutes**, **1.22 GiB** peak process-tree RAM, **86.28 MiB** evidence before the final summary, and zero sampling errors, interruptions or retries. Saved snapshots serialize 14.19 GB across 809 collection and 33 full checkpoint writes, with 5.86 minutes spent checkpointing. The separate audit takes **4.51 minutes**, including its bounded wait for native completion, and peaks at **0.87 GiB**. All costs remain within their frozen limits. Fifteen distinct focused local tests and Ruff pass; the small paired-fit replay was also checked before execution. GitHub Actions remains disabled.

The next fit should reuse the **404 original and 808 new saved feature observations together**, with a fresh optimizer and the verified earlier weights. This preserves all available training observations and avoids repeating their native collection. Freeze the combined-data contract, costs and unchanged acoustic gates before execution. The head remains **14 outputs / 355,806 parameters**. Sound training still needs both input views and the counterbalanced development panel alongside the original cases. A2, fresh confirmation, listening and release remain unfinished.
