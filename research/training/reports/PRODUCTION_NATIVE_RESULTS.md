# Production-choice sound directions pass real Vital checks

Completed and independently verified 2026-10-23. The fixed local change improves measured sound error on **12/12 original-input cases and 11/12 wider-input cases**, for both continuous and joint steps. All predeclared direction/connection gates pass. This qualifies the production-choice path for a costed paired training comparison. **No optimizer update occurs and no trained model is admitted.** The predictor remains at 14 dense outputs / 355,806 trainable parameters.

The [card](PRODUCTION_NATIVE_CARD.md), [frozen plan](data/production_native_plan.json) and [complete audited evidence](data/production_native_summary.json) preserve plan `b1aabfef73a4e32e108f54eedec98c63225659b5ec85302d512a352958356d94`, numerical source `d96d272`. These are 24 reciprocal observations of **12 training patches**, not 24 independent patches or a held-out confirmation. The accepted initializer and all input recordings remain unchanged.

| Predictor input | Starting MRSTFT | After continuous step | After joint step | Improved cases, each method |
|---|---:|---:|---:|---:|
| Original recording | 0.4622 | 0.4121 | 0.4121 | 12/12 |
| Wider recording | 1.2091 | 1.1885 | 1.1325 | 11/12 |

Lower is better. Every prediction is scored under the opposite recording's performance. Both input groups also have strictly lower mean spectral/envelope/side loss for each method. Four joint steps change a category. Source category recovery is not an acceptance criterion; actual sound scores decide the gate. No backtracking, parameter-label loss, checkpoint selection or failed-case omission is used.

Sound loss reaches the actual inverse network. **188/188 responding continuous-control checks and 48/48 responding category checks** have nonzero gradients in their corresponding prediction heads. Complete model-gradient digests, native loss tables and fixed output-space steps reproduce exactly in the fresh audit. All 24 source repeat pairs are exact and match the previously audited recordings. Model weights and Torch RNG remain unchanged throughout.

All **568 native requests** are accepted: 48 source/repeat renders, 472 gradient probes and 48 changed-sound renders, below the frozen 624-attempt cap. The fresh auditor reconstructs every request and gradient, hashes all 24 retained source waveforms, and independently recalculates all **72 starting/changed sound scores** from saved audio. Intermediate gradient-probe waveforms are not retained or independently rescored; their recorded scalar losses drive the exact gradient replay. The original source, starting and changed sounds are kept for inspection.

Native execution takes **17.34 minutes**, peaking at **1.05 GiB** process-tree RAM with zero sampling errors. Its conservative archive bound is 298.05 MiB against a 512-MiB cap; native evidence is about 133.1 MiB before the final summary. Input preparation/binding separately takes **165.00 seconds**, execution-time revalidation **307.45 seconds**, and fresh verification **177.58 seconds / 0.80 GiB**. The durable Windows service finishes with no error. Including preparation, complete service execution/setup and verification, total measured local time is **28.32 minutes**. All declared limits pass; no time or attempt counter is reset.

Twenty focused local tests pass for balanced reciprocal selection, the real-model gradient path, per-control saturation detection, waveform rescoring, corrupted native evidence, resource checks and overwrite refusal. Ruff and diff checks pass. Both generation and verification use local CPU execution; GitHub Actions stays disabled.

Next finish the paired comparison runner, with one original and one wider observation in every update, raw logits for the matched parameter control, and native sound loss for the candidate. The separately prepared paired kernel/checkpoint loop has 11 focused local checks for individual measurements, no partial optimizer advance on render failure, and exact model/Adam recovery. It has not run a native training comparison. Keep the same accepted initial weights and unchanged reconstruction gates, freeze the new allocation before execution, and retain candidate audio for rescoring. A2, fresh confirmation, listening and release remain open. Colab is deferred.
