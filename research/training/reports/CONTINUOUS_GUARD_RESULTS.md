# Corrected protection passes its local screen

The corrected rule accepts **45/48** nonzero proposals and retains **32.89%** of the original positive estimated gain. Both frozen gates pass: at least 36 proposals and at least 25% estimated gain. Every selected proposal keeps all four sampled sound estimates non-increasing within 1e-8 and all existing panel control changes within 0.025. Three rejected proposals remain unchanged rather than being counted as successful learning.

The check takes **26.55 seconds / 0.73 GiB RAM**, with zero new renders or experimental training. All original trace fields, model/Adam/RNG states and 5,451 scalar probes replay exactly. Seven focused tests pass, including proof that protection excludes category-score derivatives while the original optimizer still receives them. The [plan](data/continuous_guard_plan.json), [declaration](data/continuous_guard_declaration.json) and [complete measurements](data/continuous_guard_summary.json) are frozen before execution. Implementation: `6781aae`; numerical plan: `88d9cf4`.

| Screen | First rule | Corrected protection |
|---|---:|---:|
| Accepted nonzero proposals | 24 / 48 | 45 / 48 |
| Original full-gradient gain denominator | 0.3701789855 | 0.3701789855 |
| Estimated protected output gain | 0.1244484082 | 0.1217701236 |
| Retained estimated gain | 33.62% | 32.89% |
| Frozen usefulness gates | Fail | Pass |

Only the protection derivative changes. Adam's full native objective, category learning, step cap, backtracking, data, model and usefulness thresholds stay fixed. A fixed hard selection cannot improve its rendered sound merely by moving its category scores; protection now follows the continuous output within that region. Actual selection changes still face the conditional native-cost guard. The [failed first rule](PROTECTED_UPDATE_RESULTS.md) remains recorded.

Some accepted proposals are small: 11 use full scale, 12 use half scale, and 22 use scales from 1/16 down to 1/256. Aggregate estimated progress prevents declaring the rule useful solely from tiny accepted moves. Estimates use measured native local gradients and conditional costs; they are not after-update audio. The mixed weight-linear/output-based ratio is a screening signal, not a perceptual improvement claim.

**Next: native after-update acceptance checks before another long fit.** Use fixed historical states and fresh Vital renders of the actual proposed patches, preserving phase control and qualified preroll. Do not advance a new trajectory using historical gradients. A passing screen is not proof of native preservation, a trained model, full A2/A3 acceptance or release readiness. All 14 outputs and original sound-quality gates remain unchanged; local only, Actions disabled, Colab deferred.
