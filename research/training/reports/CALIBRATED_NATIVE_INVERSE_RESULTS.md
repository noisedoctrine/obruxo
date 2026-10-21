# Category scaling restores learning signals, but basic quality still falls short

Completed 2026-10-21 under the [frozen comparison card](CALIBRATED_NATIVE_INVERSE_CARD.md). Both models complete **128 updates**, and both training-fit gates pass. Held-out native spectral error improves **7.23% on reference MIDI** and **4.24% on transfer MIDI** against the starting model. Both miss the unchanged 10% target. All native requests, resource checks and independent replay pass. **Stage A2 remains incomplete; no model is promoted.**

[Verified evidence](data/calibrated_native_inverse_fit_summary.json) · Frozen training source `6ec6c2a` · Plan `b93183419006684feaa0dd593842a1b27511528bda2f9c6c21a4dd18dddb891b`

## What changed and what learned

Only the checked category-score scaling changes the training design: fixed divisors 20.31771924164867 for unison and 20.337628037582693 for filter, applied before both matched losses. Both arms start from the historical initial weights with fresh Adam states. Data, MIDI, update order, optimizer, renderer, losses, quality gates and resource caps stay fixed. All 64 training targets render fresh on demand. The head remains **14 dense outputs** and the inverse model has **355,806 trainable parameters**. Native scoring/export uses raw-score argmax.

| Training performance | Starting primary loss | Parameter arm | Sound arm | Sound gain vs starting model |
|---|---:|---:|---:|---:|
| Reference | 0.56675 | 0.54375 | **0.39204** | 30.83% |
| Additional training MIDI | 0.62013 | 0.61845 | **0.41882** | 32.46% |

All 32 states appear in each group. Both fit gates pass: primary loss falls at least 10%, is no worse than the parameter arm, and legacy MRSTFT improves. This establishes learning on the fitted states, not reliable generalization.

## Held-out development result

All 12 declared states are scored under both performances. Each method predicts once from reference audio and reuses that patch for transfer. Lower native MRSTFT is better.

| Method | Reference | Transfer |
|---|---:|---:|
| Template | 2.69093 | 2.70709 |
| Training-only retrieval | 1.06871 | 1.17729 |
| Historical starting model | 0.98473 | 1.00708 |
| Matched parameter arm | 1.04517 | 1.06486 |
| Sound-trained arm | **0.91358** | **0.96435** |

Gains against the starting model are **7.23% / 4.24%**, below 10% in both groups. Gains against the parameter arm are **12.59% / 9.44%**, so transfer also misses that comparator's threshold. Template/retrieval comparisons, primary-loss comparisons against the parameter arm, and tail-error limits pass. Membership, denominators and thresholds are unchanged.

Eight of 12 reference cases and six of 12 transfer cases improve against the starting model. Whole-state paired 95% MRSTFT-delta intervals are **−0.21272 to +0.06865** and **−0.17215 to +0.08799**; both include no benefit. These are descriptive development intervals, not independent confirmation. Final-test states remain untouched.

## Category gradients are active

Median raw-score gradient L1 is **8.86e-4 for unison** and **1.85e-3 for filter**, with no exact-zero gradient on any of the 128 updates. Every update has a measurable native response for both heads. The previous unscaled medians were 8.53e-7 and 1.11e-16. This verifies that the scaling repair reaches the actual training path; it does not establish the required sound improvement.

The final sound model changes one filter choice and one unison choice across the 32 training states. The parameter control changes four and seven. Exact source-category recovery is not a perceptual requirement. With both category gradients now active, do not assume that further score tuning or more category outputs is the next useful change.

## Cost and verification

- **3,004 native requests, all accepted:** 2,580 training and 424 audit requests. All **88 source-repeat checks** are sample-exact. No interruption or retry occurs.
- **128 charged matched optimizer attempts**, each committed for both arms; no extra fitting updates or checkpoint selection.
- **6,321.34 seconds (105.36 minutes)** elapsed, **1,643,577,344 bytes (1.53 GiB)** peak process-tree RAM and **744,023,522 bytes (0.69 GiB)** retained evidence. All original caps pass.
- Independent replay reproduces every actual update, both final model weights, both Adam states and Gumbel RNG exactly. Calibration identity matches the frozen plan, checkpoint and diagnostic exports. Retained source-repeat audio independently reconstructs the cached input features and spectral targets. Hard audit patches, requests, receipts and final gates are reconstructed.

Candidate audit audio is discarded as declared, so its scores are verified against evaluator receipts rather than independently rescored from retained candidate waveforms. Thirteen focused tests, Ruff and whitespace checks pass. Diagnostic exports remain `qualified_for_release=false`. GitHub Actions stays disabled and Colab remains deferred.

Next, use retained evidence to examine training coverage and the remaining per-control sound errors before selecting another comparison. The large training improvement and much smaller held-out improvement show the remaining generalization gap; they do not prove a single cause. Do not extend this run, launch a second seed of an unqualified result, lower thresholds or add prediction heads. Fresh confirmation, listening and bundle qualification remain required after a promising basic comparison.
