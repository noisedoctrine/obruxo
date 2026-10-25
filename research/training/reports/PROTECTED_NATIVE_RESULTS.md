# Native audio rejects the proposed protection rule

All **74 Vital renders** are accepted and independently re-scored, including two exact fixed-phase repeats. The proposed update lowers the weighted batch sound objective at **all 12 checkpoints**, by **0.7188% pooled**, but protects every sampled sound at only **7/12** checkpoints. The frozen requirement was 9/12. **Reject this rule as defined; do not start its proposed long fit or relax its gate.**

| Frozen native criterion | Result | Decision |
|---|---:|---|
| Points protecting all four sounds and making progress | 7 / 12; required 9 | Fail |
| Pooled actual weighted sound improvement | 0.7188%; required 0.5% | Pass |
| Exact fixed-phase repeats | 2 / 2 | Pass |
| Accepted and independently re-scored comparisons | 74 / 74 | Complete |

The failing points are updates **8, 12, 20, 36 and 48**. At least one sampled sound worsens by more than 1e-6 at each, despite a locally non-increasing estimate. This confirms that an estimated guard cannot guarantee actual sound preservation. The 7/12 result belongs to the stricter proposed protection rule; it does not replace or change the original model-quality gates.

The [complete scores](data/protected_native_measurements.jsonl), [native ledger](data/protected_native_renders.jsonl), [summary](data/protected_native_summary.json) and [independent verification](data/protected_native_verification.json) retain every comparison and failed criterion. No optimizer update or trained model is produced by this check. The earlier auxiliary weights remain rejected and the preceding sound-only model stays the stronger research reference.

The first launcher omitted the existing Vita dependency path and spent one request before generating audio. The [runtime-only repeat](PROTECTED_NATIVE_RUNTIME_REPEAT.md) fixes that path while preserving its evidence and original clock. It reaches 63 accepted results, then hits the original 15-minute limit; that miss stays recorded. The separately frozen [continuation](PROTECTED_NATIVE_CONTINUATION_CARD.md) re-scores and reuses all 63 results, rendering only the remaining 11. All 74 final request/outcome pairs and the immutable original ledger/measurement prefix verify exactly. **75 aggregate requests** include the initial missing-runtime failure; there are no native audio rejections or unresolved timeouts.

The final continuation uses **25.47 seconds of execution / 0.92 GiB RAM**; its outer service including prefix validation takes 64.35 seconds. Independent audit finishes **27 minutes 56 seconds from the original start**, within the separately declared 30-minute aggregate window. Peak RAM across the stages is **1.28 GiB** and aggregate evidence is **181 MiB**, below 4 GiB / 512 MiB. The original 15-minute result remains false. Source and numerical plans were frozen before each stage; three native/recovery tests pass, alongside seven protection and three diagnostic checks. Ruff and diff checks pass.

Next implement **actual native acceptance of the original weighted batch objective**, with bounded smaller trial steps and rollback when no trial improves it. This deliberately permits the within-batch tradeoffs inherent in the original objective, while keeping the original earlier/wider model-quality gates unchanged. It is a separate algorithm, not a claim that this rejected per-sound protection rule passed. See the [next fit protocol](NATIVE_ACCEPTANCE_FIT_CARD.md). Keep 14 outputs, fresh native estimates at changed states, phase control, test reservations, local execution, Actions disabled and Colab deferred.
