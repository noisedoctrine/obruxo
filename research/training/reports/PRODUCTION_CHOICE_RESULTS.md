# Production patches keep usable category gradients

Completed and independently reconstructed 2026-10-23. The new adapter renders exactly the patch that production would export, while keeping a nonzero backward derivative for both category heads. **The derivative/export contract passes. Native-direction and trained-model quality remain unproven.** No weights are changed or admitted.

The [card](PRODUCTION_CHOICE_CARD.md), [frozen plan](data/production_choice_plan.json) and [complete evidence](data/production_choice_summary.json) retain the result. Final plan: `bb8d2e5af90a9dfd140e1209f803f3385ee74387c2f47da4537347c7398c9476`; source `46dc48f`. The accepted initializer remains tensor hash `35cd356e97b5574df5d1a3cfaf19e79bd5b86f53df5e9ee6f56573a3da282c7b`, with **14 dense outputs / 355,806 trainable parameters**.

All 128 weighted training observations (96 distinct recordings from 32 states) retain their production patch bytes. All 256 training forward tensors match deterministic production exactly at backward temperatures 1.0 and 0.5, including canonical inactive values. No Gumbel noise is drawn. Model/frontend weights and Torch RNG remain unchanged. Fixed scales are 32.73040780087214 for unison and 29.015704035654633 for filter enable; they affect only the softmax backward derivative. Historical fit commands and their scaled parameter-control losses are unchanged. A future matched runner must give its parameter control the raw logits.

Every full category derivative matrix matches the independent formula: **512/512 matrices**. All norms are nonzero and above the existing `1e-8` connectivity floor; every median exceeds `1e-5` and passes the unchanged recovery requirement. These are derivative measurements, not perceptual scores.

| Head | Backward temperature | Unscaled median norm | New median norm | Connected fraction |
|---|---:|---:|---:|---:|
| Unison voices | 1.0 | 2.87e-7 | 0.013741 | 100% |
| Unison voices | 0.5 | 7.91e-14 | 0.024114 | 100% |
| Filter enable | 1.0 | 3.45e-20 | 0.009864 | 100% |
| Filter enable | 0.5 | 1.43e-39 | 0.005542 | 100% |

A fresh process rebuilds the accepted initializer/bootstrap provenance, complete audited cache, selected input identities, constants, hard exports, matrices and gates. It rejects altered evidence even if its outer success flag stays true. Final preparation takes **156.23 seconds**, execution **149.68 seconds**, and fresh verification **155.58 seconds**; each remains under its separate 10-minute/4-GiB allocation. The largest measured process tree is **0.78 GiB**, with zero sampling errors.

Preserve the preliminary passed check under its earlier source. A refused retry could replace its earlier status file, so the runner guard is corrected and the final source is separately frozen, executed and verified. All prior evidence stays intact. Preliminary preparation/execution costs 329.38 seconds. Including that work, all five passes total **790.88 seconds / 13.18 minutes**, add **zero native requests and zero optimizer updates**, and retain less than 2 MiB of evidence. Complete per-pass measurements and evidence hashes are in the public summary.

Twenty-seven distinct focused local checks pass for exact forward/export/tie behavior, native-loss derivatives, inactive-category jumps, RNG isolation, corrupted evidence, resource rejection and preservation on refused retries. Ruff and diff checks pass. GitHub Actions stays disabled.

Next complete the separately bounded native-direction/end-to-end check, then use reciprocal original/wider examples within each matched update. Preserve the earlier accepted weights, sound-loss objective, fixed phases and all reconstruction gates. This adapter addresses a concrete training/export mismatch; the previous regression's cause is not established, and no sound-quality gain is claimed here. A2 and later confirmation/listening/release stages remain open. Colab remains deferred.
