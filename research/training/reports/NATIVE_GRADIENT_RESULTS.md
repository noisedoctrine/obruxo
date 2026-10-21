# Direct native sound-loss gradients pass the local update check

Completed 2026-10-21 under the [frozen card](NATIVE_GRADIENT_CARD.md). The measured native-gradient path passes all declared direction and resource gates. Continuous updates improve **24/24** cases; combined continuous/category updates improve **23/24**, measured by both native MRSTFT and the primary spectral objective. This qualifies the bounded gradient method for a costed inverse-training trial. It does not establish a trained-model improvement.

[Audited evidence](data/native_gradient_summary.json) · Frozen source commit `6d61af7` · Plan `c9f829b06e4bc3993ff59d35b8e617100a89b64de41a4997368f3c52ac6a04d5`

## Native results

| Performance group | Update | Cases improving MRSTFT | Mean MRSTFT change | Mean primary-loss change | Gate |
|---|---|---:|---:|---:|---|
| Reference | Continuous | 12/12 | −0.24996 | −0.11080 | Pass |
| Reference | Continuous + categories | 12/12 | −0.73078 | −0.26121 | Pass |
| Transfer | Continuous | 12/12 | −0.19398 | −0.08524 | Pass |
| Transfer | Continuous + categories | 11/12 | −0.84059 | −0.29337 | Pass |

All four groups exceed the frozen 60% improvement requirement and have negative mean changes under both objectives. Fifteen joint candidates change a category. The transfer case at index nine worsens after the joint step: primary loss increases **0.11266**, MRSTFT **0.21177**. It remains in every denominator and mean. Local estimates do not guarantee that simultaneous category changes improve a sound.

Starts include an incorrect category in 16/24 cases. This differs from the earlier learned-helper starts, so the counts are not a matched head-to-head comparison with those helpers. These 12 source states and eight MIDI recipes were previously used for development. They are not fresh confirmation or final-test evidence. No source parameters enter the update calculation after the imperfect start is constructed.

The method renders legal nearby patches, measures the complete-recording sound loss and supplies a local estimated derivative. It uses no learned forward model. The native plugin remains nondifferentiable; the continuous finite differences and categorical conditional-loss tables are approximations. The qualification uses one deterministic straight-through category step and hard final selections. Connection to stochastic Gumbel choices is covered separately by correctness tests.

## Cost and verification

- **560/560 native attempts accepted**, with **24/24 sample-exact fixed-phase repeats**.
- **1,197.97 seconds (19.97 minutes)**, **942,465,024 bytes (0.88 GiB)** peak process-tree RSS, zero RSS sampling errors.
- **3,149,417 bytes (3.00 MiB)** retained logs/measurements/receipts. Audio is discarded after each case.
- Within the frozen 624-request, 30-minute, 8-GiB and 32-MiB caps. No retries, extra optimization steps, test states or inverse-training updates.

The publication script verifies the frozen source hashes and every attempt/outcome pair, reconstructs all native requests from the recorded patches and phase policy, and reconstructs every numerical gradient and final update from the saved scalar losses. It recomputes the gates and resource accounting. This is an audit of saved measurements and native receipts; it does not re-render or independently re-score discarded audio.

Twenty focused local tests pass: nine new estimator/inverse-connection/screen checks and eleven existing spectral/control checks. They cover bounded finite differences, inactive controls, actual Gumbel-score gradients, hard codec agreement, stale-estimate rejection, rejected renders, planned denominators, gradients reaching the actual 355,806-parameter inverse model, and exact model/optimizer/Gumbel recovery. Ruff and whitespace checks pass.

## Next step

Use the qualified method in a matched local inverse-training comparison, beginning with a small end-to-end fit check. The shared `native_reconstruction_loss` function is connected and tested, but no production training command or trained spectral candidate is claimed here. Both arms must start from the same existing basic checkpoint and retain the **14-output** head.

Cost the next card before training: one example can require 22 native queries for its gradient, in addition to target collection and real-output audits. The current A2 request cap cannot support thousands of such updates. Declare the actual fit size, update count, audit allocation and any justified budget revision before execution. Fresh validation, a second seed, listening and the final bundle remain later gates. The earlier neural helpers stay unqualified.
