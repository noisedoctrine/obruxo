# Training plan delivery status

Updated 2026-10-16. The [training plan](../modelling/TRAINING_PLAN.md) remains the scope. The implemented pilot and broader capability checks are research artifacts; the broad finished model has not passed the release gates. Final-test audio remains reserved. GitHub Actions stays disabled, and compute remains free-first.

| Plan item | Delivered evidence | Remaining acceptance work |
|---|---|---|
| M0: contracts and measurement | Versioned identities, ancestry-aware splits, bounded online benchmark, reviewed Windows full-state renderer | Complete broad capability matrix and eligible training population; Linux/Colab native qualification or an explicit local-only deployment claim |
| M1: stream and codec | Recoverable online stream; full 775-field scalar outputs; learned registered wavetable/sample choices; qualified factory references | Train-only Era2 LFO representation; custom warps and seeds; ordered 64-slot connections and routing controls; native checks and recovery across the combined representation |
| M2: first trained model | Audio-only pilot, frozen frontend, retrieval baseline, native evaluation, portable checkpoints and exact retained-feature recovery | Broad tiny-set overfit and complete valid native output coverage; the narrow pilot does not establish the broad result |
| M3: useful alpha | Two-performance pipeline, optional identity objective, bounded conditioning comparison, alpha and full-state capability integrations | Broader synthesis coverage and performance curriculum with declared held-out ranges; verified improvements without component regressions |
| M4: audio-based improvement | Sound helper, native-confirmed alternate targets and optional audio-only candidate ranking | Helper/extra-target learning checks did not qualify; establish a real-render improvement for the broader candidate before promoting an objective |
| M5: broad candidate | Frozen validation and split machinery; earlier narrow state-coverage and seed comparisons | Combined broad representation, complete coverage, measured data scaling, shortlisted seeds, frozen native quality/robustness gates |
| M6: release | Narrow offline inference/export and release probes | Broad portable bundle, fresh-install/offline verification, asset notices, listening study, model card, runtime report and gated final test |

Next implementation: reuse the existing Era2 97-point phase/gain LFO representation, fit its learned assets only on training lineages, and qualify decoded curves with native sound. Then connect custom warps/seeds and the ordered modulation slots. These are missing patch capabilities, so their implementation does not depend on showing a gain on the small development panel.

After those capabilities work together, expand synthesis/performance coverage and measure training-data scaling. Freeze candidate-selection gates before consulting the final reserve. The release still requires a blind listening result and fresh-environment reproduction; software tests or a lower error on a few development cases cannot substitute for either.

Current capability details: [full renderer](FULL_RENDER_PROFILE.md), [scalar targets](FULL_SCALAR_TRAINING.md), [factory choices](FACTORY_ASSETS.md). Historical negative learning results remain recorded in their individual reports rather than being treated as completed quality gates.
