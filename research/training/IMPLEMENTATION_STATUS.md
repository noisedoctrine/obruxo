# Training status: evidence and remaining work

Reviewed 31 October 2026. The [training plan](../modelling/TRAINING_PLAN.md) is the source of future priorities; the [lessons/review](reports/TRAINING_REVIEW_20261031.md) links plan evolution, PRs and results. This ledger records what is established. The [previous detailed ledger](https://github.com/noisedoctrine/obruxo/blob/ced6b0b/research/training/IMPLEMENTATION_STATUS.md) and individual reports preserve historical checks and run accounting. Consolidation does not change their gates or outcomes.

**No spectral-trained model has completed basic quality and release qualification.** The central unfinished task is reliable, affordable shared audio-to-patch learning across performances. Broad preset support, exact recovery and merged PRs are infrastructure progress, not evidence of a finished model.

## Learning and quality

| Requirement | Established evidence | Remaining work |
|---|---|---|
| A0: legal basic prediction/loss contract | [Basic contracts](reports/BASIC_SPECTRAL_CONTRACT_RESULTS.md); actual 14-output / 11-codec interface | Maintain under changed implementations; no broad heads in Stage A. |
| A1: useful sound feedback | [Settled native gradients](reports/SETTLED_NATIVE_GRADIENT_RESULTS.md), [production-choice qualification](reports/PRODUCTION_NATIVE_RESULTS.md) | Native path qualified only for this scope/profile. Existing learned helpers remain unqualified. |
| Individual sound fitting | [Three direct fits pass](reports/DIRECT_BASIC_CONTROL_RESULTS.md), reducing joint error 41–86% | Does not establish a shared predictor or unseen-patch quality. |
| A2: shared basic reconstruction | [Trainable-input fit](reports/TRAINABLE_BASIC_FIT_RESULTS.md) passes four of six groups; held-note groups fail. [Shared three-sound fit](reports/SHARED_BASIC_AUXILIARY_RESULTS.md) improves means but fails per-input/preservation requirements. | Finish balanced fit, then follow the plan's small shared-learning decision tree; retain all old quality gates. |
| Balanced input coverage | [Eighteen source clips and six repeats qualify](reports/BALANCED_BASIC_DATA_RESULTS.md), all three patches crossed with the same pitches/durations | [48-update fit](reports/BALANCED_BASIC_FIT_EXECUTION.md) is running; full endpoints/audit pending at this revision. No automatic extension or next run. |
| Performance/timbre separation | [Controlled probes](reports/BASIC_REPRESENTATION_PROBES_RESULTS.md) show pitch/duration information and output sensitivity | No successful separation claim. Tested [branch removal](reports/CALIBRATED_BRANCH_SCREEN_RESULTS.md) and [shared removal](reports/SHARED_BASIC_AUXILIARY_RESULTS.md) do not justify adoption or coefficient sweeps. |
| Loudness and listening | [Original-level diagnosis](reports/BASIC_LEVEL_DIAGNOSIS_RESULTS.md); [checked LUFS tone packet](reports/LOUDNESS_MATCHED_LISTENING_RESULTS.md) and usable replay/pill interface | First twelve responses contain five severe level confounds; corrected tone judgments and representative release listening remain pending. Input LUFS versus peak normalization is deferred, not selected. |
| A3: independent confirmation | [192 authored roots and fresh performances reserved](reports/BASIC_CONFIRMATION_RESERVATION_RESULTS.md), unscored | A2 pass, complete conditional execution plan, second seed, independent family/performance confirmation and listening. Reused 21-/23-case panels are development data. Final test remains reserved. |
| Later capabilities | Full codec, assets, expression, LFO/routing and mixed-source infrastructure exists | After basic quality, add one measured useful capability. Modulation starts with one LFO and at most two linear routes. No exhaustive configuration target. |

## Execution and delivery

| Area | Established evidence | Remaining work |
|---|---|---|
| Renderer correctness | [Fixed-phase worker checks](reports/RENDER_WORKER_RESULTS.md), [32,768-frame settling](reports/NATIVE_PREROLL_RESULTS.md), profile-aware targets/exports | Qualify any faster execution policy against this current basic profile. Do not reuse incompatible historical targets. |
| Training cost | [Review timing](reports/TRAINING_REVIEW_20261031.md#why-the-native-experiments-take-hours): 12,025 requests / 96 scheduled updates / 6 h 15 min | Isolate native DSP from plugin/state work; address one-process-per-request overhead and query amplification before another long fit. No speedup is claimed from the small new profile. |
| Audio-only inference | [Saved-model CLI](reports/BASIC_DIRECT_INFERENCE_RESULTS.md), [portable local bundle](reports/LOCAL_BASIC_BUNDLE_RESULTS.md), [fresh runtime](reports/CLEAN_BASIC_RUNTIME_RESULTS.md) reproduce audited presets | Deliver a quality-qualified model, final runtime/scope measurements, complete notices/model card and final-test report. Current bundle is a diagnostic research artifact. |
| Provenance/notices | [Notice inventory and frontend attribution](reports/INFERENCE_NOTICES_RESULTS.md), split/ancestry and immutable evidence tooling | Finish release obligations for the selected bundle; no release qualification is implied. |
| Local workflow | Appropriate local checks, checkpointed battery execution and ready PRs; PRs 137–139 are now merged | Actions remains disabled. Stay local/free-first; defer Colab/Linux/cloud work. Never merge without explicit approval. |

The next work order is: close the running balanced comparison, make the actual learning schedule affordable, establish shared learning on balanced small data, then independent quality and selective expansion. The current planning review runs no further experiment and changes no frozen card, test reservation or running implementation.
