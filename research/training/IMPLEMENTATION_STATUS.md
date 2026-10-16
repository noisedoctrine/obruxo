# Training-plan completion ledger

This ledger tracks the [training and release plan](../modelling/TRAINING_PLAN.md). An implemented experiment does not complete a quality gate. PRs are submitted for review and are never merged automatically. Compute remains free-first, audio is generated on demand, and GitHub Actions remains disabled.

| Plan item | Completed evidence | Remaining work |
|---|---|---|
| M0: contracts and measurement | Frozen authored splits, asset/version checks, QA, local stream/resource measurements | Linux/Colab qualification or documented local-only execution; full eligible capability matrix |
| M1: stream and codec | Bounded online rendering, acknowledged recovery, narrow and alpha codecs, five-family authored curriculum | Broad canonical assets, train-only LFO representation, sampler/voice/effect/order/routing coverage |
| M2: first trained model | Audio-only inverse model, frozen Basic Pitch, baselines, exports and native validation | Finished-model quality remains open |
| M3: useful alpha | Wider oscillators/wavetables/LFO/effect vocabulary; paired identity implementation and fixed comparison | Broad performance curriculum; one justified conditioning comparison; reliable transfer gains |
| M4: sound-based training | Forward helper qualification in #45; direct-render fallback and matched learning implemented; [fixed fallback result](reports/RENDER_SELECTED_RESULTS.md) is +12.56% native error | Validated sound-quality gain; current unsuccessful options remain disabled by default |
| M5: broad candidate | Frozen final-test reserves and reproducible development comparisons | Complete declared coverage, measured data scaling, successful combined components, shortlisted seeds and frozen release validation |
| M6: release | Portable narrow inference bundle/CLI foundation | Final test after validation gate, fresh-environment broad bundle, runtime results, listening evidence, model card and release instructions |

Current tasks: measure and reduce repeated native-render setup costs, qualify audio-only candidate ranking, and complete one conditioning comparison. Direct-render selection accepted 220/220 training renders and 120/120 validation renders, with 13 independently confirmed alternative targets from 32 source recordings. Learning from these targets did not improve held-out audio. Shipped inference remains audio-only. Detailed proxy/target-weight tuning and repeated seed studies remain deferred until the larger directions are available.
