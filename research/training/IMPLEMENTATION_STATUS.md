# Training-plan completion ledger

This ledger tracks the [training and release plan](../modelling/TRAINING_PLAN.md). An implemented experiment does not complete a quality gate. PRs are submitted for review and are never merged automatically. Compute remains free-first, audio is generated on demand, and GitHub Actions remains disabled.

| Plan item | Completed evidence | Remaining work |
|---|---|---|
| M0: contracts and measurement | Frozen authored splits, asset/version checks, QA, local stream/resource measurements | Linux/Colab qualification or documented local-only execution; full eligible capability matrix |
| M1: stream and codec | Bounded online rendering, acknowledged recovery, narrow and alpha codecs, five-family authored curriculum | Broad canonical assets, train-only LFO representation, sampler/voice/effect/order/routing coverage |
| M2: first trained model | Audio-only inverse model, frozen Basic Pitch, baselines, exports and native validation | Finished-model quality remains open |
| M3: useful alpha | Wider oscillators/wavetables/LFO/effect vocabulary; paired identity and [one matched conditioning comparison](reports/CONDITIONING_RESULTS.md) implemented and measured | Broad performance curriculum and reliable transfer gains; small initial settings do not settle the component choices |
| M4: sound-based training | Forward helper qualification in #45; direct-render fallback and matched learning implemented; [fixed fallback result](reports/RENDER_SELECTED_RESULTS.md) is +12.56% native error | Validated sound-quality gain; current unsuccessful options remain disabled by default |
| M5: broad candidate | Frozen final-test reserves, reproducible comparisons and [qualified audio-only candidate selection](reports/AUDIO_RANKING_RESULTS.md), with 10.64% lower error in the bounded alpha check | Complete declared coverage, measured data scaling, successful combined components, shortlisted seeds and frozen release validation |
| M6: release | Portable narrow inference bundle/CLI foundation | Final test after validation gate, fresh-environment broad bundle, runtime results, listening evidence, model card and release instructions |

Current task: broaden canonical asset, LFO/custom-warp, sampler, voice, effect-order and slot-indexed routing coverage, then run measured data scaling and combined training. The initial identity, audio-objective, candidate and conditioning directions are implemented and screened. Audio-only candidate selection passed its initial gates; its ten-patch development result does not complete the broad-model quality gate. Worker reuse and phase-controlled correctness are in PR #49. Detailed weight tuning and repeated seed studies remain deferred while the larger coverage work is completed. Final-test and release gates remain open.
