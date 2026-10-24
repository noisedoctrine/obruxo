# Unison choices learn from sound; original reconstruction regresses

Completed 2026-10-24 under the [frozen card](UNISON_CHOICE_FIT_CARD.md), plan `9847596ab6e0be68ff352d78186126a9593b9dd91cd0e7ef2b0b6ac143f6b744`. Both arms finish all **192 paired updates** on the same 12 training patches / 36 recordings. All **376 new native requests** are accepted: 168 alternative-choice renders and 208 fresh final comparisons. No recovery is needed. **The preservation gate fails, so no model is admitted.** No development/test case is used.

The [independent audit](data/unison_sound_fit_verification.json) exactly reconstructs both models, Adam, RNG and every update, verifies the [first-four-update witness](data/unison_sound_fit_prefix.json), reconstructs native requests and cold exports, and rescores **480 candidate comparisons** from retained audio. That covers 252 three-choice costs, 208 fresh final predictions and 20 additional initial comparisons for the original-32 panel. Frozen weights, nine continuous network outputs and both filter scores remain unchanged. Only unison decisions learn; normal codec defaults still apply to inactive detune.

Training the sound arm uses expected measured spectral/envelope/stereo cost over three legal choices, with no parameter-label loss. Production and final evaluation use hard argmax presets. The matched control receives unison cross-entropy under the same schedule and update budget.

| Wider-input group, own/reference playback averaged | Sound-loss change | MRSTFT change | Available sound gain recovered |
|---|---:|---:|---:|
| Wider held | −3.13% | −4.96% | 56.86% |
| Wider alternate | −6.29% | −7.37% | 75.80% |

Changes compare the sound-trained model with initialization; lower loss is better. The best measured fixed-control unison choices offer 5.50% and 8.29% reductions in these mean sound losses. The model recovers more than the required half of each available gain and beats the parameter control on both means. All four wider own/transfer nonregression checks pass. This establishes useful category learning on these repeated training inputs; it does not establish full reconstruction quality or fresh-family generalization.

Original performance is the problem:

- On the 12 fitting patches, original-input own sound loss rises **5.28%** and original-input transfer sound loss rises **7.28%**. Both preservation groups fail.
- Across all 32 original patches, sound loss rises **28.26%**, mean MRSTFT rises **23.75%** (`0.323321 → 0.400105`) and MRSTFT p90 rises **38.08%** (`0.543779 → 0.750871`).
- The sound-trained head changes unison choices on **10/32 original inputs**, including one of the 12 fitting patches and nine of the other 20. Weight freezing preserves the earlier predictor's tensors, but the new correction can still change its established behaviour.

The [per-input choices](data/unison_sound_fit_choices.json) also show why parameter recovery and sound quality must remain separate. That file uses category indices 0/1/2 for 1/2/4 voices. On alternate inputs, source-unison accuracy drops from 6/12 to 2/12 while actual sound error improves. The parameter control reaches 8/12 source-unison accuracy there with little sound improvement. This is consistent with category changes compensating for errors in the fixed continuous predictions; it is not proof that the resulting patches are acoustically identical. Authored labels remain diagnostic, not candidate admission criteria.

Native execution takes **773.52 seconds (12 min 54 s) / 1.00 GiB** process-tree RAM; retained evidence is about **0.47 GiB**, below the 0.88-GiB calculated bound and 1-GiB ceiling. The independent audit takes **120.84 seconds / 0.71 GiB**, adding no native requests or experimental training updates. [Native service](data/unison_sound_fit_service_001.json) and [audit service](data/unison_sound_fit_service_002.json) both exit successfully, release the scoped active-work request and report zero RSS sampling errors. Eight local checks, Ruff and diff checks pass.

The final sound-model tensor digest is `7ed837b5c0e981a7f1ce8dd1448ce22624091cd020e989f744cb0a698ae117da`; the parameter control is `3b676d5032a76ae45c3109bf504a83379021cf9931297da5f8637823ce696726`. Both remain diagnostic artifacts. The accepted initializer and the **14-output** interface stay unchanged.

Do not run a larger unison-only retry: even the measured fixed-control oracle leaves most reconstruction error unresolved. Next, design a direct correction for the existing basic continuous controls as well as category scores, with explicit reference-sound rehearsal and preservation checks. This changes how the same basic outputs learn; it adds no oscillator/LFO/curve coverage. Qualify its starting behaviour, gradients and recovery before a separately budgeted fit. All full A2/A3 sound-quality gates remain unchanged, and fresh-family confirmation, a second seed, listening and release remain open.
