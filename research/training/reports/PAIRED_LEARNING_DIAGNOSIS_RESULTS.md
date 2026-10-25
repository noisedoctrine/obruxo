# The wider-recording failure follows the sound features

Completed 2026-10-24 using only saved training evidence. The [frozen plan](data/paired_learning_diagnosis_plan.json), source `77e4d0b`, has ID `c73e3d4e2f5d5809b5a347e8027f26fce5e6869818f34e57ce34b5f4c767b218`. All 128 updates, final models, Adam and Python/NumPy/Torch state replay exactly while gradients are observed. Parent files remain unchanged. Execution takes **14.33 seconds / 0.74 GiB**, after **3.55 seconds / 0.47 GiB** preparation, with zero new renders or added training updates. Twelve focused checks pass, and every reported aggregate is independently recomputed from the retained 128 gradient pairs and 1,344 feature cases.

## Which input branch moves predictions?

The accepted initializer has 4.46% source-active continuous error on original recordings, with both categories correct for all 64 patches. Wider recordings raise that error to about 30%. Swap one feature branch between two recordings of the **same patch**, keeping weights fixed:

| Initializer input | Wider held: control error | Wider alternate: control error |
|---|---:|---:|
| Both branches from the wider recording | 29.96% | 31.67% |
| Wider sound features, original note features | 29.89% | 31.25% |
| Original sound features, wider note features | 9.71% | 9.43% |

Restoring the original note vector barely repairs the wider-recording failure. Restoring the original sound features reduces error substantially even while note features stay wider. Unison accuracy similarly rises from 50.0%/39.1% to 93.8%/90.6% when original sound features are restored. The completed sound-trained model shows the same pattern: real wider inputs have 29.33%/30.88% control error; original sound features with wider note features reduce it to 10.29%/10.22%.

These artificial hybrids are diagnostics, not valid recordings or perceptual acceptance scores. Pitch, note lengths, patterns and velocities change together in the real recordings; this does **not** isolate pitch as the sole cause. It identifies the sound-feature path as an important place to improve interpretation of changed performances. Replacing only the separate note vector is poorly supported by this evidence.

## Do paired updates oppose each other?

The two observations produce opposing whole-model gradient components in **57/128 updates**. Their median alignment is near zero (cosine 0.021). The wider observation's gradient has median strength **0.73 times** the original observation's, so consistent domination by larger wider-input gradients is not established. The note-information branch has opposing gradients in 66/128 updates, as does fusion; the sound branch has 61/128. Alignment is descriptive evidence, not proof that a particular optimizer change will repair reconstruction.

The observer leaves every original update trace unchanged, including retained output-gradient measurements. This matters because inspecting gradients can otherwise accumulate values in the tensors being diagnosed. The [full report](data/paired_learning_diagnosis_summary.json) retains every case, group, gradient pair and exact-replay/resource verdict. No development or test recording is used.

## Next structural direction

Use estimated notes and timing to help interpret sound features before final parameter prediction, rather than relying on late concatenation alone. Test a small conditional correction with the earlier predictor retained and frozen, instead of another loss-weight sweep or parameter-only initialization. The existing conditioning implementation measures **426,271 total parameters**, comprising the retained **355,806** plus a **70,465-parameter** correction. It still predicts **14 values**; no new control head is proposed.

First qualify an initialization that reproduces the accepted predictor's outputs exactly, valid export and useful sound-gradient access to the new correction. Freeze trainable groups, initialization, update/native allocation and short fitting gates before training it. The correction has not yet been qualified, and the diagnosis does not establish that it will improve sound. All original/wider quality gates, fresh-family confirmation, seed replication and listening remain required.
