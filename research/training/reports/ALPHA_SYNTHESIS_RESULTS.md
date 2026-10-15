# Broader synthesis pipeline qualification

Measured locally on 2026-10-15. This change establishes a trainable broader vocabulary and a bounded integration run. It does not promote a reconstruction model or replace the selected pilot bundle. [Commands, scope and next directions](../ALPHA.md) and [machine-readable evidence](data/alpha_synthesis_qualification.json) accompany this report.

## Coverage

The alpha codec has 19 continuous targets and nine categorical heads. It covers two oscillators, four wavetable choices per oscillator, external-table scanning, oscillator-2 octave/fine tuning, filter/envelope controls, three LFO shapes driving filter cutoff, and chorus. Conditional heads and inactive continuous controls are masked; a static init wavetable has no frame target. Unsupported state remains rejected.

The deterministic 256-state pool has 197 training, 28 validation and 31 test states. Each split contains all five authoring levels:

| Level | Train | Validation | Test |
|---|---:|---:|---:|
| Wavetable | 38 | 7 | 7 |
| Layered | 41 | 3 | 7 |
| Filtered | 37 | 6 | 8 |
| Modulated | 41 | 6 | 4 |
| Chorus | 40 | 6 | 5 |

These are mixed coverage levels, not an adaptive training schedule. No final-test state was rendered. The pilot's separate 64-state final test reserve also remains untouched.

## Asset and native checks

The initial 56-render probe failed 22 renders at the Vita round-trip boundary. Native layering, modulation and chorus already passed; the failures involved older external wavetable formats. The unpublished exploratory pool is superseded by the final pool. No validation tolerance was relaxed.

Three exact source hashes now have explicit, deterministic imports into the authored format. The factory sources are Basic Shapes, Jaw Harp and Stabbed. Original source bytes and imported content identities remain pinned in local pools and bundles. The native import comparison loaded each original directly through Vital and each converted table through the production renderer, at frames 0, 127.5 and 255. **All 18 renders passed and all nine audio pairs were sample-identical** (waveform RMSE 0). This check took 46.27 seconds.

The final codec qualification passed **14/14 interventions and 56/56 renders** in 134.08 seconds. Each case includes a source, repeat, codec reconstruction and a disabled/alternate-feature reference. Checks cover each factory asset in both oscillator roles, frame scanning, layering, both octave choices, the LFO route and alternate shapes, and chorus. Every intervention exceeds the fixed spectral floor and measured repeat noise. The source-repeat and codec-roundtrip comparisons retain their existing tolerances.

This is bounded activation and round-trip evidence, not exhaustive coverage of every parameter combination or performance. Original factory payloads, model weights and generated audio are not committed to Git.

## Training smoke run

The existing trainer now derives its output heads from the codec and exports all required asset content. The alpha model has **359,676 trainable parameters**. Frozen Basic Pitch weights and buffers remained unchanged.

The run used four CPU threads, float32, 256 optimizer updates, batch size 8, learning rate 0.001, one native render worker and a fixed 64-fresh-example target. All **64/64 acknowledged renders passed**, without errors or QA rejections. It retained **2,474,240 bytes (2.36 MiB)** of features within an 8 MiB cap. No audio archive was written.

Those 64 distinct training states cover every level: 12 wavetable, 14 layered, 16 filtered, 13 modulated and nine chorus states.

Recorded training time was **171.83 seconds** and peak sampled process-tree RSS was **1,059,864,576 bytes (0.99 GiB)**. Feature extraction used 7.70 seconds and optimizer work 1.95 seconds; the complete path includes rendering, waiting and checkpoint costs. This short integration run provides no reason to pay for a GPU or move the native renderer to Colab. Linux/Colab native qualification remains open.

Training metrics are descriptive only: the initial metric uses the warm-up population and the final metric uses the retained 64 examples, so their difference is not a controlled learning-curve comparison. The smoke checkpoint is not an optimized alpha model.

## Verification and interpretation

The two-state validation smoke passed **20/20 native renders**, including fixed-template, train-only retrieval and source-repeat comparators under reference and transfer performances. A separate audio-only load of `model.pt` reproduced the checkpoint's first prediction and passed another **2/2 source/prediction renders**. The retained pilot bundle still loads with its original nine/two head dimensions and unchanged file hash. This is 160 accepted renders across the final import, codec, training, evaluation and export checks; the earlier failed exploratory probe is separate. Prefetched but unacknowledged training work is not counted as consumed examples.

Two development states are insufficient to judge reconstruction quality. The learned alpha checkpoint remains a smoke artifact; no selected pilot export was replaced, and no final-test state was used.

The full local training/renderer regression suite passed 141 tests. After the final inactive-frame correction, the expanded alpha suite passed all 21 tests, including three additional negative checks. Together these cover 144 distinct tests. Ruff and whitespace checks passed. Tests cover strict decoding, asset hashes/path containment, split/lineage enforcement, inactive gradients, deterministic recovery, checkpoint compatibility and audio-only export. GitHub Actions remains disabled as requested.

The next work is paired synthesis-identity learning, followed by an audio-based learning objective and bounded multiple-candidate refinement. Each gets a bounded feasibility screen before replication or detailed tuning. Release-head tuning is deferred. Historical pilot experiments should be reproduced from their recorded revisions; their frozen plans and results were not rewritten for the new codec.
