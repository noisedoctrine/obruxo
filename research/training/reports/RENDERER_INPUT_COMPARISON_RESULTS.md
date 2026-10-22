# The renderer correction substantially changes inputs to the historical model

Compared 2026-10-22 using **32 saved cold/settled record pairs**. Patches, phase-controlled preset bytes, MIDI, native engine, frontend and initial weights are identical. The historical model reacts substantially to the corrected inputs: mean attack prediction changes **26.80% of its allowed range**, decay changes **24.83%**, and filter on/off flips in **4/32 cases**. This is an input-distribution and prediction-shift finding, **not a new sound-quality result or proof of the entire generalization failure's cause**.

[Paired evidence](data/renderer_input_comparison_summary.json) · [Stopped cold parent audit](data/native_inverse_prefix_summary.json) · [Settled parent audit](data/calibrated_native_inverse_fit_summary.json)

## What is held constant

The runner checks both prior audits and every parent evidence hash, all 32 record-file hashes and recipe identities, and accepted source receipts for both performances under both profiles. It verifies the same source/controlled preset bytes, native plugin fingerprint/backend, initial model digest, codec, frontend and model/phase implementations. The profiles are zero lead versus the qualified 32,768-frame lead. No examples, labels or performance recipes are reselected.

Only saved reference features enter the unchanged initial model. Production raw-score argmax determines category choices. Model weights and frozen frontend weights are checked unchanged afterward. There are **zero new native requests and zero optimizer updates**. No development or test predictions are collected.

## Measured shifts

Relative feature change is `norm(settled-cold) / max(norm(cold), 1e-12)` for each saved tensor pair:

| Input tensor | Mean relative change | Median | 90th percentile |
|---|---:|---:|---:|
| Stereo spectral features | 28.63% | 26.98% | 44.27% |
| Basic Pitch features | 2.66% | 2.38% | 4.34% |

These are feature-space differences, not percentages of audible error. The same weights produce the following mean absolute changes in bounded dense outputs, expressed as a percentage of each normalized control range:

| Control | Mean prediction change |
|---|---:|
| Oscillator level | 9.67% |
| Pan | 6.47% |
| Unison detune | 19.43% |
| Filter cutoff | 3.85% |
| Filter resonance | 16.68% |
| Attack | 26.80% |
| Decay | 24.83% |
| Sustain | 15.99% |
| Release | 16.98% |

The codec can canonicalize inactive dense outputs at export; the table does not assign them audible importance. Hard unison choice changes in **3/32 cases**, filter on/off in **4/32**. Exact source-category recovery is not a perceptual requirement.

Source-active normalized parameter errors also change in mixed directions. Attack MAE rises from 0.23755 to 0.30462 and level from 0.30702 to 0.34198; cutoff falls from 0.11640 to 0.09465 and resonance from 0.36632 to 0.29455. All nine controls and their denominators are included in the evidence. These mixed label errors do not establish that either profile sounds closer overall, and do not replace native sound scoring.

## Interpretation and next work

We corrected the native renderer but retained the historical starting checkpoint. This comparison shows that keeping the weights does not keep their inputs or predictions stable. The correction remains necessary for pipeline correctness; do not restore the cold startup transient to recover old feature behavior.

Next, implement a **render-profile-aware basic bootstrap on corrected renders**, using the existing training-only pool and the same 14 outputs. Freeze its collection/update/audit budget before training, persist phase/profile identity in caches/checkpoints, and reject cold feature reuse. This refresh is initialization work only: it does not replace the required perceptual training or unchanged sound-quality gates. Recheck category-scale admission against any new initial weights instead of relabelling the old calibration report. Compare actual hard exports before another expensive sound-training run.

## Verification and limits

The final identity-checked comparison takes **2.98 seconds** after imports and peaks at **577,880,064 bytes (0.54 GiB)** process-tree RAM, within its five-minute/8-GiB/64-MiB-evidence limits. Two focused tests cover unequal pair contracts, inference-source/profile differences, feature scaling and invalid pairs. Ruff and whitespace checks pass. An earlier draft before the extra native-identity checks is preserved privately; its numerical results are identical.

Cold input features are verified against their archived hashes and accepted receipts, **not remeasured from retained cold audio**. The settled parent's source-repeat audio previously independently reconstructed its cached features/targets. This comparison performs no waveform rescoring and claims no model-quality qualification. Cold records stay diagnostic and cannot enter corrected training. Stage A2, fresh confirmation, second seed, actual listening and bundle/release work remain unfinished. GitHub Actions stays disabled and Colab deferred.
