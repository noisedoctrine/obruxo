# Checking what the two audio branches contain

The first frozen-model screen finds performance information in the branch named “timbre” and patch information in the branch named “performance.” Those names describe intended roles, not established separation. The current architecture concatenates the two branches without enforcing independence. Detectable information alone does not prove the decoder relies on it or that a particular architecture repair will improve reconstruction.

## Existing recordings

The [frozen cached protocol](data/cached_representation_probes_plan.json) uses all **1,212 distinct recordings / 404 training patches**, removing the 404 repeated original-recording aliases. Labels come from the authoritative MIDI recipes and codec targets, not the pitch detector. Each patch and all three of its recordings stay in one of three probe folds, stratified by source unison/filter choices. The inverse encoders are unchanged. Fixed ridge readouts use training-fold standardization, regularization 0.1 and five shuffled-label controls, without a hyperparameter search.

| Frozen model / branch | Playing-pattern accuracy | Shuffled-label mean | Mean-pitch RMSE, semitones | Source-filter accuracy |
|---|---:|---:|---:|---:|
| Initial / timbre | 79.29% | 33.20% | 3.89 | 98.84% |
| Settings-trained / timbre | 79.54% | 33.40% | 3.84 | 98.93% |
| Sound-trained / timbre | 79.29% | 33.27% | 3.86 | 98.84% |
| Sound-trained / performance | 75.41% | 30.30% | 9.43 | 59.08% |
| Raw Basic Pitch features | 96.29% | 32.61% | 3.22 | 88.61% |

Playing-pattern chance is 33.33%; filter chance is 50%, with the largest filter class at 51.98%. Mean-pitch prediction from a training-fold mean has RMSE 11.21 semitones. These are descriptive linear-readout results; no significance threshold or architecture winner was selected afterward. Full source-control readouts, activity-mask denominators, continuous performance summaries, confusion matrices and cross-performance patch retrieval are in the [summary](data/cached_representation_probes_summary.json).

The timbre branch retains patch information as well: the sound-trained model retrieves the correct patch for 6.68% of original-to-held-note queries among 404 candidates (0.25% chance), and its filter readout remains strong. This rules out a completely constant representation, not all forms of collapse or adequate reconstruction. Conversely, low patch-identity retrieval from the performance branch does not establish absence of patch information; its source-filter readout remains above the shuffled controls.

All patches were already seen by the inverse model. Held-out here means unseen by the probe during its fit, not a new validation patch. Pitch, timing and velocity change together in these cached recordings. The screen therefore cannot attribute leakage independently to those factors. Velocity regression does not improve on the mean baseline in the timbre branch; it is not claimed as evidence of velocity leakage.

The screen completes in **74.02 seconds / 0.86 GiB peak RAM / 1.03 MiB evidence**, within ten minutes / 4 GiB / 16 MiB. It makes zero native requests and zero inverse-model optimizer updates. Source weights, cache and pool remain unchanged. The reserved 192 confirmation roots and final test set remain unused.

## Controlled factors and predicted-patch stability

The [controlled protocol](data/controlled_representation_probes_plan.json) completes all combinations of MIDI 48/60/72, velocity 64/112 and note duration 0.4/1.2 seconds for the twelve already-used training patches. All **144 clips** last 4.25 seconds, with identical onset, fixed oscillator/sampler random phases and 32,768-frame preroll. All **156 native requests** succeed, including one exact live repeat per patch. Original and constant-gain LUFS-matched audio are analyzed separately. The common matched level is -25.515 LUFS to preserve headroom; the independent meter's spread is **0.00614 LU** across all 144 clips.

Factor readouts hold out six entire patches at a time in two category-balanced folds. Patch-identity readouts train on MIDI 60 and test MIDI 48/72 for the same twelve patches; their chance accuracy is 8.33%. They test retention across unseen pitches, not unseen patch identities. Five shuffled-label controls and the same fixed ridge setting remain in force.

| Sound-trained branch / condition | Pitch accuracy (chance 33.33%) | Note-length accuracy (chance 50%) | Cross-pitch patch accuracy (chance 8.33%) |
|---|---:|---:|---:|
| Timbre / original | 83.33% | 95.83% | 52.08% |
| Timbre / LUFS matched | 91.67% | 95.83% | 70.83% |
| Performance / original | 93.06% | 97.22% | 12.50% |
| Performance / LUFS matched | 100.00% | 100.00% | 12.50% |

For the sound-trained timbre branch, shuffled-label pitch means are 32.78%/37.22% and note-length means are 47.50%/43.89% for original/matched audio. Useful patch information remains, but pitch and timing are readily recoverable. LUFS matching does not remove that leakage. Matching is a diagnostic condition, not a change to the original reconstruction objective or admission scores.

The output controls also respond strongly to performance changes of the same source patch. Across 144 pitch-only comparisons, the sound-trained model changes its selected unison category in **68.06%** of original-audio pairs; across 72 duration-only comparisons, it changes unison in **58.33%** and filter enable in **55.56%**. After matching, pitch-only unison changes remain at **75.00%**; duration-only changes are 25.00% for unison and 2.78% for filter enable. The [complete results](data/controlled_representation_probes_summary.json) retain each control's normalized mean/maximum changes, both category flip rates and all models/branches. Parameter instability is a diagnostic, not a new exact-parameter acceptance gate: different patches can produce similar audio, and these changed predictions have not been re-rendered here to assign a perceptual penalty.

**Velocity is uninformative in this scope.** All 72 low/high-velocity pairs have identical native waveforms, including before matching. The fixed basic template has `velocity_track = 0` and no modulation routes; the renderer does pass each note's actual velocity to Vital. Chance-level velocity readouts therefore do not demonstrate successful isolation. A meaningful velocity-response test needs a separately declared responsive template/capability. This run retains the unchanged basic template and does not silently add that capability.

Native collection, matching, both feature versions and all probes take **320.77 seconds / 0.83 GiB peak RAM**, retaining **218.14 MiB before summary**, within twenty minutes / 4 GiB / 512 MiB. No inverse-model optimizer runs and no new validation/test families are used. The [offline audit](data/controlled_representation_probes_verification.json) checks request identities, source-factor contrasts, constant-gain matching and exact regeneration of both feature versions from every saved waveform. Duplicate repeat audio was compared sample-for-sample live and not retained for a second offline comparison; the repeat receipts remain.

The [first launch](data/controlled_representation_probes_first_launch_failure.json) stops after one render call returns because its receipt subdirectory was missing. Its plan and failure record remain preserved; the corrected run uses a fresh directory with the same scientific design. Including that first call, the diagnostic uses **157 native calls**. No source patch, factor, acceptance tolerance or probe setting is changed to obtain a better score.

These checks do not establish successful separation. They identify pitch/timing information in the timbre features and substantial performance sensitivity in the predicted patch. The next shared-learning check must keep these measurements alongside reconstruction and retained patch information; reducing probe accuracy alone is insufficient. The existing fourteen-output budget, all six reconstruction groups, earlier-reference preservation, fresh confirmation and listening requirements remain unchanged. No model is promoted.
