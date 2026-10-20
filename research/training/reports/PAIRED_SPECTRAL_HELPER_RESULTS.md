# Paired sound-change training: improved, still not qualified

The frozen local run completed on 2026-10-20. **Aligned pair training helped some suggested changes, but the helper still fails the declared qualification gates.** It remains refused for inverse training. No inverse model was trained or promoted, and the basic prediction head remains 14 values.

[Frozen card](PAIRED_SPECTRAL_HELPER_CARD.md) · [Native evidence](data/paired_spectral_helper_summary.json) · [Training-pair diagnosis](data/paired_spectral_helper_diagnosis.json) · [First experiment](BASIC_SPECTRAL_HELPER_RESULTS.md)

## Real Vital results

| Check | Reference | Transfer | Required |
|---|---:|---:|---|
| Mean helper forward error | 0.576 | 0.710 | Reference ≤90% of best baseline; transfer no worse |
| Best baseline forward error | 0.909 | 0.944 | Train-only five-neighbor interpolation |
| Continuous-change ranking agreement | 12/24 (50%) | 13/24 (54%) | ≥70% each |
| Category-change ranking agreement | 9/12 (75%) | 10/12 (83%) | ≥70% each |
| Pan-change ranking agreement | 9/12 (75%) | 11/12 (92%) | ≥70% each |
| Active-detune ranking agreement | 6/8 (75%) | 4/8 (50%) | ≥70% each |
| Continuous-guided cases improved | 9/12 (75%) | 8/12 (67%) | ≥60% each |
| Joint-guided cases improved | 8/12 (67%) | 6/12 (50%) | ≥60% each |

All 240 qualification requests passed audio QA, including 24 sample-exact phase-fixed repeat pairs. No ranking ties occurred. Four inactive-detune cases per role were excluded and reported as declared. The failing gates are continuous ranking in both roles, transfer detune ranking, and transfer joint-guidance success rate. These thresholds were not changed after scoring.

| Guided update | Mean native MRSTFT change | 95% paired-family bootstrap interval | Mean new spectral-objective change |
|---|---:|---|---:|
| Reference, continuous | −0.3704 | [−0.5976, −0.1452] | −0.1660 |
| Reference, joint | −0.2970 | [−0.5837, −0.0341] | −0.1441 |
| Transfer, continuous | −0.1882 | [−0.4080, +0.0198] | −0.0712 |
| Transfer, joint | −0.1377 | [−0.4147, +0.1032] | −0.0614 |

Real error improved in 31/48 guided trials, versus 27/48 in the first experiment. Misleading decreases—lower helper error but higher native MRSTFT error—fell from 21/48 to 15/48. Those remaining errors prevent promotion. The reference intervals exclude zero; both transfer intervals include it. Joint guidance changed categories in two reference and three transfer cases.

This is a reused development panel, not an independent confirmation. The intervention changed pair construction, time alignment and the learning objective together, and includes the separately verified frame-boundary repair. It is not a loss-only ablation. Guided starts retain the source categories while perturbing continuous controls; these results do not establish recovery from incorrect initial categories. Category alternatives are tested separately.

## Diagnosis from saved data

The ranking problem persists when the helper is compared against **the same spectral objective it was trained to approximate**. Continuous agreement is 10/24 reference and 16/24 transfer under that objective, compared with 12/24 and 13/24 under the existing native metric. The two real-audio objectives disagree on only 5/48 continuous probes. Changing metrics therefore does not resolve the observed failure. This post-run comparison is diagnostic; the frozen admission decision stays unchanged.

An additional read-only pass checked all 508 complete training pairs at their stored times. It compared the helper's signed-change error against predicting no sound change at all. No new audio or optimizer updates were used. Lower errors are better; each row uses the declared paired-change objective, so errors across different control groups are not directly comparable.

| Changed control | Pairs | Mean helper change error | Mean no-change error | Pairs beating no change |
|---|---:|---:|---:|---:|
| Level | 29 | 0.1204 | 0.3019 | 29 |
| Pan | 29 | 0.1455 | 0.2140 | 26 |
| Detune | 21 | 0.5194 | 0.5235 | 14 |
| Filter cutoff | 23 | 0.2592 | 0.3504 | 23 |
| Filter resonance | 14 | 0.0339 | 0.0379 | 5 |
| Attack | 56 | 0.1985 | 0.1904 | 3 |
| Decay | 28 | 0.1025 | 0.0789 | 8 |
| Sustain | 28 | 0.1228 | 0.2261 | 26 |
| Release | 28 | 0.0458 | 0.0357 | 0 |
| Unison choice | 128 | 0.8006 | 0.8689 | 109 |
| Filter enable | 124 | 0.5566 | 1.1968 | 124 |

Attack, decay and release fail even this training-fit comparison on average; release beats the no-change baseline in none of its 28 pairs. Detune's average advantage is also small. This is evidence of an unresolved learning/representation problem within the training data, not just an unlucky held-out panel. Uniform time sampling, model conditioning and response scaling remain possible contributors; this diagnostic does not isolate one cause or justify a particular replacement automatically. It also does not measure correctness at unsampled times.

The diagnosis took 8.48 seconds, using existing checksummed features and a frozen helper. It explicitly loads the unqualified artifact only for inspection; ordinary inverse loading still refuses it. There is no checkpoint selection or additional training.

## Resources and verification

The run used 1,024 training attempts, 1,020 accepted records and 12,000 updates. Four clipping failures were retained without replacement. There were **no shape failures** after the renderer repair. Training used 508 complete pairs / 1,016 records; the other four accepted records have missing partners and were excluded from paired updates. They remain available to the train-only forward baselines.

Elapsed time was **4,987.00 seconds (83.12 minutes)**. Peak sampled process-tree RSS was **1,637,814,272 bytes (1.53 GiB)** with zero sampling errors. The retained archive occupied **1,249,218,928 bytes (1.16 GiB)**. All fit the frozen four-hour, 8-GiB RAM and 2-GiB archive ceilings. The helper still has 1,413,033 parameters. Training audio was not archived; final-test states were not used; all work was local.

Five paired-objective/collection/recovery tests and seven existing helper tests pass. The publication script recomputes the frozen qualification results, verifies plan identity and checks all native/repeat counts. The training-pair diagnostic verifies pair ancestry, MIDI, times and feature hashes. Frozen source hashes, the final 12,000-step checkpoint and refusal of the actual unqualified artifact were checked. Ruff and whitespace checks pass.

## Decision and next work

Keep this helper disconnected from inverse training. A1 remains incomplete; A2 cannot begin with this artifact. Preserve both failed runs and the unchanged admission criteria.

The next investigation should establish useful control responses directly before another full helper fit: inspect attack/release time coverage and compare a bounded real-Vital control-change probe with the helper's response. Any new native collection needs a separately frozen card and explicit render cost. This can determine whether to repair temporal supervision or pursue the plan's restricted differentiable-synth/native-optimization alternative. Do not start a third full fit by increasing model size, update count or patch coverage. Independent confirmation and listening remain later gates; Colab stays deferred.
