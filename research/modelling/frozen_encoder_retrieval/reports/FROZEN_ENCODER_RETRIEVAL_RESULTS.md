# Frozen audio encoder retrieval benchmark

Issue: [#152](https://github.com/noisedoctrine/obruxo/issues/152). This report covers frozen pretrained representations on original Vital source presets and existing MIDI performances. No synthesis controls are predicted, tuned, or optimized.

## Results and recommendation

**Finding:** LAION-CLAP Music HTSAT-base (official_clip) reached 58.9% cross-MIDI Recall@1 (n=6,450 queries, 129 candidate presets, 50 held-out MIDI performances). This is retrieval evidence, not a reconstruction result or a six-family trunk verdict.

### Original validation-paired cohort: 12 selected MIDI IDs; 12 with evaluated queries; 1548 common-gallery episodes
Validation preset gallery: 131 verified states; performances sourced from `validation`. Reserved test and confirmation sources used: **no**.
Selected-excerpt diversity: 4 monophonic / 8 polyphonic; 8 held, 4 phrased; median pitch 48–77; variable velocity in 3/12; expressive events in 0 sustain, 0 pitch bend, and 0 other CC/pressure excerpts.

| Frozen representation (output) | R@1 (%; preset-family / MIDI / crossed 95% CI) | R@5 (%) | MRR (%) | Same-MIDI negative win (%; preset-family / MIDI CI; queries) | Acoustic top-5 negative win (%; preset-family / MIDI CI; queries) | Query R@1 clean / warning (%) |
|---|---:|---:|---:|---:|---:|---:|
| LAION-CLAP Music HTSAT-base — official_clip (512-D) | 70.7% preset-family [67.3%, 74.0%] / MIDI [57.8%, 83.2%] / 2-way [57.0%, 82.7%] | 88.9% | 78.7% | 47.9% family [43.7%, 52.0%] / MIDI [33.2%, 61.9%] (q=1548) | 68.5% family [65.0%, 72.0%] / MIDI [55.4%, 80.7%] (q=1548) | 66.0% [52.9%, 79.7%] (q=941; MIDI IDs=12) / 78.1% [65.9%, 88.2%] (q=607; MIDI IDs=12) |
| Descript Audio Codec 44.1 kHz — temporal_pyramid_8 (16384-D) | 36.6% preset-family [32.1%, 41.1%] / MIDI [27.8%, 46.4%] / 2-way [26.6%, 47.3%] | 60.2% | 47.4% | 26.9% family [22.4%, 31.3%] / MIDI [20.2%, 34.9%] (q=1548) | 45.5% family [40.6%, 50.3%] / MIDI [37.5%, 54.3%] (q=1548) | 29.4% [21.1%, 42.1%] (q=941; MIDI IDs=12) / 47.8% [38.4%, 55.6%] (q=607; MIDI IDs=12) |
| MuQ-MuLan — official_clip (512-D) | 36.5% preset-family [32.7%, 40.2%] / MIDI [27.3%, 46.4%] / 2-way [26.6%, 47.0%] | 64.5% | 49.2% | 11.7% family [8.7%, 15.2%] / MIDI [7.5%, 16.7%] (q=1548) | 30.2% family [26.0%, 34.6%] / MIDI [22.2%, 40.2%] (q=1548) | 31.3% [22.6%, 41.1%] (q=941; MIDI IDs=12) / 44.5% [34.6%, 54.4%] (q=607; MIDI IDs=12) |
| Descript Audio Codec 44.1 kHz — masked_mean (1024-D) | 31.4% preset-family [27.3%, 35.6%] / MIDI [23.5%, 40.6%] / 2-way [22.2%, 41.5%] | 53.5% | 42.1% | 28.9% family [24.9%, 33.0%] / MIDI [20.9%, 38.6%] (q=1548) | 47.3% family [42.8%, 51.6%] / MIDI [38.8%, 57.2%] (q=1548) | 26.7% [18.4%, 38.0%] (q=941; MIDI IDs=12) / 38.7% [31.0%, 45.6%] (q=607; MIDI IDs=12) |
| Meta EnCodec 24 kHz — temporal_pyramid_8 (2048-D) | 29.8% preset-family [25.4%, 34.5%] / MIDI [22.8%, 39.2%] / 2-way [21.1%, 39.6%] | 51.3% | 40.9% | 18.3% family [14.4%, 22.4%] / MIDI [12.9%, 24.9%] (q=1548) | 33.9% family [28.9%, 39.0%] / MIDI [28.0%, 40.8%] (q=1548) | 21.8% [15.9%, 31.4%] (q=941; MIDI IDs=12) / 42.3% [32.8%, 51.1%] (q=607; MIDI IDs=12) |
| Meta EnCodec 24 kHz — masked_mean (128-D) | 24.5% preset-family [20.5%, 28.7%] / MIDI [16.9%, 34.0%] / 2-way [15.2%, 34.6%] | 42.1% | 34.1% | 21.3% family [17.1%, 25.6%] / MIDI [14.5%, 29.7%] (q=1548) | 34.4% family [29.3%, 39.4%] / MIDI [25.8%, 44.8%] (q=1548) | 18.4% [12.4%, 27.6%] (q=941; MIDI IDs=12) / 33.9% [23.9%, 43.3%] (q=607; MIDI IDs=12) |
| Random chance | 0.8% (fixed) | 3.9% | 4.2% | — | — | — |

Paired crossed-cluster Recall@1 differences from the leading family representation (`clap_music:official_clip`; best pooling per family):

| Comparator | ΔR@1 (percentage points) | Crossed MIDI × preset-family 95% CI |
|---|---:|---:|
| `dac_44khz:temporal_pyramid_8` | +34.1 | [+23.5, +44.4]  |
| `muq_mulan:official_clip` | +34.2 | [+27.5, +41.3]  |
| `encodec_24khz:temporal_pyramid_8` | +40.9 | [+29.9, +51.9]  |

R@1 intervals are shown in preset-family, performance-ID, and crossed performance × preset-family order (2,000 bootstrap replicates); the heading gives the independent performance denominator. Negative-win cells show query denominators and preset-family/performance-cluster intervals. Clean/warning strata use query-render QA status; the full report also retains gallery quality strata.

### Expanded held-out performance cohort: 50 selected MIDI IDs; 50 with evaluated queries; 6450 common-gallery episodes
Validation preset gallery: 131 verified states; performances sourced from `train`. Reserved test and confirmation sources used: **no**.
Performance selection: deterministic coverage-balanced sample of 50 from 96 compatible IDs; 46 sampled out and 0 exact-ID overlaps excluded.

Selected-excerpt diversity: 24 monophonic / 26 polyphonic; 30 held, 20 phrased; median pitch 29–89; variable velocity in 25/50; expressive events in 2 sustain, 0 pitch bend, and 0 other CC/pressure excerpts.

| Frozen representation (output) | R@1 (%; preset-family / MIDI / crossed 95% CI) | R@5 (%) | MRR (%) | Same-MIDI negative win (%; preset-family / MIDI CI; queries) | Acoustic top-5 negative win (%; preset-family / MIDI CI; queries) | Query R@1 clean / warning (%) |
|---|---:|---:|---:|---:|---:|---:|
| LAION-CLAP Music HTSAT-base — official_clip (512-D) | 58.9% preset-family [55.6%, 62.0%] / MIDI [50.9%, 66.6%] / 2-way [50.3%, 67.9%] | 79.8% | 68.4% | 40.1% family [36.5%, 43.8%] / MIDI [32.2%, 47.9%] (q=6450) | 59.6% family [56.0%, 63.0%] / MIDI [51.5%, 67.0%] (q=6450) | 54.6% [46.4%, 62.9%] (q=3778; MIDI IDs=50) / 65.0% [57.9%, 73.1%] (q=2672; MIDI IDs=50) |
| MuQ-MuLan — official_clip (512-D) | 30.2% preset-family [26.7%, 33.6%] / MIDI [25.7%, 34.6%] / 2-way [24.7%, 36.0%] | 52.9% | 41.3% | 10.4% family [7.7%, 13.7%] / MIDI [8.4%, 12.5%] (q=6450) | 28.2% family [24.4%, 32.4%] / MIDI [24.4%, 32.2%] (q=6450) | 26.0% [21.6%, 30.2%] (q=3778; MIDI IDs=50) / 36.1% [32.3%, 42.3%] (q=2672; MIDI IDs=50) |
| Random chance | 0.8% (fixed) | 3.9% | 4.2% | — | — | — |

Paired crossed-cluster Recall@1 differences from the leading family representation (`clap_music:official_clip`; best pooling per family):

| Comparator | ΔR@1 (percentage points) | Crossed MIDI × preset-family 95% CI |
|---|---:|---:|
| `muq_mulan:official_clip` | +28.7 | [+23.4, +34.2]  |

R@1 intervals are shown in preset-family, performance-ID, and crossed performance × preset-family order (2,000 bootstrap replicates); the heading gives the independent performance denominator. Negative-win cells show query denominators and preset-family/performance-cluster intervals. Clean/warning strata use query-render QA status; the full report also retains gallery quality strata.

Observed expanded-cohort lead: `clap_music:official_clip`. It ranked first on 50/50 individual MIDI performances and led in 100.0% of 2,000 performance-cluster resamples. Original-cohort leader: `clap_music:official_clip`; top-rank stability: stable across cohorts. Crossed-cluster ΔR@1 versus `muq_mulan:official_clip` is +28.7 percentage points [+23.4, +34.2]. Use `clap_music:official_clip` as the lead and `muq_mulan:official_clip` as a comparator for the later 14-output reconstruction screen.

**Largest descriptive performance strata for the expanded-cohort lead:**

- query-to-gallery pitch/register gap: lowest `13+ semitones` 44.8% [37.2%, 51.8%], 4386 queries / 34 MIDI IDs; highest `0-5 semitones` 96.0% [94.1%, 97.5%], 645 queries / 5 MIDI IDs.
- performance articulation: lowest `held` 54.0% [44.2%, 63.6%], 3870 queries / 30 MIDI IDs; highest `phrased` 66.3% [53.5%, 79.0%], 2580 queries / 20 MIDI IDs.
- query polyphony: lowest `monophonic` 53.8% [41.6%, 66.9%], 3096 queries / 24 MIDI IDs; highest `polyphonic` 63.6% [54.6%, 72.7%], 3354 queries / 26 MIDI IDs.
These are exploratory subgroup contrasts; the intervals resample MIDI IDs and do not imply a causal effect.

2 of six encoder families have measured local checkpoints and adapters in this run. Unavailable arms remain explicitly unranked; the final discussion records their checkpoint/license/runtime blockers.
No Vital controls are predicted or trained in retrieval. The linked [vital_control_inventory.jsonl](./vital_control_inventory.jsonl) inventories every exact source field/instance; each family is fixed with 0 enabled controls and 0 model outputs:

| Vital control family | Enabled / total | Output count | Update method / training signal |
|---|---:|---:|---|
| Global settings ❌ 0 / 562 | 0 | unchanged original preset; no training signal, loss, or gradient |
| Oscillators and voices ❌ 0 / 88 | 0 | unchanged original preset; no training signal, loss, or gradient |
| Filters ❌ 0 / 53 | 0 | unchanged original preset; no training signal, loss, or gradient |
| Envelopes ❌ 0 / 54 | 0 | unchanged original preset; no training signal, loss, or gradient |
| LFOs and modulation ❌ 0 / 170 | 0 | unchanged original preset; no training signal, loss, or gradient |
| Effects ❌ 0 / 46 | 0 | unchanged original preset; no training signal, loss, or gradient |
| Wavetables, curves and samples ❌ 0 / 6 | 0 | unchanged original preset; no training signal, loss, or gradient |
The expanded performance IDs came from the permitted `train` source split; 0 exact MIDI identities overlapping the original cohort were excluded. This is held out from the original MIDI IDs but remains within PresetShare source MIDI, not an external live-performance sample. The score is frozen retrieval on newly rendered audio, not reconstruction quality.
The acoustic top-5 negatives are selected by an independent spectral descriptor, not human-verified perceptual labels. Query and gallery QA warning strata are reported to show whether render warnings shift ranking.
**Unscored candidates and blockers:**

- CIR single-timbre AST ([source](https://github.com/gwendal-lv/CIR)): no verified local checkpoint lock entry; checkpoint license not specified in the upstream repository.
- BEATs iter3+ AS2M ([source](https://huggingface.co/lpepino/beats_ckpts)): no verified local checkpoint lock entry; mirror labels MIT; upstream checkpoint license remains unconfirmed ([license status](https://github.com/microsoft/unilm/issues/1760)).
- Meta EnCodec 24 kHz ([source](https://huggingface.co/facebook/encodec_24khz)): expanded-cohort results are incomplete; original-cohort results are reported above; MIT.
- Descript Audio Codec 44.1 kHz ([source](https://huggingface.co/descript/dac_44khz)): expanded-cohort results are incomplete; original-cohort results are reported above; MIT.


**Scope:** retrieval only. No current OBRUXO CNN baseline, shared-model training, source-label warm start, direct patch optimization, audio-loss training, or reconstruction head was run.

## Controlled preset perturbations

This secondary diagnostic uses one original validation preset, three single-control perturbations, 50 MIDI performances, and source/independently randomized Vital renders for each version. It reports cross-MIDI identity retrieval, same-MIDI retrieval against a different random seed, and perturbation distance relative to the identical-preset repeat distance. Query vectors never appear in their own galleries. The MIDI-cluster intervals summarize variation for this one preset and do not generalize across presets or controls.
R@5 is a four-version ceiling and is not interpreted. The LFO arm changes one selected routed frequency field; no phase or modulation route is optimized.
MIDI excerpt properties: 50 MIDI IDs; polyphony 1–7, median pitch 29–89, clip length 3.00–8.00s, median note duration 0.04–5.03s, variable velocity 25/50, articulation 30 held / 20 phrased. The original source carried 5 inherited Vital schema diagnostics ({"vital.parameter.unknown": 5}); perturbations introduced 0 new diagnostics. Velocity ranges are recorded, but velocity response was not isolated. The independent 64-bin log-frequency audio descriptor has no human audibility threshold. The lfo_2-routed LFO frequency perturbation had audio-descriptor distance 0.000699 [95% MIDI-cluster CI 0.000478, 0.000991]; it exceeded identical-preset seed-repeat distance on 0.4 of excerpts. This tests a routed LFO rate change, not phase reconstruction or full temporal-control recovery.
The linked [vital_control_inventory_controlled.jsonl](./vital_control_inventory_controlled.jsonl) records every exact Vital control/instance for each version (979 rows per version). No model predicts or optimizes these controls; each perturbation is manually specified, with every other source value held fixed.

### Original source preset

| Measure | Result or coverage | Output count | Update method / training signal |
|---|---:|---:|---|
| Applied Vital field | source preset unchanged | 0 | manual perturbation; no training, labels, gradients, or loss |
| LAION-CLAP Music HTSAT-base — official_clip cross-MIDI variant retrieval | R@1 0.2400 [0.12, 0.36]; R@5 1.0000; MRR 0.5683 (50 queries) | 512-D | frozen backbone; cosine retrieval |
| LAION-CLAP Music HTSAT-base — official_clip same-MIDI independent-seed control | R@1 0.7350 [0.685, 0.785]; MRR 0.8642 | 512-D | distinct source/independent seed renders; query excluded |
| LAION-CLAP Music HTSAT-base — official_clip identical-preset seed-repeat distance | 0.0400 [0.033171, 0.048484] | 512-D | same source preset/MIDI; varied Vital seed |
| LAION-CLAP Music HTSAT-base — official_clip extraction time / peak RSS / accelerator memory | 45.074 s / 4344885248 / 851259904 bytes | — | local inference only; no training |
| Descript Audio Codec 44.1 kHz — masked_mean cross-MIDI variant retrieval | R@1 0.2600 [0.14, 0.38]; R@5 1.0000; MRR 0.5683 (50 queries) | 1024-D | frozen backbone; cosine retrieval |
| Descript Audio Codec 44.1 kHz — masked_mean same-MIDI independent-seed control | R@1 0.6200 [0.555, 0.685]; MRR 0.7858 | 1024-D | distinct source/independent seed renders; query excluded |
| Descript Audio Codec 44.1 kHz — masked_mean identical-preset seed-repeat distance | 0.0085 [0.007049, 0.010177] | 1024-D | same source preset/MIDI; varied Vital seed |
| Descript Audio Codec 44.1 kHz — masked_mean extraction time / peak RSS / accelerator memory | 649.319 s / 11251425280 / 1020797952 bytes | — | local inference only; no training |
| Descript Audio Codec 44.1 kHz — temporal_pyramid_8 cross-MIDI variant retrieval | R@1 0.3000 [0.18, 0.44]; R@5 1.0000; MRR 0.5900 (50 queries) | 16384-D | frozen backbone; cosine retrieval |
| Descript Audio Codec 44.1 kHz — temporal_pyramid_8 same-MIDI independent-seed control | R@1 0.5950 [0.535, 0.655]; MRR 0.7725 | 16384-D | distinct source/independent seed renders; query excluded |
| Descript Audio Codec 44.1 kHz — temporal_pyramid_8 identical-preset seed-repeat distance | 0.0118 [0.009893, 0.014161] | 16384-D | same source preset/MIDI; varied Vital seed |
| Descript Audio Codec 44.1 kHz — temporal_pyramid_8 extraction time / peak RSS / accelerator memory | 649.319 s / 11251425280 / 1020797952 bytes | — | local inference only; no training |
| Meta EnCodec 24 kHz — masked_mean cross-MIDI variant retrieval | R@1 0.0800 [0.02, 0.16]; R@5 1.0000; MRR 0.4667 (50 queries) | 128-D | frozen backbone; cosine retrieval |
| Meta EnCodec 24 kHz — masked_mean same-MIDI independent-seed control | R@1 0.7150 [0.655, 0.77]; MRR 0.8450 | 128-D | distinct source/independent seed renders; query excluded |
| Meta EnCodec 24 kHz — masked_mean identical-preset seed-repeat distance | 0.0000 [3.8e-05, 5.2e-05] | 128-D | same source preset/MIDI; varied Vital seed |
| Meta EnCodec 24 kHz — masked_mean extraction time / peak RSS / accelerator memory | 217.528 s / 11751788544 / 249600000 bytes | — | local inference only; no training |
| Meta EnCodec 24 kHz — temporal_pyramid_8 cross-MIDI variant retrieval | R@1 0.1200 [0.04, 0.22]; R@5 1.0000; MRR 0.4700 (50 queries) | 2048-D | frozen backbone; cosine retrieval |
| Meta EnCodec 24 kHz — temporal_pyramid_8 same-MIDI independent-seed control | R@1 0.6150 [0.55, 0.675]; MRR 0.7821 | 2048-D | distinct source/independent seed renders; query excluded |
| Meta EnCodec 24 kHz — temporal_pyramid_8 identical-preset seed-repeat distance | 0.0007 [0.000608, 0.000698] | 2048-D | same source preset/MIDI; varied Vital seed |
| Meta EnCodec 24 kHz — temporal_pyramid_8 extraction time / peak RSS / accelerator memory | 217.528 s / 11751788544 / 249600000 bytes | — | local inference only; no training |
| MuQ-MuLan — official_clip cross-MIDI variant retrieval | R@1 0.3800 [0.26, 0.52]; R@5 1.0000; MRR 0.6317 (50 queries) | 512-D | frozen backbone; cosine retrieval |
| MuQ-MuLan — official_clip same-MIDI independent-seed control | R@1 0.7250 [0.675, 0.775]; MRR 0.8517 | 512-D | distinct source/independent seed renders; query excluded |
| MuQ-MuLan — official_clip identical-preset seed-repeat distance | 0.0249 [0.019934, 0.030122] | 512-D | same source preset/MIDI; varied Vital seed |
| MuQ-MuLan — official_clip extraction time / peak RSS / accelerator memory | 108.463 s / 10937692160 / 2873248768 bytes | — | local inference only; no training |
| Global settings ❌ 0 / 562 | fixed source except the listed field | 0 | fixed original preset input; no supervision or loss |
| Oscillators and voices ❌ 0 / 88 | fixed source except the listed field | 0 | fixed original preset input; no supervision or loss |
| Filters ❌ 0 / 53 | fixed source except the listed field | 0 | fixed original preset input; no supervision or loss |
| Envelopes ❌ 0 / 54 | fixed source except the listed field | 0 | fixed original preset input; no supervision or loss |
| LFOs and modulation ❌ 0 / 170 | fixed source except the listed field | 0 | fixed original preset input; no supervision or loss |
| Effects ❌ 0 / 46 | fixed source except the listed field | 0 | fixed original preset input; no supervision or loss |
| Wavetables, curves and samples ❌ 0 / 6 | fixed source except the listed field | 0 | fixed original preset input; no supervision or loss |

### Filter cutoff perturbation

| Measure | Result or coverage | Output count | Update method / training signal |
|---|---:|---:|---|
| Applied Vital field | filter_1_cutoff: 8 → 33.6 raw (0.000 → 0.200 normalized) | 0 | manual perturbation; no training, labels, gradients, or loss |
| LAION-CLAP Music HTSAT-base — official_clip cross-MIDI variant retrieval | R@1 0.6400 [0.5, 0.78]; R@5 1.0000; MRR 0.7717 (50 queries) | 512-D | frozen backbone; cosine retrieval |
| LAION-CLAP Music HTSAT-base — official_clip same-MIDI independent-seed control | R@1 0.7350 [0.685, 0.785]; MRR 0.8642 | 512-D | distinct source/independent seed renders; query excluded |
| LAION-CLAP Music HTSAT-base — official_clip perturbation distance / fraction exceeding repeat | 0.1087 [0.095936, 0.123112] / 1.0000 [1.0, 1.0] | 512-D | distance from original at same MIDI/seed; bootstrap by MIDI |
| LAION-CLAP Music HTSAT-base — official_clip extraction time / peak RSS / accelerator memory | 45.074 s / 4344885248 / 851259904 bytes | — | local inference only; no training |
| Descript Audio Codec 44.1 kHz — masked_mean cross-MIDI variant retrieval | R@1 0.6800 [0.54, 0.8]; R@5 1.0000; MRR 0.7667 (50 queries) | 1024-D | frozen backbone; cosine retrieval |
| Descript Audio Codec 44.1 kHz — masked_mean same-MIDI independent-seed control | R@1 0.6200 [0.555, 0.685]; MRR 0.7858 | 1024-D | distinct source/independent seed renders; query excluded |
| Descript Audio Codec 44.1 kHz — masked_mean perturbation distance / fraction exceeding repeat | 0.1793 [0.14381, 0.214155] / 1.0000 [1.0, 1.0] | 1024-D | distance from original at same MIDI/seed; bootstrap by MIDI |
| Descript Audio Codec 44.1 kHz — masked_mean extraction time / peak RSS / accelerator memory | 649.319 s / 11251425280 / 1020797952 bytes | — | local inference only; no training |
| Descript Audio Codec 44.1 kHz — temporal_pyramid_8 cross-MIDI variant retrieval | R@1 0.7800 [0.66, 0.88]; R@5 1.0000; MRR 0.8500 (50 queries) | 16384-D | frozen backbone; cosine retrieval |
| Descript Audio Codec 44.1 kHz — temporal_pyramid_8 same-MIDI independent-seed control | R@1 0.5950 [0.535, 0.655]; MRR 0.7725 | 16384-D | distinct source/independent seed renders; query excluded |
| Descript Audio Codec 44.1 kHz — temporal_pyramid_8 perturbation distance / fraction exceeding repeat | 0.0357 [0.032301, 0.039249] / 0.9800 [0.94, 1.0] | 16384-D | distance from original at same MIDI/seed; bootstrap by MIDI |
| Descript Audio Codec 44.1 kHz — temporal_pyramid_8 extraction time / peak RSS / accelerator memory | 649.319 s / 11251425280 / 1020797952 bytes | — | local inference only; no training |
| Meta EnCodec 24 kHz — masked_mean cross-MIDI variant retrieval | R@1 0.7000 [0.56, 0.82]; R@5 1.0000; MRR 0.7800 (50 queries) | 128-D | frozen backbone; cosine retrieval |
| Meta EnCodec 24 kHz — masked_mean same-MIDI independent-seed control | R@1 0.7150 [0.655, 0.77]; MRR 0.8450 | 128-D | distinct source/independent seed renders; query excluded |
| Meta EnCodec 24 kHz — masked_mean perturbation distance / fraction exceeding repeat | 0.0013 [0.001093, 0.001414] / 1.0000 [1.0, 1.0] | 128-D | distance from original at same MIDI/seed; bootstrap by MIDI |
| Meta EnCodec 24 kHz — masked_mean extraction time / peak RSS / accelerator memory | 217.528 s / 11751788544 / 249600000 bytes | — | local inference only; no training |
| Meta EnCodec 24 kHz — temporal_pyramid_8 cross-MIDI variant retrieval | R@1 0.7400 [0.62, 0.86]; R@5 1.0000; MRR 0.8117 (50 queries) | 2048-D | frozen backbone; cosine retrieval |
| Meta EnCodec 24 kHz — temporal_pyramid_8 same-MIDI independent-seed control | R@1 0.6150 [0.55, 0.675]; MRR 0.7821 | 2048-D | distinct source/independent seed renders; query excluded |
| Meta EnCodec 24 kHz — temporal_pyramid_8 perturbation distance / fraction exceeding repeat | 0.0030 [0.002873, 0.003226] / 1.0000 [1.0, 1.0] | 2048-D | distance from original at same MIDI/seed; bootstrap by MIDI |
| Meta EnCodec 24 kHz — temporal_pyramid_8 extraction time / peak RSS / accelerator memory | 217.528 s / 11751788544 / 249600000 bytes | — | local inference only; no training |
| MuQ-MuLan — official_clip cross-MIDI variant retrieval | R@1 0.6200 [0.48, 0.76]; R@5 1.0000; MRR 0.7417 (50 queries) | 512-D | frozen backbone; cosine retrieval |
| MuQ-MuLan — official_clip same-MIDI independent-seed control | R@1 0.7250 [0.675, 0.775]; MRR 0.8517 | 512-D | distinct source/independent seed renders; query excluded |
| MuQ-MuLan — official_clip perturbation distance / fraction exceeding repeat | 0.1150 [0.095218, 0.137388] / 1.0000 [1.0, 1.0] | 512-D | distance from original at same MIDI/seed; bootstrap by MIDI |
| MuQ-MuLan — official_clip extraction time / peak RSS / accelerator memory | 108.463 s / 10937692160 / 2873248768 bytes | — | local inference only; no training |
| Global settings ❌ 0 / 562 | fixed source except the listed field | 0 | fixed original preset input; no supervision or loss |
| Oscillators and voices ❌ 0 / 88 | fixed source except the listed field | 0 | fixed original preset input; no supervision or loss |
| Filters ✅ 1 / 53 | fixed source except the listed field | 0 | manual perturbation; no model output; no supervision or loss |
| Envelopes ❌ 0 / 54 | fixed source except the listed field | 0 | fixed original preset input; no supervision or loss |
| LFOs and modulation ❌ 0 / 170 | fixed source except the listed field | 0 | fixed original preset input; no supervision or loss |
| Effects ❌ 0 / 46 | fixed source except the listed field | 0 | fixed original preset input; no supervision or loss |
| Wavetables, curves and samples ❌ 0 / 6 | fixed source except the listed field | 0 | fixed original preset input; no supervision or loss |

### Amplitude envelope attack perturbation

| Measure | Result or coverage | Output count | Update method / training signal |
|---|---:|---:|---|
| Applied Vital field | env_1_attack: 0 → 0.475684 raw (0.000 → 0.200 normalized) | 0 | manual perturbation; no training, labels, gradients, or loss |
| LAION-CLAP Music HTSAT-base — official_clip cross-MIDI variant retrieval | R@1 0.6000 [0.46, 0.74]; R@5 1.0000; MRR 0.7500 (50 queries) | 512-D | frozen backbone; cosine retrieval |
| LAION-CLAP Music HTSAT-base — official_clip same-MIDI independent-seed control | R@1 0.7350 [0.685, 0.785]; MRR 0.8642 | 512-D | distinct source/independent seed renders; query excluded |
| LAION-CLAP Music HTSAT-base — official_clip perturbation distance / fraction exceeding repeat | 0.1061 [0.08982, 0.123686] / 0.9600 [0.9, 1.0] | 512-D | distance from original at same MIDI/seed; bootstrap by MIDI |
| LAION-CLAP Music HTSAT-base — official_clip extraction time / peak RSS / accelerator memory | 45.074 s / 4344885248 / 851259904 bytes | — | local inference only; no training |
| Descript Audio Codec 44.1 kHz — masked_mean cross-MIDI variant retrieval | R@1 0.4800 [0.34, 0.62]; R@5 1.0000; MRR 0.6650 (50 queries) | 1024-D | frozen backbone; cosine retrieval |
| Descript Audio Codec 44.1 kHz — masked_mean same-MIDI independent-seed control | R@1 0.6200 [0.555, 0.685]; MRR 0.7858 | 1024-D | distinct source/independent seed renders; query excluded |
| Descript Audio Codec 44.1 kHz — masked_mean perturbation distance / fraction exceeding repeat | 0.0119 [0.009631, 0.014092] / 0.7000 [0.56, 0.82] | 1024-D | distance from original at same MIDI/seed; bootstrap by MIDI |
| Descript Audio Codec 44.1 kHz — masked_mean extraction time / peak RSS / accelerator memory | 649.319 s / 11251425280 / 1020797952 bytes | — | local inference only; no training |
| Descript Audio Codec 44.1 kHz — temporal_pyramid_8 cross-MIDI variant retrieval | R@1 0.3400 [0.2, 0.48]; R@5 1.0000; MRR 0.5667 (50 queries) | 16384-D | frozen backbone; cosine retrieval |
| Descript Audio Codec 44.1 kHz — temporal_pyramid_8 same-MIDI independent-seed control | R@1 0.5950 [0.535, 0.655]; MRR 0.7725 | 16384-D | distinct source/independent seed renders; query excluded |
| Descript Audio Codec 44.1 kHz — temporal_pyramid_8 perturbation distance / fraction exceeding repeat | 0.0121 [0.010385, 0.01424] / 0.6200 [0.48, 0.74] | 16384-D | distance from original at same MIDI/seed; bootstrap by MIDI |
| Descript Audio Codec 44.1 kHz — temporal_pyramid_8 extraction time / peak RSS / accelerator memory | 649.319 s / 11251425280 / 1020797952 bytes | — | local inference only; no training |
| Meta EnCodec 24 kHz — masked_mean cross-MIDI variant retrieval | R@1 0.4600 [0.32, 0.6]; R@5 1.0000; MRR 0.6250 (50 queries) | 128-D | frozen backbone; cosine retrieval |
| Meta EnCodec 24 kHz — masked_mean same-MIDI independent-seed control | R@1 0.7150 [0.655, 0.77]; MRR 0.8450 | 128-D | distinct source/independent seed renders; query excluded |
| Meta EnCodec 24 kHz — masked_mean perturbation distance / fraction exceeding repeat | 0.0001 [5.5e-05, 7.8e-05] / 0.8000 [0.68, 0.9] | 128-D | distance from original at same MIDI/seed; bootstrap by MIDI |
| Meta EnCodec 24 kHz — masked_mean extraction time / peak RSS / accelerator memory | 217.528 s / 11751788544 / 249600000 bytes | — | local inference only; no training |
| Meta EnCodec 24 kHz — temporal_pyramid_8 cross-MIDI variant retrieval | R@1 0.4800 [0.34, 0.62]; R@5 1.0000; MRR 0.6583 (50 queries) | 2048-D | frozen backbone; cosine retrieval |
| Meta EnCodec 24 kHz — temporal_pyramid_8 same-MIDI independent-seed control | R@1 0.6150 [0.55, 0.675]; MRR 0.7821 | 2048-D | distinct source/independent seed renders; query excluded |
| Meta EnCodec 24 kHz — temporal_pyramid_8 perturbation distance / fraction exceeding repeat | 0.0007 [0.000657, 0.000789] / 0.8000 [0.68, 0.9] | 2048-D | distance from original at same MIDI/seed; bootstrap by MIDI |
| Meta EnCodec 24 kHz — temporal_pyramid_8 extraction time / peak RSS / accelerator memory | 217.528 s / 11751788544 / 249600000 bytes | — | local inference only; no training |
| MuQ-MuLan — official_clip cross-MIDI variant retrieval | R@1 0.5800 [0.44, 0.72]; R@5 1.0000; MRR 0.7433 (50 queries) | 512-D | frozen backbone; cosine retrieval |
| MuQ-MuLan — official_clip same-MIDI independent-seed control | R@1 0.7250 [0.675, 0.775]; MRR 0.8517 | 512-D | distinct source/independent seed renders; query excluded |
| MuQ-MuLan — official_clip perturbation distance / fraction exceeding repeat | 0.0626 [0.054102, 0.071494] / 0.9200 [0.84, 0.98] | 512-D | distance from original at same MIDI/seed; bootstrap by MIDI |
| MuQ-MuLan — official_clip extraction time / peak RSS / accelerator memory | 108.463 s / 10937692160 / 2873248768 bytes | — | local inference only; no training |
| Global settings ❌ 0 / 562 | fixed source except the listed field | 0 | fixed original preset input; no supervision or loss |
| Oscillators and voices ❌ 0 / 88 | fixed source except the listed field | 0 | fixed original preset input; no supervision or loss |
| Filters ❌ 0 / 53 | fixed source except the listed field | 0 | fixed original preset input; no supervision or loss |
| Envelopes ✅ 1 / 54 | fixed source except the listed field | 0 | manual perturbation; no model output; no supervision or loss |
| LFOs and modulation ❌ 0 / 170 | fixed source except the listed field | 0 | fixed original preset input; no supervision or loss |
| Effects ❌ 0 / 46 | fixed source except the listed field | 0 | fixed original preset input; no supervision or loss |
| Wavetables, curves and samples ❌ 0 / 6 | fixed source except the listed field | 0 | fixed original preset input; no supervision or loss |

### Routed LFO frequency perturbation

| Measure | Result or coverage | Output count | Update method / training signal |
|---|---:|---:|---|
| Applied Vital field | lfo_2_frequency: 1 → 5 raw (0.500 → 0.750 normalized); lfo_2 → filter_1_resonance | 0 | manual perturbation; no training, labels, gradients, or loss |
| LAION-CLAP Music HTSAT-base — official_clip cross-MIDI variant retrieval | R@1 0.3400 [0.2, 0.48]; R@5 1.0000; MRR 0.6133 (50 queries) | 512-D | frozen backbone; cosine retrieval |
| LAION-CLAP Music HTSAT-base — official_clip same-MIDI independent-seed control | R@1 0.7350 [0.685, 0.785]; MRR 0.8642 | 512-D | distinct source/independent seed renders; query excluded |
| LAION-CLAP Music HTSAT-base — official_clip perturbation distance / fraction exceeding repeat | 0.0391 [0.032493, 0.046802] / 0.4400 [0.3, 0.58] | 512-D | distance from original at same MIDI/seed; bootstrap by MIDI |
| LAION-CLAP Music HTSAT-base — official_clip extraction time / peak RSS / accelerator memory | 45.074 s / 4344885248 / 851259904 bytes | — | local inference only; no training |
| Descript Audio Codec 44.1 kHz — masked_mean cross-MIDI variant retrieval | R@1 0.1600 [0.06, 0.26]; R@5 1.0000; MRR 0.5133 (50 queries) | 1024-D | frozen backbone; cosine retrieval |
| Descript Audio Codec 44.1 kHz — masked_mean same-MIDI independent-seed control | R@1 0.6200 [0.555, 0.685]; MRR 0.7858 | 1024-D | distinct source/independent seed renders; query excluded |
| Descript Audio Codec 44.1 kHz — masked_mean perturbation distance / fraction exceeding repeat | 0.0099 [0.007551, 0.012784] / 0.5000 [0.36, 0.64] | 1024-D | distance from original at same MIDI/seed; bootstrap by MIDI |
| Descript Audio Codec 44.1 kHz — masked_mean extraction time / peak RSS / accelerator memory | 649.319 s / 11251425280 / 1020797952 bytes | — | local inference only; no training |
| Descript Audio Codec 44.1 kHz — temporal_pyramid_8 cross-MIDI variant retrieval | R@1 0.2000 [0.1, 0.32]; R@5 1.0000; MRR 0.5317 (50 queries) | 16384-D | frozen backbone; cosine retrieval |
| Descript Audio Codec 44.1 kHz — temporal_pyramid_8 same-MIDI independent-seed control | R@1 0.5950 [0.535, 0.655]; MRR 0.7725 | 16384-D | distinct source/independent seed renders; query excluded |
| Descript Audio Codec 44.1 kHz — temporal_pyramid_8 perturbation distance / fraction exceeding repeat | 0.0120 [0.010025, 0.014529] / 0.4800 [0.34, 0.62] | 16384-D | distance from original at same MIDI/seed; bootstrap by MIDI |
| Descript Audio Codec 44.1 kHz — temporal_pyramid_8 extraction time / peak RSS / accelerator memory | 649.319 s / 11251425280 / 1020797952 bytes | — | local inference only; no training |
| Meta EnCodec 24 kHz — masked_mean cross-MIDI variant retrieval | R@1 0.1800 [0.08, 0.28]; R@5 1.0000; MRR 0.5467 (50 queries) | 128-D | frozen backbone; cosine retrieval |
| Meta EnCodec 24 kHz — masked_mean same-MIDI independent-seed control | R@1 0.7150 [0.655, 0.77]; MRR 0.8450 | 128-D | distinct source/independent seed renders; query excluded |
| Meta EnCodec 24 kHz — masked_mean perturbation distance / fraction exceeding repeat | 0.0000 [4e-05, 5.9e-05] / 0.6000 [0.46, 0.74] | 128-D | distance from original at same MIDI/seed; bootstrap by MIDI |
| Meta EnCodec 24 kHz — masked_mean extraction time / peak RSS / accelerator memory | 217.528 s / 11751788544 / 249600000 bytes | — | local inference only; no training |
| Meta EnCodec 24 kHz — temporal_pyramid_8 cross-MIDI variant retrieval | R@1 0.1600 [0.08, 0.26]; R@5 1.0000; MRR 0.5017 (50 queries) | 2048-D | frozen backbone; cosine retrieval |
| Meta EnCodec 24 kHz — temporal_pyramid_8 same-MIDI independent-seed control | R@1 0.6150 [0.55, 0.675]; MRR 0.7821 | 2048-D | distinct source/independent seed renders; query excluded |
| Meta EnCodec 24 kHz — temporal_pyramid_8 perturbation distance / fraction exceeding repeat | 0.0007 [0.000602, 0.000717] / 0.4600 [0.32, 0.6] | 2048-D | distance from original at same MIDI/seed; bootstrap by MIDI |
| Meta EnCodec 24 kHz — temporal_pyramid_8 extraction time / peak RSS / accelerator memory | 217.528 s / 11751788544 / 249600000 bytes | — | local inference only; no training |
| MuQ-MuLan — official_clip cross-MIDI variant retrieval | R@1 0.3000 [0.18, 0.44]; R@5 1.0000; MRR 0.5600 (50 queries) | 512-D | frozen backbone; cosine retrieval |
| MuQ-MuLan — official_clip same-MIDI independent-seed control | R@1 0.7250 [0.675, 0.775]; MRR 0.8517 | 512-D | distinct source/independent seed renders; query excluded |
| MuQ-MuLan — official_clip perturbation distance / fraction exceeding repeat | 0.0255 [0.019883, 0.032278] / 0.4600 [0.32, 0.6] | 512-D | distance from original at same MIDI/seed; bootstrap by MIDI |
| MuQ-MuLan — official_clip extraction time / peak RSS / accelerator memory | 108.463 s / 10937692160 / 2873248768 bytes | — | local inference only; no training |
| Global settings ❌ 0 / 562 | fixed source except the listed field | 0 | fixed original preset input; no supervision or loss |
| Oscillators and voices ❌ 0 / 88 | fixed source except the listed field | 0 | fixed original preset input; no supervision or loss |
| Filters ❌ 0 / 53 | fixed source except the listed field | 0 | fixed original preset input; no supervision or loss |
| Envelopes ❌ 0 / 54 | fixed source except the listed field | 0 | fixed original preset input; no supervision or loss |
| LFOs and modulation ✅ 1 / 170 | fixed source except the listed field | 0 | manual perturbation; no model output; no supervision or loss |
| Effects ❌ 0 / 46 | fixed source except the listed field | 0 | fixed original preset input; no supervision or loss |
| Wavetables, curves and samples ❌ 0 / 6 | fixed source except the listed field | 0 | fixed original preset input; no supervision or loss |

## Baselines

| Condition | MIDI-only R@1 / R@5 / MRR | Family bootstrap 95% interval for R@1 | Random R@1 / R@5 / MRR |
|---|---:|---:|---:|
| Common gallery MIDI | not applicable: query MIDI source states are outside the validation preset gallery | not applicable | 0.0078 / 0.0388 / 0.0422 |
| Mixed gallery MIDI | not applicable: query MIDI source states are outside the validation preset gallery | not applicable | 0.0078 / 0.0388 / 0.0422 |

## Execution status

The frozen `validation` preset gallery contains 131 verified original presets and is paired with 50 compatible MIDI excerpts from `train`; the plan covers 13100 retrieval episodes and 6550 preset/MIDI renders.
Render profile: `vital-1.6.4-vst3-sha256-a622a2c99b4066cd7945a4ab9bbdd698e7632a30702f6f0a7ccbf26a56b576e1-dawdreamer-0.8.3-full-1.5.5-state-verified-seed-bits-v2-expression-midi-v1-curve-input-v1-corpus-readback-v1`. Selected renderer capabilities: `{"channel_pressure": true, "control_changes": [1, 64, 74], "max_channels": 16, "note_off_velocity": true, "notes": true, "pitch_bend": true, "polyphony": true, "tempo_changes": false}`.
Source pairing and MIDI compatibility audit: {"diversity_sampled_out_performances": 46, "duplicate_midi_performance_aliases_collapsed": 3, "duplicate_performance_source_state_aliases_collapsed": 2, "eligible_preset_records": 131, "midi_descriptive_meta_ignored:key_signature": 2, "midi_descriptive_meta_ignored:marker": 1, "midi_descriptive_meta_ignored:text": 20, "midi_descriptive_meta_ignored:track_name": 104, "midi_excluded:IndexError:list index out of range": 6, "midi_excluded:midi.capability.control_change": 19, "midi_excluded:midi.capability.opaque": 19, "midi_excluded:midi.capability.tempo_changes": 1, "midi_excluded:midi.note.unmatched_off": 2, "midi_excluded:midi.note.unmatched_on": 2, "midi_pair_candidates": 138, "midi_parsed": 99, "pair_excluded:pair.ambiguous": 6, "pair_excluded:pair.missing_midi": 760, "performance_midi_identity_overlap_excluded": 0, "performance_source_eligible_preset_records": 906}. These counts reflect the selected existing split and direct PresetShare preset/MIDI pairing rules; source WAV quality is not an eligibility condition.
The 50 compatible MIDI excerpts from the permitted `train` performance-source split are cross-rendered on all 131 admitted `validation` preset states. These are new Vital renders, not claims of original PresetShare preset/MIDI capture pairs.
The existing source audit counted 9,620 files and 1,153 statically eligible states (131 validation, 904 train, 118 test). The candidate gallery is validation-only; MIDI performance sources come from `train`. Static eligibility is not native render eligibility.
Vital render stage: 6480 QA accepted of 6550 planned requests; this run rendered 303 and reused 6197 cached WAVs in 1109.493 seconds.
QA policy `finite_silent_reject_clipping_tail_warn_v2` rejects malformed, silent, or failed renders and records clipping/tail thresholds as warnings. Accepted warning counts: {"amplitude_above_unit_range": 1531, "tail_rms_above_warning": 1579}; excluded renders: {"render_exception:RuntimeError": 50, "render_qa_silence": 20}.
The comparison uses 12900 identical episodes across the available arms; held-out reserved sources used: **no**.
The gallery retained 129 of 131 states with complete renders across the frozen gallery-MIDI pool; 2 incomplete states were removed once for all arms.

### Render and feature coverage

| Condition | Planned episodes | Render QA qualified | Excluded | Feature qualified by arm | Common across arms |
|---|---:|---:|---:|---|---:|
| cross_midi_common_gallery | 6550 | 6450 | 100 | clap_music:official_clip: 6450, muq_mulan:official_clip: 6450 | 6450 |
| cross_midi_mixed_gallery | 6550 | 6450 | 100 | clap_music:official_clip: 6450, muq_mulan:official_clip: 6450 | 6450 |
No known multi-state lineage/alias neighbors were present in the render-qualified gallery, so that diagnostic has zero negatives. The corpus snapshot also lacks instrument-category labels; the spectral hard-negative result is only a coarse audio-similarity proxy, not a same-category or human-verified perceptual-neighbor result.
Controlled perturbation diagnostic: 400 real Vital renders, 400 QA accepted, 4 pretrained encoders; its MIDI-cluster intervals are exploratory.

Runtime preflight: Python `3.12.13`; missing encoder runtime packages: none. Checkpoint root supplied: **yes**. Vital plugin environment variable configured: **no**.
The existing [corpus audit report](../../../training/reports/CORPUS_INPUT_RESULTS.md) records prior static eligibility findings. Those audit counts are not native render eligibility. The separate [PresetShare audit](../../basic_pitch/README.md) has its own pairing rules and counts.

The default command selects the corpus-qualified full-preset expression renderer identity, including the established curve-input, ignored-empty-map, and native scalar readback policies. It supports note/polyphony, pitch-bend, channel-pressure, CC 1/64/74, release-velocity and multiple channels; tempo changes remain unsupported and are counted as exclusions. Unsupported events are rejected rather than silently removed. Musical spans are complete-note excerpts of 2–8 seconds plus a fixed 2-second release tail; model input is mono and 4–10 seconds, downmixed by channel mean and polyphase-resampled. No shared gain normalization is applied; EnCodec retains its own checkpoint-configured RMS normalization when enabled.

Render and feature stages are independent, resumable commands. Vital remains a single-worker in-process renderer; an asynchronous manifest reader feeds a bounded queue of request batches. Render keys include exact preset/MIDI/excerpt/renderer/policy identities. Feature keys include audio hash, checkpoint digest/revision, loader revision, sample rate, preprocessing, extraction tap, pooling, and FP32 precision. Weights are never downloaded by the run commands. Full paths, identifiers, WAVs, caches, checkpoints, and per-episode results belong in ignored local outputs.

No reconstruction head, source-label warm start, shared-model training, direct patch optimization, or audio-loss training is part of this experiment. Retrieval results do not establish reconstruction quality.

### LAION-CLAP Music HTSAT-base — official_clip

Checkpoint: `lukewys/laion_clap/music_audioset_epoch_15_esc_90.14.pt` ([upstream checkpoint](https://huggingface.co/lukewys/laion_clap)); requested sample rate: 48000 Hz; representation: official audio clip embedding; output size: 512-D. Status: **executed**.

| Measure | Result or coverage | Output count | Update / supervision path |
|---|---:|---:|---|
| Checkpoint provenance / license | SHA-256 `fae3e9c087f2909c28a09dc31c8dfcdacbc42ba44c70e972b58c1bd1caf6dedd`; source `b3708341862f581175dba5c356a4ebf74a9b6651`; loader `laion_clap==1.1.7`; runtime `{"laion_clap": "1.1.7", "numpy": "1.26.4", "scipy": "1.17.1", "torch": "2.12.1+xpu", "transformers": "4.50.0"}`; CC0-1.0 | — | exact local lock; frozen inference |
| Cross-MIDI common-gallery Recall@1 | 0.5890 [0.5561, 0.6202] | — | frozen backbone; cosine retrieval |
| Cross-MIDI common-gallery Recall@5 | 0.7978 [0.7726, 0.8220] | — | frozen backbone; cosine retrieval |
| Cross-MIDI common-gallery MRR | 0.6841 [0.6563, 0.7108] | — | frozen backbone; cosine retrieval |
| Cross-MIDI family-level Recall@1 | 0.5890 [0.5561, 0.6202] | — | frozen backbone; cosine retrieval |
| Cross-MIDI mixed-gallery Recall@1 | 0.5502 [0.5180, 0.5836] | — | frozen backbone; cosine retrieval |
| Same-MIDI other-patch negative win rate | 0.4008 [0.3647, 0.4383] (6450 episodes; 825600 negatives) | — | frozen backbone; cosine retrieval |
| Acoustic hard-negative top-5 win rate | 0.5955 [0.5600, 0.6296] (6450 episodes; 32250 negatives) | — | frozen backbone; cosine retrieval |
| Known source-lineage near-neighbor win rate | not run (0 episodes; 0 negatives) | — | frozen backbone; cosine retrieval |
| Gallery candidates / episodes | [129] / 6450 | — | fixed candidate sets across encoders |
| Features created / cache reused | 304 / 6176 | 6480 | resumable extraction pass; cache identity checked |
| Extraction pass time / peak RSS / accelerator memory | 388.439 s / 3470766080 / None bytes | — | local inference and cache validation; no training |
| Resumable pass scope | 304 new and 6176 cache-reused features | 6480 | elapsed time is this pass, not cumulative across earlier partial runs |
| Stratum: duration_gap_band = 0-1s | Recall@1 0.5045; 2-way 95% CI [0.4010, 0.6070]; Recall@5 0.7549; MRR 0.6182 | 3354 episodes | frozen encoder; crossed performance/family bootstrap |
| Stratum: duration_gap_band = 1-3s | Recall@1 0.3976; 2-way 95% CI [0.2027, 0.6080]; Recall@5 0.6279; MRR 0.5089 | 903 episodes | frozen encoder; crossed performance/family bootstrap |
| Stratum: duration_gap_band = 3+s | Recall@1 0.7971; 2-way 95% CI [0.6908, 0.8865]; Recall@5 0.9334; MRR 0.8571 | 2193 episodes | frozen encoder; crossed performance/family bootstrap |
| Stratum: gallery_render_quality = warning | Recall@1 0.5890; 2-way 95% CI [0.5026, 0.6786]; Recall@5 0.7978; MRR 0.6841 | 6450 episodes | frozen encoder; crossed performance/family bootstrap |
| Stratum: pitch_register_gap_band = 0-5 semitones | Recall@1 0.9597; 2-way 95% CI [0.9302, 0.9845]; Recall@5 1.0000; MRR 0.9787 | 645 episodes | frozen encoder; crossed performance/family bootstrap |
| Stratum: pitch_register_gap_band = 13+ semitones | Recall@1 0.4480; 2-way 95% CI [0.3600, 0.5319]; Recall@5 0.7145; MRR 0.5698 | 4386 episodes | frozen encoder; crossed performance/family bootstrap |
| Stratum: pitch_register_gap_band = 6-12 semitones | Recall@1 0.8562; 2-way 95% CI [0.7808, 0.9225]; Recall@5 0.9634; MRR 0.9035 | 1419 episodes | frozen encoder; crossed performance/family bootstrap |
| Stratum: query_articulation = held | Recall@1 0.5398; 2-way 95% CI [0.4349, 0.6357]; Recall@5 0.7783; MRR 0.6477 | 3870 episodes | frozen encoder; crossed performance/family bootstrap |
| Stratum: query_articulation = phrased | Recall@1 0.6628; 2-way 95% CI [0.5240, 0.7899]; Recall@5 0.8271; MRR 0.7388 | 2580 episodes | frozen encoder; crossed performance/family bootstrap |
| Stratum: query_duration_band = 2-4s | Recall@1 0.6802; 2-way 95% CI [0.3915, 0.9612]; Recall@5 0.8547; MRR 0.7591 | 516 episodes | frozen encoder; crossed performance/family bootstrap |
| Stratum: query_duration_band = 4-6s | Recall@1 0.3776; 2-way 95% CI [0.1982, 0.5814]; Recall@5 0.6168; MRR 0.4923 | 903 episodes | frozen encoder; crossed performance/family bootstrap |
| Stratum: query_duration_band = 6-8s | Recall@1 0.6176; 2-way 95% CI [0.5176, 0.7058]; Recall@5 0.8245; MRR 0.7109 | 5031 episodes | frozen encoder; crossed performance/family bootstrap |
| Stratum: query_expression = expressive | Recall@1 0.5000; 2-way 95% CI [0.3488, 0.6512]; Recall@5 0.8411; MRR 0.6409 | 258 episodes | frozen encoder; crossed performance/family bootstrap |
| Stratum: query_expression = plain | Recall@1 0.5927; 2-way 95% CI [0.5061, 0.6810]; Recall@5 0.7960; MRR 0.6859 | 6192 episodes | frozen encoder; crossed performance/family bootstrap |
| Stratum: query_note_density_band = dense | Recall@1 0.5855; 2-way 95% CI [0.4579, 0.7044]; Recall@5 0.8124; MRR 0.6904 | 1935 episodes | frozen encoder; crossed performance/family bootstrap |
| Stratum: query_note_density_band = moderate | Recall@1 0.5891; 2-way 95% CI [0.4434, 0.7302]; Recall@5 0.7744; MRR 0.6748 | 2580 episodes | frozen encoder; crossed performance/family bootstrap |
| Stratum: query_note_density_band = sparse | Recall@1 0.5922; 2-way 95% CI [0.4517, 0.7292]; Recall@5 0.8145; MRR 0.6903 | 1935 episodes | frozen encoder; crossed performance/family bootstrap |
| Stratum: query_polyphony = monophonic | Recall@1 0.5378; 2-way 95% CI [0.4057, 0.6709]; Recall@5 0.7448; MRR 0.6348 | 3096 episodes | frozen encoder; crossed performance/family bootstrap |
| Stratum: query_polyphony = polyphonic | Recall@1 0.6363; 2-way 95% CI [0.5298, 0.7266]; Recall@5 0.8468; MRR 0.7297 | 3354 episodes | frozen encoder; crossed performance/family bootstrap |
| Stratum: query_render_quality = clean | Recall@1 0.5458; 2-way 95% CI [0.4474, 0.6375]; Recall@5 0.7642; MRR 0.6463 | 3778 episodes | frozen encoder; crossed performance/family bootstrap |
| Stratum: query_render_quality = warning | Recall@1 0.6501; 2-way 95% CI [0.5613, 0.7357]; Recall@5 0.8454; MRR 0.7376 | 2672 episodes | frozen encoder; crossed performance/family bootstrap |
| Stratum: source_preset_lineage_group_size = single-state | Recall@1 0.5890; 2-way 95% CI [0.5026, 0.6786]; Recall@5 0.7978; MRR 0.6841 | 6450 episodes | frozen encoder; crossed performance/family bootstrap |
### MuQ-MuLan — official_clip

Checkpoint: `OpenMuQ/MuQ-MuLan-large/pytorch_model.bin` ([upstream checkpoint](https://huggingface.co/OpenMuQ/MuQ-MuLan-large)); requested sample rate: 24000 Hz; representation: official audio embedding; output size: 512-D. Status: **executed**.

| Measure | Result or coverage | Output count | Update / supervision path |
|---|---:|---:|---|
| Checkpoint provenance / license | SHA-256 `fe56bc76aa587d2431c0c416d63f4c857d43ef7678ee6933fbf307b46af9d938`; artifact file SHA-256 `pytorch_model.bin` `d42ae3f7cb9b66759ee0089ddc70e2f28b130c2d8ba621457358272d32dd0444`; source `2e01c796b71dca71b45251384c04cd7b237c9020`; loader `muq==0.1.0`; runtime `{"easydict": "1.14", "einops": "0.8.2", "huggingface_hub": "0.36.0", "librosa": "1.0.0", "muq": "0.1.0", "numpy": "1.26.4", "scipy": "1.17.1", "torch": "2.12.1+xpu", "torchaudio": "2.11.0+xpu", "transformers": "4.50.0", "x_clip": "0.15.0"}`; CC BY-NC 4.0; non-commercial research use only, weights not redistributed | — | exact local lock; frozen inference |
| Offline architecture configs | audio `OpenMuQ/MuQ-large-msd-iter@0562a57814f6f8bbd9fdea0a25921a2fce1a841a` (bundle SHA-256 `a7e0d590e649147ce6af2c8bf0126f8457332054481f67ecea422b5aed28c5ce`); text `FacebookAI/xlm-roberta-base@e73636d4f797dec63c3081bb6ed5c7b0bb3f2089` (bundle SHA-256 `2e17c855a6b77ba1fad3e980612f7f6f9ca51e7ab77da583014605e53cea9bfe`) | — | local config-only initialization; no transitive pretrained weights loaded |
| Cross-MIDI common-gallery Recall@1 | 0.3017 [0.2668, 0.3363] | — | frozen backbone; cosine retrieval |
| Cross-MIDI common-gallery Recall@5 | 0.5290 [0.4896, 0.5660] | — | frozen backbone; cosine retrieval |
| Cross-MIDI common-gallery MRR | 0.4128 [0.3784, 0.4470] | — | frozen backbone; cosine retrieval |
| Cross-MIDI family-level Recall@1 | 0.3017 [0.2668, 0.3363] | — | frozen backbone; cosine retrieval |
| Cross-MIDI mixed-gallery Recall@1 | 0.2619 [0.2316, 0.2938] | — | frozen backbone; cosine retrieval |
| Same-MIDI other-patch negative win rate | 0.1042 [0.0767, 0.1372] (6450 episodes; 825600 negatives) | — | frozen backbone; cosine retrieval |
| Acoustic hard-negative top-5 win rate | 0.2825 [0.2439, 0.3242] (6450 episodes; 32250 negatives) | — | frozen backbone; cosine retrieval |
| Known source-lineage near-neighbor win rate | not run (0 episodes; 0 negatives) | — | frozen backbone; cosine retrieval |
| Gallery candidates / episodes | [129] / 6450 | — | fixed candidate sets across encoders |
| Features created / cache reused | 304 / 6176 | 6480 | resumable extraction pass; cache identity checked |
| Extraction pass time / peak RSS / accelerator memory | 386.212 s / 7372480512 / 2873438208 bytes | — | local inference and cache validation; no training |
| Resumable pass scope | 304 new and 6176 cache-reused features | 6480 | elapsed time is this pass, not cumulative across earlier partial runs |
| Stratum: duration_gap_band = 0-1s | Recall@1 0.2606; 2-way 95% CI [0.1995, 0.3256]; Recall@5 0.4848; MRR 0.3714 | 3354 episodes | frozen encoder; crossed performance/family bootstrap |
| Stratum: duration_gap_band = 1-3s | Recall@1 0.2027; 2-way 95% CI [0.1107, 0.3156]; Recall@5 0.3953; MRR 0.3023 | 903 episodes | frozen encoder; crossed performance/family bootstrap |
| Stratum: duration_gap_band = 3+s | Recall@1 0.4054; 2-way 95% CI [0.3151, 0.4943]; Recall@5 0.6516; MRR 0.5216 | 2193 episodes | frozen encoder; crossed performance/family bootstrap |
| Stratum: gallery_render_quality = warning | Recall@1 0.3017; 2-way 95% CI [0.2467, 0.3595]; Recall@5 0.5290; MRR 0.4128 | 6450 episodes | frozen encoder; crossed performance/family bootstrap |
| Stratum: pitch_register_gap_band = 0-5 semitones | Recall@1 0.5194; 2-way 95% CI [0.4403, 0.5907]; Recall@5 0.7907; MRR 0.6404 | 645 episodes | frozen encoder; crossed performance/family bootstrap |
| Stratum: pitch_register_gap_band = 13+ semitones | Recall@1 0.2166; 2-way 95% CI [0.1683, 0.2636]; Recall@5 0.4261; MRR 0.3232 | 4386 episodes | frozen encoder; crossed performance/family bootstrap |
| Stratum: pitch_register_gap_band = 6-12 semitones | Recall@1 0.4658; 2-way 95% CI [0.3805, 0.5560]; Recall@5 0.7280; MRR 0.5864 | 1419 episodes | frozen encoder; crossed performance/family bootstrap |
| Stratum: query_articulation = held | Recall@1 0.2760; 2-way 95% CI [0.2165, 0.3424]; Recall@5 0.5018; MRR 0.3875 | 3870 episodes | frozen encoder; crossed performance/family bootstrap |
| Stratum: query_articulation = phrased | Recall@1 0.3403; 2-way 95% CI [0.2504, 0.4236]; Recall@5 0.5698; MRR 0.4508 | 2580 episodes | frozen encoder; crossed performance/family bootstrap |
| Stratum: query_duration_band = 2-4s | Recall@1 0.4128; 2-way 95% CI [0.1667, 0.6783]; Recall@5 0.6279; MRR 0.5163 | 516 episodes | frozen encoder; crossed performance/family bootstrap |
| Stratum: query_duration_band = 4-6s | Recall@1 0.1883; 2-way 95% CI [0.1030, 0.3001]; Recall@5 0.3776; MRR 0.2878 | 903 episodes | frozen encoder; crossed performance/family bootstrap |
| Stratum: query_duration_band = 6-8s | Recall@1 0.3107; 2-way 95% CI [0.2524, 0.3685]; Recall@5 0.5460; MRR 0.4247 | 5031 episodes | frozen encoder; crossed performance/family bootstrap |
| Stratum: query_expression = expressive | Recall@1 0.1667; 2-way 95% CI [0.0853, 0.2636]; Recall@5 0.3721; MRR 0.2732 | 258 episodes | frozen encoder; crossed performance/family bootstrap |
| Stratum: query_expression = plain | Recall@1 0.3073; 2-way 95% CI [0.2506, 0.3660]; Recall@5 0.5355; MRR 0.4187 | 6192 episodes | frozen encoder; crossed performance/family bootstrap |
| Stratum: query_note_density_band = dense | Recall@1 0.2941; 2-way 95% CI [0.2274, 0.3654]; Recall@5 0.5359; MRR 0.4097 | 1935 episodes | frozen encoder; crossed performance/family bootstrap |
| Stratum: query_note_density_band = moderate | Recall@1 0.3058; 2-way 95% CI [0.2186, 0.3946]; Recall@5 0.5225; MRR 0.4135 | 2580 episodes | frozen encoder; crossed performance/family bootstrap |
| Stratum: query_note_density_band = sparse | Recall@1 0.3039; 2-way 95% CI [0.2119, 0.3995]; Recall@5 0.5307; MRR 0.4151 | 1935 episodes | frozen encoder; crossed performance/family bootstrap |
| Stratum: query_polyphony = monophonic | Recall@1 0.2952; 2-way 95% CI [0.2209, 0.3795]; Recall@5 0.5165; MRR 0.4035 | 3096 episodes | frozen encoder; crossed performance/family bootstrap |
| Stratum: query_polyphony = polyphonic | Recall@1 0.3077; 2-way 95% CI [0.2403, 0.3727]; Recall@5 0.5405; MRR 0.4215 | 3354 episodes | frozen encoder; crossed performance/family bootstrap |
| Stratum: query_render_quality = clean | Recall@1 0.2599; 2-way 95% CI [0.2011, 0.3227]; Recall@5 0.4823; MRR 0.3700 | 3778 episodes | frozen encoder; crossed performance/family bootstrap |
| Stratum: query_render_quality = warning | Recall@1 0.3608; 2-way 95% CI [0.2884, 0.4386]; Recall@5 0.5951; MRR 0.4734 | 2672 episodes | frozen encoder; crossed performance/family bootstrap |
| Stratum: source_preset_lineage_group_size = single-state | Recall@1 0.3017; 2-way 95% CI [0.2467, 0.3595]; Recall@5 0.5290; MRR 0.4128 | 6450 episodes | frozen encoder; crossed performance/family bootstrap |
### CIR single-timbre AST — official_clip

Checkpoint: `Google Drive: ICASSP26_inst-direct-with-data-aug_final.pt` ([upstream checkpoint](https://github.com/gwendal-lv/CIR)); requested sample rate: 16000 Hz; representation: mean of final CLS and distillation tokens; output size: 768-D. Status: **unavailable / not run**.

| Measure | Result or coverage | Output count | Update / supervision path |
|---|---:|---:|---|
| Cross-MIDI common-gallery Recall@1 | not run | — | frozen backbone; cosine retrieval |
| Cross-MIDI common-gallery Recall@5 | not run | — | frozen backbone; cosine retrieval |
| Cross-MIDI common-gallery MRR | not run | — | frozen backbone; cosine retrieval |
| Cross-MIDI family-level Recall@1 | not run | — | frozen backbone; cosine retrieval |
| Cross-MIDI mixed-gallery Recall@1 | not run | — | frozen backbone; cosine retrieval |
| Same-MIDI other-patch negative win rate | not run | — | frozen backbone; cosine retrieval |
| Acoustic hard-negative top-5 win rate | not run | — | frozen backbone; cosine retrieval |
| Known source-lineage near-neighbor win rate | not run | — | frozen backbone; cosine retrieval |
| Gallery candidates / episodes | not run / not run | — | fixed candidate sets across encoders |
| Extraction pass time / peak RSS / accelerator memory | not run s / not measured / not measured bytes | — | local inference and cache validation; no training |

Unavailable: no verified local checkpoint lock entry. Checkpoint/license: checkpoint license not specified in the upstream repository.

### BEATs iter3+ AS2M — masked_mean

Checkpoint: `lpepino/beats_ckpts/BEATs_iter3_plus_AS2M.pt` ([upstream checkpoint](https://huggingface.co/lpepino/beats_ckpts)); requested sample rate: 16000 Hz; representation: pretrained iter3+ final pre-predictor features; output size: 768-D. Status: **unavailable / not run**.

| Measure | Result or coverage | Output count | Update / supervision path |
|---|---:|---:|---|
| Cross-MIDI common-gallery Recall@1 | not run | — | frozen backbone; cosine retrieval |
| Cross-MIDI common-gallery Recall@5 | not run | — | frozen backbone; cosine retrieval |
| Cross-MIDI common-gallery MRR | not run | — | frozen backbone; cosine retrieval |
| Cross-MIDI family-level Recall@1 | not run | — | frozen backbone; cosine retrieval |
| Cross-MIDI mixed-gallery Recall@1 | not run | — | frozen backbone; cosine retrieval |
| Same-MIDI other-patch negative win rate | not run | — | frozen backbone; cosine retrieval |
| Acoustic hard-negative top-5 win rate | not run | — | frozen backbone; cosine retrieval |
| Known source-lineage near-neighbor win rate | not run | — | frozen backbone; cosine retrieval |
| Gallery candidates / episodes | not run / not run | — | fixed candidate sets across encoders |
| Extraction pass time / peak RSS / accelerator memory | not run s / not measured / not measured bytes | — | local inference and cache validation; no training |

Unavailable: no verified local checkpoint lock entry. Checkpoint/license: mirror labels MIT; upstream checkpoint license remains unconfirmed.

### BEATs iter3+ AS2M — temporal_pyramid_8

Checkpoint: `lpepino/beats_ckpts/BEATs_iter3_plus_AS2M.pt` ([upstream checkpoint](https://huggingface.co/lpepino/beats_ckpts)); requested sample rate: 16000 Hz; representation: pretrained iter3+ final pre-predictor features; output size: 12288-D. Status: **unavailable / not run**.

| Measure | Result or coverage | Output count | Update / supervision path |
|---|---:|---:|---|
| Cross-MIDI common-gallery Recall@1 | not run | — | frozen backbone; cosine retrieval |
| Cross-MIDI common-gallery Recall@5 | not run | — | frozen backbone; cosine retrieval |
| Cross-MIDI common-gallery MRR | not run | — | frozen backbone; cosine retrieval |
| Cross-MIDI family-level Recall@1 | not run | — | frozen backbone; cosine retrieval |
| Cross-MIDI mixed-gallery Recall@1 | not run | — | frozen backbone; cosine retrieval |
| Same-MIDI other-patch negative win rate | not run | — | frozen backbone; cosine retrieval |
| Acoustic hard-negative top-5 win rate | not run | — | frozen backbone; cosine retrieval |
| Known source-lineage near-neighbor win rate | not run | — | frozen backbone; cosine retrieval |
| Gallery candidates / episodes | not run / not run | — | fixed candidate sets across encoders |
| Extraction pass time / peak RSS / accelerator memory | not run s / not measured / not measured bytes | — | local inference and cache validation; no training |

Unavailable: no verified local checkpoint lock entry. Checkpoint/license: mirror labels MIT; upstream checkpoint license remains unconfirmed.

### Meta EnCodec 24 kHz — masked_mean

Checkpoint: `facebook/encodec_24khz` ([upstream checkpoint](https://huggingface.co/facebook/encodec_24khz)); requested sample rate: 24000 Hz; representation: continuous encoder output before RVQ; output size: 128-D. Status: **unavailable / not run**.

| Measure | Result or coverage | Output count | Update / supervision path |
|---|---:|---:|---|
| Checkpoint provenance / license | SHA-256 `304c7656c7f978fa9d6838fbc7312108d2345ba6b1b816c8599653da8a71bdb3`; source `c5e61996c75613cfd18f4c53cc06667932675776`; loader `transformers==4.50.0`; runtime `{"numpy": "1.26.4", "scipy": "1.17.1", "torch": "2.12.1+xpu", "transformers": "4.50.0"}`; MIT | — | exact local lock; frozen inference |
| Cross-MIDI common-gallery Recall@1 | not run | — | frozen backbone; cosine retrieval |
| Cross-MIDI common-gallery Recall@5 | not run | — | frozen backbone; cosine retrieval |
| Cross-MIDI common-gallery MRR | not run | — | frozen backbone; cosine retrieval |
| Cross-MIDI family-level Recall@1 | not run | — | frozen backbone; cosine retrieval |
| Cross-MIDI mixed-gallery Recall@1 | not run | — | frozen backbone; cosine retrieval |
| Same-MIDI other-patch negative win rate | not run | — | frozen backbone; cosine retrieval |
| Acoustic hard-negative top-5 win rate | not run | — | frozen backbone; cosine retrieval |
| Known source-lineage near-neighbor win rate | not run | — | frozen backbone; cosine retrieval |
| Gallery candidates / episodes | not run / not run | — | fixed candidate sets across encoders |
| Features created / cache reused | 4316 / 0 | 4316 | resumable extraction pass; cache identity checked |
| Extraction pass time / peak RSS / accelerator memory | 2561.054 s / 2304942080 / 253531648 bytes | — | local inference and cache validation; no training |
### Meta EnCodec 24 kHz — temporal_pyramid_8

Checkpoint: `facebook/encodec_24khz` ([upstream checkpoint](https://huggingface.co/facebook/encodec_24khz)); requested sample rate: 24000 Hz; representation: continuous encoder output before RVQ; output size: 2048-D. Status: **unavailable / not run**.

| Measure | Result or coverage | Output count | Update / supervision path |
|---|---:|---:|---|
| Checkpoint provenance / license | SHA-256 `304c7656c7f978fa9d6838fbc7312108d2345ba6b1b816c8599653da8a71bdb3`; source `c5e61996c75613cfd18f4c53cc06667932675776`; loader `transformers==4.50.0`; runtime `{"numpy": "1.26.4", "scipy": "1.17.1", "torch": "2.12.1+xpu", "transformers": "4.50.0"}`; MIT | — | exact local lock; frozen inference |
| Cross-MIDI common-gallery Recall@1 | not run | — | frozen backbone; cosine retrieval |
| Cross-MIDI common-gallery Recall@5 | not run | — | frozen backbone; cosine retrieval |
| Cross-MIDI common-gallery MRR | not run | — | frozen backbone; cosine retrieval |
| Cross-MIDI family-level Recall@1 | not run | — | frozen backbone; cosine retrieval |
| Cross-MIDI mixed-gallery Recall@1 | not run | — | frozen backbone; cosine retrieval |
| Same-MIDI other-patch negative win rate | not run | — | frozen backbone; cosine retrieval |
| Acoustic hard-negative top-5 win rate | not run | — | frozen backbone; cosine retrieval |
| Known source-lineage near-neighbor win rate | not run | — | frozen backbone; cosine retrieval |
| Gallery candidates / episodes | not run / not run | — | fixed candidate sets across encoders |
| Features created / cache reused | 4316 / 0 | 4316 | resumable extraction pass; cache identity checked |
| Extraction pass time / peak RSS / accelerator memory | 2561.054 s / 2304942080 / 253531648 bytes | — | local inference and cache validation; no training |
### Descript Audio Codec 44.1 kHz — masked_mean

Checkpoint: `descript/dac_44khz` ([upstream checkpoint](https://huggingface.co/descript/dac_44khz)); requested sample rate: 44100 Hz; representation: continuous encoder output before RVQ; output size: 1024-D. Status: **unavailable / not run**.

| Measure | Result or coverage | Output count | Update / supervision path |
|---|---:|---:|---|
| Cross-MIDI common-gallery Recall@1 | not run | — | frozen backbone; cosine retrieval |
| Cross-MIDI common-gallery Recall@5 | not run | — | frozen backbone; cosine retrieval |
| Cross-MIDI common-gallery MRR | not run | — | frozen backbone; cosine retrieval |
| Cross-MIDI family-level Recall@1 | not run | — | frozen backbone; cosine retrieval |
| Cross-MIDI mixed-gallery Recall@1 | not run | — | frozen backbone; cosine retrieval |
| Same-MIDI other-patch negative win rate | not run | — | frozen backbone; cosine retrieval |
| Acoustic hard-negative top-5 win rate | not run | — | frozen backbone; cosine retrieval |
| Known source-lineage near-neighbor win rate | not run | — | frozen backbone; cosine retrieval |
| Gallery candidates / episodes | not run / not run | — | fixed candidate sets across encoders |
| Extraction pass time / peak RSS / accelerator memory | not run s / not measured / not measured bytes | — | local inference and cache validation; no training |

Checkpoint/license: MIT.

### Descript Audio Codec 44.1 kHz — temporal_pyramid_8

Checkpoint: `descript/dac_44khz` ([upstream checkpoint](https://huggingface.co/descript/dac_44khz)); requested sample rate: 44100 Hz; representation: continuous encoder output before RVQ; output size: 16384-D. Status: **unavailable / not run**.

| Measure | Result or coverage | Output count | Update / supervision path |
|---|---:|---:|---|
| Cross-MIDI common-gallery Recall@1 | not run | — | frozen backbone; cosine retrieval |
| Cross-MIDI common-gallery Recall@5 | not run | — | frozen backbone; cosine retrieval |
| Cross-MIDI common-gallery MRR | not run | — | frozen backbone; cosine retrieval |
| Cross-MIDI family-level Recall@1 | not run | — | frozen backbone; cosine retrieval |
| Cross-MIDI mixed-gallery Recall@1 | not run | — | frozen backbone; cosine retrieval |
| Same-MIDI other-patch negative win rate | not run | — | frozen backbone; cosine retrieval |
| Acoustic hard-negative top-5 win rate | not run | — | frozen backbone; cosine retrieval |
| Known source-lineage near-neighbor win rate | not run | — | frozen backbone; cosine retrieval |
| Gallery candidates / episodes | not run / not run | — | fixed candidate sets across encoders |
| Extraction pass time / peak RSS / accelerator memory | not run s / not measured / not measured bytes | — | local inference and cache validation; no training |

Checkpoint/license: MIT.

## Evaluation protocol

The primary condition uses one common, existing gallery MIDI per episode and a query render of the same target preset on another existing MIDI. Query/gallery MIDI IDs differ; selection rotates deterministically. Every admitted `validation` preset is paired with all 50 compatible MIDI performances from `train`, making multiple cross-renders per preset. Gallery candidates are all admitted source states. The mixed-gallery condition uses a deterministic candidate-specific MIDI assignment and is reported separately. A same-MIDI different-patch negative check compares each cross-MIDI positive against other patches rendered on the query MIDI. Known frozen lineage groups are reported as family-level retrieval; confidence intervals resample family, performance identity, and crossed family/performance units.

Metrics use cosine similarity over FP32 cached vectors; candidate sets and query episodes are paired across available arms. Acoustic hard negatives are selected independently of the tested encoders using cosine similarity over normalized 64-bin log-frequency power spectra of the target query and same-MIDI candidate renders; each encoder’s cross-MIDI positive is then compared against the five nearest such negatives on that query MIDI. The report includes exact-state and family-level Recall@1, Recall@5, MRR, family-cluster 95% bootstrap intervals, MIDI-only metadata and random baselines, and strata for polyphony, articulation, expression, source-lineage group size, pitch gap, duration, and note density. Sequence outputs include a valid-frame mean and an 8-bin ordered mean/std pyramid. No pooling or similarity setting is selected on reserved test labels.

This benchmark uses new Vital renders only; source WAVs are not used as model inputs or mixed with renders.
