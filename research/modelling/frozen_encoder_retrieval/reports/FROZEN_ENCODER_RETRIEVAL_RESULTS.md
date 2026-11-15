# Frozen audio encoder retrieval benchmark

Issue: [#152](https://github.com/noisedoctrine/obruxo/issues/152). This report covers frozen pretrained representations on original Vital source presets and existing MIDI performances. No synthesis controls are predicted, tuned, or optimized.

## Results and recommendation

**Finding:** CIR single-timbre AST (official_clip) reached 81.4% cross-MIDI Recall@1 (n=6,450 queries, 129 candidate presets, 50 held-out MIDI performances). This ranks frozen representations across six model families; it is retrieval evidence, not a reconstruction result.

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

R@1 intervals are shown in preset-family, performance-ID, and crossed performance × preset-family order (2,000 bootstrap replicates); the heading gives the independent performance denominator. Negative-win cells show query denominators and preset-family/performance-cluster intervals. Clean/warning strata use query-render QA status; differences are descriptive because warning status can covary with preset acoustics. The full report also retains gallery quality strata.

**Query-render QA strata:** clean: `clap_music:official_clip` 66.0% [52.9%, 79.7%] (q=941; 12 MIDI IDs); warning: `clap_music:official_clip` 78.1% [65.9%, 88.2%] (q=607; 12 MIDI IDs). These are per-stratum MIDI-cluster intervals; accepted warning status is associated with the rendered preset and does not imply a causal quality effect.

**Per-performance ranking stability** (best observed pooling within each family and condition; bootstrap resamples 12 MIDI IDs):

| Encoder family | Common-gallery arm; R@1; first-place performances; bootstrap top-rank probability | Mixed-gallery arm; R@1; first-place performances; bootstrap top-rank probability |
|---|---|---|
| LAION-CLAP Music HTSAT-base | `clap_music:official_clip`; 70.7%; 12/12; 100.0% | `clap_music:official_clip`; 64.8%; 12/12; 100.0% |
| Descript Audio Codec 44.1 kHz | `dac_44khz:temporal_pyramid_8`; 36.6%; 0/12; 0.0% | `dac_44khz:temporal_pyramid_8`; 38.8%; 0/12; 0.0% |
| Meta EnCodec 24 kHz | `encodec_24khz:temporal_pyramid_8`; 29.8%; 0/12; 0.0% | `encodec_24khz:temporal_pyramid_8`; 29.6%; 0/12; 0.0% |
| MuQ-MuLan | `muq_mulan:official_clip`; 36.5%; 0/12; 0.0% | `muq_mulan:official_clip`; 31.1%; 0/12; 0.0% |

Pooling is selected descriptively within each evaluated condition; the primary common-gallery table above lists every arm. Top-rank probabilities compare all scored arms within the condition.

### Expanded held-out performance cohort: 50 selected MIDI IDs; 50 with evaluated queries; 6450 common-gallery episodes
Validation preset gallery: 131 verified states; performances sourced from `train`. Reserved test and confirmation sources used: **no**.
Performance selection: deterministic coverage-balanced sample of 50 from 96 compatible IDs; 46 sampled out and 0 exact-ID overlaps excluded.

Selected-excerpt diversity: 24 monophonic / 26 polyphonic; 30 held, 20 phrased; median pitch 29–89; variable velocity in 25/50; expressive events in 2 sustain, 0 pitch bend, and 0 other CC/pressure excerpts.

| Frozen representation (output) | R@1 (%; preset-family / MIDI / crossed 95% CI) | R@5 (%) | MRR (%) | Same-MIDI negative win (%; preset-family / MIDI CI; queries) | Acoustic top-5 negative win (%; preset-family / MIDI CI; queries) | Query R@1 clean / warning (%) |
|---|---:|---:|---:|---:|---:|---:|
| CIR single-timbre AST — official_clip (768-D) | 81.4% preset-family [78.9%, 83.9%] / MIDI [76.0%, 86.7%] / 2-way [75.3%, 87.1%] | 93.5% | 86.8% | 69.4% family [66.4%, 72.4%] / MIDI [61.5%, 77.0%] (q=6450) | 84.5% family [81.7%, 87.0%] / MIDI [79.8%, 89.0%] (q=6450) | 79.4% [73.8%, 85.2%] (q=3778; MIDI IDs=50) / 84.2% [79.7%, 89.2%] (q=2672; MIDI IDs=50) |
| BEATs iter3+ AS2M — temporal_pyramid_8 (12288-D) | 62.2% preset-family [58.9%, 65.3%] / MIDI [54.4%, 69.7%] / 2-way [53.7%, 70.9%] | 81.3% | 70.9% | 26.1% family [22.1%, 30.3%] / MIDI [21.0%, 32.0%] (q=6450) | 47.2% family [42.9%, 51.5%] / MIDI [40.9%, 53.6%] (q=6450) | 57.1% [48.7%, 65.5%] (q=3778; MIDI IDs=50) / 69.3% [63.4%, 76.8%] (q=2672; MIDI IDs=50) |
| BEATs iter3+ AS2M — masked_mean (768-D) | 59.3% preset-family [56.1%, 62.3%] / MIDI [50.7%, 67.5%] / 2-way [50.3%, 68.8%] | 78.6% | 68.2% | 41.3% family [37.9%, 44.7%] / MIDI [32.4%, 50.3%] (q=6450) | 57.4% family [53.8%, 60.9%] / MIDI [49.1%, 65.6%] (q=6450) | 54.1% [44.9%, 63.4%] (q=3778; MIDI IDs=50) / 66.5% [59.6%, 74.4%] (q=2672; MIDI IDs=50) |
| LAION-CLAP Music HTSAT-base — official_clip (512-D) | 58.9% preset-family [55.6%, 62.0%] / MIDI [50.9%, 66.6%] / 2-way [50.3%, 67.9%] | 79.8% | 68.4% | 40.1% family [36.5%, 43.8%] / MIDI [32.2%, 47.9%] (q=6450) | 59.6% family [56.0%, 63.0%] / MIDI [51.5%, 67.0%] (q=6450) | 54.6% [46.4%, 62.9%] (q=3778; MIDI IDs=50) / 65.0% [57.9%, 73.1%] (q=2672; MIDI IDs=50) |
| Descript Audio Codec 44.1 kHz — masked_mean (1024-D) | 31.4% preset-family [27.5%, 35.4%] / MIDI [25.7%, 37.4%] / 2-way [24.9%, 38.7%] | 52.2% | 41.8% | 28.9% family [25.2%, 32.6%] / MIDI [23.3%, 34.9%] (q=6450) | 48.1% family [43.9%, 51.9%] / MIDI [41.9%, 54.1%] (q=6450) | 27.5% [21.0%, 33.0%] (q=3778; MIDI IDs=50) / 36.9% [32.5%, 44.0%] (q=2672; MIDI IDs=50) |
| Descript Audio Codec 44.1 kHz — temporal_pyramid_8 (16384-D) | 30.7% preset-family [26.6%, 34.8%] / MIDI [26.2%, 35.6%] / 2-way [24.3%, 37.1%] | 52.9% | 41.8% | 17.6% family [13.9%, 21.2%] / MIDI [14.6%, 21.2%] (q=6450) | 36.5% family [32.4%, 40.4%] / MIDI [32.0%, 41.3%] (q=6450) | 24.2% [19.1%, 28.5%] (q=3778; MIDI IDs=50) / 39.8% [36.0%, 46.9%] (q=2672; MIDI IDs=50) |
| MuQ-MuLan — official_clip (512-D) | 30.2% preset-family [26.7%, 33.6%] / MIDI [25.7%, 34.6%] / 2-way [24.7%, 36.0%] | 52.9% | 41.3% | 10.4% family [7.7%, 13.7%] / MIDI [8.4%, 12.5%] (q=6450) | 28.2% family [24.4%, 32.4%] / MIDI [24.4%, 32.2%] (q=6450) | 26.0% [21.6%, 30.2%] (q=3778; MIDI IDs=50) / 36.1% [32.3%, 42.3%] (q=2672; MIDI IDs=50) |
| Meta EnCodec 24 kHz — masked_mean (128-D) | 24.2% preset-family [20.2%, 28.3%] / MIDI [19.8%, 29.3%] / 2-way [18.2%, 30.5%] | 41.1% | 33.6% | 21.1% family [17.4%, 25.1%] / MIDI [16.6%, 26.4%] (q=6450) | 36.0% family [31.7%, 40.4%] / MIDI [30.3%, 41.9%] (q=6450) | 19.3% [14.5%, 24.1%] (q=3778; MIDI IDs=50) / 31.1% [27.2%, 37.5%] (q=2672; MIDI IDs=50) |
| Meta EnCodec 24 kHz — temporal_pyramid_8 (2048-D) | 23.4% preset-family [19.5%, 27.5%] / MIDI [20.0%, 27.1%] / 2-way [17.9%, 29.0%] | 43.5% | 33.9% | 12.0% family [8.8%, 15.4%] / MIDI [9.8%, 14.6%] (q=6450) | 26.3% family [22.3%, 30.6%] / MIDI [22.9%, 30.2%] (q=6450) | 16.6% [13.0%, 19.9%] (q=3778; MIDI IDs=50) / 33.0% [29.9%, 38.8%] (q=2672; MIDI IDs=50) |
| Random chance | 0.8% (fixed) | 3.9% | 4.2% | — | — | — |

Paired crossed-cluster Recall@1 differences from the leading family representation (`cir_ast:official_clip`; best pooling per family):

| Comparator | ΔR@1 (percentage points) | Crossed MIDI × preset-family 95% CI |
|---|---:|---:|
| `beats_iter3_plus:temporal_pyramid_8` | +19.2 | [+14.8, +24.1]  |
| `clap_music:official_clip` | +22.5 | [+17.8, +27.5]  |
| `dac_44khz:masked_mean` | +50.0 | [+43.7, +56.4]  |
| `muq_mulan:official_clip` | +51.2 | [+46.4, +56.2]  |
| `encodec_24khz:masked_mean` | +57.2 | [+51.2, +63.1]  |

R@1 intervals are shown in preset-family, performance-ID, and crossed performance × preset-family order (2,000 bootstrap replicates); the heading gives the independent performance denominator. Negative-win cells show query denominators and preset-family/performance-cluster intervals. Clean/warning strata use query-render QA status; differences are descriptive because warning status can covary with preset acoustics. The full report also retains gallery quality strata.

**Query-render QA strata:** clean: `cir_ast:official_clip` 79.4% [73.8%, 85.2%] (q=3,778; 50 MIDI IDs); warning: `cir_ast:official_clip` 84.2% [79.7%, 89.2%] (q=2,672; 50 MIDI IDs). These are per-stratum MIDI-cluster intervals; accepted warning status is associated with the rendered preset and does not imply a causal quality effect.

**Per-performance ranking stability** (best observed pooling within each family and condition; bootstrap resamples 50 MIDI IDs):

| Encoder family | Common-gallery arm; R@1; first-place performances; bootstrap top-rank probability | Mixed-gallery arm; R@1; first-place performances; bootstrap top-rank probability |
|---|---|---|
| BEATs iter3+ AS2M | `beats_iter3_plus:temporal_pyramid_8`; 62.2%; 1/50; 0.0% | `beats_iter3_plus:masked_mean`; 55.3%; 0/50; 0.0% |
| CIR single-timbre AST | `cir_ast:official_clip`; 81.4%; 47/50; 100.0% | `cir_ast:official_clip`; 80.6%; 50/50; 100.0% |
| LAION-CLAP Music HTSAT-base | `clap_music:official_clip`; 58.9%; 0/50; 0.0% | `clap_music:official_clip`; 55.0%; 0/50; 0.0% |
| Descript Audio Codec 44.1 kHz | `dac_44khz:masked_mean`; 31.4%; 0/50; 0.0% | `dac_44khz:masked_mean`; 32.9%; 0/50; 0.0% |
| Meta EnCodec 24 kHz | `encodec_24khz:masked_mean`; 24.2%; 0/50; 0.0% | `encodec_24khz:temporal_pyramid_8`; 24.6%; 0/50; 0.0% |
| MuQ-MuLan | `muq_mulan:official_clip`; 30.2%; 0/50; 0.0% | `muq_mulan:official_clip`; 26.2%; 0/50; 0.0% |

Pooling is selected descriptively within each evaluated condition; the primary common-gallery table above lists every arm. Top-rank probabilities compare all scored arms within the condition.

Observed expanded-cohort lead: `cir_ast:official_clip`. It ranked first on 47/50 individual MIDI performances and led in 100.0% of 2,000 performance-cluster resamples. Earlier 12-MIDI validation-source cohort leader: `clap_music:official_clip`; the earlier and expanded cohorts use different MIDI pools, so this is contextual rather than a paired stability comparison. Crossed-cluster ΔR@1 versus `beats_iter3_plus:temporal_pyramid_8` is +19.2 percentage points [+14.8, +24.1]. Use `cir_ast:official_clip` as the lead and `beats_iter3_plus:temporal_pyramid_8` as a comparator for the later 14-output reconstruction screen. Keep CLAP Music as an established reference in that screen.

**Largest descriptive performance strata for the expanded-cohort lead:**

- query-to-gallery pitch/register gap: lowest `13+ semitones` 73.9% [67.4%, 79.6%], 4386 queries / 34 MIDI IDs; highest `0-5 semitones` 99.8% [99.5%, 100.0%], 645 queries / 5 MIDI IDs.
- performance articulation: lowest `held` 79.1% [72.2%, 85.9%], 3870 queries / 30 MIDI IDs; highest `phrased` 84.7% [77.0%, 92.4%], 2580 queries / 20 MIDI IDs.
- query polyphony: lowest `monophonic` 76.6% [68.1%, 85.1%], 3096 queries / 24 MIDI IDs; highest `polyphonic` 85.7% [79.8%, 91.3%], 3354 queries / 26 MIDI IDs.
These are exploratory subgroup contrasts; the intervals resample MIDI IDs and do not imply a causal effect.

All six encoder families have expanded-cohort retrieval results on 6,450 identical common-gallery episodes per family.
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

**Scope:** retrieval only. No current OBRUXO CNN baseline, shared-model training, source-label warm start, direct patch optimization, audio-loss training, or reconstruction head was run.

## Controlled preset perturbations

This secondary diagnostic uses one original validation preset, three single-control perturbations, 50 MIDI performances, and source/independently randomized Vital renders for each version. It reports cross-MIDI identity retrieval, same-MIDI retrieval against a different random seed, and perturbation distance relative to the identical-preset repeat distance. Every query uses a source-seeded render and its gallery uses independent-seed renders; exact query audio is excluded from every gallery. BEATs and CIR controlled passes ran in isolated processes after a combined Windows XPU run faulted during a pandas/pyarrow import; their summaries were combined with the existing four-model results only after the same preset, MIDI IDs, controls, seeds, and 400 accepted renders were verified. MIDI-cluster intervals summarize variation for this one preset and do not generalize across presets or controls.
The expanded-cohort lead `cir_ast:official_clip` scored among the four controlled versions as filter cutoff 90.0% [82.0%, 98.0%]; amplitude attack 72.0% [60.0%, 84.0%]; routed LFO frequency 50.0% [36.0%, 64.0%] (95% MIDI-cluster intervals, 50 performances; one preset).
R@5 is a four-version ceiling and is not interpreted. The LFO arm changes one selected routed frequency field; no phase or modulation route is optimized.
MIDI excerpt properties: 50 MIDI IDs; polyphony 1–7, median pitch 29–89, clip length 3.00–8.00s, median note duration 0.04–5.03s, variable velocity 25/50, articulation 30 held / 20 phrased. The original source carried 5 inherited Vital schema diagnostics ({"vital.parameter.unknown": 5}); perturbations introduced 0 new diagnostics. Velocity ranges are recorded, but velocity response was not isolated. The independent 64-bin log-frequency audio descriptor has no human audibility threshold. The lfo_2-routed LFO frequency perturbation had audio-descriptor distance 0.000699 [95% MIDI-cluster CI 0.000478, 0.000991]; it exceeded identical-preset seed-repeat distance on 0.4 of excerpts. This tests a routed LFO rate change, not phase reconstruction or full temporal-control recovery.
The linked [vital_control_inventory_controlled.jsonl](./vital_control_inventory_controlled.jsonl) records every exact Vital control/instance for each version (979 rows per version). No model predicts or optimizes these controls; each perturbation is manually specified, with every other source value held fixed.

### Original source preset

| Measure | Result or coverage | Output count | Update method / training signal |
|---|---:|---:|---|
| Applied Vital field | source preset unchanged | 0 | manual perturbation; no training, labels, gradients, or loss |
| BEATs iter3+ AS2M — masked_mean cross-MIDI variant retrieval | R@1 0.3800 [0.24, 0.52]; R@5 1.0000; MRR 0.6467 (50 queries) | 768-D | frozen backbone; cosine retrieval |
| BEATs iter3+ AS2M — masked_mean same-MIDI independent-seed control | R@1 0.7400 [0.69, 0.79]; MRR 0.8675 | 768-D | distinct source/independent seed renders; query excluded |
| BEATs iter3+ AS2M — masked_mean identical-preset seed-repeat distance | 0.0064 [0.005461, 0.007414] | 768-D | same source preset/MIDI; varied Vital seed |
| BEATs iter3+ AS2M — masked_mean extraction time / peak RSS / accelerator memory | 10.468 s / 1768079360 / 361782784 bytes | — | local inference only; no training |
| BEATs iter3+ AS2M — temporal_pyramid_8 cross-MIDI variant retrieval | R@1 0.2800 [0.16, 0.42]; R@5 1.0000; MRR 0.5983 (50 queries) | 12288-D | frozen backbone; cosine retrieval |
| BEATs iter3+ AS2M — temporal_pyramid_8 same-MIDI independent-seed control | R@1 0.7650 [0.715, 0.81]; MRR 0.8800 | 12288-D | distinct source/independent seed renders; query excluded |
| BEATs iter3+ AS2M — temporal_pyramid_8 identical-preset seed-repeat distance | 0.0122 [0.010764, 0.0137] | 12288-D | same source preset/MIDI; varied Vital seed |
| BEATs iter3+ AS2M — temporal_pyramid_8 extraction time / peak RSS / accelerator memory | 10.468 s / 1768079360 / 361782784 bytes | — | local inference only; no training |
| CIR single-timbre AST — official_clip cross-MIDI variant retrieval | R@1 0.3600 [0.22, 0.5]; R@5 1.0000; MRR 0.6417 (50 queries) | 768-D | frozen backbone; cosine retrieval |
| CIR single-timbre AST — official_clip same-MIDI independent-seed control | R@1 0.7450 [0.7, 0.79]; MRR 0.8708 | 768-D | distinct source/independent seed renders; query excluded |
| CIR single-timbre AST — official_clip identical-preset seed-repeat distance | 0.0441 [0.040611, 0.047636] | 768-D | same source preset/MIDI; varied Vital seed |
| CIR single-timbre AST — official_clip extraction time / peak RSS / accelerator memory | 26.627 s / 2387619840 / 433472512 bytes | — | local inference only; no training |
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
| BEATs iter3+ AS2M — masked_mean cross-MIDI variant retrieval | R@1 0.7200 [0.58, 0.84]; R@5 1.0000; MRR 0.8183 (50 queries) | 768-D | frozen backbone; cosine retrieval |
| BEATs iter3+ AS2M — masked_mean same-MIDI independent-seed control | R@1 0.7400 [0.69, 0.79]; MRR 0.8675 | 768-D | distinct source/independent seed renders; query excluded |
| BEATs iter3+ AS2M — masked_mean perturbation distance / fraction exceeding repeat | 0.0723 [0.064615, 0.080357] / 1.0000 [1.0, 1.0] | 768-D | distance from original at same MIDI/seed; bootstrap by MIDI |
| BEATs iter3+ AS2M — masked_mean extraction time / peak RSS / accelerator memory | 10.468 s / 1768079360 / 361782784 bytes | — | local inference only; no training |
| BEATs iter3+ AS2M — temporal_pyramid_8 cross-MIDI variant retrieval | R@1 0.7000 [0.58, 0.82]; R@5 1.0000; MRR 0.7983 (50 queries) | 12288-D | frozen backbone; cosine retrieval |
| BEATs iter3+ AS2M — temporal_pyramid_8 same-MIDI independent-seed control | R@1 0.7650 [0.715, 0.81]; MRR 0.8800 | 12288-D | distinct source/independent seed renders; query excluded |
| BEATs iter3+ AS2M — temporal_pyramid_8 perturbation distance / fraction exceeding repeat | 0.1005 [0.092792, 0.108489] / 1.0000 [1.0, 1.0] | 12288-D | distance from original at same MIDI/seed; bootstrap by MIDI |
| BEATs iter3+ AS2M — temporal_pyramid_8 extraction time / peak RSS / accelerator memory | 10.468 s / 1768079360 / 361782784 bytes | — | local inference only; no training |
| CIR single-timbre AST — official_clip cross-MIDI variant retrieval | R@1 0.9000 [0.82, 0.98]; R@5 1.0000; MRR 0.9317 (50 queries) | 768-D | frozen backbone; cosine retrieval |
| CIR single-timbre AST — official_clip same-MIDI independent-seed control | R@1 0.7450 [0.7, 0.79]; MRR 0.8708 | 768-D | distinct source/independent seed renders; query excluded |
| CIR single-timbre AST — official_clip perturbation distance / fraction exceeding repeat | 0.3298 [0.306267, 0.35382] / 1.0000 [1.0, 1.0] | 768-D | distance from original at same MIDI/seed; bootstrap by MIDI |
| CIR single-timbre AST — official_clip extraction time / peak RSS / accelerator memory | 26.627 s / 2387619840 / 433472512 bytes | — | local inference only; no training |
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
| BEATs iter3+ AS2M — masked_mean cross-MIDI variant retrieval | R@1 0.5200 [0.38, 0.66]; R@5 1.0000; MRR 0.6933 (50 queries) | 768-D | frozen backbone; cosine retrieval |
| BEATs iter3+ AS2M — masked_mean same-MIDI independent-seed control | R@1 0.7400 [0.69, 0.79]; MRR 0.8675 | 768-D | distinct source/independent seed renders; query excluded |
| BEATs iter3+ AS2M — masked_mean perturbation distance / fraction exceeding repeat | 0.0356 [0.027836, 0.044216] / 0.9800 [0.94, 1.0] | 768-D | distance from original at same MIDI/seed; bootstrap by MIDI |
| BEATs iter3+ AS2M — masked_mean extraction time / peak RSS / accelerator memory | 10.468 s / 1768079360 / 361782784 bytes | — | local inference only; no training |
| BEATs iter3+ AS2M — temporal_pyramid_8 cross-MIDI variant retrieval | R@1 0.5800 [0.44, 0.72]; R@5 1.0000; MRR 0.7433 (50 queries) | 12288-D | frozen backbone; cosine retrieval |
| BEATs iter3+ AS2M — temporal_pyramid_8 same-MIDI independent-seed control | R@1 0.7650 [0.715, 0.81]; MRR 0.8800 | 12288-D | distinct source/independent seed renders; query excluded |
| BEATs iter3+ AS2M — temporal_pyramid_8 perturbation distance / fraction exceeding repeat | 0.0411 [0.034767, 0.047573] / 1.0000 [1.0, 1.0] | 12288-D | distance from original at same MIDI/seed; bootstrap by MIDI |
| BEATs iter3+ AS2M — temporal_pyramid_8 extraction time / peak RSS / accelerator memory | 10.468 s / 1768079360 / 361782784 bytes | — | local inference only; no training |
| CIR single-timbre AST — official_clip cross-MIDI variant retrieval | R@1 0.7200 [0.6, 0.84]; R@5 1.0000; MRR 0.8317 (50 queries) | 768-D | frozen backbone; cosine retrieval |
| CIR single-timbre AST — official_clip same-MIDI independent-seed control | R@1 0.7450 [0.7, 0.79]; MRR 0.8708 | 768-D | distinct source/independent seed renders; query excluded |
| CIR single-timbre AST — official_clip perturbation distance / fraction exceeding repeat | 0.1203 [0.108013, 0.132703] / 1.0000 [1.0, 1.0] | 768-D | distance from original at same MIDI/seed; bootstrap by MIDI |
| CIR single-timbre AST — official_clip extraction time / peak RSS / accelerator memory | 26.627 s / 2387619840 / 433472512 bytes | — | local inference only; no training |
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
| BEATs iter3+ AS2M — masked_mean cross-MIDI variant retrieval | R@1 0.3200 [0.2, 0.46]; R@5 1.0000; MRR 0.6350 (50 queries) | 768-D | frozen backbone; cosine retrieval |
| BEATs iter3+ AS2M — masked_mean same-MIDI independent-seed control | R@1 0.7400 [0.69, 0.79]; MRR 0.8675 | 768-D | distinct source/independent seed renders; query excluded |
| BEATs iter3+ AS2M — masked_mean perturbation distance / fraction exceeding repeat | 0.0066 [0.005373, 0.008098] / 0.4400 [0.32, 0.58] | 768-D | distance from original at same MIDI/seed; bootstrap by MIDI |
| BEATs iter3+ AS2M — masked_mean extraction time / peak RSS / accelerator memory | 10.468 s / 1768079360 / 361782784 bytes | — | local inference only; no training |
| BEATs iter3+ AS2M — temporal_pyramid_8 cross-MIDI variant retrieval | R@1 0.4000 [0.26, 0.54]; R@5 1.0000; MRR 0.6767 (50 queries) | 12288-D | frozen backbone; cosine retrieval |
| BEATs iter3+ AS2M — temporal_pyramid_8 same-MIDI independent-seed control | R@1 0.7650 [0.715, 0.81]; MRR 0.8800 | 12288-D | distinct source/independent seed renders; query excluded |
| BEATs iter3+ AS2M — temporal_pyramid_8 perturbation distance / fraction exceeding repeat | 0.0123 [0.010631, 0.01421] / 0.3800 [0.26, 0.52] | 12288-D | distance from original at same MIDI/seed; bootstrap by MIDI |
| BEATs iter3+ AS2M — temporal_pyramid_8 extraction time / peak RSS / accelerator memory | 10.468 s / 1768079360 / 361782784 bytes | — | local inference only; no training |
| CIR single-timbre AST — official_clip cross-MIDI variant retrieval | R@1 0.5000 [0.36, 0.64]; R@5 1.0000; MRR 0.7333 (50 queries) | 768-D | frozen backbone; cosine retrieval |
| CIR single-timbre AST — official_clip same-MIDI independent-seed control | R@1 0.7450 [0.7, 0.79]; MRR 0.8708 | 768-D | distinct source/independent seed renders; query excluded |
| CIR single-timbre AST — official_clip perturbation distance / fraction exceeding repeat | 0.0412 [0.03817, 0.044377] / 0.4400 [0.3, 0.58] | 768-D | distance from original at same MIDI/seed; bootstrap by MIDI |
| CIR single-timbre AST — official_clip extraction time / peak RSS / accelerator memory | 26.627 s / 2387619840 / 433472512 bytes | — | local inference only; no training |
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
| cross_midi_common_gallery | 6550 | 6450 | 100 | beats_iter3_plus:masked_mean: 6450, beats_iter3_plus:temporal_pyramid_8: 6450, cir_ast:official_clip: 6450, clap_music:official_clip: 6450, dac_44khz:masked_mean: 6450, dac_44khz:temporal_pyramid_8: 6450, encodec_24khz:masked_mean: 6450, encodec_24khz:temporal_pyramid_8: 6450, muq_mulan:official_clip: 6450 | 6450 |
| cross_midi_mixed_gallery | 6550 | 6450 | 100 | beats_iter3_plus:masked_mean: 6450, beats_iter3_plus:temporal_pyramid_8: 6450, cir_ast:official_clip: 6450, clap_music:official_clip: 6450, dac_44khz:masked_mean: 6450, dac_44khz:temporal_pyramid_8: 6450, encodec_24khz:masked_mean: 6450, encodec_24khz:temporal_pyramid_8: 6450, muq_mulan:official_clip: 6450 | 6450 |
No known multi-state lineage/alias neighbors were present in the render-qualified gallery, so that diagnostic has zero negatives. The corpus snapshot also lacks instrument-category labels; the spectral hard-negative result is only a coarse audio-similarity proxy, not a same-category or human-verified perceptual-neighbor result.
Controlled perturbation diagnostic: 400 real Vital renders, 400 QA accepted, 6 pretrained encoders; its MIDI-cluster intervals are exploratory.

Runtime preflight: Python `3.12.13`; missing encoder runtime packages: none. Checkpoint root supplied: **yes**. Vital plugin environment variable configured: **no**.
The existing [corpus audit report](../../../training/reports/CORPUS_INPUT_RESULTS.md) records prior static eligibility findings. Those audit counts are not native render eligibility. The separate [PresetShare audit](../../basic_pitch/README.md) has its own pairing rules and counts.

The default command selects the corpus-qualified full-preset expression renderer identity, including the established curve-input, ignored-empty-map, and native scalar readback policies. It supports note/polyphony, pitch-bend, channel-pressure, CC 1/64/74, release-velocity and multiple channels; tempo changes remain unsupported and are counted as exclusions. Unsupported events are rejected rather than silently removed. Musical spans are complete-note excerpts of 2–8 seconds plus a fixed 2-second release tail; model input is mono and 4–10 seconds, downmixed by channel mean and polyphase-resampled. No shared gain normalization is applied; EnCodec retains its own checkpoint-configured RMS normalization when enabled.

Render and feature stages are independent, resumable commands. Vital remains a single-worker in-process renderer; an asynchronous manifest reader feeds a bounded queue of request batches. Render keys include exact preset/MIDI/excerpt/renderer/policy identities. Feature keys include audio hash, checkpoint digest/revision, loader revision, sample rate, preprocessing, extraction tap, pooling, and FP32 precision. Weights are never downloaded by the run commands. Full paths, identifiers, WAVs, caches, checkpoints, and per-episode results belong in ignored local outputs.

No reconstruction head, source-label warm start, shared-model training, direct patch optimization, or audio-loss training is part of this experiment. Retrieval results do not establish reconstruction quality.

Checkpoint-term note: BEATs iter3+ AS2M, CIR single-timbre AST were evaluated under authorized local non-commercial research use; their terms remain unresolved. Checkpoint weights are kept local and are not redistributed. This records the research run only and does not establish future deployment or distribution rights.
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

Vital controls for this representation version:

| Vital control family | Enabled / exact instances | Vital output count | Update method / training signal |
|---|---:|---:|---|
| Global settings ❌ 0 / 562 | 0 | unchanged source preset; no head, optimization, loss, or gradient |
| Oscillators and voices ❌ 0 / 88 | 0 | unchanged source preset; no head, optimization, loss, or gradient |
| Filters ❌ 0 / 53 | 0 | unchanged source preset; no head, optimization, loss, or gradient |
| Envelopes ❌ 0 / 54 | 0 | unchanged source preset; no head, optimization, loss, or gradient |
| LFOs and modulation ❌ 0 / 170 | 0 | unchanged source preset; no head, optimization, loss, or gradient |
| Effects ❌ 0 / 46 | 0 | unchanged source preset; no head, optimization, loss, or gradient |
| Wavetables, curves and samples ❌ 0 / 6 | 0 | unchanged source preset; no head, optimization, loss, or gradient |
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

Vital controls for this representation version:

| Vital control family | Enabled / exact instances | Vital output count | Update method / training signal |
|---|---:|---:|---|
| Global settings ❌ 0 / 562 | 0 | unchanged source preset; no head, optimization, loss, or gradient |
| Oscillators and voices ❌ 0 / 88 | 0 | unchanged source preset; no head, optimization, loss, or gradient |
| Filters ❌ 0 / 53 | 0 | unchanged source preset; no head, optimization, loss, or gradient |
| Envelopes ❌ 0 / 54 | 0 | unchanged source preset; no head, optimization, loss, or gradient |
| LFOs and modulation ❌ 0 / 170 | 0 | unchanged source preset; no head, optimization, loss, or gradient |
| Effects ❌ 0 / 46 | 0 | unchanged source preset; no head, optimization, loss, or gradient |
| Wavetables, curves and samples ❌ 0 / 6 | 0 | unchanged source preset; no head, optimization, loss, or gradient |
### CIR single-timbre AST — official_clip

Checkpoint: `Google Drive: ICASSP26_inst-direct-with-data-aug_final.pt` ([upstream checkpoint](https://github.com/gwendal-lv/CIR)); requested sample rate: 16000 Hz; representation: mean of final CLS and distillation tokens; output size: 768-D. Status: **executed**.

| Measure | Result or coverage | Output count | Update / supervision path |
|---|---:|---:|---|
| Checkpoint provenance / license | SHA-256 `b08996e4c42511e69af7245695d735d85585a2c41e49ab1fdc67dc308c604bbb`; source `google-drive:1rbudnDYkc8-7KHm2jYk8ouyhob0zzCHI`; loader `ca551b8df4234e40e3acc84202f8b921854be89c`; runtime `{"librosa": "1.0.0", "numpy": "1.26.4", "scipy": "1.17.1", "torch": "2.12.1+xpu", "tqdm": "4.68.4", "transformers": "4.50.0"}`; Checkpoint terms unspecified. Authorized local non-commercial research only; weights are not redistributed. | — | exact local lock; frozen inference |
| Offline architecture configs | ast `MIT/ast-finetuned-audioset-12-12-0.447@820128982ca1044325d77662565c4dc77411f709` (bundle SHA-256 `b43827aafcd9a0238c505c50b1315e1ebf191d292d6dc436b39d73b0975a1016`) | — | local config-only initialization; no transitive pretrained weights loaded |
| Cross-MIDI common-gallery Recall@1 | 0.8138 [0.7887, 0.8389] | — | frozen backbone; cosine retrieval |
| Cross-MIDI common-gallery Recall@5 | 0.9350 [0.9202, 0.9484] | — | frozen backbone; cosine retrieval |
| Cross-MIDI common-gallery MRR | 0.8684 [0.8491, 0.8871] | — | frozen backbone; cosine retrieval |
| Cross-MIDI family-level Recall@1 | 0.8138 [0.7887, 0.8389] | — | frozen backbone; cosine retrieval |
| Cross-MIDI mixed-gallery Recall@1 | 0.8056 [0.7806, 0.8313] | — | frozen backbone; cosine retrieval |
| Same-MIDI other-patch negative win rate | 0.6936 [0.6637, 0.7243] (6450 episodes; 825600 negatives) | — | frozen backbone; cosine retrieval |
| Acoustic hard-negative top-5 win rate | 0.8447 [0.8174, 0.8701] (6450 episodes; 32250 negatives) | — | frozen backbone; cosine retrieval |
| Known source-lineage near-neighbor win rate | not run (0 episodes; 0 negatives) | — | frozen backbone; cosine retrieval |
| Gallery candidates / episodes | [129] / 6450 | — | fixed candidate sets across encoders |
| Features created / cache reused | 6480 / 0 | 6480 | resumable extraction pass; cache identity checked |
| Extraction pass time / peak RSS / accelerator memory | 1629.597 s / 2189545472 / 433472512 bytes | — | local inference and cache validation; no training |
| Stratum: duration_gap_band = 0-1s | Recall@1 0.7639; 2-way 95% CI [0.6846, 0.8411]; Recall@5 0.9159; MRR 0.8324 | 3354 episodes | frozen encoder; crossed performance/family bootstrap |
| Stratum: duration_gap_band = 1-3s | Recall@1 0.6700; 2-way 95% CI [0.4983, 0.8461]; Recall@5 0.8671; MRR 0.7592 | 903 episodes | frozen encoder; crossed performance/family bootstrap |
| Stratum: duration_gap_band = 3+s | Recall@1 0.9494; 2-way 95% CI [0.9083, 0.9808]; Recall@5 0.9922; MRR 0.9684 | 2193 episodes | frozen encoder; crossed performance/family bootstrap |
| Stratum: gallery_render_quality = warning | Recall@1 0.8138; 2-way 95% CI [0.7527, 0.8709]; Recall@5 0.9350; MRR 0.8684 | 6450 episodes | frozen encoder; crossed performance/family bootstrap |
| Stratum: pitch_register_gap_band = 0-5 semitones | Recall@1 0.9984; 2-way 95% CI [0.9907, 1.0000]; Recall@5 1.0000; MRR 0.9992 | 645 episodes | frozen encoder; crossed performance/family bootstrap |
| Stratum: pitch_register_gap_band = 13+ semitones | Recall@1 0.7389; 2-way 95% CI [0.6619, 0.8089]; Recall@5 0.9054; MRR 0.8138 | 4386 episodes | frozen encoder; crossed performance/family bootstrap |
| Stratum: pitch_register_gap_band = 6-12 semitones | Recall@1 0.9612; 2-way 95% CI [0.9295, 0.9852]; Recall@5 0.9972; MRR 0.9777 | 1419 episodes | frozen encoder; crossed performance/family bootstrap |
| Stratum: query_articulation = held | Recall@1 0.7915; 2-way 95% CI [0.7142, 0.8607]; Recall@5 0.9214; MRR 0.8502 | 3870 episodes | frozen encoder; crossed performance/family bootstrap |
| Stratum: query_articulation = phrased | Recall@1 0.8473; 2-way 95% CI [0.7574, 0.9264]; Recall@5 0.9554; MRR 0.8957 | 2580 episodes | frozen encoder; crossed performance/family bootstrap |
| Stratum: query_duration_band = 2-4s | Recall@1 0.9128; 2-way 95% CI [0.8023, 1.0000]; Recall@5 0.9845; MRR 0.9444 | 516 episodes | frozen encoder; crossed performance/family bootstrap |
| Stratum: query_duration_band = 4-6s | Recall@1 0.6556; 2-way 95% CI [0.4895, 0.8095]; Recall@5 0.8627; MRR 0.7501 | 903 episodes | frozen encoder; crossed performance/family bootstrap |
| Stratum: query_duration_band = 6-8s | Recall@1 0.8320; 2-way 95% CI [0.7617, 0.8881]; Recall@5 0.9430; MRR 0.8818 | 5031 episodes | frozen encoder; crossed performance/family bootstrap |
| Stratum: query_expression = expressive | Recall@1 0.7868; 2-way 95% CI [0.6589, 0.8915]; Recall@5 0.9380; MRR 0.8524 | 258 episodes | frozen encoder; crossed performance/family bootstrap |
| Stratum: query_expression = plain | Recall@1 0.8149; 2-way 95% CI [0.7518, 0.8721]; Recall@5 0.9349; MRR 0.8691 | 6192 episodes | frozen encoder; crossed performance/family bootstrap |
| Stratum: query_note_density_band = dense | Recall@1 0.8140; 2-way 95% CI [0.7199, 0.8935]; Recall@5 0.9390; MRR 0.8697 | 1935 episodes | frozen encoder; crossed performance/family bootstrap |
| Stratum: query_note_density_band = moderate | Recall@1 0.7953; 2-way 95% CI [0.6930, 0.8845]; Recall@5 0.9291; MRR 0.8559 | 2580 episodes | frozen encoder; crossed performance/family bootstrap |
| Stratum: query_note_density_band = sparse | Recall@1 0.8382; 2-way 95% CI [0.7292, 0.9276]; Recall@5 0.9390; MRR 0.8837 | 1935 episodes | frozen encoder; crossed performance/family bootstrap |
| Stratum: query_polyphony = monophonic | Recall@1 0.7665; 2-way 95% CI [0.6725, 0.8566]; Recall@5 0.9163; MRR 0.8342 | 3096 episodes | frozen encoder; crossed performance/family bootstrap |
| Stratum: query_polyphony = polyphonic | Recall@1 0.8575; 2-way 95% CI [0.7862, 0.9141]; Recall@5 0.9523; MRR 0.8999 | 3354 episodes | frozen encoder; crossed performance/family bootstrap |
| Stratum: query_render_quality = clean | Recall@1 0.7935; 2-way 95% CI [0.7225, 0.8579]; Recall@5 0.9285; MRR 0.8546 | 3778 episodes | frozen encoder; crossed performance/family bootstrap |
| Stratum: query_render_quality = warning | Recall@1 0.8424; 2-way 95% CI [0.7770, 0.8979]; Recall@5 0.9442; MRR 0.8878 | 2672 episodes | frozen encoder; crossed performance/family bootstrap |
| Stratum: source_preset_lineage_group_size = single-state | Recall@1 0.8138; 2-way 95% CI [0.7527, 0.8709]; Recall@5 0.9350; MRR 0.8684 | 6450 episodes | frozen encoder; crossed performance/family bootstrap |

Vital controls for this representation version:

| Vital control family | Enabled / exact instances | Vital output count | Update method / training signal |
|---|---:|---:|---|
| Global settings ❌ 0 / 562 | 0 | unchanged source preset; no head, optimization, loss, or gradient |
| Oscillators and voices ❌ 0 / 88 | 0 | unchanged source preset; no head, optimization, loss, or gradient |
| Filters ❌ 0 / 53 | 0 | unchanged source preset; no head, optimization, loss, or gradient |
| Envelopes ❌ 0 / 54 | 0 | unchanged source preset; no head, optimization, loss, or gradient |
| LFOs and modulation ❌ 0 / 170 | 0 | unchanged source preset; no head, optimization, loss, or gradient |
| Effects ❌ 0 / 46 | 0 | unchanged source preset; no head, optimization, loss, or gradient |
| Wavetables, curves and samples ❌ 0 / 6 | 0 | unchanged source preset; no head, optimization, loss, or gradient |
### BEATs iter3+ AS2M — masked_mean

Checkpoint: `lpepino/beats_ckpts/BEATs_iter3_plus_AS2M.pt` ([upstream checkpoint](https://huggingface.co/lpepino/beats_ckpts)); requested sample rate: 16000 Hz; representation: pretrained iter3+ final pre-predictor features; output size: 768-D. Status: **executed**.

| Measure | Result or coverage | Output count | Update / supervision path |
|---|---:|---:|---|
| Checkpoint provenance / license | SHA-256 `d43cbfad4d7b56381c061d7a24774f908d4d94c72961f6eb1d9090ff18cd8d34`; source `5b53b0404df452a3a607d7e67687227730e5bad1`; loader `31c5b904ca1bf2afb4c234a6675c683a4e5fc7cd`; runtime `{"numpy": "1.26.4", "scipy": "1.17.1", "torch": "2.12.1+xpu", "torchaudio": "2.11.0+xpu"}`; Mirror labels MIT; upstream checkpoint term unresolved. Authorized local non-commercial research only; weights are not redistributed. | — | exact local lock; frozen inference |
| Cross-MIDI common-gallery Recall@1 | 0.5927 [0.5608, 0.6234] | — | frozen backbone; cosine retrieval |
| Cross-MIDI common-gallery Recall@5 | 0.7864 [0.7626, 0.8093] | — | frozen backbone; cosine retrieval |
| Cross-MIDI common-gallery MRR | 0.6821 [0.6559, 0.7078] | — | frozen backbone; cosine retrieval |
| Cross-MIDI family-level Recall@1 | 0.5927 [0.5608, 0.6234] | — | frozen backbone; cosine retrieval |
| Cross-MIDI mixed-gallery Recall@1 | 0.5532 [0.5229, 0.5854] | — | frozen backbone; cosine retrieval |
| Same-MIDI other-patch negative win rate | 0.4127 [0.3792, 0.4470] (6450 episodes; 825600 negatives) | — | frozen backbone; cosine retrieval |
| Acoustic hard-negative top-5 win rate | 0.5743 [0.5377, 0.6093] (6450 episodes; 32250 negatives) | — | frozen backbone; cosine retrieval |
| Known source-lineage near-neighbor win rate | not run (0 episodes; 0 negatives) | — | frozen backbone; cosine retrieval |
| Gallery candidates / episodes | [129] / 6450 | — | fixed candidate sets across encoders |
| Features created / cache reused | 6480 / 0 | 6480 | resumable extraction pass; cache identity checked |
| Extraction pass time / peak RSS / accelerator memory | 1484.423 s / 3577622528 / 445343232 bytes | — | local inference and cache validation; no training |
| Stratum: duration_gap_band = 0-1s | Recall@1 0.4866; 2-way 95% CI [0.3798, 0.5996]; Recall@5 0.7233; MRR 0.5963 | 3354 episodes | frozen encoder; crossed performance/family bootstrap |
| Stratum: duration_gap_band = 1-3s | Recall@1 0.4142; 2-way 95% CI [0.2060, 0.6434]; Recall@5 0.6578; MRR 0.5260 | 903 episodes | frozen encoder; crossed performance/family bootstrap |
| Stratum: duration_gap_band = 3+s | Recall@1 0.8285; 2-way 95% CI [0.7191, 0.9234]; Recall@5 0.9357; MRR 0.8777 | 2193 episodes | frozen encoder; crossed performance/family bootstrap |
| Stratum: gallery_render_quality = warning | Recall@1 0.5927; 2-way 95% CI [0.5031, 0.6884]; Recall@5 0.7864; MRR 0.6821 | 6450 episodes | frozen encoder; crossed performance/family bootstrap |
| Stratum: pitch_register_gap_band = 0-5 semitones | Recall@1 0.9798; 2-way 95% CI [0.9566, 0.9953]; Recall@5 0.9984; MRR 0.9876 | 645 episodes | frozen encoder; crossed performance/family bootstrap |
| Stratum: pitch_register_gap_band = 13+ semitones | Recall@1 0.4412; 2-way 95% CI [0.3495, 0.5317]; Recall@5 0.6956; MRR 0.5592 | 4386 episodes | frozen encoder; crossed performance/family bootstrap |
| Stratum: pitch_register_gap_band = 6-12 semitones | Recall@1 0.8851; 2-way 95% CI [0.8013, 0.9549]; Recall@5 0.9704; MRR 0.9233 | 1419 episodes | frozen encoder; crossed performance/family bootstrap |
| Stratum: query_articulation = held | Recall@1 0.5349; 2-way 95% CI [0.4318, 0.6416]; Recall@5 0.7553; MRR 0.6364 | 3870 episodes | frozen encoder; crossed performance/family bootstrap |
| Stratum: query_articulation = phrased | Recall@1 0.6795; 2-way 95% CI [0.5291, 0.8151]; Recall@5 0.8329; MRR 0.7507 | 2580 episodes | frozen encoder; crossed performance/family bootstrap |
| Stratum: query_duration_band = 2-4s | Recall@1 0.6628; 2-way 95% CI [0.3585, 0.9593]; Recall@5 0.8333; MRR 0.7406 | 516 episodes | frozen encoder; crossed performance/family bootstrap |
| Stratum: query_duration_band = 4-6s | Recall@1 0.3931; 2-way 95% CI [0.2016, 0.6035]; Recall@5 0.6512; MRR 0.5117 | 903 episodes | frozen encoder; crossed performance/family bootstrap |
| Stratum: query_duration_band = 6-8s | Recall@1 0.6213; 2-way 95% CI [0.5170, 0.7156]; Recall@5 0.8058; MRR 0.7067 | 5031 episodes | frozen encoder; crossed performance/family bootstrap |
| Stratum: query_expression = expressive | Recall@1 0.5233; 2-way 95% CI [0.3023, 0.7442]; Recall@5 0.7403; MRR 0.6232 | 258 episodes | frozen encoder; crossed performance/family bootstrap |
| Stratum: query_expression = plain | Recall@1 0.5956; 2-way 95% CI [0.5055, 0.6870]; Recall@5 0.7883; MRR 0.6846 | 6192 episodes | frozen encoder; crossed performance/family bootstrap |
| Stratum: query_note_density_band = dense | Recall@1 0.5995; 2-way 95% CI [0.4481, 0.7463]; Recall@5 0.8010; MRR 0.6917 | 1935 episodes | frozen encoder; crossed performance/family bootstrap |
| Stratum: query_note_density_band = moderate | Recall@1 0.6109; 2-way 95% CI [0.4539, 0.7543]; Recall@5 0.7752; MRR 0.6900 | 2580 episodes | frozen encoder; crossed performance/family bootstrap |
| Stratum: query_note_density_band = sparse | Recall@1 0.5618; 2-way 95% CI [0.4176, 0.7013]; Recall@5 0.7866; MRR 0.6620 | 1935 episodes | frozen encoder; crossed performance/family bootstrap |
| Stratum: query_polyphony = monophonic | Recall@1 0.5417; 2-way 95% CI [0.3986, 0.6809]; Recall@5 0.7322; MRR 0.6298 | 3096 episodes | frozen encoder; crossed performance/family bootstrap |
| Stratum: query_polyphony = polyphonic | Recall@1 0.6398; 2-way 95% CI [0.5239, 0.7424]; Recall@5 0.8363; MRR 0.7304 | 3354 episodes | frozen encoder; crossed performance/family bootstrap |
| Stratum: query_render_quality = clean | Recall@1 0.5413; 2-way 95% CI [0.4359, 0.6403]; Recall@5 0.7512; MRR 0.6387 | 3778 episodes | frozen encoder; crossed performance/family bootstrap |
| Stratum: query_render_quality = warning | Recall@1 0.6654; 2-way 95% CI [0.5703, 0.7513]; Recall@5 0.8361; MRR 0.7435 | 2672 episodes | frozen encoder; crossed performance/family bootstrap |
| Stratum: source_preset_lineage_group_size = single-state | Recall@1 0.5927; 2-way 95% CI [0.5031, 0.6884]; Recall@5 0.7864; MRR 0.6821 | 6450 episodes | frozen encoder; crossed performance/family bootstrap |

Vital controls for this representation version:

| Vital control family | Enabled / exact instances | Vital output count | Update method / training signal |
|---|---:|---:|---|
| Global settings ❌ 0 / 562 | 0 | unchanged source preset; no head, optimization, loss, or gradient |
| Oscillators and voices ❌ 0 / 88 | 0 | unchanged source preset; no head, optimization, loss, or gradient |
| Filters ❌ 0 / 53 | 0 | unchanged source preset; no head, optimization, loss, or gradient |
| Envelopes ❌ 0 / 54 | 0 | unchanged source preset; no head, optimization, loss, or gradient |
| LFOs and modulation ❌ 0 / 170 | 0 | unchanged source preset; no head, optimization, loss, or gradient |
| Effects ❌ 0 / 46 | 0 | unchanged source preset; no head, optimization, loss, or gradient |
| Wavetables, curves and samples ❌ 0 / 6 | 0 | unchanged source preset; no head, optimization, loss, or gradient |
### BEATs iter3+ AS2M — temporal_pyramid_8

Checkpoint: `lpepino/beats_ckpts/BEATs_iter3_plus_AS2M.pt` ([upstream checkpoint](https://huggingface.co/lpepino/beats_ckpts)); requested sample rate: 16000 Hz; representation: pretrained iter3+ final pre-predictor features; output size: 12288-D. Status: **executed**.

| Measure | Result or coverage | Output count | Update / supervision path |
|---|---:|---:|---|
| Checkpoint provenance / license | SHA-256 `d43cbfad4d7b56381c061d7a24774f908d4d94c72961f6eb1d9090ff18cd8d34`; source `5b53b0404df452a3a607d7e67687227730e5bad1`; loader `31c5b904ca1bf2afb4c234a6675c683a4e5fc7cd`; runtime `{"numpy": "1.26.4", "scipy": "1.17.1", "torch": "2.12.1+xpu", "torchaudio": "2.11.0+xpu"}`; Mirror labels MIT; upstream checkpoint term unresolved. Authorized local non-commercial research only; weights are not redistributed. | — | exact local lock; frozen inference |
| Cross-MIDI common-gallery Recall@1 | 0.6216 [0.5890, 0.6529] | — | frozen backbone; cosine retrieval |
| Cross-MIDI common-gallery Recall@5 | 0.8130 [0.7895, 0.8349] | — | frozen backbone; cosine retrieval |
| Cross-MIDI common-gallery MRR | 0.7092 [0.6823, 0.7348] | — | frozen backbone; cosine retrieval |
| Cross-MIDI family-level Recall@1 | 0.6216 [0.5890, 0.6529] | — | frozen backbone; cosine retrieval |
| Cross-MIDI mixed-gallery Recall@1 | 0.5440 [0.5116, 0.5766] | — | frozen backbone; cosine retrieval |
| Same-MIDI other-patch negative win rate | 0.2609 [0.2208, 0.3031] (6450 episodes; 825600 negatives) | — | frozen backbone; cosine retrieval |
| Acoustic hard-negative top-5 win rate | 0.4718 [0.4285, 0.5149] (6450 episodes; 32250 negatives) | — | frozen backbone; cosine retrieval |
| Known source-lineage near-neighbor win rate | not run (0 episodes; 0 negatives) | — | frozen backbone; cosine retrieval |
| Gallery candidates / episodes | [129] / 6450 | — | fixed candidate sets across encoders |
| Features created / cache reused | 6480 / 0 | 6480 | resumable extraction pass; cache identity checked |
| Extraction pass time / peak RSS / accelerator memory | 1484.423 s / 3577622528 / 445343232 bytes | — | local inference and cache validation; no training |
| Stratum: duration_gap_band = 0-1s | Recall@1 0.5328; 2-way 95% CI [0.4299, 0.6404]; Recall@5 0.7651; MRR 0.6385 | 3354 episodes | frozen encoder; crossed performance/family bootstrap |
| Stratum: duration_gap_band = 1-3s | Recall@1 0.4662; 2-way 95% CI [0.2724, 0.6800]; Recall@5 0.6966; MRR 0.5761 | 903 episodes | frozen encoder; crossed performance/family bootstrap |
| Stratum: duration_gap_band = 3+s | Recall@1 0.8212; 2-way 95% CI [0.7155, 0.9156]; Recall@5 0.9343; MRR 0.8722 | 2193 episodes | frozen encoder; crossed performance/family bootstrap |
| Stratum: gallery_render_quality = warning | Recall@1 0.6216; 2-way 95% CI [0.5367, 0.7093]; Recall@5 0.8130; MRR 0.7092 | 6450 episodes | frozen encoder; crossed performance/family bootstrap |
| Stratum: pitch_register_gap_band = 0-5 semitones | Recall@1 0.9752; 2-way 95% CI [0.9442, 0.9953]; Recall@5 0.9969; MRR 0.9845 | 645 episodes | frozen encoder; crossed performance/family bootstrap |
| Stratum: pitch_register_gap_band = 13+ semitones | Recall@1 0.4838; 2-way 95% CI [0.3960, 0.5643]; Recall@5 0.7364; MRR 0.5997 | 4386 episodes | frozen encoder; crossed performance/family bootstrap |
| Stratum: pitch_register_gap_band = 6-12 semitones | Recall@1 0.8865; 2-way 95% CI [0.8076, 0.9549]; Recall@5 0.9662; MRR 0.9228 | 1419 episodes | frozen encoder; crossed performance/family bootstrap |
| Stratum: query_articulation = held | Recall@1 0.5685; 2-way 95% CI [0.4711, 0.6638]; Recall@5 0.7811; MRR 0.6663 | 3870 episodes | frozen encoder; crossed performance/family bootstrap |
| Stratum: query_articulation = phrased | Recall@1 0.7012; 2-way 95% CI [0.5620, 0.8267]; Recall@5 0.8609; MRR 0.7737 | 2580 episodes | frozen encoder; crossed performance/family bootstrap |
| Stratum: query_duration_band = 2-4s | Recall@1 0.6628; 2-way 95% CI [0.3837, 0.9380]; Recall@5 0.8527; MRR 0.7424 | 516 episodes | frozen encoder; crossed performance/family bootstrap |
| Stratum: query_duration_band = 4-6s | Recall@1 0.4452; 2-way 95% CI [0.2713, 0.6368]; Recall@5 0.6921; MRR 0.5629 | 903 episodes | frozen encoder; crossed performance/family bootstrap |
| Stratum: query_duration_band = 6-8s | Recall@1 0.6490; 2-way 95% CI [0.5496, 0.7374]; Recall@5 0.8306; MRR 0.7321 | 5031 episodes | frozen encoder; crossed performance/family bootstrap |
| Stratum: query_expression = expressive | Recall@1 0.5078; 2-way 95% CI [0.2791, 0.7287]; Recall@5 0.7403; MRR 0.6062 | 258 episodes | frozen encoder; crossed performance/family bootstrap |
| Stratum: query_expression = plain | Recall@1 0.6263; 2-way 95% CI [0.5425, 0.7106]; Recall@5 0.8161; MRR 0.7135 | 6192 episodes | frozen encoder; crossed performance/family bootstrap |
| Stratum: query_note_density_band = dense | Recall@1 0.6300; 2-way 95% CI [0.4966, 0.7607]; Recall@5 0.8284; MRR 0.7193 | 1935 episodes | frozen encoder; crossed performance/family bootstrap |
| Stratum: query_note_density_band = moderate | Recall@1 0.6360; 2-way 95% CI [0.4950, 0.7702]; Recall@5 0.8132; MRR 0.7168 | 2580 episodes | frozen encoder; crossed performance/family bootstrap |
| Stratum: query_note_density_band = sparse | Recall@1 0.5938; 2-way 95% CI [0.4537, 0.7251]; Recall@5 0.7974; MRR 0.6891 | 1935 episodes | frozen encoder; crossed performance/family bootstrap |
| Stratum: query_polyphony = monophonic | Recall@1 0.5766; 2-way 95% CI [0.4467, 0.7070]; Recall@5 0.7697; MRR 0.6659 | 3096 episodes | frozen encoder; crossed performance/family bootstrap |
| Stratum: query_polyphony = polyphonic | Recall@1 0.6631; 2-way 95% CI [0.5575, 0.7588]; Recall@5 0.8530; MRR 0.7492 | 3354 episodes | frozen encoder; crossed performance/family bootstrap |
| Stratum: query_render_quality = clean | Recall@1 0.5707; 2-way 95% CI [0.4717, 0.6664]; Recall@5 0.7777; MRR 0.6669 | 3778 episodes | frozen encoder; crossed performance/family bootstrap |
| Stratum: query_render_quality = warning | Recall@1 0.6935; 2-way 95% CI [0.6070, 0.7740]; Recall@5 0.8630; MRR 0.7691 | 2672 episodes | frozen encoder; crossed performance/family bootstrap |
| Stratum: source_preset_lineage_group_size = single-state | Recall@1 0.6216; 2-way 95% CI [0.5367, 0.7093]; Recall@5 0.8130; MRR 0.7092 | 6450 episodes | frozen encoder; crossed performance/family bootstrap |

Vital controls for this representation version:

| Vital control family | Enabled / exact instances | Vital output count | Update method / training signal |
|---|---:|---:|---|
| Global settings ❌ 0 / 562 | 0 | unchanged source preset; no head, optimization, loss, or gradient |
| Oscillators and voices ❌ 0 / 88 | 0 | unchanged source preset; no head, optimization, loss, or gradient |
| Filters ❌ 0 / 53 | 0 | unchanged source preset; no head, optimization, loss, or gradient |
| Envelopes ❌ 0 / 54 | 0 | unchanged source preset; no head, optimization, loss, or gradient |
| LFOs and modulation ❌ 0 / 170 | 0 | unchanged source preset; no head, optimization, loss, or gradient |
| Effects ❌ 0 / 46 | 0 | unchanged source preset; no head, optimization, loss, or gradient |
| Wavetables, curves and samples ❌ 0 / 6 | 0 | unchanged source preset; no head, optimization, loss, or gradient |
### Meta EnCodec 24 kHz — masked_mean

Checkpoint: `facebook/encodec_24khz` ([upstream checkpoint](https://huggingface.co/facebook/encodec_24khz)); requested sample rate: 24000 Hz; representation: continuous encoder output before RVQ; output size: 128-D. Status: **executed**.

| Measure | Result or coverage | Output count | Update / supervision path |
|---|---:|---:|---|
| Checkpoint provenance / license | SHA-256 `304c7656c7f978fa9d6838fbc7312108d2345ba6b1b816c8599653da8a71bdb3`; source `c5e61996c75613cfd18f4c53cc06667932675776`; loader `transformers==4.50.0`; runtime `{"numpy": "1.26.4", "scipy": "1.17.1", "torch": "2.12.1+xpu", "transformers": "4.50.0"}`; MIT | — | exact local lock; frozen inference |
| Cross-MIDI common-gallery Recall@1 | 0.2422 [0.2017, 0.2828] | — | frozen backbone; cosine retrieval |
| Cross-MIDI common-gallery Recall@5 | 0.4112 [0.3657, 0.4575] | — | frozen backbone; cosine retrieval |
| Cross-MIDI common-gallery MRR | 0.3364 [0.2966, 0.3774] | — | frozen backbone; cosine retrieval |
| Cross-MIDI family-level Recall@1 | 0.2422 [0.2017, 0.2828] | — | frozen backbone; cosine retrieval |
| Cross-MIDI mixed-gallery Recall@1 | 0.2457 [0.2081, 0.2860] | — | frozen backbone; cosine retrieval |
| Same-MIDI other-patch negative win rate | 0.2113 [0.1744, 0.2507] (6450 episodes; 825600 negatives) | — | frozen backbone; cosine retrieval |
| Acoustic hard-negative top-5 win rate | 0.3600 [0.3171, 0.4037] (6450 episodes; 32250 negatives) | — | frozen backbone; cosine retrieval |
| Known source-lineage near-neighbor win rate | not run (0 episodes; 0 negatives) | — | frozen backbone; cosine retrieval |
| Gallery candidates / episodes | [129] / 6450 | — | fixed candidate sets across encoders |
| Features created / cache reused | 2077 / 4403 | 6480 | resumable extraction pass; cache identity checked |
| Extraction pass time / peak RSS / accelerator memory | 1220.946 s / 2156445696 / 253531648 bytes | — | local inference and cache validation; no training |
| Resumable pass scope | 2077 new and 4403 cache-reused features | 6480 | elapsed time is this pass, not cumulative across earlier partial runs |
| Stratum: duration_gap_band = 0-1s | Recall@1 0.2066; 2-way 95% CI [0.1428, 0.2746]; Recall@5 0.3551; MRR 0.2934 | 3354 episodes | frozen encoder; crossed performance/family bootstrap |
| Stratum: duration_gap_band = 1-3s | Recall@1 0.1838; 2-way 95% CI [0.0997, 0.2802]; Recall@5 0.3212; MRR 0.2664 | 903 episodes | frozen encoder; crossed performance/family bootstrap |
| Stratum: duration_gap_band = 3+s | Recall@1 0.3206; 2-way 95% CI [0.2193, 0.4405]; Recall@5 0.5340; MRR 0.4311 | 2193 episodes | frozen encoder; crossed performance/family bootstrap |
| Stratum: gallery_render_quality = warning | Recall@1 0.2422; 2-way 95% CI [0.1820, 0.3047]; Recall@5 0.4112; MRR 0.3364 | 6450 episodes | frozen encoder; crossed performance/family bootstrap |
| Stratum: pitch_register_gap_band = 0-5 semitones | Recall@1 0.5225; 2-way 95% CI [0.2512, 0.7907]; Recall@5 0.7116; MRR 0.6234 | 645 episodes | frozen encoder; crossed performance/family bootstrap |
| Stratum: pitch_register_gap_band = 13+ semitones | Recall@1 0.1783; 2-way 95% CI [0.1265, 0.2330]; Recall@5 0.3224; MRR 0.2631 | 4386 episodes | frozen encoder; crossed performance/family bootstrap |
| Stratum: pitch_register_gap_band = 6-12 semitones | Recall@1 0.3122; 2-way 95% CI [0.2417, 0.3883]; Recall@5 0.5490; MRR 0.4325 | 1419 episodes | frozen encoder; crossed performance/family bootstrap |
| Stratum: query_articulation = held | Recall@1 0.2016; 2-way 95% CI [0.1452, 0.2625]; Recall@5 0.3499; MRR 0.2883 | 3870 episodes | frozen encoder; crossed performance/family bootstrap |
| Stratum: query_articulation = phrased | Recall@1 0.3031; 2-way 95% CI [0.2085, 0.4047]; Recall@5 0.5031; MRR 0.4086 | 2580 episodes | frozen encoder; crossed performance/family bootstrap |
| Stratum: query_duration_band = 2-4s | Recall@1 0.1977; 2-way 95% CI [0.0891, 0.3178]; Recall@5 0.3857; MRR 0.3007 | 516 episodes | frozen encoder; crossed performance/family bootstrap |
| Stratum: query_duration_band = 4-6s | Recall@1 0.1905; 2-way 95% CI [0.1008, 0.2879]; Recall@5 0.3234; MRR 0.2713 | 903 episodes | frozen encoder; crossed performance/family bootstrap |
| Stratum: query_duration_band = 6-8s | Recall@1 0.2560; 2-way 95% CI [0.1898, 0.3284]; Recall@5 0.4295; MRR 0.3518 | 5031 episodes | frozen encoder; crossed performance/family bootstrap |
| Stratum: query_expression = expressive | Recall@1 0.1395; 2-way 95% CI [0.0465, 0.2403]; Recall@5 0.2984; MRR 0.2295 | 258 episodes | frozen encoder; crossed performance/family bootstrap |
| Stratum: query_expression = plain | Recall@1 0.2464; 2-way 95% CI [0.1854, 0.3110]; Recall@5 0.4159; MRR 0.3409 | 6192 episodes | frozen encoder; crossed performance/family bootstrap |
| Stratum: query_note_density_band = dense | Recall@1 0.2036; 2-way 95% CI [0.1390, 0.2760]; Recall@5 0.3426; MRR 0.2897 | 1935 episodes | frozen encoder; crossed performance/family bootstrap |
| Stratum: query_note_density_band = moderate | Recall@1 0.2860; 2-way 95% CI [0.1934, 0.3926]; Recall@5 0.4764; MRR 0.3865 | 2580 episodes | frozen encoder; crossed performance/family bootstrap |
| Stratum: query_note_density_band = sparse | Recall@1 0.2222; 2-way 95% CI [0.1401, 0.3142]; Recall@5 0.3928; MRR 0.3163 | 1935 episodes | frozen encoder; crossed performance/family bootstrap |
| Stratum: query_polyphony = monophonic | Recall@1 0.2422; 2-way 95% CI [0.1609, 0.3395]; Recall@5 0.4086; MRR 0.3346 | 3096 episodes | frozen encoder; crossed performance/family bootstrap |
| Stratum: query_polyphony = polyphonic | Recall@1 0.2421; 2-way 95% CI [0.1756, 0.3154]; Recall@5 0.4135; MRR 0.3381 | 3354 episodes | frozen encoder; crossed performance/family bootstrap |
| Stratum: query_render_quality = clean | Recall@1 0.1932; 2-way 95% CI [0.1328, 0.2616]; Recall@5 0.3573; MRR 0.2881 | 3778 episodes | frozen encoder; crossed performance/family bootstrap |
| Stratum: query_render_quality = warning | Recall@1 0.3114; 2-way 95% CI [0.2228, 0.3975]; Recall@5 0.4873; MRR 0.4048 | 2672 episodes | frozen encoder; crossed performance/family bootstrap |
| Stratum: source_preset_lineage_group_size = single-state | Recall@1 0.2422; 2-way 95% CI [0.1820, 0.3047]; Recall@5 0.4112; MRR 0.3364 | 6450 episodes | frozen encoder; crossed performance/family bootstrap |

Vital controls for this representation version:

| Vital control family | Enabled / exact instances | Vital output count | Update method / training signal |
|---|---:|---:|---|
| Global settings ❌ 0 / 562 | 0 | unchanged source preset; no head, optimization, loss, or gradient |
| Oscillators and voices ❌ 0 / 88 | 0 | unchanged source preset; no head, optimization, loss, or gradient |
| Filters ❌ 0 / 53 | 0 | unchanged source preset; no head, optimization, loss, or gradient |
| Envelopes ❌ 0 / 54 | 0 | unchanged source preset; no head, optimization, loss, or gradient |
| LFOs and modulation ❌ 0 / 170 | 0 | unchanged source preset; no head, optimization, loss, or gradient |
| Effects ❌ 0 / 46 | 0 | unchanged source preset; no head, optimization, loss, or gradient |
| Wavetables, curves and samples ❌ 0 / 6 | 0 | unchanged source preset; no head, optimization, loss, or gradient |
### Meta EnCodec 24 kHz — temporal_pyramid_8

Checkpoint: `facebook/encodec_24khz` ([upstream checkpoint](https://huggingface.co/facebook/encodec_24khz)); requested sample rate: 24000 Hz; representation: continuous encoder output before RVQ; output size: 2048-D. Status: **executed**.

| Measure | Result or coverage | Output count | Update / supervision path |
|---|---:|---:|---|
| Checkpoint provenance / license | SHA-256 `304c7656c7f978fa9d6838fbc7312108d2345ba6b1b816c8599653da8a71bdb3`; source `c5e61996c75613cfd18f4c53cc06667932675776`; loader `transformers==4.50.0`; runtime `{"numpy": "1.26.4", "scipy": "1.17.1", "torch": "2.12.1+xpu", "transformers": "4.50.0"}`; MIT | — | exact local lock; frozen inference |
| Cross-MIDI common-gallery Recall@1 | 0.2336 [0.1946, 0.2746] | — | frozen backbone; cosine retrieval |
| Cross-MIDI common-gallery Recall@5 | 0.4353 [0.3898, 0.4795] | — | frozen backbone; cosine retrieval |
| Cross-MIDI common-gallery MRR | 0.3394 [0.2996, 0.3797] | — | frozen backbone; cosine retrieval |
| Cross-MIDI family-level Recall@1 | 0.2336 [0.1946, 0.2746] | — | frozen backbone; cosine retrieval |
| Cross-MIDI mixed-gallery Recall@1 | 0.2464 [0.2104, 0.2856] | — | frozen backbone; cosine retrieval |
| Same-MIDI other-patch negative win rate | 0.1202 [0.0881, 0.1540] (6450 episodes; 825600 negatives) | — | frozen backbone; cosine retrieval |
| Acoustic hard-negative top-5 win rate | 0.2634 [0.2229, 0.3056] (6450 episodes; 32250 negatives) | — | frozen backbone; cosine retrieval |
| Known source-lineage near-neighbor win rate | not run (0 episodes; 0 negatives) | — | frozen backbone; cosine retrieval |
| Gallery candidates / episodes | [129] / 6450 | — | fixed candidate sets across encoders |
| Features created / cache reused | 2077 / 4403 | 6480 | resumable extraction pass; cache identity checked |
| Extraction pass time / peak RSS / accelerator memory | 1220.946 s / 2156445696 / 253531648 bytes | — | local inference and cache validation; no training |
| Resumable pass scope | 2077 new and 4403 cache-reused features | 6480 | elapsed time is this pass, not cumulative across earlier partial runs |
| Stratum: duration_gap_band = 0-1s | Recall@1 0.2487; 2-way 95% CI [0.1762, 0.3253]; Recall@5 0.4436; MRR 0.3493 | 3354 episodes | frozen encoder; crossed performance/family bootstrap |
| Stratum: duration_gap_band = 1-3s | Recall@1 0.2060; 2-way 95% CI [0.1152, 0.3068]; Recall@5 0.3887; MRR 0.3028 | 903 episodes | frozen encoder; crossed performance/family bootstrap |
| Stratum: duration_gap_band = 3+s | Recall@1 0.2221; 2-way 95% CI [0.1605, 0.2877]; Recall@5 0.4419; MRR 0.3393 | 2193 episodes | frozen encoder; crossed performance/family bootstrap |
| Stratum: gallery_render_quality = warning | Recall@1 0.2336; 2-way 95% CI [0.1794, 0.2899]; Recall@5 0.4353; MRR 0.3394 | 6450 episodes | frozen encoder; crossed performance/family bootstrap |
| Stratum: pitch_register_gap_band = 0-5 semitones | Recall@1 0.3550; 2-way 95% CI [0.2233, 0.5178]; Recall@5 0.6047; MRR 0.4826 | 645 episodes | frozen encoder; crossed performance/family bootstrap |
| Stratum: pitch_register_gap_band = 13+ semitones | Recall@1 0.1920; 2-way 95% CI [0.1430, 0.2442]; Recall@5 0.3719; MRR 0.2897 | 4386 episodes | frozen encoder; crossed performance/family bootstrap |
| Stratum: pitch_register_gap_band = 6-12 semitones | Recall@1 0.3073; 2-way 95% CI [0.2142, 0.4094]; Recall@5 0.5546; MRR 0.4279 | 1419 episodes | frozen encoder; crossed performance/family bootstrap |
| Stratum: query_articulation = held | Recall@1 0.2204; 2-way 95% CI [0.1581, 0.2871]; Recall@5 0.4103; MRR 0.3199 | 3870 episodes | frozen encoder; crossed performance/family bootstrap |
| Stratum: query_articulation = phrased | Recall@1 0.2535; 2-way 95% CI [0.1930, 0.3159]; Recall@5 0.4729; MRR 0.3687 | 2580 episodes | frozen encoder; crossed performance/family bootstrap |
| Stratum: query_duration_band = 2-4s | Recall@1 0.2248; 2-way 95% CI [0.1163, 0.3450]; Recall@5 0.4380; MRR 0.3407 | 516 episodes | frozen encoder; crossed performance/family bootstrap |
| Stratum: query_duration_band = 4-6s | Recall@1 0.2082; 2-way 95% CI [0.1118, 0.3079]; Recall@5 0.3976; MRR 0.3086 | 903 episodes | frozen encoder; crossed performance/family bootstrap |
| Stratum: query_duration_band = 6-8s | Recall@1 0.2391; 2-way 95% CI [0.1829, 0.2995]; Recall@5 0.4419; MRR 0.3448 | 5031 episodes | frozen encoder; crossed performance/family bootstrap |
| Stratum: query_expression = expressive | Recall@1 0.0891; 2-way 95% CI [0.0310, 0.1550]; Recall@5 0.2636; MRR 0.1799 | 258 episodes | frozen encoder; crossed performance/family bootstrap |
| Stratum: query_expression = plain | Recall@1 0.2397; 2-way 95% CI [0.1862, 0.2946]; Recall@5 0.4425; MRR 0.3461 | 6192 episodes | frozen encoder; crossed performance/family bootstrap |
| Stratum: query_note_density_band = dense | Recall@1 0.2460; 2-way 95% CI [0.1726, 0.3333]; Recall@5 0.4460; MRR 0.3492 | 1935 episodes | frozen encoder; crossed performance/family bootstrap |
| Stratum: query_note_density_band = moderate | Recall@1 0.2372; 2-way 95% CI [0.1756, 0.3035]; Recall@5 0.4539; MRR 0.3511 | 2580 episodes | frozen encoder; crossed performance/family bootstrap |
| Stratum: query_note_density_band = sparse | Recall@1 0.2165; 2-way 95% CI [0.1364, 0.3018]; Recall@5 0.4000; MRR 0.3141 | 1935 episodes | frozen encoder; crossed performance/family bootstrap |
| Stratum: query_polyphony = monophonic | Recall@1 0.2216; 2-way 95% CI [0.1651, 0.2829]; Recall@5 0.4167; MRR 0.3259 | 3096 episodes | frozen encoder; crossed performance/family bootstrap |
| Stratum: query_polyphony = polyphonic | Recall@1 0.2448; 2-way 95% CI [0.1786, 0.3175]; Recall@5 0.4526; MRR 0.3519 | 3354 episodes | frozen encoder; crossed performance/family bootstrap |
| Stratum: query_render_quality = clean | Recall@1 0.1657; 2-way 95% CI [0.1163, 0.2209]; Recall@5 0.3542; MRR 0.2687 | 3778 episodes | frozen encoder; crossed performance/family bootstrap |
| Stratum: query_render_quality = warning | Recall@1 0.3297; 2-way 95% CI [0.2452, 0.4133]; Recall@5 0.5501; MRR 0.4394 | 2672 episodes | frozen encoder; crossed performance/family bootstrap |
| Stratum: source_preset_lineage_group_size = single-state | Recall@1 0.2336; 2-way 95% CI [0.1794, 0.2899]; Recall@5 0.4353; MRR 0.3394 | 6450 episodes | frozen encoder; crossed performance/family bootstrap |

Vital controls for this representation version:

| Vital control family | Enabled / exact instances | Vital output count | Update method / training signal |
|---|---:|---:|---|
| Global settings ❌ 0 / 562 | 0 | unchanged source preset; no head, optimization, loss, or gradient |
| Oscillators and voices ❌ 0 / 88 | 0 | unchanged source preset; no head, optimization, loss, or gradient |
| Filters ❌ 0 / 53 | 0 | unchanged source preset; no head, optimization, loss, or gradient |
| Envelopes ❌ 0 / 54 | 0 | unchanged source preset; no head, optimization, loss, or gradient |
| LFOs and modulation ❌ 0 / 170 | 0 | unchanged source preset; no head, optimization, loss, or gradient |
| Effects ❌ 0 / 46 | 0 | unchanged source preset; no head, optimization, loss, or gradient |
| Wavetables, curves and samples ❌ 0 / 6 | 0 | unchanged source preset; no head, optimization, loss, or gradient |
### Descript Audio Codec 44.1 kHz — masked_mean

Checkpoint: `descript/dac_44khz` ([upstream checkpoint](https://huggingface.co/descript/dac_44khz)); requested sample rate: 44100 Hz; representation: continuous encoder output before RVQ; output size: 1024-D. Status: **executed**.

| Measure | Result or coverage | Output count | Update / supervision path |
|---|---:|---:|---|
| Checkpoint provenance / license | SHA-256 `128180a5e4d41ceabfeb952b687fb9da7cf468c2291acf5be9cadb5ab091cc46`; source `c1bc521685adf9cfe247bc39a5ca58917eda1ac4`; loader `transformers==4.50.0`; runtime `{"numpy": "1.26.4", "scipy": "1.17.1", "torch": "2.12.1+xpu", "transformers": "4.50.0"}`; MIT | — | exact local lock; frozen inference |
| Cross-MIDI common-gallery Recall@1 | 0.3138 [0.2752, 0.3541] | — | frozen backbone; cosine retrieval |
| Cross-MIDI common-gallery Recall@5 | 0.5223 [0.4806, 0.5636] | — | frozen backbone; cosine retrieval |
| Cross-MIDI common-gallery MRR | 0.4178 [0.3810, 0.4554] | — | frozen backbone; cosine retrieval |
| Cross-MIDI family-level Recall@1 | 0.3138 [0.2752, 0.3541] | — | frozen backbone; cosine retrieval |
| Cross-MIDI mixed-gallery Recall@1 | 0.3290 [0.2926, 0.3668] | — | frozen backbone; cosine retrieval |
| Same-MIDI other-patch negative win rate | 0.2887 [0.2519, 0.3259] (6450 episodes; 825600 negatives) | — | frozen backbone; cosine retrieval |
| Acoustic hard-negative top-5 win rate | 0.4806 [0.4386, 0.5186] (6450 episodes; 32250 negatives) | — | frozen backbone; cosine retrieval |
| Known source-lineage near-neighbor win rate | not run (0 episodes; 0 negatives) | — | frozen backbone; cosine retrieval |
| Gallery candidates / episodes | [129] / 6450 | — | fixed candidate sets across encoders |
| Features created / cache reused | 6426 / 54 | 6480 | resumable extraction pass; cache identity checked |
| Extraction pass time / peak RSS / accelerator memory | 8132.595 s / 3227729920 / 990163968 bytes | — | local inference and cache validation; no training |
| Resumable pass scope | 6426 new and 54 cache-reused features | 6480 | elapsed time is this pass, not cumulative across earlier partial runs |
| Stratum: duration_gap_band = 0-1s | Recall@1 0.2615; 2-way 95% CI [0.1855, 0.3432]; Recall@5 0.4505; MRR 0.3612 | 3354 episodes | frozen encoder; crossed performance/family bootstrap |
| Stratum: duration_gap_band = 1-3s | Recall@1 0.2602; 2-way 95% CI [0.1417, 0.3942]; Recall@5 0.4385; MRR 0.3541 | 903 episodes | frozen encoder; crossed performance/family bootstrap |
| Stratum: duration_gap_band = 3+s | Recall@1 0.4159; 2-way 95% CI [0.3010, 0.5417]; Recall@5 0.6667; MRR 0.5308 | 2193 episodes | frozen encoder; crossed performance/family bootstrap |
| Stratum: gallery_render_quality = warning | Recall@1 0.3138; 2-way 95% CI [0.2485, 0.3874]; Recall@5 0.5223; MRR 0.4178 | 6450 episodes | frozen encoder; crossed performance/family bootstrap |
| Stratum: pitch_register_gap_band = 0-5 semitones | Recall@1 0.6109; 2-way 95% CI [0.3318, 0.8760]; Recall@5 0.8155; MRR 0.7039 | 645 episodes | frozen encoder; crossed performance/family bootstrap |
| Stratum: pitch_register_gap_band = 13+ semitones | Recall@1 0.2298; 2-way 95% CI [0.1705, 0.2923]; Recall@5 0.4195; MRR 0.3302 | 4386 episodes | frozen encoder; crossed performance/family bootstrap |
| Stratum: pitch_register_gap_band = 6-12 semitones | Recall@1 0.4383; 2-way 95% CI [0.3397, 0.5462]; Recall@5 0.7068; MRR 0.5588 | 1419 episodes | frozen encoder; crossed performance/family bootstrap |
| Stratum: query_articulation = held | Recall@1 0.2548; 2-way 95% CI [0.1845, 0.3271]; Recall@5 0.4566; MRR 0.3588 | 3870 episodes | frozen encoder; crossed performance/family bootstrap |
| Stratum: query_articulation = phrased | Recall@1 0.4023; 2-way 95% CI [0.2942, 0.5081]; Recall@5 0.6209; MRR 0.5065 | 2580 episodes | frozen encoder; crossed performance/family bootstrap |
| Stratum: query_duration_band = 2-4s | Recall@1 0.2655; 2-way 95% CI [0.1240, 0.4322]; Recall@5 0.4864; MRR 0.3713 | 516 episodes | frozen encoder; crossed performance/family bootstrap |
| Stratum: query_duration_band = 4-6s | Recall@1 0.2614; 2-way 95% CI [0.1406, 0.3887]; Recall@5 0.4363; MRR 0.3534 | 903 episodes | frozen encoder; crossed performance/family bootstrap |
| Stratum: query_duration_band = 6-8s | Recall@1 0.3282; 2-way 95% CI [0.2502, 0.4083]; Recall@5 0.5414; MRR 0.4342 | 5031 episodes | frozen encoder; crossed performance/family bootstrap |
| Stratum: query_expression = expressive | Recall@1 0.1744; 2-way 95% CI [0.1163, 0.2326]; Recall@5 0.3721; MRR 0.2790 | 258 episodes | frozen encoder; crossed performance/family bootstrap |
| Stratum: query_expression = plain | Recall@1 0.3196; 2-way 95% CI [0.2497, 0.3947]; Recall@5 0.5286; MRR 0.4236 | 6192 episodes | frozen encoder; crossed performance/family bootstrap |
| Stratum: query_note_density_band = dense | Recall@1 0.2630; 2-way 95% CI [0.1783, 0.3628]; Recall@5 0.4749; MRR 0.3717 | 1935 episodes | frozen encoder; crossed performance/family bootstrap |
| Stratum: query_note_density_band = moderate | Recall@1 0.3705; 2-way 95% CI [0.2597, 0.4853]; Recall@5 0.5698; MRR 0.4691 | 2580 episodes | frozen encoder; crossed performance/family bootstrap |
| Stratum: query_note_density_band = sparse | Recall@1 0.2889; 2-way 95% CI [0.1917, 0.3907]; Recall@5 0.5065; MRR 0.3957 | 1935 episodes | frozen encoder; crossed performance/family bootstrap |
| Stratum: query_polyphony = monophonic | Recall@1 0.3130; 2-way 95% CI [0.2190, 0.4202]; Recall@5 0.5129; MRR 0.4137 | 3096 episodes | frozen encoder; crossed performance/family bootstrap |
| Stratum: query_polyphony = polyphonic | Recall@1 0.3145; 2-way 95% CI [0.2326, 0.4028]; Recall@5 0.5310; MRR 0.4216 | 3354 episodes | frozen encoder; crossed performance/family bootstrap |
| Stratum: query_render_quality = clean | Recall@1 0.2747; 2-way 95% CI [0.2013, 0.3494]; Recall@5 0.4839; MRR 0.3800 | 3778 episodes | frozen encoder; crossed performance/family bootstrap |
| Stratum: query_render_quality = warning | Recall@1 0.3690; 2-way 95% CI [0.2777, 0.4577]; Recall@5 0.5767; MRR 0.4713 | 2672 episodes | frozen encoder; crossed performance/family bootstrap |
| Stratum: source_preset_lineage_group_size = single-state | Recall@1 0.3138; 2-way 95% CI [0.2485, 0.3874]; Recall@5 0.5223; MRR 0.4178 | 6450 episodes | frozen encoder; crossed performance/family bootstrap |

Vital controls for this representation version:

| Vital control family | Enabled / exact instances | Vital output count | Update method / training signal |
|---|---:|---:|---|
| Global settings ❌ 0 / 562 | 0 | unchanged source preset; no head, optimization, loss, or gradient |
| Oscillators and voices ❌ 0 / 88 | 0 | unchanged source preset; no head, optimization, loss, or gradient |
| Filters ❌ 0 / 53 | 0 | unchanged source preset; no head, optimization, loss, or gradient |
| Envelopes ❌ 0 / 54 | 0 | unchanged source preset; no head, optimization, loss, or gradient |
| LFOs and modulation ❌ 0 / 170 | 0 | unchanged source preset; no head, optimization, loss, or gradient |
| Effects ❌ 0 / 46 | 0 | unchanged source preset; no head, optimization, loss, or gradient |
| Wavetables, curves and samples ❌ 0 / 6 | 0 | unchanged source preset; no head, optimization, loss, or gradient |
### Descript Audio Codec 44.1 kHz — temporal_pyramid_8

Checkpoint: `descript/dac_44khz` ([upstream checkpoint](https://huggingface.co/descript/dac_44khz)); requested sample rate: 44100 Hz; representation: continuous encoder output before RVQ; output size: 16384-D. Status: **executed**.

| Measure | Result or coverage | Output count | Update / supervision path |
|---|---:|---:|---|
| Checkpoint provenance / license | SHA-256 `128180a5e4d41ceabfeb952b687fb9da7cf468c2291acf5be9cadb5ab091cc46`; source `c1bc521685adf9cfe247bc39a5ca58917eda1ac4`; loader `transformers==4.50.0`; runtime `{"numpy": "1.26.4", "scipy": "1.17.1", "torch": "2.12.1+xpu", "transformers": "4.50.0"}`; MIT | — | exact local lock; frozen inference |
| Cross-MIDI common-gallery Recall@1 | 0.3068 [0.2660, 0.3479] | — | frozen backbone; cosine retrieval |
| Cross-MIDI common-gallery Recall@5 | 0.5290 [0.4860, 0.5718] | — | frozen backbone; cosine retrieval |
| Cross-MIDI common-gallery MRR | 0.4182 [0.3787, 0.4568] | — | frozen backbone; cosine retrieval |
| Cross-MIDI family-level Recall@1 | 0.3068 [0.2660, 0.3479] | — | frozen backbone; cosine retrieval |
| Cross-MIDI mixed-gallery Recall@1 | 0.3146 [0.2767, 0.3512] | — | frozen backbone; cosine retrieval |
| Same-MIDI other-patch negative win rate | 0.1757 [0.1388, 0.2119] (6450 episodes; 825600 negatives) | — | frozen backbone; cosine retrieval |
| Acoustic hard-negative top-5 win rate | 0.3650 [0.3242, 0.4043] (6450 episodes; 32250 negatives) | — | frozen backbone; cosine retrieval |
| Known source-lineage near-neighbor win rate | not run (0 episodes; 0 negatives) | — | frozen backbone; cosine retrieval |
| Gallery candidates / episodes | [129] / 6450 | — | fixed candidate sets across encoders |
| Features created / cache reused | 6426 / 54 | 6480 | resumable extraction pass; cache identity checked |
| Extraction pass time / peak RSS / accelerator memory | 8132.595 s / 3227729920 / 990163968 bytes | — | local inference and cache validation; no training |
| Resumable pass scope | 6426 new and 54 cache-reused features | 6480 | elapsed time is this pass, not cumulative across earlier partial runs |
| Stratum: duration_gap_band = 0-1s | Recall@1 0.2913; 2-way 95% CI [0.2087, 0.3795]; Recall@5 0.4976; MRR 0.3965 | 3354 episodes | frozen encoder; crossed performance/family bootstrap |
| Stratum: duration_gap_band = 1-3s | Recall@1 0.2735; 2-way 95% CI [0.1517, 0.4009]; Recall@5 0.4950; MRR 0.3824 | 903 episodes | frozen encoder; crossed performance/family bootstrap |
| Stratum: duration_gap_band = 3+s | Recall@1 0.3443; 2-way 95% CI [0.2599, 0.4273]; Recall@5 0.5910; MRR 0.4660 | 2193 episodes | frozen encoder; crossed performance/family bootstrap |
| Stratum: gallery_render_quality = warning | Recall@1 0.3068; 2-way 95% CI [0.2433, 0.3712]; Recall@5 0.5290; MRR 0.4182 | 6450 episodes | frozen encoder; crossed performance/family bootstrap |
| Stratum: pitch_register_gap_band = 0-5 semitones | Recall@1 0.4961; 2-way 95% CI [0.3147, 0.6806]; Recall@5 0.7488; MRR 0.6154 | 645 episodes | frozen encoder; crossed performance/family bootstrap |
| Stratum: pitch_register_gap_band = 13+ semitones | Recall@1 0.2417; 2-way 95% CI [0.1833, 0.3037]; Recall@5 0.4414; MRR 0.3471 | 4386 episodes | frozen encoder; crossed performance/family bootstrap |
| Stratum: pitch_register_gap_band = 6-12 semitones | Recall@1 0.4221; 2-way 95% CI [0.3206, 0.5201]; Recall@5 0.6998; MRR 0.5482 | 1419 episodes | frozen encoder; crossed performance/family bootstrap |
| Stratum: query_articulation = held | Recall@1 0.2778; 2-way 95% CI [0.2016, 0.3548]; Recall@5 0.4832; MRR 0.3826 | 3870 episodes | frozen encoder; crossed performance/family bootstrap |
| Stratum: query_articulation = phrased | Recall@1 0.3504; 2-way 95% CI [0.2756, 0.4233]; Recall@5 0.5977; MRR 0.4714 | 2580 episodes | frozen encoder; crossed performance/family bootstrap |
| Stratum: query_duration_band = 2-4s | Recall@1 0.3391; 2-way 95% CI [0.1705, 0.5039]; Recall@5 0.5640; MRR 0.4505 | 516 episodes | frozen encoder; crossed performance/family bootstrap |
| Stratum: query_duration_band = 4-6s | Recall@1 0.2780; 2-way 95% CI [0.1473, 0.4075]; Recall@5 0.4873; MRR 0.3829 | 903 episodes | frozen encoder; crossed performance/family bootstrap |
| Stratum: query_duration_band = 6-8s | Recall@1 0.3087; 2-way 95% CI [0.2415, 0.3755]; Recall@5 0.5329; MRR 0.4212 | 5031 episodes | frozen encoder; crossed performance/family bootstrap |
| Stratum: query_expression = expressive | Recall@1 0.1124; 2-way 95% CI [0.0581, 0.1744]; Recall@5 0.2946; MRR 0.2195 | 258 episodes | frozen encoder; crossed performance/family bootstrap |
| Stratum: query_expression = plain | Recall@1 0.3149; 2-way 95% CI [0.2529, 0.3763]; Recall@5 0.5388; MRR 0.4264 | 6192 episodes | frozen encoder; crossed performance/family bootstrap |
| Stratum: query_note_density_band = dense | Recall@1 0.2894; 2-way 95% CI [0.2021, 0.3881]; Recall@5 0.5013; MRR 0.3970 | 1935 episodes | frozen encoder; crossed performance/family bootstrap |
| Stratum: query_note_density_band = moderate | Recall@1 0.3275; 2-way 95% CI [0.2484, 0.4093]; Recall@5 0.5620; MRR 0.4432 | 2580 episodes | frozen encoder; crossed performance/family bootstrap |
| Stratum: query_note_density_band = sparse | Recall@1 0.2966; 2-way 95% CI [0.1907, 0.4031]; Recall@5 0.5127; MRR 0.4060 | 1935 episodes | frozen encoder; crossed performance/family bootstrap |
| Stratum: query_polyphony = monophonic | Recall@1 0.3062; 2-way 95% CI [0.2355, 0.3831]; Recall@5 0.5349; MRR 0.4181 | 3096 episodes | frozen encoder; crossed performance/family bootstrap |
| Stratum: query_polyphony = polyphonic | Recall@1 0.3074; 2-way 95% CI [0.2302, 0.3924]; Recall@5 0.5236; MRR 0.4182 | 3354 episodes | frozen encoder; crossed performance/family bootstrap |
| Stratum: query_render_quality = clean | Recall@1 0.2425; 2-way 95% CI [0.1834, 0.3076]; Recall@5 0.4656; MRR 0.3572 | 3778 episodes | frozen encoder; crossed performance/family bootstrap |
| Stratum: query_render_quality = warning | Recall@1 0.3978; 2-way 95% CI [0.3030, 0.4863]; Recall@5 0.6186; MRR 0.5043 | 2672 episodes | frozen encoder; crossed performance/family bootstrap |
| Stratum: source_preset_lineage_group_size = single-state | Recall@1 0.3068; 2-way 95% CI [0.2433, 0.3712]; Recall@5 0.5290; MRR 0.4182 | 6450 episodes | frozen encoder; crossed performance/family bootstrap |

Vital controls for this representation version:

| Vital control family | Enabled / exact instances | Vital output count | Update method / training signal |
|---|---:|---:|---|
| Global settings ❌ 0 / 562 | 0 | unchanged source preset; no head, optimization, loss, or gradient |
| Oscillators and voices ❌ 0 / 88 | 0 | unchanged source preset; no head, optimization, loss, or gradient |
| Filters ❌ 0 / 53 | 0 | unchanged source preset; no head, optimization, loss, or gradient |
| Envelopes ❌ 0 / 54 | 0 | unchanged source preset; no head, optimization, loss, or gradient |
| LFOs and modulation ❌ 0 / 170 | 0 | unchanged source preset; no head, optimization, loss, or gradient |
| Effects ❌ 0 / 46 | 0 | unchanged source preset; no head, optimization, loss, or gradient |
| Wavetables, curves and samples ❌ 0 / 6 | 0 | unchanged source preset; no head, optimization, loss, or gradient |
## Evaluation protocol

The primary condition uses one common, existing gallery MIDI per episode and a query render of the same target preset on another existing MIDI. Query/gallery MIDI IDs differ; selection rotates deterministically. Every admitted `validation` preset is paired with all 50 compatible MIDI performances from `train`, making multiple cross-renders per preset. Gallery candidates are all admitted source states. The mixed-gallery condition uses a deterministic candidate-specific MIDI assignment and is reported separately. A same-MIDI different-patch negative check compares each cross-MIDI positive against other patches rendered on the query MIDI. Known frozen lineage groups are reported as family-level retrieval; confidence intervals resample family, performance identity, and crossed family/performance units.

Metrics use cosine similarity over FP32 cached vectors; candidate sets and query episodes are paired across available arms. Acoustic hard negatives are selected independently of the tested encoders using cosine similarity over normalized 64-bin log-frequency power spectra of the target query and same-MIDI candidate renders; each encoder’s cross-MIDI positive is then compared against the five nearest such negatives on that query MIDI. The report includes exact-state and family-level Recall@1, Recall@5, MRR, family-cluster 95% bootstrap intervals, MIDI-only metadata and random baselines, and strata for polyphony, articulation, expression, source-lineage group size, pitch gap, duration, and note density. Sequence outputs include a valid-frame mean and an 8-bin ordered mean/std pyramid. No pooling or similarity setting is selected on reserved test labels.

This benchmark uses new Vital renders only; source WAVs are not used as model inputs or mixed with renders.
