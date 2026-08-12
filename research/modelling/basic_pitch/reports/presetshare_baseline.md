# PresetShare Basic Pitch baseline

## Corpus pairing

- Layout: `direct child directory with exact one-to-one extensions`.
- Candidate directories: `9620`.
- Eligible pairs: `1769`.
- Excluded candidates: `7837`; ambiguous: `14`.
- Exclusions by reason: `pair.ambiguous`=14; `pair.derived_render_failed`=1; `pair.empty_reference`=75; `pair.invalid_midi`=23; `pair.missing_midi`=7738.
- Pairing methods: `same_directory_exact_one_midi_derived_render`.
- Pair identity uses the observed direct-directory relationship with exactly one MIDI and one audio file; no fuzzy matching is used.
- Derived-render opt-in: `True`. Any derived audio is labeled separately and remains ignored local output.
- Existing audio remains read-only; derived audio uses the parent-approved Vital/Pedalboard path and is never described as historical source audio.
- Private source-stat records: `11518`; mismatches: `0`.

## Runtime provenance and #24 route decision

- Exact #24 decision consumed: backend `pytorch_xpu`; device `xpu:0`; boundary `end_to_end_audio_to_note_event`; precision `float32`.
- Runtime-selection source: `#24 corpus_inference_decision`.
- Selection rule: highest median end-to-end audio-seconds/wall-second among successful parity-safe inference routes.
- Supporting #24 measurements: batch-1 model-call `225.8292173388524` audio-seconds/second; end-to-end `95.72711839188128` audio-seconds/wall-second.
- Supporting decision identity matched the #24 report and recorded evaluator contract: `True`.
- Consistency assessment: `exact_issue_24_decision_consumed`.
- Interpretation: The full-corpus quality result is attributed to the exact #24-selected runtime. It does not establish quality equivalence for any other backend.
- The full-corpus result is not relabeled as another backend and no quality equivalence is inferred for routes not executed here.

## Evaluation status

- Status: `ok`.
- Failure code: `none`.
- The report distinguishes unavailable execution from measured quality; unavailable rows are not treated as zero-quality predictions.

## Overall quality

| Metric | Reference | Predicted | TP | FP | FN | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Onset + pitch | 146842 | 183249 | 45907 | 137342 | 100935 | 0.278148 |
| Onset + pitch + offset | 146842 | 183249 | 16220 | 167029 | 130622 | 0.098276 |
| Frame | 5811924 | 6281996 | 2413241 | 3868755 | 3398683 | 0.399083 |

- Pair coverage: `1769/1769`.
- Micro metrics are derived from total counts. Pair-macro values and preset-cluster bootstrap intervals remain separate.

## Uncertainty and support

- Preset-cluster bootstrap: `10000` replicates, seed `0`, `1769` clusters.

| Metric | F1 | Bootstrap 95% interval |
| --- | ---: | ---: |
| `onset_pitch` | 0.278148 | 0.255319 - 0.300418 |
| `onset_pitch_offset` | 0.098276 | 0.087249 - 0.110546 |
| `frames` | 0.399083 | 0.377048 - 0.420062 |

## Category summaries

The committed report retains counts and support for objective MIDI categories and explicit source metadata categories. Unknown metadata remains unknown; no style labels are inferred from filenames.

## Category interpretation

The following statements use onset+pitch F1 and retain category support. `well_supported` means at least 100 pairs, `moderately_supported` 30-99, and `small_subset` fewer than 30; small-subset extremes are descriptive, not robust corpus-wide findings.

- `duration_class`: highest `medium` `0.333236` (388 pairs, `well_supported`); lowest `short` `0.165138` (13 pairs, `small_subset`). The well-supported range is `0.273903`-`0.333236` across `long` to `medium`.
- `note_density_class`: highest `high` `0.328499` (222 pairs, `well_supported`); lowest `low` `0.102055` (512 pairs, `well_supported`). The well-supported range is `0.102055`-`0.328499` across `low` to `high`.
- `pitch_register_class`: highest `mid` `0.290658` (1292 pairs, `well_supported`); lowest `low` `0.173113` (322 pairs, `well_supported`). The well-supported range is `0.173113`-`0.290658` across `low` to `mid`.
- `polyphony_class`: `polyphonic` `0.278413` vs `monophonic` `0.277012`; near tie across `1150` and `619` pairs. Frame behavior should be read separately from event F1.
- `type`: the overall high is `Sub` `0.727273` on `2` pairs (`small_subset`); among well-supported types, `Pluck` is highest at `0.509789` on `107` pairs, while `Pad` is lowest at `0.073721` on `161` pairs. This separates robust patterns from tiny type strata.

The category tables below retain every observed stratum so these summaries can be checked against the underlying aggregate counts.

### duration_class

| Category | Pairs | Successful | Coverage | Onset + pitch F1 |
| --- | ---: | ---: | ---: | ---: |
| `long` | 1368 | 1368 | 1.000 | 0.273903 |
| `medium` | 388 | 388 | 1.000 | 0.333236 |
| `short` | 13 | 13 | 1.000 | 0.165138 |

### genre

| Category | Pairs | Successful | Coverage | Onset + pitch F1 |
| --- | ---: | ---: | ---: | ---: |
| `Ambient` | 99 | 99 | 1.000 | 0.221069 |
| `Bass House` | 21 | 21 | 1.000 | 0.186881 |
| `Breakbeat / Breaks` | 13 | 13 | 1.000 | 0.156098 |
| `Cinematic` | 84 | 84 | 1.000 | 0.130666 |
| `Drum and Bass` | 148 | 148 | 1.000 | 0.165130 |
| `Dubstep` | 67 | 67 | 1.000 | 0.145444 |
| `Future Bass` | 46 | 46 | 1.000 | 0.219750 |
| `Halfstep` | 1 | 1 | 1.000 | 0.000000 |
| `Hardcore / Hardstyle` | 143 | 143 | 1.000 | 0.309296 |
| `Hip-Hop / R&B` | 92 | 92 | 1.000 | 0.348684 |
| `House` | 113 | 113 | 1.000 | 0.378929 |
| `Industrial` | 24 | 24 | 1.000 | 0.097712 |
| `Midtempo` | 50 | 50 | 1.000 | 0.155595 |
| `Moombahton / Reggae` | 5 | 5 | 1.000 | 0.254601 |
| `Multigenre` | 443 | 443 | 1.000 | 0.313160 |
| `Other` | 149 | 149 | 1.000 | 0.318157 |
| `Psytrance` | 15 | 15 | 1.000 | 0.364267 |
| `Synthwave` | 108 | 108 | 1.000 | 0.305947 |
| `Tech House` | 13 | 13 | 1.000 | 0.368159 |
| `Techno` | 67 | 67 | 1.000 | 0.276490 |
| `Trance` | 40 | 40 | 1.000 | 0.330598 |
| `Trap` | 22 | 22 | 1.000 | 0.328426 |
| `UK Garage` | 6 | 6 | 1.000 | 0.223947 |

### instrument

| Category | Pairs | Successful | Coverage | Onset + pitch F1 |
| --- | ---: | ---: | ---: | ---: |
| `Vital` | 1769 | 1769 | 1.000 | 0.278148 |

### note_density_class

| Category | Pairs | Successful | Coverage | Onset + pitch F1 |
| --- | ---: | ---: | ---: | ---: |
| `high` | 222 | 222 | 1.000 | 0.328499 |
| `low` | 512 | 512 | 1.000 | 0.102055 |
| `medium` | 1035 | 1035 | 1.000 | 0.307027 |

### pitch_register_class

| Category | Pairs | Successful | Coverage | Onset + pitch F1 |
| --- | ---: | ---: | ---: | ---: |
| `high` | 155 | 155 | 1.000 | 0.281268 |
| `low` | 322 | 322 | 1.000 | 0.173113 |
| `mid` | 1292 | 1292 | 1.000 | 0.290658 |

### polyphony_class

| Category | Pairs | Successful | Coverage | Onset + pitch F1 |
| --- | ---: | ---: | ---: | ---: |
| `monophonic` | 619 | 619 | 1.000 | 0.277012 |
| `polyphonic` | 1150 | 1150 | 1.000 | 0.278413 |

### type

| Category | Pairs | Successful | Coverage | Onset + pitch F1 |
| --- | ---: | ---: | ---: | ---: |
| `Arp` | 25 | 25 | 1.000 | 0.330997 |
| `Atmosphere` | 27 | 27 | 1.000 | 0.102306 |
| `Bass` | 446 | 446 | 1.000 | 0.115646 |
| `Chord` | 32 | 32 | 1.000 | 0.190594 |
| `Drone` | 4 | 4 | 1.000 | 0.039106 |
| `Drums` | 45 | 45 | 1.000 | 0.001230 |
| `FX` | 39 | 39 | 1.000 | 0.044815 |
| `Keys` | 281 | 281 | 1.000 | 0.364495 |
| `Lead` | 393 | 393 | 1.000 | 0.380731 |
| `Miscellaneous` | 16 | 16 | 1.000 | 0.187441 |
| `Other` | 18 | 18 | 1.000 | 0.047305 |
| `Pad` | 161 | 161 | 1.000 | 0.073721 |
| `Pluck` | 107 | 107 | 1.000 | 0.509789 |
| `Reese` | 16 | 16 | 1.000 | 0.020888 |
| `Seq` | 42 | 42 | 1.000 | 0.093319 |
| `Stab` | 17 | 17 | 1.000 | 0.312152 |
| `Sub` | 2 | 2 | 1.000 | 0.727273 |
| `Synth` | 95 | 95 | 1.000 | 0.255692 |
| `Vox` | 3 | 3 | 1.000 | 0.015238 |

### vital_style

| Category | Pairs | Successful | Coverage | Onset + pitch F1 |
| --- | ---: | ---: | ---: | ---: |
| `unknown` | 1769 | 1769 | 1.000 | 0.278148 |

## Failure analysis

- Onset + pitch false negatives: `100935`; false positives: `137342`.
- Additional offset false negatives: `29687`.
- Assigned near-onset pitch errors: `16170`; octave errors: `11418`; diagnostically unassigned residual references/predictions: `205937`.
- Pair-level failures and private best/worst rows remain under ignored local outputs. This committed view contains only aggregate coverage and stable exclusion counts.

## Interpretation

Aggregate values are computed from successful pair results using fixed stock Basic Pitch decoding and no threshold tuning.

Quality describes the frozen Basic Pitch performance prior; it is not a claim about source audio reconstruction or an OBRUXO training objective.

## Provenance and limits

- Model: `spotify-basic-pitch-icassp-2022-v0.4.0`.
- Backend: `pytorch_xpu` on `xpu:0`; precision: `float32`.
- Stock settings are onset threshold 0.5, frame threshold 0.3, minimum note length 11 frames, inferred onsets enabled, Melodia fallback enabled, and no frequency limits.
- No composite score is used and no upstream model/runtime setting is tuned from corpus results.
