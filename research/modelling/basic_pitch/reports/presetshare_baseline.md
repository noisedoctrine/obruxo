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

- Recorded #25 corpus backend: `pytorch_cpu`; boundary: `#24_end_to_end_audio_to_note_events_batch_1`; precision: `float32`.
- Runtime-selection source: the backend contract recorded by the existing #25 run.
- Selection rationale recorded by the artifacts: The existing #25 run fixed pytorch_cpu in its backend contract. Its artifacts do not record why that route was selected; current #24 measurements identify pytorch_xpu as the batch-1 model-call throughput leader and openvino_gpu as the warmed end-to-end leader. This report does not relabel the existing quality result or infer alternate-backend quality.
- Existing #24 report's highest batch-1 inference route: `pytorch_xpu` at `217.87797768889692` audio-seconds/second.
- Existing #24 report's highest warmed end-to-end route: `openvino_gpu` at `98.26220356768816` audio-seconds/wall-second.
- Consistency assessment: `recorded_backend_does_not_match_issue_24_observed_leader`.
- Interpretation: The existing #25 quality run records a backend different from one or both current #24 leaders. Its quality result remains attributed only to the recorded backend; no alternate-backend full-corpus quality or equivalence claim is made.
- This report revision does not rerun the corpus evaluation. The backend mismatch is surfaced for review rather than silently reassigning the existing F1 result to XPU.

## Evaluation status

- Status: `ok`.
- Failure code: `none`.
- The report distinguishes unavailable execution from measured quality; unavailable rows are not treated as zero-quality predictions.

## Overall quality

| Metric | Reference | Predicted | TP | FP | FN | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Onset + pitch | 146842 | 183299 | 45917 | 137382 | 100925 | 0.278166 |
| Onset + pitch + offset | 146842 | 183299 | 16242 | 167057 | 130600 | 0.098394 |
| Frame | 5811924 | 6290580 | 2413242 | 3877338 | 3398682 | 0.398800 |

- Pair coverage: `1769/1769`.
- Micro metrics are derived from total counts. Pair-macro values and preset-cluster bootstrap intervals remain separate.

## Uncertainty and support

- Preset-cluster bootstrap: `10000` replicates, seed `0`, `1769` clusters.

| Metric | F1 | Bootstrap 95% interval |
| --- | ---: | ---: |
| `onset_pitch` | 0.278166 | 0.255365 - 0.300425 |
| `onset_pitch_offset` | 0.098394 | 0.087394 - 0.110627 |
| `frames` | 0.398800 | 0.376780 - 0.419742 |

## Category summaries

The committed report retains counts and support for objective MIDI categories and explicit source metadata categories. Unknown metadata remains unknown; no style labels are inferred from filenames.

## Category interpretation

The following statements use onset+pitch F1 and retain category support. `well_supported` means at least 100 pairs, `moderately_supported` 30-99, and `small_subset` fewer than 30; small-subset extremes are descriptive, not robust corpus-wide findings.

- `duration_class`: highest `medium` `0.333264` (388 pairs, `well_supported`); lowest `short` `0.170616` (13 pairs, `small_subset`). The well-supported range is `0.273906`-`0.333264` across `long` to `medium`.
- `note_density_class`: highest `high` `0.328722` (222 pairs, `well_supported`); lowest `low` `0.102239` (512 pairs, `well_supported`). The well-supported range is `0.102239`-`0.328722` across `low` to `high`.
- `pitch_register_class`: highest `mid` `0.290710` (1292 pairs, `well_supported`); lowest `low` `0.173343` (322 pairs, `well_supported`). The well-supported range is `0.173343`-`0.290710` across `low` to `mid`.
- `polyphony_class`: `polyphonic` `0.278429` vs `monophonic` `0.277039`; near tie across `1150` and `619` pairs. Frame behavior should be read separately from event F1.
- `type`: the overall high is `Sub` `0.738462` on `2` pairs (`small_subset`); among well-supported types, `Pluck` is highest at `0.509418` on `107` pairs, while `Pad` is lowest at `0.073700` on `161` pairs. This separates robust patterns from tiny type strata.

The category tables below retain every observed stratum so these summaries can be checked against the underlying aggregate counts.

### duration_class

| Category | Pairs | Successful | Coverage | Onset + pitch F1 |
| --- | ---: | ---: | ---: | ---: |
| `long` | 1368 | 1368 | 1.000 | 0.273906 |
| `medium` | 388 | 388 | 1.000 | 0.333264 |
| `short` | 13 | 13 | 1.000 | 0.170616 |

### genre

| Category | Pairs | Successful | Coverage | Onset + pitch F1 |
| --- | ---: | ---: | ---: | ---: |
| `Ambient` | 99 | 99 | 1.000 | 0.221013 |
| `Bass House` | 21 | 21 | 1.000 | 0.185874 |
| `Breakbeat / Breaks` | 13 | 13 | 1.000 | 0.156098 |
| `Cinematic` | 84 | 84 | 1.000 | 0.130470 |
| `Drum and Bass` | 148 | 148 | 1.000 | 0.165337 |
| `Dubstep` | 67 | 67 | 1.000 | 0.147802 |
| `Future Bass` | 46 | 46 | 1.000 | 0.219482 |
| `Halfstep` | 1 | 1 | 1.000 | 0.000000 |
| `Hardcore / Hardstyle` | 143 | 143 | 1.000 | 0.309131 |
| `Hip-Hop / R&B` | 92 | 92 | 1.000 | 0.348116 |
| `House` | 113 | 113 | 1.000 | 0.378773 |
| `Industrial` | 24 | 24 | 1.000 | 0.097666 |
| `Midtempo` | 50 | 50 | 1.000 | 0.155497 |
| `Moombahton / Reggae` | 5 | 5 | 1.000 | 0.254211 |
| `Multigenre` | 443 | 443 | 1.000 | 0.313369 |
| `Other` | 149 | 149 | 1.000 | 0.318034 |
| `Psytrance` | 15 | 15 | 1.000 | 0.364742 |
| `Synthwave` | 108 | 108 | 1.000 | 0.305881 |
| `Tech House` | 13 | 13 | 1.000 | 0.366812 |
| `Techno` | 67 | 67 | 1.000 | 0.276560 |
| `Trance` | 40 | 40 | 1.000 | 0.330791 |
| `Trap` | 22 | 22 | 1.000 | 0.327437 |
| `UK Garage` | 6 | 6 | 1.000 | 0.225166 |

### instrument

| Category | Pairs | Successful | Coverage | Onset + pitch F1 |
| --- | ---: | ---: | ---: | ---: |
| `Vital` | 1769 | 1769 | 1.000 | 0.278166 |

### note_density_class

| Category | Pairs | Successful | Coverage | Onset + pitch F1 |
| --- | ---: | ---: | ---: | ---: |
| `high` | 222 | 222 | 1.000 | 0.328722 |
| `low` | 512 | 512 | 1.000 | 0.102239 |
| `medium` | 1035 | 1035 | 1.000 | 0.306856 |

### pitch_register_class

| Category | Pairs | Successful | Coverage | Onset + pitch F1 |
| --- | ---: | ---: | ---: | ---: |
| `high` | 155 | 155 | 1.000 | 0.280760 |
| `low` | 322 | 322 | 1.000 | 0.173343 |
| `mid` | 1292 | 1292 | 1.000 | 0.290710 |

### polyphony_class

| Category | Pairs | Successful | Coverage | Onset + pitch F1 |
| --- | ---: | ---: | ---: | ---: |
| `monophonic` | 619 | 619 | 1.000 | 0.277039 |
| `polyphonic` | 1150 | 1150 | 1.000 | 0.278429 |

### type

| Category | Pairs | Successful | Coverage | Onset + pitch F1 |
| --- | ---: | ---: | ---: | ---: |
| `Arp` | 25 | 25 | 1.000 | 0.331238 |
| `Atmosphere` | 27 | 27 | 1.000 | 0.102285 |
| `Bass` | 446 | 446 | 1.000 | 0.115803 |
| `Chord` | 32 | 32 | 1.000 | 0.190337 |
| `Drone` | 4 | 4 | 1.000 | 0.038997 |
| `Drums` | 45 | 45 | 1.000 | 0.001229 |
| `FX` | 39 | 39 | 1.000 | 0.044773 |
| `Keys` | 281 | 281 | 1.000 | 0.364452 |
| `Lead` | 393 | 393 | 1.000 | 0.380835 |
| `Miscellaneous` | 16 | 16 | 1.000 | 0.187705 |
| `Other` | 18 | 18 | 1.000 | 0.047409 |
| `Pad` | 161 | 161 | 1.000 | 0.073700 |
| `Pluck` | 107 | 107 | 1.000 | 0.509418 |
| `Reese` | 16 | 16 | 1.000 | 0.020833 |
| `Seq` | 42 | 42 | 1.000 | 0.093289 |
| `Stab` | 17 | 17 | 1.000 | 0.312674 |
| `Sub` | 2 | 2 | 1.000 | 0.738462 |
| `Synth` | 95 | 95 | 1.000 | 0.255664 |
| `Vox` | 3 | 3 | 1.000 | 0.015355 |

### vital_style

| Category | Pairs | Successful | Coverage | Onset + pitch F1 |
| --- | ---: | ---: | ---: | ---: |
| `unknown` | 1769 | 1769 | 1.000 | 0.278166 |

## Failure analysis

- Onset + pitch false negatives: `100925`; false positives: `137382`.
- Additional offset false negatives: `29675`.
- Assigned near-onset pitch errors: `16167`; octave errors: `11418`; ambiguous/unassigned: `24954`.
- Pair-level failures and private best/worst rows remain under ignored local outputs. This committed view contains only aggregate coverage and stable exclusion counts.

## Interpretation

Aggregate values are computed from successful pair results using fixed stock Basic Pitch decoding and no threshold tuning.

Quality describes the frozen Basic Pitch performance prior; it is not a claim about source audio reconstruction or an OBRUXO training objective.

## Provenance and limits

- Model: `spotify-basic-pitch-icassp-2022-v0.4.0`.
- Backend: `pytorch_cpu`; precision: `float32`.
- Stock settings are onset threshold 0.5, frame threshold 0.3, minimum note length 11 frames, inferred onsets enabled, Melodia fallback enabled, and no frequency limits.
- No composite score is used and no upstream model/runtime setting is tuned from corpus results.
