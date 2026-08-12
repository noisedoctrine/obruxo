# PresetShare Basic Pitch baseline

## Corpus pairing

- Layout: `direct child directory with exact one-to-one extensions`.
- Candidate directories: `9620`.
- Eligible pairs: `18`.
- Excluded candidates: `9588`; ambiguous: `14`.
- Exclusions by reason: `pair.ambiguous`=14; `pair.derived_render_unavailable`=1752; `pair.empty_reference`=75; `pair.invalid_midi`=23; `pair.missing_midi`=7738.
- Pairing methods: `same_directory_exact_one_midi_derived_render, same_directory_exact_one_midi_one_audio`.
- Pair identity uses the observed direct-directory relationship with exactly one MIDI and one audio file; no fuzzy matching is used.
- Derived-render opt-in: `True`. Any derived audio is labeled separately and remains ignored local output.
- Existing audio remains read-only; derived audio uses the parent-approved Vital/Pedalboard path and is never described as historical source audio.
- Private source-stat records: `11518`; mismatches: `0`.

## Runtime provenance and #24 route decision

- Recorded #25 corpus backend: `pytorch_cpu`; boundary: `#24_end_to_end_audio_to_note_events_batch_1`; precision: `float32`.
- Runtime-selection source: the backend contract recorded by the existing #25 run.
- Selection rationale recorded by the artifacts: The existing #25 run fixed pytorch_cpu in its backend contract. Its artifacts do not record why that route was selected; current #24 measurements identify pytorch_xpu as the batch-1 model-call throughput leader and openvino_gpu as the warmed end-to-end leader. This report does not relabel the existing quality result or infer alternate-backend quality..
- Existing #24 report's highest batch-1 inference route: `pytorch_xpu` at `217.87797768889692` audio-seconds/second.
- Existing #24 report's highest warmed end-to-end route: `openvino_gpu` at `98.26220356768816` audio-seconds/wall-second.
- Consistency assessment: `recorded_backend_does_not_match_issue_24_observed_leader`.
- Interpretation: The existing #25 quality run records a backend different from one or both current #24 leaders. Its quality result remains attributed only to the recorded backend; no alternate-backend full-corpus quality or equivalence claim is made.
- This report revision does not rerun the corpus evaluation. The backend mismatch is surfaced for review rather than silently reassigning the existing F1 result to XPU.

## Evaluation status

- Status: `unavailable`.
- Failure code: `no_eligible_pairs`.
- The report distinguishes unavailable execution from measured quality; unavailable rows are not treated as zero-quality predictions.

## Overall quality

| Metric | Reference | Predicted | TP | FP | FN | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Onset + pitch | 0 | 0 | 0 | 0 | 0 | n/a |
| Onset + pitch + offset | 0 | 0 | 0 | 0 | 0 | n/a |
| Frame | 0 | 0 | 0 | 0 | 0 | n/a |

- Pair coverage: `0/0`.
- Micro metrics are derived from total counts. Pair-macro values and preset-cluster bootstrap intervals remain separate.

## Uncertainty and support

- Preset-cluster bootstrap: `10000` replicates, seed `0`, `0` clusters.

| Metric | F1 | Bootstrap 95% interval |
| --- | ---: | ---: |
| `onset_pitch` | n/a | n/a |
| `onset_pitch_offset` | n/a | n/a |
| `frames` | n/a | n/a |

## Category summaries

The committed report retains counts and support for objective MIDI categories and explicit source metadata categories. Unknown metadata remains unknown; no style labels are inferred from filenames.

## Category interpretation

The following statements use onset+pitch F1 and retain category support. `well_supported` means at least 100 pairs, `moderately_supported` 30-99, and `small_subset` fewer than 30; small-subset extremes are descriptive, not robust corpus-wide findings.


The category tables below retain every observed stratum so these summaries can be checked against the underlying aggregate counts.

## Failure analysis

- Onset + pitch false negatives: `0`; false positives: `0`.
- Additional offset false negatives: `0`.
- Assigned near-onset pitch errors: `0`; octave errors: `0`; ambiguous/unassigned: `0`.
- Pair-level failures and private best/worst rows remain under ignored local outputs. This committed view contains only aggregate coverage and stable exclusion counts.

## Interpretation

Aggregate values are computed from successful pair results using fixed stock Basic Pitch decoding and no threshold tuning.

Quality describes the frozen Basic Pitch performance prior; it is not a claim about source audio reconstruction or an OBRUXO training objective.

## Provenance and limits

- Model: `spotify-basic-pitch-icassp-2022-v0.4.0`.
- Backend: `pytorch_cpu`; precision: `float32`.
- Stock settings are onset threshold 0.5, frame threshold 0.3, minimum note length 11 frames, inferred onsets enabled, Melodia fallback enabled, and no frequency limits.
- No composite score is used and no upstream model/runtime setting is tuned from corpus results.
