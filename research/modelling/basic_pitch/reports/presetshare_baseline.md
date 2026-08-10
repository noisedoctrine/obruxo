# PresetShare Basic Pitch baseline

## Corpus pairing

- Layout: `direct child directory with exact one-to-one extensions`.
- Candidate directories: `9620`.
- Eligible pairs: `0`.
- Excluded candidates: `9606`; ambiguous: `14`.
- Pairing methods: `same_directory_exact_one_midi_derived_render`.
- Pair identity uses the observed direct-directory relationship with exactly one MIDI and one audio file; no fuzzy matching is used.
- Derived-render opt-in: `True`. Any derived audio is labeled separately and remains ignored local output.
- Private source-stat records: `11518`; mismatches: `0`.

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

## Category summaries

The committed report retains counts and support for objective MIDI categories and explicit source metadata categories. Unknown metadata remains unknown; no style labels are inferred from filenames.

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
