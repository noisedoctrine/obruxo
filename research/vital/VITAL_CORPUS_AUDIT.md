# Vital Corpus Audit Summary

This is the sanitized, tracked summary of the Vital preset corpus audit used
to check revision drift in `PRESET_SCHEMA.md`. The raw corpus is external and
the generated per-file audit is intentionally not tracked because it contains
source-file inventory and embedded preset payload metadata.

The audit was run with [`build_vital_corpus_audit.py`](build_vital_corpus_audit.py)
(script version `1.0.0`) over 9,620 discovered `.vital` files: 9,618 parsed
successfully and two were malformed. The audit did not retain embedded sample
or wavetable payloads in this summary.

## Version and shape distribution

“Scalar settings” means numeric top-level values in `settings`. “Total
settings keys” includes every top-level `settings` key, including nested
objects and arrays such as `sample`, `wavetables`, `lfos`, `modulations`,
`custom_warps`, and `random_values`.

| `synth_version` | Files | Scalar settings | Total `settings` keys |
|---|---:|---:|---:|
| 1.0.0 | 1 | 771 | 775 |
| 1.0.2 | 17 | 771 | 775 |
| 1.0.3 | 220 | 771 | 775 |
| 1.0.4 | 12 | 771 | 775 |
| 1.0.5 | 47 | 772 | 776 |
| 1.0.7 | 1,500 | 772 | 776 |
| 1.0.7 (`flanger_depth` extension) | 2 | 773 | 777 |
| 1.0.8 | 302 | 772 | 776 |
| 1.5.1 | 146 | 775 | 781 |
| 1.5.3 | 445 | 775 | 781 |
| 1.5.4 | 58 | 775 | 781 |
| 1.5.5 | 6,581 | 775 | 781 |
| 1.6.0 | 30 | 903 | 909 |
| 1.6.1 | 112 | 903 | 909 |
| 1.6.2 | 3 | 903 | 909 |
| 1.6.4 | 142 | 903 | 909 |

No 1.0.1 files were present in the discovered corpus. The two malformed
files are excluded from the version table. The two 1.0.7 extensions were
`Action Pluck_p8182/Action_Pluck.vital` and
`Helpfind preset called DROPLET_p15742/Helpfind_preset_called_DROPLET.vital`;
both add only `flanger_depth` to the 772-scalar baseline.

## Interpreted revision changes

- Versions 1.5.1–1.5.5 add three scalar fields:
  `osc_1_spectral_morph_phase`, `osc_2_spectral_morph_phase`, and
  `osc_3_spectral_morph_phase`. They also add the array-valued fields
  `custom_warps` and `random_values`; those arrays are not scalar controls.
- Versions 1.6.x add 128 scalar fields:
  `modulation_<n>_ramp_up` and `modulation_<n>_ramp_down` for each of 64
  modulation slots.
- Every parsed preset has three wavetables, eight LFO shape objects, and 64
  modulation connection slots. The sample object has either four fields
  (`length`, `name`, `sample_rate`, `samples`) or an additional
  `samples_stereo` field.
- The modulation vocabulary audit observed all 32 registered non-empty
  sources and 366 of the 428 registered destinations. The three observed
  destination names absent from the pinned source vocabulary are the spectral
  morph phase fields listed above; the remaining difference is expected
  source vocabulary that was not used by this corpus.

## Reproduction and limits

The source revision used for the schema reconciliation is
[`mtytel/vital@636ca0ef`](https://github.com/mtytel/vital/commit/636ca0ef517a4db087a6a08a6a8a5e704e21f836).
The executable 772-parameter bundle that this summary cross-checks is under
[`research/data_generation/obruxo_data/vital/schema/vital-1.0.8-vita-0.1.0/`](../data_generation/obruxo_data/vital/schema/vital-1.0.8-vita-0.1.0/).
Re-running the audit requires the external corpus; this file records the
result and the counting definitions without committing that corpus or its raw
audit archive.
