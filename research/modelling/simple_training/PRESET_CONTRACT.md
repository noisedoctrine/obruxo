# What makes an exported preset valid?

The separate [full 1.5.5 contract](../../vital/FULL_CONTRACT_1_5_5.md) now covers all
775 scalar fields and six nested structures. This page describes the experiment's
narrower template contract; it remains the guard on the current two-control head.

The decoder must either produce a preset that satisfies a specific contract or
refuse to export. Valid JSON alone is insufficient: a file can parse correctly
yet contain impossible controls, damaged wavetable data, or an unsupported version.

For this experiment, we export a **1.5.5-labelled, oscillator-only subset** and
verify it in the installed **Vital 1.6.4** plugin. The model can change only the
two controls below. It cannot write metadata, routing, envelopes or embedded assets.

| Control | Prediction | Mapping into Vital |
| --- | --- | --- |
| `osc_1_transpose` | Three logits (unnormalized scores) | Highest score selects `-12`, `0`, or `12` semitones |
| `osc_1_level` | One logit | `0.2 + 0.6 * sigmoid(logit)`, in raw Vital units |

Tied octave scores select the first category. Wrong-sized, nonfinite, boolean or
complex predictions are rejected. For accepted finite predictions, the mapping
can only produce one of the three legal octaves and a level between 0.2 and 0.8.
Every other value comes from the pinned, tested template.

## Why this version and subset?

The existing [PresetShare corpus audit](../../vital/VITAL_CORPUS_AUDIT.md) found
6,581 version-1.5.5 files among 9,618 parsed presets: **68.4%**, the largest group.
It also found different control inventories across versions. We therefore do not
combine all observed keys into an assumed universal schema.

| Contract component | Decision |
| --- | --- |
| Exported preset and wavetable version headers | `1.5.5`, chosen from corpus frequency |
| Supplied control inventory | The existing 772-scalar, source-reconciled baseline |
| Predicted controls | Only oscillator 1 octave and level |
| All other supplied fields | Exact match to the pinned template, including metadata and payloads |
| Verified loader | Official Vital `1.6.4`, with the repository's accepted binary fingerprint |
| Omitted later controls | Allowed only because their default filling was measured for this subset |

This is a **loadable subset**, not the complete 1.5.5 serializer output. The corpus
shows full 1.5.5 presets usually have 775 scalars and two extra arrays. Our 772
supplied scalars use the older reconciled inventory; the unused newer settings
are left to the verified loader.

The native compatibility probe loaded the same authored subset with `1.0.8`,
`1.5.5` and `1.6.4` headers in Vital 1.6.4. In all three cases, every supplied
scalar survived readback within float32 tolerance. The plugin supplied **133
additional settings**, all equal to its initial defaults: 128 modulation ramp
controls, three spectral-morph phase controls, and the `custom_warps` and
`random_values` arrays. The requested lower octave rendered at 130 Hz, as expected.

This tests the installed 1.6.4 loader. It does not claim that a 1.5.5 binary or
every future release has been tested. A new loader, new predicted controls or new
assets needs a new compatibility review.

## Validation gates

| Gate | What is rejected |
| --- | --- |
| Strict JSON parsing | Syntax errors, duplicate keys, invalid UTF-8, `NaN`/`Infinity`, oversized files |
| Structure and types | Missing/extra keys, wrong array lengths, booleans or strings used as controls |
| Supported controls | Unsupported octave values, fractional octave ordinals, out-of-range levels, nonfinite numbers |
| Fixed-state comparison | Any changed effect, route, metadata, LFO shape, wavetable or sample payload |
| Template identity | A template whose SHA-256 differs from the versioned contract |
| Vita round-trip, optional | Loading failure or changes outside explicitly classified numeric/header/sample re-encoding |
| Official Vital check, optional | Load/readback mismatch, nonfinite/silent audio or incorrect octave |

The machine-readable definition is [preset_contract.json](preset_contract.json).
The validator is separate from the neural network in
[preset_contract.py](preset_contract.py). It compares the full document against
the pinned template, allowing only the two declared control values to differ.
That closes the generic validator's gaps around nested assets and metadata.

All training and inference exports use this validator automatically. Export first
checks the document, then serializes with nonfinite JSON values disabled, writes a
temporary file, and strictly validates those exact bytes. Only then is the final
file published. Existing files are not overwritten, and failed checks leave no
destination file. Publication uses a same-directory hard link; a filesystem that
does not support it fails explicitly rather than falling back to a partial write.

The native renderer additionally saves the plugin's loaded state and checks every
supplied scalar against the request. This catches a plugin silently ignoring the
preset even when it produces non-silent audio. It accepts only float32-sized
numeric differences (`1e-6` relative / `1e-7` absolute), not changed controls.

## Validate a file

From the repository root:

```powershell
# Strict JSON, structure, fixed-state and control checks; no native plugin needed.
python research/modelling/simple_training/pipeline.py validate path/to/preset.vital

# Also round-trip through Vita and load/render through the reviewed official plugin.
python research/modelling/simple_training/pipeline.py validate path/to/preset.vital --render

# Reproduce the version/default-filling probe; requires the native dependencies.
python research/modelling/simple_training/audit_compatibility.py --output research/data_generation/outputs/compatibility.json
```

The validator prints a JSON report with diagnostic codes and JSON pointers, and
exits nonzero on failure. `--runtime` requests Vita round-trip validation without
the official-plugin render. Native dependency setup is described in the experiment
[README](README.md#run). A community preset may be perfectly valid in Vital and
still be rejected here because it is outside this deliberately narrow contract.

## Evidence and remaining limits

Tests exercise 384 seeded prediction vectors spanning scales from `1e-30` to
`1e300`, malformed files, corrupted nested assets, unknown controls, missing keys,
and failure before publication. A native boundary sweep exports and loads all
nine combinations of the three octaves and the minimum, midpoint and maximum level.

On 31 August 2026, **105 unit tests passed** and the native sweep passed **9/9
combinations**, alongside the six existing native integration tests. Standalone
`validate --render` also passed on the XPU model's reloaded-checkpoint prediction.
A fresh two-step transformer training run
on XPU also completed with this contract: frozen BasicPitch stayed unchanged,
trainable BasicPitch and transformer weights changed, and validated presets were
exported. Its local [checkpoint, contract snapshot and report](../../data_generation/outputs/simple-training-validated-xpu/)
are separate from the earlier 1.6.4-header experiments. The measured compatibility
probe is stored locally in
[`preset-compatibility-1.json`](../../data_generation/outputs/preset-compatibility-1.json).

Run the tests with:

```powershell
python -m pytest research/modelling/simple_training
$env:OBRUXO_VALIDATE_NATIVE = '1'
python -m pytest research/modelling/simple_training/test_preset_contract.py -k native
```

The finite tests support the construction-and-validation guarantee; they are not
a proof about arbitrary Vital patches. Structural validity and successful loading
also do not imply similarity to the input audio, good sound design, or successful
performance/timbre separation. Those remain separate evaluation questions.
