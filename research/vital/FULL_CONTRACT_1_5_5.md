# Full preset verification, with a small learning target

The executable contract now covers **775 scalar controls and all six nested
structures in 1.5.5 presets**. The model still predicts only octave and oscillator
level in the small experiment. The decoder supplies everything else from a
reviewed template, and invalid predictions are rejected before export.

Modulation ramps remain outside the model and outside exported 1.5.5 documents.
The reviewed Vital 1.6.4 renderer fills all 128 ramp fields with their measured
default of `-10`. We do not import newer presets by silently deleting controls or
changing version labels.

This separates three questions: does the JSON satisfy our contract, does Vital
preserve the requested state, and does its rendered audio resemble the input?
This work verifies the first two within the limits below. It does not establish
reconstruction quality.

## What we found

All 6,581 corpus presets labeled 1.5.5 have the same 775 scalar names and 781 total
settings keys. The full scan discovers 9,620 files; two cannot be parsed as JSON.
Other versions are counted separately rather than pooled into one schema.

**Field names alone do not establish compatibility.** An initial audit using the
older source-era metadata rejected 951 presets for range violations. Measuring
the installed renderer showed that several existing controls have expanded
their value ranges. We now have native raw endpoint measurements for all 775
controls, matched to unique parameter names and verified through saved state.

| Control family | Earlier metadata | Measured 1.6.4 range | Why it matters |
| --- | --- | --- | --- |
| `osc_<n>_spectral_morph_type` | `0..11` | `0..16` | More morph choices use the same field name. |
| `osc_<n>_stack_style` | `0..10` | `0..12` | Old bounds reject additional stack choices. |
| `lfo_<n>_sync_type` | `0..5` | `0..6` | The ordinal vocabulary has grown. |
| `osc_<n>_spectral_morph_phase` | Absent | `0..1`, default `0.5` | These three controls belong to the full 1.5.5 inventory. |

The [final corpus report](contract_1_5_5/corpus_verification.json) records pass
counts and rejection categories using measured renderer bounds. Rejection means
outside this authoring contract, not necessarily unplayable in Vital. We do not
widen bounds merely because a corpus file contains an outlier.

| Final check on the 6,581 target presets | Pass | Flagged |
| --- | ---: | ---: |
| Structure, relationships, and payload checks | 6,526 | 55 |
| Those checks plus renderer authoring bounds | 6,251 | 330 |

The rejection categories overlap. In particular, 273 files exceed measured raw
bounds, often on compressor ratios; four have nonintegral categorical values.
Other findings include unordered or duplicate keyframes, missing keyframes,
invalid audio-source windows, and malformed base64. Flagged files are left
unchanged for review rather than repaired automatically.

Four [native test cases](contract_1_5_5/native_verification.json) preserved all
775 supplied scalars, filled all 128 ramps correctly, and rendered finite,
non-silent audio. They exercise phase endpoints/midpoint, a live modulation
connection, custom-curve and seed edits, and Shepard Tone Source. That last
component exists in the pinned source but is absent from the 1.5.5 cohort.

The native checks found a sampler detail: readback adds **four leading zero
samples and drops the final four**, with at most one PCM16 unit of rounding after
alignment. The test checks that exact pattern; arbitrary payload changes fail.
Our experiment disables the sampler, but sampler round trips must not be called
byte-exact. Wavetable version headers become `1.6.4`; other nested changes fail
the native probe.

## What is checked

| Part | Contract | Current prediction policy |
| --- | --- | --- |
| Metadata | Required string fields, exact `synth_version: "1.5.5"`, no unknown fields | Supplied by decoder. |
| Scalars | 775 required finite numbers; raw bounds; integral categorical ordinals | Small model predicts two controls. |
| `modulations` | 64 connections; known sources/destinations; no half-connected routes; optional remap checked | No active routes in learning experiment. |
| `lfos` | Eight curves; consistent point/power lengths; ordered, bounded coordinates | Factory shapes. |
| `custom_warps` | Three curves with the same checks | Factory shapes. |
| `random_values` | Three seed objects; unsigned 32-bit integral seeds | Factory seeds. |
| `sample` | Mono/stereo base64 PCM16 byte lengths match declared sample length | Factory sample, disabled in experiment. |
| `wavetables` | Three tables; nonempty groups; ten known component types; component-specific keyframes | Factory assets in scalar decoder. |
| `wave_data` | Exactly 2,048 finite float32 samples in base64 | Never predicted as raw JSON. |
| `audio_file` | Nonempty PCM16 base64, even byte length, positive rate/window | Never predicted as raw JSON. |
| Ramp controls | Excluded from scalar outputs and modulation destinations | Renderer supplies defaults. |

Parsing rejects duplicate keys, invalid UTF-8, nonstandard numeric tokens,
nonfinite values, excessive nesting, and files above the configured 64 MiB safety
limit. These are our authoring limits, not claimed Vital product limits. Export
validates the object and exact serialized bytes, then publishes atomically
without overwriting an existing file.

The existing [experiment contract](../modelling/simple_training/PRESET_CONTRACT.md)
remains a narrower, independently verified template contract. Its 772-scalar
internal representation is a loadable subset, not a complete 775-scalar document.
This broader validator does not silently change the training head or checkpoints.
It is the full-document validation and scalar-authoring building block for the
next expansion of the experiment.

## Use and reproduction

Run from the repository root in the project Python environment:

```powershell
python research/vital/verify_155.py path/to/preset.vital
python research/vital/verify_155.py path/to/preset.vital --structure-only
python research/vital/verify_155.py datasets/presetshare/raw/presetshare_files/data --corpus --output audit.json
python -m pytest research/vital/test_full_contract.py -q
```

A single-file check exits nonzero on rejection. A corpus audit finishes with a
report even when individual files fail; its exit status is not an all-files-pass
signal. Output reports are never overwritten implicitly.

Import `FullPresetContract` from `research/vital/full_contract.py` for authoring:

```python
contract = FullPresetContract()
preset = contract.decode_scalars({"osc_1_transpose": -12, "osc_1_level": 0.4})
contract.save(preset, "prediction.vital")
```

This helper accepts **decoded raw scalars**, not logits. Argmax/sigmoid decisions
belong to the model's explicit prediction mapping. The helper starts from a
hash-verified factory preset and preserves nested assets; unknown controls,
including ramps, are rejected.

The native commands require DawDreamer and the reviewed Vital 1.6.4 plugin. They
check its fingerprint before loading it:

```powershell
python research/vital/probe_155_ranges.py --output renderer_ranges.json
python research/vital/verify_155_native.py --output native_verification.json
python research/vital/build_155_contract.py datasets/presetshare/raw/presetshare_files/data --native-init native-init-1.6.4.json --renderer-ranges renderer_ranges.json
```

`--native-init` is the JSON extracted from a fresh plugin state with
`VitalVst3StateTemplate(...).preset_document`. It supplies the three phase defaults,
custom curves, seeds, and ramp-default evidence. Sample and wavetable assets come
from the existing pinned factory baseline, never from corpus presets. Rebuilding
is an explicit authoring operation that updates the generated contract bundle.

The bundle pins inventory, modulation vocabulary, renderer bounds, factory
template, and cohort content hashes. Nine component structures come from corpus
observations. [Shepard Tone Source](https://github.com/mtytel/vital/blob/636ca0ef517a4db087a6a08a6a8a5e704e21f836/src/common/wavetable/shepard_tone_source.h)
inherits Wave Source serialization in the pinned source. The
[audio-source serializer](https://github.com/mtytel/vital/blob/636ca0ef517a4db087a6a08a6a8a5e704e21f836/src/common/wavetable/file_source.cpp)
confirms that `audio_file` contains PCM16 bytes, not a WAV container.

## Limits of the evidence

The target is **the 1.5.5 field inventory rendered with verified Vital 1.6.4**.
This is not a recovered official 1.5.5 specification: historical binary bounds
and behavior remain untested. Static checks cannot prove that every legal
combination sounds right or avoids every plugin bug. Nested DSP amounts do not
yet have a fully recovered range catalog. Unknown structures fail closed.

The corpus is not an independent holdout for the structure it helped define.
Only four authored cases received native readback and audio checks; the corpus
verification is static. Reconstruction accuracy, performance/timbre separation,
and diffusion/flow training remain separate experiments.
