# MIDI expression in on-demand training

The expression profile carries pitch bend, channel pressure, modulation wheel (CC1), sustain pedal (CC64), MPE slide (CC74), note-off velocity and MIDI channels 1–16 into Vital. This lets training exercise the corresponding patch sources. It extends the [custom-curve and seed model](CUSTOM_CURVES_AND_SEEDS.md) through a separate codec, renderer and performance-bank identity. Existing profiles keep their identities and note transport.

The installed DawDreamer 0.8.3 API supports `load_midi(..., beats=False, all_events=True)`. The [upstream API documentation](https://dirt.design/DawDreamer/api_reference/index.html) describes these flags; the installed version and native behavior were checked locally. MIDI files are temporary transport files. Audio stays in memory and is generated during the training loop.

## Timing and native behavior

The compiler first converts each source tick into an integer audio sample using the existing half-even `TempoMap`. It writes a fixed-clock MIDI file with four ticks per sample, placing the timestamp one quarter sample inside its intended sample. This prevents floating-point floor/nearest conversion from moving an event to the previous sample. The source tempo has already been applied to event offsets; the render engine receives the actual fixed performance BPM independently.

Native checks cover 44.1, 48 and 96 kHz. Boundary fixtures include half-sample ties, simultaneous notes and offsets around processing-block boundaries. Note-only output matches the earlier API exactly in ten fixtures. The remaining chord exposes a legacy API note-off at sample 61,739 instead of the intended 61,740; a file shifted one sample earlier reproduces the old output exactly. The new path retains the intended boundary. Its repeated audio is exact with generator phases fixed.

Controller response is processed at Vital's native control rate. It can become audible within the processing block containing the event, including slightly before its nominal sample. The qualified 128-sample host configuration bounds the measured response within one host buffer. This is not sample-exact modulation response. Per-note bend, pressure and slide are independently checked while another MPE note remains unchanged.

Tempo changes, opaque messages, polyphonic key pressure and controllers beyond CC1/64/74 remain unsupported by this profile. The compiler can map tempo changes to absolute event offsets, but host tempo automation has not been qualified and the renderer rejects those performances. Unsupported events fail explicitly. Legacy profiles also now reject nonzero release velocity, which their note-only API previously discarded.

## Frozen performance bank and source patches

The new bank retains the eight initial performances and adds eight expression performances: gesture hold, independent polyphonic expression, sustain phrase and release-velocity phrase, each with reference and transfer variants. There are eight reference and eight reserved transfer performances. Training rejects transfer performances and held-out states as before. The old bank remains unchanged.

`ExpressionCodec` inherits all 7,187 continuous and 898 categorical outputs. It adds four authored connections using empty slots: modulation wheel to oscillator-1 pan, pressure to oscillator-2 pan, slide to oscillator-3 pan, and lift/release velocity to oscillator-1 pan. Existing slot positions and amount references remain intact. The original LFOs, custom curves, mapping shapes and seed labels retain their source bindings. MPE is enabled, and this new curriculum sets master volume to `2500` (-30 dB) to provide more headroom. Those are declared source changes; rendered audio is not normalized afterward.

Only audio-derived timbre and frozen Basic Pitch features enter the model. MIDI remains part of rendering recipes and evaluation, not an inference input. This implementation does not estimate controller gestures from an arbitrary input clip or establish a finished model's performance invariance.

The optional `performance_round_robin` sampler completes a sweep over all reference performances before moving to the next rotated set of shuffled training states. Every state/performance pair appears once per epoch. It makes early bounded feature collection cover the entire reference bank. Defaults and earlier sampling rules remain unchanged. Saved performance IDs are verified against their recipes during recovery.

## Run locally

```powershell
python research/training/reports/scripts/run_expression_check.py --output research/training/outputs/expression-native
python research/training/run.py expression-pool --asset-bank research/training/outputs/factory-bank/bank.json --lfo-bank research/training/outputs/lfo-bank-check/bank.json --output research/training/outputs/expression --count 256
python research/training/reports/scripts/run_expression_integration.py --asset-bank research/training/outputs/factory-bank/bank.json --lfo-bank research/training/outputs/lfo-bank-check/bank.json --native-qualification research/training/outputs/expression-native --output research/training/outputs/expression-integration
```

The bounded integration plans at most 32 native training attempts, sixteen retained examples and three optimizer updates, then checks feature/label integrity, frozen weights, bank coverage and export reload. Its purpose is to prove expressive feature ingestion and updates. It is not another small reconstruction-quality experiment and does not promote its diagnostic weights.

Results and resource measurements are in the [expression report](reports/EXPRESSIVE_MIDI_RESULTS.md). Free local CPU remains the qualified path; Linux/Colab and the final-test reserve are untouched. Broader combined control coverage, measured data scaling, useful reconstruction and release gates remain in the [completion ledger](IMPLEMENTATION_STATUS.md). GitHub Actions stays disabled.
