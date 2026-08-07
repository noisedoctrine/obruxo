# Training-data generation foundation

This workspace provides reusable, offline-first authoring and rendering primitives for OBRUXO. It deliberately contains no curriculum policy: callers construct or load a `VitalPreset`, construct or load a `Performance`, and submit both to a renderer.

Run commands from the repository root with the `py312` Conda environment:

```powershell
conda run --no-capture-output -n py312 python research/data_generation/run.py --help
conda run --no-capture-output -n py312 python -m pytest research/data_generation/tests
```

Generated presets, MIDI, audio, reports, and batch state belong under `outputs/`, which is ignored except for its placeholder.

## Renderer decision (WP0)

The production backend is a locally installed Vital VST3 hosted by DawDreamer. Current Vita `0.1.0` at `DBraun/Vita@342bc90aca7ab2b6e7a487f8e54a0158a5ccab76` still exposes only `render(midi_note, midi_velocity, note_dur, render_dur)`. It has no arbitrary event-stream binding, so it is retained for optional preset runtime validation and schema probing, not multi-event rendering.

DawDreamer `0.8.3` hosts one Vital plugin instance, injects the complete MIDI stream, and runs one continuous render. Vital's `SynthPlugin::setStateInformation` parses the received bytes as Vital JSON, while DawDreamer's `load_state` forwards raw plugin-state bytes. This lets the backend load an authored `.vital` document directly without converting it to a bank preset or summing separately rendered notes.

The first backend capability contract supports note events, polyphony, one fixed tempo, 16 MIDI channels, and an explicit render end plus tail. Tempo changes, pitch bend, pressure, arbitrary CC messages, and unsupported/opaque MIDI events are rejected before the plugin is opened. Events are converted from absolute ticks to deterministic sample offsets by the shared timing layer; adjacent segments use one half-even rounding rule.

Pinned implementation references:

- Vital source/state format: `mtytel/vital@636ca0ef517a4db087a6a08a6a8a5e704e21f836`.
- Vita runtime API audit: `DBraun/Vita@342bc90aca7ab2b6e7a487f8e54a0158a5ccab76` (`0.1.0`).
- DawDreamer source audit: `DBraun/DawDreamer@b891902bae3ef5cb9041373b888ceeb2a016f9d4`.
- DawDreamer wheel: `0.8.3`; Windows CPython 3.12 wheel SHA-256 `b4eff365bf40f373406279c88728c8545b9a3cbd8b1e6ba29458dc4964e90e2c`.
- Local reference Vital VST3 used for the Windows spike: SHA-256 `a622a2c99b4066cd7945a4ab9bdda698e7632a30702f6f0a7ccbf26a56b576e1`. The plugin itself is never committed or redistributed.

The host is GPLv3 and depends on JUCE and the Steinberg VST3 SDK. Users must install Vital themselves and comply with the licenses of Vital, DawDreamer, and their transitive dependencies. The plugin path and accepted binary hashes are local YAML configuration, never inferred by downloading a proprietary/factory asset.

## Workspace map

- `run.py`: human CLI entry point.
- `obruxo_data/`: reusable authoring, validation, rendering, hashing, and batch modules.
- `configs/`: typed YAML examples for runtime settings and profiles.
- `tests/`: unit tests plus opt-in `vita`, `integration`, and `reference_plugin` tests.
- `outputs/`: ignored generated artifacts.

Static authoring and validation do not import Vita, DawDreamer, or an installed plugin. Integration tests are explicitly skipped when their user-supplied local dependency is unavailable.
