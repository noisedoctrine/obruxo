# Training-data generation foundation

This workspace provides reusable, offline-first Vital preset authoring, MIDI performance authoring, continuous rendering, and audio QA for OBRUXO. It deliberately contains no curriculum, sampling, ancestry, or dataset-sharding policy: later planners produce `VitalPreset`, `Performance`, and `RenderRequest` values through these APIs.

Generated presets, MIDI, audio, reports, and batch state belong under `outputs/`, which is ignored except for its placeholder. No factory bank, proprietary plugin, or generated audio corpus is stored in Git.

## Setup

Create the pinned research environment from the repository root:

```powershell
conda env create --file research/data_generation/environment.yml
conda activate obruxo-data-generation
$env:PYTHONPATH = (Resolve-Path research/data_generation)
python research/data_generation/run.py --help
```

The environment pins Python 3.12.13, Mido 1.2.10, NumPy 2.4.6, SciPy 1.18.0, PyYAML 6.0.3, pytest 9.1.1, Ruff 0.12.2, DawDreamer 0.8.3, and Vita at commit `342bc90aca7ab2b6e7a487f8e54a0158a5ccab76`. Once that environment and a local Vital plugin are installed, authoring, validation, rendering, and tests do not access the network.

Rendering requires a user-installed official Vital VST3. Set `plugin_path` in `configs/renderer.yaml`, pass `--plugin-path`, or set `OBRUXO_VITAL_PLUGIN`. The usual Windows path is `C:\Program Files\Common Files\VST3\Vital.vst3`. The renderer hashes the binary before opening it and rejects hashes not listed in `accepted_plugin_sha256`; obtain a local hash with:

```powershell
(Get-FileHash 'C:\Program Files\Common Files\VST3\Vital.vst3' -Algorithm SHA256).Hash.ToLower()
```

The reviewed Windows reference is Vital 1.6.4 with SHA-256 `a622a2c99b4066cd7945a4ab9bbdd698e7632a30702f6f0a7ccbf26a56b576e1`. Other builds must be reviewed and explicitly added to local configuration. Never copy the plugin or a factory preset bank into this repository.

## Python API

Static Vital and MIDI authoring does not import Vita, DawDreamer, or the plugin:

```python
from obruxo_data.midi import Performance
from obruxo_data.render import RenderRequest, VitalRenderer
from obruxo_data.vital import ComponentProfile, VitalPreset

preset = VitalPreset.init()
preset.set_raw("osc_1_level", 0.7)
preset.apply_profile(ComponentProfile.only(oscillators=[1], lfos=[1]))
preset.save("research/data_generation/outputs/simple.vital")

performance = Performance(ticks_per_beat=480, bpm=120)
performance.add_note(pitch=60, velocity=100, start_tick=0, duration_ticks=960)
performance.end_tick = 1440
performance.save_midi("research/data_generation/outputs/simple.mid")

renderer = VitalRenderer()
request = RenderRequest(
    preset=preset, performance=performance, sample_rate=44_100,
    end_tick=1440, tail_seconds=2.0, renderer_id=renderer.renderer_id,
)
result = renderer.render(request)
result.write_wav("research/data_generation/outputs/simple.wav")
result.write_json("research/data_generation/outputs/simple.json")
```

`VitalPreset` edits raw or normalized atlas controls, resets complete component-owned state, clears affected routes and paired modulation scalars, and applies component profiles atomically. `Performance` uses absolute integer ticks, preserves an explicit end boundary, and provides atomic transpose, velocity, quantize, and profile transforms. Both models return structured validation diagnostics instead of silently clamping or dropping unsupported data.

## CLI

Run the thin `argparse` wrappers from the repository root:

```powershell
python research/data_generation/run.py vital init --output research/data_generation/outputs/init.vital
python research/data_generation/run.py vital set research/data_generation/outputs/init.vital --raw osc_1_level=0.7 --output research/data_generation/outputs/edited.vital
python research/data_generation/run.py vital apply-profile research/data_generation/outputs/edited.vital --profile one_osc_one_lfo --output research/data_generation/outputs/simple.vital
python research/data_generation/run.py vital validate research/data_generation/outputs/simple.vital --runtime --json

python research/data_generation/run.py midi create-note --pitch 60 --velocity 100 --beats 2 --end-beats 3 --output research/data_generation/outputs/simple.mid
python research/data_generation/run.py midi apply-profile research/data_generation/outputs/simple.mid --profile one_held_note --output research/data_generation/outputs/profiled.mid
python research/data_generation/run.py midi validate research/data_generation/outputs/profiled.mid --json

python research/data_generation/run.py render research/data_generation/outputs/simple.vital research/data_generation/outputs/profiled.mid --plugin-path 'C:\Program Files\Common Files\VST3\Vital.vst3' --tail 2.0 --output research/data_generation/outputs/simple.wav --result research/data_generation/outputs/simple.json
```

Commands return non-zero on invalid input or rendering failure. Output commands refuse to overwrite a path unless `--force` is explicit.

## Schema bundle and probing

`obruxo_data/vital/schema/vital-1.0.8-vita-0.1.0/` is the executable schema contract. It contains the pinned runtime init document, 772-control inventory, legal modulation vocabulary, source/runtime reconciliation, exact revisions, and content hashes. The fixed `load_init_preset()` white-noise sampler payload is committed as the legal deterministic init asset. Runtime re-encoding of `/settings/sample/samples` and float32 control quantization within `1e-6` relative/`1e-7` absolute tolerance are explicitly classified warning diagnostics; any other round-trip drift is an error.

Probe a candidate bundle without overwriting reviewed output:

```powershell
python research/data_generation/run.py vital schema-probe --output research/data_generation/outputs/schema-probe
```

The offline default compares against the committed reviewed runtime inventory plus the 22 explicitly classified migration-only source registrations. Developers with a freshly extracted full Vital source atlas can pass `--source-atlas <path>` to repeat source-default reconciliation. Use `--force` only when intentionally replacing an existing probe directory.

## Renderer decision and capability boundary

The production backend is one locally installed Vital VST3 instance per render, hosted by DawDreamer 0.8.3. Vita 0.1.0 at `DBraun/Vita@342bc90aca7ab2b6e7a487f8e54a0158a5ccab76` exposes only `render(midi_note, midi_velocity, note_dur, render_dur)`, so it is retained for schema probing and runtime preset validation rather than falsely treating independent note renders as a continuous synth.

DawDreamer injects the complete MIDI note stream into one continuously running Vital plugin graph. A fresh plugin instance is created for every request, its JUCE VST3 state template is loaded with the complete canonical `.vital` JSON, and absolute ticks are converted to half-even-rounded sample offsets. Stereo `float32` is returned as `[frames, channels]` through the explicit performance end plus release tail.

The initial capability contract supports notes, polyphony, one fixed tempo, MIDI channel 0, and an explicit end plus tail. Tempo changes, pitch bend, pressure, CC messages, other channels, and opaque events fail validation before the plugin opens. `renderer.yaml` declares this boundary and is checked against the implemented backend rather than trusted silently.

Pinned implementation references:

- Vital source/state format: `mtytel/vital@636ca0ef517a4db087a6a08a6a8a5e704e21f836`.
- Vita runtime: `DBraun/Vita@342bc90aca7ab2b6e7a487f8e54a0158a5ccab76` (`0.1.0`).
- DawDreamer source audit: `DBraun/DawDreamer@b891902bae3ef5cb9041373b888ceeb2a016f9d4`.
- DawDreamer wheel: `0.8.3`; Windows CPython 3.12 wheel SHA-256 `b4eff365bf40f373406279c88728c8545b9a3cbd8b1e6ba29458dc4964e90e2c`.

DawDreamer is GPLv3 and depends on JUCE and the Steinberg VST3 SDK. Users are responsible for the licenses of Vital, DawDreamer, and their transitive dependencies.

## Batch seam

`write_requests()` writes canonical JSONL jobs. `run_batch()` creates a fresh synth inside each render, preserves request order, and resumes only when both the result request ID and decoded WAV float-buffer hash match. The generic seam supports a bounded thread pool, but the pinned Windows DawDreamer/Vital host stalled under concurrent in-process plugin creation during the release test; `VitalRenderer.max_workers` is therefore enforced as 1 instead of making an unsafe concurrency claim.

```python
from obruxo_data.render import RenderRequest, write_requests

write_requests("research/data_generation/outputs/requests.jsonl", [
    RenderRequest(preset=preset, performance=performance, renderer_id=renderer.renderer_id),
])
```

```powershell
python research/data_generation/run.py batch research/data_generation/outputs/requests.jsonl --plugin-path 'C:\Program Files\Common Files\VST3\Vital.vst3' --output research/data_generation/outputs/batch --workers 1
```

This seam intentionally has no curriculum, remote queue, sharding, bank sampler, or storage service.

## QA, fixtures, and repeatability

Every result records expected/actual shape, finiteness, peak and clipping count, RMS and silence threshold, DC offset, tail energy, the canonical float-buffer SHA-256, request ID, DawDreamer version, schema ID, scheduling policy, and the full plugin fingerprint. The complete plugin SHA-256 is also part of `renderer_id` and therefore request identity; a configured build ID that omits it is rejected. Audio is never loudness-normalized.

Vital 1.6.4 is not bit-deterministic even with oscillator random phase disabled. The legally generated 5-preset × 3-performance release matrix therefore records pointwise waveform RMSE and enforces numeric repeatability on stable aggregate/spectral measures: at most 1% relative RMS difference, 0.10 absolute peak difference, and 0.35 whole-render `log1p` spectral-magnitude RMSE. Calibration measured maxima of 0.64%, 0.092, and 0.320 respectively. Each render keeps its own lossless float-buffer hash; request identity is stable while bit-identical audio is not claimed.

Regenerate authored fixtures and reference metrics explicitly:

```powershell
python research/data_generation/tests/fixtures/generate_fixtures.py --force
python research/data_generation/tests/fixtures/generate_reference_results.py --plugin-path 'C:\Program Files\Common Files\VST3\Vital.vst3' --force
```

`tests/fixtures/reference_results.json` retains aggregate metrics and hashes, not audio. The tempo-change fixture is intentionally rejected because the backend declares no tempo-change support.

## Tests and CI

From the repository root:

```powershell
ruff check research/data_generation/obruxo_data research/data_generation/tests research/data_generation/run.py
python -m pytest research/data_generation/tests/unit
python -m pytest research/data_generation/tests -m 'not reference_plugin'
$env:OBRUXO_VITAL_PLUGIN = 'C:\Program Files\Common Files\VST3\Vital.vst3'
python -m pytest research/data_generation/tests/integration
```

Unit tests require neither Vita nor a proprietary binary. The distributable Vita runtime tests run when Vita is installed; reference-plugin tests skip if the user-supplied path is unavailable. `.github/workflows/data-generation.yml` recreates the pinned environment on Ubuntu, runs Ruff, all static tests, and the distributable Vita integration path. The Windows reference-plugin, fresh-state, and concurrency-boundary tests are local release gates.

## Workspace map

- `run.py`: human CLI entry point.
- `obruxo_data/vital/`: schema, presets, component ownership, profiles, and validation.
- `obruxo_data/midi/`: canonical events, SMF I/O, timing, transforms, and profiles.
- `obruxo_data/render/`: capability contract, Vital backend, QA, provenance, and batch seam.
- `configs/`: versioned YAML runtime, renderer, component, and performance profiles.
- `tests/`: static tests, legal fixtures, and opt-in native integration tests.
- `outputs/`: ignored local artifacts.
