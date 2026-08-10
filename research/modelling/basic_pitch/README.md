# OBRUXO Basic Pitch

This workspace contains the single Spotify Basic Pitch ICASSP-2022 architecture used by OBRUXO. The public reference is Spotify Basic Pitch v0.4.0 at revision `9991303bba609a3b93089d13ec80d1d495083596`, specifically `basic_pitch/saved_models/icassp_2022/nmp.onnx` (Git blob `c30e5f9438e798604b7177aa26be1fe64482f767`, 230444 bytes). The upstream project is Apache-2.0 licensed. The CQT implementation follows the upstream nnAudio-derived semantics; nnAudio's corresponding implementation is MIT licensed.

The repository's local validation uses the existing user-managed `py312` environment and does not apply this metadata file locally. The runtime identity recorded for the current validation is Python 3.12.13, NumPy 2.4.6, SciPy 1.18.0, PyTorch 2.12.1+xpu, ONNX 1.22.0, ONNX Runtime 1.28.0, pytest 9.1.1, and Ruff 0.16.2. `environment.yml` is reproducibility/CI metadata only.

Acquire the pinned public ONNX file separately, then convert it into the one authorized public checkpoint and metadata sidecar:

```text
python research/modelling/basic_pitch/run.py import-onnx \
  --onnx research/modelling/basic_pitch/outputs/nmp.onnx \
  --checkpoint research/modelling/basic_pitch/artifacts/basic_pitch_icassp_2022.pt \
  --metadata research/modelling/basic_pitch/artifacts/basic_pitch_icassp_2022.json
```

The native module loads as a plain PyTorch state dict and does not require ONNX, ONNX Runtime, TensorFlow, Basic Pitch, nnAudio, or network access at runtime:

```python
import torch
from obruxo_basic_pitch import BasicPitchICASSP2022

model = BasicPitchICASSP2022()
state = torch.load("artifacts/basic_pitch_icassp_2022.pt", map_location="cpu", weights_only=True)
model.load_state_dict(state, strict=True)
model.eval()
with torch.inference_mode():
    output = model(torch.zeros(1, 43_844, 1, dtype=torch.float32))
```

Run deterministic CPU float32 parity against the public ONNX oracle:

```text
python research/modelling/basic_pitch/run.py parity \
  --onnx research/modelling/basic_pitch/outputs/nmp.onnx \
  --checkpoint research/modelling/basic_pitch/artifacts/basic_pitch_icassp_2022.pt \
  --json research/modelling/basic_pitch/reports/onnx_parity.json \
  --markdown research/modelling/basic_pitch/reports/onnx_parity.md
```

The committed parity report contains aggregate synthetic/public results only. Optional `--audio` validation reads existing local WAVs without modifying or publishing them; only sanitized aggregate counts and errors may be retained. CPU/XPU/OpenVINO cost measurement belongs to #24. PresetShare pairing and transcription evaluation belong to #25.

## Fixed backend benchmark

The #24 benchmark measures the canonical checkpoint on the fixed inference matrix (`pytorch_cpu`, `pytorch_xpu`, `openvino_cpu`, `openvino_gpu`) and the fixed full forward-plus-backward training matrix (`pytorch_cpu`, `pytorch_xpu`). It uses float32, fresh subprocesses, three repetitions, three warmups, ten timed calls, and model-call batches `[1, 2, 4, 8]`. OpenVINO is converted from the native PyTorch module and compiled for the explicitly requested device; it never uses automatic device selection or fallback.

Create the private ignored smoke manifest from existing paired WAV/MIDI files, then run:

```text
python research/modelling/basic_pitch/run.py benchmark \
  --config research/modelling/basic_pitch/configs/backend_benchmark.yaml \
  --manifest research/modelling/basic_pitch/outputs/smoke_manifest.json \
  --checkpoint research/modelling/basic_pitch/artifacts/basic_pitch_icassp_2022.pt \
  --json research/modelling/basic_pitch/reports/backend_benchmark.json \
  --markdown research/modelling/basic_pitch/reports/backend_benchmark.md \
  --xpu-index 0 \
  --openvino-gpu-device GPU
```

The manifest and all source-derived scratch state remain local and ignored. Reports contain only anonymous labels, sanitized aggregate timings, memory availability, parity summaries, and route statuses. Existing WAVs are always read-only. When an otherwise unambiguous source relationship has an existing Vital patch and MIDI performance but no WAV, the parent contract permits an explicitly opted-in derived render under `research/modelling/basic_pitch/outputs/`:

```text
python research/modelling/basic_pitch/run.py benchmark \
  --config research/modelling/basic_pitch/configs/backend_benchmark.yaml \
  --manifest research/modelling/basic_pitch/outputs/smoke_manifest.json \
  --checkpoint research/modelling/basic_pitch/artifacts/basic_pitch_icassp_2022.pt \
  --json research/modelling/basic_pitch/reports/backend_benchmark.json \
  --markdown research/modelling/basic_pitch/reports/backend_benchmark.md \
  --allow-derived-render \
  --force
```

The opt-in flag defaults to false. Such a private manifest adds `audio_source: "derived_render"` and `preset_path` to a case, points `audio_path` at a new WAV below the approved ignored output root, and leaves the source patch/MIDI untouched. The runner performs a resolved destination check, refuses to overwrite an existing output or source directory, records renderer provenance in the local sidecar, and never publishes source paths or identities. If the validated Vital/DawDreamer runtime is unavailable, the run reports `derived_render_unavailable` rather than falling back or changing the environment. #25 owns the comprehensive PresetShare pairing/evaluation manifest; #24 only consumes its fixed smoke contract.

## PresetShare evaluation

The #25 evaluator first inspects the local PresetShare-derived layout and uses the observed concrete rule: a direct child directory is eligible only when its MIDI/audio relationship is unambiguous under the directory contract. Ambiguities, invalid MIDI, missing MIDI, missing audio, and unavailable derived rendering are retained in a private pairing audit. Legacy audio without a render-QA sidecar remains eligible. The current sanitized report records the observed coverage and does not contain source paths, IDs, filenames, or row-level results.

Build the private manifest from the repository root:

```text
python research/modelling/basic_pitch/run.py build-eval-manifest \
  --corpus-root datasets/presetshare/raw/presetshare_files/data \
  --output research/modelling/basic_pitch/outputs/presetshare_evaluation/manifest.jsonl \
  --audit research/modelling/basic_pitch/outputs/presetshare_evaluation/pairing_audit.json
```

When a source directory contains exactly one readable Vital patch and one valid MIDI performance but no WAV, derived rendering is available only behind the explicit, default-off flag:

```text
python research/modelling/basic_pitch/run.py build-eval-manifest \
  --corpus-root datasets/presetshare/raw/presetshare_files/data \
  --output research/modelling/basic_pitch/outputs/presetshare_evaluation/manifest.jsonl \
  --audit research/modelling/basic_pitch/outputs/presetshare_evaluation/pairing_audit.json \
  --allow-derived-render
```

This opt-in exists because the patch and MIDI provide an exact, reproducible local rendering input under the current parent contract. It writes only a labeled `derived_render` WAV and provenance sidecar below the ignored Basic Pitch output root, refuses source overlap/overwrite, and never calls the result an original or historical WAV. Existing WAVs are read-only. If the validated renderer is unavailable, the audit records `pair.derived_render_unavailable`; the environment is not changed and no fallback renderer is used.

Run the fixed stock evaluation with the #24 CPU route and no threshold controls:

```text
python research/modelling/basic_pitch/run.py evaluate-corpus \
  --manifest research/modelling/basic_pitch/outputs/presetshare_evaluation/manifest.jsonl \
  --output research/modelling/basic_pitch/outputs/presetshare_evaluation
```

The private output stores resumable per-pair metrics, aggregate counts, timing/velocity/pitch diagnostics, frame metrics, failure cases, and source-stat checks. The tracked `reports/presetshare_baseline.json` and `.md` files contain sanitized aggregate evidence only. `mir_eval==0.8.2`, `Performance`, and `TempoMap` define the fixed evaluation semantics; no model, backend, decoder threshold, or corpus pairing rule is tuned from the results.
