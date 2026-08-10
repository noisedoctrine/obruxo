# Basic Pitch backend benchmark

- Model: `spotify-basic-pitch-icassp-2022-v0.4.0`
- Smoke set: `unavailable` (8 cases)
- Runtime: `3.12.13` / float32
- Missing-WAV derived rendering: opt-in only; source patches and MIDI remain read-only.

## Inference routes

| Route | Status | Batch-1 audio seconds/second |
| --- | --- | ---: |
| `pytorch_cpu` | `unavailable` | — |
| `pytorch_xpu` | `unavailable` | — |
| `openvino_cpu` | `unavailable` | — |
| `openvino_gpu` | `unavailable` | — |

## Training routes

| Route | Status | Batch-1 audio seconds/second |
| --- | --- | ---: |
| `pytorch_cpu` | `unavailable` | — |
| `pytorch_xpu` | `unavailable` | — |

## Scope and caveats

- `the opted-in validated Vital derived-render path was unavailable`
- The report contains no source paths, filenames, IDs, hashes, or per-source predictions.
