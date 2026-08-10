# Basic Pitch backend benchmark

- Model: `spotify-basic-pitch-icassp-2022-v0.4.0`
- Smoke set: `ok` (8 cases)
- Runtime: `3.12.13` / float32
- Missing-WAV derived rendering: opt-in only; source patches and MIDI remain read-only.

## Inference routes

| Route | Status | Batch-1 audio seconds/second |
| --- | --- | ---: |
| `pytorch_cpu` | `ok` | 129.654 |
| `pytorch_xpu` | `ok` | 280.008 |
| `openvino_cpu` | `ok` | 102.785 |
| `openvino_gpu` | `parity_failed` | — |

## Training routes

| Route | Status | Batch-1 audio seconds/second |
| --- | --- | ---: |
| `pytorch_cpu` | `ok` | 29.3444 |
| `pytorch_xpu` | `ok` | 92.5048 |

## Scope and caveats

- The report contains no source paths, filenames, IDs, hashes, or per-source predictions.
