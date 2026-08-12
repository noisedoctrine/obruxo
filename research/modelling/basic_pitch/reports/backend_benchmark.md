# Basic Pitch backend benchmark

This is a fixed measurement of the canonical #23 float32 model, not an optimization search. Markdown shows medians across three fresh-process repetitions; the JSON retains min/max/total values and anonymous per-case timing.

## Executive findings

- Measured steady-state inference winners by model-call batch: batch 1 `pytorch_xpu` (238.368 audio-seconds/second), batch 2 `pytorch_xpu` (417.493 audio-seconds/second), batch 4 `pytorch_xpu` (616.100 audio-seconds/second), batch 8 `pytorch_xpu` (727.274 audio-seconds/second).
- On the warmed end-to-end smoke boundary, `pytorch_xpu` is fastest at `96.027` audio-seconds per wall-second.
- The fixed startup/throughput calculation retains `2` positive finite crossover point(s); these are descriptive model-only results, not claims about all application workloads.
- First-call and startup trade-offs remain visible in the dedicated tables, so short interactive calls must be interpreted separately from reused or longer workloads.
- The corrected OpenVINO GPU route now has measured FP32 + PERFORMANCE startup, batch scaling, end-to-end, parity, and resource results.
- The original/default FP16 + PERFORMANCE OpenVINO GPU parity failure remains preserved as historical evidence; it is not conflated with the corrected timed route.
- The bounded post-fix FP32 + PERFORMANCE parity diagnostic is retained as a separate correctness result; the corrected route's performance/resource measurements are now included in the tables below.

## Runtime and benchmark setup

- Model: `spotify-basic-pitch-icassp-2022-v0.4.0`; precision: `float32`; smoke set: `ok` with `8` cases.
- Runtime: Python `3.12.13`, PyTorch `2.12.1+xpu`, OpenVINO `2026.3.0-000--`, NumPy `2.4.6`, SciPy `1.18.0`.
- Smoke-set coverage gate: `complete`; required representative categories are recorded in the JSON contract and missing categories are `{}`.
- Each route used a fresh process for each of 3 repetitions; each fixed batch used 3 warmups and 10 timed calls.
- Model-only inference and full forward+backward training used batches `[1, 2, 4, 8]`. End-to-end inference used batch 1 and covered read-only audio preparation through stock note-event materialization.
- Missing-WAV derived rendering was opt-in only; source patches, MIDI, audio, and metadata remained read-only.

## Corpus inference decision for #25

- Status: `selected`; selected backend: `pytorch_xpu`; device: `xpu:0`; precision: `float32`.
- Boundary: `end_to_end_audio_to_note_event`.
- Selection rule: highest median end-to-end audio-seconds/wall-second among successful parity-safe inference routes.
- Supporting run identity code revision: `c61ffc23b89abc09ce645760849e9be3a004d6ab`.

## Timed inference route identity

The route-specific properties below are recorded from the actual compiled/runtime objects. `n/a` means the property was not exposed by that route; no device fallback is inferred.

| Route | Parity | Selected device | Full device name | Execution devices | Effective inference precision | Execution mode | Available devices |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `pytorch_cpu` | `passed` | n/a | n/a | n/a | n/a | n/a | n/a |
| `pytorch_xpu` | `passed` | n/a | n/a | n/a | n/a | n/a | n/a |
| `openvino_gpu` | `passed` | GPU | Intel(R) Arc(TM) 140T GPU (16GB) (iGPU) | GPU.0 | float32 | PERFORMANCE | CPU, GPU |

## Inference startup and initialization

Startup is separated from first-call, warmup, and steady-state timing. The corrected OpenVINO GPU timing rows below use the explicit float32 inference hint and the plugin-reported PERFORMANCE execution mode. The historical pre-fix failure remains preserved in the diagnostic sections; `n/a` means the route failed before that phase or the phase does not apply.

| Route | Status | Import | Construct | Checkpoint | Device move | OV convert | OV compile | Total startup |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `pytorch_cpu` | `ok` | 0.778 | 0.002 | 0.004 | n/a | n/a | n/a | 0.783 |
| `pytorch_xpu` | `ok` | 0.782 | 0.002 | 0.005 | 0.016 | n/a | n/a | 0.804 |
| `openvino_cpu` | `parity_failed` | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| `openvino_gpu` | `ok` | 0.777 | 0.002 | 0.004 | n/a | 1.592 | 0.346 | 2.700 |

### First-call and warmup observations

| Route | Batch | First call (s) | Warmup (s) | Steady median call (s) |
| --- | ---: | ---: | ---: | ---: |
| `pytorch_cpu` | 1 | 0.026 | 0.072 | 0.025442 |
| `pytorch_cpu` | 8 | 0.090 | 0.245 | 0.089291 |
| `pytorch_xpu` | 1 | 0.624 | 0.023 | 0.008390 |
| `pytorch_xpu` | 8 | 0.657 | 0.069 | 0.022000 |
| `openvino_gpu` | 1 | 0.038 | 0.062 | 0.017905 |
| `openvino_gpu` | 8 | 0.095 | 0.162 | 0.052546 |

## Steady-state inference scaling

The corrected timing tables expose both throughput and call latency for every tested batch, including OpenVINO GPU under explicit float32 inference. Throughput is the model-call audio-equivalent rate; it excludes audio decode and stock postprocessing.

### Audio-equivalent throughput (audio-seconds/second)

| Route | Batch 1 | Batch 2 | Batch 4 | Batch 8 |
| --- | ---: | ---: | ---: | ---: |
| `pytorch_cpu` | 78.610 | 141.531 | 225.268 | 179.190 |
| `pytorch_xpu` | 238.368 | 417.493 | 616.100 | 727.274 |
| `openvino_gpu` | 111.703 | 242.847 | 287.862 | 304.494 |

### Median model-call latency (seconds)

| Route | Batch 1 | Batch 2 | Batch 4 | Batch 8 |
| --- | ---: | ---: | ---: | ---: |
| `pytorch_cpu` | 0.025442 | 0.028262 | 0.035513 | 0.089291 |
| `pytorch_xpu` | 0.008390 | 0.009581 | 0.012985 | 0.022000 |
| `openvino_gpu` | 0.017905 | 0.016471 | 0.027791 | 0.052546 |

Interpretation: batch 1 winner is `pytorch_xpu` at 238.368 audio-seconds/second; batch 2 winner is `pytorch_xpu` at 417.493 audio-seconds/second; batch 4 winner is `pytorch_xpu` at 616.100 audio-seconds/second; batch 8 winner is `pytorch_xpu` at 727.274 audio-seconds/second. These are fixed-workload observations, not tuning targets.

## End-to-end audio-to-note-event throughput

This is the realistic batch-1 boundary: read-only audio open/decode, in-memory preparation, model windows, unwrapping, and stock note-event materialization. The smoke set totals 162.073 audio seconds across 8 cases.

| Route | Median wall time (s) | Min-max wall time (s) | Median audio-seconds/wall-second | Median RTF (wall/audio) |
| --- | ---: | ---: | ---: | ---: |
| `pytorch_cpu` | 3.123 | 3.092-3.272 | 51.896 | 0.01927 |
| `pytorch_xpu` | 1.688 | 1.610-1.728 | 96.027 | 0.01041 |
| `openvino_gpu` | 3.612 | 3.542-3.616 | 44.868 | 0.02229 |

The measured end-to-end winner is `pytorch_xpu`; this ordering is specific to the fixed smoke boundary and includes no failed route.

## CPU versus XPU full forward+backward cost

These rows measure the explicitly allowed backward cost at the native PyTorch boundary. They do not train, update, or save weights.

### Effective throughput (audio-seconds/second)

| Route | Batch 1 | Batch 2 | Batch 4 | Batch 8 |
| --- | ---: | ---: | ---: | ---: |
| `pytorch_cpu` | 28.533 | 32.766 | 31.880 | 28.735 |
| `pytorch_xpu` | 87.951 | 142.903 | 209.984 | 227.634 |

### Median forward+backward step latency (seconds)

| Route | Batch 1 | Batch 2 | Batch 4 | Batch 8 |
| --- | ---: | ---: | ---: | ---: |
| `pytorch_cpu` | 0.070094 | 0.122076 | 0.250945 | 0.556816 |
| `pytorch_xpu` | 0.022740 | 0.027991 | 0.038098 | 0.070288 |

Training winners by batch are: batch 1 `pytorch_xpu` (87.951 audio-seconds/second); batch 2 `pytorch_xpu` (142.903 audio-seconds/second); batch 4 `pytorch_xpu` (209.984 audio-seconds/second); batch 8 `pytorch_xpu` (227.634 audio-seconds/second). This is a cost observation, not a recommendation to change the current training architecture.

## Memory and resource observations

Host RSS is a peak process measurement and is not directly interchangeable with device allocation. `n/a` is an unavailable measurement, not zero.

| Mode | Route | Host peak RSS (MiB) | XPU allocated (MiB) | XPU reserved (MiB) | OV GPU current allocation (MiB) | OV GPU device memory (MiB) | Measurement note |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| inference | `pytorch_cpu` | 609.1 | n/a | n/a | n/a | n/a | available |
| inference | `pytorch_xpu` | 2331.6 | 70.3 | 124.0 | n/a | n/a | available |
| inference | `openvino_cpu` | n/a | n/a | n/a | n/a | n/a | parity_failed before timing |
| inference | `openvino_gpu` | 1209.5 | n/a | n/a | 241.5 | 16762.6 | available |
| training | `pytorch_cpu` | 910.0 | n/a | n/a | n/a | n/a | available |
| training | `pytorch_xpu` | 2925.7 | 182.1 | 316.0 | n/a | n/a | available |

OpenVINO GPU memory is reported as the post-measurement GPU_MEMORY_STATISTICS allocation total, not a peak; host RSS remains a separate process-level measurement.

## Startup versus throughput crossover

- Each point uses median one-time startup and median batch-1 model-call throughput; only positive finite distances are retained.
- `pytorch_cpu` versus `pytorch_xpu`: `2.471` audio seconds.
- `pytorch_cpu` versus `openvino_gpu`: `508.519` audio seconds.
- These model-only crossover points describe when the measured steady-state rate repays the measured startup difference; they are not universal short-clip latency guarantees.

## Parity diagnostics by framework and processor

The gate was evaluated on `3` fresh-process repetitions of `canonical float32 model on five public synthetic windows; no private smoke audio or rendering`. Each cell reports the maximum observed value across repetitions; the JSON retains each repetition separately.
| Parity check (applied threshold) | PyTorch CPU | PyTorch XPU | OpenVINO CPU | OpenVINO GPU |
| --- | --- | --- | --- | --- |
| Route status (must be `ok`) | passed | passed | parity_failed | passed |
| Non-finite contour values (must be 0) | 0 | 0 | 0 | 0 |
| Non-finite note values (must be 0) | 0 | 0 | 0 | 0 |
| Non-finite onset values (must be 0) | 0 | 0 | 0 | 0 |
| Maximum contour absolute error (<= 0.0205306) | 0 | 8.94069672e-07 | 0.00046145916 | 1.43051147e-06 |
| Maximum note absolute error (<= 0.0038445) | 0 | 7.4505806e-07 | 0.000364899635 | 7.15255737e-07 |
| Maximum onset absolute error (<= 0.2089347) | 0 | 1.10268593e-06 | 0.000262662768 | 1.13248825e-06 |
| Note-frame threshold disagreements (threshold 0.3; must be 0) | 0 | 0 | 0 | 0 |
| Onset threshold disagreements (threshold 0.5; must be 0) | 0 | 0 | 0 | 0 |
| Generated note-event count disagreements (must be 0) | 0 | 0 | 0 | 0 |
| Generated note-event structural disagreements (must be 0) | 0 | 0 | 0 | 0 |
| (start_time_s, end_time_s, MIDI pitch) disagreements (must be 0) | 0 | 0 | 0 | 0 |
| Pitch-bend element disagreements (must be 0) | 0 | 0 | 1 | 0 |

## OpenVINO GPU historical parity failure (pre-fix configuration)

- Status: `parity_failed` across the preserved pre-fix diagnostic repetitions.
- The worker performs the parity gate before model-only or end-to-end timing. The gate compares contour, note, and onset numeric outputs plus note/onset threshold decisions and stock note-event structure against the canonical PyTorch CPU route.
- Preserved pre-fix maximums were contour/non-finite `227040`, note/non-finite `75680`, onset/non-finite `75680`, note-threshold `733`, onset-threshold `190`, event-count `1`, event-tuple `0`, and pitch-bend `8`. These are the fixed synthetic gate only; they do not authorize timing a route that failed parity.
- This failure is historical default FP16 + PERFORMANCE evidence. The corrected FP32 + PERFORMANCE route is measured separately in the current inference, startup, end-to-end, and memory tables.

## OpenVINO GPU precision correction

This bounded post-fix diagnostic used `5` synthetic windows in one batch; it did not use private smoke audio, render audio, or measure benchmark throughput. It is retained as historical bounded diagnostic.

- Runtime: OpenVINO `2026.3.0-000--` on `Intel(R) Arc(TM) 140T GPU (16GB) (iGPU)`; device architecture `GPU: vendor=0x8086 arch=v12.74.4`.
- Driver version: `not exposed by OpenVINO; host query permission denied`.
- Compiled inference precision (`INFERENCE_PRECISION_HINT`): `float32` (requested `float32`).
- Compiled execution mode: `PERFORMANCE`; execution-mode request: `unconfigured`.
- The post-fix FP32 + PERFORMANCE diagnostic passes GPU parity; its bounded correctness result is separate from the corrected timed-route measurements.
- Diagnostic status: `parity_passed`. The corrected FP32 + PERFORMANCE route is now timed separately below under the fixed #24 contract.

| Check (applied threshold) | Result |
| --- | ---: |
| Compiled inference precision (must be float32) | float32 |
| Compiled execution mode (must remain PERFORMANCE) | PERFORMANCE |
| Non-finite contour values (must be 0) | 0 |
| Non-finite note values (must be 0) | 0 |
| Non-finite onset values (must be 0) | 0 |
| Maximum contour absolute error (<= 0.0205306) | 0.000001431 |
| Maximum note absolute error (<= 0.0038445) | 0.000000715 |
| Maximum onset absolute error (<= 0.2089347) | 0.000001132 |
| Note-frame threshold disagreements (threshold 0.3; must be 0) | 0 |
| Onset threshold disagreements (threshold 0.5; must be 0) | 0 |
| Generated note-event count disagreements (must be 0) | 0 |
| Generated note-event structural disagreements (must be 0) | 0 |
| (start_time_s, end_time_s, MIDI pitch) disagreements (must be 0) | 0 |
| Pitch-bend element disagreements (must be 0) | 0 |

## Practical conclusions supported by this run

- The batch-scaling table is the direct comparison of the four inference routes; it should be read together with startup/first-call and resource rows rather than reduced to one universal winner.
- The measured end-to-end ordering is led by `pytorch_xpu` on this fixed 8-case smoke set; it is not a claim about other audio distributions or application integration overhead.
- Startup, first-call, and crossover sections quantify the trade-off between short interactive use and longer/reused workloads without changing backend settings.
- The corrected OpenVINO GPU route is measured under requested float32 + plugin-reported PERFORMANCE; its practical position relative to PyTorch XPU, PyTorch CPU, and OpenVINO CPU is visible in all four inference tables.
- The measured full forward+backward comparison is limited to native PyTorch CPU/XPU routes.
- The original/default FP16 + PERFORMANCE failure, bounded corrected FP32 + PERFORMANCE parity pass, and newly measured corrected performance are kept as separate evidence states.

## Scope and caveats

- The report contains no source paths, filenames, IDs, hashes, or per-source predictions. Per-case end-to-end rows are anonymous case indexes only.
