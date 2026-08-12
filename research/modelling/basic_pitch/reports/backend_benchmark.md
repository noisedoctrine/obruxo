# Basic Pitch backend benchmark

This is a fixed measurement of the canonical #23 float32 model, not an optimization search. Markdown shows medians across three fresh-process repetitions; the JSON retains min/max/total values and anonymous per-case timing.

## Executive findings

- Measured steady-state inference winners by model-call batch: batch 1 `pytorch_xpu` (225.829 audio-seconds/second), batch 2 `pytorch_xpu` (379.212 audio-seconds/second), batch 4 `pytorch_xpu` (578.354 audio-seconds/second), batch 8 `pytorch_xpu` (779.279 audio-seconds/second).
- On the warmed end-to-end smoke boundary, `pytorch_xpu` is fastest at `95.727` audio-seconds per wall-second.
- The fixed startup/throughput calculation retains `1` positive finite crossover point(s); these are descriptive model-only results, not claims about all application workloads.
- First-call and startup trade-offs remain visible in the dedicated tables, so short interactive calls must be interpreted separately from reused or longer workloads.
- The corrected OpenVINO GPU route now has measured FP32 + PERFORMANCE startup, batch scaling, end-to-end, parity, and resource results.
- The original/default FP16 + PERFORMANCE OpenVINO GPU parity failure remains preserved as historical evidence; it is not conflated with the corrected timed route.
- The bounded post-fix FP32 + PERFORMANCE parity diagnostic is retained as a separate correctness result; the corrected route's performance/resource measurements are now included in the tables below.

## Runtime and benchmark setup

- Model: `spotify-basic-pitch-icassp-2022-v0.4.0`; precision: `float32`; smoke set: `ok` with `8` cases.
- Runtime: Python `3.12.13`, PyTorch `2.12.1+xpu`, OpenVINO `2026.3.0-000--`, NumPy `2.4.6`, SciPy `1.18.0`.
- Each route used a fresh process for each of 3 repetitions; each fixed batch used 3 warmups and 10 timed calls.
- Model-only inference and full forward+backward training used batches `[1, 2, 4, 8]`. End-to-end inference used batch 1 and covered read-only audio preparation through stock note-event materialization.
- Missing-WAV derived rendering was opt-in only; source patches, MIDI, audio, and metadata remained read-only.

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
| `pytorch_cpu` | `ok` | 1.015 | 0.004 | 0.010 | n/a | n/a | n/a | 1.028 |
| `pytorch_xpu` | `ok` | 0.948 | 0.003 | 0.009 | 0.024 | n/a | n/a | 0.983 |
| `openvino_cpu` | `parity_failed` | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| `openvino_gpu` | `ok` | 0.968 | 0.002 | 0.006 | n/a | 2.539 | 0.516 | 3.977 |

### First-call and warmup observations

| Route | Batch | First call (s) | Warmup (s) | Steady median call (s) |
| --- | ---: | ---: | ---: | ---: |
| `pytorch_cpu` | 1 | 0.030 | 0.077 | 0.025544 |
| `pytorch_cpu` | 8 | 0.120 | 0.300 | 0.096693 |
| `pytorch_xpu` | 1 | 0.773 | 0.028 | 0.008856 |
| `pytorch_xpu` | 8 | 0.948 | 0.062 | 0.020532 |
| `openvino_gpu` | 1 | 0.047 | 0.061 | 0.016407 |
| `openvino_gpu` | 8 | 0.094 | 0.163 | 0.050029 |

## Steady-state inference scaling

The corrected timing tables expose both throughput and call latency for every tested batch, including OpenVINO GPU under explicit float32 inference. Throughput is the model-call audio-equivalent rate; it excludes audio decode and stock postprocessing.

### Audio-equivalent throughput (audio-seconds/second)

| Route | Batch 1 | Batch 2 | Batch 4 | Batch 8 |
| --- | ---: | ---: | ---: | ---: |
| `pytorch_cpu` | 78.297 | 110.257 | 140.898 | 165.472 |
| `pytorch_xpu` | 225.829 | 379.212 | 578.354 | 779.279 |
| `openvino_gpu` | 121.901 | 262.879 | 310.162 | 319.814 |

### Median model-call latency (seconds)

| Route | Batch 1 | Batch 2 | Batch 4 | Batch 8 |
| --- | ---: | ---: | ---: | ---: |
| `pytorch_cpu` | 0.025544 | 0.036279 | 0.056778 | 0.096693 |
| `pytorch_xpu` | 0.008856 | 0.010548 | 0.013832 | 0.020532 |
| `openvino_gpu` | 0.016407 | 0.015216 | 0.025793 | 0.050029 |

Interpretation: batch 1 winner is `pytorch_xpu` at 225.829 audio-seconds/second; batch 2 winner is `pytorch_xpu` at 379.212 audio-seconds/second; batch 4 winner is `pytorch_xpu` at 578.354 audio-seconds/second; batch 8 winner is `pytorch_xpu` at 779.279 audio-seconds/second. These are fixed-workload observations, not tuning targets.

## End-to-end audio-to-note-event throughput

This is the realistic batch-1 boundary: read-only audio open/decode, in-memory preparation, model windows, unwrapping, and stock note-event materialization. The smoke set totals 115.021 audio seconds across 8 cases.

| Route | Median wall time (s) | Min-max wall time (s) | Median audio-seconds/wall-second | Median RTF (wall/audio) |
| --- | ---: | ---: | ---: | ---: |
| `pytorch_cpu` | 2.379 | 2.357-2.383 | 48.355 | 0.02068 |
| `pytorch_xpu` | 1.202 | 1.196-1.203 | 95.727 | 0.01045 |
| `openvino_gpu` | 1.629 | 1.566-1.686 | 70.608 | 0.01416 |

The measured end-to-end winner is `pytorch_xpu`; this ordering is specific to the fixed smoke boundary and includes no failed route.

## CPU versus XPU full forward+backward cost

These rows measure the explicitly allowed backward cost at the native PyTorch boundary. They do not train, update, or save weights.

### Effective throughput (audio-seconds/second)

| Route | Batch 1 | Batch 2 | Batch 4 | Batch 8 |
| --- | ---: | ---: | ---: | ---: |
| `pytorch_cpu` | 33.053 | 33.343 | 32.575 | 35.637 |
| `pytorch_xpu` | 98.683 | 151.480 | 205.752 | 252.572 |

### Median forward+backward step latency (seconds)

| Route | Batch 1 | Batch 2 | Batch 4 | Batch 8 |
| --- | ---: | ---: | ---: | ---: |
| `pytorch_cpu` | 0.060509 | 0.119964 | 0.245590 | 0.448974 |
| `pytorch_xpu` | 0.020267 | 0.026406 | 0.038882 | 0.063348 |

Training winners by batch are: batch 1 `pytorch_xpu` (98.683 audio-seconds/second); batch 2 `pytorch_xpu` (151.480 audio-seconds/second); batch 4 `pytorch_xpu` (205.752 audio-seconds/second); batch 8 `pytorch_xpu` (252.572 audio-seconds/second). This is a cost observation, not a recommendation to change the current training architecture.

## Memory and resource observations

Host RSS is a peak process measurement and is not directly interchangeable with device allocation. `n/a` is an unavailable measurement, not zero.

| Mode | Route | Host peak RSS (MiB) | XPU allocated (MiB) | XPU reserved (MiB) | OV GPU current allocation (MiB) | OV GPU device memory (MiB) | Measurement note |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| inference | `pytorch_cpu` | 598.9 | n/a | n/a | n/a | n/a | available |
| inference | `pytorch_xpu` | 2310.6 | 70.3 | 124.0 | n/a | n/a | available |
| inference | `openvino_cpu` | n/a | n/a | n/a | n/a | n/a | parity_failed before timing |
| inference | `openvino_gpu` | 1188.8 | n/a | n/a | 236.6 | 16762.6 | available |
| training | `pytorch_cpu` | 907.5 | n/a | n/a | n/a | n/a | available |
| training | `pytorch_xpu` | 2926.7 | 182.1 | 316.0 | n/a | n/a | available |

OpenVINO GPU memory is reported as the post-measurement GPU_MEMORY_STATISTICS allocation total, not a peak; host RSS remains a separate process-level measurement.

## Startup versus throughput crossover

- Each point uses median one-time startup and median batch-1 model-call throughput; only positive finite distances are retained.
- `pytorch_cpu` versus `openvino_gpu`: `645.619` audio seconds.
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
