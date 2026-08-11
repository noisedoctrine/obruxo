# Basic Pitch backend benchmark

This is a fixed measurement of the canonical #23 float32 model, not an optimization search. Markdown shows medians across three fresh-process repetitions; the JSON retains min/max/total values and anonymous per-case timing.

## Executive findings

- PyTorch XPU is the observed steady-state inference leader at every tested model-call batch (`280.008` audio-seconds/second at batch 1 and `747.332` at batch 8), ahead of PyTorch CPU (`129.654` at batch 1).
- On the warmed end-to-end smoke boundary, PyTorch XPU processes `130.873` audio-seconds per wall-second, versus `56.559` for CPU and `60.948` for OpenVINO CPU.
- The model-call startup/throughput calculation records one positive CPU/XPU crossover at `3.317` audio seconds. This is a descriptive model-only crossover, not a claim about all application workloads.
- XPU pays a much larger first model call in the stored run than CPU, so short interactive calls should account for initialization and first-call latency; reused or longer workloads benefit from XPU steady-state throughput.
- The persisted OpenVINO GPU row failed the parity gate in all three repetitions before timing under the original default GPU precision configuration. It has no pre-fix throughput or memory result.
- A separate post-fix FP32 + PERFORMANCE diagnostic passes GPU parity, but it did not measure startup, throughput, end-to-end rate, or memory; those measurements remain pending a later full benchmark rerun.

## Runtime and benchmark setup

- Model: `spotify-basic-pitch-icassp-2022-v0.4.0`; precision: `float32`; smoke set: `ok` with `8` cases.
- Runtime: Python `3.12.13`, PyTorch `2.12.1+xpu`, OpenVINO `2026.3.0-000--`, NumPy `2.4.6`, SciPy `1.18.0`.
- Each route used a fresh process for each of 3 repetitions; each fixed batch used 3 warmups and 10 timed calls.
- Model-only inference and full forward+backward training used batches `[1, 2, 4, 8]`. End-to-end inference used batch 1 and covered read-only audio preparation through stock note-event materialization.
- Missing-WAV derived rendering was opt-in only; source patches, MIDI, audio, and metadata remained read-only.
- Current OpenVINO compilation explicitly sets `INFERENCE_PRECISION_HINT=float32` and leaves `EXECUTION_MODE_HINT` unconfigured; the bounded GPU diagnostic observed compiled `float32` with `PERFORMANCE` execution.

## Inference startup and initialization

Startup is separated from first-call, warmup, and steady-state timing. These persisted timing values were collected before the OpenVINO float32 precision correction; `n/a` means the route failed before that phase or the phase does not apply.

| Route | Status | Import | Construct | Checkpoint | Device move | OV convert | OV compile | Total startup |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `pytorch_cpu` | `ok` | 0.816 | 0.002 | 0.005 | n/a | n/a | n/a | 0.823 |
| `pytorch_xpu` | `ok` | 0.813 | 0.002 | 0.004 | 0.017 | n/a | n/a | 0.836 |
| `openvino_cpu` | `ok` | 0.774 | 0.002 | 0.004 | n/a | 1.542 | 0.159 | 2.519 |
| `openvino_gpu` | `parity_failed` | n/a | n/a | n/a | n/a | n/a | n/a | n/a |

### First-call and warmup observations

| Route | Batch | First call (s) | Warmup (s) | Steady median call (s) |
| --- | ---: | ---: | ---: | ---: |
| `pytorch_cpu` | 1 | 0.018 | 0.043 | 0.015426 |
| `pytorch_cpu` | 8 | 0.103 | 0.261 | 0.085077 |
| `pytorch_xpu` | 1 | 0.419 | 0.020 | 0.007143 |
| `pytorch_xpu` | 8 | 0.636 | 0.062 | 0.021410 |
| `openvino_cpu` | 1 | 0.039 | 0.056 | 0.019458 |
| `openvino_cpu` | 8 | 0.096 | 0.196 | 0.071773 |

## Steady-state inference scaling

The persisted pre-fix timing tables expose both throughput and call latency for every tested batch. Throughput is the model-call audio-equivalent rate; it excludes audio decode and stock postprocessing.

### Audio-equivalent throughput (audio-seconds/second)

| Route | Batch 1 | Batch 2 | Batch 4 | Batch 8 |
| --- | ---: | ---: | ---: | ---: |
| `pytorch_cpu` | 129.654 | 158.542 | 175.522 | 188.064 |
| `pytorch_xpu` | 280.008 | 408.659 | 623.172 | 747.332 |
| `openvino_cpu` | 102.785 | 96.907 | 146.818 | 222.926 |

### Median model-call latency (seconds)

| Route | Batch 1 | Batch 2 | Batch 4 | Batch 8 |
| --- | ---: | ---: | ---: | ---: |
| `pytorch_cpu` | 0.015426 | 0.025230 | 0.045578 | 0.085077 |
| `pytorch_xpu` | 0.007143 | 0.009788 | 0.012838 | 0.021410 |
| `openvino_cpu` | 0.019458 | 0.041277 | 0.054489 | 0.071773 |

Interpretation: XPU scales strongly with batching in this fixed workload. CPU improves more gradually. OpenVINO CPU is not monotonic at batch 2 in this run, then reaches `222.926` audio-seconds/second at batch 8; this is an observed measurement, not a tuning target.

## End-to-end audio-to-note-event throughput

This is the realistic batch-1 boundary: read-only audio open/decode, in-memory preparation, model windows, unwrapping, and stock note-event materialization. The smoke set totals 115.021 audio seconds across 8 cases.

| Route | Median wall time (s) | Min-max wall time (s) | Median audio-seconds/wall-second | Median RTF (wall/audio) |
| --- | ---: | ---: | ---: | ---: |
| `pytorch_cpu` | 2.034 | 1.973-2.072 | 56.559 | 0.01768 |
| `pytorch_xpu` | 0.879 | 0.790-1.229 | 130.873 | 0.00764 |
| `openvino_cpu` | 1.887 | 1.592-1.922 | 60.948 | 0.01641 |

The persisted pre-fix end-to-end result preserves the same ordering as model-only batch 1: XPU is fastest, OpenVINO CPU is slightly ahead of CPU, and neither timing includes the failed OpenVINO GPU route.

## CPU versus XPU full forward+backward cost

These rows measure the explicitly allowed backward cost at the native PyTorch boundary. They do not train, update, or save weights.

### Effective throughput (audio-seconds/second)

| Route | Batch 1 | Batch 2 | Batch 4 | Batch 8 |
| --- | ---: | ---: | ---: | ---: |
| `pytorch_cpu` | 29.344 | 34.481 | 35.434 | 33.566 |
| `pytorch_xpu` | 92.505 | 116.588 | 191.100 | 254.338 |

### Median forward+backward step latency (seconds)

| Route | Batch 1 | Batch 2 | Batch 4 | Batch 8 |
| --- | ---: | ---: | ---: | ---: |
| `pytorch_cpu` | 0.068156 | 0.116005 | 0.225774 | 0.476667 |
| `pytorch_xpu` | 0.021621 | 0.034309 | 0.041863 | 0.062908 |

XPU is faster than CPU at all four tested training batches in this fixed scalar-loss forward+backward measurement; this is a cost observation, not a recommendation to change the current training architecture.

## Memory and resource observations

Host RSS is a peak process measurement and is not directly interchangeable with device allocation. `n/a` is an unavailable measurement, not zero.

| Mode | Route | Host peak RSS (MiB) | XPU allocated (MiB) | XPU reserved (MiB) | Measurement note |
| --- | --- | ---: | ---: | ---: | --- |
| inference | `pytorch_cpu` | 590.6 | n/a | n/a | available |
| inference | `pytorch_xpu` | 2300.1 | 70.3 | 124.0 | available |
| inference | `openvino_cpu` | 807.0 | n/a | n/a | available |
| inference | `openvino_gpu` | n/a | n/a | n/a | parity_failed before timing |
| training | `pytorch_cpu` | 902.8 | n/a | n/a | available |
| training | `pytorch_xpu` | 2915.3 | 182.1 | 316.0 | available |

The observed XPU routes use substantially more host RSS than CPU in these fresh processes, while their recorded device allocations are much smaller than host RSS. OpenVINO GPU has no pre-fix memory observation because it failed parity before timing; the post-fix diagnostic did not measure memory.

## Startup versus throughput crossover

- The only positive finite crossover retained by the fixed formula is `pytorch_cpu` versus `pytorch_xpu` at `3.317` audio seconds.
- The formula uses median one-time startup and median batch-1 model-call throughput. In this run, CPU startup is `0.823` s and XPU startup is `0.836` s, while their batch-1 rates are `129.654` and `280.008` audio-seconds/second.
- This crossover means the XPU steady-state advantage repays its measured startup difference after roughly 3.3 audio seconds under the model-only abstraction. It does not erase XPU's larger first-call observation, and it is not a universal short-clip latency guarantee.

## Parity diagnostics by framework and processor

The gate was evaluated on `3` fresh-process repetitions of `canonical float32 model on five public synthetic windows; no private smoke audio or rendering`. Each cell reports the maximum observed value across repetitions; the JSON retains each repetition separately.

This matrix was collected before the explicit OpenVINO float32 correction. The GPU compile path did not request inference precision; the runtime default was observed as float16 with PERFORMANCE execution.

| Parity check (applied threshold) | PyTorch CPU | PyTorch XPU | OpenVINO CPU | OpenVINO GPU |
| --- | --- | --- | --- | --- |
| Route status (must be `ok`) | ok | ok | ok | parity_failed |
| Non-finite contour values (must be 0) | 0 | 0 | 0 | 227040 |
| Non-finite note values (must be 0) | 0 | 0 | 0 | 75680 |
| Non-finite onset values (must be 0) | 0 | 0 | 0 | 75680 |
| Maximum contour absolute error (<= 0.0205306) | 0 | 8.94069672e-07 | 0.00046145916 | non_finite |
| Maximum note absolute error (<= 0.0038445) | 0 | 7.4505806e-07 | 0.000364899635 | non_finite |
| Maximum onset absolute error (<= 0.2089347) | 0 | 1.10268593e-06 | 0.000262662768 | non_finite |
| Note-frame threshold disagreements (threshold 0.3; must be 0) | 0 | 0 | 0 | 733 |
| Onset threshold disagreements (threshold 0.5; must be 0) | 0 | 0 | 0 | 190 |
| Generated note-event count disagreements (must be 0) | 0 | 0 | 0 | 1 |
| (start_time_s, end_time_s, MIDI pitch) disagreements (must be 0) | 0 | 0 | 0 | 0 |

## OpenVINO GPU parity failure (pre-fix configuration)

- Status: `parity_failed` for `3` repetitions; failure code: `parity_failed`.
- The worker performs the parity gate before model-only or end-to-end timing. The gate compares contour, note, and onset numeric outputs plus note/onset threshold decisions and stock note-event structure against the canonical PyTorch CPU route.
- The component-level parity values are tabulated above. They describe the fixed synthetic gate only; they do not authorize timing a route that failed parity.
- No OpenVINO GPU throughput, latency, end-to-end rate, or memory claim is supported; no CPU fallback or substitute-device measurement was used.

## OpenVINO GPU precision correction

This bounded post-fix diagnostic used `5` synthetic windows in one batch; it did not use private smoke audio, render audio, or measure benchmark throughput.

- Runtime: OpenVINO `2026.3.0-000--` on `Intel(R) Arc(TM) 140T GPU (16GB) (iGPU)`; device architecture `GPU: vendor=0x8086 arch=v12.74.4`.
- Driver version: `not exposed by OpenVINO; host query permission denied`.
- Compiled inference precision: `float32` (requested `float32`).
- Compiled execution mode: `PERFORMANCE`; execution-mode request: `unconfigured`.
- Diagnostic status: `parity_passed`. The original full benchmark timing rows remain pre-fix and require a later rerun.

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
| (start_time_s, end_time_s, MIDI pitch) disagreements (must be 0) | 0 |

## Practical conclusions supported by this run

- For longer-running or reused workloads, PyTorch XPU is the strongest observed route: it leads steady-state model-call throughput at every batch and the warmed end-to-end smoke rate.
- For short interactive workloads, initialization and first-call behavior should be treated as part of the product latency budget. XPU's first batch-1 call is materially slower than CPU in the stored measurements even though its warmed rate is higher; the benchmark does not establish a product policy for hiding or amortizing that cost.
- OpenVINO CPU has a larger one-time conversion/compile cost and lower model-call throughput than XPU, but its warmed end-to-end smoke rate is close to CPU. It remains a measured explicit route, not an automatically preferred backend.
- XPU is also faster for the measured full forward+backward cost at all tested batches, with the resource caveat above.
- These timing conclusions describe the fixed 8-case smoke workload and the persisted pre-fix OpenVINO configuration. The bounded post-fix FP32 + PERFORMANCE diagnostic validates GPU parity but provides no speed, startup, end-to-end, or memory result; a post-fix full benchmark is still required.

## Scope and caveats

- The report contains no source paths, filenames, IDs, hashes, or per-source predictions. Per-case end-to-end rows are anonymous case indexes only.
