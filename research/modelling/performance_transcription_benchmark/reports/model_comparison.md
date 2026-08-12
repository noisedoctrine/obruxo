# Performance transcription benchmark

## Research status

**Comparative status: `incomplete_alternatives_unavailable`.** Exactly `1` of `8` configured candidates produced executable benchmark evidence in the permitted local state. The result is a Basic Pitch baseline plus explicit alternative-model blockers, not a completed comparative benchmark.

The JSON is authoritative, but this Markdown is intended to stand alone as the research finding. Quality, execution/resource cost, backward cost, representation, licensing, and quantization remain separate evidence classes; no composite winner is computed.

## What was successfully established

- Measured candidates: `basic_pitch`.
- Metadata-only or unavailable candidates: `timbre_trap_base, ymt3_plus, yptf_multi, yptf_moe_multi, muscriptor_small, muscriptor_medium, muscriptor_large`.
- Directly measured scope: Only Basic Pitch produced executable #24/#25 evidence in the permitted existing runtime and storage. Its quality and cost evidence are inherited, not rerun by #26; the current #24 cost rows include the corrected OpenVINO GPU FP32 + PERFORMANCE route when present.
- Sourced/model-level scope: Candidate source, checkpoint, representation, architecture boundary, native sample rate, batch semantics, and license fields are verified inventory facts; they are not performance measurements.
- Adapter implementation scope: The repository contains an implemented pinned-official adapter path for every required candidate family. An implemented adapter is not treated as executed when its source, checkpoint, dependency, or credential prerequisite is unavailable.
- Unresolved comparative scope: No comparative quality, execution cost, resource, backward-cost, or quantization result exists for the unavailable alternatives. The intended comparative benchmark remains incomplete.

## Candidate identity and known properties

These are verified inventory facts, separated from observations produced by executing a model. A known source, representation, or license does not imply that the candidate was runnable here.

| Candidate | Family | Status | Output / representation | Native rate | Native batch | Code / weight license | Differentiable boundary |
| --- | --- | --- | --- | ---: | --- | --- | --- |
| `basic_pitch` | `basic_pitch` | `ok` | `note_and_frame`; dense=note_posterior_on_model_frames; event=stock_polyphonic_decoder | 22050 | `1, 2, 4, 8` | `Apache-2.0 / Apache-2.0` | `native_pytorch_forward` |
| `timbre_trap_base` | `timbre_trap` | `unavailable` | `frame_pitch`; native_frame_pitch | 22050 | `1` | `MIT / upstream_space_license_not_separately_declared` | `native_forward_if_upstream_runtime_is_present` |
| `ymt3_plus` | `yourmt3` | `unavailable` | `note_events`; stock_midi_note_events | 16000 | `1` | `GPL-3.0-only / Apache-2.0` | `not_exposed_by_stock_inference_path` |
| `yptf_multi` | `yourmt3` | `unavailable` | `note_events`; stock_midi_note_events | 16000 | `1` | `GPL-3.0-only / Apache-2.0` | `not_exposed_by_stock_inference_path` |
| `yptf_moe_multi` | `yourmt3` | `unavailable` | `note_events`; stock_midi_note_events | 16000 | `1` | `GPL-3.0-only / Apache-2.0` | `not_exposed_by_stock_inference_path` |
| `muscriptor_small` | `muscriptor` | `unavailable` | `note_events`; timing_corrected_midi_note_events | 16000 | `1` | `MIT / CC-BY-NC-4.0` | `not_exposed_by_stock_generation_path` |
| `muscriptor_medium` | `muscriptor` | `unavailable` | `note_events`; timing_corrected_midi_note_events | 16000 | `1` | `MIT / CC-BY-NC-4.0` | `not_exposed_by_stock_generation_path` |
| `muscriptor_large` | `muscriptor` | `unavailable` | `note_events`; timing_corrected_midi_note_events | 16000 | `1` | `MIT / CC-BY-NC-4.0` | `not_exposed_by_stock_generation_path` |

### Identity/source inventory

| Candidate | Source identity | Checkpoint identity | Availability reason |
| --- | --- | --- | --- |
| `basic_pitch` | `spotify/basic-pitch @ 9991303bba609a3b93089d13ec80d1d495083596` | `noisedoctrine/obruxo @ 918a1678465815c6f0a70009910737c164ed5a02` | available and verified |
| `timbre_trap_base` | `sony/timbre-trap @ 7afe7e9b327929c099baeccd4b21973aedb94d9b` | `cwitkowitz/timbre-trap @ 84fbd38582435c863a2d65197a75edd794888f19` | approved py312 storage has no pinned Timbre-Trap checkout or checkpoint |
| `ymt3_plus` | `mimbres/YourMT3 @ 5e66c1ea173a8186e0d20432b841d3180cc015b5` | `mimbres/YourMT3 @ 5e66c1ea173a8186e0d20432b841d3180cc015b5` | official source and checkpoint are not present in permitted local storage |
| `yptf_multi` | `mimbres/YourMT3 @ 5e66c1ea173a8186e0d20432b841d3180cc015b5` | `mimbres/YourMT3 @ 5e66c1ea173a8186e0d20432b841d3180cc015b5` | official source and checkpoint are not present in permitted local storage |
| `yptf_moe_multi` | `mimbres/YourMT3 @ 5e66c1ea173a8186e0d20432b841d3180cc015b5` | `mimbres/YourMT3 @ 5e66c1ea173a8186e0d20432b841d3180cc015b5` | official source and checkpoint are not present in permitted local storage |
| `muscriptor_small` | `muscriptor/muscriptor @ c3a50ec3f7a54361495b79ed8875ba330240324c` | `MuScriptor/muscriptor-small @ 8c127f603b807520fa465c838e9bfee8a91ada4e` | checkpoint is gated and no approved credential or local copy is available |
| `muscriptor_medium` | `muscriptor/muscriptor @ c3a50ec3f7a54361495b79ed8875ba330240324c` | `MuScriptor/muscriptor-medium @ f32236969308476e01fd3aae67357de5feb05a2d` | checkpoint is gated and no approved credential or local copy is available |
| `muscriptor_large` | `muscriptor/muscriptor @ c3a50ec3f7a54361495b79ed8875ba330240324c` | `MuScriptor/muscriptor-large @ 8809fdfbed2affa7ade94a7059e746e3880720e7` | checkpoint is gated and no approved credential or local copy is available |

Checkpoint lock status is explicit in the JSON source of truth: `locked` means the public digest and byte size are fixed; `gated_digest_not_exposed_without_access` means the immutable model revision and public size are recorded but the upstream gated service did not expose a digest without access. Neither state implies local executability.

Sourced representation notes: Timbre-Trap is retained as a native frame/pitch output and is not given a fabricated note-event decoder; YourMT3 variants expose stock note-event output; MuScriptor exposes timing-corrected MIDI note events with stock prelude forcing. These facts describe upstream interfaces, not measured OBRUXO performance.

## What was actually executed

Only Basic Pitch produced executable evidence. The following sections consume the landed #24 and #25 reports; #26 did not rerun inference, evaluation, rendering, or quantization for this reporting revision.

### Basic Pitch quality evidence inherited from #25

- Source: `landed_issue_25_report`; eligible population: `1769`; coverage: `1.000`.
- Recorded #25 backend: `pytorch_xpu`; boundary: `end_to_end_audio_to_note_event`; precision: `float32`.
- #25 route provenance assessment: `exact_issue_24_decision_consumed`. The full-corpus quality result is attributed to the exact #24-selected runtime. It does not establish quality equivalence for any other backend.

| Quality view | Eligible | Succeeded | Failed | Coverage | Onset+pitch F1 | Onset+pitch+offset F1 | Frame F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `success_only` | 1769 | 1769 | 0 | 1.000 | 0.278 | 0.098 | 0.399 |
| `failure_penalized` | 1769 | 1769 | 0 | 1.000 | 0.278 | 0.098 | 0.399 |

- Uncertainty: `10000` seed-`0` preset-cluster replicates over `1769` clusters. These are Basic Pitch baseline intervals, not alternative-model comparisons.

### Basic Pitch execution and resource evidence inherited from #24

- Source: `landed_issue_24_report`; Routes and findings are consumed from the landed #24 report; #26 does not rerun Basic Pitch cost measurements.
- Cost evidence is route-specific; a route failure is not converted into a score or a fallback result.
- The table below consumes the current #24 route rows. The historical default OpenVINO GPU failure and the bounded corrected parity diagnostic remain separate from the corrected timed route row.

| Mode | Route | Evidence state | Batch-1 throughput | Batch-8 throughput | E2E rate | Startup (s) | Host RSS (MiB) | XPU allocated/reserved (MiB) | OpenVINO GPU memory (MiB) |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| `inference` | `pytorch_cpu` | `ok` | 78.297 | 165.472 | 48.355 | 1.028 | 598.9 | n/a / n/a | n/a |
| `inference` | `pytorch_xpu` | `ok` | 225.829 | 779.279 | 95.727 | 0.983 | 2310.6 | 70.3 / 124.0 | n/a |
| `inference` | `openvino_cpu` | `parity_failed` | n/a | n/a | n/a | n/a | n/a | n/a / n/a | n/a |
| `inference` | `openvino_gpu` | `measured_corrected_fp32_performance` | 121.901 | 319.814 | 70.608 | 3.977 | 1188.8 | n/a / n/a | 236.6 / 16762.6 |
| `training` | `pytorch_cpu` | `ok` | 33.053 | 35.637 | n/a | 0.800 | 907.5 | n/a / n/a | n/a |
| `training` | `pytorch_xpu` | `ok` | 98.683 | 252.572 | n/a | 0.784 | 2926.7 | 182.1 / 316.0 | n/a |

Historical route records:
- `openvino_cpu`: `parity_failed`. The landed #24 report suppresses timing for this route.

### OpenVINO GPU evidence state

- Historical pre-fix/default result: requested `plugin default (observed float16)`, observed `plugin default (observed PERFORMANCE)`; status `parity_failed` before timing. Non-finite contour/note/onset values were `227040` / `75680` / `75680`; candidate/reference event counts were `0` / `8`. No performance or resource result is inferred from this failed route.
- Bounded diagnostic result (corrected): requested `INFERENCE_PRECISION_HINT=float32`, compiled `float32` + `PERFORMANCE`; status `parity_passed` on `5` public synthetic windows.
- Bounded parity metrics: non-finite values and threshold/event disagreements were `0`; maximum contour/note/onset errors were `0.000001431`, `0.000000715`, and `0.000001132`.
- Corrected measured result: the actual #24 smoke route compiled `float32` on `Intel(R) Arc(TM) 140T GPU (16GB) (iGPU)` with `PERFORMANCE` execution; timed-route parity is `passed`.
- Corrected startup medians: backend import `0.968` s, model conversion `2.539` s, GPU compilation `0.516` s, total startup `3.977` s; first-call / warmup at batch 1 `0.047` / `0.061` s.
- Corrected steady-state audio-equivalent throughput (audio-s/s): batch 1 `121.901`, batch 2 `262.879`, batch 4 `310.162`, batch 8 `319.814`.
- Corrected end-to-end throughput: `70.608` audio-s/s; host peak RSS `1188.8` MiB; OpenVINO GPU memory `236.6` / `16762.6` MiB where reported.
- Timed-route parity errors: non-finite values and threshold/event disagreements were `0`; maximum contour/note/onset errors were `0.000001431`, `0.000000715`, and `0.000001132`.

### Basic Pitch quantization evidence

- Status: `not recorded`; ordinary Linear modules `n/a` -> `n/a`; engine `n/a`.
- No quantized artifact was produced, and no quantized XPU/OpenVINO/backward/batch-sweep result exists.

## What could not be executed

No alternative candidate produced a quality, execution-cost, resource, backward-cost, or quantization measurement. The unavailability reasons are concrete local prerequisites, not claims that the models are intrinsically impossible to run.

| Candidate | Status | Concrete blocker | What this prevents |
| --- | --- | --- | --- |
| `timbre_trap_base` | `unavailable` | approved py312 storage has no pinned Timbre-Trap checkout or checkpoint | no comparative quality/cost result |
| `ymt3_plus` | `unavailable` | official source and checkpoint are not present in permitted local storage | no comparative quality/cost result |
| `yptf_multi` | `unavailable` | official source and checkpoint are not present in permitted local storage | no comparative quality/cost result |
| `yptf_moe_multi` | `unavailable` | official source and checkpoint are not present in permitted local storage | no comparative quality/cost result |
| `muscriptor_small` | `unavailable` | checkpoint is gated and no approved credential or local copy is available | no comparative quality/cost result |
| `muscriptor_medium` | `unavailable` | checkpoint is gated and no approved credential or local copy is available | no comparative quality/cost result |
| `muscriptor_large` | `unavailable` | checkpoint is gated and no approved credential or local copy is available | no comparative quality/cost result |

- `timbre_trap_base`: the pinned checkpoint size is not locally verifiable and no approved Timbre-Trap checkout is present in the existing runtime/storage.
- `ymt3_plus`, `yptf_multi`, `yptf_moe_multi`: the official source/checkpoint material is not present in permitted local storage.
- `muscriptor_small`, `muscriptor_medium`, `muscriptor_large`: checkpoints are gated and no approved credential or local copy is available; no login, terms acceptance, or acquisition was attempted.

## Conclusions by evidence class

### Directly supported by measured results

- Directly measured evidence exists only for Basic Pitch: #25 quality on 1,769 eligible pairs and #24 route/cost evidence. The current #24 inference comparison includes corrected OpenVINO GPU FP32 + PERFORMANCE startup, throughput, end-to-end, parity, and resource measurements when that route is present.
- The Basic Pitch evidence is a complete baseline for the landed #24/#25 contracts, not a comparison against the unavailable alternatives.
- The measured #24 model-call throughput winners were batch 1: `pytorch_xpu` (225.829 audio-s/s), batch 2: `pytorch_xpu` (379.212 audio-s/s), batch 4: `pytorch_xpu` (578.354 audio-s/s), batch 8: `pytorch_xpu` (779.279 audio-s/s); the end-to-end winner was `pytorch_xpu` (95.727 audio-s/s). These are Basic Pitch route findings, not alternative-model results.
- The #25 quality result remains CPU-provenanced until route provenance is resolved.

### Bounded diagnostic results

- A separate bounded #24 diagnostic compiled OpenVINO GPU with INFERENCE_PRECISION_HINT=float32 while retaining PERFORMANCE and passed parity on five public synthetic windows. This is a parity result, not a performance/resource result.
- The bounded corrected parity diagnostic remains a correctness result; the corrected GPU timing/resource rows above are the separate measured result.

### Supported only by verified model characteristics

- The candidate inventory establishes model identity, representation, architecture boundary, native rate/batch semantics, and licensing where verified, but none of these facts ranks execution quality or cost.
- Representation and licensing facts can inform later integration design, but they do not establish transcription quality, runtime cost, memory, or suitability.

### Comparative questions that remain unanswered

- Alternative-model quality, latency, throughput, memory, backward cost, quantization response, and quality-versus-cost trade-offs remain unanswered because those candidates were not executable in the permitted local state.
- The alternative-model comparison remains incomplete because the required candidate executions were unavailable; this reporting update did not rerun #25 evaluation or #26 candidate inference.
- No ranking, quality estimate, cost estimate, quantization effect, or integration recommendation is assigned to any unavailable candidate.

## What is required before the intended comparison can be completed

The following prerequisites must become available before a legitimate comparative run can be attempted; none is acquired or changed by this report revision:

- An approved, immutable Timbre-Trap source checkout plus a verifiable pinned `tt-orig.pt` checkpoint and the already-permitted runtime prerequisites.
- The approved YourMT3 source checkout and all three exact immutable checkpoints selected in `models.yaml`.
- Approved access and local copies of the three gated MuScriptor checkpoints, without exposing credentials or taking account actions in this task.
- A permitted existing-runtime preflight for each candidate, followed by the fixed common #25 quality population and applicable #24 cost routes. Missing dependencies must be recorded as blockers rather than installed or substituted.
- After executable results exist, report both success-only and failure-penalized quality views, route/resource/backward applicability, and the fixed CPU dynamic-Linear quantization scope separately; do not synthesize a composite winner.

## Contract and privacy limits

- Quality: `landed_issue_25_metrics_and_10000_replicate_seed_0_bootstrap`; cost: `landed_issue_24_end_to_end_boundary`; quantization: `cpu_dynamic_qint8_ordinary_linear_only`.
- #26 reused the exact #25 eligible population/audio provenance and the landed #24 results; it did not render, rebuild audio, rerun Basic Pitch, or use alternative fallbacks.
- Private paths, pair identifiers, source filenames, row predictions, gated weights, and local run state are excluded from this public report.
