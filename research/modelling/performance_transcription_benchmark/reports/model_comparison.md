# Performance transcription benchmark

## Research status

**Comparative status: `incomplete_alternatives_unavailable`.** Exactly `1` of `8` configured candidates produced executable benchmark evidence in the permitted local state. The result is a Basic Pitch baseline plus explicit alternative-model blockers, not a completed comparative benchmark.

The JSON is authoritative, but this Markdown is intended to stand alone as the research finding. Quality, execution/resource cost, backward cost, representation, licensing, and quantization remain separate evidence classes; no composite winner is computed.

## What was successfully established

- Measured candidates: `basic_pitch`.
- Metadata-only or unavailable candidates: `timbre_trap_base, ymt3_plus, yptf_multi, yptf_moe_multi, muscriptor_small, muscriptor_medium, muscriptor_large`.
- Directly measured scope: Only Basic Pitch produced executable #24/#25 evidence in the permitted existing runtime and storage. Its quality and cost evidence are inherited, not rerun by #26; the persisted OpenVINO GPU cost row is pre-fix evidence.
- Sourced/model-level scope: Candidate source, checkpoint, representation, architecture boundary, native sample rate, batch semantics, and license fields are verified inventory facts; they are not performance measurements.
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
| `timbre_trap_base` | `sony/timbre-trap @ 7afe7e9b327929c099baeccd4b21973aedb94d9b` | `cwitkowitz/timbre-trap @ c1112a0` | pinned checkpoint size is not locally verifiable and the approved py312 runtime has no Timbre-Trap checkout |
| `ymt3_plus` | `mimbres/YourMT3-HuggingFace-Space @ a03c9b4d8c3b97c6a7a556768726794042127628` | `mimbres/YourMT3 @ e45ebd70398682d54b7bb1901a5216e18f3b1824` | official source and checkpoint are not present in permitted local storage |
| `yptf_multi` | `mimbres/YourMT3-HuggingFace-Space @ a03c9b4d8c3b97c6a7a556768726794042127628` | `mimbres/YourMT3 @ e45ebd70398682d54b7bb1901a5216e18f3b1824` | official source and checkpoint are not present in permitted local storage |
| `yptf_moe_multi` | `mimbres/YourMT3-HuggingFace-Space @ a03c9b4d8c3b97c6a7a556768726794042127628` | `mimbres/YourMT3 @ e45ebd70398682d54b7bb1901a5216e18f3b1824` | official source and checkpoint are not present in permitted local storage |
| `muscriptor_small` | `muscriptor/muscriptor @ c3b82e7` | `MuScriptor/muscriptor-small @ gated_revision_not_available_locally` | checkpoint is gated and no approved credential or local copy is available |
| `muscriptor_medium` | `muscriptor/muscriptor @ c3b82e7` | `MuScriptor/muscriptor-medium @ gated_revision_not_available_locally` | checkpoint is gated and no approved credential or local copy is available |
| `muscriptor_large` | `muscriptor/muscriptor @ c3b82e7` | `MuScriptor/muscriptor-large @ gated_revision_not_available_locally` | checkpoint is gated and no approved credential or local copy is available |

Sourced representation notes: Timbre-Trap is retained as a native frame/pitch output and is not given a fabricated note-event decoder; YourMT3 variants expose stock note-event output; MuScriptor exposes timing-corrected MIDI note events with stock prelude forcing. These facts describe upstream interfaces, not measured OBRUXO performance.

## What was actually executed

Only Basic Pitch produced executable evidence. The following sections consume the landed #24 and #25 reports; #26 did not rerun inference, evaluation, rendering, or quantization for this reporting revision.

### Basic Pitch quality evidence inherited from #25

- Source: `landed_issue_25_report`; eligible population: `1769`; coverage: `1.000`.
- Recorded #25 backend: `pytorch_cpu`; boundary: `#24_end_to_end_audio_to_note_events_batch_1`; precision: `float32`.
- #25 route provenance assessment: `recorded_backend_does_not_match_issue_24_observed_leader`. The existing #25 quality run records a different backend from the fastest observed #24 route. Its quality result remains a CPU-provenance measurement; no XPU full-corpus quality or equivalence claim is made.

| Quality view | Eligible | Succeeded | Failed | Coverage | Onset+pitch F1 | Onset+pitch+offset F1 | Frame F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `success_only` | 1769 | 1769 | 0 | 1.000 | 0.278 | 0.098 | 0.399 |
| `failure_penalized` | 1769 | 1769 | 0 | 1.000 | 0.278 | 0.098 | 0.399 |

- Uncertainty: `10000` seed-`0` preset-cluster replicates over `1769` clusters. These are Basic Pitch baseline intervals, not alternative-model comparisons.

### Basic Pitch execution and resource evidence inherited from #24

- Source: `landed_issue_24_report`; Routes and findings are consumed from the landed #24 report; #26 does not rerun Basic Pitch cost measurements.
- Cost evidence is route-specific; a route failure is not converted into a score or a fallback result.
- The persisted #24 cost rows below are pre-fix OpenVINO evidence where indicated; a bounded post-fix parity pass is not a corrected performance measurement.

| Mode | Route | Evidence state | Batch-1 throughput | Batch-8 throughput | E2E rate | Startup (s) | Host RSS (MiB) | XPU allocated/reserved (MiB) |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `inference` | `pytorch_cpu` | `ok` | 129.654 | 188.064 | 56.559 | 0.823 | 590.6 | n/a / n/a |
| `inference` | `pytorch_xpu` | `ok` | 280.008 | 747.332 | 130.873 | 0.836 | 2300.1 | 70.3 / 124.0 |
| `inference` | `openvino_cpu` | `ok` | 102.785 | 222.926 | 60.948 | 2.519 | 807.0 | n/a / n/a |
| `inference` | `openvino_gpu` | `pre_fix_parity_failed` | n/a | n/a | n/a | n/a | n/a | n/a / n/a |
| `training` | `pytorch_cpu` | `ok` | 29.344 | 33.566 | n/a | 0.763 | 902.8 | n/a / n/a |
| `training` | `pytorch_xpu` | `ok` | 92.505 | 254.338 | n/a | 0.833 | 2915.3 | 182.1 / 316.0 |

Historical route records:
- `openvino_gpu`: pre-fix/default `float16` + `PERFORMANCE` -> `parity_failed` before timing. The corrected route has a bounded parity result only.

### OpenVINO GPU evidence state

- Measured result (pre-fix/default): the plugin compiled the GPU route as `float16` + `PERFORMANCE`; parity failed catastrophically before timing, so the persisted cost row contains no GPU performance or resource result.
- Bounded diagnostic result (corrected): requested `INFERENCE_PRECISION_HINT=float32`, compiled `float32` + `PERFORMANCE`; status `parity_passed` on `5` public synthetic windows.
- Bounded parity metrics: non-finite values and threshold/event disagreements were `0`; maximum contour/note/onset errors were `0.000001431`, `0.000000715`, and `0.000001132`.
- Not yet measured: corrected FP32 startup, throughput, end-to-end, and resource measurements remain `not_run`; no corrected OpenVINO GPU performance claim is made.

### Basic Pitch quantization evidence

- Status: `not_applicable_no_linear`; ordinary Linear modules `0` -> `0`; engine `onednn`.
- No quantized artifact was produced, and no quantized XPU/OpenVINO/backward/batch-sweep result exists.

## What could not be executed

No alternative candidate produced a quality, execution-cost, resource, backward-cost, or quantization measurement. The unavailability reasons are concrete local prerequisites, not claims that the models are intrinsically impossible to run.

| Candidate | Status | Concrete blocker | What this prevents |
| --- | --- | --- | --- |
| `timbre_trap_base` | `unavailable` | pinned checkpoint size is not locally verifiable and the approved py312 runtime has no Timbre-Trap checkout | no comparative quality/cost result |
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

- Directly measured evidence exists only for Basic Pitch: #25 quality on 1,769 eligible pairs and #24 route/cost evidence. The persisted OpenVINO GPU cost row is the pre-fix default FP16 + PERFORMANCE parity failure and contains no GPU timing/resource result.
- The Basic Pitch evidence is a complete baseline for the landed #24/#25 contracts, not a comparison against the unavailable alternatives.
- The observed Basic Pitch route trade-offs and the historical pre-fix OpenVINO GPU failure are findings of #24; the #25 quality result remains CPU-provenanced until route provenance is resolved.

### Bounded diagnostic results

- A separate bounded #24 diagnostic compiled OpenVINO GPU with INFERENCE_PRECISION_HINT=float32 while retaining PERFORMANCE and passed parity on five public synthetic windows. This is a parity result, not a performance/resource result.
- This bounded result validates numerical parity only; it does not add a corrected OpenVINO GPU speed, startup, end-to-end, memory, or resource result.

### Supported only by verified model characteristics

- The candidate inventory establishes model identity, representation, architecture boundary, native rate/batch semantics, and licensing where verified, but none of these facts ranks execution quality or cost.
- Representation and licensing facts can inform later integration design, but they do not establish transcription quality, runtime cost, memory, or suitability.

### Comparative questions that remain unanswered

- Alternative-model quality, latency, throughput, memory, backward cost, quantization response, and quality-versus-cost trade-offs remain unanswered because those candidates were not executable in the permitted local state.
- Corrected OpenVINO GPU startup, throughput, end-to-end, and resource measurements have not yet been run; alternative-model comparisons remain blocked by unavailable prerequisites.
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
