# Performance transcription benchmark

The JSON report is authoritative. Quality, execution cost, resources, licensing, representation, and quantization are reported separately; no composite winner is computed.

## Candidate status

| Model | Family | Status | Quality |
| --- | --- | --- | --- |
| `basic_pitch` | `basic_pitch` | `ok` | `reported` |
| `timbre_trap_base` | `timbre_trap` | `unavailable` | `unavailable` |
| `ymt3_plus` | `yourmt3` | `unavailable` | `unavailable` |
| `yptf_multi` | `yourmt3` | `unavailable` | `unavailable` |
| `yptf_moe_multi` | `yourmt3` | `unavailable` | `unavailable` |
| `muscriptor_small` | `muscriptor` | `unavailable` | `unavailable` |
| `muscriptor_medium` | `muscriptor` | `unavailable` | `unavailable` |
| `muscriptor_large` | `muscriptor` | `unavailable` | `unavailable` |

## Contract

- Quality: `landed_issue_25_metrics_and_10000_replicate_seed_0_bootstrap`
- Cost: `landed_issue_24_end_to_end_boundary`
- Quantization: `cpu_dynamic_qint8_ordinary_linear_only`
- Timbre-Trap remains frame-only; no synthetic note-event decoder is included.
- Private paths, pair identifiers, source filenames, and row predictions are excluded.

## Quality, cost, and quantization details

### `basic_pitch`


| Quality view | Eligible | Succeeded | Failed | Coverage | Onset+pitch F1 | Onset+pitch+offset F1 | Frame F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `success_only` | 1769 | 1769 | 0 | 1.000000 | 0.278166 | 0.098394 | 0.398800 |
| `failure_penalized` | 1769 | 1769 | 0 | 1.000000 | 0.278166 | 0.098394 | 0.398800 |

- Execution status: `measured`.

| Route | Status |
| --- | --- |
| `pytorch_cpu` | `ok` |
| `pytorch_xpu` | `ok` |
| `openvino_cpu` | `ok` |
| `openvino_gpu` | `parity_failed` |
| `pytorch_cpu` | `ok` |
| `pytorch_xpu` | `ok` |
- Quantization: `not_applicable_no_linear`; ordinary Linear modules 0 → 0; engine `onednn`.

### `timbre_trap_base`

- Availability: `unavailable` — pinned checkpoint size is not locally verifiable and the approved py312 runtime has no Timbre-Trap checkout.
- Quality: unavailable; no score is synthesized.
- Execution status: `unavailable`.

### `ymt3_plus`

- Availability: `unavailable` — official source and checkpoint are not present in permitted local storage.
- Quality: unavailable; no score is synthesized.
- Execution status: `unavailable`.

### `yptf_multi`

- Availability: `unavailable` — official source and checkpoint are not present in permitted local storage.
- Quality: unavailable; no score is synthesized.
- Execution status: `unavailable`.

### `yptf_moe_multi`

- Availability: `unavailable` — official source and checkpoint are not present in permitted local storage.
- Quality: unavailable; no score is synthesized.
- Execution status: `unavailable`.

### `muscriptor_small`

- Availability: `unavailable` — checkpoint is gated and no approved credential or local copy is available.
- Quality: unavailable; no score is synthesized.
- Execution status: `unavailable`.

### `muscriptor_medium`

- Availability: `unavailable` — checkpoint is gated and no approved credential or local copy is available.
- Quality: unavailable; no score is synthesized.
- Execution status: `unavailable`.

### `muscriptor_large`

- Availability: `unavailable` — checkpoint is gated and no approved credential or local copy is available.
- Quality: unavailable; no score is synthesized.
- Execution status: `unavailable`.

## Interpretation

- No quality score is published for unavailable models or an empty eligible population.
- Cost rows remain separate by route and are unavailable when the fixed smoke input or candidate runtime is unavailable.
- Representation and licensing are inventories, not an integration decision.
- No candidate is promoted without executable common-corpus evidence.
