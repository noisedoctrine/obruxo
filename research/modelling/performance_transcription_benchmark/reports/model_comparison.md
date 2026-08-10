# Performance transcription benchmark

The JSON report is authoritative. Quality, execution cost, resources, licensing, representation, and quantization are reported separately; no composite winner is computed.

## Candidate status

| Model | Family | Status | Quality |
| --- | --- | --- | --- |
| `basic_pitch` | `basic_pitch` | `unavailable` | `unavailable` |
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

## Interpretation

- No quality score is published for unavailable models or an empty eligible population.
- Cost rows remain separate by route and are unavailable when the fixed smoke input or candidate runtime is unavailable.
- Representation and licensing are inventories, not an integration decision.
- No candidate is promoted without executable common-corpus evidence.
