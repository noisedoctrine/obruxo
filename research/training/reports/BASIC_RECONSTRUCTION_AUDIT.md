# Basic reconstruction evidence and next diagnostic

Prepared 2026-10-19 from existing reports. **The small model learns useful structure, but reliable sound reconstruction remains unproven.** This audit starts the revised staged plan without another training run. It does not promote a checkpoint or consume test data.

The unchanged model predicts nine continuous controls and two categorical choices (five logits), with 355,806 trainable parameters. The new explicit basic scope enforces this existing output space and rejects broad pools. Full-preset parsing/rendering remains available independently.

## Existing sound evidence

Lower spectral error is better. Each primary score is the equal-weight mean of reference and transfer error. These are synthetic development measurements, not calibrated listening scores.

| Development panel | Basic model | Training-only retrieval | Fixed template | Model reference | Model transfer |
|---|---:|---:|---:|---:|---:|
| Original 21 patches | 0.9124 | 1.2746 | 3.1011 | 0.8304 | 0.9943 |
| Additional 23 patches | 0.8397 | 1.9768 | 6.5702 | 0.7760 | 0.9034 |

The candidate is the existing 404-state, seed-0, 16,000-update model. Retrieval is the frozen original 205-training-record bank in both panels. Every displayed method scored every planned case. The model improves on both baselines and transfers some sound structure to new performances, but mean error alone cannot establish that the resulting patches sound sufficiently similar. The historical comparisons involved development selection and repeated inspection; neither panel is a fresh confirmation set now.

Source-setting diagnostics identify questions, not established acoustic causes. On these panels, envelope release normalized error is about 0.332 and 0.325. Unison voices are recovered in 12/21 and 17/23 cases. Relative RMS error is about 0.20–0.21. Different valid settings can sound similar, so these numbers do not justify chasing exact parameter recovery or expanding the decoder.

## Fixed diagnostic sample

Use the additional panel's five-number summary of the candidate's existing primary error. Case numbers below are the stored zero-based case IDs. The [audit JSON](data/basic_reconstruction_audit.json) retains full state/recipe/performance IDs, input hashes and the selection rule.

| Error rank out of 23 | Stored case | Historical primary error |
|---:|---:|---:|
| 1 (minimum) | 7 | 0.4047 |
| 6 (lower quartile) | 1 | 0.6014 |
| 12 (median) | 8 | 0.8993 |
| 18 (upper quartile) | 22 | 1.0400 |
| 23 (maximum) | 14 | 1.3436 |

The next step is the [70-request controlled native diagnostic](../STAGED_RECONSTRUCTION.md): compare source and baseline prediction with separate envelope, level, filter and unison/stereo corrections, under both performances and fixed random phase. Corrections use source parameters solely to diagnose possible learning directions. They cannot be presented as model predictions. Listening and the intervention results are still pending.

The next training change will follow the diagnosed sound failure. Eight-LFO fitting, full routing graphs, mapping curves, custom oscillator curves and exact seeds are outside this stage.

## Reproduction and checks

```powershell
python research/training/reports/scripts/summarize_basic_reconstruction.py --output research/training/outputs/basic-audit-reproduction.json
```

The script uses only the published [original results](data/state_coverage_summary.json) and [additional results](data/fresh_validation_summary.json). It validates complete cases, records their file hashes and refuses to overwrite an existing result. No native renders, audio files, model updates or external downloads are needed.

Local validation: 32 scope/training tests passed, including rejection before frontend creation, actual head counts, same-pool feature reuse, exact interrupted recovery, export/evaluation reload and unchanged legacy contract fields. The actual historical 16,000-update checkpoint also reloads under its original contract with nine continuous outputs and five categorical logits. Ruff and whitespace checks pass. These checks establish implementation behavior, not new reconstruction quality.
