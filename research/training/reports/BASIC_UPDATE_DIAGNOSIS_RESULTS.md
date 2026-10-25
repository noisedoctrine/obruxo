# Saved updates reveal competing sound objectives

Both audited 48-update fits replay exactly, including all model/Adam/RNG state and 10,896 measured scalar probes. Inspection takes **21.52 seconds / 0.73 GiB RAM**, with zero new renders or experimental training updates. The frozen [plan](data/basic_update_diagnosis_plan.json), [declaration](data/basic_update_diagnosis_declaration.json) and [complete measurements](data/basic_update_diagnosis_summary.json) retain both trajectories and every observation.

| Measurement | Sound only | Continuous auxiliary |
|---|---:|---:|
| Updates inspected | 48 | 48 |
| Updates with sampled control change over 0.025 | 6 | 6 |
| Updates with any panel input over 0.025 | 21 | 21 |
| Median sampled maximum control change | 0.00985 | 0.00966 |
| Largest panel control change | 0.04705 | 0.06632 |
| Reference observations with estimated worsening | 38 / 96 | 34 / 96 |
| Updates with any reference estimated worsening | 31 | 30 |
| Updates with estimated combined sound improvement | 45 | 46 |

The 0.025 value is the native finite-difference spacing, not a proven safe movement limit. Large jumps occur, but cannot explain every conflict: the sound-only final update moves the held-input controls by at most 0.01715 and one reference by 0.00287, yet both sounds worsen. No sampled observation changes either hard category during its update. This does not establish that categorical learning is sufficient or that unobserved panel choices never change.

Only the final update has independently audited after-update audio. For that update, both runs improve the original paired input and one reference, while worsening the held input and the other reference. Every estimated sign agrees with those eight measured changes:

| Final update: combined sound-error change | Sound only | Auxiliary |
|---|---:|---:|
| Original paired input | -0.039053 | -0.018180 |
| Held paired input | +0.004419 | +0.003529 |
| First reference | +0.000962 | +0.003653 |
| Second reference | -0.002621 | -0.001327 |

These are eight observations from one update in each run, not an independent population-wide result. The other updates' changes are approximations from saved native gradients and conditional category costs; curvature and interactions are unmeasured. The result supports checking an update rule that protects each sampled sound before another long fit. Merely reducing large steps would not directly address the observed conflict at smaller steps. A protected proposal must first demonstrate useful progress rather than achieving preservation by doing nothing.

No model is admitted. The auxiliary candidate remains rejected; the preceding sound-only model remains the stronger research reference. Keep all 14 outputs and all original quality gates. Full A2/A3, broader generalization, second-seed, listening and release work remain unfinished. Actions stays disabled and Colab remains deferred.
