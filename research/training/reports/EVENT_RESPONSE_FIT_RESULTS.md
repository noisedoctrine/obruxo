# Corrected samples do not yet produce learned control responses

The matched short fitting check completed all 2,000 declared update attempts. **Both helpers still respond almost like a model that ignores the control change.** The corrected measurements improve absolute prediction error, but fail every declared response-fit group. A third full fit of this helper is not justified by this result.

[Frozen card](EVENT_RESPONSE_FIT_CARD.md) · [Re-scored checkpoint evidence](data/event_response_fit_summary.json)

Both arms started from identical seed-zero initialization and used the same 12 training pairs, pair selections, architecture, optimizer and 1,000 updates. The uniform arm used the old samples; the event arm used the corrected spectral frames and complete envelopes. Both final checkpoints were evaluated on the same corrected training measurements. This is training-fit evidence, not independent reconstruction quality.

| Control group | No-change error | Uniform helper error | Event helper error | Event pairs beating no change | Gate |
|---|---:|---:|---:|---:|---|
| Attack | 0.10075 | 0.10076 | 0.10238 | 0/3 | Failed |
| Decay | 0.12668 | 0.12668 | 0.12893 | 1/3 | Failed |
| Release | 0.07267 | 0.07267 | 0.07361 | 0/3 | Failed |
| Detune | 0.69727 | 0.69727 | 0.69782 | 0/3 | Failed |

The event arm's mean absolute measurement error was **0.84232**, versus **0.89164** for uniform sampling: a **5.53% reduction**. This passes the declared absolute-error safeguard but does not offset the failed control-response gates. Neither mean response error nor case-level success establishes useful envelope learning.

The sampling repair remains valid: its native preflight proved that the signals are present in the corrected measurements. This fit shows that preserving those measurements alone did not make the current model learn the responses within this budget. It does not prove all neural helpers are incapable, or that no longer fit could help. It also provides no evidence for spending another 12,000 updates on this unchanged mechanism.

## Resources and verification

The run took **468.44 seconds (7.81 minutes)**: 146.22 seconds for uniform sampling and 321.69 seconds for event sampling, including each arm's scoring. Equal updates were not equal compute; the event arm processed more frames. Peak process-tree RSS was **1,001,074,688 bytes (0.93 GiB)** with zero sampling errors. Retained checkpoints, logs and reports occupy **68,037,424 bytes (64.89 MiB)**.

Exactly 1,000 attempts per arm were charged before optimizer execution, with both final checkpoints at step 1,000. There were no native requests, validation/test examples, inverse-model updates or qualified exports. The run stayed within its 30-minute, 8-GiB RAM and 256-MiB output caps.

Twenty focused local tests pass, including shared-loop recovery and preservation of the full-screen minimum-pair safeguard. The publication script reloads and hashes the final checkpoints, verifies every input feature, re-scores both models, recomputes all gates and checks the durable 2,000-attempt sequence. Both re-scores exactly match the saved results. Ruff and whitespace checks pass.

## Next mechanism

Keep these checkpoints diagnostic-only and both earlier helpers unqualified. Investigate the training plan's separately costed native-gradient alternative: measure the spectral-loss effect of legal control changes directly in Vital, then use those measurements to supply local continuous and categorical learning signals. These would be numerical gradient estimates, not derivatives supplied by the plugin. Preserve phase control, complete-recording spectral scoring, deterministic export and the same 14-output scope.

First qualify the proposed updates against real audio under a frozen native-request budget. Only then integrate them into a matched inverse-training comparison. No claim of Stage A completion follows from this failed fit or from implementing the alternative.
