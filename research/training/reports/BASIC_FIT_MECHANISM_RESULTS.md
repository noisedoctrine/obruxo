# Held-note failures need more than pitch repair or retaining old probes

The [read-only audit](data/basic_fit_mechanism_summary.json) finds the correct source note at the strongest saved note-feature bin for **all twelve held recordings**. All contour peaks are within half a semitone. This pooled summary does not test full transcription or prove that every note feature is adequate, but it gives no support for wrong held-note identity as the immediate explanation.

Each held and alternate recording receives only **two scheduled primary sound-loss updates** in the 48-update fit. Each original recording receives four, plus the original-reference rehearsal schedule. This is an exposure count, not evidence that simply extending the fit would solve it.

The audit compares complete, already measured patches under both the input performance and the original performance. It never combines the lowest errors from different patches. Both wider panels retain all twelve patches; incomplete candidate pairs are counted separately. A final sound-model prediction remains eligible if it beats its earlier probes.

| Input type | Parameter-control mean | Final sound-model mean | Best earlier probe or final sound prediction | Best measured sound choice versus control |
|---|---:|---:|---:|---:|
| Held note | 0.48669 | 0.53329 | 0.51180 | **5.16% worse** |
| Alternate | 0.65737 | 0.59573 | 0.52723 | **19.80% better** |

The final prediction is worse than an earlier probed patch on 9/12 held inputs and 11/12 alternate inputs. Retaining alternatives may help, especially on alternate inputs, but the measured held alternatives still do not close the mean gap. This is post-hoc selection from known training measurements, not a deployable predictor or new quality confirmation. Probe waveforms were discarded; their scalar values are covered by the completed fit's independent replay, rather than a new waveform rescore.

Implementation `ae6d389` precedes the [frozen audit](data/basic_fit_mechanism_plan.json). Four focused tests, Ruff and diff checks pass. The real check takes **6.66 seconds / 0.56 GiB**, retaining 19,288 bytes within five minutes / 4 GiB / 8 MiB. It adds zero renders, inference calls, optimizer applications or confirmation scores. Known MIDI is used only to check the saved note evidence, never as a predictor input.

The proposed next component check is [direct control fitting on three existing training sounds](DIRECT_BASIC_CONTROL_CARD.md). It asks whether the sound objective and bounded native updates can fit individual sounds before another network change. Software preparation can proceed; execution selection should incorporate the actual listening feedback and a complete validated implementation. No extra heads, longer repeat of the failed network fit, parameter-only initializer, fresh confirmation or second seed is launched here.
