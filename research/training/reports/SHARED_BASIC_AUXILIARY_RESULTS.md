# Pitch-removal loss has little effect on shared reconstruction

The [frozen three-sound comparison](SHARED_BASIC_AUXILIARY_CARD.md) is complete and audited. The auxiliary version reduces mean combined sound error by **0.235%** and MRSTFT by **0.214%** against sound-only, below the declared 5% directional target. Independent probes still recover pitch and note length. **Do not adopt this auxiliary setting or sweep its coefficient.** This result does not reject performance separation as a goal.

Both shared models learn substantial average improvements on the three sounds. Neither passes the per-input learning prerequisite or earlier-sound preservation, so **A2 remains active and neither model is promoted**. The fourteen-output prediction space stays fixed.

## Reconstruction on the three sounds

Each row averages the same seven input/playback comparisons, covering all three input recordings. Lower combined sound error is better. These are reused training/development cases, not fresh validation or human listening results.

| Sound, identified by its earlier held note | Initial | Sound-only | Sound plus auxiliary | Sound-only improvement |
|---|---:|---:|---:|---:|
| MIDI 36 | 0.47856 | 0.26600 | 0.26564 | 44.42% |
| MIDI 61 | 0.56477 | 0.30772 | 0.30703 | 45.51% |
| MIDI 83 | 0.63750 | 0.46245 | 0.46107 | 27.46% |
| All 21 comparisons | 0.56028 | 0.34539 | 0.34458 | 38.35% |

All three mean reductions exceed 25%, but every sound misses at least one of its individual requirements. On MIDI 36, the held-input pair remains worse than the saved settings-trained control: **0.36372 versus 0.26912** for sound-only. MIDI 61 improves only 7.4% on that pair on average, and one playback remains worse than the control. For MIDI 83, the original-input transfer comparisons get slightly worse. The [complete per-input breakdown](data/shared_basic_auxiliary_publication.json) preserves all nine input groups and both metrics.

The original 32-patch panel also regresses: sound-only combined error rises **14.81%** and MRSTFT **9.85%** against initialization. The stronger earlier-sound preservation gate fails too. Both models pass only two of the six full-panel groups, the alternate-input groups. Concentrating updates on three sounds improves their average fit while harming other known sounds; this is not reliable reconstruction across the supported inputs.

## What the probes show

Fresh readouts use 125 development patches excluded from this run's sound training and rehearsal, with fitting on the fixed 259-patch auxiliary split. The baseline encoder had seen this population during its earlier initialization. Higher note-readout MSE means less recoverable performance information.

| Readout | Auxiliary MSE change against sound-only |
|---|---:|
| Linear pitch | +0.33% |
| Linear note length | +1.20% |
| Nonlinear pitch | -1.42% |
| Nonlinear note length | +0.68% |

These small, mixed changes do not demonstrate separation. Patch retrieval and source-control readouts remain similar. The auxiliary term has a measured nonzero encoder gradient and uses the full timbre representation, but that does not establish a useful representation or reconstruction benefit.

## Execution and verification

Both arms complete **48 updates**, accepting 45 proposed updates each. The original initializer, data order, Adam settings, actual-sound acceptance, inward derivative, 14 outputs, qualified renderer, fixed oscillator/sampler phases and 32,768-frame preroll remain as declared. The auxiliary coefficient is measured once at native update two and then follows its frozen ramp.

Native execution uses **12,025 accepted requests**, with no timeouts or recovery, in **6 h 15 min / 1.65 GiB peak process-tree RAM**. The archive is **0.60 GiB before summary**, within the ten-hour / 4-GiB / 2-GiB allocation. The scoped awake request is released normally. The [run summary](data/shared_basic_auxiliary_summary.json) retains the actual accounting.

The [offline audit](data/shared_basic_auxiliary_verification.json) exactly reconstructs all 48 updates of both models, Adam, calibration and RNG; it cold-loads both exports and independently rescores **324 retained endpoint/repeat waveforms**. Fixed-phase repeats are exact. Training-probe audio was transient: its saved scalar measurements are replayed, not independently rescored. The audit takes **162.63 seconds / 1.09 GiB**, adding no native requests or experimental fitting.

The publication check revalidates the frozen source/input bindings and all **383 scientific evidence files**. Operational heartbeat/completion files are excluded from that immutable scientific manifest. The earlier execution declaration and prefix report remain historical snapshots. Eighteen focused software tests, Ruff 0.16.2 and diff checks passed for the unchanged implementation; Actions remains disabled.

## Next training task

Keep the small model and sound objective. The next data change should give every one of these three patches the same note and duration coverage. In the current focused subset, the held notes are tied to patch identity: 36, 61 and 83 respectively. Cross the same low/middle/high notes and short/long durations with every patch, then freeze a comparison against this audited sound-only control before another fit. This tests a concrete data hypothesis; the present result does not prove that the missing combinations caused the failure.

Do not expand controls, tune another adversary, extend these fitted weights or score the 192 reserved roots. The shared-learning prerequisite, complete reconstruction/preservation gates, fresh confirmation, second seed, clean human listening and release remain unfinished.
