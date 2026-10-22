# Joint replay helps retention, but still fails admission

Completed 2026-10-22. Training on original and wider recordings together produces a smaller original-input regression than the wider-only run, but it still fails the unchanged preservation gate. **Retain the earlier full-bootstrap initializer; do not admit these weights.** The next learning trial should return to the primary sound-loss objective with corrected data, rather than repeat parameter-only initialization.

The [card](MIXED_BOOTSTRAP_CARD.md), [frozen plan](data/mixed_bootstrap_plan.json) and [complete results](data/mixed_bootstrap_summary.json) retain the evidence. Plan ID: `e82e52730ab4c22444ff86a90d42356b789a43ce479304d8b7d00808f6a6b85f`. Executed training source: `2294bb71c505be96b2d6df0beb586ba9ac8a510f`. Both panels contain the same eight training patches as before; no validation or test source is scored.

| Input panel | Playback | Starting model | Joint replay | Joint / starting |
|---|---|---:|---:|---:|
| Original recordings | Reference | 0.3062 | 0.5583 | **1.824 — fail** |
| Original recordings | Transfer | 0.3382 | 0.5679 | **1.679 — fail** |
| Wider-note recordings | Reference | 1.3885 | 0.7692 | **0.554 — pass** |
| Wider-note recordings | Transfer | 1.2617 | 0.7276 | **0.577 — pass** |

Wider-note error falls **44.6% / 42.3%**, while original-input error rises **82.4% / 67.9%**. The allowed original-input ratio remains at most 1.1. Candidate/template comparisons pass on both panels, and every source repeat is exact, but those results cannot override preservation failure. The prior wider-only run had ratios 3.085/3.098 on original inputs. Both data weighting and update count changed, so this comparison does not isolate the causal effect of replay alone.

The model starts from the verified earlier weights with a fresh optimizer and runs 12,000 updates. All 1,212 distinct cached observations remain available through 1,616 weighted rows, with half the input draws original recordings. Source-active normalized settings error falls from 17.73% to 11.41%; category accuracy rises from 82.92% to 99.94%. Parameter diagnostics do not establish sound quality. The architecture stays at **14 dense outputs / 355,806 parameters**.

There are **zero source renders** and **192 accepted native comparison requests**, with no retries. All 32 source repeat pairs match exactly. The independent audit reproduces model weights, Adam, batch RNG, global Torch RNG and metrics exactly across all 12,000 updates. It revalidates both audited source banks, all 1,616 weighted records and 1,212 distinct observations against the actual cache, and reconstructs all 192 comparison requests and 32 hard exports. Cached/fresh prediction differences are at most `2.22e-16`, within the declared `1e-6` tolerance; categories match exactly. No additional renders are used, and discarded waveforms are not independently rescored.

Measured fit/comparison runner cost: **17.06 minutes**, **1.38 GiB** peak process-tree RAM and **113.90 MiB** evidence before the final summary. Checkpointing takes 51.20 seconds and serializes 2.64 GB across one collection and 49 full writes. The [cache preparation](MIXED_FEATURES_RESULTS.md) separately takes 3.39 minutes with zero renders/updates. Frozen-plan validation precedes the runner timer. The independent audit takes **9.35 minutes**, including its bounded native wait and cached-origin validation, and peaks at **1.14 GiB**. All declared limits pass, with zero RSS sampling errors.

Twenty-five distinct focused local checks pass across combined-data contracts, corruption rejection, RNG isolation, exact recovery, cache binding, historical/varied training and the comparison driver. Ruff and diff checks pass. The serialized real cache retains all 404 original tensor aliases and every distinct observation. GitHub Actions remains disabled.

Keep both failed initializer results and their data. The corrected observations remain useful independently of their rejected model weights. Next use the retained accepted initializer for a matched **actual sound-loss** trial, qualify its category feedback against the selected training inputs, train from both recorded views, and retain original-input regression plus counterbalanced development cases. Fix phase metadata/export consistency before the next trained artifact. A2, fresh confirmation, listening and release remain unfinished; no broader prediction heads are added.
