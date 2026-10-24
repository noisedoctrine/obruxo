# Both-performance training still fails original-note reconstruction

Completed and independently verified on 2026-10-24. **Reject both new models and keep the accepted initializer.** Scoring each inferred patch against both performances corrects the earlier objective gap, but does not establish useful basic reconstruction. Original-note error still increases; both wider-input groups improve. The retained original 32 patches also fail. Development scoring is skipped and no model is promoted.

The [frozen card](INPUT_TRANSFER_NATIVE_FIT_CARD.md) keeps the same 14 outputs, 355,806 parameters, initial weights, 64 training patches, three input recordings, 128 paired updates, optimizer and quality gates. Each identical candidate patch is now rendered under its input and reciprocal performances; their measured sound losses have equal weight. The parameter control's final weight hash is identical to the preceding paired trial, confirming its matched execution remains unchanged.

| Input recordings, all 64 patches | Initial spectral error | Parameter training | Sound training | Change from initial | Fit gate |
|---|---:|---:|---:|---:|---|
| Original notes | 0.313734 | 0.483039 | 0.368790 | +17.5% | Fail |
| Wider held notes | 1.477426 | 1.379246 | 1.282328 | -13.2% | Pass |
| Wider alternate pattern | 1.955185 | 1.944166 | 1.580946 | -19.1% | Pass |

Values are mean multiresolution spectral distance, where lower is better. The unchanged gates also check the combined sound objective against initialization and the parameter control. On the retained 32 patches, original-note error rises from 0.323321 to 0.405556 (+25.4%). The prior transfer-only fit's original-input regression was +23.1%; the corrected objective reduces that regression but still misses the required improvement. This is a training-fit failure, before independent generalization or listening.

The [complete independent proof](data/input_transfer_fit_verification.json) exactly replays all 128 updates of both models, Adam and Python/NumPy/Torch state, reconstructs every hard request, checks source inputs/targets and rescores all **576 candidate comparison waveforms**. Its added native requests and training updates are zero. The [terminal service records](data/input_transfer_fit_completion.json) confirm both execution and the automatically queued audit exit successfully and release their scoped active-work requests.

All **10,952 native requests are accepted**: 10,184 training/source requests and 768 final comparisons. There are 128 charged optimizer attempts per arm, no interrupted attempts and no recovered timeouts. The native window takes **4 hours 55 minutes**, peaks at **2.41 GiB** process-tree RAM, and retains **2.87 GiB** of evidence. The complete audit takes **17 minutes 50 seconds / 2.47 GiB**. Every declared time, request, RAM and archive limit passes. Infrastructure failure cannot explain this result.

Before another costly fit, use the saved training evidence to separate input-representation sensitivity from opposing update directions. Inspect how the sound and note-information branches change predictions across performances, and whether the two observations in a paired update push shared weights in opposite directions. These are diagnostics, not alternative quality scores. Choose a structural next change from that evidence, retain the 14-output scope and sound objective, and keep every existing preservation and development gate. Do not retry parameter-only initialization, select an earlier quality checkpoint or promote either diagnostic model.
