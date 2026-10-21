# Native inverse run stopped on an invalid training probe

The frozen [matched training run](NATIVE_INVERSE_FIT_CARD.md) stopped during update **35**, after **34 committed updates per arm**. A categorical probe of four unison voices produced one sample at **1.0319849**, above the render QA amplitude limit. The run correctly retained the rejection. Subsequent [renderer diagnostics](NATIVE_PREROLL_RESULTS.md) traced this sample to a startup transient while controls were settling. No fit or validation comparison was performed; this result establishes neither model improvement nor a reconstruction-quality failure.

[Verified checkpoint-prefix evidence](data/native_inverse_prefix_summary.json) · Frozen training source `468cd25` · Plan `1530130a22f4e6b7143a3ba66ab2248a231e20d77e3805b83e960d23f5734365`

- **743 native attempts:** 742 accepted, one rejected for `audio.clipping`. All source recordings were accepted; the rejection was a proposed alternative, not a source example.
- **35 charged optimizer attempts**, with 34 committed matched updates. The uncommitted attempt is not treated as training progress.
- **2,274.72 seconds (37.91 minutes)** and **1,584,209,920 bytes (1.48 GiB)** peak process-tree RAM. Within the original caps.
- Both model weights, both Adam states and the Gumbel RNG replay exactly from the recorded native losses, with no additional renders. Every accepted probe used by that prefix has a matching native receipt.
- The audit reconstructs the exact rejected patch and request from the checkpoint. Four voices remain in the vocabulary. The corrected renderer accepts this same patch after settling its controls, with peak **0.24395**.

The original exception summary reports zero committed updates because the outer function lost access to the returned trace when fitting raised. The checkpoint proves 34. The original summary is preserved unchanged, and the audit explicitly records both values. Subsequent code must retain committed progress when an exception occurs.

The initial diagnostic hypothesis was to penalize clipping-only probes. That policy was investigated but **is not enabled in training**: waveform and state checks instead identified a renderer defect. The [qualified correction](NATIVE_PREROLL_RESULTS.md) renders a 32,768-frame silent lead before MIDI and removes only the added lead. Exact repeats, the quadratic level response and agreement with a longer lead pass the unchanged gates. Strict target, probe and final-output QA remains in force.

This failed run and its original records remain historical evidence. Its cold targets and 34-update model prefix cannot be reused as corrected training. Requalify the numerical-gradient method on newly rendered settled audio, then freeze a fresh costed trial from the same historical starting model with fresh optimizers and targets. No new prediction heads, test states or model promotion are involved.
