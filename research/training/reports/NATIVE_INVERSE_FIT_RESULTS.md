# Native inverse run stopped on an invalid training probe

The frozen [matched training run](NATIVE_INVERSE_FIT_CARD.md) stopped during update **35**, after **34 committed updates per arm**. A categorical probe of four unison voices produced one sample at **1.0319849**, above the render QA amplitude limit. The run correctly retained the rejection, but its training loop could not turn an invalid trial into a learning signal. No fit or validation comparison was performed; this result establishes neither model improvement nor a reconstruction-quality failure.

[Verified checkpoint-prefix evidence](data/native_inverse_prefix_summary.json) · Frozen training source `468cd25` · Plan `1530130a22f4e6b7143a3ba66ab2248a231e20d77e3805b83e960d23f5734365`

- **743 native attempts:** 742 accepted, one rejected for `audio.clipping`. All source recordings were accepted; the rejection was a proposed alternative, not a source example.
- **35 charged optimizer attempts**, with 34 committed matched updates. The uncommitted attempt is not treated as training progress.
- **2,274.72 seconds (37.91 minutes)** and **1,584,209,920 bytes (1.48 GiB)** peak process-tree RAM. Within the original caps.
- Both model weights, both Adam states and the Gumbel RNG replay exactly from the recorded native losses, with no additional renders. Every accepted probe used by that prefix has a matching native receipt.
- The audit reconstructs the exact rejected patch and request from the checkpoint. Four voices remain in the vocabulary; it is this combination of controls and performance that exceeds the amplitude limit.

The original exception summary reports zero committed updates because the outer function lost access to the returned trace when fitting raised. The checkpoint proves 34. The original summary is preserved unchanged, and the audit explicitly records both values. Subsequent code must retain committed progress when an exception occurs.

The original failed run will not be silently resumed or relabelled successful. The correction is a declared constrained objective: valid probes use the same measured spectral loss, while clipping-only trials receive a finite validity penalty. Targets and final output audits must continue to pass strict QA. A small native check must first reproduce this failure and verify that the penalty points away from it.

A continuation can reuse the checked records and exact matched checkpoint, while preserving all spent attempts. Its cumulative budget must include this failed run, the new check and the remaining training/audits within the original 3,328 requests and four-hour deadline. No new prediction heads, test states or model promotion are involved.
