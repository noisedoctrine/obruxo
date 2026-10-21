# Native startup-state diagnostic

Declared before execution on 2026-10-21. The rejected four-voice probe and its valid source/base choices all peak at sample 63. Lowering the oscillator level changes that peak much less than its quadratic amplitude scale suggests. Before using the tested clipping penalty in training, determine whether native controls are still settling when notes begin. The proposed penalty continuation remains disconnected.

Freeze **six native requests**, at most ten minutes, 8 GiB process-tree RAM and 64 MiB output. Plan ID: `b84d2218a1ece0e1abddb1618b0a5ecf1c09ffb877ecbfaa7415de4276a21095`.

Capture the original source and rejected patch with no timing change. Their audio hashes must match historical renders, proving the observation hook does not change audio. Save native control state after rendering as well as the state loaded before rendering.

Then render the source, rejected patch and level-reduced patch after a silent MIDI lead, repeating the rejected patch once. The deterministic timing calculation chooses a 24,055-sample lead for this performance. Every note-on and note-off moves by exactly that many samples, and the recording grows by the same number. Crop only the added lead; retain all original requested note and tail frames. Fix all generator phases as before.

Report before/after native settings, cold and delayed peaks, silent-prefix peak, the repeated delayed waveform equality and the level response. The schema marks oscillator level as quadratic; compare the level-reduced waveform with the corresponding squared raw-level ratio. These are diagnostic questions, not a model-quality acceptance test. Keep every original QA decision and retain the unmodified captured waveforms. Do not normalize, limit or discard a true note attack.

This check adds six requests to the 743 stopped training requests and 11 previous validity-check requests. It neither restarts training nor authorizes extending the failed run. If the renderer's observation changes after settling, old target caches and the partial sound-trained models cannot be silently reused as corrected evidence. Freeze the corrected renderer and a new costed plan before further training.
