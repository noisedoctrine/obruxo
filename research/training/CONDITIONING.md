# Performance conditioning comparison

The optional `cross_attention` model lets each short section of the sound representation attend to the frozen performance features. The existing model combines pooled sound and performance features once. This comparison tests whether a more direct connection helps recover the same patch across different notes.

The new path uses four sound positions, eight performance positions and a 64-wide attention layer with four heads. It starts with a zero residual gain: all shared weights and initial predictions exactly match the baseline under the same seed. It adds 70,465 parameters; this is a comparison of practical architectures, not a claim to isolate attention from extra model capacity.

`train --conditioning cross_attention` selects the new model. The default remains `concat`, with unchanged saved contracts and outputs. Exports record the architecture and restore it for ordinary audio-only inference. Comparing different architectures requires explicit `compare --allow-architecture-comparison`; codec, frontend, renderer and precision contracts must still match.

The fixed screen in `reports/scripts/run_conditioning_screen.py` reuses the qualified 126-view paired-alpha feature cache. Both arms start fresh, see identical batches for 1,000 updates, keep the same labels and frozen frontend, and perform no new training renders. It verifies identical common initialization, initial predictions, replay contents, batch RNG and per-example use counts.

Native validation compares ordinary predictions on ten balanced validation patches, each with reference and reserved transfer performances. The render budget is 120 planned attempts plus 12 recovery attempts. Retrieval and the validation panel stay fixed. Final-test states remain reserved, and this small single-seed screen cannot promote a finished release model.
