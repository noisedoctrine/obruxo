# Let sound training change the audio representation

The [measured-acceptance fit](NATIVE_ACCEPTANCE_FIT_RESULTS.md) improves the sound objective on 44/48 training batches, but fails two performance groups and the earlier error-tail rule. All three recording types already receive training. This is a failure to fit those training performances reliably, before fresh-family generalization. Earlier full-network fits also exist and failed their quality gates; this proposal does not claim that a trainable encoder has never been tried.

The current direct correction acts on a frozen stereo CNN and pitch projection. Keep its successful direct-control coordinate system, reference-sound rehearsal, production choices and measured native acceptance, but let a copied input branch learn from the sound objective as well. This changes which audio features can be learned. It adds no prediction capabilities, source parameters, routes, curves or random-seed heads.

| Model component | Frozen weights | Trainable weights |
|---|---:|---:|
| Accepted original predictor, used as the baseline | 355,806 | 0 |
| Copied stereo CNN | 0 | 5,456 |
| Copied pitch projection | 0 | 225,408 |
| Existing 832 → 32 → 14 direct correction | 0 | 27,118 |
| **Total** | **355,806** | **257,982** |

There are **613,788 total weights**. The final output remains **9 continuous values + 3 unison scores + 2 filter scores = 14 dense values**, reduced to the same 11 codec selections at export. The Basic Pitch audio frontend and input preprocessing remain frozen. Only the inverse network's copied CNN/pitch projection and direct correction train. Both matched arms must use this same architecture and mask.

Initialize the copies from the accepted original predictor, with the same seeded hidden layer and zero final correction layer. Every starting continuous value, calibrated category probability and exported preset must remain identical. The zero final layer initially blocks the copied input layers' gradients; after the first update, sound-loss derivatives must reach both branches. The original predictor stays byte-identical, in evaluation mode and without gradients throughout training and restoration.

Before any native training, freeze this bridge's implementation, input hashes and numerical plan. Allocate **five minutes / 4 GiB process-tree RAM / 64 MiB evidence** separately for preparation and execution. Use the complete existing 1,616-recipe / 1,212-recording / 404-training-state cache. Reject changed or held-out inputs, changed renderer identities and missing phase/preroll metadata. Check all starting predictions and cold exports; bind all 32 reference sounds to their accepted baseline.

Reuse the 24 previously qualified native estimates only at their identical initial raw controls and hard choices. Verify their existing responding output derivatives, with zero new renders. After predictions move, use a clearly labelled software cost evaluator that recomputes its costs at every new patch to test gradients and recovery. Two software updates plus one exact replay must reproduce model/Adam/Python/NumPy/Torch state, change both copied input branches and preserve cold predictions on all 1,616 recipes. These three simulated optimizer applications are not experimental training or evidence of better Vital reconstruction.

The implementation passes **22 local tests**, including unchanged old-model behavior, initial/cold identity, input-gradient access, both feature branches changing under sound-cost derivatives, complete measured-acceptance fitting and exact recovery after a simulated native-trial timeout. Ruff and diff checks pass. The real-cache bridge result remains pending until execution.

Only a passed bridge permits preparation of a separately costed small native comparison. It must keep the current 48-update source schedule, real spectral/envelope/stereo objective, no source-label auxiliary, fresh native gradients, reference rehearsal and final quality gates. Use the same untrained initial predictions in both arms; do not import rejected trained weights. Recalculate control identities and evidence bounds for the new architecture before launch. Real prefix replay must verify the input-layer optimizer state as well as the original correction. Do not expand scope or call a working gradient path a reconstruction success.

Local CPU float32, two threads, fixed oscillator/sampler random phases and 32,768-frame preroll remain in force. Battery/checkpoint execution is already authorized; scoped awake requests only. Actions stays disabled and Colab remains deferred. A2/A3, fresh confirmation, second seed, listening and release are still open.
