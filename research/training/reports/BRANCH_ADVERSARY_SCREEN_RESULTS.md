# First branch-specific note-removal setting shows no useful separation gain

The cheap [predeclared screen](BRANCH_ADVERSARY_SCREEN_CARD.md) completes both matched encoder arms. Adding the reversed note-prediction gradient produces essentially the same independent pitch/timing readouts as paired-view training alone. **None of the four leakage-reduction conditions passes.** Source-control information and representation spread remain similar between arms. This setting does not qualify for an expensive sound-training trial; it does not settle whether a stronger or differently designed separation objective would help.

This is an encoder-only experiment. Spectral reconstruction remains primary in the inverse-model training plan, but no inverse sound-loss fit or native audio comparison is performed here. Neither research encoder is a deployable patch model, and no A2/A3 gate advances.

## Fixed comparison and excluded patches

The [frozen machine-readable protocol](data/branch_adversary_screen_plan.json) uses all 1,212 distinct cached observations. It fits each encoder on **259 original training patches** and evaluates fresh readouts on the **145 excluded patches / 435 recordings**. All twelve controlled-grid patches are excluded from this new training. These patches were previously seen during the original baseline initialization, so this is development evidence rather than fresh final validation.

Both arms start from the same qualified untrained basic initializer and use identical head warmup, paired minibatches and 500 encoder updates. Only the timbre CNN is copied and trained. The sole difference is whether the temporary two-output pitch/duration predictor sends a reversed gradient into it. The imported Basic Pitch frontend, performance branch, decoder, original initializer and fourteen codec outputs remain unchanged. The temporary head has **24,674 parameters and two training outputs**; neither enters inference.

## Fresh predictors still recover pitch and timing

These predictors are fitted after encoder training, independently of the head being defeated. Standardization and predictor fitting use training patches only. Values below convert the archived normalized RMSE back to semitones and seconds. **Higher error means less recoverable performance information**, not better sound reconstruction.

| Independent held-out readout | Paired-only control | With note removal |
|---|---:|---:|
| Linear pitch RMSE, semitones | 3.9801 | 3.9802 |
| Nonlinear pitch RMSE, semitones | 3.4356 | 3.4501 |
| Linear note-length RMSE, seconds | 0.4745 | 0.4753 |
| Nonlinear note-length RMSE, seconds | 0.3789 | 0.3769 |

The frozen rule requires at least 10% higher MSE for both factors with both fresh predictors. Changes are much smaller, and the nonlinear timing predictor improves slightly instead. Shuffled-label ridge controls remain in the [complete summary](data/branch_adversary_screen_summary.json). The untouched encoder reference is retained, rather than treating changes from paired-only training as an auxiliary effect.

Patch information is similar between the two trained arms: filter balanced accuracy is 100%, unison balanced accuracy is 45.87%, and mean active continuous-control readout MSE is 0.057889 versus 0.057878. Mean cross-performance patch retrieval is 9.48% versus 9.66% among 145 candidates. No zero vectors appear and normalized spread remains above the required half-control level. Passing these retention checks alone is insufficient; all four performance-removal checks fail.

The independent twelve-patch grid also gives identical factor and patch readouts for both trained arms. On original audio, pitch accuracy is 79.17%, note-length accuracy 95.83% and cross-pitch patch accuracy 33.33%; after LUFS matching these are 90.28%, 95.83% and 54.17%. Velocity remains uninformative. Both trained arms have lower grid patch readout than the untouched encoder's 56.25%/70.83% original/matched result. That is an additional reason not to assume paired training itself is already a useful replacement.

## Compute and recovery

The complete screen takes **193.66 seconds / 0.76 GiB peak process-tree RAM**, retaining **3.41 MiB before summary**, within twenty minutes / 4 GiB / 64 MiB. It adds **zero native requests**. Each arm receives 500 real encoder updates and 100 detached-head warmup updates; a live cold-recovery check adds one software replay update per arm. The three fresh nonlinear readouts add 250 predictor updates each. These counts are separate from inverse-model sound training, which remains zero.

The first standalone audit matches all first 250 training-trace entries but stops because it compares a fresh process's global RNG without restoring the archived RNG first. Its [failure record](data/branch_adversary_screen_first_audit_failure.json) remains preserved. The corrected verifier restores checkpoint RNG before checking its preservation; it does not change data, encoders, losses or experimental settings. [Audit evidence](data/branch_adversary_screen_verification.json) records full replay and fresh-probe verification. Checkpoints and research exports retain an explicitly unqualified, refused-for-inverse-inference format.

## Consequence for the plan

The [saved-gradient inspection](data/branch_adversary_screen_gradient_magnitudes.json) measures eight fixed training minibatches at each saved checkpoint without another optimizer step. At update 250 the reversed note-gradient norm averages **5.65%** of the main paired-identity gradient norm; at update 500 it averages **3.90%**. All sixteen measured ratios range from 1.82% to 9.59%. These are unclipped encoder-gradient norms, not loss-value ratios. Correct gradient sign and nonzero pressure are established; this inspection does not establish why the setting has little effect.

The valid audit takes **54.29 seconds / 0.74 GiB**, replays all 500 updates per arm including encoder/head/Adam/RNG state, and reproduces both trained arms' fresh cached probes. The failed audit's first 250 matching trace entries remain charged separately. No new experimental arm or native request is introduced by either audit.

Do not launch a larger sound fit with this setting or sweep its weights automatically. The measured signal is relatively weak, so the result is not a strong rejection of branch-specific separation. A next test needs a justified training signal and capable adversary, not an assumption that a coefficient of one means equal pressure. Any next encoder method must retain independent leakage probes, patch information and actual reconstruction checks. Keep all fourteen outputs, original reconstruction gates, earlier-reference preservation, reserved 192-family confirmation data and final-test reservations unchanged.
