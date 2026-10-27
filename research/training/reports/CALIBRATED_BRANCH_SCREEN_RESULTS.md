# Equal gradient pressure gives a small directional effect, below the fixed gate

The [single calibrated follow-up](CALIBRATED_BRANCH_SCREEN_CARD.md) increases the note-removal coefficient from one to **25.22367**, derived from eight training-only batches before encoder training. The median auxiliary/main encoder-gradient ratio is exactly one at calibration; measured mean ratios remain **1.21 at update 250** and **0.93 at update 500**. This is a materially acting auxiliary signal, unlike the first weak setting. No held-out result selects the coefficient.

All four cached held-out performance readouts move toward less readable pitch/timing, with **2.6–4.8% higher MSE** than the exact paired-only control. However, every change remains below the predeclared 10% requirement. **The directional screen does not pass.** Patch-information retention checks pass between arms. Do not promote an encoder or launch a larger sound fit with this setting.

## Unchanged comparison and fresh readouts

The [frozen plan](data/calibrated_branch_screen_plan.json) inherits the same initializer, head, 259/145 whole-patch split, 500 updates, data order, learning rates and probes from the [first experiment](BRANCH_ADVERSARY_SCREEN_RESULTS.md). The independently replayed paired-only control is reused exactly, without retraining. All twelve controlled-grid patches remain excluded from new encoder training. The only experimental change is the training-derived coefficient, ramped over the first 100 updates and then fixed.

| Fresh held-out readout | Paired-only control | Calibrated note removal | MSE increase |
|---|---:|---:|---:|
| Linear pitch RMSE, semitones | 3.9801 | 4.0311 | 2.58% |
| Nonlinear pitch RMSE, semitones | 3.4356 | 3.5168 | 4.78% |
| Linear note-length RMSE, seconds | 0.4745 | 0.4841 | 4.10% |
| Nonlinear note-length RMSE, seconds | 0.3789 | 0.3839 | 2.65% |

Higher prediction error here means less readable performance information, not better reconstructed audio. Source-filter balanced accuracy stays at 100%; unison balanced accuracy changes from 45.87% to 46.39%. Mean active continuous-control readout MSE changes from 0.057889 to 0.057824, and mean patch retrieval from 9.48% to 9.83%. Spread and zero-vector checks pass. All complete metrics, denominators and decisions remain in the [summary](data/calibrated_branch_screen_summary.json).

The independent grid does not show a general separation gain: original-audio pitch classification rises from 79.17% to 83.33%, note-length classification stays at 95.83%, and cross-pitch patch identity falls from 33.33% to 29.17%. LUFS-matched grid results remain unchanged. These descriptive checks reinforce the decision to withhold admission rather than treating the small regression-probe trend as success. Velocity still cannot be tested with this template.

## Saved-head and normalization inspection

A separate [read-only post-hoc inspection](data/calibrated_branch_normalization_summary.json) checks the exact input seen by the temporary head, its actual saved predictions and the raw encoder mean/scale. It changes no screen rule or candidate weight.

The trained adversary is not simply predicting a constant: on excluded patches it recovers pitch with **3.62-semitone RMSE**, versus an 11.54-semitone mean baseline. A fresh linear probe on its exact normalized input achieves **3.67 semitones**, so substantial pitch information remains accessible even to that input. The adversary's known task is still readily solvable after training.

There is also a coverage limitation: per-example layer normalization removes the raw vector mean and rescales its spread, while the downstream encoder representation retains those channels. A two-value mean/scale readout still reduces pitch MSE by **18.69%** against the mean baseline. This identifies a training-head coverage gap; it does not establish the cause of the small effect or demonstrate a successful normalization repair. Do not respond with another coefficient increase. The next separation design needs to address the objective/data and information exposed to its adversary, while retaining real reconstruction checks.

## Compute, replay and limits

Candidate learning and evaluation take **39.14 seconds / 0.74 GiB peak RAM / 0.93 MiB before summary**, within twenty minutes / 4 GiB / 64 MiB. There are 100 detached-head warmup updates and 500 candidate encoder/head updates. The prior control is not retrained. No new native renders, inverse sound-fit updates, final-test use or reserved-confirmation audio is added. The original inverse model remains unchanged at fourteen dense outputs.

The [valid standalone replay](data/calibrated_branch_screen_verification.json) takes **42.10 seconds / 0.74 GiB**, reproducing the coefficient, all candidate encoder/head/Adam/RNG state, fresh probes, grid results, diagnostics, frozen decision and reused control. The first direct verification matches all 500 training-trace entries but stops at a full-RNG comparison because only Torch was seeded in its new process. Its [failure record](data/calibrated_branch_first_audit_failure.json) is preserved. Use `audit_calibrated_branch_screen.py` for standalone verification: it restores archived Python, NumPy and Torch RNG before invoking unchanged replay code. Both audits add only software replay; neither adds an experimental arm or changes the candidate.

The original [weak screen](BRANCH_ADVERSARY_SCREEN_RESULTS.md) remains intact. This follow-up rules out insufficient gradient magnitude as the only explanation for that setting's small effect, but does not reject branch-specific separation in general. These are two encoder-only settings, one seed and reused development data. Spectral reconstruction remains primary in inverse training; the three-case shared-learning prerequisite, all six sound groups, earlier-reference preservation, fresh confirmation, listening and release remain unfinished.
