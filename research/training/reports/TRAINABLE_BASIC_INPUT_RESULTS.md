# Trainable audio features: bridge passed

The new input branch passes its correctness/resource checks. **Every one of 1,616 starting patches remains identical**, and both the copied stereo CNN and pitch projection receive sound-cost gradients after the first correction update. This permits preparation of a native fitting comparison; it does not establish improved sound reconstruction.

The [frozen card](TRAINABLE_BASIC_INPUT_CARD.md) changes one architectural constraint: the direct correction can now learn its audio representation. The accepted original predictor remains frozen. The old 27,118-weight correction and copied 230,864 input weights train together, for **257,982 trainable / 613,788 total weights**. The Basic Pitch frontend remains frozen. Outputs stay **9 continuous + 3 unison scores + 2 filter scores = 14 values**, with the same 11 codec selections, defaults and assets.

The current direct model's three recording types are all training inputs. The previous failure is therefore incomplete fitting across performances, before independent confirmation. Earlier fully trainable models also failed their quality gates; the new trial retains the more recent direct-control, rehearsal and actual-render acceptance repairs rather than assuming that unfreezing alone is proven to help.

## Evidence

Implementation `fa00aa3`, plan `34716c6aefca3faa930e8e18d2fd2a4b3de9d9564f22f9b765a814988747e062` and the [declaration](data/trainable_basic_input_bridge_declaration.json) are frozen before execution. Preparation takes 2.31 seconds. The complete real-cache check takes **50.44 seconds / 0.74 GiB**, with zero RSS sampling errors. The [external service](data/trainable_basic_input_bridge_execution_service.json) also exits successfully within five minutes / 4 GiB and releases its scoped awake request.

- All **1,616 recipes / 1,212 recordings / 404 training states** retain identical starting continuous values, calibrated choice probabilities and hard presets. Initial and simulated models both cold-load with matching outputs across all recipes.
- All **24 previously qualified native points** match their exact raw controls, hard choices, input features and source identities. Their responding control/choice derivatives reach the output correction. No stale native estimate is used at a changed prediction.
- The zero final layer correctly makes both copied input branches' initial gradients zero. After a simulated update, gradient L1 is 0.207940779 for the stereo CNN and 0.021618705 for the pitch projection; both branches change while original predictor bytes stay fixed.
- Two software updates and one replay reproduce model/Adam/Python/NumPy/Torch state exactly. All **224 software cost evaluations** are recomputed at their current controls. These **three simulated optimizer applications** are separately labelled: **zero new native requests and zero experimental training updates**.
- All 32 reference recordings retain their accepted initial patches and sound baselines. Every source evidence hash is checked before and after execution. **22 local tests**, Ruff and diff checks pass, including a complete simulated measured-acceptance fit and exact encoder/Adam/RNG recovery after an interrupted native trial.

Full rows, native derivative checks and recovery measurements are in the [summary](data/trainable_basic_input_bridge_summary.json); the [plan](data/trainable_basic_input_bridge_plan.json) binds the inputs and implementation.

## Next

Prepare the small matched native comparison from this untrained initializer. Both arms use the new trainable mask, unchanged initial presets and the fixed 48-update observation schedule. Keep sound loss primary, fresh native gradients, actual-render acceptance, no label auxiliary, earlier-sound preservation and all six reconstruction gates. Compute the new parameter-control identities and evidence bound before launch; verify the first real prefix includes the copied input layers' optimizer state. Do not import failed trained weights or widen the output vocabulary.

No A2/A3 admission or release is claimed. Fresh families/performances, second seed, listening and bundle qualification remain open. Work stays local; GitHub Actions remains disabled and Colab deferred.
