# Native curve ingestion and active supervision

The [new curve-input profile](../CURVE_INPUTS.md) supports original one-point sources and decimal curve coordinates/powers without changing older renderer or decoder identities. The requested source remains in the recipe. Its applied constant/float32 import is separately hashed and checked against native state. Only metadata is published.

## Native behavior and qualification

The [failure-control plan](data/curve_input_discovery_plan.json) and [result](data/curve_input_discovery_summary.json) make nine phase-controlled requests with exact float32 coordinates. The two active one-point LFO requests exit the pinned native worker. Their explicit constant reference and all mapping/custom-curve controls succeed: seven accepted, two expected worker exits. This is a reproduced boundary defect, not an audio-QA rejection or a failed model prediction. A separate readback inspection shows that arbitrary decimal coordinates also round to float32; rounding is distinct from the LFO rendering failure.

The [singleton qualification](data/curve_input_native_plan.json) and [result](data/curve_input_native_summary.json) accept all 144 attempts. They cover all eight LFO slots, mapping slots 1/32/64, all three oscillator slots under custom modes 13/14/15 and two constant levels. The new import matches the explicit constant references exactly, including repeats. Mapping and custom-curve cases additionally match unexpanded, native-float32 one-point sources under the earlier profile. Applied state, original source hashes and the complete curve-change ledger pass. All seven gates pass. The panel takes 339.44 seconds with 399.45 MiB peak process-tree memory, below 8 GiB.

The [multi-point precision plan](data/curve_input_precision_plan.json) and [result](data/curve_input_precision_summary.json) accept all 30 requests. Three-point LFOs, mappings and custom curves in all three modes retain their smooth/unsmoothed behavior and match explicitly rounded float32 references exactly, including repeats. All five gates pass. Across the two positive panels, all 174 requests succeed. Correctness rendering fixes oscillator/sampler random phases and uses a fresh process per request.

The one-point LFO import defines constant semantics so the source becomes renderable. It does not claim an audio comparison with the crashing unmodified request. Full 775-scalar and six-family native readback remains required; curve expansion and precision conversion are the only new import operations.

## Correcting the training population

The first [training plan](data/curve_input_initial_training_plan.json) and [transport/recovery result](data/curve_input_initial_training_summary.json) pass their original eleven gates, but the [activity audit](data/curve_input_initial_activity.json) shows that presence was insufficient:

| Singleton family | Stored in initial replay | Active in initial replay | Active in corrected replay |
|---|---:|---:|---:|
| LFO | 9 | 0 | 9 |
| Connection mapping | 9 | 5 | 8 |
| Custom oscillator curve | 12 | 0 | 11 |

The source snapshot for that initial result is recorded in the [compatibility receipt](data/curve_input_compatibility.json). No model is promoted from it. The corrected source policy deliberately connects the chosen LFO/mapping and enables the chosen oscillator/custom mode. It fails source generation if the selected curve is masked out. The batches use every retained example in each update. This fixes source coverage rather than removing the components or weakening a quality threshold.

The corrected [training plan](data/curve_input_training_plan.json) and [result](data/curve_input_training_summary.json) pass all twelve gates. Sixteen acknowledged attempts produce sixteen accepted examples from distinct training states, with no rejections or errors. All 28 selected singleton curves are actively supervised. Every retained example is sampled in all three updates. Model, optimizer, Python/NumPy/Torch and batch RNG, replay and cursor match after recovery from update one; recovery renders no new audio. Export/reload and frozen frontend checks pass. Original/applied source hashes and per-curve ledgers match for every retained example.

The measured training segment takes 63.87 seconds, excluding initial pool verification. Peak measured training/recovery process memory is 1,126.92 MiB, below 8 GiB. The source pool contains 32 states: 25 training, two validation and five reserved test descendants under frozen ancestry. Only training reference performances are rendered. The 5,725,689-parameter model retains the existing output dimensions; three updates verify ingestion and supervision, not useful reconstruction or strict tiny-overfit.

All 60 distinct selected local checks pass across the initial/corrected runs, including renderer contracts, expression MIDI, recipe isolation, codec/pool/export round trips and recovery. The corrected 17-test curve suite passes in 168.94 seconds. Ruff, whitespace and current/historical source fingerprints pass. The [compatibility receipt](data/curve_input_compatibility.json) verifies that the parent combined codec still matches its prior qualified identity and the existing train-only LFO bank is unchanged.

Factory assets, raw source pools, audio and weights remain local. Broader corpus/layout admission, decoder refit policy, active graph/range coverage, data scaling, useful validation quality and release remain open. Final-test audio stays reserved, Actions stays disabled and no model is promoted.
