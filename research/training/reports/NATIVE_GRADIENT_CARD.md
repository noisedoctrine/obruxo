# Native sound-loss gradient qualification

Declared 2026-10-21 before collection. The learned helpers failed to provide useful control gradients. This separately costed Stage A1 alternative measures the local sound-loss changes directly in Vital. It does not extend the completed helper runs, alter their gates, train the inverse predictor or expand its 14 outputs.

Frozen plan ID: `c9f829b06e4bc3993ff59d35b8e617100a89b64de41a4997368f3c52ac6a04d5`. Seven focused local tests pass, including numerical-gradient scaling/bounds, inactive controls, actual Gumbel-score gradients, stale-estimate rejection, fixed qualification denominators and the complete sound-measurement path.

## Method

For a raw nine-control row and its two selected categories, render the canonical patch and measure the unchanged full-recording A0 spectral/envelope/side loss against the target sound. For every active continuous control, render ±0.025 in normalized units, clipped to legal bounds; divide the loss difference by the actual control span. Inactive gradients are zero. Evaluate every option of each category while holding the other category fixed. Reuse identical canonical patches within that estimate. This costs at most **22 native queries per example**: base + 18 continuous probes + three category alternatives.

The tensor objective retains the measured loss as its forward value and supplies these local estimates during backward. Category tables can connect to the existing hard Gumbel-softmax adapter. Use raw continuous values: differentiating the adapter's inactive-default blend again would double-count category effects already measured by the native tables. Export still uses raw-score argmax. These are finite-difference and conditional categorical estimates, not exact derivatives through Vital; interactions between category changes are not estimated.

## Fixed qualification

Use the existing basic pool's first 12 category-balanced validation states and both reference/transfer performances, covering all eight reserved MIDI recipes. These are previously used development states, not fresh confirmation or test data. Freeze every state, source, start and performance in `plan.json`.

Starts retain the earlier deterministic ±0.2 continuous perturbations. For index modulo three: keep both categories, change unison to its next option, or toggle filter. Thus eight of 24 cases keep correct categories and 16 start with one wrong category. Raw inactive values are preserved until native codec canonicalization.

Each case renders source and exact repeat, one gradient estimate, and two final patches. The continuous candidate takes one descent step with maximum coordinate change 0.05, scaling the measured gradient by its maximum absolute component. The joint candidate uses that same continuous step plus one category-logit step. Category logits start at margin 0.1 for the selected option; deterministic hard argmax has softmax backward at temperature one. Each logit head takes a step of one after scaling its gradient by its maximum absolute component. This deterministic qualification removes sampling variability; a separate correctness test checks connection to the actual hard Gumbel adapter. Both final candidates use hard selections.

There is no line search, best-candidate selection, source-parameter loss, neural helper, trained inverse model or additional optimization step. Every final patch receives a new native render. Fixed generator phase, fresh native workers, complete recordings, native state verification and QA remain mandatory.

## Gates and limits

Both continuous and joint candidates must independently improve legacy native MRSTFT beyond 1e-7 in at least 60% of cases, with strictly negative mean changes in both MRSTFT and the A0 spectral objective. Apply these gates separately to the 12 reference and 12 transfer cases. Missing cases, ties, rejected renders and duplicate cases cannot be counted as successes. All 24 source repeats must be sample-exact. This retains the earlier useful-update requirement; learned-forward accuracy/ranking gates are inapplicable to this direct-measurement alternative.

Caps: **624 native attempts**, **30 minutes**, **8 GiB process-tree RAM**, **32 MiB retained output**, one native worker and two CPU Torch threads in float32. The cap is 24 × (two source/repeat + at most 22 gradient queries + two final patches). No retry allowance or resume is added; interrupted/rejected requests still count and an attempted run cannot be silently restarted. Expected rendering cost is about 15–25 minutes based on prior native timings. Abort on the first reached cap.

Retain scalar losses, exact candidates, audio hashes, request/QA receipts and source identities; discard audio after each case. No training corpus is pre-rendered. The plan is frozen and source committed before execution.

A pass permits designing a matched inverse-training trial with explicit rendering cost. It does not establish inverse reconstruction quality, Stage A completion or a finished model. A failure remains a failure and does not authorize extending this card.

```powershell
python research/training/reports/scripts/run_native_gradient_screen.py --pool EXISTING_BASIC_POOL --output NEW_DIRECTORY
python research/training/reports/scripts/run_native_gradient_screen.py --pool EXISTING_BASIC_POOL --output SAME_DIRECTORY --execute
```
