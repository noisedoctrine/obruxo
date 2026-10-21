# Numerical sound-loss gradients on the settled renderer

Declared before collection on 2026-10-21. The [renderer check](NATIVE_PREROLL_CARD.md) qualifies a 32,768-frame silent lead: exact repeats, zero waveform difference versus a longer lead, and correct quadratic oscillator-level response. New requests hash the lead length, and old targets/qualification cannot be relabelled. Recheck the useful-gradient gates on this corrected observation before fresh inverse training.

Frozen plan ID: `a23dee76b9dff187770e50680a601c530877d5915b8beae5db383dc6808a9b30`. Nine targeted profile/cache/training checks pass, including refusal to relabel legacy qualification or reuse cold cached targets.

Reuse the [original native-gradient protocol](NATIVE_GRADIENT_CARD.md): the same 12 category-balanced development states, reference/transfer recipes and deterministic imperfect starts; the same nine normalized continuous controls, two category heads and **14 dense outputs**. Sixteen starts contain one wrong category. The estimator uses bounded ±0.025 native finite differences and conditional categorical loss tables. One fixed continuous step and one joint step are evaluated as hard patches, without search or source-parameter loss.

All native requests now use the explicit settled profile: **32,768 silent frames before MIDI**, followed by removal of only that added lead. No requested note or tail frame is removed. Generate all source measurements and scores anew. Do not reuse the failed inverse run's targets, model prefix or historical qualification scores. Both learned helpers remain unqualified and unused.

Keep every original gate: separately for the 12 reference and 12 transfer cases, continuous and joint updates must each improve native MRSTFT in at least 60% of cases beyond 1e-7, with strictly negative mean changes in both MRSTFT and the A0 spectral/envelope/side objective. Every source repeat must be sample-exact, all native requests must pass strict QA, every planned case must be present, and all resource limits must pass. No clipping penalty is enabled.

Caps: **624 native attempts**, **45 minutes elapsed**, **8 GiB process-tree RAM**, **32 MiB retained evidence**, one native worker, two CPU Torch threads and float32. This new renderer qualification replaces no previous result and extends no failed run. Its time allocation includes observed local startup variation; the number of cases, request cap and quality thresholds are unchanged. No inverse-model updates, test examples, Colab or GitHub Actions changes.

Freeze `plan.json` and commit source before executing `run_native_gradient_screen.py` in a new output directory. The plan pins the renderer profile and source/QA/transport hashes. Its publication audit reconstructs each request with that profile. A passing result permits a new costed A2 trial from the same historical starting model, with fresh optimizers and newly rendered targets. It does not establish trained-model quality.
