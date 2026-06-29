# Experiment 3 Findings: Frequency-First Residual Peeling

## What question did we test?

Can a fixed base of 15 provisional stock shapes plus 17 observed corpus medoids support a residual stack that handles common shapes early, sends completed cases to an explicit no-op, and leaves later layers to work on the remaining presets?

We also tested whether one or two scaling values per layer are worth their model-output cost, whether topology conditioning remains useful, and whether a small circular-shift decoder offers a useful convolution-like extension.

## How were the codebooks constructed?

Experiment 2 capped repeated shape signatures at eight while fitting. Experiment 3 removes that cap. Every training occurrence contributes to the objective, so common configurations exert proportionally more pressure on early code selection.

The base layer is fixed at 32 choices:

- 15 provisional stock geometries;
- 17 observed training medoids, selected greedily by total reduction in corpus MSE relative to the stock set.

Each of four residual layers contains 16 choices. Code 0 is an exact no-op. The other 15 codes are observed training residuals selected by total error reduction across the residual population left by the frozen preceding layers. This is frequency-first residual peeling, not equal weighting by unique signature.

Authors remain disjoint between codebook fitting and the 1,605-instance held-out evaluation set. Final RMSE is measured over all 1,024 sampled phases of each curve.

## Did presets move onto the no-op branch?

Yes, particularly with topology conditioning. For categorical decoding without gains:

| Layer | Shared no-op share | Topology-conditioned no-op share |
|---|---:|---:|
| 1 | 45.0% | 54.1% |
| 2 | 47.1% | 54.1% |
| 3 | 47.7% | 60.5% |
| 4 | 49.1% | 60.6% |

This supports the proposed mechanism: many common cases are handled by the base or early residual layers, while later topology dictionaries operate on a smaller unresolved population. The shared chain shows only a modest rise, so the peeling is real but incomplete.

With learned gains, raw code-0 frequency is less meaningful because choosing any residual with a gain near zero is functionally another no-op. The categorical branch is therefore the cleaner test of the peeling hypothesis.

## What did compact scaling buy us?

The full four-layer results are:

| Configuration | Dense outputs | Median RMSE | P95 RMSE |
|---|---:|---:|---:|
| Shared, no scaling | 96 | 0.013317 | 0.143424 |
| Topology, no scaling | 99 | 0.013849 | 0.140571 |
| Shared, one scalar/layer | 100 | 0.002894 | 0.108700 |
| Topology, one scalar/layer | 103 | 0.002961 | 0.110446 |
| Shared, linear envelope | 104 | 0.001181 | 0.099376 |
| Shared, two-half step envelope | 104 | 0.002104 | 0.097610 |
| Topology, linear envelope | 107 | 0.001128 | 0.093353 |
| Topology, two-half step envelope | 107 | 0.001740 | 0.098736 |

The scaling budget was kept deliberately small:

- scalar: one clipped value per layer;
- linear: offset and slope, two values per layer;
- step: independent gains for the first and second half, two values per layer.

The linear envelope gives the best median. The step envelope is slightly worse at the median but competitive in the tail, especially for the shared chain. Both justify further rendered-audio testing, but the single-scalar version remains the cleaner low-parameter candidate.

## Was topology conditioning useful?

Yes, but its value appears mostly in the difficult tail rather than the common case. The best topology-linear result reduces P95 RMSE from the shared linear result's 0.099376 to 0.093353 while using three more dense outputs and substantially more decoder storage.

Topology conditioning also produced substantially higher no-op use in later categorical layers. It should remain a primary candidate rather than being discarded.

Base-conditioned dictionaries were not retrained as the main Experiment 3 branch. Experiment 2 already established them as a complementary high-storage upper bound. The frequency-first shared and topology branches are more relevant to the desired decoder footprint.

## How did this compare with Experiment 2?

At the same B32/K16/L4 categorical structure:

- Experiment 2 shared: median 0.018647, P95 0.148889;
- Experiment 3 frequency-first shared: median 0.013317, P95 0.143424.

With one scalar per layer:

- Experiment 2 shared: median 0.005735, P95 0.116220;
- Experiment 3 frequency-first shared: median 0.002894, P95 0.108700.

Frequency-first construction therefore improved both the common case and the tail. At 100 outputs it also slightly improves the P95 of Experiment 2's much larger base-conditioned gain model (0.110708), although that model retains a better median of 0.001286.

## Did the convolutional idea help?

The decoder-side probe allowed each residual code four circular placements. This added four placement logits per layer: 16 additional dense outputs over the four-layer stack.

After preserving the ordinary unshifted solution as an explicit fallback, the shift variant produced no meaningful aggregate improvement. The extra branch factor also made unconstrained beam search less stable. This particular decoder formulation is not worth its output cost.

This does not rule out convolutions in the audio encoder. A 1D convolutional encoder could recognize local LFO features without adding any parameters to the predicted representation. That question belongs to the inferability experiment rather than the oracle codec experiment.

## Current survivors

The useful next-stage candidates are:

1. Frequency-first shared B32/K16/L4 with one scalar per layer: compact learned candidate, 100 outputs.
2. Frequency-first shared B32/K16/L4 with a two-value linear or step envelope: localized-correction diagnostic, 104 outputs.
3. Frequency-first topology B32/K16/L4 with a linear envelope: best geometry/tail candidate, 107 outputs.
4. Grid64: simple low-output baseline.

The next comparison should include sparse Vital point/power refitting and rendered modulation tests. Geometry alone cannot establish whether the additional four to seven outputs produce an audible benefit or are inferable from reference audio.
