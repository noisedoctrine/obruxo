# LFO Representation Experiment Findings

Experiment 2's stacked residual-codebook results are reported separately in
[experiment-02-findings.md](experiment-02-findings.md).

## 2026-06-21: Initial corpus and oracle run

This run evaluates **representation capacity under perfect encoding**. For every corpus LFO, the experiment samples the original Vital curve at 1,024 phases, lets each codec encode that complete curve, reconstructs it, and compares the reconstruction with the original.

There is no audio-to-LFO model in this run. The results answer which compact representations are worth carrying forward, not which ones can already be inferred from audio.


## Question 1: How many relevant LFO examples do we have?

### Method

The catalog scanned 9,620 PresetShare records. It included an LFO when a modulation route named it as its source and had a non-empty destination. It then separately marked the LFO as materially active when at least one corresponding route was non-bypassed and had non-zero modulation amount.

### Answer

The scan found:

* **16,534 routed LFO instances**;
* **14,900 materially active instances** used by the oracle benchmark;
* all 15 known stock shape names.

This is enough data to characterize the shape distribution. It does not mean that all 14,900 shapes are independently authored: repeated stock shapes and templates are intentionally present in the population.


## Question 2: How was the stock codebook obtained?

### Method

For each of the 15 known stock names, the experiment selected the most frequent exact point/power geometry carrying that name. The resulting shape became the provisional codebook entry.

### Answer

All 15 names produced an entry, so the stock-codebook experiments could run. This does **not** prove that every entry is canonical. An edited shape may retain a stock name; controlled Vital saves should replace these modal corpus entries.


## Question 3: Are the 15 stock shapes sufficient by themselves?

### Method

For each reference curve, the oracle selected whichever of the 15 stock curves had the lowest mean squared error. This is more favorable than a trained classifier because it always makes the best possible stock choice.

### Answer

No. Stock-only reconstruction had median RMSE **0.1663** and 95th-percentile RMSE **0.3960**. Even with perfect nearest-stock selection, too much corpus geometry lies outside the stock vocabulary.

This rejects a stock-only model as the complete LFO representation. The stock shapes may still be useful as a common-case prior.


## Question 4: How much fidelity do direct fixed grids buy?

### Method

The oracle sampled each reference curve at 8, 16, 32, or 64 uniformly spaced phases. The decoder linearly expanded those values back to the 1,024-point comparison grid.

### Answer

| Codec | Dense dimensions | Median RMSE | 95th percentile RMSE |
|---|---:|---:|---:|
| Grid 8 | 8 | 0.0776 | 0.3490 |
| Grid 16 | 16 | 0.0321 | 0.2684 |
| Grid 32 | 32 | 0.0111 | 0.1842 |
| Grid 64 | 64 | 0.0036 | 0.1251 |

Direct grids improve steadily as dimensions increase and behave comparatively well in the tail. They are a strong representation-neutral baseline. The remaining tail error and near-unit maximum errors indicate that uniform samples handle some sharp jumps poorly.


## Question 5: Does a stock prior make residual prediction more efficient?

### Method

The oracle selected the nearest stock curve, subtracted it from the reference, sampled the residual on a fixed grid, expanded the residual, and added it back to the stock curve.

The dense output cost is:

```text
15 stock logits + 1 residual-presence value + N residual samples
```

### Answer

| Codec | Dense dimensions | Median RMSE | 95th percentile RMSE |
|---|---:|---:|---:|
| Stock 15 + residual 8 | 24 | 0.0121 | 0.3288 |
| Stock 15 + residual 16 | 32 | 0.0032 | 0.2951 |
| Stock 15 + residual 32 | 48 | 0.0008 | 0.2094 |

Yes for the common case, but not uniformly.

At the same nominal 32-dimensional cost, stock plus a 16-value residual has much lower median error than Grid 32 (`0.0032` versus `0.0111`). However, its 95th-percentile error is substantially worse (`0.2951` versus `0.1842`). The stock prior compresses regular or stock-derived shapes very efficiently, but it is a poor basis for part of the free-form tail.


## Question 6: Does one representation currently dominate?

### Answer

No single tested codec dominates both the common case and the tail.

The current evidence supports testing a **gated hybrid**:

```text
regular path:
  stock shape + compact residual

free-form path:
  direct sampled grid
```

The gate would be another fixed model output. During oracle analysis it can choose the lower-error branch; in the eventual model it would be predicted from audio along with the selected branch's parameters.

This is a hypothesis for the next benchmark, not yet the chosen production representation.


## Question 7: Why are tail and maximum errors still large?

### Current status

Not answered yet. Duplicate x-coordinates and other sharp discontinuities are the leading hypothesis because uniform grid sampling can miss the exact location of a jump. The current aggregate report has not yet stratified error by discontinuity, smoothing, point count, or route context.

This needs measurement before changing the codec. Possible remedies—explicit jump slots, non-uniform samples, or sparse segment prediction—should only be added if the stratified results show that discontinuities explain the tail.


## What this run does not establish

The current results do not answer:

* whether low geometry RMSE produces perceptually equivalent modulation;
* whether reconstructed dense curves can be fit back into compact valid Vital points and powers;
* whether the compact representation can be inferred from one reference audio clip;
* whether errors matter equally for amplitude, pitch, filter, wavetable-position, and other destinations;
* whether the provisional codebook geometry exactly matches the Vital stock presets.

No audio-quality conclusion should be drawn from this report yet.


## Next questions and corresponding experiments

1. **Are the codebook entries canonical?** Replace modal entries with controlled saves and rerun exact/nearest coverage.
2. **What causes the reconstruction tail?** Stratify each codec by exact-stock match, duplicate-x jumps, point count, smoothing, shape name, and repeated signature.
3. **Does a hybrid dominate its branches?** Add an oracle gate between stock-residual and direct-grid codecs and measure its cost and error.
4. **Can the dense reconstruction become valid sparse Vital state?** Fit curves back to points/powers and rerun metrics after that conversion.
5. **Are the surviving differences audible?** Render original and reconstructed shapes across controlled destinations, rates, and modulation amounts.
6. **Can audio predict the compact code?** Train a small probe only after the representation and rendered-effect tests have reduced the candidate set.
