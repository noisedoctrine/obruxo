# Experiment 2 Findings: Stacked Residual Codebooks

## What did this experiment ask?

Can an LFO be represented efficiently as one categorical base choice followed by several categorical residual corrections?

The proposed decoder was:

```text
curve = base[b] + residual_1[r1] + ... + residual_N[rN]
```

The experiment compared base widths `{15, 16, 24, 32}`, residual widths `{4, 8, 16}`, and depths `{1, 2, 3, 4}`. Conditional and scalar-gain variants were evaluated for three shared-codebook finalists.


## How were the codebooks constructed?

The 15 provisional stock shapes were fixed. Additional bases were selected from observed training shapes. At each residual layer:

1. reconstruct every training curve using the frozen prefix;
2. calculate the remaining curve error;
3. cluster those residuals on a reduced grid;
4. replace every cluster center with an observed full-resolution residual;
5. reserve code zero as a no-op;
6. freeze the layer before constructing the next one.

The averages used for clustering were never decoded. Every stored non-zero base or residual was snapped to an observed training vector.

Authors were deterministically separated before codebook construction: 13,295 active LFO instances from 582 authors were used for fitting, while 1,605 instances from 123 different authors were held out. No author or learned-code source crossed the split.

Held-out encoding used beam search width 32. Greedy output was retained whenever it had lower full-resolution error, and a preceding-depth path with a new zero code was retained whenever another layer would increase error.


## Question 1: Does categorical stacking improve over the stock codebook?

Yes. The stock-only baseline had median RMSE `0.1663` and 95th-percentile RMSE `0.3960`. Shared stacked codebooks improved continuously with width and depth:

| Shared configuration | Dense logits | Effective bits | Median RMSE | P95 RMSE |
|---|---:|---:|---:|---:|
| B15 / K4 / L1 | 19 | 5.91 | 0.1450 | 0.3142 |
| B15 / K8 / L2 | 31 | 9.91 | 0.0513 | 0.2495 |
| B16 / K16 / L3 | 64 | 16.00 | 0.0259 | 0.1812 |
| B24 / K16 / L4 | 88 | 20.58 | 0.0198 | 0.1495 |
| B32 / K16 / L4 | 96 | 21.00 | 0.0186 | 0.1489 |

The stacked construction works as intended, but the gains diminish near the largest shared configurations.


## Question 2: Does shared categorical stacking beat a direct sampled grid?

Not at comparable neural-output width.

Experiment 1's direct Grid 64 used 64 continuous outputs and achieved median RMSE `0.0036` and P95 RMSE `0.1251`. A 64-logit shared stack achieved approximately median `0.024–0.028` and P95 `0.181` depending on its factorization.

The strongest shared categorical-only stack used 96 logits and still had median RMSE `0.0186` and P95 `0.1489`. Pure categorical stacking therefore does not currently replace direct grid prediction on geometry efficiency.

The comparison is not entirely symmetric: the categorical stack carries about 21 bits after argmax, whereas the grid contains 64 continuous values. Whether the categorical bottleneck is easier to infer from audio remains an untested question.


## Question 3: Do scalar residual gains help?

Substantially. Adding one continuous gain per residual layer changed the largest shared stack from:

| Configuration | Dense outputs | Median RMSE | P95 RMSE |
|---|---:|---:|---:|
| Shared B32 / K16 / L4 | 96 | 0.0186 | 0.1489 |
| Shared B32 / K16 / L4 + gains | 100 | 0.0057 | 0.1162 |

This slightly beats Grid 64 in tail error but remains worse in median error and uses more model outputs. It suggests that fixed residual magnitude—not residual direction alone—is a major limitation of pure categorical RVQ.


## Question 4: Does conditioning residual dictionaries help?

Base conditioning helps dramatically; topology conditioning helps only slightly.

| B32 / K16 / L4 + gains | Dense outputs | Stored floats | Median RMSE | P95 RMSE |
|---|---:|---:|---:|---:|
| Shared | 100 | 98,304 | 0.0057 | 0.1162 |
| Topology-conditioned | 103 | 229,376 | 0.0052 | 0.1162 |
| Base-conditioned | 100 | 2,129,920 | 0.0013 | 0.1107 |

Topology conditioning does not earn its additional gate and 2.3× dictionary storage in this experiment.

Base conditioning gives the best geometry result, but its dictionary is about 21.7× larger than the shared version. This shifts complexity from the model output into a large decoder table. It is a viable upper bound or retrieval-style representation, not an obvious default.


## Question 5: Did the implementation satisfy its structural checks?

Yes:

* 101,115 held-out result rows were produced across 63 configurations;
* no metric contains NaNs;
* the author partitions are disjoint;
* learned codeword sources come only from the training partition;
* every residual dictionary reserves an exact zero code;
* every learned non-zero code has an observed source;
* held-out RMSE never increases when another residual layer is added;
* beam output falls back to greedy whenever greedy is better at full resolution;
* all eight unit tests pass.

Some condition-specific codes are unused by the held-out partition. That is expected and is reported separately; it is not the same as a code being dead during fitting.


## Current conclusion

Stacked residual codebooks are technically sound, and scalar gains make them much more expressive. The current geometry evidence does **not** support replacing direct-grid prediction with a globally shared categorical stack.

The useful survivors are:

1. **Grid 64** as the strongest simple output-space baseline.
2. **Shared B32/K16/L4 with gains** as a compact-index-plus-small-continuous hybrid.
3. **Base-conditioned B32/K16/L4 with gains** as a high-storage upper bound.

These should next be compared after fitting back to valid Vital points/powers and in controlled rendered-audio contexts. Geometry RMSE alone cannot determine whether the remaining differences are audible or inferable from one reference clip.

