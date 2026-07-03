# LFO Representation Experiments 1–5: Consolidated Design Report

## Executive summary

The core research question is:

> How far can we reduce custom LFO modeling complexity by replacing raw variable-length LFO geometry with a compact structured code, while preserving enough reconstruction capacity to plausibly reproduce musical behavior?

Experiments 1–5 answer the representation-ceiling part of that question. They do not yet answer audio inferability or perceptual equivalence.

The answer so far is encouraging:

- Stock-only LFO prediction is nowhere near enough.
- Direct fixed grids are a strong and annoyingly honest baseline.
- Pure categorical residual stacks are too weak unless we add small continuous factors.
- Residual gain is essential.
- Frequency-weighted fitting and explicit no-op branches are useful.
- Topology conditioning helps mostly in the tail.
- Phase factorization is a major win.
- Per-code phase alignment before code selection is necessary for a sound oracle.
- Additive shared + topology phase-aware residuals are the current best design family.

The strongest current practical candidate is:

```text
32-way base
+ 4 residual layers
+ additive K8 shared + K8 topology correction per layer
+ one phase and one gain per selected residual
+ exact no-op code
+ exact per-code XPU alignment for oracle labels
```

This is the best compromise we currently have between compact output, expressive power, and design cleanliness.

The highest-quality upper bound is `phase_additive_k16`, but its output/storage cost is probably too high to choose before a learnability test.

## Current recommended design stance

Do not frame the solution as “choose one LFO from a discrete library.”

The evidence supports a more specific representation:

```text
canonical discrete atoms
+ circular phase offsets
+ clipped scalar gains
+ residual stacking
+ exact no-op branches
+ optional topology-specific correction branch
```

That distinction matters. Without phase and gain, the codebook has to waste slots on translated and differently scaled copies. With phase and gain, a smaller library can cover much more of the useful shape space.

## Experiment 1: simple baselines

Experiment 1 established the initial corpus and simple oracle baselines.

Corpus:

- `16,534` routed LFO instances;
- `14,900` materially active routed LFO instances;
- all 15 known stock LFO shape names represented.

Stock-only failed as a complete representation:

| Codec | Median RMSE | P95 RMSE |
|---|---:|---:|
| stock-only nearest of 15 | 0.1663 | 0.3960 |

Direct fixed grids were strong:

| Codec | Dense outputs | Median RMSE | P95 RMSE |
|---|---:|---:|---:|
| Grid 8 | 8 | 0.0776 | 0.3490 |
| Grid 16 | 16 | 0.0321 | 0.2684 |
| Grid 32 | 32 | 0.0111 | 0.1842 |
| Grid 64 | 64 | 0.0036 | 0.1251 |

Stock + direct residual improved the common case but not the tail:

| Codec | Dense outputs | Median RMSE | P95 RMSE |
|---|---:|---:|---:|
| Stock 15 + residual 8 | 24 | 0.0121 | 0.3288 |
| Stock 15 + residual 16 | 32 | 0.0032 | 0.2951 |
| Stock 15 + residual 32 | 48 | 0.0008 | 0.2094 |

Design implication:

Grid64 is the minimum serious baseline. Any structured code has to beat or justify itself against it, not just against stock-only.

## Experiment 2: stacked residual codebooks

Experiment 2 tested categorical residual-vector-quantized LFO reconstruction:

```text
curve = base[b] + residual_1[r1] + ... + residual_N[rN]
```

The learned codebooks were structurally sound: author-held-out split, observed-source codewords, zero/no-op residual code, deterministic seeds, no NaNs.

Categorical stacking improved greatly over stock-only, but did not beat Grid64 at comparable output width:

| Configuration | Dense logits | Effective bits | Median RMSE | P95 RMSE |
|---|---:|---:|---:|---:|
| B15 / K4 / L1 | 19 | 5.91 | 0.1450 | 0.3142 |
| B15 / K8 / L2 | 31 | 9.91 | 0.0513 | 0.2495 |
| B16 / K16 / L3 | 64 | 16.00 | 0.0259 | 0.1812 |
| B24 / K16 / L4 | 88 | 20.58 | 0.0198 | 0.1495 |
| B32 / K16 / L4 | 96 | 21.00 | 0.0186 | 0.1489 |

Adding one residual gain per layer was the big unlock:

| Configuration | Dense outputs | Median RMSE | P95 RMSE |
|---|---:|---:|---:|
| Shared B32/K16/L4 | 96 | 0.0186 | 0.1489 |
| Shared B32/K16/L4 + gains | 100 | 0.0057 | 0.1162 |

Base-conditioned dictionaries produced a high-quality but high-storage upper bound:

| Configuration | Dense outputs | Stored floats | Median RMSE | P95 RMSE |
|---|---:|---:|---:|---:|
| Shared + gains | 100 | 98,304 | 0.0057 | 0.1162 |
| Topology-conditioned + gains | 103 | 229,376 | 0.0052 | 0.1162 |
| Base-conditioned + gains | 100 | 2,129,920 | 0.0013 | 0.1107 |

Design implication:

Pure categorical RVQ is not enough. Small continuous factors are not optional; they are doing real work. Base conditioning is useful as an upper bound, but too storage-heavy to be the default unless later evidence says storage is cheap and output complexity is the only bottleneck.

## Experiment 3: frequency-first residual peeling

Experiment 3 changed construction philosophy:

- no cap on repeated shapes;
- common shapes influence early layers proportionally;
- base = 15 provisional stock + 17 observed medoids;
- four 16-way residual layers;
- code 0 is exact no-op;
- topology conditioning retained as a serious branch.

This matched the intended mechanism: common cases should be solved early, hit no-op, and let later layers focus on remaining shapes.

No-op use supported the mechanism:

| Layer | Shared no-op share | Topology no-op share |
|---|---:|---:|
| 1 | 45.0% | 54.1% |
| 2 | 47.1% | 54.1% |
| 3 | 47.7% | 60.5% |
| 4 | 49.1% | 60.6% |

Frequency-first improved over Experiment 2:

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

Design implication:

Frequency-first fitting is the right construction principle. It aligns with the user’s intuition that common shapes should be handled early rather than capped away.

Topology conditioning is not a huge median win, but it helps in the tail and increases no-op behavior. Keep it.

Envelope variants are promising, but they increase predicted continuous parameters. Since phase/gain later produced larger wins, envelope complexity should not be expanded yet.

## Experiment 4: phase-factorized residual codebooks

Experiment 4 tested the key idea that one atom should cover translated versions of itself:

```text
decoded residual = gain * circular_shift(code, phase)
```

This was a major improvement.

For the shared B32/K16/L4 representation:

| Controls | Dense outputs | Median RMSE | P95 RMSE |
|---|---:|---:|---:|
| gains only | 100 | 0.041890 | 0.194077 |
| gains + base phase | 101 | 0.015964 | 0.126878 |
| gains + residual phases | 104 | 0.019141 | 0.109909 |
| gains + base and residual phases | 105 | 0.007051 | 0.076554 |

The bad gains-only result is not a failure; it shows that phase-trained dictionaries expect phase controls. Once phase is enabled, the result is much better than Experiment 3.

Phase quantization was not good enough at small category counts:

| Phase positions | Median RMSE | P95 RMSE |
|---:|---:|---:|
| 8 | 0.042835 | 0.184050 |
| 16 | 0.034114 | 0.144886 |
| 32 | 0.030924 | 0.125037 |
| 64 | 0.015252 | 0.092368 |

Design implication:

Phase should be a continuous circular scalar, not a low-cardinality categorical output.

Shared/topology mixtures:

| Configuration | Dense outputs | Median RMSE | P95 RMSE |
|---|---:|---:|---:|
| Shared | 105 | 0.007051 | 0.076554 |
| Shared layer 1, topology layers 2–4 | 108 | 0.007806 | 0.071169 |
| Shared layers 1–2, topology layers 3–4 | 108 | 0.006913 | 0.071610 |
| Additive K8 shared + K8 topology | 116 | 0.005739 | 0.053648 |
| Additive K16 + K16 upper bound | 180 | 0.002900 | 0.043579 |

Design implication:

The additive shared + topology branch is the best family so far. It lets a shape use one common correction and one topology-specific correction per layer. That is more expressive than deciding between shared and topology dictionaries.

## Experiment 5: per-code phase-alignment oracle

Experiment 5 corrected the alignment question:

> Every code must receive its own best phase/gain before we compare code identities.

This matters because a preset may be better represented by code `X` at phase `r` than code `Y` at phase `0`. If we compare before alignment, we mis-measure the value of phase factorization.

Experiment 5 implemented and compared:

- old FFT/coarse-local alignment;
- dense phase grids;
- exact CPU residual-space solver;
- exact XPU batched residual-space solver;
- clipped-prefix numerical reference;
- greedy and beam search separation.

The XPU exact solver was the best production/reference direction:

| Solver | Benchmark | Pairs/sec | Median error | P95 error |
|---|---:|---:|---:|---:|
| exact CPU batch | full dictionary | 2,084 | 0.001561 | 0.082681 |
| exact XPU batch | full dictionary | 33,865 | 0.001561 | 0.082681 |

Greedy exact alignment alone was diagnostic, not the headline:

| Configuration | Median RMSE | P95 RMSE |
|---|---:|---:|
| phase_additive_k16 | 0.003527 | 0.047108 |
| phase_additive_k8 | 0.005964 | 0.055083 |
| phase_shared | 0.007787 | 0.079160 |
| phase_switch_1 | 0.008131 | 0.081875 |
| phase_switch_2 | 0.007917 | 0.080018 |

The real win came from exact alignment inside beam search:

| Configuration | E4 median | E4 P95 | E5 beam-128 median | E5 beam-128 P95 | Median gain | P95 gain |
|---|---:|---:|---:|---:|---:|---:|
| phase_shared | 0.007051 | 0.076554 | 0.005445 | 0.056057 | 22.8% | 26.8% |
| phase_switch_1 | 0.007806 | 0.071169 | 0.006010 | 0.057407 | 23.0% | 19.3% |
| phase_switch_2 | 0.006913 | 0.071610 | 0.005686 | 0.056667 | 17.8% | 20.9% |
| phase_additive_k8 | 0.005739 | 0.053648 | 0.002806 | 0.028633 | 51.1% | 46.6% |
| phase_additive_k16 | 0.002900 | 0.043579 | 0.001446 | 0.023839 | 50.1% | 45.3% |

Beam-64 was usually close to beam-128:

| Configuration | Beam-64 median | Beam-64 P95 | Beam-128 median | Beam-128 P95 |
|---|---:|---:|---:|---:|
| phase_shared | 0.005464 | 0.056084 | 0.005445 | 0.056057 |
| phase_switch_1 | 0.006010 | 0.057407 | 0.006010 | 0.057407 |
| phase_switch_2 | 0.005686 | 0.056667 | 0.005686 | 0.056667 |
| phase_additive_k8 | 0.002844 | 0.029188 | 0.002806 | 0.028633 |
| phase_additive_k16 | 0.001481 | 0.024418 | 0.001446 | 0.023839 |

Design implication:

Use exact XPU per-code alignment for oracle labeling. Use beam-64 for routine labels and beam-128 for final/tail audits.

Do not use FFT-128/local refinement as the reference oracle. It is useful for rough diagnostics only.

## Sampling grid audit

The current reference grid is `1024`, and the search/feature grid is often `128`.

Those are computationally convenient powers of two, but they are not musically neutral. The corpus is power-of-two dominated, but custom-ish shapes show much more triplet and non-power-of-two structure.

Materially active corpus:

- shapes with denominator divisible by `3`: 8.1%;
- shapes with any non-power-of-two denominator: 18.9%.

Custom-ish subset (`stock_name_hint=False`):

- triplet-denominator shapes: 29.8%;
- non-power-of-two-denominator shapes: 48.9%.

Candidate denominator coverage:

| Grid | Boundary denominator coverage |
|---:|---:|
| 128 | 85.6% |
| 192 | 88.2% |
| 1024 | 85.6% |
| 1536 | 88.3% |
| 1920 | 89.2% |

Design implication:

We should test musically composite grids before locking the representation:

- feature/search grid: `192` instead of `128`;
- reference/evaluation grid: `1920` instead of `1024`;
- optional middle cases: `384`, `1536`.

This may matter most for gates, pulses, discontinuities, and custom-ish shapes.

## Design choices that are now reasonably supported

### 1. Keep residual stacking

Residual stacking is structurally sound and improves capacity. It also gives us no-op paths, which are important for common/simple shapes.

### 2. Keep a 32-way base

The 15 stock + 17 medoid base is a good current default. It is simple, interpretable, and outperformed stock-only assumptions.

The exact 15 stock geometries are still provisional and should be replaced by controlled canonical saves, but the 32-way structure is reasonable.

### 3. Keep scalar gain per residual

Gain is too valuable to remove. Fixed-magnitude categorical residuals are a bad fit for this problem.

### 4. Keep continuous circular phase

Phase factorization is one of the largest improvements in the whole sequence. Low-bin phase quantization is not enough.

### 5. Keep exact no-op codes

No-op is not just a convenience. It is how solved/common cases leave the residual stack and avoid muddy average corrections.

### 6. Keep topology as a branch, but prefer additive over exclusive switching

Topology conditioning alone was not a giant median improvement, but it helped tail behavior and no-op structure. Experiment 4–5 suggest the additive shared + topology setup is better than forcing a single branch.

### 7. Use exact XPU per-code alignment for oracle labels

Every candidate code must be independently aligned before code selection. This is the sound version of the oracle.

## Design choices that should not be locked yet

### 1. Final output budget

The main candidates are:

| Candidate | Dense outputs | Role |
|---|---:|---|
| Grid64 | 64 | simple continuous baseline |
| phase_shared | 105 | compact structured code |
| phase_switch_1 / phase_switch_2 | 108 | modest topology-tail variants |
| phase_additive_k8 | 116 | current practical favorite |
| phase_additive_k16 | 180 | high-quality upper bound |

`phase_additive_k8` is the current favorite, but not yet final.

### 2. Envelope scaling

Experiment 3’s linear/step envelopes helped, but phase/gain later gave a cleaner and larger improvement. We should not add envelope parameters to the main representation until we know what error remains after phase-aware additive codebooks on the improved grid.

### 3. Reference/search grid

`1024/128` is not bad, but `1920/192` may be more musically appropriate. This should be audited before training neural predictors.

### 4. Sparse Vital reconstruction

All current metrics operate on sampled curves. We still need to confirm that decoded curves can be converted back into valid Vital point/power/smooth state without losing the gains.

### 5. Audio relevance

Geometry RMSE is useful for representation design, but modulation audibility depends on destination, depth, rate, and other synth context.

## Recommended candidate set for the next phase

Carry forward four codecs:

1. `Grid64`
   - baseline;
   - simple, low-output, continuous;
   - hard to beat on simplicity.

2. `phase_shared`
   - compact structured code;
   - useful lower-budget phase-aware candidate.

3. `phase_additive_k8`
   - current practical favorite;
   - strong median and tail performance;
   - manageable output budget relative to the upper bound.

4. `phase_additive_k16`
   - upper bound;
   - use to measure how much quality remains on the table.

Optionally keep one switch topology candidate only if we want a low-increment topology comparison. Otherwise, additive K8 is the cleaner topology story.

## Proposed Experiment 6

Experiment 6 should be a design-locking audit before neural audio inference.

### 6A. Grid sensitivity

Rerun the selected oracle codecs with:

- feature/search: `128` versus `192`;
- reference/evaluation: `1024` versus `1920`;
- focus metrics by topology and custom-ish status.

Questions:

- Do selected code paths change~
- Does tail error improve for discontinuous/gate shapes~
- Does phase alignment change materially~
- Is runtime still acceptable on XPU~

### 6B. Dense-curve learnability probe

Before audio, test whether the representation is learnable from the target curve itself.

Inputs:

- sampled LFO curve;
- maybe topology label as either input or predicted auxiliary target.

Targets:

- base code;
- residual code choices;
- topology branch where applicable;
- gains;
- circular phases.

Models:

- circular-padding 1D CNN;
- parameter-matched MLP baseline;
- Grid64 regression baseline.

Metrics:

- decoded reconstruction RMSE;
- oracle gap;
- phase error modulo symmetry;
- code accuracy as diagnostic, not primary;
- topology errors;
- tail behavior.

### 6C. Sparse Vital refit

Convert decoded dense curves back into valid Vital LFO editor state:

- points;
- powers;
- smooth flag/mode if needed;
- duplicate-x discontinuities.

Then re-sample and re-score. This tells us whether the representation remains useful as an actual preset-editable object rather than just a dense internal curve.

### 6D. Rendered modulation audit

Render controlled contexts:

- amplitude;
- pitch;
- filter cutoff;
- wavetable position;
- maybe spectral/macro-like destinations.

Vary:

- LFO rate;
- modulation amount;
- sync mode;
- smooth versus discontinuous shapes.

The goal is not perfect audio identity yet. The goal is to identify which geometric errors matter musically.

## Current best design recommendation

For now, design around this representation:

```text
base:
  32-way categorical
  = 15 stock + 17 observed medoids
  + continuous base phase

residual layers:
  4 layers
  each layer has:
    shared residual code
    topology residual code
    no-op code available
    continuous circular phase per active correction
    clipped scalar gain per active correction

oracle:
  exact per-code XPU alignment
  beam-64 routine labels
  beam-128 audit labels

main candidate:
  additive K8 shared + K8 topology

upper bound:
  additive K16 shared + K16 topology
```

Do not yet choose `phase_additive_k16` as the production representation. It is better, but the extra output budget must earn itself in learnability and rendered-audio tests.

Do not discard Grid64. It remains the clean baseline and may still win if structured codes are hard to infer from audio.

## Open risks

1. **Canonical stock shapes are still provisional.**
   The 15 stock entries should be replaced by controlled Vital saves.

2. **Current grid may underrepresent musical subdivisions.**
   Run the `192/1920` audit.

3. **Dense reconstruction may not map cleanly back to Vital sparse state.**
   This is a real product constraint.

4. **Geometry RMSE may not track audio relevance.**
   Rendered modulation tests are still necessary.

5. **Oracle labels may be multi-modal.**
   Many code/phase/gain paths can reconstruct the same curve. Predictor training should optimize decoded reconstruction, not only exact code accuracy.

6. **Output dimensions are not the same as effective complexity.**
   Dense output count, effective index bits, continuous scalar precision, and decoder storage must stay separated in reports.

## Bottom line

Experiments 1–5 support continuing with a structured discrete-plus-continuous LFO representation. The strongest design idea is not a pure codebook; it is a phase-factorized residual codebook with gain, no-op paths, and additive shared/topology corrections.

The next work should stop expanding the oracle family and start testing design lock criteria:

- musically composite sampling grids;
- learnability from dense curves;
- sparse Vital refitting;
- rendered modulation impact.

If `phase_additive_k8` survives those checks with a small oracle gap, it is the leading candidate for the custom-LFO output representation.
