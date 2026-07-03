# Experiment 5 Findings: Per-Code Phase-Alignment Oracle

Experiment 5 asked a narrow but important question:

> If every candidate code gets its own best phase and gain before code selection, how much of Experiment 4’s error was caused by approximate alignment rather than by the codebooks themselves?

The short answer is that phase factorization is sound, but it only pays off fully when alignment is solved independently per code and then used inside beam search. Greedy selection with the stronger aligner is useful diagnostically, but the practical reconstruction gains come from separating alignment quality from layered search quality.

The full run completed successfully at `2026-06-26T18:35:40+08:00`; stderr was empty. Artifacts are under `../artifacts/phase_alignment_oracle/`.

## What was tested

The experiment froze the Experiment 4 codebooks:

- `phase_shared`
- `phase_switch_1`
- `phase_switch_2`
- `phase_additive_k8`
- `phase_additive_k16`

For each target and each stage, it aligned every available code independently, then selected the best aligned code afterward. This is the important conceptual correction: code identity is no longer judged at phase zero, or by a phase chosen globally before code comparison.

Experiment 5 separates three error sources:

1. Alignment approximation: whether the phase/gain solver finds the best fit for a fixed code.
2. Code-selection error: whether better per-code alignment changes which code is selected.
3. Layered beam-pruning error: whether the right local choices survive through multiple residual layers.

## Solver result

The exact residual-space solver and dense references agree closely. On the full-dictionary benchmark, XPU batching is the clear production direction:

| Solver | Benchmark | Pairs/sec | Median error | P95 error |
|---|---:|---:|---:|---:|
| exact CPU batch | full dictionary | 2,084 | 0.001561 | 0.082681 |
| exact XPU batch | full dictionary | 33,865 | 0.001561 | 0.082681 |

So the XPU analytic interval solver is not just a speed trick; it preserves the CPU reference result while making exhaustive per-code alignment practical.

For a single code per target, the 4,096 and 65,536 dense grids essentially match the exact solver, while the old FFT-128 plus local refinement is slightly worse at P95. The old solver is probably acceptable for rough exploratory runs, but it should not be the reference oracle.

## Corpus result

Greedy exact alignment gives the following held-out reconstruction errors:

| Configuration | Median RMSE | P95 RMSE |
|---|---:|---:|
| `phase_additive_k16` | 0.003527 | 0.047108 |
| `phase_additive_k8` | 0.005964 | 0.055083 |
| `phase_shared` | 0.007787 | 0.079160 |
| `phase_switch_1` | 0.008131 | 0.081875 |
| `phase_switch_2` | 0.007917 | 0.080018 |

This greedy result is not the headline win. In fact, some greedy rows are worse than Experiment 4’s oracle-style results, which tells us that alignment quality and layered search quality are separate. Better phase fitting does not rescue a locally bad residual path.

With beam search, the story changes:

| Configuration | E4 median | E4 P95 | E5 beam-128 median | E5 beam-128 P95 | Median gain | P95 gain |
|---|---:|---:|---:|---:|---:|---:|
| `phase_shared` | 0.007051 | 0.076554 | 0.005445 | 0.056057 | 22.8% | 26.8% |
| `phase_switch_1` | 0.007806 | 0.071169 | 0.006010 | 0.057407 | 23.0% | 19.3% |
| `phase_switch_2` | 0.006913 | 0.071610 | 0.005686 | 0.056667 | 17.8% | 20.9% |
| `phase_additive_k8` | 0.005739 | 0.053648 | 0.002806 | 0.028633 | 51.1% | 46.6% |
| `phase_additive_k16` | 0.002900 | 0.043579 | 0.001446 | 0.023839 | 50.1% | 45.3% |

That is strong evidence that the phase-aligned residual-code approach is worth keeping.

## Beam width

Full-corpus beam-64 already captures most of the benefit. Beam-128 improves further but with roughly doubled runtime:

| Configuration | Beam-64 median | Beam-64 P95 | Beam-128 median | Beam-128 P95 |
|---|---:|---:|---:|---:|
| `phase_shared` | 0.005464 | 0.056084 | 0.005445 | 0.056057 |
| `phase_switch_1` | 0.006010 | 0.057407 | 0.006010 | 0.057407 |
| `phase_switch_2` | 0.005686 | 0.056667 | 0.005686 | 0.056667 |
| `phase_additive_k8` | 0.002844 | 0.029188 | 0.002806 | 0.028633 |
| `phase_additive_k16` | 0.001481 | 0.024418 | 0.001446 | 0.023839 |

Beam-128 is useful for oracle reporting. Beam-64 is likely the better default for routine runs unless we are specifically characterizing tail behavior.

## Did per-code phase alignment change code choices?

Yes, but not violently. The largest per-stage code-change rates were:

| Configuration | Max stage code-change rate |
|---|---:|
| `phase_additive_k16` | 6.23% |
| `phase_additive_k8` | 5.98% |
| `phase_shared` | 5.73% |
| `phase_switch_1` | 3.74% |
| `phase_switch_2` | 2.68% |

This is exactly the shape we wanted to measure. Phase alignment does not rewrite the whole codebook story, but it is large enough to matter, especially in later residual layers where translated corrections compete with each other.

## Interpretation

The result supports the user’s hypothesis: a preset can use code `X` with a nonzero translation instead of code `Y` at zero phase, but only if the oracle actually gives every code its own alignment before choosing. If we compare code identities before alignment, we undercount this possibility.

The most important result is not “phase is good” in the abstract. It is:

> Phase factorization plus per-code alignment plus beam search materially reduces both median and tail error without needing translated duplicate codes.

The additive shared+topology configurations remain the strongest. `phase_additive_k16` is the best reconstruction upper bound, while `phase_additive_k8` is the more practical Pareto candidate because it captures much of the win with a smaller output budget.

## Recommendation

Use the exact XPU batched residual-space aligner as the production/reference oracle for Experiment 6-style predictor work.

Keep:

- per-code phase alignment before code selection;
- residual gain;
- no-op residual paths;
- beam search for oracle labeling;
- additive shared + topology branches as the main high-quality family.

Use beam-64 for routine labeling and beam-128 for final oracle audits or tail-sensitive reports.

Do not use FFT-128/local refinement as the reference oracle. It is fine as a rough fast diagnostic, but Experiment 5 shows we can afford the stronger XPU aligner.

## Open questions for the next experiment

1. Can a predictor learn these oracle labels from the dense LFO curve reliably?
2. Does the model learn phase as a circular variable cleanly, or do we need symmetry-aware phase labels?
3. Is `phase_additive_k8` the right practical target, or does `phase_additive_k16` buy enough tail improvement to justify the larger output space?
4. Can beam-64 labels be used for training while beam-128 remains an evaluation-only oracle?
5. How much of the remaining tail error is codebook insufficiency versus envelope/gain limitations?

The next logical experiment is a learnability probe: freeze one or two E5-selected codecs, train a small circular-padding CNN and a parameter-matched MLP to predict base/residual choices, gains, and phases, then measure predicted reconstruction RMSE versus the E5 oracle gap.
