# Issue #38 quick-research results

These are the reviewed, sanitized results from the bounded issue #38 run. Every command used a 30-minute supervisor budget. Recorded worker wall times were about 282 seconds for A, 160 seconds for B, and 66 seconds for C. D's citation validator and report formatter took about 2 seconds after reviewer-led source inspection; manual review time was separate and was not recorded. Raw presets, source paths, rendered audio, feature caches, native logs, and source checkouts remain ignored local artifacts.

| Task | Outcome | Useful conclusion |
|---|---|---|
| A: stored-state identity | Complete: 9,618 parsed, 2 excluded | Strict JSON canonicalization did not merge any additional files. Removing only `author`, `comments`, and `preset_name` introduced three duplicate groups, all explained by `/preset_name`. |
| B: measurement variability | Complete with failures: 54/54 attempted, 33 passed QA | Nine fixture/control/metric combinations separated the 5% parameter change from repeat variation. Detectability depended on the fixture and metric; no single metric was uniformly informative. |
| C: controlled Basic Pitch | Complete with failures: 24/24 attempted, 15 scored | Dense posteriors distinguished lower-octave, upper-octave, weak-pitch, and weak-onset cases that decoded MIDI alone would collapse together. |
| D: modulation semantics | Five questions reviewed from pinned Vital and Vita source | A slot is a positional connection plus its line map and five indexed controls. Relabeling must move all of them and rewrite slot-amount references; cycles and accumulation order prevent a general bit-identity claim. |

## A. Stored-state identity

Raw bytes and strictly parsed/key-ordered JSON both produced 9,595 groups with 23 excess duplicate files. Dropping the three declared display fields produced 9,592 groups with 26 excess duplicate files. The three newly merged groups differ only at `/preset_name`.

The deterministic 80/10/10 file-path simulation placed eight raw/canonical duplicate groups across split boundaries and nine state-identity groups across boundaries. This is a simulated split diagnostic, not evidence about an existing training split. Future split construction should group by state identity, while retaining the raw and canonical hashes for provenance.

See the [identity report](DATASET_IDENTITY_REPORT.md) and [aggregate JSON](data/identity_summary.json).

## B. Measurement signal and variability

Five of eight fixtures produced valid audio for every scheduled comparison. All attempts for the other three fixtures failed the clipping gate, leaving 33 valid renders and 21 explicit QA failures.

The useful separations were local:

- `bass_00940b8f3955`: oscillator level separated spectral distance and envelope MAE; filter cutoff separated waveform RMSE.
- `pad_0066d9cb520f`: oscillator level separated all three metrics from a deterministic baseline.
- `lead_00d6e56919f9`: oscillator level separated spectral distance; filter cutoff separated spectral distance and envelope MAE.
- The other valid pad and pluck comparisons overlapped repeat variation at this perturbation size.

These results support keeping waveform, spectral, and envelope distances as complementary diagnostics. They do not support a global metric ranking or a perceptual-loss claim. A larger render study should first define a prospective, non-clipping fixture policy so exclusions do not depend on the tested treatment.

See the [measurement report](MEASUREMENT_SIGNAL_REPORT.md), [range table](data/measurement_ranges.csv), [comparison table](data/comparisons.csv), and [summary JSON](data/measurement_summary.json).

![Observed repeat and treatment ranges](images/measurement_ranges.png)

## C. Controlled Basic Pitch behavior

The frozen Basic Pitch frontend ran in float32 on XPU. The five fixtures that passed audio QA yielded 15 scored performances:

- The bass had a clear lower-octave preference on held notes: mean lower-octave posterior 0.828/0.834 versus correct-pitch 0.160/0.256.
- One pad had strong correct-pitch support, while its held C3 onset peak was 0.466, below the stock 0.5 decoder threshold.
- The second pad had weak correct-pitch and onset evidence across held and staccato cases.
- The pluck mixed correct-pitch support with lower-octave ambiguity, especially on held C4.
- The lead had weak correct-pitch support, intermittent upper-octave preference, and onset peaks below the stock threshold.

These are fixture-level mechanisms. They support retaining dense posterior and onset diagnostics alongside decoded event scores; they do not establish category-wide behavior or a conditioning benefit.

See the [Basic Pitch report](CONTROLLED_BASIC_PITCH_REPORT.md), [posterior diagnostics](data/posterior_diagnostics.csv), and [summary JSON](data/encoder_summary.json).

## D. Modulation-slot semantics

The source audit resolves the code question without permutation renders. Vital saves 64 positional connection objects, while `modulation_<slot>_{amount,power,bipolar,stereo,bypass}` supplies that slot's indexed controls. A safe acyclic relabel must also carry the optional line map and rewrite every `modulation_<slot>_amount` destination reference.

Graph dependencies determine processor scheduling, but contributors are accumulated in connection/input order. Cycles introduce a previous-block feedback node on the edge that closes the cycle. The source therefore supports semantic equivalence only for a complete acyclic relabel from equivalent state; it does not support general bit-identical output. Dataset canonicalization should preserve modulation-array order unless it implements and verifies the full graph-aware rewrite.

The findings apply directly to `mtytel/vital@636ca0ef517a4db087a6a08a6a8a5e704e21f836` and the cited unchanged blocks in `DBraun/Vita@342bc90aca7ab2b6e7a487f8e54a0158a5ccab76`. Source equivalence to the installed Vital 1.6.4 VST3 was not established.

See the citation-checked [Vital audit](MODULATION_SLOT_SEMANTICS_AUDIT.md) and independent [Vita cross-check](MODULATION_SLOT_SEMANTICS_AUDIT_VITA.md).

## Next bounded experiment

Define a prospective fixture-eligibility rule using one baseline render per candidate, then repeat B/C on a fresh frozen set that passes the clipping gate. Reuse the existing MIDI, render cache, metric implementation, and Basic Pitch feature cache. The quick decision is whether the nine observed separations and four posterior failure mechanisms recur across at least two independently selected fixtures; stop within 30 minutes and report incomplete coverage rather than silently shrinking the population.
