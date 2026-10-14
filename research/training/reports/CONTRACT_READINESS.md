# Contracts readiness — 2026-10-04

This is the historical corpus/contracts report. The subsequent [first trainer report](FIRST_TRAINER_RESULTS.md) records a native loading correction and a newly versioned authored pilot; it does not change the static corpus counts below. The old contract ID remains historical and is incompatible with the current renderer identity.

The initial training contracts are implemented and tested. **The downloaded corpus currently has zero statically eligible states under the deliberately limited init-only asset catalog and legacy schema.** This measures what has been verified; it does not establish that the corpus is unusable or that its unresolved assets are imported/custom.

The [workspace README](../README.md) documents the executable interfaces and reproduction commands. The [machine-readable aggregate](data/contracts_summary.json) contains the full version and unsupported-field counts. Private membership paths, presets, asset payloads and audio are not included in this report.

## Full local corpus audit

| Measure | Result |
|---|---:|
| Files discovered and inspected | 9,620 |
| Strictly parsed JSON presets | 9,618 |
| Excluded before schema/asset inspection | 2: one invalid JSON, one invalid UTF-8 |
| Raw byte identities | 9,595 |
| Canonical JSON identities | 9,595 |
| Stored-state identities | 9,592 |
| Initial lineage groups | 9,592 |
| Statically eligible files / states | 0 / 0 |
| Native loading / audio QA / codec reconstruction | Not run |

The identity totals match the [earlier corpus identity audit](../../experiments/quick_research/reports/DATASET_IDENTITY_REPORT.md). No extra parent relationships were supplied for this audit, and near-duplicate ancestry was not inferred. Exact stored-state duplicates share a split; any additional reviewed relationships must be included before treating the corpus splits as research-ready.

The seed-0 frozen manifest assigns **7,637 train, 980 validation and 975 test states** across the entire strictly parsed population, including presently ineligible states. Eligible counts in all three splits are zero. Split assignment is kept separate from eligibility so filtering does not reshuffle the held-out population.

| Static ineligibility reason | Files, with overlap |
|---|---:|
| At least one active asset lacks a registered content match | 9,617 |
| Preset fails the reviewed 772-control schema | 7,864 |
| Preset version is outside the initial allowlist | 7,517 |

There are 2,101 parsed files in versions 1.0.0–1.0.8 and 7,517 in versions 1.5.x–1.6.x. The most common newer version is 1.5.5, with 6,581 files. `custom_warps` occurs outside the reviewed scalar schema in 7,517 files; each newer modulation ramp-up/down field occurs in 287. These fields were reported, not stripped from their sources.

The default catalog contains **two references**: the reviewed init wavetable and init sample. It is not a factory-bank census. Exact content matching retains versions and encodings, so a serialization change can also leave an otherwise equivalent asset unresolved. The earlier stock-name heuristic count of 3,568 files is therefore not a verified training population. Register reviewed reference exports by hash and expand schema support before counting broad corpus coverage.

## Capability matrix for this increment

| Boundary | Executable coverage now | Remaining gate |
|---|---|---|
| Identity | Strict JSON, raw/canonical/stored-state hashes; ordered arrays preserved | Acoustic or near-duplicate review is separate |
| Ancestry and splits | Transitive declared parents; cycle/missing-parent rejection; frozen hash splits; online descendants inherit; cross-lineage collisions fail | Complete reviewed ancestry for a research dataset |
| Legacy controls | Existing 772-control ranges/ordinals and required field validation; source and schema hashes retained | Learned codec/head coverage and native reconstruction |
| Oscillators and sample | Three stored wavetable slots plus one sample; active assets must match registered content | Verified factory asset catalog; broader asset resolver |
| Envelopes, filters, effects, voice and global state | Existing scalar schema retained without selecting a neural target layout | Explicit codec masks/heads and per-family render tests |
| Modulation | All 64 slots retain their stored order and graph-reference vocabulary | Structured decoder, graph-preserving codec and render checks |
| LFOs | Eight stored shapes; existing structural validation; no reordering | Train-only learned LFO assets and codec integration with the 97-point design |
| Modern 1.5.x/1.6.x state | Excluded with version and unsupported-field reasons | Reviewed schema/defaults and native tests; no silent projection |
| Performance | Eight exact MIDI patterns; four reference, four transfer; content-based role checks | Broader versioned curriculum and statistical evaluation suite |
| MIDI | Notes, polyphony, fixed tempo and channel 0 | CC, bend, pressure and tempo changes need renderer support |
| Render recipe | Immutable request, source identities, lineage/split/contract IDs, sequence and purpose | Worker execution, audio hash, QA result, retry and consumption records |
| Audio | Declared 44.1 kHz stereo float32, two-second tail, existing Windows renderer identity | Native QA on each example; longer-tail coverage where needed |
| Stream budget | One worker, 32-clip and 256 MiB prefetch limits declared | Bounded queue/backpressure implementation and coupled profiling |
| Snapshot recovery | Artifact integrity, ancestry reconstruction and contract compatibility checked | Optimizer/RNG/sampler/queue checkpoint recovery |
| Usage and redistribution | Reference source hashes and caller-declared provenance | Per-source usage review; fingerprints are not licenses |

No learned inverse-model capability is claimed by schema validation. The full architecture remains governed by the [training plan](../../modelling/TRAINING_PLAN.md).

## Reproduction and evidence

Run from the repository root, choosing a new output directory for another audit:

```sh
python research/training/run.py audit --corpus datasets/presetshare/raw/presetshare_files --output research/training/outputs/corpus-contracts-20261004-final
python research/training/run.py verify --snapshot research/training/outputs/corpus-contracts-20261004-final
```

The complete snapshot generated for this report uses:

| Identity | Value |
|---|---|
| Schema ID | `vital-1.0.8-vita-0.1.0-026ce213403d` |
| Contract ID | `75e64f048f3b2868fcb2bb704e2440650f9e4561f76bddbae44f2bb38f088983` |
| Asset catalog ID | `7350ded77d830620a2bda95695d2051da8bc576c034d051e9ae476fec8d4e893` |
| Split manifest ID | `6929fc181f3344b7b84022ca6ff097efb10b079d37befa136af05aeb977b9242` |

Validation on the local Windows/Python 3.12 environment: **35 training-contract tests passed**, **45 existing non-native data-generation tests passed** (five native tests deselected), and **Ruff passed**. Tests cover audit/source immutability, golden identity compatibility, changed schema/artifact rejection, partial-population rejection, ancestry grouping, variant split inheritance/collisions, asset identity and provenance checks, reserved-performance leakage and immutable recipe inputs. A Linux CI workflow is included; it has not been run remotely in this increment.

The audit is CPU-only and requires no audio pre-rendering or GPU. It does not measure accepted renders per second, training throughput, total RAM/VRAM, or Colab runtime availability. The renderer ID pins the existing reviewed Windows Vital 1.6.4/DawDreamer foundation; no new native integration run or Linux parity claim is made here. The project remains free-first, using the local machine and optional free Colab after renderer qualification.

## Work unlocked next

Build a small authored pilot population inside the reviewed schema and asset vocabulary, freeze its ancestry/splits, and implement the bounded online renderer queue and first codec. Native QA and a coupled producer/consumer benchmark then determine practical throughput and memory budgets. This route does not depend on classifying the entire downloaded corpus first.

For broader corpus use, register actual reviewed factory references, extend the schema for selected modern versions, review additional ancestry and usage status, and rerun static eligibility plus native QA. Keep resulting evaluation groups frozen before sampling variants or fitting any LFO assets. Linux/Colab qualification remains a separate platform gate. The contracts unit is complete; the larger M0 contracts-and-measurement milestone remains open.
