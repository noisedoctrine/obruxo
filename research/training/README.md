# Training contracts, online pilot and trainer

The [delivery status](TRAINING_STATUS.md) tracks the remaining training and release work. [Learned factory choices](FACTORY_ASSETS.md) add registered wavetables for all three oscillator slots and registered sampler content, with hash-pinned native import qualification. Curves and ordered modulation connections are the next missing capabilities.

The [full scalar training path](FULL_SCALAR_TRAINING.md) connects all 775 scalar fields to bounded audio-only learning, legal effect-order and pitch-mask outputs, modern render recipes and inference export. Nested assets, curves, seeds and connections remain explicitly fixed in this increment; learned structured coverage follows.

The [full-preset render profile](FULL_RENDER_PROFILE.md) adds an explicit path for all 775 controls and six nested structures in the reviewed 1.5.5 inventory. It checks the broader native state before broader model training; existing pilot and alpha artifacts retain their identities.

The [performance-conditioning comparison](CONDITIONING.md) is complete. Its [fixed result](reports/CONDITIONING_RESULTS.md) did not show a benefit from attention (+1.96% mean sound error, interval spanning zero), so the original model stays the default. The next major work is full declared synthesis/performance coverage, followed by data scaling and combined training; further narrow tuning remains deferred.

The [audio-only candidate ranker](AUDIO_CANDIDATES.md) can choose among up to four legal patches using notes estimated from the recording. Its [completed native check](reports/AUDIO_RANKING_RESULTS.md) reduced mean validation sound error by 10.64% against ordinary prediction, with full ranked coverage and a 15.99s model-warm p95. It passed the bounded optional-path gates; this is not a finished release model or final-test result.

The [direct-render training path](RENDER_SELECTED_TRAINING.md) collects bounded candidate guesses, confirms improvements with independent Vital renders and trains matched inverse models with unchanged source supervision. Its [completed comparison](reports/RENDER_SELECTED_RESULTS.md) found confirmed alternatives for 13/32 training recordings, but extra-target training raised held-out sound error by 12.56%. It remains a research option; no release model is promoted. The next independent capabilities are audio-only candidate ranking and a conditioning comparison. Renderer throughput is being measured before larger native runs.

The [sound-prediction helper](AUDIO_PROXY.md) implements the next audio-based learning step: learn patch-and-performance-to-sound measurements, then check its rankings and suggested changes against native Vital before connecting it to inverse training. `proxy-screen` provides bounded on-demand collection, resumable CPU learning and a reserved-performance validation panel.

The [completed helper check](reports/SOUND_PROXY_RESULTS.md) passes familiar-performance prediction but fails transfer and native-update qualification. Its predicted error improves on all 20 guided cases while real sound error worsens on all 20. The helper remains disconnected from inverse training, and ordinary artifact loading refuses it. This is a limitation of the current helper/update, not a rejection of audio-based training.

This workspace implements the training pipeline from the [training plan](../modelling/TRAINING_PLAN.md). It identifies and audits presets, resolves registered asset content, freezes ancestry-aware splits, reserves evaluation performances, and builds immutable on-demand render recipes using `obruxo_data`.

The [authored online pilot](ONLINE_PILOT.md) supplies a limited codec and bounded rendering during training. The [trainer](TRAINER.md) adds audio-only prediction, frozen Basic Pitch features, bounded feature replay, optimizer recovery, resumable native evaluation and offline inference.

The [broader alpha path](ALPHA.md) adds two oscillators, pinned factory wavetable selection and scanning, LFO filter modulation and chorus, with a five-level authored coverage curriculum. Its model heads, inactive masks and offline assets follow the saved codec. This is the current capability-expansion direction; further pilot release-head tuning is deferred while synthesis identity, audio-based learning and candidate refinement are screened.

The [alpha qualification report](reports/ALPHA_SYNTHESIS_RESULTS.md) records native asset equivalence, feature activation and a bounded CPU training run. It establishes the broader pipeline, with no alpha quality promotion.

[Paired synthesis-identity learning](PAIRED_IDENTITY.md) adds an explicit cross-performance representation objective and a matched, bounded screen on the alpha vocabulary. The next independent direction is audio-based learning; further identity-weight or pilot-head tuning is deferred until the larger directions have been screened.

The [identity feasibility result](reports/PAIRED_IDENTITY_RESULTS.md) shows no held-out benefit for the fixed weight-one setting: native error increases 8.25% and cross-performance retrieval drops from 25% to 20%. The optional loss keeps a zero default. All 120 comparison renders passed; the training cache retained 63 complete QA-qualified pairs after one clipped source view.

The [fresh validation](reports/FRESH_VALIDATION_RESULTS.md) compares the existing 205- and 404-state checkpoints on all 23 previously unused validation states. Expanded-minus-205 deltas are -0.0717 and -0.0589 for learning seeds 1 and 2; both intervals include zero. Audio means improve, while unison accuracy and release MAE regress. All 506 renders and 161 presets passed, with no retraining or paid compute.

The subsequent [release diagnostic](reports/RELEASE_DIAGNOSTIC_RESULTS.md) separates training-label fit, train-only feature retrieval and a native release-only intervention. Training unison accuracy is already 100%; release shows a large validation gap, while pitch features contain useful release information under a fixed neighbor probe. Changing only release lowers native audio distance by 3.0% and 3.2% across the two replication seeds, with all 276 renders passing in 18.72 CPU minutes. These are the same 23 cases reused as development validation. The seed-0 selected pilot export remains current, and all 64 pilot test states stay reserved. The release-head comparison proposed in that historical report is deferred in favor of the broader directions above.

The [state-coverage discovery](reports/STATE_COVERAGE_RESULTS.md), [original-panel replication](reports/STATE_COVERAGE_REPLICATION_RESULTS.md), [cached-feature curve](reports/CACHED_CURVE_RESULTS.md), [two-view experiment](reports/TWO_VIEW_RESULTS.md), [full-pilot results](reports/FULL_PILOT_RESULTS.md) and [pool expansion readiness](reports/POOL_EXPANSION_READINESS.md) retain the earlier evidence and controls.

The local corpus result and current capability limits are in [CONTRACT_READINESS.md](reports/CONTRACT_READINESS.md). A complete audit is not a training-ready dataset: static eligibility, native loading, audio QA, codec coverage, and permission to use/distribute a source are separate checks. The trained model covers only the narrow authored pilot.

The first trainer exposed silent native rejection of Vita's synthetic version marker. The renderer and pilot codec now have new identities. Old pilot pools, render cursors and training checkpoints must not be resumed under these contracts. Historical static corpus counts remain useful, but a historical snapshot fails current executable-contract verification; create a new audit to obtain a current snapshot. The older [native report](reports/ONLINE_PILOT_RESULTS.md) is explicitly superseded as evidence of patch fidelity.

## Run from the repository root

Python 3.12 is required. The existing data-generation environment already supplies the dependencies. For a small, CPU-only contracts environment:

```sh
python -m pip install -r research/training/requirements-contracts.txt
python research/training/run.py describe
python research/training/run.py audit --corpus datasets/presetshare/raw/presetshare_files --output research/training/outputs/my-contract-snapshot
python research/training/run.py verify --snapshot research/training/outputs/my-contract-snapshot
python -m pytest research/training/tests -q
ruff check research/training/obruxo_training research/training/tests research/training/run.py
```

These commands do not need PyTorch, a GPU, Vita, a Vital installation, Colab, or rendered audio. Native rendering later uses the separately pinned [data-generation environment](../data_generation/environment.yml). The [CI workflow](../../.github/workflows/training-contracts.yml) runs static contract tests on Linux; passing it does **not** qualify the Linux synth.

`audit` never edits source presets. Its output must be a **new directory outside the corpus**. Source traversal errors and skipped linked directories prevent a complete snapshot. `--limit N` supports a bounded diagnostic, marked partial if it omits files. Exit status is 0 for a complete audit, 2 for a partial audit, and 1 for a command failure. A complete audit can have zero eligible states.

`verify` refuses partial snapshots, changed artifacts, inconsistent ancestry/splits, or incompatible current contracts. It validates the saved snapshot; it does not rescan source paths. Recipes recheck the actual input preset and require its state to be registered in the split ledger.

## Snapshot files

All generated output under `outputs/` is Git-ignored. Keep private membership files local; commit only reviewed aggregate reports.

| File | Contract |
|---|---|
| `contract.json` | Schema content hashes, eligibility policy, asset and performance bank IDs, renderer ID, MIDI limits, audio format, tail and buffer budgets |
| `assets.json` | Registered asset content fingerprints, source hashes and declared provenance; no asset payloads |
| `performances.json` | Exact MIDI performance bank, identities and reference/transfer roles |
| `splits.json` | Frozen state-to-lineage-to-split assignments, seed and identity/split rules |
| `members.json` | Private relative paths, raw/canonical/state hashes, declared parent edges, static diagnostics and exclusions |
| `summary.json`, `REPORT.md` | Aggregate eligibility/coverage counts; overlapping reasons are not additive |
| `status.json` | Completion flag and hashes for the six JSON artifacts; written last |

Preserve the snapshot ID with experiment metadata. Start a new run for a changed contract or population, and freeze the split seed before evaluating models.

## Preset identity and ancestry

`patch_identity_v1` matches the existing corpus identity audit, including a fixed regression vector. It records:

- **Raw:** SHA-256 of original bytes, including formatting and any UTF-8 BOM.
- **Canonical:** SHA-256 of a sorted, type-tagged JSON tree, preserving array order and the string representation of parsed `Decimal` values. It distinguishes integers from decimals and retains decimal precision without binary-float rounding.
- **Stored state:** The same tree with only top-level `author`, `comments`, and `preset_name` removed. Nested metadata, macro labels, array order, and all other fields remain significant.

Duplicate JSON keys, nonfinite constants, invalid UTF-8 and non-object roots are rejected. Stored-state identity is not acoustic equivalence or a near-duplicate detector.

Pass `--parents path/to/parents.json` when reviewed ancestry is known. Its JSON object maps child stored-state hashes to lists of parent stored-state hashes. Every parent must be in the audited population, and cycles are rejected. Exact stored-state duplicates collapse automatically. Declared ancestry is joined transitively **before** splitting; the lexicographically smallest member identifies each initial lineage. Unknown near-duplicate relationships are not inferred.

`lineage_hash_80_10_10_v1` hashes the seed and lineage ID into train/validation/test with 80/10/10 probabilities. Small populations need not have those exact counts. Freeze the manifest before generating variants or fitting any learned codebook, normalization, retrieval index or decoder asset.

For on-demand mutations, `SplitLedger.register_variant(new_state_id, parent_state_id)` inherits the parent's existing lineage and split without rehashing. It refuses a collision with another frozen lineage, even when both lineages happen to be in the same split. Save `ledger.to_dict()` with future checkpoints and restore it with `SplitLedger.from_dict(saved, manifest)`. Augmented audio also belongs to its source lineage; it does not create a new independent split unit.

## Static schema and asset gate

The initial policy accepts version strings `1.0.0` through `1.0.8` and the reviewed Vita init marker `99999.9.9`, then applies the existing **772-control schema**. This version allowlist is not native-render certification. Unknown/missing controls, invalid scalar values, malformed required structures, and unsupported versions remain explicit failures; no fields are dropped or silently default-filled.

The default asset catalog contains only the reviewed init wavetable and sample fingerprints. It does not claim to enumerate Vital's factory library. `asset_content_v1` ignores only an asset object's root `name` and `author`; it preserves DSP content, versions, encodings, array order and numeric representation. A renamed identical asset can match; an edited asset with a familiar name cannot. This conservative comparison may leave equivalent alternate encodings unresolved.

Each of the three stored oscillator assets and the sample is inspected. Unresolved **active** assets block eligibility. Unresolved inactive content is recorded; enabling it in a later variant requires a new successful inspection. A malformed wavetable object cannot bypass the asset gate. An unresolved fingerprint is not a claim that the asset is custom, imported, unsafe, or unlicensed.

To supply reviewed references, pass `--asset-references` **before** the command:

```sh
python research/training/run.py --asset-references local-references.json audit --corpus path/to/presets --output research/training/outputs/reviewed-references
```

The manifest format is:

```json
{
  "version": 1,
  "references": [
    {
      "asset_id": "my-reviewed-wavetable",
      "kind": "wavetable",
      "slot": 1,
      "path": "relative/path/to/reference.vital",
      "source_sha256": "<SHA-256 of the exact reference file bytes>",
      "provenance": "factory_export"
    }
  ]
}
```

Use `wavetable` slots 1–3 or `sample` with no slot. External provenance is `factory_export` or `authored_reference`; the caller must review that declaration and the applicable usage terms. File hashes verify which export was used, not where it originally came from. Reference paths resolve relative to the manifest. This manifest **replaces** the default init-only catalog. Duplicate IDs or ambiguous fingerprints fail rather than selecting one arbitrarily. No reference blobs are copied into the repository.

## Performance and render recipe boundary

The frozen initial bank has four reference and four transfer patterns. Exact notes, velocities, ticks and tempos are exported in `performances.json`.

| Role | Patterns | Allowed use |
|---|---|---|
| Reference | Mid-register held note, staccato, melody, chord | Training states for training; validation/test states for evaluation |
| Transfer | Low held note, high staccato, wide melody, wide chord, changed tempos/velocities | Evaluation only |

The initial bank is deliberately small. It is not the later randomized curriculum or a statistically sufficient evaluation suite. Adding a broader performance distribution requires a versioned sampling policy with explicit held-out ranges/families. Unknown performances are currently rejected. Roles are checked by actual MIDI content hash, so relabeling a transfer performance cannot admit it to training.

The current renderer boundary is fixed tempo, notes/polyphony, MIDI channel 0, 44.1 kHz stereo float32 audio and an explicit two-second tail. CC, pitch bend, pressure and tempo changes require renderer qualification first. Slow envelopes/effects may require longer tails in a new contract.

From Python, put `research/training` and `research/data_generation` on the import path. The integration seam is:

```python
from pathlib import Path

from obruxo_data.vital import VitalSchema
from obruxo_training.audit import load_snapshot
from obruxo_training.contracts import RenderRecipe
from obruxo_training.performances import initial_bank
from obruxo_training.splits import SplitLedger

schema = VitalSchema.load()
contract, catalog, manifest = load_snapshot(Path("research/training/outputs/my-contract-snapshot"), schema)
ledger = SplitLedger(manifest)
# Select an actually eligible TRAIN member from members.json, or register an authored variant first.
raw_preset = Path("path/to/eligible-training-member.vital").read_bytes()
recipe = RenderRecipe.create(raw_preset, initial_bank()[0].performance(), schema=schema, catalog=catalog,
                             contract=contract, ledger=ledger, purpose="train", sequence_id=0)
request = recipe.render_request()
# Later, in a qualified worker: result = renderer.render(request)
```

The recipe binds raw/canonical/state identity, lineage, split manifest, contract, sequence, performance and purpose to a serialized render request. Subsequent changes to the caller's mutable `Performance` cannot change the recipe. `purpose="train"` rejects held-out states and transfer performances; `purpose="evaluate"` permits the frozen evaluation combinations.

Request and recipe hashes identify experimental conditions, **not exact audio bytes**. Vital repeat renders are not guaranteed bit-identical. The stream and trainer record the actual audio hash, native diagnostics, QA acceptance, failure/retry and consumption separately. Static eligibility alone never authorizes an optimizer update.

## Compute and next implementation boundary

The [native latency report](reports/VITAL_RENDER_LATENCY.md) shows Vital synthesis running well above playback speed. Repeated process/plugin setup and patch-state handling dominate short clips. Bulk state conversion reduced the measured five-case wall time by 20.9%, preserving identical native state and audio. The worker measurement below covers the next source of setup overhead.

The [reusable worker](RENDER_WORKER.md) now keeps Python imports alive while constructing a fresh Vital instance for every patch. The phase-controlled local check accepted all 66 native renders, matched fresh-process audio exactly across 32 jobs and a restart, and reduced paired mean call time by 16.7%. New runs can opt in with `--worker-jobs 32`; existing contracts retain one-job processes. Pipeline correctness checks fix oscillator phases; random-phase variation is tested separately.

The contract declares **one render worker, at most 32 prefetched clips and at most 256 MiB of prefetched audio**. The online stream now enforces both limits, with a smaller 4-clip/32 MiB default. These are buffer budgets, not total application memory estimates. Training audio is rendered in memory on demand; no pre-rendered audio archive or GPU allocation is needed for the contracts and pipeline checks.

The authored pilot, limited codec, online queue, native QA, inverse model, optimizer, training recovery, full authored-split and cached-feature learning curves, verified pool expansion, and resumable real-render evaluation are implemented. Same-pool feature seeding supports multiple training performances per state without re-rendering the first views; named comparisons hold evaluation and retrieval fixed across verified training-pool expansions. The fixed-case two-seed replication is complete; the next work should address broader held-out reconstruction and remaining capability gaps, including wider native QA/coverage, verified factory references, modern schema support and Linux/Colab renderer qualification. Keep the final test split reserved unless a separately planned gate authorizes it. The trainer checkpoints model/optimizer/RNG/replay state with the cursor; a standalone recipe cursor remains insufficient for optimization recovery.
