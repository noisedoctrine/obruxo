# Fresh validation of training-state coverage

Completed 2026-10-15 (Asia/Kuala_Lumpur). Seven existing checkpoints were evaluated without retraining on all 23 previously unused validation states added by the frozen 256-to-512-state pool expansion. All 64 final-test states remain reserved. This is a development diagnostic for the limited authored vocabulary, not a finished general synthesizer model.

The [pre-run plan](data/fresh_validation_plan.json) has SHA-256 `7dc79b0935bb0eda2c046dd54f77b051fa9f4c86686d96d107042c364018b90a`. It fixes all checkpoints, source hashes, retrieval records, case/performance pairs, metrics, failure rules and the decision rule before rendering. The [data](data/fresh_validation_summary.json) retain per-state scores and artifact hashes; the [plot](images/fresh_validation.png) shows all paired differences.

## Paired results

Each state averages reference and transfer MR-STFT equally. Negative expanded-minus-205 deltas favor the larger training population. Seeds 1 and 2 are the prespecified primary comparisons; seed 0 remains descriptive discovery. Bootstrap intervals resample states within each seed (2,000 resamples, seed 0); they do not measure variation across learning seeds.

| Learning seed | Paired cases | Mean delta | State bootstrap 95% interval | Expanded wins |
|---|---:|---:|---:|---:|
| 0 (discovery) | 23/23 | -0.1114 | [-0.2457, 0.0187] | 14/23 |
| 1 | 23/23 | -0.0717 | [-0.2346, 0.0824] | 12/23 |
| 2 | 23/23 | -0.0589 | [-0.1909, 0.0706] | 11/23 |

The descriptive mean across replication seeds 1 and 2 is **-0.0653**. Both favor expanded coverage on this panel. The prespecified rule supports prioritizing broader coverage, subject to the parameter tradeoffs below; two seeds and 23 states do not establish general superiority.

Both intervals include zero. Expanded training also lowers unison accuracy from 91.3% to 82.6% for seed 1 and from 95.7% to 78.3% for seed 2, and increases release MAE for both seeds. Attack and decay improve for both; oscillator-level and RMS effects are mixed. Keep the 404-state population as the working pool, but investigate these unison/release regressions with a controlled ablation on the existing cached features before expanding again. Freeze that experiment separately; this evaluation does not identify the cause or establish that changing loss weights will fix it.

The original 21-state panel produced deltas of +0.0026 and -0.0405 for seeds 1 and 2. This panel is disjoint from that validation set and all training records. It is still validation: prior development influenced the candidates and protocol, and these 23 states are now consumed for development. No final-test or significance claim follows.

| Candidate | Scored | Primary | Reference | Transfer | Active MAE | Unison accuracy | Filter accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|
| incumbent | 23/23 | 1.0321 | 0.9710 | 1.0932 | 0.2079 | 0.9130 | 1.0000 |
| one_seed0_16000 | 23/23 | 0.9511 | 0.8574 | 1.0448 | 0.2324 | 0.8261 | 1.0000 |
| expanded_seed0_16000 | 23/23 | 0.8397 | 0.7760 | 0.9034 | 0.1965 | 0.7391 | 1.0000 |
| one_seed1_16000 | 23/23 | 1.0135 | 0.9207 | 1.1063 | 0.2269 | 0.9130 | 0.9130 |
| expanded_seed1_16000 | 23/23 | 0.9418 | 0.8933 | 0.9904 | 0.2176 | 0.8261 | 1.0000 |
| one_seed2_16000 | 23/23 | 0.9369 | 0.8439 | 1.0298 | 0.2267 | 0.9565 | 0.9565 |
| expanded_seed2_16000 | 23/23 | 0.8779 | 0.7972 | 0.9587 | 0.2189 | 0.7826 | 1.0000 |

| Candidate | Attack MAE | Decay MAE | Sustain MAE | Release MAE | Oscillator level MAE | Relative RMS reference | Relative RMS transfer |
|---|---:|---:|---:|---:|---:|---:|---:|
| incumbent | 0.1809 | 0.2762 | 0.2036 | 0.3339 | 0.1783 | 0.2383 | 0.2563 |
| one_seed0_16000 | 0.1196 | 0.3097 | 0.2809 | 0.3568 | 0.2264 | 0.2487 | 0.2299 |
| expanded_seed0_16000 | 0.1029 | 0.2404 | 0.2323 | 0.3249 | 0.2499 | 0.2012 | 0.2105 |
| one_seed1_16000 | 0.1606 | 0.2790 | 0.2544 | 0.3479 | 0.2325 | 0.2093 | 0.2701 |
| expanded_seed1_16000 | 0.1335 | 0.2473 | 0.2420 | 0.3907 | 0.2359 | 0.2816 | 0.2556 |
| one_seed2_16000 | 0.1163 | 0.2840 | 0.2557 | 0.3354 | 0.3013 | 0.1954 | 0.2299 |
| expanded_seed2_16000 | 0.1000 | 0.2661 | 0.2711 | 0.3625 | 0.2529 | 0.2289 | 0.2563 |

Parameter MAE uses normalized codec units and excludes inactive controls. Categorical accuracy and level/envelope errors describe the emitted patches; they do not replace audio distance.

| Baseline | Scored | Primary |
|---|---:|---:|
| fixed_template | 23/23 | 6.5702 |
| train_retrieval | 23/23 | 1.9768 |
| source_repeat | 23/23 | 0.0000 |

## Rendering compatibility and verification

Main had changed the native header adapter, scalar tolerance and provenance while retaining the historical v2 renderer ID. The general renderer now uses a distinct v3 ID; the training contract explicitly retains v2 behavior. Vita's known 1.0.8-to-development-header round trip is classified narrowly; unknown header drift still fails. Before fresh validation, six training replay records covering all unison/filter categories matched their retained audio hashes and every timbre/pitch tensor exactly ([qualification](data/fresh_validation_renderer_qualification.json), 18.51 seconds). This is evidence for those six records on this runtime, not a universal bit-determinism guarantee.

The evaluation attempted **506/506** planned renders: **506 accepted, 0 rejected, 0 errors**. Emitted prediction presets passed codec validation: **161/161**. All accepted render identities match the checkpoints, and the fixed 205-state retrieval digest remains `1831f6a3aeedfcd00991ab98fa8ee106bb4c70970d0c2686a3709e7b6b60d6ac`. The 23 cases were checked against 105 retained prior case files before rendering, with no overlap.

Evaluation took **20.97 recorded minutes**, excluding setup, tests, qualification and analysis. It used local CPU float32, four PyTorch threads and one native worker. No optimizer steps, new training features, GPU, paid compute or WAV archive were needed. The qualification cost is separate; local tests ran concurrently during part of evaluation, so timing is an observation rather than a throughput benchmark. Colab was unnecessary for this run.

GitHub Actions remains disabled as requested. The local final gate passed **238 tests with no skips in 264.41 seconds**, including training tests, data-generation unit/integration tests and the simple-training pipeline/contract tests with `OBRUXO_VALIDATE_NATIVE=1`. Ruff and whitespace checks passed locally as well.

The evaluator's automatic best checkpoint is `expanded_seed0_16000`. No candidate is promoted: the existing seed-0 bundle remains byte-for-byte unchanged at `research/training/outputs/state-coverage-validation-20261013/selected-model.pt`, SHA-256 `5517db59dec68f0daca30789d08e75f41183e5577ba5a2082a4dc9488da15d4f`. Raw evaluation artifacts remain under ignored `research/training/outputs/fresh-validation-20261014`.

## Reproduction

Use the dependency/runtime and source hashes in the frozen plan. From the repository root with the retained checkpoint and pool files available:

```sh
python research/training/run.py compare \
  --pool research/training/outputs/authored-pilot-v2-20261005/pool.json \
  --validation-pool research/training/outputs/authored-expanded-20261013/pool.json \
  --checkpoints incumbent=research/training/outputs/two-view-train-20261010/checkpoint.pt \
    one_seed0_16000=research/training/outputs/cached-curve-one-step16000-20261011/checkpoint.pt \
    expanded_seed0_16000=research/training/outputs/state-coverage-expanded-16000-20261013/checkpoint.pt \
    one_seed1_16000=research/training/outputs/state-coverage-replication-one-seed1-16000-20261014/checkpoint.pt \
    expanded_seed1_16000=research/training/outputs/state-coverage-replication-expanded-seed1-16000-20261014/checkpoint.pt \
    one_seed2_16000=research/training/outputs/state-coverage-replication-one-seed2-16000-20261014/checkpoint.pt \
    expanded_seed2_16000=research/training/outputs/state-coverage-replication-expanded-seed2-16000-20261014/checkpoint.pt \
  --checkpoint-pools expanded_seed0_16000=research/training/outputs/authored-expanded-20261013/pool.json \
    expanded_seed1_16000=research/training/outputs/authored-expanded-20261013/pool.json \
    expanded_seed2_16000=research/training/outputs/authored-expanded-20261013/pool.json \
  --retrieval-checkpoint research/training/outputs/full-pilot-step4000-20261009/checkpoint.pt \
  --output research/training/outputs/fresh-validation-reproduction --count 23 --timeout 30
```

Use a new output directory, or add `--resume` only for an interrupted identical plan. Preserve every failed case and all logged attempts. Existing complete cases are hash-verified and reused; an unfinished case is rendered again in full. A native request must pass the declared QA and renderer-identity gates.
