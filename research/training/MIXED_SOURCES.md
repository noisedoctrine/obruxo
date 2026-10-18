# Original corpus and authored coverage in one training stream

`MixedPool` combines the original corpus reader with the authored connection-coverage pool under the long-observation model contract. It retains the registered asset and LFO banks. The source artifacts remain independently identified and unchanged; the mixture stores private, relocatable references to them, plus corpus validation labels and a canonical-program census. Model exports contain the model codec, not dataset paths or source presets.

The union manifest preserves both parent ledgers, including every original state, lineage and split. It never rehashes a subset to assign new splits. Scheduled entries contain all training and validation sources; final-test entries are excluded. A separate static census includes the reserved sources only to check split isolation. It renders no audio and fits no representation. The corpus validation labels use the existing frozen decoder, without fitting on validation or test data.

The program census runs before lossy curve encoding. It applies the declared corpus label/native-curve rules, canonical registered asset metadata, curve names and outer display fields. All scalar controls, phases, seed bits and ordered connections stay in the identity. A program alias crossing frozen splits rejects the mixture. Same-split aliases share an evaluation unit along with their existing authored family, so variants and duplicates are not counted as independent validation cases. This is a declared program-alias check, not acoustic or general near-duplicate detection.

## Labels, source reads and caching

Both source artifacts use the reviewed curve encoder and identical assets/decoder banks. Continuous layouts must match exactly; categorical domains may only extend the existing ordered prefix. Existing corpus labels retain their exact encoder identity. Authored target reuse additionally requires the common encoder and unchanged label preparation; a changed preparation path is re-encoded and checked explicitly. Every source program is inspected for the alias census. These checks avoid rebuilding the same authored programs twice during startup.

The existing authored loader retains full source verification by default. The mixed reader requests metadata-only startup verification and then verifies the source when it is read. Each read checks the source index, frozen state, original byte hash, program identity and re-encoded target before rendering. Changed targets, source bytes, source artifacts, banks, aliases or memberships fail closed. Original source documents and phase controls supply training audio; label-prepared or decoded presets are not substituted.

## Source-balanced schedule

`--sampling-policy source_balanced` alternates authored and corpus **attempts**, with equal counts per complete epoch. Both groups receive seeded permutations. The smaller group cycles only after all of its states have been visited; a complete epoch visits every training state in both sources. QA failures are recorded and acknowledged without silently changing which source is scheduled next. Accepted replay proportions are measured separately and need not be equal.

Each state keeps the attack-compatible reference views from the earlier bank where possible, retaining the expressive performances. A state that needs a longer note uses its shortest adequate extended reference duration. This avoids choosing forty-second notes unnecessarily for ordinary patches. The long-observation contract still retains its full declared bank, thirty-two-second tail and eighty-second feature cap. Longer tails add CPU work; measure coupled throughput before scaling.

## Commands

```text
python research/training/run.py mix-sources --corpus-pool ORIGINAL_OR_EXTENDED_PRIVATE_POOL.json --authored-pool GRAPH_POOL.json --snapshot CORPUS_AUDIT/snapshot --output PRIVATE_MIXTURE.json
python research/training/reports/scripts/qualify_mixed_sources.py --pool PRIVATE_MIXTURE.json --coverage research/training/reports/data/graph_coverage.json --output NEW_QUALIFICATION_DIRECTORY
python research/training/run.py train --pool PRIVATE_MIXTURE.json --output NEW_TRAINING_RUN --sampling-policy source_balanced
```

Keep the private source-artifact references and original corpus root with the dataset, or update their private wrapper locations after copying identical artifacts. IDs and source checks still apply. The [qualification report](reports/MIXED_SOURCE_RESULTS.md) distinguishes preserved authored supervision, original-source ingestion, exact checkpoint recovery and held-out validation ingestion from useful learned quality. Final-test audio, broad data scaling, decoder policy, source robustness and release remain separate gates. Compute remains free-first and GitHub Actions remains disabled.
