# Attribution sources for local inference snapshots

`upstream/` preserves the original Basic Pitch LICENSE and NOTICE at the model's pinned Spotify revision, plus nnAudio's MIT text at the separately recorded license-source revision. `manifest.json` records original byte hashes and source URLs. The nnAudio license-source revision identifies the retrieved license, not an asserted exact code ancestry revision.

OBRUXO converts the pinned ICASSP-2022 ONNX weights to PyTorch, implements its CQT/normalized-log/harmonic frontend in PyTorch, and ports its note decoder into a dependency-light runtime. The source checkpoint/model identity remains in the Basic Pitch artifact metadata. Upstream's NOTICE is retained verbatim, including references to dependencies/data used by the broader upstream project; those references do not declare them installed in this inference runtime.

For a locked local environment, `reports/scripts/collect_runtime_notices.py` collects original license/notice files and checks each against its installed distribution's RECORD. It records wheel pins, exact versions, any declared license expressions, original paths and hashes. It neither installs packages nor infers a license expression when metadata omits one. Include that archive with `bundle-basic --runtime-notices ARCHIVE`.

The inventory and snapshot keep `release_notices_complete=false`. The repository has no selected project-wide license file, and the copied Vital initialization/schema data need their distribution terms resolved against their recorded upstream provenance. This work does not choose an owner license or certify distribution permission. Model quality, listening and release qualification also remain separate gates.
