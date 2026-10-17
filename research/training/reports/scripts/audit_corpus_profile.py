"""Audit the complete corpus with the curve-input profile and qualified factory catalog."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

RESEARCH = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(RESEARCH / "training"), str(RESEARCH / "data_generation")]


def main():
    from obruxo_data.hashing import canonical_sha256
    from obruxo_data.render.base import FULL_CURVE_RENDERER_ID
    from obruxo_data.vital.full_contract import FullPresetContract
    from obruxo_training.asset_codec import FactoryAssetCodec
    from obruxo_training.audit import audit_corpus, load_snapshot, write_json
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("corpus", "asset-bank", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    bank = json.loads(args.asset_bank.read_text())
    assets, schema = FactoryAssetCodec(bank), FullPresetContract(renderer_id=FULL_CURVE_RENDERER_ID)
    root = args.corpus.resolve(strict=True)
    if not root.is_dir() or args.output.resolve().is_relative_to(root):
        raise ValueError("audit output must be outside the source corpus")
    files = [Path(__file__), *(RESEARCH / "training/obruxo_training").glob("*.py"),
             RESEARCH / "data_generation/obruxo_data/render/curve_inputs.py", RESEARCH / "data_generation/obruxo_data/vital/full_contract.py"]

    def sources():
        return {path.relative_to(RESEARCH).as_posix(): hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest() for path in files}

    plan = {"audit": "full_curve_registered_factory_corpus_admission_v1", "source_sha256_lf": sources(),
            "factory_bank_id": bank["bank_id"], "renderer_id": schema.renderer_id, "catalog_id": assets.catalog.catalog_id,
            "claim": "static_eligibility_only_no_native_audio_or_model_quality", "final_test_audio": False,
            "fits_assets_or_decoders": False, "corpus_modified": False}
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "plan.json", plan)
    started = perf_counter()
    result = audit_corpus(root, args.output / "snapshot", schema=schema, catalog=assets.catalog,
                         progress=lambda done, total: print(json.dumps({"inspected": done, "total": total}), flush=True))
    contract, catalog, manifest = load_snapshot(args.output / "snapshot", schema)
    performances = json.loads((args.output / "snapshot/performances.json").read_text())
    gates = {"source_unchanged": sources() == plan["source_sha256_lf"], "population_complete": result["population_complete"],
             "performance_bank_matches_contract": canonical_sha256(performances) == contract.performance_bank_id,
             "catalog_and_manifest_verified": catalog.catalog_id == plan["catalog_id"] and manifest.manifest_id == result["split_manifest_id"]}
    summary = {"plan_sha256": canonical_sha256(plan), "passed": all(gates.values()), "gates": gates,
               "elapsed_seconds": perf_counter() - started, "summary": result, "final_test_audio": False,
               "fits_assets_or_decoders": False, "corpus_modified": False}
    write_json(args.output / "summary.json", summary)
    print(json.dumps({"passed": summary["passed"], **{key: result[key] for key in (
        "population_complete", "inspected_files", "static_eligible_files", "static_eligible_states", "eligible_states_by_split")}}), flush=True)
    return 0 if summary["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
