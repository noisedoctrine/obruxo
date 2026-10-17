"""Check canonical program aliases across frozen splits before any lossy curve encoding."""
import argparse
from collections import defaultdict
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
    from obruxo_data.render.curve_inputs import prepare_curve_inputs
    from obruxo_data.vital.full_contract import FullPresetContract
    from obruxo_training.asset_codec import FactoryAssetCodec, _content
    from obruxo_training.audit import load_snapshot, write_json
    from obruxo_training.corpus_inputs import prepare_corpus_labels
    from obruxo_training.identity import DISPLAY_FIELDS
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("snapshot", "corpus", "asset-bank", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    assets = FactoryAssetCodec(json.loads(args.asset_bank.read_text()))
    schema = FullPresetContract(renderer_id=FULL_CURVE_RENDERER_ID)
    _, catalog, manifest = load_snapshot(args.snapshot, schema)
    if catalog.catalog_id != assets.catalog.catalog_id:
        raise ValueError("alias audit requires the snapshot's registered asset catalog")
    root = args.corpus.resolve(strict=True)
    if not root.is_dir() or args.output.resolve().is_relative_to(root):
        raise ValueError("audit output must be outside the source corpus")
    records = json.loads((args.snapshot / "members.json").read_text())["records"]
    members = {row["identity"]["state"]: row for row in records if row["static_eligible"]}
    known = {kind: {_content(kind, value): value for value in values}
             for kind, values in (("wavetable", assets._waves), ("sample", assets._samples))}
    files = [Path(__file__), *(RESEARCH / "training/obruxo_training").glob("*.py"),
             RESEARCH / "data_generation/obruxo_data/render/curve_inputs.py", RESEARCH / "data_generation/obruxo_data/vital/full_contract.py"]

    def sources():
        return {path.relative_to(RESEARCH).as_posix(): hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest() for path in files}

    plan = {"audit": "canonical_source_program_aliases_before_lossy_encoding_v1", "source_sha256_lf": sources(),
            "split_manifest_id": manifest.manifest_id, "catalog_id": catalog.catalog_id, "eligible_states": len(members),
            "canonicalization": ["declared_corpus_label_policy", "native_curve_input_precision", "registered_asset_names_and_authors", "curve_names", "outer_display_fields"],
            "scalar_changes": False, "random_phases_fixed": False, "splits_changed": False, "fits_or_renders": False,
            "final_test_audio": False, "claim": "declared_program_alias_check_not_acoustic_or_near_duplicate_detection"}
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "plan.json", plan)
    groups, started = defaultdict(list), perf_counter()
    for index, (state, row) in enumerate(sorted(members.items()), 1):
        source = root / row["relative_path"]
        path = source.resolve(strict=True)
        if not path.is_relative_to(root) or source.is_symlink():
            raise ValueError("corpus member escapes its frozen source root")
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != row["identity"]["raw"]:
            raise ValueError("frozen source changed")
        document, _ = prepare_corpus_labels(json.loads(raw), assets, schema)
        document, _ = prepare_curve_inputs(document)
        for name in DISPLAY_FIELDS:
            document.pop(name, None)
        settings = document["settings"]
        settings["wavetables"] = [known["wavetable"][_content("wavetable", asset)] for asset in settings["wavetables"]]
        settings["sample"] = known["sample"][_content("sample", settings["sample"])]
        shapes = settings["lfos"] + settings["custom_warps"] + [route["line_mapping"] for route in settings["modulations"] if "line_mapping" in route]
        for shape in shapes:
            shape.pop("name", None)
        groups[canonical_sha256(document)].append({"state_id": state, "split": row["split"]})
        if index % 250 == 0 or index == len(members):
            print(json.dumps({"inspected": index, "total": len(members)}), flush=True)
    aliases = {key: rows for key, rows in groups.items() if len(rows) > 1}
    cross = {key: rows for key, rows in aliases.items() if len({row["split"] for row in rows}) > 1}
    unchanged = sources() == plan["source_sha256_lf"]
    summary = {"plan_sha256": canonical_sha256(plan), "passed": unchanged and not cross, "source_unchanged": unchanged,
               "eligible_states": len(members), "distinct_programs_before_lossy_encoding": len(groups), "alias_groups": aliases,
               "cross_split_alias_groups": cross, "elapsed_seconds": perf_counter() - started, "splits_changed": False,
               "source_presets_modified": False, "fits_or_renders": False, "final_test_audio": False}
    write_json(args.output / "summary.json", summary)
    print(json.dumps({"passed": summary["passed"], "alias_groups": len(aliases), "cross_split_alias_groups": len(cross)}), flush=True)
    return 0 if summary["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
