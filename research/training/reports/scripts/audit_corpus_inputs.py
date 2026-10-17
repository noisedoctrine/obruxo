"""Check corpus training labels against a complete frozen snapshot without rendering or fitting."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

RESEARCH = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(RESEARCH / "training"), str(RESEARCH / "data_generation")]


def main():
    from obruxo_data.errors import ValidationError
    import numpy as np
    from obruxo_data.hashing import canonical_sha256
    from obruxo_data.render.curve_inputs import prepare_curve_inputs
    from obruxo_training.audit import load_snapshot, write_json
    from obruxo_training.corpus_inputs import CorpusInputCodec, prepare_corpus_labels
    from obruxo_training.lfo_bank import sample_lfo
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("snapshot", "corpus", "asset-bank", "lfo-bank", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    codec = CorpusInputCodec(json.loads(args.asset_bank.read_text()), json.loads(args.lfo_bank.read_text()))
    _, catalog, manifest = load_snapshot(args.snapshot, codec.schema)
    if catalog.catalog_id != codec.catalog.catalog_id:
        raise ValueError("corpus snapshot uses a different registered asset catalog")
    records = json.loads((args.snapshot / "members.json").read_text())["records"]
    members = {}
    for row in records:
        if row["static_eligible"] and row["split"] == "train":
            members.setdefault(row["identity"]["state"], row)
    root = args.corpus.resolve(strict=True)
    if not root.is_dir() or args.output.resolve().is_relative_to(root):
        raise ValueError("audit output must be outside the source corpus")
    files = [Path(__file__), *(RESEARCH / "training/obruxo_training").glob("*.py"),
             RESEARCH / "data_generation/obruxo_data/render/curve_inputs.py", RESEARCH / "data_generation/obruxo_data/vital/full_contract.py"]

    def sources():
        return {path.relative_to(RESEARCH).as_posix(): hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest() for path in files}

    plan = {"audit": "frozen_corpus_training_label_compatibility_v1", "source_sha256_lf": sources(), "codec_id": codec.codec_id,
            "split_manifest_id": manifest.manifest_id, "catalog_id": catalog.catalog_id, "eligible_training_states": len(members),
            "split": "train", "fits_assets_or_decoders": False, "render_attempts": 0, "final_test_audio": False}
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "plan.json", plan)
    started, rows, entries, counts, changes_by_path = perf_counter(), [], [], Counter(), Counter()
    curve_error, active_curve_values, active_curves = 0., 0, 0
    for index, (state_id, member) in enumerate(sorted(members.items()), 1):
        source = root / member["relative_path"]
        path = source.resolve(strict=True)
        if not path.is_relative_to(root) or source.is_symlink():
            raise ValueError("snapshot member escapes the source corpus")
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != member["identity"]["raw"]:
            raise ValueError("frozen corpus member changed")
        changes, status = [], "encoded"
        try:
            prepared, changes = prepare_corpus_labels(json.loads(raw), codec.base, codec.schema)
            target = codec.encode(raw)
            native, _ = prepare_curve_inputs(prepared)
            original_curves = np.stack([sample_lfo(shape) for shape in native["settings"]["lfos"]])
            mask = np.asarray(codec.active_lfos(target), dtype=bool)
            error = (original_curves - codec.curves(target))[mask] ** 2
            curve_error += float(error.sum())
            active_curve_values += error.size
            active_curves += int(mask.sum())
            entries.append({"state_id": state_id, "lineage_id": member["lineage_id"], "relative_path": member["relative_path"],
                            "raw_sha256": member["identity"]["raw"], "target": target.to_dict(), "label_changes": changes})
        except (ValidationError, ValueError, TypeError, KeyError) as error:
            status = type(error).__name__ + ": " + str(error)[:160]
        counts[status] += 1
        changes_by_path.update(change["path"] for change in changes)
        rows.append({"state_id": state_id, "status": status, "label_changes": changes})
        if index % 100 == 0 or index == len(members):
            print(json.dumps({"checked_training_states": index, "total": len(members), "encoded": len(entries)}), flush=True)
    unchanged = sources() == plan["source_sha256_lf"]
    summary = {"plan_sha256": canonical_sha256(plan), "source_unchanged": unchanged, "codec_id": codec.codec_id,
               "eligible_training_states": len(members), "encoded_training_states": len(entries), "counts": dict(counts),
               "label_changes_by_path": dict(changes_by_path), "elapsed_seconds": perf_counter() - started, "checks": rows,
               "active_lfo_projection": {"shapes": active_curves, "sample_values": active_curve_values,
                                         "mse": curve_error / active_curve_values if active_curve_values else None},
               "render_attempts": 0, "fits_assets_or_decoders": False, "final_test_audio": False,
               "claim": "training_label_compatibility_not_native_source_admission_or_model_quality"}
    write_json(args.output / "candidates.json", {"codec": codec.to_dict(), "split_manifest": manifest.to_dict(), "entries": entries})
    write_json(args.output / "summary.json", summary)
    if not unchanged:
        raise ValueError("corpus label implementation changed during the audit")
    print(json.dumps({key: summary[key] for key in ("eligible_training_states", "encoded_training_states", "counts", "elapsed_seconds")}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
