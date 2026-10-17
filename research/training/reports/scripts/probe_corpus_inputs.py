"""Measure real corpus label-copy equivalence and retain original native source failures."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

RESEARCH = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(RESEARCH / "training"), str(RESEARCH / "data_generation")]


def main():
    from obruxo_data.errors import ValidationError
    from obruxo_data.hashing import canonical_json, canonical_sha256
    from obruxo_data.render.base import RenderRequest
    from obruxo_training.audit import load_snapshot, write_json
    from obruxo_training.benchmark import MemoryMonitor
    from obruxo_training.corpus_inputs import CorpusInputCodec, prepare_corpus_labels
    from obruxo_training.identity import PresetIdentity
    from obruxo_training.performance_eligibility import eligible, longest_held_seconds
    from obruxo_training.performances import bank_for
    from obruxo_training.reproducibility import FULL_PHASE_CONTROL
    from obruxo_training.stream import NativeRendererFactory, RenderWorker
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("snapshot", "corpus", "asset-bank", "lfo-bank", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    codec = CorpusInputCodec(json.loads(args.asset_bank.read_text()), json.loads(args.lfo_bank.read_text()))
    _, catalog, manifest = load_snapshot(args.snapshot, codec.schema)
    if catalog.catalog_id != codec.catalog.catalog_id:
        raise ValueError("snapshot and codec asset catalogs differ")
    records = json.loads((args.snapshot / "members.json").read_text())["records"]
    members = {row["identity"]["state"]: row for row in records if row["static_eligible"] and row["split"] == "train"}
    root = args.corpus.resolve(strict=True)
    if not root.is_dir() or args.output.resolve().is_relative_to(root):
        raise ValueError("probe output must be outside the corpus")
    files = [Path(__file__), *(RESEARCH / "training/obruxo_training").glob("*.py"),
             *(RESEARCH / "data_generation/obruxo_data/render").glob("*.py"), RESEARCH / "data_generation/obruxo_data/vital/full_contract.py"]

    def sources():
        return {path.relative_to(RESEARCH).as_posix(): hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest() for path in files}

    plan = {"probe": "training_corpus_label_copy_equivalence_v1", "source_sha256_lf": sources(), "codec_id": codec.codec_id,
            "split_manifest_id": manifest.manifest_id, "positive_pair_limit": 6, "source_candidate_limit": 16,
            "maximum_native_attempts": 48, "phase_control": FULL_PHASE_CONTROL, "worker_jobs": 1,
            "performance_policy": "longest_attack_compatible_reference", "rss_limit_bytes": 8 * 1024**3,
            "source_failures": "record_and_leave_unadmitted_never_treat_as_success", "final_test_audio": False,
            "fits_assets_or_decoders": False, "model_promotion": False}
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "plan.json", plan)
    checks, events, candidates, accepted, skipped = [], [], 0, 0, []
    monitor, started = MemoryMonitor(), perf_counter()
    monitor.start()
    try:
        with RenderWorker(renderer_factory=NativeRendererFactory(renderer_id=codec.contract.renderer_id), max_jobs=1) as worker:
            for state, member in sorted(members.items()):
                if candidates >= plan["source_candidate_limit"] or accepted >= plan["positive_pair_limit"]:
                    break
                source = root / member["relative_path"]
                path = source.resolve(strict=True)
                if not path.is_relative_to(root) or source.is_symlink():
                    raise ValueError("corpus member escapes its source root")
                raw = path.read_bytes()
                identity = PresetIdentity.from_bytes(raw)
                if identity.state != state or identity.raw != member["identity"]["raw"] or manifest.by_state[state].split != "train":
                    raise ValueError("frozen training source changed")
                try:
                    target = codec.encode(raw)
                    prepared, changes = prepare_corpus_labels(json.loads(raw), codec.base, codec.schema)
                except (ValidationError, ValueError, TypeError, KeyError) as error:
                    skipped.append({"state_id": state, "reason": type(error).__name__ + ": " + str(error)[:160]})
                    continue
                if not changes:
                    continue
                performances = [entry for entry in bank_for(codec.contract.renderer_id)
                                if entry.role == "reference" and eligible(codec, target, entry.performance())]
                if not performances:
                    skipped.append({"state_id": state, "reason": "no_attack_compatible_reference"})
                    continue
                entry = max(performances, key=lambda value: longest_held_seconds(value.performance()))
                original = json.loads(raw)
                original["settings"].update(FULL_PHASE_CONTROL)
                prepared["settings"].update(FULL_PHASE_CONTROL)
                candidates += 1
                pair = []
                for role, document in (("original", original), ("prepared", prepared), ("repeat", prepared)):
                    request = RenderRequest(preset_json=canonical_json(document), performance=entry.performance(),
                                            tail_seconds=4., renderer_id=codec.contract.renderer_id)
                    _, metadata = worker.render(request, timeout=60.)
                    event = {"state_id": state, "role": role, "request_id": request.request_id, **metadata}
                    events.append(event)
                    pair.append(event)
                    write_json(args.output / "events.json", events)
                    if role == "original" and metadata["status"] != "accepted":
                        break
                valid = len(pair) == 3 and all(event["status"] == "accepted" for event in pair)
                exact = valid and len({event["qa"]["audio_float32_sha256"] for event in pair}) == 1
                accepted += valid
                row = {"state_id": state, "performance_id": entry.performance_id, "label_changes": changes,
                       "statuses": [event["status"] for event in pair], "accepted_pair": valid, "source_prepared_repeat_exact": exact,
                       "source_failure": {key: pair[0][key] for key in ("error_type", "message") if key in pair[0]}}
                checks.append(row)
                write_json(args.output / "checks.json", checks)
                print(json.dumps(row), flush=True)
    finally:
        monitor.close()
    positive_checks = [row for row in checks if row["accepted_pair"]]
    gates = {"bounded_positive_panel_complete": accepted == plan["positive_pair_limit"],
             "accepted_originals_keep_prepared_and_repeat": all(row["accepted_pair"] for row in checks if row["statuses"][0] == "accepted"),
             "all_positive_pairs_exact": all(row["source_prepared_repeat_exact"] for row in positive_checks),
             "attempts_bounded": len(events) <= plan["maximum_native_attempts"], "source_unchanged": sources() == plan["source_sha256_lf"],
             "memory_bounded": monitor.errors == 0 and monitor.peak_bytes is not None and monitor.peak_bytes <= plan["rss_limit_bytes"]}
    summary = {"plan_sha256": canonical_sha256(plan), "label_copy_panel_passed": all(gates.values()), "gates": gates,
               "source_candidates": candidates, "accepted_pairs": accepted, "attempts": len(events), "checks": checks, "skipped": skipped,
               "unadmitted_native_sources": [row["state_id"] for row in checks if not row["accepted_pair"]],
               "peak_process_rss_bytes": monitor.peak_bytes, "elapsed_seconds": perf_counter() - started,
               "claim": "bounded_label_copy_equivalence_not_complete_corpus_native_admission", "final_test_audio": False,
               "fits_assets_or_decoders": False, "model_promotion": False}
    write_json(args.output / "summary.json", summary)
    print(json.dumps({key: summary[key] for key in ("label_copy_panel_passed", "accepted_pairs", "attempts", "unadmitted_native_sources")}), flush=True)
    return 0 if summary["label_copy_panel_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
