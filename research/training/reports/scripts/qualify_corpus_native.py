"""Qualify exact corpus readback with frozen training sources and fixed-phase native repeats."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

RESEARCH = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(RESEARCH / "training"), str(RESEARCH / "data_generation")]
STATES = {"scalar": "00084aa654184d7508e6c3a5cb74791bcfbf9995cf653f1a877e6c7c9a5e9fc0",
          "empty_map": "4919dc18bb9c247b7923bed4864ac78e8aacfc5e2cb502ef8033292f03df39f9"}


def main():
    from obruxo_data.hashing import canonical_json, canonical_sha256
    from obruxo_data.midi import Performance
    from obruxo_data.render.base import FULL_CORPUS_RENDERER_ID, FULL_CURVE_RENDERER_ID, RenderRequest
    from obruxo_data.render.corpus_inputs import native_scalar_grid, prepare_native_corpus
    from obruxo_data.vital.full_contract import FullPresetContract
    from obruxo_training.audit import load_snapshot, write_json
    from obruxo_training.benchmark import MemoryMonitor
    from obruxo_training.corpus_codec import CorpusNativeCodec
    from obruxo_training.corpus_inputs import prepare_corpus_labels
    from obruxo_training.identity import PresetIdentity
    from obruxo_training.performance_eligibility import eligible, longest_held_seconds
    from obruxo_training.performances import bank_for
    from obruxo_training.reproducibility import FULL_PHASE_CONTROL
    from obruxo_training.stream import NativeRendererFactory, RenderWorker
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("snapshot", "corpus", "asset-bank", "lfo-bank", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    codec = CorpusNativeCodec(json.loads(args.asset_bank.read_text()), json.loads(args.lfo_bank.read_text()))
    source_schema = FullPresetContract(renderer_id=FULL_CURVE_RENDERER_ID)
    source_contract, catalog, manifest = load_snapshot(args.snapshot, source_schema)
    if source_schema.spec != {key: value for key, value in codec.schema.spec.items() if key != "corpus_native_policy"}:
        raise ValueError("source and target schemas differ beyond the declared native policy")
    if catalog.catalog_id != codec.catalog.catalog_id:
        raise ValueError("source and target asset catalogs differ")
    root = args.corpus.resolve(strict=True)
    if not root.is_dir() or args.output.resolve().is_relative_to(root):
        raise ValueError("qualification output must be outside the source corpus")
    records = json.loads((args.snapshot / "members.json").read_text())["records"]
    members = {row["identity"]["state"]: row for row in records if row["static_eligible"] and row["split"] == "train"}
    cases, groups, source_files = [], {}, []

    def add(group, role, document, performance, renderer=FULL_CORPUS_RENDERER_ID):
        name = group + ":" + role
        cases.append((name, document, performance, renderer))
        groups.setdefault(group, []).append(name)

    for group, state in STATES.items():
        member = members[state]
        source = root / member["relative_path"]
        path = source.resolve(strict=True)
        if source.is_symlink() or not path.is_relative_to(root) or manifest.by_state[state].split != "train":
            raise ValueError("qualification source escapes frozen training membership")
        raw = path.read_bytes()
        identity = PresetIdentity.from_bytes(raw)
        if identity.state != state or identity.raw != member["identity"]["raw"]:
            raise ValueError("frozen training source changed")
        source_files.append((path, identity.raw))
        target = codec.encode(raw)
        entries = [entry for entry in bank_for(codec.contract.renderer_id)
                   if entry.role == "reference" and eligible(codec, target, entry.performance())]
        entry = max(entries, key=lambda value: longest_held_seconds(value.performance()))
        document = json.loads(raw)
        document["settings"].update(FULL_PHASE_CONTROL)
        labels, _ = prepare_corpus_labels(document, codec.base, codec.schema)
        reference, _ = prepare_native_corpus(document)
        # Change only the diagnosed field in this reference, never unrelated controls.
        if group == "scalar":
            name = "lfo_1_smooth_time"
            reference["settings"][name] = native_scalar_grid(name, reference["settings"][name], codec.schema)
        add(group, "legacy_negative", document, entry.performance(), FULL_CURVE_RENDERER_ID)
        add(group, "original", document, entry.performance())
        add(group, "repeat", document, entry.performance())
        add(group, "labels", labels, entry.performance())
        add(group, "legacy_reference", reference, entry.performance(), FULL_CURVE_RENDERER_ID)

    performance = Performance(bpm=120)
    performance.add_note(pitch=60, velocity=90, start_tick=240, duration_ticks=1440)
    performance.end_tick = 2400
    base = codec.schema.template()
    base["settings"].update(FULL_PHASE_CONTROL, volume=2500., osc_1_destination=4., osc_1_level=.15)
    for mode in (13, 14, 15):
        for layout in ("singleton", "decimal"):
            document = deepcopy(base)
            shape = ({"name": "one point", "num_points": 1, "points": [.37, .4], "powers": [.37], "smooth": True}
                     if layout == "singleton" else {"name": "decimal", "num_points": 3, "points": [0., .1, .37, .8, 1., .2],
                                                    "powers": [.37, -.17, .13], "smooth": False})
            settings = document["settings"]
            settings["lfos"][0] = deepcopy(shape)
            settings["custom_warps"][0] = deepcopy(shape)
            settings["modulations"][0] = {"source": "lfo_1", "destination": "osc_1_pan", "line_mapping": deepcopy(shape)}
            settings.update(osc_1_spectral_morph_type=float(mode), osc_1_spectral_morph_amount=.5,
                            modulation_1_amount=.05, modulation_1_bypass=0.)
            group = f"authored_{mode}_{layout}"
            add(group, "legacy", document, performance, FULL_CURVE_RENDERER_ID)
            add(group, "original", document, performance)
            add(group, "repeat", document, performance)

    files = [Path(__file__), *(RESEARCH / "training/obruxo_training").glob("*.py"),
             *(RESEARCH / "data_generation/obruxo_data/render").glob("*.py"),
             RESEARCH / "data_generation/obruxo_data/vital/full_contract.py", RESEARCH / "data_generation/obruxo_data/vital/atlas.py"]

    def sources():
        return {path.relative_to(RESEARCH).as_posix(): hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest() for path in files}

    plan = {"qualification": "original_corpus_exact_native_readback_v1", "source_sha256_lf": sources(),
            "source_contract_id": source_contract.contract_id, "target_contract_id": codec.contract.contract_id,
            "codec_id": codec.codec_id, "split_manifest_id": manifest.manifest_id, "training_states": STATES,
            "case_request_sha256": {name: canonical_sha256(doc) for name, doc, _, _ in cases}, "attempt_budget": len(cases),
            "phase_control": FULL_PHASE_CONTROL, "worker_jobs": 1, "rss_limit_bytes": 8 * 1024**3,
            "actual_source_scalar_changes": False, "legacy_reference_scalars": "only_diagnosed_lfo_1_smooth_time_projected_in_scalar_reference",
            "final_test_audio": False, "fits_assets_or_decoders": False, "model_promotion": False}
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "plan.json", plan)
    rows, monitor, started = {}, MemoryMonitor(), perf_counter()
    monitor.start()
    try:
        for name, document, performance, renderer in cases:
            request = RenderRequest(preset_json=canonical_json(document), performance=performance, tail_seconds=4., renderer_id=renderer)
            with RenderWorker(renderer_factory=NativeRendererFactory(renderer_id=renderer), max_jobs=1) as worker:
                _, metadata = worker.render(request, timeout=60.)
            rows[name] = {"request_id": request.request_id, **metadata}
            write_json(args.output / "events.json", rows)
            print(json.dumps({"case": name, "status": metadata["status"], "message": metadata.get("message")}), flush=True)
    finally:
        monitor.close()
    negative = {group: rows[group + ":legacy_negative"] for group in STATES}
    positive = {name: row for name, row in rows.items() if not name.endswith(":legacy_negative")}
    exact = {group: len({rows[name].get("qa", {}).get("audio_float32_sha256") for name in names
                        if not name.endswith(":legacy_negative")}) == 1
             and all(rows[name]["status"] == "accepted" for name in names if not name.endswith(":legacy_negative"))
             for group, names in groups.items()}
    native = {name: row["renderer_qa"]["native_state"] for name, row in positive.items()
              if row["status"] == "accepted" and "corpus_native_policy" in row["renderer_qa"]["native_state"]}
    gates = {"legacy_negative_controls_retained": negative["scalar"].get("message", "").startswith(
                 "Vital did not load the requested scalar controls: lfo_1_smooth_time")
                 and negative["empty_map"].get("message") == "Vital did not load the requested modulations",
             "all_positive_requests_accepted": all(row["status"] == "accepted" for row in positive.values()),
             "every_profile_reference_and_fresh_repeat_exact": all(exact.values()),
             "scalar_roundtrip_ledger_present": bool(native.get("scalar:original", {}).get("scalar_native_roundtrips")),
             "empty_mapping_ledger_present": bool(native.get("empty_map:original", {}).get("corpus_changes")),
             "full_readback_retained": bool(native) and all(value["verified_loaded_scalars"] == 775
                 and value["scalar_lfo_route_readback_verified"] and len(value["nested_families_verified"]) == 6 for value in native.values()),
             "attempt_budget_respected": len(rows) == plan["attempt_budget"], "source_implementation_unchanged": sources() == plan["source_sha256_lf"],
             "corpus_bytes_unchanged": all(hashlib.sha256(path.read_bytes()).hexdigest() == digest for path, digest in source_files),
             "memory_bounded": monitor.errors == 0 and monitor.peak_bytes is not None and monitor.peak_bytes <= plan["rss_limit_bytes"]}
    summary = {"plan_sha256": canonical_sha256(plan), "passed": all(gates.values()), "gates": gates,
               "attempts": len(rows), "accepted": sum(row["status"] == "accepted" for row in rows.values()),
               "exact_groups": exact, "negative_controls": negative, "native_state_checks": native,
               "peak_process_rss_bytes": monitor.peak_bytes, "elapsed_seconds": perf_counter() - started,
               "claim": "bounded_native_contract_qualification_not_complete_corpus_admission_or_model_quality",
               "final_test_audio": False, "model_promotion": False}
    write_json(args.output / "summary.json", summary)
    print(json.dumps({key: summary[key] for key in ("passed", "gates", "attempts", "accepted")}), flush=True)
    return 0 if summary["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
