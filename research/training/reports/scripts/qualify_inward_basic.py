"""Qualify inward-only gradients without changing any existing basic prediction."""

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter
from unittest.mock import patch

REPOSITORY = Path(__file__).resolve().parents[4]
sys.path[:0] = [str(REPOSITORY / "research/training"), str(REPOSITORY / "research/data_generation")]


def main():
    import torch
    from obruxo_data.hashing import canonical_sha256
    from obruxo_training import basic_direct_bridge
    from obruxo_training.audit import write_json
    from obruxo_training.basic_direct import restore_direct
    from obruxo_training.benchmark import MemoryMonitor
    from obruxo_training.category_scale_refresh import cpu_settings
    from obruxo_training.checkpoints import atomic_torch_save
    from obruxo_training.conditioned_bridge import read, training_bank
    from obruxo_training.features import tensor_digest
    from obruxo_training.inward_basic import InwardClamp, POLICY, export_inward, training_model
    from obruxo_training.native_fit import digest
    from obruxo_training.pilot import PilotPool
    from obruxo_training.trainable_basic import COUNTS, input_gradients
    from obruxo_training.trainable_basic_fit import source_code
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("initial", "features", "pool", "direct", "diagnosis", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    direct, proof = read(args.direct / "plan.json"), read(args.direct / "verification.json")
    diagnosis, diagnosis_plan = read(args.diagnosis / "summary.json"), read(args.diagnosis / "plan.json")
    current, differences = source_code(direct)
    if (not proof["verified"] or not proof["direct_quality_passed"] or proof["plan_id"] != canonical_sha256(direct)
            or any(digest(args.direct / name) != sha for name, sha in proof["evidence_sha256"].items())
            or differences or current != direct["source"] or digest(args.initial) != direct["inputs"]["bridge/model.pt"]
            or digest(args.features) != direct["inputs"]["features"] or not diagnosis["model_Adam_RNG_replay_exact"]
            or diagnosis["new_native_requests"] or diagnosis["experimental_optimizer_updates"]
            or diagnosis_plan["fit_plan_id"] != direct["fit_plan_id"]):
        raise ValueError("inward qualification requires unchanged verified training and direct-fit evidence")
    args.output.mkdir(parents=True, exist_ok=False)
    sources = ("obruxo_training/inward_basic.py", "reports/scripts/qualify_inward_basic.py", "reports/scripts/diagnose_shared_basic_learning.py")
    plan = {"format": "inward_basic_software_qualification_v1", "gradient_policy": POLICY,
            "source": {name: digest(REPOSITORY / "research/training" / name) for name in sources},
            "initializer_sha256": digest(args.initial), "feature_cache_sha256": digest(args.features),
            "direct_verification_sha256": digest(args.direct / "verification.json"),
            "diagnosis_sha256": {name: digest(args.diagnosis / name) for name in ("plan.json", "summary.json", "measurements.json")},
            "limits": {"wall_seconds": 600, "process_tree_bytes": 4 * 1024**3, "archive_bytes": 64 * 1024**2},
            "new_native_requests": 0, "experimental_optimizer_updates": 0, "model_budget": COUNTS}
    write_json(args.output / "plan.json", plan)
    started, monitor = perf_counter(), MemoryMonitor()
    monitor.start()
    try:
        with cpu_settings():
            initial, frontend, codec = restore_direct(torch.load(args.initial, map_location="cpu", weights_only=True))
            pool = PilotPool.load(args.pool)
            if pool.pool_id != direct["pool_id"] or codec.to_dict() != pool.codec.to_dict():
                raise ValueError("inward qualification changed its training pool or codec")
            bank = training_bank(torch.load(args.features, map_location="cpu", weights_only=True), pool)
            model = training_model(initial)
            atomic_torch_save(args.output / "model.pt", export_inward(model, frontend, codec, pool.pool_id))
            cold, _front, _codec = restore_direct(torch.load(args.output / "model.pt", map_location="cpu", weights_only=True))
            with torch.no_grad():
                preserved = basic_direct_bridge.preserve(initial.base, model, cold, codec, bank, initial=True)
            measurements = read(args.diagnosis / "measurements.json")
            raw = torch.tensor([row["raw_proposal"] for row in measurements], requires_grad=True)
            incoming = torch.tensor([row["received_output_gradient"] for row in measurements])
            value = InwardClamp.apply(raw)
            if not torch.equal(value.detach(), raw.detach().clamp(0., 1.)):
                raise ValueError("inward gradients changed a saved native forward value")
            value.backward(incoming)
            repaired = 0
            for index, row in enumerate(measurements):
                expected = row["received_output_gradient"] if not row["blocked"] or row["inward"] else 0.
                if float(raw.grad[index]) != expected:
                    raise ValueError("inward policy lost an allowed derivative or passed an outward derivative")
                repaired += bool(row["inward"] and row["correction_output_gradient"] == 0. and raw.grad[index] != 0.)
            if repaired != diagnosis["clipped_with_inward_native_gradient"]:
                raise ValueError("inward policy did not recover every measured blocked inward direction")

            def restore_training(saved):
                restored, front, current_codec = restore_direct(saved)
                return training_model(restored), front, current_codec

            features = [row["features"] for row in list(bank.values())[:4]]
            with patch.object(basic_direct_bridge, "export_direct", export_inward), patch.object(basic_direct_bridge, "restore_direct", restore_training):
                trial, resumed, recovery = basic_direct_bridge.simulate(model, frontend, codec, pool.pool_id, features, args.output)
            access = input_gradients(trial)
            if not all(value > 0 for value in access.values()):
                raise ValueError("sound derivatives do not reach every copied input branch")
            with torch.no_grad():
                simulated = basic_direct_bridge.preserve(initial.base, trial, resumed, codec, bank, initial=False)
            if tensor_digest(model.state_dict()) != tensor_digest(initial.state_dict()) or digest(args.initial) != plan["initializer_sha256"]:
                raise ValueError("qualification changed the original initializer")
    finally:
        monitor.close()
    resources = {"elapsed_seconds": perf_counter() - started, "peak_process_tree_rss_bytes": monitor.peak_bytes, "rss_sampling_errors": monitor.errors}
    archive = sum(path.stat().st_size for path in args.output.rglob("*") if path.is_file())
    if resources["elapsed_seconds"] > 600 or not monitor.peak_bytes or monitor.peak_bytes > 4 * 1024**3 or monitor.errors or archive + 1024**2 > 64 * 1024**2:
        raise ValueError("inward qualification exceeded its fixed allocation")
    summary = {"plan_id": canonical_sha256(plan), "qualified_gradient_policy": True, "model_budget": model.require_frozen(),
               "initialization_cases": len(preserved), "simulated_cold_cases": len(simulated), "saved_active_native_components": len(measurements),
               "restored_inward_native_directions": repaired, "software_recovery": recovery, "later_input_gradient_l1": access,
               "model_file_sha256": digest(args.output / "model.pt"), "model_sha256": tensor_digest(model.state_dict()),
               "resources": resources, "archive_bytes_before_summary": archive, "new_native_requests": 0,
               "experimental_optimizer_updates": 0, "qualified_for_release": False}
    write_json(args.output / "summary.json", summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
