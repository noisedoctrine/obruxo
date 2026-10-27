"""Replay the saved fit and measure useful native gradients blocked by output clipping."""

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
    from obruxo_training import basic_direct_fit, basic_direct_fit_step
    from obruxo_training.audit import write_json
    from obruxo_training.basic_direct_fit_data import Inputs
    from obruxo_training.benchmark import MemoryMonitor
    from obruxo_training.category_scale_refresh import cpu_settings
    from obruxo_training.codec import CONTINUOUS
    from obruxo_training.conditioned_bridge import read
    from obruxo_training.model import batch_features
    from obruxo_training.native_fit import digest
    from obruxo_training.paired_native_audit import equal_tree
    from obruxo_training.trainer import capture_rng
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("fit", "parent", "bridge", "features", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    plan, verified = read(args.fit / "plan.json"), read(args.fit / "verification.json")
    if (not verified["verified"] or not verified["offline_training_replay_exact"] or verified["plan_id"] != canonical_sha256(plan)
            or digest(args.features) != plan["input_cache_sha256"] or digest(args.fit / "checkpoint.pt") != verified["evidence_sha256"]["checkpoint.pt"]):
        raise ValueError("shared learning diagnosis requires the unchanged verified fit")
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "plan.json", {"fit_plan_id": verified["plan_id"], "fit_verification_sha256": digest(args.fit / "verification.json"),
               "checkpoint_sha256": digest(args.fit / "checkpoint.pt"), "diagnostic_script_sha256": digest(Path(__file__)),
               "limits": {"wall_seconds": 600, "process_tree_bytes": 4 * 1024**3, "archive_bytes": 16 * 1024**2},
               "new_native_requests": 0, "experimental_optimizer_updates": 0, "software_replay_updates_per_arm": 48})
    started, monitor = perf_counter(), MemoryMonitor()
    monitor.start()
    try:
        with cpu_settings():
            inputs = Inputs(args.parent, args.bridge, args.features, plan)
            models, optimizers = basic_direct_fit.initialize(inputs)
            store = basic_direct_fit.SoundStore(args.fit, inputs)
            saved = torch.load(args.fit / "checkpoint.pt", map_location="cpu", weights_only=True)
            original_predict, observed, rows = basic_direct_fit_step.predict, [], []

            def predict(model, features):
                if model is not models["spectral"]:
                    return original_predict(model, features)
                deltas = []
                hook = model.correction.register_forward_hook(lambda _module, _args, output: deltas.append(output))
                try:
                    result, bounds = original_predict(model, features)
                finally:
                    hook.remove()
                deltas[0].retain_grad()
                result["continuous"].retain_grad()
                with torch.no_grad():
                    raw = model.base(batch_features([{"features": features}]))["continuous"] + deltas[0][:, :9]
                observed.append((raw.detach(), deltas[0], result["continuous"]))
                return result, bounds

            with patch.object(basic_direct_fit_step, "predict", predict):
                for index, expected in enumerate(saved["trace"]):
                    observed.clear()
                    result = basic_direct_fit.step(models, optimizers, inputs, store, index)
                    actual = {"step": index + 1, "update": plan["updates"][index], **result}
                    if actual != {key: value for key, value in expected.items() if key != "attempt"} or len(observed) != 4:
                        raise ValueError("diagnostic hooks changed the recorded training trajectory")
                    for observation, ((raw, delta, output), estimate, cells) in enumerate(zip(observed, result["estimates"], inputs.observations(plan["updates"][index]), strict=True)):
                        from obruxo_training.codec import PilotTarget
                        active = inputs.codec.continuous_mask(PilotTarget.from_dict(estimate["anchor"]))
                        cell = plan["fit_audit_cells"][cells[0]]
                        for control, enabled in enumerate(active):
                            if not enabled:
                                continue
                            value, native = float(raw[0, control]), estimate["continuous"][control]
                            blocked = value < 0. or value > 1.
                            inward = value < 0. and native < 0. or value > 1. and native > 0.
                            if blocked and float(delta.grad[0, control]) != 0.:
                                raise ValueError("a strictly clipped output unexpectedly passed its derivative")
                            rows.append({"step": index + 1, "observation": observation, "training_index": cell["training_index"],
                                         "input_view": cell["input_view"], "control": CONTINUOUS[control][0], "raw_proposal": value,
                                         "native_gradient": native, "received_output_gradient": float(output.grad[0, control]),
                                         "correction_output_gradient": float(delta.grad[0, control]), "blocked": blocked, "inward": inward})
            if (not all(equal_tree(model.state_dict(), saved["models"][arm]) and equal_tree(optimizers[arm].state_dict(), saved["optimizers"][arm]) for arm, model in models.items())
                    or not equal_tree(capture_rng("cpu"), saved["rng"])):
                raise ValueError("shared diagnosis changed final models, Adam or RNG")
    finally:
        monitor.close()
    elapsed = perf_counter() - started
    if elapsed > 600 or not monitor.peak_bytes or monitor.peak_bytes > 4 * 1024**3 or monitor.errors:
        raise ValueError("shared learning diagnosis exceeded its fixed resource allocation")
    summary = {"active_control_observations": len(rows), "strictly_clipped": sum(row["blocked"] for row in rows),
               "clipped_with_inward_native_gradient": sum(row["inward"] for row in rows), "replayed_updates_per_arm": len(saved["trace"]),
               "model_Adam_RNG_replay_exact": True, "new_native_requests": 0, "experimental_optimizer_updates": 0,
               "elapsed_seconds": elapsed, "peak_process_tree_rss_bytes": monitor.peak_bytes, "rss_sampling_errors": monitor.errors,
               "by_input_view": {str(view): {"active": sum(row["input_view"] == view for row in rows),
                    "blocked_inward": sum(row["input_view"] == view and row["inward"] for row in rows)} for view in (0, 1, 2)},
               "qualified_for_release": False}
    write_json(args.output / "measurements.json", rows)
    write_json(args.output / "summary.json", summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
