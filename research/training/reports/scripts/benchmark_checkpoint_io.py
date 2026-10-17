"""Compare persistence of identical growing replay prefixes with fixed initial state."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

RESEARCH = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(RESEARCH / "training"), str(RESEARCH / "data_generation")]


def main():
    import torch
    from obruxo_data.hashing import canonical_sha256
    from obruxo_training.audit import write_json
    from obruxo_training.benchmark import MemoryMonitor
    from obruxo_training.checkpoints import TrainingCheckpointWriter, atomic_torch_save, load_training_checkpoint
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    value = load_training_checkpoint(args.checkpoint)
    if value["format"] != "obruxo_pilot_training_v1" or len(value["replay"]) != 64:
        raise ValueError("benchmark requires a complete 64-example training checkpoint")
    source_step = value["step"]
    # This is an I/O fixture using fixed existing weights, not a new training run.
    value.update(step=0, initial_metrics=None)
    value["optimizer"] = {**value["optimizer"], "state": {}}
    value["totals"].update(optimizer_seconds=0., sampled_examples=0, reused_feature_draws=0, maximum_gradient_norm=0.)
    replay = [{**item, "uses": 0} for item in value["replay"]]
    args.output.mkdir(parents=True, exist_ok=False)
    for name in ("full", "compact"):
        (args.output / name).mkdir()
    files = [Path(__file__), RESEARCH / "training/obruxo_training/checkpoints.py"]

    def sources():
        return {path.relative_to(RESEARCH).as_posix(): hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest() for path in files}

    plan = {"benchmark": "growing_replay_checkpoint_io_v1", "source_sha256_lf": sources(),
            "input_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(), "input_step": source_step,
            "pool_id": value["run_contract"]["pool_id"], "codec_id": canonical_sha256(value["run_contract"]["codec"]),
            "fixture": "fixed_existing_weights_empty_optimizer_no_learning", "prefixes": list(range(65)),
            "order": "alternate_full_first_and_compact_first_by_prefix", "minimum_time_reduction": .2,
            "minimum_serialized_byte_reduction": .5, "rss_limit_bytes": 8 * 1024**3,
            "audio_files_written": 0, "final_test_audio": False, "model_promotion": False}
    write_json(args.output / "plan.json", plan)
    writer = TrainingCheckpointWriter(args.output / "compact")
    rows, totals = [], {"full_seconds": 0., "compact_seconds": 0., "full_serialized_bytes": 0}
    monitor = MemoryMonitor()
    monitor.start()
    try:
        for count in plan["prefixes"]:
            next_sequence = replay[count - 1]["sequence_id"] + 1 if count else 0
            current = {**value, "replay": replay[:count], "cursor": {**value["cursor"], "next_sequence_id": next_sequence},
                       "render_ready": count > 0, "renderer_identity": value["renderer_identity"] if count else None,
                       "totals": {**value["totals"], "accepted": count, "render_attempts": next_sequence, "rejected": next_sequence - count}}
            row = {"replay_examples": count}
            for kind in (("full", "compact") if count % 2 == 0 else ("compact", "full")):
                tick = perf_counter()
                if kind == "full":
                    atomic_torch_save(args.output / "full/checkpoint.pt", current)
                    totals["full_serialized_bytes"] += (args.output / "full/checkpoint.pt").stat().st_size
                else:
                    writer.save(current)
                row[kind + "_seconds"] = perf_counter() - tick
                totals[kind + "_seconds"] += row[kind + "_seconds"]
            rows.append(row)
        full = load_training_checkpoint(args.output / "full/checkpoint.pt")
        compact = load_training_checkpoint(args.output / "compact/checkpoint.pt")
        # Standard tensor equality is exact; Python numeric labels are not requantized.
        def equal(left, right):
            if torch.is_tensor(left):
                return torch.is_tensor(right) and torch.equal(left, right)
            if isinstance(left, dict):
                return isinstance(right, dict) and left.keys() == right.keys() and all(equal(left[key], right[key]) for key in left)
            if isinstance(left, (tuple, list)):
                return type(left) is type(right) and len(left) == len(right) and all(equal(a, b) for a, b in zip(left, right, strict=True))
            return left == right
        identical = equal(full, compact)
    finally:
        monitor.close()
    totals["compact_serialized_bytes"] = writer.statistics["serialized_bytes"]
    time_reduction = 1 - totals["compact_seconds"] / totals["full_seconds"]
    byte_reduction = 1 - totals["compact_serialized_bytes"] / totals["full_serialized_bytes"]
    gates = {"resolved_states_identical": identical, "initial_state_written_once": writer.statistics["initial_writes"] == 1,
             "every_prefix_persisted": writer.statistics["collection_writes"] == 65,
             "material_time_reduction": time_reduction >= plan["minimum_time_reduction"],
             "material_byte_reduction": byte_reduction >= plan["minimum_serialized_byte_reduction"],
             "source_unchanged": sources() == plan["source_sha256_lf"],
             "memory_bounded": monitor.errors == 0 and monitor.peak_bytes is not None and monitor.peak_bytes <= plan["rss_limit_bytes"]}
    summary = {"plan_sha256": canonical_sha256(plan), "passed": all(gates.values()), "gates": gates, **totals,
               "time_reduction_fraction": time_reduction, "serialized_byte_reduction_fraction": byte_reduction,
               "compact_writes": deepcopy(writer.statistics), "prefix_measurements": rows, "peak_process_rss_bytes": monitor.peak_bytes,
               "source_step_is_not_a_training_comparison": True, "final_test_audio": False, "model_promotion": False}
    write_json(args.output / "summary.json", summary)
    print(json.dumps({key: value for key, value in summary.items() if key != "prefix_measurements"}), flush=True)
    return 0 if summary["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
