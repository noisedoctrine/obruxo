"""Reconstruct saved category derivatives and gates before publishing the diagnosis."""

import argparse
import hashlib
import inspect
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

RESEARCH = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(RESEARCH / "training"), str(RESEARCH / "data_generation")]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(source, parent, parent_audit):
    import torch
    from obruxo_data.hashing import canonical_sha256
    from obruxo_training.category_signal import conditional_score_gradient, fit_scales
    from obruxo_training.spectral_controls import temperature_at
    from check_category_signal import direction_summary
    torch.set_num_threads(2)
    result = json.loads((source / "summary.json").read_text())
    plan = json.loads((source / "plan.json").read_text())
    if result["plan"] != plan or result["plan_id"] != canonical_sha256(plan) or result["failure"] is not None:
        raise ValueError("incomplete or changed category diagnosis")
    if digest(parent_audit) != plan["parent_audit_sha256"]:
        raise ValueError("parent audit changed")
    for name, expected in plan["parent_evidence_sha256"].items():
        if digest(parent / name) != expected:
            raise ValueError("parent evidence changed")
    for name, expected in plan["source"].items():
        path = RESEARCH / "training/obruxo_training" / name if name == "category_signal.py" else Path(__file__).with_name(name)
        if hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest() != expected:
            raise ValueError("frozen diagnostic source changed")
    if hashlib.sha256(inspect.getsource(torch.nn.functional.gumbel_softmax).encode()).hexdigest() != plan["gumbel_implementation_sha256"]:
        raise ValueError("Gumbel implementation changed")
    initial = result["initial_training_inputs"]
    scales = fit_scales([torch.tensor([row["raw_logits"][head] for row in initial]) for head in (0, 1)])
    if len(initial) != 32 or list(scales) != result["fixed_scales"] or not all(row["hard_patch_unchanged"] for row in initial):
        raise ValueError("initial calibration inputs or fixed scales changed")
    trace = torch.load(parent / "checkpoint.pt", map_location="cpu", weights_only=True)["trace"]
    if len(result["points"]) != len(trace) or len(trace) != 128:
        raise ValueError("diagnostic point denominator changed")
    for step, (point, committed) in enumerate(zip(result["points"], trace, strict=True)):
        if point["step"] != step + 1 or len(point["heads"]) != 2:
            raise ValueError("diagnostic point order changed")
        for head, row in enumerate(point["heads"]):
            estimate = committed["estimates"][0]
            if (row["head"] != head or row["conditional_losses"] != estimate["categorical"][head]
                    or row["selected"] != estimate["anchor"]["categorical"][head]
                    or row["original_gradient_l1"] != committed["gradients"]["category_l1"][head]):
                raise ValueError("saved derivative point differs from actual native training")
            raw, noise = torch.tensor([row["raw_logits"]]), torch.tensor([row["noise"]])
            for scale, field in ((1., "original_gradient_l1"), (scales[head], "calibrated_gradient_l1")):
                gradient = conditional_score_gradient(raw, noise, row["conditional_losses"], scale=scale, temperature=temperature_at(step, 128))
                if not math.isclose(float(gradient.abs().sum()), row[field], rel_tol=1e-5, abs_tol=1e-30):
                    raise ValueError("saved score derivative cannot be reconstructed")
            if (int((raw + noise).argmax(-1).item()) != row["selected"]
                    or row["better_alternative"] != (min(row["conditional_losses"]) < row["conditional_losses"][row["selected"]] - 1e-7)
                    or row["raw_top_two_margin"] != float(raw.topk(2).values.diff(dim=-1).abs().item())):
                raise ValueError("saved choice/margin/native-alternative diagnosis changed")
    heads = direction_summary(SimpleNamespace(points=result["points"], scales=scales))
    archive = sum(path.stat().st_size for path in source.rglob("*") if path.is_file())
    limits = plan["limits"]
    resources = (0 < result["peak_process_tree_rss_bytes"] <= limits["process_tree_bytes"] and result["rss_sampling_errors"] == 0
                 and result["elapsed_seconds"] <= limits["wall_seconds"] and archive <= limits["archive_bytes"])
    passed = result["historical_replay_unchanged"] and resources and all(head["pass"] for head in heads)
    if heads != result["heads"] or resources != result["within_resource_limits"] or passed != result["passed"]:
        raise ValueError("saved diagnosis gates differ from reconstructed results")
    return {**result, "all_saved_derivatives_reconstructed": True, "actual_training_points_verified": True,
            "retained_archive_bytes": archive,
            "evidence_sha256": {name: digest(source / name) for name in ("plan.json", "run.json", "summary.json")}}


def main():
    from obruxo_training.audit import write_json
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "parent", "parent-audit", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("refusing to replace published evidence")
    result = summarize(args.source, args.parent, args.parent_audit)
    write_json(args.output, result)
    print(json.dumps({"passed": result["passed"], "all_saved_derivatives_reconstructed": True,
                      "elapsed_seconds": result["elapsed_seconds"], "peak_process_tree_rss_bytes": result["peak_process_tree_rss_bytes"]}))


if __name__ == "__main__":
    main()
