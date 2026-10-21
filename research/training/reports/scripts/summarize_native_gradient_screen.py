"""Verify native receipts and reconstruct every numerical update before publishing."""

import argparse
import json
from pathlib import Path
import sys

RESEARCH = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(RESEARCH / "training"), str(RESEARCH / "data_generation")]


def summarize(source):
    import torch
    from obruxo_data.hashing import canonical_sha256, sha256_bytes
    from obruxo_data.midi import Performance
    from obruxo_data.render.base import RenderRequest
    from obruxo_training.codec import PilotTarget, load_codec
    from obruxo_training.native_gradient import guided_candidates, measure_native_gradient
    from obruxo_training.native_gradient_screen import qualification_summary
    from obruxo_training.reproducibility import fixed_generator_phases
    torch.set_num_threads(2)
    plan = json.loads((source / "plan.json").read_text())
    summary = json.loads((source / "summary.json").read_text())
    cases = json.loads((source / "cases.json").read_text())
    events = [json.loads(line) for line in (source / "renders.jsonl").read_text().splitlines()]
    if (summary["plan"] != plan or summary["plan_id"] != canonical_sha256(plan) or summary["cases"] != cases
            or summary["failure"] is not None or not summary["result"]["complete"]):
        raise ValueError("incomplete or mismatched native gradient qualification")
    for name, expected in plan["source"].items():
        path = (RESEARCH / "data_generation/obruxo_data/render/vita.py" if name == "vita.py" else
                Path(__file__).with_name(name) if name == "run_native_gradient_screen.py" else
                RESEARCH / "training/obruxo_training" / name)
        if sha256_bytes(path.read_bytes().replace(b"\r\n", b"\n")) != expected:
            raise ValueError(f"frozen source changed: {name}")
    attempts = [row for row in events if row["event"] == "attempt"]
    outcomes = [row for row in events if row["event"] == "outcome"]
    rendered = [row for case in cases for row in case["renders"]]
    if len(attempts) != len(outcomes) or len(attempts) != len(rendered) or len(attempts) != summary["native_attempts"]:
        raise ValueError("native attempt/outcome accounting mismatch")
    for index, (attempt, outcome, row) in enumerate(zip(attempts, outcomes, rendered, strict=True)):
        if (attempt["attempt"] != index or outcome["attempt"] != index or outcome["status"] != "accepted"
                or attempt["request_id"] != outcome["request_id"] or row["request_id"] != outcome["request_id"]
                or row["audio_sha256"] != outcome["qa"]["audio_float32_sha256"]
                or outcome["renderer_identity"] != summary["renderer_identity"]):
            raise ValueError("native receipt identity or acceptance mismatch")
    codec, published, cursor = load_codec(plan["codec"]), [], 0
    for planned, case in zip(plan["cases"], cases, strict=True):
        if any(case[key] != planned[key] for key in ("index", "role", "state_id")):
            raise ValueError("qualification case changed")
        rows, raw = case["renders"], PilotTarget.from_dict(planned["start"])
        expected_source = PilotTarget.from_dict(planned["source"])
        for row in rows:
            candidate = PilotTarget.from_dict(row["candidate"])
            request = RenderRequest(preset_json=fixed_generator_phases(codec.decode(candidate)).decode(),
                                    performance=Performance.from_dict(planned["performance"]),
                                    renderer_id=codec.contract.renderer_id, tail_seconds=codec.contract.tail_seconds)
            receipt = outcomes[cursor]
            if (request.request_id != row["request_id"] or sha256_bytes(codec.decode(candidate)) != receipt["source_sha256"]
                    or sha256_bytes(request.preset_json.encode()) != receipt["controlled_sha256"]):
                raise ValueError("native receipt does not describe the recorded candidate")
            cursor += 1
        if (any(row["candidate"] != expected_source.to_dict() for row in rows[:2])
                or rows[0]["audio_sha256"] != rows[1]["audio_sha256"] or not case["repeat_exact"]):
            raise ValueError("source/repeat identity changed")
        measured = {PilotTarget.from_dict(row["candidate"]): row["scores"]["spectral"] for row in rows[2:-2]}
        estimate = measure_native_gradient(codec, raw, measured.__getitem__)
        candidates, gradients = guided_candidates(codec, estimate)
        if estimate.to_dict() != case["estimate"] or gradients != case["gradients"] or len(rows) != estimate.queries + 4:
            raise ValueError("recorded probe losses do not reproduce the gradient/update")
        if case["scores"]["start"] != rows[2]["scores"]:
            raise ValueError("starting score changed")
        for row, method in zip(rows[-2:], ("continuous", "joint"), strict=True):
            if row["candidate"] != candidates[method].to_dict() or row["scores"] != case["scores"][method]:
                raise ValueError("final native candidate or score changed")
        published.append({key: value for key, value in case.items() if key not in {"renders", "estimate"}} | {
            "native_queries_for_gradient": estimate.queries, "start": planned["start"],
            "final_candidates": {name: candidate.to_dict() for name, candidate in candidates.items()},
            "source_audio_sha256": rows[0]["audio_sha256"]})
    result = qualification_summary(cases, plan)
    archive = sum(path.stat().st_size for path in source.rglob("*") if path.is_file())
    limits = plan["limits"]
    resources = (summary["rss_sampling_errors"] == 0 and 0 < summary["peak_process_tree_rss_bytes"] <= limits["process_tree_bytes"]
                 and summary["elapsed_seconds"] <= limits["wall_seconds"] and archive <= limits["archive_bytes"]
                 and len(attempts) <= limits["qualification_attempts"])
    qualified = result["direction_gates_passed"] and resources
    if (result != summary["result"] or qualified != summary["qualified_native_gradient_path"]
            or not summary["all_native_accepted"] or resources != summary["within_resource_limits"]):
        raise ValueError("published gates differ from recomputed native evidence")
    digests = {name: sha256_bytes((source / name).read_bytes()) for name in ("plan.json", "run.json", "cases.json", "renders.jsonl", "summary.json")}
    return {key: value for key, value in summary.items() if key not in {"cases", "archive_bytes_before_summary"}} | {
        "cases": published, "retained_archive_bytes": archive, "evidence_sha256": digests,
        "all_numerical_updates_reconstructed": True, "all_requests_reconstructed": True,
        "audio_rescored": False, "verification": "native_receipts_and_saved_scalar_losses_not_rerendered_audio"}


def main():
    from obruxo_training.audit import write_json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.source)
    if args.output.exists():
        raise ValueError("refusing to replace published evidence")
    write_json(args.output, result)
    print(json.dumps({"qualified_native_gradient_path": result["qualified_native_gradient_path"],
                      "native_attempts": result["native_attempts"], "resources_ok": result["within_resource_limits"]}))


if __name__ == "__main__":
    main()
