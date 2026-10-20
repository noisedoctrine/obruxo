"""Re-score final checkpoints and publish the bounded matched response-fit evidence."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

RESEARCH = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(RESEARCH / "training"), str(RESEARCH / "data_generation")]

from run_event_response_fit import digest, load_inputs, make_plan, score, summarize  # noqa: E402


def publish(directory, parent, repaired):
    import torch
    from obruxo_data.hashing import canonical_sha256
    from obruxo_training.codec import load_codec
    from obruxo_training.features import tensor_digest
    from obruxo_training.spectral_controls import BasicControlAdapter
    from obruxo_training.temporal_proxy import TemporalSpectralProxy
    torch.set_num_threads(2)
    names = ("plan.json", "summary.json", "updates.jsonl", "uniform-scores.json", "events-scores.json")
    raw = {name: (directory / name).read_bytes() for name in names}
    plan, summary = [json.loads(raw[name]) for name in names[:2]]
    records, identities, codec, preflight = load_inputs(parent, repaired)
    if (plan != make_plan(parent, repaired, identities, codec, preflight) or canonical_sha256(plan) != summary["plan_id"]
            or not summary["complete"] or summary["failure"] is not None or not summary["within_resource_limits"]
            or summary["qualified_for_inverse_training"]):
        raise ValueError("response-fit completion, identity or resource evidence failed")
    charges = [json.loads(line) for line in raw["updates.jsonl"].splitlines()]
    if len(charges) != 2000:
        raise ValueError("unexpected optimizer-attempt count")
    cases, artifacts = {}, {}
    adapter = BasicControlAdapter(load_codec(codec))
    for arm in plan["arms"]:
        if [row["attempt"] for row in charges if row["arm"] == arm] != list(range(1, 1001)):
            raise ValueError("optimizer attempt sequence changed")
        path = directory / f"{arm}.pt"
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if (checkpoint["step"] != 1000 or checkpoint["contract"] != {"plan_id": summary["plan_id"], "arm": arm}
                or tensor_digest(checkpoint["model"]) != checkpoint["model_sha256"]):
            raise ValueError("final checkpoint or fitting contract changed")
        helper = TemporalSpectralProxy()
        helper.load_state_dict(checkpoint["model"])
        cases[arm] = score(helper.freeze(), records["events"], adapter, identities)
        if cases[arm] != summary["cases"][arm] or cases[arm] != json.loads(raw[f"{arm}-scores.json"]):
            raise ValueError("checkpoint re-score disagrees with saved result")
        artifacts[arm] = {"checkpoint_sha256": digest(path), "model_sha256": checkpoint["model_sha256"],
                          "committed_updates": checkpoint["step"], "training_trace": checkpoint["trace"]}
    result = summarize(cases, plan["gates"])
    if result != summary["result"]:
        raise ValueError("saved fitting decision disagrees with recomputed gates")
    archive_bytes = sum(path.stat().st_size for path in directory.rglob("*") if path.is_file())
    if archive_bytes > plan["limits"]["archive_bytes"]:
        raise ValueError("fitting archive exceeded its cap")
    return {**summary, "plan": plan, "artifacts": artifacts, "retained_archive_bytes": archive_bytes,
            "source_hashes": {name: hashlib.sha256(value).hexdigest() for name, value in raw.items()},
            "final_checkpoints_rescored": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--repaired", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = publish(args.source, args.parent, args.repaired)
    with args.output.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"fit_gate_passed": value["result"]["fit_gate_passed"], "resources_ok": value["within_resource_limits"],
                      "retained_archive_bytes": value["retained_archive_bytes"]}))


if __name__ == "__main__":
    main()
