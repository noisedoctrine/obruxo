"""Freeze terminal service metadata and independently replay a completed direct check."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

REPOSITORY = Path(__file__).resolve().parents[4]
sys.path[:0] = [str(REPOSITORY / "research/training"), str(REPOSITORY / "research/data_generation")]


def main():
    from obruxo_training.audit import write_json
    from obruxo_training.category_scale_refresh import cpu_settings
    from obruxo_training.direct_basic_audit import verify
    from obruxo_training.native_fit import digest
    from obruxo_training.trainable_basic_fit import source_code
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "fit", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads((args.run / "plan.json").read_bytes())
    first = json.loads((args.run / "verification.json").read_bytes())
    differences = [name for name, sha in first["evidence_sha256"].items() if digest(args.run / name) != sha]
    operational = json.loads((args.run / "automatic-verification.json").read_bytes())
    pending = {key: operational[key] for key in ("supervisor_pid", "supervisor_created_unix", "observed_pid", "observed_created_unix", "started_unix", "native_requests_allowed")}
    pending["complete"] = False
    pending_sha = hashlib.sha256(json.dumps(pending, indent=2).encode()).hexdigest()
    current, compatible = source_code(plan)
    if (differences != ["automatic-verification.json"] or pending_sha != first["evidence_sha256"]["automatic-verification.json"]
            or not operational["complete"] or not operational["verified"] or operational["exit_code"] != 0
            or operational["verification_sha256"] != digest(args.run / "verification.json")
            or compatible or current != plan["source"]):
        raise ValueError("scientific data, frozen code or the exact terminal operational transition changed")
    args.output.mkdir(parents=True, exist_ok=False)
    for path in args.run.rglob("*"):
        if not path.is_file() or path.suffix not in (".pt", ".json", ".jsonl") or path.name == "verification.json":
            continue
        target = args.output / path.relative_to(args.run)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    shutil.copyfile(args.run / "verification.json", args.output / "first-verification.json")
    write_json(args.output / "publication-plan.json", {"original_verification_sha256": digest(args.run / "verification.json"),
               "unchanged_scientific_evidence_files": len(first["evidence_sha256"]) - 1, "only_changed_record": "automatic-verification.json",
               "exact_original_pending_record_reconstructed_sha256": pending_sha,
               "completed_operational_record_sha256": digest(args.run / "automatic-verification.json"),
               "operational_transition": "pending_to_completed_after_verifier_return; original_verification_unchanged",
               "frozen_source_files_unchanged": len(plan["source"]), "publication_script_sha256": digest(Path(__file__)),
               "new_native_requests": 0, "optimizer_applications": 0, "purpose": "independent_replay_on_terminal_static_publication_snapshot"})
    with cpu_settings():
        report = verify(plan, args.fit, args.output)
    if any(digest(args.output / name) != sha for name, sha in report["evidence_sha256"].items()):
        raise ValueError("terminal direct publication snapshot changed during verification")
    print(json.dumps({key: value for key, value in report.items() if key not in ("evidence_sha256", "result")}), flush=True)


if __name__ == "__main__":
    main()
