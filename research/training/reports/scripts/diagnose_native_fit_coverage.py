"""Describe saved basic-fit coverage and errors, without new renders or training."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
from time import perf_counter

RESEARCH = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(RESEARCH / "training"), str(RESEARCH / "data_generation")]


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def main():
    from obruxo_training.audit import write_json
    from obruxo_training.fit_coverage import diagnose
    from obruxo_training.native_fit_report import public_cases
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "audit", "pool", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("refusing to overwrite diagnosis evidence")
    started = perf_counter()
    report = json.loads(args.audit.read_text())
    for name, expected in report["evidence_sha256"].items():
        if digest(args.source / name) != expected:
            raise ValueError(f"parent evidence changed: {name}")
    if json.loads((args.source / "plan.json").read_text()) != report["plan"]:
        raise ValueError("parent plan changed")
    for name in ("fit", "validation"):
        if public_cases(json.loads((args.source / f"{name}_cases.json").read_text())) != report[f"{name}_cases"]:
            raise ValueError("published cases differ from actual native receipts")
    result = diagnose(report, json.loads(args.pool.read_text()))
    result.update(input_sha256={"audited_fit": digest(args.audit), "pool": digest(args.pool), **report["evidence_sha256"]},
                  source_sha256={path.name: sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest() for path in
                    (Path(__file__), RESEARCH / "training/obruxo_training/fit_coverage.py", RESEARCH / "training/obruxo_training/native_fit_report.py")},
                  actual_native_cases_checked=len(report["fit_cases"]) + len(report["validation_cases"]), elapsed_seconds=perf_counter() - started)
    write_json(args.output, result)
    print(json.dumps({"fitted_states": result["coverage"]["fitted"]["states"],
                      "all_training_states": result["coverage"]["all_training"]["states"], "native_requests": 0, "optimizer_updates": 0}))


if __name__ == "__main__":
    main()
