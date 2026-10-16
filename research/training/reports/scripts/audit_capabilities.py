"""Audit frozen training recipes without rendering audio or learning weights."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

RESEARCH = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(RESEARCH / "training"), str(RESEARCH / "data_generation")]


def main():
    from obruxo_training.audit import write_json
    from obruxo_training.capability_audit import audit_capabilities
    from obruxo_training.pilot import PilotPool
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    files = [Path(__file__), *(RESEARCH / "training/obruxo_training").glob("*.py"),
             *(RESEARCH / "data_generation/obruxo_data" / name for name in ("render/base.py", "render/capabilities.py", "vital/full_contract.py"))]

    def sources():
        return {path.relative_to(RESEARCH).as_posix(): hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest() for path in sorted(files)}

    fingerprints = sources()
    started = perf_counter()
    pool = PilotPool.load(args.pool)
    result = audit_capabilities(pool)
    if sources() != fingerprints:
        raise ValueError("coverage runtime changed during the audit")
    result.update(pool_file_sha256=hashlib.sha256(args.pool.read_bytes()).hexdigest(), elapsed_seconds=perf_counter() - started,
                  runtime_source_sha256_lf=fingerprints, source_unchanged=True)
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "coverage.json", result)
    print(json.dumps({key: result[key] for key in ("pool_id", "audited_train_states", "scalar_count", "scalar_status_counts", "scalar_families", "elapsed_seconds")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
