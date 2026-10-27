"""Restore complete archived RNG before independently replaying the calibrated screen."""

import argparse
import hashlib
from pathlib import Path
import sys

REPOSITORY = Path(__file__).resolve().parents[4]
sys.path[:0] = [str(REPOSITORY / "research/training"), str(REPOSITORY / "research/data_generation")]


def main():
    import torch
    from obruxo_training.audit import write_json
    from obruxo_training.conditioned_bridge import read
    from obruxo_training.native_fit import digest
    from obruxo_training.trainer import restore_rng
    from run_calibrated_branch_screen import main as replay
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--mode", choices=("verify",), default="verify")
    args, _remaining = parser.parse_known_args()
    checkpoint = args.output / "checkpoint.pt"
    restore_rng(torch.load(checkpoint, map_location="cpu", weights_only=True)["rng"], "cpu")
    if "--mode" not in sys.argv:
        sys.argv.extend(("--mode", "verify"))
    replay()
    bindings = {"checkpoint_sha256": digest(checkpoint), "full_global_rng_restored": ["python", "numpy", "torch"],
                "wrapper_sha256_lf": hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest()}
    for name in ("plan.json", "verification.json"):
        path = args.audit_output / name
        write_json(path, {**read(path), "rng_restoration": bindings})


if __name__ == "__main__":
    main()
