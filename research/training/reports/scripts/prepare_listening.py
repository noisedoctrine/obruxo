"""Prepare an anonymous local listening page or audit the returned judgments."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main():
    from obruxo_training.listening import prepare, summarize, write_json
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    packet = commands.add_parser("prepare")
    packet.add_argument("--manifest", type=Path, required=True)
    packet.add_argument("--output", type=Path, required=True)
    packet.add_argument("--seed", type=int)
    score = commands.add_parser("summarize")
    score.add_argument("--key", type=Path, required=True)
    score.add_argument("--responses", type=Path, required=True)
    score.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        print(prepare(args.manifest, args.output, seed=args.seed))
        return
    if args.output.exists():
        raise ValueError("refusing to replace listening results")
    result = summarize(args.key, args.responses)
    write_json(args.output, result)
    print({"completed_trials": result["completed_trials"], "counts": result["counts"]})


if __name__ == "__main__":
    main()
