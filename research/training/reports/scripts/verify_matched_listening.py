"""Verify loudness-matched WAVs with the independent pyloudnorm implementation."""

import argparse
from pathlib import Path
import sys

REPOSITORY = Path(__file__).resolve().parents[4]
sys.path[:0] = [str(REPOSITORY / "research/training"), str(REPOSITORY / "research/data_generation")]


def main():
    from obruxo_training.audit import write_json
    from obruxo_training.listening_loudness import verify_packet
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("key", "manifest", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = verify_packet(args.key, args.manifest)
    write_json(args.output, result)
    print({key: value for key, value in result.items() if key != "cases"})


if __name__ == "__main__":
    main()
