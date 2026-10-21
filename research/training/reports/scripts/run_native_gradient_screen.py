"""Freeze, then execute a bounded local test of numerical native sound-loss gradients."""

import argparse
import json
from pathlib import Path
import sys

RESEARCH = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(RESEARCH / "training"), str(RESEARCH / "data_generation")]


def main():
    from obruxo_training.native_gradient_screen import run_screen
    from obruxo_training.pilot import PilotPool
    from obruxo_training.render_profile import BASIC_PREROLL_FRAMES
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--preroll-frames", type=int, default=BASIC_PREROLL_FRAMES)
    args = parser.parse_args()
    print(json.dumps(run_screen(PilotPool.load(args.pool), args.output, execute=args.execute, preroll_frames=args.preroll_frames)), flush=True)


if __name__ == "__main__":
    main()
