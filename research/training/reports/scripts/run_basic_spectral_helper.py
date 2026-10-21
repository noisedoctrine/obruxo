"""Run the bounded local basic spectral-helper experiment; no Colab dependencies."""

import argparse
import json
from pathlib import Path
import sys

RESEARCH = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(RESEARCH / "training"), str(RESEARCH / "data_generation")]


def main():
    from obruxo_training.pilot import PilotPool
    from obruxo_training.temporal_proxy_screen import run_temporal_proxy_screen
    from obruxo_training.temporal_proxy_data import TemporalProxyConfig
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--paired-changes", action="store_true", help="Use aligned pairs and the declared local-change objective")
    args = parser.parse_args()
    def progress(event):
        if event["stage"] != "collect" or event["completed"] in (1, event["planned"]) or event["completed"] % 32 == 0:
            print(json.dumps(event, allow_nan=False), flush=True)
    result = run_temporal_proxy_screen(PilotPool.load(args.pool), args.output, resume=args.resume, progress=progress,
                                      config=TemporalProxyConfig(paired_changes=args.paired_changes))
    print(json.dumps(result, allow_nan=False), flush=True)
    return 0 if result["complete"] and result["within_resource_limits"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
