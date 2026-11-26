"""Export saved checkpoint traces as lightweight JSON for the Tk monitor."""

import argparse
import json
from pathlib import Path
import time

import torch

BACKBONES = ("residual_mlp", "transformer", "conformer")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--interval", type=float, default=2.)
    args = parser.parse_args()
    state = {}
    destination = args.output / "live-validation-loss.json"
    while True:
        curves, errors = {}, {}
        for backbone in BACKBONES:
            path = args.output / f"expanded-independent-seed-{backbone}.pt"
            try:
                stat = path.stat()
                stamp = (stat.st_mtime_ns, stat.st_size)
                if state.get(backbone, (None,))[0] != stamp:
                    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
                    state[backbone] = (stamp, checkpoint.get("trace", []))
                curves[backbone] = state[backbone][1]
            except FileNotFoundError:
                curves[backbone] = []
            except (OSError, RuntimeError, ValueError, EOFError) as error:
                errors[backbone] = str(error)
                curves[backbone] = state.get(backbone, (None, []))[1]
        payload = {"updated_at": time.time(), "curves": curves, "errors": errors}
        # The Tk process may have the JSON file open while reading it. On Windows
        # that can block replace(); the reader tolerates an occasional partial write.
        destination.write_text(json.dumps(payload), encoding="utf-8")
        try:
            status = json.loads((args.output / "status.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            status = {}
        if status.get("state") in {"complete", "failed", "stopped"}:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
