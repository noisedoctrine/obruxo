"""Preserve original license and notice files from a locked local environment."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

if __name__ == "__main__":
    from obruxo_training.runtime_notices import collect
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("lock", "environment", "site-packages", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    result = collect(args.lock, args.environment, args.site_packages, args.output)
    print(json.dumps({"inventory_id": result["inventory_id"], "packages": len(result["packages"]), "files": len(result["files"]), "release_notices_complete": False}))
