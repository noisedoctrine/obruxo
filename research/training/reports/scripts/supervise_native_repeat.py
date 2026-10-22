"""Prepare or supervise one explicitly costed repeat of an interrupted native fit."""

import argparse
import json
from pathlib import Path
import sys
import traceback

RESEARCH = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(RESEARCH / "training"), str(RESEARCH / "data_generation")]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--request", type=Path)
    args = parser.parse_args()
    if bool(args.plan) == bool(args.request):
        parser.error("choose exactly one of --plan or --request")
    try:
        request = json.loads((args.plan or args.request).read_text())
        sys.path.insert(0, request["plan"]["pythonpath"] if args.plan else request["pythonpath"])
        from obruxo_training.native_repeat import prepare_repeat, supervise_repeat
        if args.plan:
            supervise_repeat(args.plan)
        else:
            plan = prepare_repeat(request["source"], request["output"], request["command"],
                                  working_directory=request["working_directory"], pythonpath=request["pythonpath"], prelaunches=request.get("prelaunches", ()))
            print(json.dumps({"limits": plan["limits"], "training_plan_id": plan["training_plan_id"]}))
    except BaseException:
        if args.plan:
            (args.plan.parent / "supervisor.traceback.log").write_text(traceback.format_exc(), encoding="utf-8")
        raise
