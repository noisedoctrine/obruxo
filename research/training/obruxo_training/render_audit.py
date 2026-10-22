"""Small child-side audit wrapper; importing it does not load the training runtime."""

from dataclasses import dataclass
import json
import os
from pathlib import Path
from time import time

from obruxo_data.hashing import sha256_bytes


@dataclass(frozen=True)
class AuditedFactory:
    output: str
    phase: str
    base: object

    def __call__(self):
        return AuditedRenderer(self)


class AuditedRenderer:
    def __init__(self, factory):
        self.factory = factory

    def render(self, request):
        root, phase = Path(self.factory.output), self.factory.phase
        plan = json.loads((root / "plan.json").read_text())
        started = json.loads((root / "run.json").read_text())["started_unix"]
        path = root / "attempts.jsonl"
        attempts = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
        if time() - started >= plan["limits"]["wall_seconds"]:
            raise RuntimeError("bootstrap original wall-time budget exhausted")
        if sum(row["phase"] == phase for row in attempts) >= plan["limits"][phase + "_attempts"]:
            raise RuntimeError("bootstrap native attempt budget exhausted")
        with path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps({"phase": phase, "request_id": request.request_id,
                                     "controlled_sha256": sha256_bytes(request.preset_json.encode()), "started_unix": time()}, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        return self.factory.base().render(request)
