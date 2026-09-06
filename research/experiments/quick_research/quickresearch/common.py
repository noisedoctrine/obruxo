from __future__ import annotations

from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any

WORKSPACE = Path(__file__).resolve().parents[1]
REPO = WORKSPACE.parents[2]
OUTPUTS = WORKSPACE / "outputs"


class BudgetExpired(RuntimeError):
    pass


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_hash(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def code_hash(*directories: Path) -> str:
    return digest([(str(path.relative_to(REPO)), file_hash(path)) for directory in directories
                   for path in sorted(directory.rglob("*.py")) if "outputs" not in path.parts])


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(value)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def write_json(path: Path, value: Any) -> None:
    write_text(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    import io
    stream = io.StringIO(newline="")
    if rows:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(key for row in rows for key in row)))
        writer.writeheader()
        writer.writerows(rows)
    write_text(path, stream.getvalue())


def checked_output(path: Path) -> Path:
    resolved = path.resolve()
    root = OUTPUTS.resolve()
    if resolved == root or not resolved.is_relative_to(root):
        raise ValueError(f"output must be a child of {root}; raw corpus and source checkouts are read-only")
    return resolved


def status(output: Path, state: str, **fields: Any) -> None:
    previous = read_json(output / "status.json") if (output / "status.json").exists() else {}
    write_json(output / "status.json", {**previous, **fields, "state": state, "updated_unix": time.time()})


class Context:
    def __init__(self, output: Path, seconds: float):
        self.output = output
        self.started = time.monotonic()
        self.deadline = self.started + seconds
        self.last_status = self.started - 1
        self.last_console = self.started - 30

    @property
    def remaining(self) -> float:
        return max(0.0, self.deadline - time.monotonic())

    def check(self, reserve: float = 0.0) -> None:
        if self.remaining <= reserve:
            raise BudgetExpired("execution budget exhausted; cached work remains available")

    def progress(self, **fields: Any) -> None:
        self.check()
        now = time.monotonic()
        if now - self.last_status < 0.25:
            return
        status(self.output, "running", elapsed_seconds=time.monotonic() - self.started, **fields)
        self.last_status = now
        if now - self.last_console >= 30:
            print(json.dumps(fields), flush=True)
            self.last_console = now


def outcome_counts(rows: list[dict]) -> dict:
    return dict(Counter(row["state"] for row in rows))


def finish_report(context: Context, filename: str, text: str, state: str, **fields: Any) -> dict:
    write_text(context.output / filename, text)
    result = {"state": state, **fields, "elapsed_seconds": time.monotonic() - context.started}
    status(context.output, **result)
    return result
