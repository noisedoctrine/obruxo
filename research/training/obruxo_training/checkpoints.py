"""Durable training snapshots without repeatedly writing unchanged initial state."""
import hashlib
import io
import os
from pathlib import Path
import shutil

import torch

COLLECTION_FORMAT = "obruxo_initial_collection_v1"
INITIAL_FORMAT = "obruxo_initial_state_v1"
INITIAL_FILE = "checkpoint.initial.pt"
STATIC_KEYS = frozenset({"run_contract", "step", "model", "optimizer", "scheduler", "scaler", "basic_pitch", "initial_metrics", "runtime", "recovery"})
DYNAMIC_KEYS = frozenset({"rng", "batch_rng", "cursor", "replay", "render_ready", "renderer_identity", "totals"})


def atomic_torch_save(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    previous_tmp = path.with_name(path.stem + ".previous.pt").with_suffix(".partial")
    try:
        with temporary.open("wb") as handle:
            torch.save(value, handle)
            handle.flush()
            os.fsync(handle.fileno())
        torch.load(temporary, map_location="cpu", weights_only=True)
        if path.exists():
            previous = path.with_name(path.stem + ".previous.pt")
            with path.open("rb") as source, previous_tmp.open("wb") as target:
                shutil.copyfileobj(source, target)
                target.flush()
                os.fsync(target.fileno())
            os.replace(previous_tmp, previous)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
        previous_tmp.unlink(missing_ok=True)


def load_training_checkpoint(path: Path) -> dict:
    """Resolve collection state against its exact sibling; ordinary artifacts pass through."""
    path = Path(path)
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(saved, dict):
        raise ValueError("training checkpoint must contain a dictionary")
    if saved.get("format") != COLLECTION_FORMAT:
        return saved
    if set(saved) != {"format", "initial", "state"} or not isinstance(saved["state"], dict) or set(saved["state"]) != DYNAMIC_KEYS:
        raise ValueError("invalid initial-collection checkpoint fields")
    initial = saved["initial"]
    if not isinstance(initial, dict) or set(initial) != {"file", "sha256"} or initial["file"] != INITIAL_FILE:
        raise ValueError("invalid initial-state reference")
    try:
        raw = path.with_name(INITIAL_FILE).read_bytes()
    except FileNotFoundError as exc:
        raise ValueError("collection recovery requires its sibling checkpoint.initial.pt") from exc
    if hashlib.sha256(raw).hexdigest() != initial["sha256"]:
        raise ValueError("initial-state checkpoint hash differs")
    base = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    if (not isinstance(base, dict) or set(base) != {"format", "state"} or base["format"] != INITIAL_FORMAT
            or not isinstance(base["state"], dict) or set(base["state"]) != STATIC_KEYS):
        raise ValueError("invalid initial-state checkpoint fields")
    state = base["state"]
    if state["step"] != 0 or state["initial_metrics"] is not None or state["optimizer"].get("state") != {}:
        raise ValueError("collection recovery requires unchanged pre-training state")
    return {"format": "obruxo_pilot_training_v1", **state, **saved["state"]}


class TrainingCheckpointWriter:
    def __init__(self, output: Path):
        self.path, self.initial = Path(output) / "checkpoint.pt", None
        self.statistics = {"initial_writes": 0, "collection_writes": 0, "full_writes": 0, "serialized_bytes": 0}

    def _write(self, path, value, kind):
        atomic_torch_save(path, value)
        self.statistics[kind + "_writes"] += 1
        self.statistics["serialized_bytes"] += path.stat().st_size

    def save(self, value):
        if set(value) != STATIC_KEYS | DYNAMIC_KEYS | {"format"} or value["format"] != "obruxo_pilot_training_v1":
            raise ValueError("unsupported training checkpoint schema")
        if value["step"] != 0 or value["initial_metrics"] is not None:
            self._write(self.path, value, "full")
            return
        if value["optimizer"].get("state") != {}:
            raise ValueError("initial collection cannot contain optimizer updates")
        if self.initial is None:
            path = self.path.with_name(INITIAL_FILE)
            if path.exists():
                raise FileExistsError(path)
            self._write(path, {"format": INITIAL_FORMAT, "state": {key: value[key] for key in STATIC_KEYS}}, "initial")
            self.initial = {"file": INITIAL_FILE, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        self._write(self.path, {"format": COLLECTION_FORMAT, "initial": self.initial,
                               "state": {key: value[key] for key in DYNAMIC_KEYS}}, "collection")
