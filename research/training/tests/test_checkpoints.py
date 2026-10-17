from copy import deepcopy
import hashlib
import shutil

import pytest
import torch

from obruxo_training.checkpoints import COLLECTION_FORMAT, INITIAL_FILE, TrainingCheckpointWriter, atomic_torch_save, load_training_checkpoint
from obruxo_training.pilot import PilotPool
from obruxo_training.trainer import TrainConfig, train_pilot
from test_stream import FakeRendererFactory


def initial_value():
    return {"format": "obruxo_pilot_training_v1", "run_contract": {"name": "fixture"}, "step": 0,
            "model": {"weight": torch.tensor([1., 2.])}, "optimizer": {"state": {}}, "scheduler": None, "scaler": None,
            "basic_pitch": {"weight": torch.tensor([3.])}, "initial_metrics": None, "runtime": {"device": "cpu"}, "recovery": "exact",
            "rng": {"torch": torch.tensor([1], dtype=torch.uint8)}, "batch_rng": torch.tensor([2], dtype=torch.uint8),
            "cursor": {"next_sequence_id": 0}, "replay": [], "render_ready": False, "renderer_identity": None, "totals": {"accepted": 0}}


def assert_equal(left, right):
    if torch.is_tensor(left):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert type(left) is type(right) and len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            assert_equal(a, b)
    else:
        assert left == right


def test_collection_writes_initial_state_once_and_full_checkpoints_stay_self_contained(tmp_path):
    value = initial_value()
    writer = TrainingCheckpointWriter(tmp_path)
    writer.save(value)
    original_hash = hashlib.sha256((tmp_path / INITIAL_FILE).read_bytes()).hexdigest()
    value["cursor"]["next_sequence_id"] = 1
    value["replay"] = [{"target": {"continuous": [float(.1234567890123)]}, "features": torch.tensor([4.])}]
    writer.save(value)
    assert_equal(load_training_checkpoint(writer.path), value)
    assert writer.statistics["initial_writes"] == 1 and writer.statistics["collection_writes"] == 2
    assert hashlib.sha256((tmp_path / INITIAL_FILE).read_bytes()).hexdigest() == original_hash
    value.update(step=1, initial_metrics={"loss": 2.})
    writer.save(value)
    previous = load_training_checkpoint(tmp_path / "checkpoint.previous.pt")
    assert previous["step"] == 0 and previous["cursor"]["next_sequence_id"] == 1
    assert writer.statistics["full_writes"] == 1
    standalone = tmp_path / "standalone"
    standalone.mkdir()
    shutil.copyfile(writer.path, standalone / "checkpoint.pt")
    assert_equal(load_training_checkpoint(standalone / "checkpoint.pt"), value)
    assert_equal(torch.load(standalone / "checkpoint.pt", weights_only=True), value)


@pytest.mark.parametrize("change,reason", [("missing", "requires its sibling"), ("corrupt", "hash differs"),
    ("traversal", "invalid initial-state reference"), ("override", "invalid initial-collection"), ("trained_base", "unchanged pre-training")])
def test_collection_recovery_fails_closed_on_missing_corrupt_or_replaced_base(tmp_path, change, reason):
    writer = TrainingCheckpointWriter(tmp_path)
    writer.save(initial_value())
    saved = torch.load(writer.path, weights_only=True)
    base_path = tmp_path / INITIAL_FILE
    if change == "missing":
        base_path.unlink()
    elif change == "corrupt":
        base_path.write_bytes(b"corrupt")
    elif change == "traversal":
        saved["initial"]["file"] = "../checkpoint.initial.pt"
    elif change == "override":
        saved["state"]["model"] = {}
    else:
        base = torch.load(base_path, weights_only=True)
        base["state"]["step"] = 1
        torch.save(base, base_path)
        saved["initial"]["sha256"] = hashlib.sha256(base_path.read_bytes()).hexdigest()
    torch.save(saved, writer.path)
    with pytest.raises(ValueError, match=reason):
        load_training_checkpoint(writer.path)


def test_failed_verification_preserves_latest_and_backup(tmp_path, monkeypatch):
    import obruxo_training.checkpoints as checkpoints
    path = tmp_path / "checkpoint.pt"
    atomic_torch_save(path, {"value": 1})
    atomic_torch_save(path, {"value": 2})
    before = path.read_bytes(), (tmp_path / "checkpoint.previous.pt").read_bytes()
    with monkeypatch.context() as context:
        def broken_load(*args, **kwargs):
            raise OSError("verification interrupted")
        context.setattr(checkpoints.torch, "load", broken_load)
        with pytest.raises(OSError, match="interrupted"):
            atomic_torch_save(path, {"value": 3})
    assert before == (path.read_bytes(), (tmp_path / "checkpoint.previous.pt").read_bytes())
    assert not list(tmp_path.glob("*.partial"))


def test_interruption_during_initial_collection_recovers_model_optimizer_rng_replay_and_cursor(tmp_path):
    pool = PilotPool.create(12)
    config = TrainConfig(steps=3, batch_size=2, replay_capacity=4, render_budget=4, checkpoint_every=1, cpu_threads=2)

    def run(name, **kwargs):
        return train_pilot(pool, tmp_path / name, config=config, renderer_factory=FakeRendererFactory(), **kwargs)

    def crash(event):
        if event["stage"] == "render" and event["replay_examples"] == 2:
            raise InterruptedError("stop during initial collection")

    full = run("full")
    with pytest.raises(InterruptedError, match="initial collection"):
        run("crash", progress=crash)
    raw = torch.load(tmp_path / "crash/checkpoint.pt", weights_only=True)
    assert raw["format"] == COLLECTION_FORMAT and "model" not in raw
    collected = load_training_checkpoint(tmp_path / "crash/checkpoint.pt")
    assert collected["step"] == 0 and len(collected["replay"]) == collected["cursor"]["next_sequence_id"] == 2
    resumed = run("resumed", resume=tmp_path / "crash/checkpoint.pt")
    assert full["model_state_sha256"] == resumed["model_state_sha256"]
    left, right = [load_training_checkpoint(tmp_path / name / "checkpoint.pt") for name in ("full", "resumed")]
    for key in ("model", "optimizer", "rng", "batch_rng", "replay", "cursor"):
        assert_equal(left[key], right[key])
    assert full["checkpoint_storage"]["initial_writes"] == resumed["checkpoint_storage"]["initial_writes"] == 1


def test_old_self_contained_artifacts_load_without_conversion(tmp_path):
    path = tmp_path / "legacy.pt"
    value = initial_value()
    torch.save(value, path)
    assert_equal(load_training_checkpoint(path), value)
    with pytest.raises(ValueError, match="unsupported training checkpoint schema"):
        TrainingCheckpointWriter(tmp_path).save({"format": "other"})
    changed = deepcopy(value)
    changed["optimizer"]["state"] = {0: {"step": 1}}
    with pytest.raises(ValueError, match="cannot contain optimizer updates"):
        TrainingCheckpointWriter(tmp_path).save(changed)
