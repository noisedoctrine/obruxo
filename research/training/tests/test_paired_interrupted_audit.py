import copy
import json
import random

import pytest
import torch

from obruxo_training.native_fit import digest
from obruxo_training.paired_interrupted_audit import anchor_external_rng, project_committed_prefix
from obruxo_training.paired_native_audit import equal_tree
from obruxo_training.trainer import capture_rng, restore_rng


def fixture_source(tmp_path):
    source, projected = tmp_path / "source", tmp_path / "projected"
    source.mkdir()
    projected.mkdir()
    (source / "records").mkdir()
    for name in ("plan.json", "run.json", "checkpoint.pt"):
        (source / name).write_bytes(b"retained original")
    rows = {str(index): {"file": f"{index:03d}.pt"} for index in range(3)}
    (source / "records.json").write_text(json.dumps(rows))
    for row in rows.values():
        (source / "records" / row["file"]).write_bytes(b"retained record")
    events = []
    for step in range(3):
        for label in [*(f"source:{step}:{view}" for view in range(3)), f"gradient:{step}:0:0"]:
            events.extend(({"event": "attempt", "label": label}, {"event": "outcome", "label": label, "status": "accepted"}))
    events[-1]["status"] = "error"
    for name, values in (("renders.jsonl", events), ("updates.jsonl", [{"attempt": i} for i in range(3)]),
                         ("measurements.jsonl", [{"optimizer_attempt": i} for i in range(3)])):
        (source / name).write_text("".join(json.dumps(row) + "\n" for row in values))
    return source, projected, {"step": 2, "trace": [{"attempt": 0}, {"attempt": 1}]}


def test_projects_committed_prefix_and_keeps_failed_tail_original(tmp_path):
    source, projected, saved = fixture_source(tmp_path)
    before = {p.relative_to(source): p.read_bytes() for p in source.rglob("*") if p.is_file()}
    result = project_committed_prefix(source, projected, saved)
    assert result["included_render_ledger_rows"] == 16
    assert result["excluded_render_ledger_rows"] == 8
    assert result["excluded_optimizer_charges"] == 1
    assert result["source_record_indices"] == [0, 1]
    assert json.loads((projected / "records.json").read_text()).keys() == {"0", "1"}
    assert {p.relative_to(source): p.read_bytes() for p in source.rglob("*") if p.is_file()} == before


def test_rejects_noncontiguous_committed_history(tmp_path):
    source, projected, saved = fixture_source(tmp_path)
    saved["trace"][1]["attempt"] = 2
    with pytest.raises(ValueError, match="uninterrupted committed prefix"):
        project_committed_prefix(source, projected, saved)


def test_rejects_a_failure_inside_the_committed_projection(tmp_path):
    source, projected, saved = fixture_source(tmp_path)
    path = source / "renders.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[1]["status"] = "error"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="failed native request"):
        project_committed_prefix(source, projected, saved)


def rng_witness(tmp_path):
    witness = {"plan_id": "frozen", "step": 4, "trace": list(range(4)), "rng": capture_rng("cpu")}
    path = tmp_path / "prefix-checkpoint.pt"
    torch.save(witness, path)
    evidence = {"plan_id": "frozen", "replayed_updates": 4, "offline_training_replay_exact": True, "checkpoint_file_sha256": digest(path)}
    (tmp_path / "prefix-verification.json").write_text(json.dumps(evidence))
    saved = {**copy.deepcopy(witness), "step": 49, "trace": list(range(49))}
    return saved


def test_external_rng_anchor_does_not_import_torch_state(tmp_path):
    before = capture_rng("cpu")
    try:
        saved = rng_witness(tmp_path)
        random.seed(819)
        torch.manual_seed(273)
        current = capture_rng("cpu")
        report = anchor_external_rng(tmp_path, saved)
        actual = capture_rng("cpu")
        assert equal_tree(actual["torch"], current["torch"])
        assert equal_tree(actual["python"], saved["rng"]["python"])
        assert equal_tree(actual["numpy"], saved["rng"]["numpy"])
        assert report["checkpoint_step"] == 4
    finally:
        restore_rng(before, "cpu")


@pytest.mark.parametrize("changed", ["python", "numpy", "witness"])
def test_external_rng_anchor_rejects_a_changed_saved_stream_or_witness(tmp_path, changed):
    saved = rng_witness(tmp_path)
    if changed == "python":
        saved["rng"]["python"] = random.Random(991).getstate()
    elif changed == "numpy":
        saved["rng"]["numpy"][2] = (saved["rng"]["numpy"][2] + 1) % 624
    else:
        with (tmp_path / "prefix-checkpoint.pt").open("ab") as handle:
            handle.write(b"changed")
    with pytest.raises(ValueError, match="early RNG witness|immutable early witness"):
        anchor_external_rng(tmp_path, saved)
