import copy
import json
from collections import Counter
from pathlib import Path

import pytest
from obruxo_training.balanced_basic_data import grid_rows
from obruxo_training.balanced_basic_fit import SPEC, extend_cells, schedule, train
from obruxo_training.basic_direct import rehearsal_indices
from obruxo_training.paired_native_audit import equal_tree
from obruxo_training.paired_native_step import reciprocal_updates
from test_shared_basic_auxiliary import synthetic  # noqa: F401


def control_schedule():
    updates = []
    for seed in range(8):
        for update in reciprocal_updates(3, seed=seed):
            examples = [{**row, "state": SPEC["training_indices"][row["state"]]} for row in update["examples"]]
            updates.append({"examples": examples, "references": list(rehearsal_indices(len(updates)))})
    return updates


def test_patch_identity_does_not_change_the_grid_sampling_distribution():
    original = control_schedule()
    unchanged = copy.deepcopy(original)
    updates = schedule(original)
    assert original == unchanged
    assert [row["examples"][0]["state"] for row in updates] == [row["examples"][0]["state"] for row in original]
    assert [row["references"] for row in updates] == [row["references"] for row in original]
    counts = {state: Counter(row["examples"][1]["input_view"] - 3 for row in updates if row["examples"][0]["state"] == state)
              for state in SPEC["training_indices"]}
    assert counts[2] == counts[8] == counts[11] and sum(counts[2].values()) == 16 and set(counts[2]) == set(range(6))
    assert sum(count for grid, count in counts[2].items() if grid % 2 == 0) == 8
    assert [sum(count for grid, count in counts[2].items() if grid // 2 == pitch) for pitch in range(3)] == [6, 5, 5]
    with pytest.raises(ValueError, match="sixteen"):
        schedule(original[:-1])


def test_legacy_endpoints_stay_intact_and_every_reciprocal_training_cell_exists():
    path = Path(__file__).parents[1] / "reports/data/trainable_basic_fit_plan.json"
    prior = json.loads(path.read_text())
    original = copy.deepcopy(prior["fit_audit_cells"])
    patches = [{"training_index": state, "state_id": prior["training"][state]["state_id"], "target": prior["training"][state]["target"]}
               for state in SPEC["training_indices"]]
    rows = grid_rows(patches)
    records = {row["index"]: {"receipt": {"qa": {"audio_float32_sha256": f"audio-{row['index']}"}}, "features_sha256": f"features-{row['index']}"} for row in rows}
    cells = extend_cells(prior, {"observations": rows}, records)
    assert prior["fit_audit_cells"] == original == cells[:104]
    lookup = {(cell["recipe"]["state_id"], cell["input_view"], cell["playback_view"]): index for index, cell in enumerate(cells)}
    assert len(lookup) == 158 and len({(state, view) for state, view, _playback in lookup}) == 74
    for update in schedule(control_schedule()):
        state = prior["training"][update["examples"][0]["state"]]["state_id"]
        view = update["examples"][1]["input_view"]
        assert lookup[state, 0, 0] < 104
        assert 140 <= lookup[state, 0, view] < 158
        assert 104 <= lookup[state, view, view] < 140 and 104 <= lookup[state, view, 0] < 140


def test_single_candidate_recovers_model_adam_trace_and_rng(synthetic):  # noqa: F811
    inputs, _rows, store = synthetic
    _model, expected = train(inputs, store, stop_after=3)
    _model, first = train(inputs, store, stop_after=1)
    _model, recovered = train(inputs, store, saved=first, stop_after=3)
    assert equal_tree(expected, recovered)
    assert all(set(row["arms"]) == {"balanced"} and row["arms"]["balanced"]["coefficient"] == 0 for row in expected["trace"])
