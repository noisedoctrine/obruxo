import copy
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

from obruxo_training.spectral_controls import BasicControlAdapter
from obruxo_training.temporal_proxy_screen import fit_temporal_records
from test_event_frames import measured_record

spec = importlib.util.spec_from_file_location("response_fit", Path(__file__).parents[1] / "reports/scripts/run_event_response_fit.py")
diagnostic = importlib.util.module_from_spec(spec)
# dataclasses resolves the defining module when inspecting annotations.
sys.modules[spec.name] = diagnostic
spec.loader.exec_module(diagnostic)


def test_small_record_fit_uses_exact_pairs_and_recovers():
    torch.set_num_threads(2)
    first, _, _ = measured_record()
    second, _, _ = measured_record(3)
    records = [first, copy.deepcopy(first), second, copy.deepcopy(second)]
    for index, row in enumerate(records):
        row.update(pair_id=index // 2, source_state_id=str(index // 2))
    config = SimpleNamespace(seed=0, learning_rate=.001, steps=4, batch_size=4, checkpoint_every=1)
    contract, adapter, pairs = {"diagnostic_pairs": [0, 1]}, BasicControlAdapter(), [(0, 1), (2, 3)]
    _, reference = fit_temporal_records(records, adapter, config, contract, pairs=pairs)
    saved = {}
    def checkpoint(value):
        saved.clear()
        saved.update(copy.deepcopy(value))
    def interrupt(step):
        if step == 2:
            raise InterruptedError
    with pytest.raises(InterruptedError):
        fit_temporal_records(records, adapter, config, contract, pairs=pairs, checkpoint=checkpoint, after_step=interrupt)
    _, resumed = fit_temporal_records(records, adapter, config, contract, pairs=pairs, saved=saved)
    assert resumed["model_sha256"] == reference["model_sha256"] and resumed["trace"] == reference["trace"]
    for invalid in ([], [(0, 0)], [(0, 4)]):
        with pytest.raises(ValueError, match="valid original/changed"):
            fit_temporal_records(records, adapter, config, contract, pairs=invalid)


def test_response_gate_cannot_hide_a_failed_envelope_group_in_overall_average():
    rows = [{"group": group, "pair_id": 3 * index + i, "response_error": .5, "no_change_error": 1., "absolute_error": .4}
            for index, group in enumerate(("attack", "decay", "release", "detune")) for i in range(3)]
    result = {"uniform": copy.deepcopy(rows), "events": copy.deepcopy(rows)}
    gates = {"detune_mean_vs_no_change": 1., "attack_decay_release_mean_vs_no_change": .8,
             "attack_decay_release_cases_beating_no_change": 2, "events_absolute_error_vs_uniform": 1.1}
    assert diagnostic.summarize(result, gates)["fit_gate_passed"]
    for row in result["events"]:
        if row["group"] == "release":
            row["response_error"] = 1.01
    summary = diagnostic.summarize(result, gates)
    assert not summary["fit_gate_passed"] and not summary["groups"]["release"]["response_fit_pass"]
