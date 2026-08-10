from __future__ import annotations

from pathlib import Path

import numpy as np
import openvino as ov
import torch
from obruxo_basic_pitch.benchmark_worker import _candidate_parity
from obruxo_basic_pitch.model import BasicPitchICASSP2022
from obruxo_basic_pitch.parity import synthetic_windows

ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_PATH = ROOT / "artifacts" / "basic_pitch_icassp_2022.pt"


def test_openvino_cpu_dynamic_batch_matches_canonical_pytorch() -> None:
    state = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True)
    model = BasicPitchICASSP2022().eval()
    model.load_state_dict(state, strict=True)
    converted = ov.convert_model(model, example_input=torch.zeros((1, 43_844, 1), dtype=torch.float32))
    converted.reshape({converted.inputs[0]: ov.PartialShape([-1, 43_844, 1])})
    compiled = ov.Core().compile_model(converted, "CPU")
    public = synthetic_windows()
    windows = np.concatenate((public, public[:3]), axis=0)

    for batch_size in (1, 2, 4, 8):
        batch = windows[:batch_size]
        with torch.inference_mode():
            reference_values = model(torch.from_numpy(batch))
        reference = {name: value.detach().numpy() for name, value in reference_values.items()}
        result = compiled(batch)
        candidate_values = [np.asarray(result[output], dtype=np.float32) for output in compiled.outputs]
        candidate = dict(zip(("note", "onset", "contour"), candidate_values, strict=True))
        summary = _candidate_parity(np, reference, candidate)
        assert summary["parity_passed"]
        assert summary["note_threshold_disagreements"] == 0
        assert summary["onset_threshold_disagreements"] == 0
        assert summary["event_structure_disagreements"] == 0
        assert "pitch_bend_element_disagreements" in summary
