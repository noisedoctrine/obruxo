"""Verify Stage 2 MIDI preprocessing and encoder output against the pinned upstream code."""

import argparse
import importlib
import json
from pathlib import Path
import sys

import torch

TRAINING = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(TRAINING.parent / "data_generation"), str(TRAINING)]

from obruxo_data.midi import Performance
from obruxo_training.issue187_stage2 import (
    EventSetEncoder, encode_midi_spans, load_midi_encoder, performance_to_spans,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder", type=Path, required=True)
    parser.add_argument("--upstream-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.upstream_source))
    upstream_model = importlib.import_module("spansynth.model")
    upstream_midi = importlib.import_module("spansynth.midi")
    score = Performance(bpm=120)
    score.add_note(pitch=48, velocity=72, start_tick=0, duration_ticks=360)
    score.add_note(pitch=55, velocity=104, start_tick=120, duration_ticks=720)
    score.add_note(pitch=60, velocity=48, start_tick=240, duration_ticks=240)
    notes = performance_to_spans(score, default_program=80)
    local_rows = encode_midi_spans(notes, duration_seconds=1.0)
    reference_notes = [{**note} for note in notes]
    valid = torch.arange(512) < 25
    reference_rows = upstream_midi.encode_midi(reference_notes, valid)
    row_differences = {
        name: float((local_rows[name].float() - reference_rows[name].float()).abs().max())
        for name in ("kind_id", "pitch_id", "numeric", "event_valid")
    }
    if any(value != 0 for value in row_differences.values()):
        raise RuntimeError(f"MIDI row preprocessing differs from upstream: {row_differences}")
    local = load_midi_encoder(args.encoder)
    reference = upstream_model.EventSetEncoder(upstream_model.EventSetConfig())
    reference.load_state_dict(local.state_dict(), strict=True)
    local.eval().requires_grad_(False)
    reference.eval().requires_grad_(False)
    rows = (local_rows["kind_id"], local_rows["pitch_id"], local_rows["numeric"], local_rows["event_valid"])
    with torch.inference_mode():
        actual = local(*rows)
        expected = reference(*rows)
    difference = float((actual - expected).abs().max())
    if difference != 0.0:
        raise RuntimeError(f"MIDI encoder output differs from upstream: max absolute error {difference}")
    report = {"device": "cpu", "midi_row_max_absolute_differences": row_differences,
              "encoder_output_shape": list(actual.shape), "encoder_max_absolute_difference": difference,
              "active_notes": len(notes), "active_event_rows": int(local_rows["event_valid"].sum()),
              "nonzero_feature_frames": int((actual.abs().sum(-1) > 0).sum())}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
