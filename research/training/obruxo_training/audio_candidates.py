"""Optional bounded candidate ranking whose sole external conditioning input is audio."""

from dataclasses import asdict
from time import perf_counter

import numpy as np
import torch

from obruxo_data.hashing import canonical_sha256
from obruxo_data.midi import Performance
from obruxo_data.render.base import RenderRequest

from .candidates import PROPOSAL_SPEC, propose_candidates
from .codec import PilotTarget
from .evaluate import audio_distance, load_evaluation_checkpoint, retrieval_digest
from .features import tensor_digest, validate_features
from .inference_profile import PROFILE_KEYS, model_rendering, prediction_preset
from .model import batch_features
from .performances import CAPABILITIES
from .stream import NativeRendererFactory, render_isolated
from .trainer import atomic_torch_save, inference_state, restore_model


RANKING_SPEC = {"ranker": "frozen_basic_pitch_native_four_candidates_v1", "input": "audio_only", "maximum_candidates": 4,
                "maximum_notes": 32, "maximum_input_seconds": 16, "maximum_bank_states": 32,
                "performance": "stock_basic_pitch_notes_linear_posterior_amplitude_velocity_no_bends_tempo120",
                "score": "mrstft_zero_pad_to_common_length_without_gain_alignment", "ties": "proposal_order",
                "failure": "explicit_unranked_model_prediction", "proposal": PROPOSAL_SPEC}


def candidate_bank(source, maximum=32):
    if type(maximum) is not int or not 1 <= maximum <= RANKING_SPEC["maximum_bank_states"]:
        raise ValueError("candidate bank needs one through 32 training states")
    bank, seen = [], set()
    for item in source["replay"]:
        if item["state_id"] in seen:
            continue
        bank.append({"state_id": item["state_id"], "target": item["target"], "features": item["features"]})
        seen.add(item["state_id"])
        if len(bank) == maximum:
            break
    if not bank:
        raise ValueError("candidate bank is empty")
    return bank


def export_candidate_model(pool, checkpoint, output, *, qualification=None):
    source, model, frontend, _ = load_evaluation_checkpoint(checkpoint, pool)
    bank = candidate_bank(source)
    identity = {"model_sha256": tensor_digest(model.state_dict()), "bank_sha256": retrieval_digest(bank),
                "frontend": frontend.describe(), "pool_id": pool.pool_id, "ranking_spec": RANKING_SPEC,
                "renderer_identity": source["renderer_identity"], **model_rendering(model)}
    qualified = qualification is not None and qualification.get("qualified") is True and qualification.get("identity") == identity
    if qualification is not None and not qualified:
        raise ValueError("candidate ranking qualification does not match this model and bank")
    atomic_torch_save(output, {"format": "obruxo_candidate_inference_v1", "base": inference_state(model, frontend, pool.codec, pool.pool_id),
                               "bank": bank, "identity": identity, "renderer_identity": source["renderer_identity"],
                               "qualified": qualified, "qualification": qualification})
    return identity


def load_candidate_model(path, *, allow_unqualified=False):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved.get("format") != "obruxo_candidate_inference_v1" or saved["identity"]["ranking_spec"] != RANKING_SPEC:
        raise ValueError("unsupported audio-only candidate bundle")
    model, frontend, codec = restore_model(saved["base"])
    bank = saved["bank"]
    for item in bank:
        validate_features(item["features"])
        codec.validate_target(PilotTarget.from_dict(item["target"]))
    if (not 1 <= len(bank) <= 32 or len({item["state_id"] for item in bank}) != len(bank)
            or saved["identity"]["model_sha256"] != tensor_digest(model.state_dict())
            or saved["identity"]["bank_sha256"] != retrieval_digest(bank) or saved["identity"]["frontend"] != frontend.describe()
            or saved["identity"]["pool_id"] != saved["base"]["training_pool_id"]
            or saved["identity"]["renderer_identity"] != saved["renderer_identity"]):
        raise ValueError("candidate bundle model, bank or frozen frontend changed")
    rendering = model_rendering(model)
    if any(saved["identity"].get(key) != rendering.get(key) for key in PROFILE_KEYS):
        raise ValueError("candidate bundle rendering metadata changed")
    qualification = saved["qualification"]
    if saved["qualified"] and (not qualification or not qualification.get("qualified") or qualification.get("identity") != saved["identity"]):
        raise ValueError("candidate bundle qualification changed")
    if not saved["qualified"] and not allow_unqualified:
        raise ValueError("audio-only ranking has not passed native qualification; diagnostic use requires allow_unqualified")
    return model, frontend, codec, bank, saved["renderer_identity"]


def performance_from_notes(notes, audio_seconds):
    if not 0 < audio_seconds <= 16 or len(notes) > RANKING_SPEC["maximum_notes"]:
        raise ValueError("estimated performance exceeds note or duration budget")
    if not notes:
        raise ValueError("no notes detected in the input recording")
    performance = Performance(bpm=120)
    for note in notes:
        if (not np.isfinite([note.start_time_s, note.end_time_s, note.amplitude]).all()
                or not 0 <= note.start_time_s < note.end_time_s <= audio_seconds + .05
                or type(note.pitch_midi) is not int or not 0 <= note.pitch_midi <= 127 or not 0 <= note.amplitude <= 1):
            raise ValueError("invalid estimated note")
        start = max(0, round(note.start_time_s * 960))
        end = min(round(audio_seconds * 960), round(note.end_time_s * 960))
        if end <= start:
            raise ValueError("estimated note has zero quantized duration")
        velocity = max(1, min(127, round(note.amplitude * 127)))
        performance.add_note(pitch=int(note.pitch_midi), velocity=velocity, start_tick=start, duration_ticks=end - start)
    performance.validate(CAPABILITIES).require_valid()
    return performance


def aligned_distance(reference, candidate):
    length = max(len(reference), len(candidate))
    arrays = [np.pad(value, ((0, length - len(value)), (0, 0))).astype(np.float32) for value in (reference, candidate)]
    return audio_distance(*arrays)


@torch.no_grad()
def rank_audio(model, frontend, codec, bank, audio, *, sample_rate=44100, renderer_factory=None, timeout=30.,
               expected_renderer=None, progress=None):
    """No supplied MIDI, source labels or evaluator state can enter this interface."""
    from obruxo_basic_pitch.postprocess import posteriorgrams_to_note_events
    if next(model.parameters()).device.type != "cpu" or model.training:
        raise ValueError("candidate ranking requires the CPU evaluation model")
    started = perf_counter()
    features, posterior = frontend.analyze(audio, sample_rate)
    prediction = model(batch_features([{"features": features}]))
    candidates = propose_candidates(prediction, features, bank, codec, seed=0)
    result = {"ranked": False, "selected": "model", "candidates": {name: target.to_dict() for name, target in candidates.items()},
              "scores": {}, "renders": [], "input": "audio_only", "source_midi_supplied": False}
    try:
        notes = posteriorgrams_to_note_events(posterior)
        performance = performance_from_notes(notes, len(audio) / sample_rate)
    except ValueError as error:
        result.update(failure=str(error), elapsed_seconds=perf_counter() - started)
        frontend.require_frozen()
        return candidates["model"], result
    result["estimated_notes"] = [asdict(note) for note in notes]
    result["estimated_performance"] = performance.to_dict()
    result["estimated_performance_id"] = canonical_sha256(performance.to_dict())
    for name, target in candidates.items():
        rendering = model_rendering(model)
        request = RenderRequest(preset_json=prediction_preset(model, codec, target).decode(), performance=performance, sample_rate=sample_rate,
                                channels=2, tail_seconds=codec.contract.tail_seconds, renderer_id=codec.contract.renderer_id,
                                preroll_frames=rendering.get("render_profile", {}).get("preroll_frames", 0))
        rendered, metadata = render_isolated(request, renderer_factory=renderer_factory or NativeRendererFactory(), timeout=timeout)
        event = {"method": name, "request_id": request.request_id, **metadata}
        result["renders"].append(event)
        if progress:
            progress(event)
        if metadata["status"] != "error":
            identity = {key: value for key, value in metadata["provenance"].items() if key != "request_id"}
            if expected_renderer is not None and identity != expected_renderer:
                raise ValueError("candidate inference renderer differs from its trained artifact")
        if rendered is not None:
            result["scores"][name] = aligned_distance(audio, rendered)
    if result["scores"]:
        result["selected"] = min(result["scores"], key=lambda name: result["scores"][name]["mrstft"])
        result["ranked"] = True
    else:
        result["failure"] = "all_candidate_renders_rejected"
    frontend.require_frozen()
    result["elapsed_seconds"] = perf_counter() - started
    return candidates[result["selected"]], result
