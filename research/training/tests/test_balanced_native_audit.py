import json
from pathlib import Path
import runpy
from types import SimpleNamespace

import numpy as np
import torch

from obruxo_data.render.qa import audio_float32_sha256
from obruxo_training import balanced_native_fit, native_fit, temporal_proxy_data
from obruxo_training.balanced_sound_data import cross_view_updates
from obruxo_training.codec import PilotCodec
from obruxo_training.features import AudioFeatures, PITCH_SIZE, TIMBRE_SHAPE
from obruxo_training.model import PilotInverse
from obruxo_training.performances import initial_bank
from obruxo_training.render_profile import basic_render_profile
from obruxo_training.trainer import export_model


def test_complete_multiview_job_replays_and_audits_repeated_development_states(tmp_path, monkeypatch):
    """Exercise the real runner/auditor bookkeeping; a simulated worker supplies deterministic audio."""
    torch.set_num_threads(2)
    torch.manual_seed(4)
    codec, frontend = PilotCodec(), AudioFeatures()
    pool = SimpleNamespace(codec=codec, pool_id="auditor-fixture")
    initial, retrieval = tmp_path / "initial.pt", tmp_path / "retrieval.pt"
    export_model(initial, PilotInverse(codec), frontend, codec, pool.pool_id, preroll_frames=32768)
    target = codec.default_target().to_dict()
    torch.save({"replay": [{"target": target, "features": {"pitch": torch.zeros(PITCH_SIZE), "timbre": torch.zeros(*TIMBRE_SHAPE)}}]}, retrieval)
    training = [{"state_id": "train", "target": target, "input_views": "all",
                 "performances": [initial_bank()[view].performance().to_dict() for view in range(3)]}]
    validation = [{"state_id": "validation", **({"case_id": f"added-{case}"} if case else {}), "target": target,
                   "performances": [initial_bank()[offset + case % 4].performance().to_dict() for offset in (0, 4)]}
                  for case in range(3)]
    plan = {"format": "matched_native_inverse_fit_v4", "codec": codec.to_dict(), "training": training, "validation": validation,
            "training_view_roles": ["original", "held", "alternate"], "updates": cross_view_updates(1),
            "validation_panels": {"original": ["validation"], "counterbalanced": ["added-1", "added-2"]},
            "render_profile": basic_render_profile(), "optimizer": {"seed": 0, "learning_rate": .0001},
            "category_training": {"scales": [3., 4.]},
            "initial_file_sha256": native_fit.digest(initial), "retrieval_file_sha256": native_fit.digest(retrieval),
            "limits": {"wall_seconds": 600, "process_tree_bytes": 4 * 1024**3, "archive_bytes": 512 * 1024**2,
                       "replay_bytes": 64 * 1024**2, "replay_records": 4, "optimizer_attempts": 8,
                       "training_attempts": 110, "qualification_attempts": 64}}

    class Worker:
        def __init__(self, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

        def render(self, request, **_kwargs):
            payload = json.loads(request.preset_json)
            payload["settings"].update({f"osc_{index}_random_phase": 1. for index in (1, 2, 3)})
            controls = codec.encode(json.dumps(payload).encode())
            pitch = request.performance.note_spans()[0].pitch
            amplitude = .001 * (1 + sum(controls.continuous) + .2 * sum(controls.categorical))
            wave = (np.sin(np.arange(8192) * (.03 + .0001 * pitch)) * amplitude).astype(np.float32)
            audio = np.stack((wave, wave * .8), axis=1)
            return audio, {"status": "accepted", "qa": {"audio_float32_sha256": audio_float32_sha256(audio)},
                           "provenance": {"request_id": request.request_id, "simulated_worker": True}}

    monkeypatch.setattr(temporal_proxy_data, "RenderWorker", Worker)
    monkeypatch.setattr(temporal_proxy_data, "NativeRendererFactory", lambda **_kwargs: object())
    actual_summary = native_fit.audit_summary

    def exercise_development(cases, current_plan, *, validation):
        result = actual_summary(cases, current_plan, validation=validation)
        # Force entry to the development branch; its real quality gates still fail against exact retrieval.
        return result if validation else {**result, "pass": True}

    monkeypatch.setattr(native_fit, "audit_summary", exercise_development)
    monkeypatch.setattr(balanced_native_fit, "make_plan", lambda *_args, **_kwargs: plan)
    output = tmp_path / "fit"
    native_fit.run_plan(pool, initial, retrieval, output, plan)
    result = native_fit.run_plan(pool, initial, retrieval, output, plan, execute=True)
    assert result["complete"] and result["committed_updates_per_arm"] == 4 and not result["matched_comparison_passed"]
    verifier = runpy.run_path(str(Path(__file__).parents[1] / "reports/scripts/verify_native_inverse_fit.py"))["verify"]
    report = verifier(output, pool, initial, retrieval, tmp_path / "qualification.json",
                      calibration_path=tmp_path / "calibration.json", input_cache_path=tmp_path / "cache.pt")
    assert report["offline_training_replay_exact"] and report["hard_audit_patches_reconstructed"]
    assert report["cached_targets_and_inputs_rescored"] and len(report["fit_cases"]) == 3 and len(report["validation_cases"]) == 6
    panels = report["descriptive_validation_comparison"]["panels"]
    assert all(panel["independent_source_states"] == 1 for panel in panels.values())
