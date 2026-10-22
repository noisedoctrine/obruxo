import copy
import json
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from obruxo_data.hashing import canonical_sha256  # noqa: E402
from obruxo_training import category_scale_refresh as refresh  # noqa: E402
from obruxo_training.codec import PilotCodec  # noqa: E402
from obruxo_training.profiled_sampler import profile_metadata  # noqa: E402


class FrozenScores(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("controls", torch.full((1, 9), .5))

    def forward(self, features):
        indices = features["pitch"][:, 0].long()
        return {"continuous": self.controls.expand(len(indices), -1),
                "categorical": [20. * (2 * torch.nn.functional.one_hot(indices % size, size).float() - 1) for size in (3, 2)]}


@pytest.fixture(scope="module")
def evidence():
    model = FrozenScores()
    frontend = SimpleNamespace(basic_pitch=torch.nn.Linear(1, 1))
    codec = PilotCodec()
    items = [{"features": {"pitch": torch.full((3521,), float(index)), "timbre": torch.zeros(2, 96, 32)}} for index in range(refresh.CALIBRATION["observations"])]
    inputs = [{"state_id": str(index // 4), "input_view": index % 4, "features_sha256": str(index)} for index in range(len(items))]
    with refresh.cpu_settings():
        result = refresh.mathematical_evidence(model, frontend, codec, items, inputs)
    return (model, frontend, codec, items, inputs), result


def fixture_report(tmp_path, evidence, monkeypatch):
    bundle, mathematical = evidence
    paths = [tmp_path / name for name in ("model.pt", "checkpoint.pt", "qualification.json", "features.pt")]
    for path in paths:
        path.write_bytes(path.name.encode())
    initial, retrieval, qualification, cache = paths
    pool = SimpleNamespace(pool_id="fixture", manifest=SimpleNamespace(manifest_id="splits"))
    audit = {"model_optimizer_batch_and_global_rng_exact": True, "replayed_updates": 16000,
             "training_source_receipts_verified": 404, "native_renders_added": 0, "plan_id": "accepted-bootstrap",
             "all_eighty_audit_requests_reconstructed": True,
             "native_source_repeat_pairs_exact": 16, "hard_patch_request_exports_reconstructed": 8,
             "hard_categories_exact": True, "cached_vs_fresh_input_predictions_max_normalized_delta": 0.,
             "evidence_sha256": {"fit-000/model.pt": refresh.digest(initial), "fit-000/checkpoint.pt": refresh.digest(retrieval)}}
    plan = {"format": "basic_category_jacobian_check_v2", "pool_id": pool.pool_id, "manifest_id": pool.manifest.manifest_id,
            "calibration": copy.deepcopy(refresh.CALIBRATION), "limits": refresh.LIMITS.copy(),
            "initialization": refresh.INITIALIZATION.copy(),
            "derivative_gates": refresh.JACOBIAN_GATES.copy(), "source": refresh.source_identity(),
            "runtime": refresh.runtime_identity(), "initial_sha256": refresh.digest(initial), "retrieval_sha256": refresh.digest(retrieval),
            "qualification_sha256": refresh.digest(qualification), "bootstrap_audit_id": canonical_sha256(audit),
            "input_cache_sha256": refresh.digest(cache),
            "bootstrap_plan_id": audit["plan_id"], "initial_artifact": "fit-000/model.pt", "retrieval_artifact": "fit-000/checkpoint.pt",
            "inference": refresh.model_identity(*bundle[:3]),
            "training_inputs": bundle[-1], **profile_metadata(32768)}
    report = {"plan": plan, "plan_id": canonical_sha256(plan), "passed": True, "failure": None, "within_resource_limits": True,
              "native_requests": 0, "optimizer_updates": 0, "test_split_used": False, "qualified_for_release": False,
              "bootstrap_audit": audit, "elapsed_seconds": 1., "peak_process_tree_rss_bytes": 1024**2, "rss_sampling_errors": 0,
              "evidence_bytes_upper_bound": 8 * 1024**2, **copy.deepcopy(mathematical)}
    monkeypatch.setattr(refresh, "load_inputs", lambda *_args: bundle)
    path = tmp_path / "report.json"
    return pool, paths, path, report


def test_report_reconstructs_all_inputs_constants_and_choice_derivatives(tmp_path, evidence, monkeypatch):
    pool, paths, path, report = fixture_report(tmp_path, evidence, monkeypatch)
    path.write_text(json.dumps(report))
    rng, threads = torch.get_rng_state().clone(), torch.get_num_threads()
    result = refresh.validate_refresh(path, pool, *paths)
    assert result["diagnosis_format"] == "basic_category_jacobian_check_v2"
    assert result["scales"] == report["fixed_scales"] and result["initial_weights_changed"] is False
    assert torch.equal(rng, torch.get_rng_state()) and threads == torch.get_num_threads()
    assert len(report["derivatives"]["points"]) == refresh.CALIBRATION["observations"] * 2 * 16


@pytest.mark.parametrize("change", ("failed_gradient", "wrong_phase", "wrong_bootstrap_model", "changed_model", "changed_scale", "missing_evidence", "wrong_draw_count", "wrong_frontend", "changed_cache", "rejected_initializer"))
def test_report_refuses_failed_or_mismatched_evidence_even_when_outer_pass_is_true(tmp_path, evidence, monkeypatch, change):
    pool, paths, path, report = fixture_report(tmp_path, evidence, monkeypatch)
    if change == "failed_gradient":
        report["derivatives"]["passed"] = False
    elif change == "wrong_phase":
        report["plan"]["render_profile"]["preroll_frames"] = 0
    elif change == "wrong_bootstrap_model":
        report["bootstrap_audit"]["evidence_sha256"]["fit-000/model.pt"] = "wrong-model"
        report["plan"]["bootstrap_audit_id"] = canonical_sha256(report["bootstrap_audit"])
    elif change == "changed_model":
        paths[0].write_bytes(b"changed")
    elif change == "changed_scale":
        report["fixed_scales"][0] *= 2
    elif change == "missing_evidence":
        report.pop("derivatives")
    elif change == "wrong_draw_count":
        report["plan"]["calibration"]["draws_per_input_and_temperature"] = 8
    elif change == "changed_cache":
        paths[3].write_bytes(b"changed")
    elif change == "rejected_initializer":
        report["plan"]["initialization"] = {"experiment": "varied_basic_initialization_v1", "records": 808, "states": 404, "updates": 8000}
    else:
        report["plan"]["inference"]["frontend_sha256"] = "wrong-frontend"
    report["plan_id"] = canonical_sha256(report["plan"])
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="evidence|bootstrap replay|reconstructed"):
        refresh.validate_refresh(path, pool, *paths)


def test_training_loader_requires_data_proof_and_reconstructs_new_report(tmp_path, evidence, monkeypatch):
    from obruxo_training.category_calibration import load_calibration
    pool, paths, path, report = fixture_report(tmp_path, evidence, monkeypatch)
    path.write_text(json.dumps(report))
    initial, retrieval, qualification, cache = paths
    with pytest.raises(ValueError, match="checkpoint and audited input cache"):
        load_calibration(path, pool, initial, qualification)
    result = load_calibration(path, pool, initial, qualification, retrieval_path=retrieval, input_cache_path=cache)
    assert result["input_cache_sha256"] == refresh.digest(cache)
