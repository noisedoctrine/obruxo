import copy
import json
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from obruxo_training import production_choice_check as check  # noqa: E402
from obruxo_training.codec import PilotCodec  # noqa: E402
from obruxo_training.features import tensor_digest  # noqa: E402


class FrozenScores(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("controls", torch.full((1, 9), .4))

    def forward(self, features):
        indices = features["pitch"][:, 0].long()
        return {"continuous": self.controls.expand(len(indices), -1),
                "categorical": [40. * (2 * torch.nn.functional.one_hot(indices % size, size).float() - 1) for size in (3, 2)]}


@pytest.fixture(scope="module")
def bundle():
    model = FrozenScores()
    frontend = SimpleNamespace(basic_pitch=torch.nn.Linear(1, 1))
    items = [{"features": {"pitch": torch.full((3521,), float(index)), "timbre": torch.zeros(2, 96, 32)}} for index in range(6)]
    inputs = [{"state_id": str(index), "features_sha256": tensor_digest(item["features"])} for index, item in enumerate(items)]
    return model, frontend, PilotCodec(), items, inputs


def test_real_export_and_full_matrices_are_reconstructed_without_mutating_model_or_rng(bundle):
    model, frontend, *_rest = bundle
    before = tensor_digest(model.state_dict()), tensor_digest(frontend.basic_pitch.state_dict())
    rng, mode = torch.get_rng_state().clone(), model.training
    with check.refresh.cpu_settings():
        result = check.mathematical_evidence(*bundle)
    assert result["derivatives"]["passed"] and result["training_forward_matches_production"]
    assert len(result["derivatives"]["points"]) == 12
    assert len(result["initial_training_inputs"]) == 6
    assert before == (tensor_digest(model.state_dict()), tensor_digest(frontend.basic_pitch.state_dict()))
    assert torch.equal(rng, torch.get_rng_state()) and model.training == mode


@pytest.fixture
def report(tmp_path, bundle, monkeypatch):
    plan, audit = {"format": "fixture", "limits": check.LIMITS}, {"accepted_initialization": True}
    monkeypatch.setattr(check, "make_plan", lambda *_args: (plan, bundle, audit))
    monkeypatch.setattr(check, "MemoryMonitor", lambda: SimpleNamespace(peak_bytes=1024**2, errors=0, start=lambda: None, close=lambda: None))
    output = tmp_path / "check"
    prepared = check.run_check(None, None, None, None, None, output)
    assert not prepared["executed"]
    result = check.run_check(None, None, None, None, None, output, execute=True)
    assert result["passed"] and not result["native_direction_qualified"]
    return output, plan


def test_validator_recomputes_derivatives_exports_and_scales(report):
    output, _plan = report
    result = check.validate_check(output / "summary.json", None, None, None, None, None)
    assert result["scales"] and not result["native_direction_qualified"]
    with pytest.raises(ValueError, match="already started"):
        check.run_check(None, None, None, None, None, output, execute=True)


def test_verification_resources_and_overwrite_guard(report):
    output, _plan = report
    result = check.run_validation(output, None, None, None, None, None)
    assert result["verified"] and result["within_resource_limits"]
    assert result["archive_bytes_upper_bound"] < check.LIMITS["archive_bytes"]
    with pytest.raises(ValueError, match="already recorded"):
        check.run_validation(output, None, None, None, None, None)


@pytest.mark.parametrize("change", ("matrix", "scale", "export", "raw_scores", "rng_claim", "scope", "cache_binding", "resources"))
def test_validator_rejects_corruption_even_if_outer_pass_is_true(report, change):
    output, _plan = report
    path = output / "summary.json"
    report = json.loads(path.read_text())
    if change == "matrix":
        report["derivatives"]["points"][0]["heads"][0]["jacobian"][0][0] += .01
    elif change == "scale":
        report["fixed_scales"][0] *= 2
    elif change == "export":
        report["initial_training_inputs"][0]["hard_patch_sha256"] = "changed"
    elif change == "raw_scores":
        report["initial_training_inputs"][0]["raw_logits"][0][0] += 1
    elif change == "rng_claim":
        report["model_frontend_and_rng_unchanged"] = False
    elif change == "scope":
        report["native_direction_qualified"] = True
    elif change == "cache_binding":
        report["plan"] = {**copy.deepcopy(report["plan"]), "input_cache_sha256": "changed"}
        report["plan_id"] = check.canonical_sha256(report["plan"])
    else:
        report["peak_process_tree_rss_bytes"] = check.LIMITS["process_tree_bytes"] + 1
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="evidence"):
        check.validate_check(path, None, None, None, None, None)


def test_source_identity_binds_new_adapter_and_preserves_old_qualified_sources():
    sources = check.source_identity()
    assert sources["native_gradient.py"] and sources["production_controls.py"] and sources["native_inverse.py"]
    assert all(sources[name] == value for name, value in check.refresh.source_identity().items())
