import copy
import itertools

import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from obruxo_training.codec import PilotCodec, PilotTarget  # noqa: E402
from obruxo_training.features import tensor_digest  # noqa: E402
from obruxo_training.spectral import (SpectralTargets, prepare_spectral_target, require_spectral_contract, spectral_loss,  # noqa: E402
                                      spectral_spec, target_bytes)
from obruxo_training.spectral_controls import BasicControlAdapter, temperature_at  # noqa: E402


@pytest.fixture(autouse=True)
def threads():
    torch.set_num_threads(2)


def audio(samples=8192):
    time = torch.arange(samples) / 44100
    wave = .1 * torch.sin(2 * torch.pi * 440 * time)
    return torch.stack((wave, .6 * wave))[None]


def test_identity_silence_gain_stereo_and_detuning():
    measure = SpectralTargets()
    source = audio()
    targets = measure(source)
    assert torch.equal(spectral_loss(targets, targets)["loss"], torch.zeros(1))
    silence = measure(torch.zeros_like(source))
    assert spectral_loss(silence, silence)["loss"].item() == 0
    for changed in (source * .5, source.flip(1), torch.roll(source, 256, -1), audio() * torch.cos(torch.arange(8192) * .002)):
        terms = spectral_loss(measure(changed), targets)
        assert torch.isfinite(terms["loss"]).all() and terms["loss"].item() > 0
    assert spectral_loss(measure(source.mean(1, keepdim=True).expand_as(source)), targets)["side"].item() > 0
    assert torch.isfinite(spectral_loss(targets, silence)["loss"]).all()
    assert target_bytes(targets) > 0 and spectral_spec()["pooling"] is None


def test_padding_masks_match_individual_clip_measurements_and_losses():
    measure = SpectralTargets()
    original = audio(4096)
    padded = torch.cat((original, torch.full_like(original, 100.)), dim=-1)
    expected, batched = measure(original), measure(padded, torch.tensor([4096]))
    for left, right, mask in zip(expected["magnitudes"], batched["magnitudes"], expected["masks"], strict=True):
        assert torch.allclose(left[..., mask[0]], right[..., :left.shape[-1]][..., mask[0]])
    def quieter(target):
        return {"magnitudes": tuple(value * .7 for value in target["magnitudes"]), "envelope": target["envelope"] * .7}
    assert torch.allclose(spectral_loss(quieter(expected), expected)["loss"], spectral_loss(quieter(batched), batched)["loss"])
    for bad in (torch.tensor([0]), torch.tensor([9000]), torch.tensor([4096.])):
        with pytest.raises(ValueError, match="lengths"):
            measure(padded, bad)


@pytest.mark.parametrize("silent", [False, True])
def test_prepared_reference_preserves_every_loss_and_prediction_gradient(silent):
    source = audio().expand(2, -1, -1).clone()
    if silent:
        source.zero_()
    target = SpectralTargets()(source, torch.tensor([8192, 4096]))
    prepared = prepare_spectral_target(target)
    results, gradients = [], []
    for reference in (target, prepared):
        magnitudes = tuple((value * .7 + .001).requires_grad_() for value in target["magnitudes"])
        envelope = (target["envelope"] * .8 + .001).requires_grad_()
        prediction = {"magnitudes": magnitudes, "envelope": envelope}
        result = spectral_loss(prediction, reference)
        result["loss"].sum().backward()
        results.append(result)
        gradients.append([value.grad for value in (*magnitudes, envelope)])
    assert all(torch.equal(results[0][key], results[1][key]) for key in results[0])
    assert all(torch.equal(left, right) for left, right in zip(*gradients, strict=True))
    assert "_loss_reference" not in target and target_bytes(prepared) > target_bytes(target)
    invalid = {**target, "envelope": target["envelope"].clone().requires_grad_()}
    with pytest.raises(ValueError, match="fixed measurements"):
        prepare_spectral_target(invalid)


@pytest.mark.parametrize("voices,filtered", tuple(itertools.product(range(3), range(2))))
def test_all_hard_choices_match_codec_and_mask_inactive_gradients(voices, filtered):
    codec, adapter = PilotCodec(), BasicControlAdapter()
    values = torch.full((1, 9), .7, requires_grad=True)
    scores = [torch.nn.functional.one_hot(torch.tensor([voices]), 3).float(),
              torch.nn.functional.one_hot(torch.tensor([filtered]), 2).float()]
    encoded = adapter(values, scores, sample=False)
    target = codec.canonical_target(PilotTarget(tuple(values.detach()[0].tolist()), (voices, filtered)))
    assert encoded[0, :9].tolist() == pytest.approx(target.continuous)
    assert codec.decode(adapter.export(values, scores)[0]) == codec.decode(target)
    encoded.sum().backward()
    assert values.grad[0].tolist() == [float(value) for value in codec.continuous_mask(target)]
    assert torch.equal(encoded, adapter(values, scores, sample=False))


class GradientFixture(nn.Module):
    """Connectivity fixture only; never exported or described as a qualified sound helper."""

    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(14, 5)

    def forward(self, controls, target):
        scale = torch.nn.functional.softplus(self.projection(controls)) + .1
        spectra = tuple(value * scale[:, i, None, None, None] for i, value in enumerate(target["magnitudes"]))
        envelope = torch.cat((target["envelope"][:, :2] * scale[:, 3, None, None],
                              target["envelope"][:, 2:] * scale[:, 4, None, None]), dim=1)
        return {"magnitudes": spectra, "envelope": envelope}


def test_sound_only_gradients_reach_continuous_and_gumbel_heads_and_freeze_helper():
    torch.manual_seed(4)
    adapter, helper = BasicControlAdapter(), GradientFixture().eval().requires_grad_(False)
    before = tensor_digest(helper.state_dict())
    values = torch.full((8, 9), .6, requires_grad=True)
    scores = [torch.tensor([[0., 4., 3.]] * 8, requires_grad=True), torch.tensor([[0., 4.]] * 8, requires_grad=True)]
    target = SpectralTargets()(audio().expand(8, -1, -1))
    controls = adapter(values, scores)
    assert (controls[:, 9:] == controls[:, 9:].round()).all()
    spectral_loss(helper(controls, target), target)["loss"].mean().backward()
    assert torch.isfinite(values.grad).all() and (values.grad.abs().sum(0) > 0).all()
    assert all(torch.isfinite(scores_i.grad).all() and (scores_i.grad.abs().sum(0) > 0).all() for scores_i in scores)
    assert before == tensor_digest(helper.state_dict()) and all(p.grad is None for p in helper.parameters())


def test_exact_gumbel_optimizer_resume_and_contract_rejection(tmp_path):
    torch.manual_seed(12)
    adapter, helper = BasicControlAdapter(), GradientFixture().eval().requires_grad_(False)
    target = SpectralTargets()(audio())
    initial = [torch.full((1, 9), .1), torch.zeros(1, 3), torch.zeros(1, 2)]
    params = [nn.Parameter(value.clone()) for value in initial]
    optimizer = torch.optim.Adam(params, lr=.01)
    contract = {"adapter": adapter.contract(), "helper_sha256": tensor_digest(helper.state_dict()), "steps": 6}
    def update(parameters, opt, step):
        opt.zero_grad(set_to_none=True)
        encoded = adapter(parameters[0].sigmoid(), parameters[1:], temperature=temperature_at(step, 6))
        loss = spectral_loss(helper(encoded, target), target)["loss"].mean()
        loss.backward()
        opt.step()
        return encoded.detach().clone(), loss.item()
    for step in range(3):
        update(params, optimizer, step)
    saved = {"contract": contract, "params": [p.detach().clone() for p in params], "optimizer": copy.deepcopy(optimizer.state_dict()),
             "rng": torch.get_rng_state(), "step": 3, "helper": helper.state_dict(), "target": target}
    torch.save(saved, tmp_path / "checkpoint.pt")
    expected = [update(params, optimizer, step) for step in range(3, 6)]
    restored = torch.load(tmp_path / "checkpoint.pt", weights_only=True)
    require_spectral_contract(restored["contract"], contract)
    recovered = [nn.Parameter(value) for value in restored["params"]]
    resumed_optimizer = torch.optim.Adam(recovered, lr=.01)
    resumed_optimizer.load_state_dict(restored["optimizer"])
    torch.set_rng_state(restored["rng"])
    actual = [update(recovered, resumed_optimizer, step) for step in range(restored["step"], 6)]
    assert all(torch.equal(a[0], b[0]) and a[1] == b[1] for a, b in zip(expected, actual, strict=True))
    assert all(torch.equal(a, b) for a, b in zip(params, recovered, strict=True))
    assert optimizer.state_dict()["param_groups"] == resumed_optimizer.state_dict()["param_groups"]
    for index, state in optimizer.state_dict()["state"].items():
        assert all(torch.equal(value, resumed_optimizer.state_dict()["state"][index][key]) for key, value in state.items())
    for key in ("helper_sha256", "adapter", "steps"):
        with pytest.raises(ValueError, match="contract changed"):
            require_spectral_contract({**contract, key: None}, contract)


def test_invalid_scores_audio_and_schedule_rejected():
    adapter = BasicControlAdapter()
    with pytest.raises(ValueError):
        adapter(torch.full((1, 9), float("nan")), [torch.zeros(1, 3), torch.zeros(1, 2)])
    with pytest.raises(ValueError):
        adapter(torch.ones(1, 9), [torch.zeros(1, 4), torch.zeros(1, 2)])
    with pytest.raises(ValueError):
        SpectralTargets()(audio().double())
    with pytest.raises(ValueError):
        temperature_at(6, 6)
    assert temperature_at(0, 6) == 1. and temperature_at(5, 6) == .5
