import json

import pytest
import torch

from obruxo_training.codec import PilotCodec, PilotTarget
from obruxo_training.proxy import performance_inputs
from obruxo_training.performances import initial_bank
from obruxo_training.surrogate_backbones import BACKBONES, VitalSurrogate
from obruxo_training.surrogate_codec import SurrogateCodec
from obruxo_training.surrogate_pilot import control_tensor, widen_model
from obruxo_training.surrogate_pilot_data import perturb
from obruxo_training.surrogate_pilot_eval import candidate_categories


@pytest.fixture(scope="module")
def codecs():
    codec = PilotCodec()
    return tuple(SurrogateCodec(codec, expanded=expanded) for expanded in (False, True))


def test_binary_width_and_exact_basic_preset_reuse(codecs):
    basic, expanded = codecs
    assert (basic.width, expanded.width) == (13, 33)
    assert (len(basic.legal_choices()), len(expanded.legal_choices())) == (6, 1248)
    for choices in basic.legal_choices():
        original = PilotTarget(basic.defaults, choices)
        lifted = expanded.from_basic(original)
        assert expanded.decode(lifted) == basic.decode(original)
        assert basic.category_vector(choices)[-1] == choices[-1]
    assert len(set(candidate_categories(expanded))) == 24


def test_expanded_activity_and_perturbations(codecs):
    _, expanded = codecs
    requested = PilotTarget(tuple(.3 for _ in expanded.continuous), (0, 1, 1, 2, 1, 3, 1, 1))
    settings = json.loads(expanded.decode(requested))["settings"]
    assert settings["filter_1_blend"] == 2.
    assert settings["wavetables"][0]["name"] == "OBRUXO sine"
    assert settings["wavetables"][1]["name"] == "OBRUXO square"
    assert settings["osc_2_transpose"] == -12.
    assert settings["reverb_on"] == 1.
    for i in range(len(expanded.continuous) + len(expanded.categorical)):
        changed, field = perturb(expanded, requested, i, "test-root")
        assert expanded.decode(changed) != expanded.decode(requested)
        assert field != "reverb_on"


@pytest.mark.parametrize("backbone", BACKBONES)
def test_widening_preserves_basic_outputs_and_input_gradients(codecs, backbone):
    basic, expanded = codecs
    previous = VitalSurrogate(backbone, basic.width)
    widened = VitalSurrogate(backbone, expanded.width)
    widen_model(widened, previous.state_dict())
    patch = basic.from_basic(basic.basic.default_target())
    original = control_tensor(basic, [patch], "cpu")
    lifted = control_tensor(expanded, [expanded.from_basic(patch)], "cpu").requires_grad_()
    played = performance_inputs(initial_bank()[0].performance())[None]
    times = tuple(torch.tensor([[0., .3]]) for _ in range(4))
    lengths = torch.tensor([40000])
    before = previous(original, played, times, lengths)
    after = widened(lifted, played, times, lengths)
    for first, second in zip((*before["magnitudes"], before["envelope"]),
                             (*after["magnitudes"], after["envelope"]), strict=True):
        torch.testing.assert_close(first, second)
    after["envelope"].sum().backward()
    assert torch.isfinite(lifted.grad).all() and lifted.grad.abs().sum().item() > 0.
