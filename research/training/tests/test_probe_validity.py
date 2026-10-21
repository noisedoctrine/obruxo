import copy
import json
from pathlib import Path

import pytest
import torch

from obruxo_training.probe_validity import clipping_penalty, probe_loss
from obruxo_training.spectral import SpectralTargets, spectral_loss


def rejected():
    report = json.loads((Path(__file__).parents[1] / "reports/data/native_inverse_prefix_summary.json").read_text())
    receipt = report["rejected_probe"]["receipt"]
    return receipt, receipt["renderer_identity"]


def test_actual_rejected_probe_has_explicit_penalty_that_increases_with_excess():
    receipt, identity = rejected()
    value = clipping_penalty(receipt, identity)
    assert value > 1000
    worse = copy.deepcopy(receipt)
    worse["qa"]["peak"] += .1
    assert clipping_penalty(worse, identity) > value
    assert probe_loss(None, receipt, None, identity) == (value, "clipping_penalty")


def test_other_render_failures_and_identity_or_state_mismatches_remain_fatal():
    receipt, identity = rejected()
    variants = []
    for key, value in (("finite", False), ("actual_frames", 1), ("peak", float("nan"))):
        changed = copy.deepcopy(receipt)
        changed["qa"][key] = value
        variants.append(changed)
    changed = copy.deepcopy(receipt)
    changed["diagnostic_codes"].append("audio.non_finite")
    variants.append(changed)
    changed = copy.deepcopy(receipt)
    changed["renderer_qa"]["native_state"]["scalar_lfo_route_readback_verified"] = False
    variants.append(changed)
    for changed in variants:
        with pytest.raises(RuntimeError, match="clipping-only"):
            clipping_penalty(changed, identity)
    with pytest.raises(RuntimeError, match="clipping-only"):
        clipping_penalty(receipt, {"different": "renderer"})


def test_valid_probe_keeps_the_exact_original_sound_loss():
    torch.set_num_threads(2)
    waveform = (.01 * torch.sin(torch.arange(8192) / 10)).numpy()
    import numpy as np
    audio = np.stack((waveform, .8 * waveform), axis=1)
    target = SpectralTargets()(torch.from_numpy(audio).T[None])
    expected = float(spectral_loss(SpectralTargets()(torch.from_numpy(audio * .8).T[None]), target)["loss"].item())
    assert probe_loss(audio * .8, {"status": "accepted"}, target, {}) == (expected, "native_spectral")
