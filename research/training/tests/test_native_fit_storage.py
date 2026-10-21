import pytest
import torch

from obruxo_training.native_fit_storage import target_tensor_bytes
from obruxo_training.spectral import SpectralTargets, target_bytes


@pytest.mark.parametrize("samples", [256, 154350, 176400])
def test_planned_target_bytes_match_actual_saved_tensor_layout(samples):
    torch.set_num_threads(2)
    measured = SpectralTargets()(torch.zeros(1, 2, samples))
    assert target_tensor_bytes(samples) == target_bytes(measured)
