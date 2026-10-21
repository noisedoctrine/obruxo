"""Keep invalid native trials visible and supply an explicit feasibility cost."""

import math

import torch

from .spectral import SpectralTargets, spectral_loss


ALLOWED_WARNINGS = {"audio.clipping", "audio.tail_truncation", "vital.runtime.canonicalization",
                    "vital.runtime.numeric_canonicalization", "vital.runtime.version_canonicalization"}


def probe_validity_spec():
    return {"format": "clipping_probe_feasibility_cost_v1", "valid_probe": "unchanged_A0_spectral_loss",
            "clipping_probe": "1000+100*(peak-1)^2+clipped_samples/(frames*channels)",
            "only_supported_rejection": "audio.clipping", "other_allowed_warnings": sorted(ALLOWED_WARNINGS - {"audio.clipping"}),
            "minimum_penalty": 1000., "threshold": 1., "target_and_final_QA": "strict_unchanged",
            "rejected_audio_returned_or_scored": False, "parameter_loss": False}


def clipping_penalty(receipt, renderer_identity):
    qa, codes = receipt.get("qa", {}), set(receipt.get("diagnostic_codes", []))
    provenance = {key: value for key, value in receipt.get("provenance", {}).items() if key != "request_id"}
    state = receipt.get("renderer_qa", {}).get("native_state", {})
    peak = qa.get("peak", float("nan"))
    valid = (receipt.get("status") == "rejected" and "audio.clipping" in codes and codes <= ALLOWED_WARNINGS
             and qa.get("finite") is True and math.isfinite(peak) and peak >= 1. and qa.get("clipping_count", 0) > 0
             and qa.get("actual_frames", 0) == qa.get("expected_frames") and qa.get("actual_frames", 0) > 0
             and qa.get("actual_channels") == qa.get("expected_channels") == 2
             and state.get("scalar_lfo_route_readback_verified") is True
             and state.get("applied_preset_sha256") == state.get("requested_preset_sha256") == receipt.get("controlled_sha256")
             and provenance == renderer_identity)
    if not valid:
        raise RuntimeError("native probe rejection is not a verified clipping-only trial")
    return 1000. + 100. * (peak - 1.) ** 2 + qa["clipping_count"] / (qa["actual_frames"] * qa["actual_channels"])


def probe_loss(audio, receipt, target, renderer_identity):
    if audio is None:
        return clipping_penalty(receipt, renderer_identity), "clipping_penalty"
    if receipt.get("status") != "accepted":
        raise RuntimeError("native probe audio must have an accepted receipt")
    value = float(spectral_loss(SpectralTargets()(torch.from_numpy(audio.copy()).T[None]), target)["loss"].item())
    return value, "native_spectral"
