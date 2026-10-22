"""Apply declared rendering constants at the prediction/preset boundary."""

from .render_profile import BASIC_PREROLL_FRAMES, basic_render_profile
from .reproducibility import FULL_PHASE_CONTROL, fixed_generator_phases

PROFILE_KEYS = ("render_profile", "render_phase_controls")


def artifact_rendering(saved):
    if not any(key in saved for key in PROFILE_KEYS):
        return {}
    profile, phases = saved.get("render_profile"), saved.get("render_phase_controls")
    frames = profile.get("preroll_frames") if isinstance(profile, dict) else None
    if (type(frames) is not int or frames not in (0, BASIC_PREROLL_FRAMES) or profile != basic_render_profile(frames)
            or not isinstance(phases, dict) or phases != FULL_PHASE_CONTROL
            or any(type(value) not in (int, float) for value in phases.values())):
        raise ValueError("inference render profile or phase controls are incomplete or unsupported")
    return {"render_profile": dict(profile), "render_phase_controls": dict(phases)}


def bind_rendering(model, saved):
    model.inference_rendering = artifact_rendering(saved)


def model_rendering(model):
    return artifact_rendering(getattr(model, "inference_rendering", {}))


def prediction_preset(model, codec, target):
    raw = codec.decode(target)
    return fixed_generator_phases(raw) if model_rendering(model) else raw
