"""Explicit native startup timing, separate from the predicted patch vocabulary."""

BASIC_PREROLL_FRAMES = 32768


def basic_render_profile(frames=BASIC_PREROLL_FRAMES):
    if type(frames) is not int or not 0 <= frames <= 44100 or frames % 128:
        raise ValueError("basic pre-roll must be whole 128-frame buffers within 44100 frames")
    return {"preroll_frames": frames, "midi_timing": "exact_sample_shift", "trim": "added_lead_only",
            "startup": "settled_controls_before_MIDI" if frames else "legacy_immediate_MIDI"}
