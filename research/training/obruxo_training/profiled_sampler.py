"""Keep source identities/schedules while explicitly controlling native rendering."""

from dataclasses import asdict, dataclass, replace
import json

from obruxo_data.hashing import canonical_json, canonical_sha256

from .codec import PilotCodec
from .contracts import RenderRecipe
from .pilot import PilotSampler
from .render_profile import BASIC_PREROLL_FRAMES, basic_render_profile
from .reproducibility import FULL_PHASE_CONTROL, fixed_generator_phases


def profile_metadata(preroll_frames):
    if type(preroll_frames) is not int:
        raise ValueError("bootstrap pre-roll must be an integer")
    if preroll_frames == 0:
        return {}
    if preroll_frames != BASIC_PREROLL_FRAMES:
        raise ValueError("basic bootstrap supports only the qualified 32768-frame render profile")
    return {"render_profile": basic_render_profile(preroll_frames), "render_phase_controls": dict(FULL_PHASE_CONTROL)}


@dataclass(frozen=True)
class ProfiledRecipe(RenderRecipe):
    render_profile_json: str
    phase_policy: str = "fixed_oscillator_1_3_sampler_random_phase_v1"

    def render_request(self):
        profile = json.loads(self.render_profile_json)
        if (profile != basic_render_profile(BASIC_PREROLL_FRAMES)
                or self.phase_policy != "fixed_oscillator_1_3_sampler_random_phase_v1"):
            raise ValueError("unsupported profiled source recipe")
        original = super().render_request()
        if original.preroll_frames:
            raise ValueError("source recipe already contains a render lead")
        return replace(original, preset_json=fixed_generator_phases(original.preset_json.encode()).decode(),
                       preroll_frames=BASIC_PREROLL_FRAMES)


class ProfiledSampler:
    """Source/lineage fields describe the original patch; the request is its controlled copy."""

    def __init__(self, pool, *, preroll_frames, **kwargs):
        if type(pool.codec) is not PilotCodec:
            raise ValueError("qualified bootstrap render profile requires the exact basic codec")
        self.metadata = profile_metadata(preroll_frames)
        if not self.metadata:
            raise ValueError("use the legacy sampler for immediate MIDI")
        self.base = PilotSampler(pool, **kwargs)
        self.pool, self.policy = pool, self.base.policy
        self.sampler_id = canonical_sha256({"sampler": "profiled_basic_source_v1", "base_sampler_id": self.base.sampler_id, **self.metadata})

    @property
    def epoch_size(self):
        return self.base.epoch_size

    def recipe_at(self, sequence_id):
        recipe = self.base.recipe_at(sequence_id)
        return ProfiledRecipe(**asdict(recipe), render_profile_json=canonical_json(self.metadata["render_profile"]))


def training_sampler(pool, config):
    kwargs = {"seed": config.seed, "policy": config.sampling_policy}
    if config.input_performances == "varied":
        from .input_performances import VariedSampler
        return VariedSampler(pool, preroll_frames=config.preroll_frames, **kwargs)
    if config.preroll_frames:
        return ProfiledSampler(pool, preroll_frames=config.preroll_frames, **kwargs)
    return PilotSampler(pool, **kwargs)
