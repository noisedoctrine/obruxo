"""Varied training observations have their own contract; the patch codec stays basic."""

from dataclasses import asdict, dataclass, replace
from hashlib import sha256
import inspect
import json
from pathlib import Path

from obruxo_data.hashing import canonical_json, canonical_sha256
from obruxo_data.midi import Performance
from obruxo_data.render.base import RenderRequest

from .performances import CAPABILITIES, initial_bank
from .pilot import seeded_random
from .profiled_sampler import ProfiledRecipe, ProfiledSampler

PATTERNS = ("held", "staccato", "melody", "chord")


def varied_spec():
    return {"policy": "basic_varied_inputs_v1", "pitch_range": [36, 84], "held_seconds": 2.,
            "views": ["informative_held_note", "balanced_alternate_pattern"], "alternate_patterns": list(PATTERNS[1:]),
            "pattern_assignment": "balanced_within_each_source_category_independent_shuffled_state_order",
            "source_sha256_lf": sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest(),
            "seed_function_sha256": sha256(inspect.getsource(seeded_random).encode()).hexdigest()}


def observation_contract(codec):
    payload = {"observations": "basic_varied_observations_v1", "patch_contract_id": codec.contract.contract_id,
               "performance_policy": varied_spec(), "split": "train", "role": "reference"}
    return {**payload, "contract_id": canonical_sha256(payload)}


def performance_for(seed, state_id, epoch, view, mode):
    rng = seeded_random("basic_varied_inputs_v1", seed, state_id, epoch, view)
    bpm, velocity = rng.choice((90, 110, 130, 150)), rng.choice((64, 80, 96, 112))
    performance = Performance(bpm=bpm)
    base = rng.randrange(36, 85 if mode in (0, 1) else 78)
    held_ticks = 16 * bpm  # Two seconds at 480 ticks per beat, including the slowest supported envelope.
    if mode == 0:
        notes = [(base, 0, held_ticks)]
    elif mode == 1:
        notes = [(base, index * 360, rng.choice((180, 240, 300))) for index in range(4)]
    elif mode == 2:
        notes = [(base + offset, index * 360, 300) for index, offset in enumerate((0, 4, 7, 2))]
    elif mode == 3:
        notes = [(base + offset, 0, held_ticks) for offset in (0, 4, 7)]
    else:
        raise ValueError("unknown varied-input pattern")
    for pitch, start, duration in notes:
        performance.add_note(pitch=pitch, velocity=velocity, start_tick=start, duration_ticks=duration)
    performance.end_tick += 120
    performance.validate(CAPABILITIES).require_valid()
    if canonical_sha256(performance.to_dict()) in {row.performance_id for row in initial_bank()}:
        raise ValueError("varied training observation overlaps a reserved contract performance")
    return performance


@dataclass(frozen=True, kw_only=True)
class VariedRecipe(ProfiledRecipe):
    observation_contract_json: str
    input_view: int
    input_pattern: str

    def render_request(self):
        contract = json.loads(self.observation_contract_json)
        payload = {key: value for key, value in contract.items() if key != "contract_id"}
        if (contract["contract_id"] != canonical_sha256(payload) or self.contract_id != contract["contract_id"]
                or contract["performance_policy"] != varied_spec() or contract["split"] != "train" or contract["role"] != "reference"
                or self.split != "train" or self.purpose != "train" or self.performance_role != "reference"):
            raise ValueError("varied observation contract changed or contains held-out data")
        request = super().render_request()
        if self.performance_id != canonical_sha256(request.performance.to_dict()):
            raise ValueError("varied observation performance identity changed")
        return request


class VariedSampler(ProfiledSampler):
    def __init__(self, pool, *, preroll_frames=32768, seed=0, policy="paired_views"):
        if policy != "paired_views":
            raise ValueError("varied basic inputs require paired_views sampling")
        super().__init__(pool, preroll_frames=preroll_frames, seed=seed, policy=policy)
        self.seed, self.observation_contract = seed, observation_contract(pool.codec)
        self.sampler_id = canonical_sha256({"sampler": "varied_basic_inputs_v1", "profiled_sampler": self.sampler_id,
                                           "observation_contract": self.observation_contract})
        groups = {}
        for entry in pool.entries:
            if pool.manifest.by_state[entry.state_id].split == "train":
                groups.setdefault(entry.target.categorical, []).append(entry.state_id)
        self.alternate_modes = {}
        for category, states in sorted(groups.items()):
            states = sorted(states)
            seeded_random("basic_varied_pattern_assignment_v1", seed, category).shuffle(states)
            self.alternate_modes.update({state: 1 + index % 3 for index, state in enumerate(states)})

    def recipe_at(self, sequence_id):
        base = super().recipe_at(sequence_id)
        epoch, view = sequence_id // self.epoch_size, sequence_id % 2
        mode = 0 if view == 0 else 1 + (self.alternate_modes[base.state_id] - 1 + epoch) % 3
        performance = performance_for(self.seed, base.state_id, epoch, view, mode)
        raw_request = RenderRequest.from_dict(json.loads(base.request_json))
        # A changed performance must carry its own end, rather than the original bank's fixed 1920 ticks.
        request = replace(raw_request, performance=performance, end_tick=None)
        return VariedRecipe(**{**asdict(base), "contract_id": self.observation_contract["contract_id"],
                               "performance_id": canonical_sha256(performance.to_dict()), "request_json": canonical_json(request.to_dict())},
                            observation_contract_json=canonical_json(self.observation_contract), input_view=view, input_pattern=PATTERNS[mode])
