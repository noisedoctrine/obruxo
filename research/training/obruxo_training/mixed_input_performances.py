"""Pair each original recording with both wider views, retaining exact origins."""

from dataclasses import dataclass, fields
import json
from pathlib import Path

from obruxo_data.hashing import canonical_json, canonical_sha256, sha256_bytes

from .input_performances import VariedSampler
from .profiled_sampler import ProfiledRecipe, ProfiledSampler


@dataclass(frozen=True, kw_only=True)
class MixedRecipe(ProfiledRecipe):
    observation_contract_json: str
    origin: str
    origin_sequence_id: int
    origin_recipe_id: str
    input_view: int

    def render_request(self):
        contract = json.loads(self.observation_contract_json)
        payload = {key: value for key, value in contract.items() if key != "contract_id"}
        if (canonical_sha256(payload) != self.contract_id or contract["contract_id"] != self.contract_id
                or contract["observations"] != "basic_mixed_original_and_varied_v1"
                or contract["split"] != "train" or contract["role"] != "reference"
                or self.split != "train" or self.purpose != "train" or self.performance_role != "reference"
                or self.origin not in {"original", "wide"}):
            raise ValueError("mixed observation contract changed or contains held-out data")
        request = super().render_request()
        if canonical_sha256(request.performance.to_dict()) != self.performance_id:
            raise ValueError("mixed performance identity changed")
        return request


class MixedSampler:
    def __init__(self, pool, *, preroll_frames=32768, seed=0, policy="paired_views"):
        if policy != "paired_views":
            raise ValueError("mixed observations require paired_views")
        self.pool, self.policy = pool, policy
        self.original = ProfiledSampler(pool, preroll_frames=preroll_frames, seed=seed, policy="state_round_robin")
        self.wide = VariedSampler(pool, preroll_frames=preroll_frames, seed=seed, policy=policy)
        self.old_by_state = {recipe.state_id: recipe for recipe in (self.original.recipe_at(index) for index in range(self.original.epoch_size))}
        payload = {"observations": "basic_mixed_original_and_varied_v1", "patch_contract_id": pool.codec.contract.contract_id,
                   "policy_sha256_lf": sha256_bytes(Path(__file__).read_bytes().replace(b"\r\n", b"\n")),
                   "origin_samplers": {"original": self.original.sampler_id, "wide": self.wide.sampler_id},
                   "schedule": ["original", "wide_held", "original", "wide_alternate"], "split": "train", "role": "reference"}
        self.observation_contract = {**payload, "contract_id": canonical_sha256(payload)}
        self.sampler_id = canonical_sha256({"sampler": "mixed_basic_inputs_v1", "observation_contract": self.observation_contract})

    @property
    def epoch_size(self):
        return self.wide.epoch_size * 2

    def recipe_at(self, sequence_id):
        if type(sequence_id) is not int or not 0 <= sequence_id < self.epoch_size:
            raise ValueError("mixed cached observations use exactly one frozen epoch")
        state_index, view = divmod(sequence_id, 4)
        wide = self.wide.recipe_at(2 * state_index + (view == 3))
        origin = self.old_by_state[wide.state_id] if view in (0, 2) else wide
        values = {field.name: getattr(origin, field.name) for field in fields(ProfiledRecipe)}
        values.update(sequence_id=sequence_id, contract_id=self.observation_contract["contract_id"])
        return MixedRecipe(**values, observation_contract_json=canonical_json(self.observation_contract),
                           origin="original" if view in (0, 2) else "wide", origin_sequence_id=origin.sequence_id,
                           origin_recipe_id=origin.recipe_id, input_view=view)
