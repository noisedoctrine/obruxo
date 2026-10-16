"""Opt-in expressive performances and four MIDI-driven connections over nested state."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from obruxo_data.hashing import canonical_json
from obruxo_data.render.base import FULL_EXPRESSION_RENDERER_ID
from obruxo_data.vital.full_contract import FullPresetContract

from .codec import PilotTarget
from .contracts import TrainingContract
from .nested_codec import NestedStateCodec
from .pilot import seeded_random

EXPRESSION_ROUTES = (("mod_wheel", "osc_1_pan"), ("aftertouch", "osc_2_pan"), ("slide", "osc_3_pan"), ("lift", "osc_1_pan"))


class ExpressionCodec(NestedStateCodec):
    model_family = "expression_nested_state"

    def __init__(self, asset_bank, lfo_bank):
        super().__init__(asset_bank, lfo_bank)
        self.schema = FullPresetContract(renderer_id=FULL_EXPRESSION_RENDERER_ID)
        self.contract = TrainingContract.create(self.schema, self.catalog)

    def to_dict(self):
        spec = super().to_dict()
        spec.update(codec="authored_expression_nested_state_v1", renderer_id=self.contract.renderer_id,
                    performance_bank_id=self.contract.performance_bank_id,
                    expression_curriculum={"policy": "four_expression_sources_on_unused_slots_v1",
                        "source_sha256": hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest(),
                        "connections": [list(value) for value in EXPRESSION_ROUTES], "master_volume_stored": 2500., "mpe_enabled": True},
                    scope=spec["scope"] + "_with_expression_performances_and_four_MIDI_sources")
        return spec

    def _original_routes(self, index):
        root = self.lineage_root(seeded_random(self.generator, self.roots.seed, index), index)
        return self.routing._authored(index, root, False)[0]

    def _route_target(self, target, routes):
        routing = self.routing._join(self.routing._curve_target(self._routing_target(target)), routes)
        return PilotTarget(routing.continuous + target.continuous[self._warp_start:], routing.categorical + target.categorical[self._routing_cat:])

    def _inject(self, target, index):
        routes = deepcopy(self._original_routes(index))
        slots = [slot for slot in range(63, -1, -1) if not routes[slot]["source"]][:4]
        base = self.routing.curve._base_target(self.routing._curve_target(self._routing_target(target)))
        settings = self.base._values(base)
        settings.update(volume=2500., mpe_enabled=1., pitch_bend_range=2.)
        for slot, (source, destination) in zip(slots, EXPRESSION_ROUTES, strict=True):
            routes[slot] = {"source": source, "destination": destination}
            settings.update({f"modulation_{slot + 1}_amount": .1, f"modulation_{slot + 1}_bipolar": 0.,
                             f"modulation_{slot + 1}_power": 0., f"modulation_{slot + 1}_stereo": 0., f"modulation_{slot + 1}_bypass": 0.})
        scalar = self.base.scalar._target(settings)
        actual = PilotTarget(scalar.continuous + target.continuous[self._ncont:], scalar.categorical + target.categorical[len(scalar.categorical):])
        return self._route_target(actual, routes)

    def sample_target(self, rng, index):
        return self.canonical_target(self._inject(super().sample_target(rng, index), index))

    def source_preset(self, target, index):
        parent = self._route_target(target, self._original_routes(index))
        document = json.loads(super().source_preset(parent, index))
        routes = deepcopy(document["settings"]["modulations"])
        slots = [slot for slot in range(63, -1, -1) if not routes[slot]["source"]][:4]
        for slot, (source, destination) in zip(slots, EXPRESSION_ROUTES, strict=True):
            routes[slot] = {"source": source, "destination": destination}
        document["settings"]["modulations"] = routes
        raw = canonical_json(document).encode()
        if self._inject(parent, index) != self.canonical_target(target):
            raise ValueError("saved expression target does not match its original source policy")
        self.schema.validate(document).require_valid()
        return raw
