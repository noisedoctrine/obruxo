from __future__ import annotations

from dataclasses import dataclass
import json
import math

from obruxo_data.hashing import canonical_json, canonical_sha256
from obruxo_data.vital import VitalPreset, VitalSchema

from .assets import AssetCatalog
from .contracts import TrainingContract
from .identity import DISPLAY_FIELDS, parse_document


# Bounds are in Vital's stored control units, not seconds, hertz or display units.
CONTINUOUS = (
    ("osc_1_level", 0.15, 0.35, 0.25), ("osc_1_pan", -0.7, 0.7, 0.0),
    ("osc_1_unison_detune", 0.5, 2.5, 1.0), ("filter_1_cutoff", 50.0, 120.0, 96.0),
    ("filter_1_resonance", 0.1, 0.6, 0.25), ("env_1_attack", 0.08, 0.6, 0.15),
    ("env_1_decay", 0.25, 1.0, 0.7), ("env_1_sustain", 0.25, 0.9, 0.7),
    ("env_1_release", 0.2, 0.65, 0.4),
)
CATEGORICAL = (("osc_1_unison_voices", (1, 2, 4)), ("filter_1_on", (0, 1)))


@dataclass(frozen=True)
class PilotTarget:
    continuous: tuple[float, ...]
    categorical: tuple[int, ...]

    def to_dict(self) -> dict:
        return {"continuous": list(self.continuous), "categorical": list(self.categorical)}

    @classmethod
    def from_dict(cls, value: dict) -> PilotTarget:
        if set(value) != {"continuous", "categorical"}:
            raise ValueError("invalid pilot target fields")
        return cls(tuple(value["continuous"]), tuple(value["categorical"]))


class PilotCodec:
    """Strict, limited proof-of-pipeline codec. Unsupported state is never projected away."""

    continuous = CONTINUOUS
    categorical = CATEGORICAL
    generator = "independent_authored_pilot_v1"

    def __init__(self, schema: VitalSchema | None = None):
        self.schema = schema or VitalSchema.load()
        self.catalog = AssetCatalog.reviewed_init(self.schema)
        self.contract = TrainingContract.create(self.schema, self.catalog)
        template = self.schema.load_init_document()
        template.update(author="OBRUXO", comments="Authored pilot v2", preset_name="OBRUXO pilot", synth_version="1.0.8")
        for name, lower, upper, default in CONTINUOUS:
            for value in (lower, upper, default):
                message = self.schema.parameters[name].validate_raw(value)
                if message:
                    raise ValueError(f"codec domain is outside schema: {name} {message}")
            template["settings"][name] = float(default)
        for name, options in CATEGORICAL:
            for value in options:
                if self.schema.parameters[name].validate_raw(value):
                    raise ValueError(f"invalid codec category: {name}")
            template["settings"][name] = float(options[0])
        VitalPreset(template, self.schema).validate().require_valid()
        self._template_json = canonical_json(template)

    @property
    def codec_id(self) -> str:
        return canonical_sha256(self.to_dict())

    def to_dict(self) -> dict:
        return {"version": 2, "codec": "authored_osc_filter_env_v2", "contract_id": self.contract.contract_id,
                "template_sha256": canonical_sha256(json.loads(self._template_json)),
                "continuous": [{"name": name, "minimum": lo, "maximum": hi, "inactive_default": default}
                               for name, lo, hi, default in CONTINUOUS],
                "categorical": [{"name": name, "values": list(values)} for name, values in CATEGORICAL],
                "normalization": "linear_in_stored_units", "inactive_policy": "fixed_template_value",
                "scope": "one_init_oscillator_optional_filter1_env1_no_routes_or_effects"}

    def validate_target(self, target: PilotTarget) -> None:
        if len(target.continuous) != len(self.continuous) or len(target.categorical) != len(self.categorical):
            raise ValueError("pilot target shape mismatch")
        if any(type(value) not in {int, float} or not math.isfinite(value) or not 0 <= value <= 1 for value in target.continuous):
            raise ValueError("continuous targets must be finite numbers in [0, 1]")
        if any(type(index) is not int or not 0 <= index < len(options)
               for index, (_, options) in zip(target.categorical, self.categorical, strict=True)):
            raise ValueError("categorical targets must be valid integer indices")

    def continuous_mask(self, target: PilotTarget) -> tuple[bool, ...]:
        self.validate_target(target)
        voices, filtered = target.categorical
        return (True, True, voices != 0, filtered != 0, filtered != 0, True, True, True, True)

    def canonical_target(self, target: PilotTarget) -> PilotTarget:
        mask = self.continuous_mask(target)
        values = tuple(float(value) if active else (default - lo) / (hi - lo)
                       for value, active, (_, lo, hi, default) in zip(target.continuous, mask, CONTINUOUS, strict=True))
        return PilotTarget(values, tuple(target.categorical))

    def decode(self, target: PilotTarget) -> bytes:
        target = self.canonical_target(target)
        document = json.loads(self._template_json)
        for (name, lo, hi, _), value in zip(CONTINUOUS, target.continuous, strict=True):
            document["settings"][name] = lo + float(value) * (hi - lo)
        for (name, options), index in zip(CATEGORICAL, target.categorical, strict=True):
            document["settings"][name] = float(options[index])
        return canonical_json(document).encode()

    def encode(self, raw: bytes) -> PilotTarget:
        parse_document(raw)  # Reject duplicate keys and nonfinite constants before float parsing.
        document = json.loads(raw.decode("utf-8-sig"))
        VitalPreset(document, self.schema).validate().require_valid()
        settings = document["settings"]
        values = tuple((settings[name] - lo) / (hi - lo) for name, lo, hi, _ in CONTINUOUS)
        try:
            categories = tuple(options.index(settings[name]) for name, options in CATEGORICAL)
        except ValueError as error:
            raise ValueError("preset uses a category outside the pilot vocabulary") from error
        target = PilotTarget(values, categories)
        self.validate_target(target)
        mask = self.continuous_mask(target)
        expected = json.loads(self._template_json)
        for (name, _, _, default), active in zip(CONTINUOUS, mask, strict=True):
            if not active and not math.isclose(settings[name], default, rel_tol=0, abs_tol=1e-12):
                raise ValueError(f"inactive pilot control must retain its fixed value: {name}")
            expected["settings"][name] = settings[name]
        for name, _ in CATEGORICAL:
            expected["settings"][name] = settings[name]
        source_state = {key: value for key, value in document.items() if key not in DISPLAY_FIELDS}
        expected_state = {key: value for key, value in expected.items() if key not in DISPLAY_FIELDS}
        if source_state != expected_state:
            raise ValueError("preset has fixed or unsupported state outside the pilot codec")
        return self.canonical_target(target)

    def preset(self, target: PilotTarget) -> VitalPreset:
        return VitalPreset(json.loads(self.decode(target)), self.schema)

    def default_target(self) -> PilotTarget:
        return self.canonical_target(PilotTarget(tuple((default - lo) / (hi - lo) for _, lo, hi, default in self.continuous),
                                                 (0,) * len(self.categorical)))


def load_codec(value: dict) -> PilotCodec:
    if value.get("codec") == "authored_expression_nested_state_v1":
        from .expression_codec import ExpressionCodec
        codec = ExpressionCodec(value["asset_bank"], value["lfo_bank"])
    elif value.get("codec") == "authored_nested_state_v1":
        from .nested_codec import NestedStateCodec
        codec = NestedStateCodec(value["asset_bank"], value["lfo_bank"])
    elif value.get("codec") == "authored_ordered_routing_v1":
        from .routing import OrderedRoutingCodec
        codec = OrderedRoutingCodec(value["asset_bank"], value["lfo_bank"])
    elif value.get("codec") == "authored_lfo_curves_v1":
        from .lfo_codec import LfoCurveCodec
        codec = LfoCurveCodec(value["asset_bank"], value["lfo_bank"])
    elif value.get("codec") == "authored_factory_assets_v1":
        from .asset_codec import FactoryAssetCodec
        codec = FactoryAssetCodec(value["asset_bank"])
    elif value.get("codec") == "authored_full_scalar_v1":
        from .full_scalar import FullScalarCodec
        codec = FullScalarCodec()
    elif value.get("codec") == "authored_alpha_v1":
        from .alpha import AlphaCodec
        codec = AlphaCodec(value["wavetable_bank"])
    else:
        codec = PilotCodec()
    if value != codec.to_dict():
        raise ValueError("unsupported or changed codec contract")
    return codec
