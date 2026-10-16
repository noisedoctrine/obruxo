"""Train-only bridge to the existing Era2 W8D16 phase/gain representation."""
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from obruxo_data.hashing import canonical_sha256
from obruxo_data.vital.full_contract import FullPresetContract

from .identity import PresetIdentity, require_digest
from .splits import SplitManifest

ERA2 = Path(__file__).resolve().parents[2] / "experiments/lfo_representation/era2/code"
sys.path.insert(0, str(ERA2))
from lfo_era2.assets import ReconstructionAssets  # noqa: E402
from lfo_era2.component_ladder import ComponentEncoding, _construct_assets, _encode_decode, _screening_spec, decode_component_encoding  # noqa: E402
from lfo_era2.dataset import LfoShape, sample_shape  # noqa: E402

ALGORITHM = "era2_w8d16_phase_gain_beam4_final_clip_97_v1"
SPEC = _screening_spec("construction_policy", "BestOverallRepair", "PhaseAndResidualGain")
INDEX_SIZES = (32,) + (8,) * 16
RESOLUTION = 97


def runtime_sources():
    paths = [Path(__file__), *(ERA2 / "lfo_era2").glob("*.py")]
    research = Path(__file__).resolve().parents[2]
    return {path.relative_to(research).as_posix(): hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest() for path in sorted(paths)}


def sample_lfo(value):
    shape = LfoShape.from_json(value)
    result = sample_shape(shape, RESOLUTION, x_grid_mode="inclusive")
    if not np.isfinite(result).all() or np.any((result < 0) | (result > 1)):
        raise ValueError("LFO samples must be finite and within [0,1]")
    return result


def curve_to_lfo(values, *, name="OBRUXO Era2 curve"):
    values = np.asarray(values, dtype=np.float32)
    if values.shape != (RESOLUTION,) or not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError("decoded LFO requires 97 finite values within [0,1]")
    points = np.column_stack((np.linspace(0, 1, RESOLUTION, dtype=np.float32), values))
    return {"name": name, "num_points": RESOLUTION, "points": points.reshape(-1).tolist(),
            "powers": [0.] * RESOLUTION, "smooth": False}


@dataclass(frozen=True)
class LfoCode:
    indices: tuple[int, ...]
    phases: tuple[float, ...]
    gains: tuple[float, ...]

    def __post_init__(self):
        for name in ("indices", "phases", "gains"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if len(self.indices) != 17 or any(type(index) is not int or not 0 <= index < size for index, size in zip(self.indices, INDEX_SIZES)):
            raise ValueError("LFO code requires a base index and sixteen legal residual indices")
        if len(self.phases) != 17 or any(type(value) not in {float, int} or not np.isfinite(value) or not 0 <= value < 1 for value in self.phases):
            raise ValueError("LFO code requires seventeen finite phases within [0,1)")
        if len(self.gains) != 16 or any(type(value) not in {float, int} or not np.isfinite(value) or not -2 <= value <= 2 for value in self.gains):
            raise ValueError("LFO code requires sixteen finite residual gains within [-2,2]")

    def to_dict(self):
        return {key: list(value) for key, value in asdict(self).items()}

    @classmethod
    def from_dict(cls, value):
        return cls(*(tuple(value[key]) for key in ("indices", "phases", "gains")))


class LfoBank:
    def __init__(self, document):
        if document.get("format") != "obruxo_lfo_bank_v1" or document.get("bank_id") != canonical_sha256(document["bank"]):
            raise ValueError("LFO bank content or format changed")
        bank = document["bank"]
        if (bank.get("algorithm") != ALGORITHM or bank.get("source_hash_policy") != "lf_normalized_bytes"
                or bank.get("runtime_sources") != runtime_sources()):
            raise ValueError("LFO bank uses an incompatible decoder or source implementation")
        if canonical_sha256(bank.get("spec")) != canonical_sha256(asdict(SPEC)):
            raise ValueError("LFO bank representation policy changed")
        manifest = SplitManifest.from_dict(bank["split_manifest"])
        members = bank["fitted_rows"]
        if type(bank["maximum_curves"]) is not int or not 1 <= bank["maximum_curves"] <= 4096:
            raise ValueError("invalid LFO fitting budget")
        if not members or len(members) > bank["maximum_curves"]:
            raise ValueError("LFO fitting members are empty or exceed the declared budget")
        keys = set()
        for member in members:
            entry = manifest.by_state.get(member["state_id"])
            if entry is None or entry.split != "train" or entry.lineage_id != member["lineage_id"]:
                raise ValueError("LFO dictionary contains an unregistered or held-out lineage")
            if type(member["slot"]) is not int or not 1 <= member["slot"] <= 8:
                raise ValueError("LFO fitting slot must be within 1..8")
            for key in ("raw_sha256", "curve_sha256"):
                require_digest(member[key])
            key = member["state_id"], member["slot"]
            if key in keys:
                raise ValueError("duplicate LFO fitting member")
            keys.add(key)
        base = np.asarray(bank["base_dictionary"], dtype=np.float32)
        residuals = np.asarray(bank["residual_dictionaries"], dtype=np.float32)
        if base.shape != (32, RESOLUTION) or residuals.shape != (16, 8, RESOLUTION):
            raise ValueError("LFO bank dictionary dimensions changed")
        if not np.isfinite(base).all() or not np.isfinite(residuals).all() or np.any((base < 0) | (base > 1)):
            raise ValueError("LFO dictionary values must be finite; base curves must lie within [0,1]")
        if np.any(residuals[:, 0] != 0):
            raise ValueError("every LFO residual layer must retain its zero NoOp atom")
        # Copy the document so caller mutations cannot alter a bank after validation.
        self._document = json.loads(json.dumps(document))
        self.assets = ReconstructionAssets(base.copy(), [layer.copy() for layer in residuals])

    @property
    def bank_id(self):
        return self._document["bank_id"]

    def to_dict(self):
        return json.loads(json.dumps(self._document))

    def encode(self, values):
        values = np.asarray(values, dtype=np.float32)
        if values.ndim == 1:
            values = values[None]
        if values.ndim != 2 or values.shape[1] != RESOLUTION or not len(values) or not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
            raise ValueError("LFO encoding needs finite rows of 97 values within [0,1]")
        encoding, reconstructed, _, _ = _encode_decode(SPEC, values, self.assets, backend="numpy", chunk_size=16,
                                                        progress=None, progress_label="training_bridge")
        codes = []
        for row in range(len(values)):
            indices = (int(encoding.base_index[row]), *(int(layer[row]) for layer in encoding.residual_layer_indices))
            phases = (float(encoding.base_phase[row] % 1), *(float(layer[row] % 1) for layer in encoding.residual_layer_phases))
            gains = tuple(float(layer[row]) for layer in encoding.residual_layer_gains)
            codes.append(LfoCode(indices, phases, gains))
        return codes, reconstructed

    def decode(self, codes):
        if not codes or any(not isinstance(code, LfoCode) for code in codes):
            raise ValueError("LFO decoding needs validated codes")
        indices = np.asarray([code.indices for code in codes], dtype=np.int32)
        phases = np.asarray([code.phases for code in codes], dtype=np.float32)
        gains = np.asarray([code.gains for code in codes], dtype=np.float32)
        encoding = ComponentEncoding(indices[:, 0], phases[:, 0], np.ones(len(codes), dtype=np.float32),
                                     [indices[:, layer + 1] for layer in range(16)], [phases[:, layer + 1] for layer in range(16)],
                                     [gains[:, layer] for layer in range(16)])
        return decode_component_encoding(SPEC, encoding, self.assets)


def fit_lfo_bank(presets, manifest, *, maximum_curves=256, progress=None):
    """Validate every supplied state before fitting; held-out input is an error."""
    if type(maximum_curves) is not int or not 1 <= maximum_curves <= 4096:
        raise ValueError("LFO fit budget must be within 1..4096 curves")
    schema, records, seen = FullPresetContract(), [], set()
    source_hashes = runtime_sources()
    for raw in presets:
        identity = PresetIdentity.from_bytes(raw)
        entry = manifest.by_state.get(identity.state)
        if entry is None or entry.split != "train":
            raise ValueError("LFO fitting requires registered training states only")
        if identity.state in seen:
            raise ValueError("duplicate preset in LFO fitting population")
        if len(seen) >= 4096:
            raise ValueError("LFO fitting population exceeds 4096 preset budget")
        seen.add(identity.state)
        document, report = schema.loads(raw)
        report.require_valid()
        for slot, shape in enumerate(document["settings"]["lfos"], 1):
            values = sample_lfo(shape)
            curve_id = canonical_sha256(values.tolist())
            records.append(({"state_id": identity.state, "lineage_id": entry.lineage_id, "slot": slot,
                             "raw_sha256": identity.raw, "curve_sha256": curve_id}, values))
    if not records:
        raise ValueError("no registered training LFOs to fit")
    # Duplicate shapes get one vote. Selection is stable under input-file order.
    unique = {}
    for member, values in sorted(records, key=lambda pair: (pair[0]["state_id"], pair[0]["slot"])):
        unique.setdefault(member["curve_sha256"], (member, values))
    selected = [unique[key] for key in sorted(unique)[:maximum_curves]]
    curves = np.stack([values for _, values in selected])
    assets = _construct_assets(SPEC, curves, np.zeros(len(curves), dtype=np.int32), backend="numpy", chunk_size=16, progress=progress)
    if source_hashes != runtime_sources():
        raise RuntimeError("LFO implementation changed while fitting")
    bank = {"algorithm": ALGORITHM, "spec": asdict(SPEC), "runtime_sources": source_hashes, "source_hash_policy": "lf_normalized_bytes",
            "split_manifest": manifest.to_dict(),
            "maximum_curves": maximum_curves, "source_presets": len(seen), "source_curve_rows": len(records), "unique_training_curves": len(unique),
            "fitted_rows": [member for member, _ in selected], "base_dictionary": assets.base_dictionary.tolist(),
            "residual_dictionaries": [layer.tolist() for layer in assets.residual_layer_dictionaries]}
    return LfoBank({"format": "obruxo_lfo_bank_v1", "bank_id": canonical_sha256(bank), "bank": bank})
