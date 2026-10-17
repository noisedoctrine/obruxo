"""Explicit corpus label preparation; original presets remain the audio sources."""
from copy import deepcopy
import hashlib
from pathlib import Path

from obruxo_data.hashing import canonical_json, canonical_sha256

from .asset_codec import _content
from .curve_codec import CurveInputCodec

DISPLAY_METADATA = ("preset_style", "macro1", "macro2", "macro3", "macro4")


def corpus_input_policy():
    return {"policy": "canonical_display_and_unknown_disabled_assets_for_labels_only_v1",
            "source_sha256_lf": hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest(),
            "display_metadata": list(DISPLAY_METADATA), "unknown_active_assets": "reject",
            "unknown_disabled_assets": "reviewed_init_for_masked_labels_only", "scalar_changes": False,
            "original_audio_source": "required_preserve_literal_source_bytes"}


def prepare_corpus_labels(document, assets, schema):
    """Validate first, then change only known display fields or disabled unknown assets."""
    schema.validate(document).require_valid()
    prepared, changes = deepcopy(document), []
    template = schema.template()
    for name in DISPLAY_METADATA:
        if name in template and (name not in prepared or prepared[name] != template[name]):
            prepared[name] = deepcopy(template[name])
            changes.append({"path": "/" + name, "reason": "display_metadata"})
        elif name not in template and name in prepared:
            del prepared[name]
            changes.append({"path": "/" + name, "reason": "display_metadata"})
    settings = prepared["settings"]
    registered = {(reference.kind, reference.content_sha256) for reference in assets.catalog.references}

    def replace_unknown(kind, slot, enabled, value, default):
        if enabled != 0. or (kind, _content(kind, value)) in registered:
            return value
        changes.append({"path": f"/settings/wavetables/{slot}" if kind == "wavetable" else "/settings/sample",
                        "reason": "unregistered_disabled_asset", "kind": kind})
        return deepcopy(default)

    for slot, value in enumerate(settings["wavetables"]):
        settings["wavetables"][slot] = replace_unknown("wavetable", slot, settings[f"osc_{slot + 1}_on"], value, assets._waves[0])
    settings["sample"] = replace_unknown("sample", 0, settings["sample_on"], settings["sample"], assets._samples[0])
    schema.validate(prepared).require_valid()
    return prepared, changes


class CorpusInputCodec:
    """Same output model, with a separately identified corpus-label input policy."""

    def __init__(self, asset_bank, lfo_bank):
        self.parent = CurveInputCodec(asset_bank, lfo_bank)

    def __getattr__(self, name):
        return getattr(self.parent, name)

    @property
    def codec_id(self):
        return canonical_sha256(self.to_dict())

    def to_dict(self):
        spec = self.parent.to_dict()
        spec.update(codec="canonical_corpus_inputs_v1", corpus_input_policy=corpus_input_policy(),
                    scope=spec["scope"] + "_with_explicit_corpus_label_preparation")
        return spec

    def encode(self, raw):
        document, report = self.schema.loads(raw)
        report.require_valid()
        prepared, _ = prepare_corpus_labels(document, self.base, self.schema)
        return self.parent.encode(canonical_json(prepared).encode())
