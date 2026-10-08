from __future__ import annotations

from collections import Counter, defaultdict
import csv
import json
from pathlib import Path

from .common import Context, digest, file_hash, finish_report, read_json, write_csv, write_json
from .identity import RULE, identities

CATEGORIES = ("Bass", "Pad", "Pluck", "Lead")


def performance(name: str):
    from obruxo_data.midi import Performance
    result = Performance(ticks_per_beat=480, bpm=120)
    if name in ("held_c4", "held_c3"):
        result.add_note(pitch=60 if name == "held_c4" else 48, velocity=100, start_tick=0, duration_ticks=1920)
    elif name == "staccato_c4":
        for start in (0, 480, 960, 1440):
            result.add_note(pitch=60, velocity=100, start_tick=start, duration_ticks=240)
    else:
        raise ValueError(f"unknown performance: {name}")
    result.end_tick = 1920
    return result


def control_names(preset) -> list[str]:
    settings = preset.to_dict()["settings"]
    names = []
    for component, slots, control in (("osc", range(1, 4), "level"), ("filter", range(1, 3), "cutoff")):
        active = next((slot for slot in slots if settings.get(f"{component}_{slot}_on", 0) != 0), None)
        if active is not None:
            names.append(f"{component}_{active}_{control}")
    return names


def perturb(preset, name: str):
    from obruxo_data.vital import VitalPreset
    spec = preset.schema.parameters[name]
    if spec.is_discrete or spec.minimum == spec.maximum:
        raise ValueError("perturbation requires a variable continuous parameter")
    original = preset.get_raw(name)
    delta = 0.05 * (spec.maximum - spec.minimum)
    value = original + delta if original + delta <= spec.maximum else original - delta
    changed = VitalPreset(preset.to_dict(), preset.schema)
    changed.set_raw(name, value)
    changed.validate().require_valid()
    before, after = preset.to_dict(), changed.to_dict()
    after["settings"][name] = original
    if before != after:
        raise ValueError("perturbation modified more than one parameter")
    return changed, {"name": name, "original_raw": original, "changed_raw": value, "raw_fraction": 0.05}


def load_fixture(row: dict):
    from obruxo_data.vital import VitalPreset
    source = Path(row["source_path"])
    if file_hash(source) != row["source_sha256"]:
        raise ValueError("source changed since fixture selection; prepare a new run")
    preset = VitalPreset.load(source)
    preset.validate().require_valid()
    # Public load/save must preserve the document, including unknown version fields.
    if json.loads(source.read_text(encoding="utf-8-sig")) != preset.to_dict():
        raise ValueError("preset loading changed stored state")
    return preset


def load_manifest(path: Path) -> dict:
    state_path = path.parent / "status.json"
    if state_path.exists() and read_json(state_path).get("state") != "complete":
        raise ValueError("fixture preparation is not complete; do not consume a stale manifest")
    manifest = read_json(path)
    if manifest.get("schema") != "quick_fixtures_v1" or not manifest.get("complete"):
        raise ValueError("requires a complete, frozen eight-fixture manifest")
    if digest(manifest["fixtures"]) != manifest["fixtures_hash"]:
        raise ValueError("fixture manifest was edited after freezing")
    if Counter(row["category"] for row in manifest["fixtures"]) != Counter({name: 2 for name in CATEGORIES}):
        raise ValueError("fixture category coverage must be exactly two per stratum")
    groups = [row["group_id"] for row in manifest["fixtures"]]
    if len(set(groups)) != 8:
        raise ValueError("fixture groups are not distinct")
    return manifest


def run(context: Context, args) -> dict:
    from obruxo_data.errors import ValidationError
    from obruxo_data.vital import VitalSchema
    identity_path = Path(args.identity).resolve()
    state_path = identity_path.parent / "status.json"
    if state_path.exists() and read_json(state_path).get("state") != "complete":
        raise ValueError("A's latest invocation is not complete; do not consume stale audit artifacts")
    audit = read_json(identity_path)
    if not audit.get("complete") or audit.get("rule") != RULE:
        raise ValueError("fixture selection requires a completed A audit with the current identity rule")
    members_path = identity_path.parent / audit["members_file"]
    if file_hash(members_path) != audit["members_sha256"]:
        raise ValueError("identity membership hash mismatch")
    root = Path(audit["corpus_root"])
    members = {str((root / row["relative_path"]).resolve()): row for row in map(json.loads, members_path.read_text().splitlines())}
    metadata_path = Path(args.metadata).resolve(strict=True)
    candidates = defaultdict(list)
    exclusions = []
    with metadata_path.open(encoding="utf-8-sig", newline="") as stream:
        for metadata in csv.DictReader(stream):
            context.check(5)
            category = metadata.get("type", "").strip()
            value = metadata.get("preset_file", "")
            if category not in CATEGORIES or not value:
                continue
            path = Path(value)
            if not path.is_absolute():
                path = metadata_path.parent / path
            member = members.get(str(path.resolve()))
            if member:
                candidates[category].append({"source_path": str(path.resolve()), "source_sha256": member["raw"],
                                             "group_id": member["state"], "category": category,
                                             "category_evidence": {"field": "type", "value": category, "preset_id": metadata.get("preset_id")}})
    selected, used = [], set()
    for category in CATEGORIES:
        for row in sorted(candidates[category], key=lambda row: (row["group_id"], row["source_path"])):
            context.check(5)
            if row["group_id"] in used or sum(item["category"] == category for item in selected) == 2:
                continue
            try:
                preset = load_fixture(row)
                if identities(Path(row["source_path"]).read_bytes())["state"] != row["group_id"]:
                    raise ValueError("identity group changed")
                names = control_names(preset)
                if not any(name.startswith("osc_") for name in names):
                    raise ValueError("no enabled oscillator")
                treatments = [perturb(preset, name)[1] for name in names]
            except (OSError, ValueError, TypeError, KeyError, RuntimeError, ValidationError) as error:
                exclusions.append({"group_id": row["group_id"], "category": category, "reason": str(error)})
                continue
            row.update(fixture_id=f"{category.lower()}_{row['group_id'][:12]}", controls=treatments,
                       synth_version=preset.to_dict().get("synth_version"), schema_id=preset.schema.schema_id)
            selected.append(row)
            used.add(row["group_id"])
            context.progress(selected=len(selected), static_exclusions=len(exclusions))
    complete = len(selected) == 8
    manifest = {"schema": "quick_fixtures_v1", "complete": complete, "fixtures": selected, "fixtures_hash": digest(selected),
                "identity_manifest_sha256": file_hash(identity_path), "metadata_sha256": file_hash(metadata_path),
                "schema_id": VitalSchema.load().schema_id,
                "selection": "two per exact metadata type, ordered by group ID; static checks only; no outcome selection"}
    write_json(context.output / "fixtures.json", manifest)
    write_csv(context.output / "static_exclusions.csv", exclusions)
    report = (f"# Fixture preparation\n\nSelected {len(selected)}/8 statically compatible, distinct groups. "
              f"Recorded {len(exclusions)} static exclusions.\n\n"
              "No native rendering or inference occurs in this command. Newer unsupported fields remain in source documents and cause exclusions; "
              "they are never stripped to gain compatibility. Selection is frozen before outcomes. Native runtime/QA failures stay in later coverage "
              "and do not trigger outcome-based replacement. Full source paths and exclusion detail stay local.\n")
    return finish_report(context, "FIXTURE_PREPARATION.md", report, "complete" if complete else "insufficient_fixtures", selected=len(selected))
