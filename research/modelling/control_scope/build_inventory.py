from __future__ import annotations

import ast
import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SCHEMA_DIR = ROOT / "research/data_generation/obruxo_data/vital/schema/vital-1.0.8-vita-0.1.0"
PROFILE_PATH = ROOT / "research/data_generation/obruxo_data/render/basic_profile.py"
OUTPUT_PATH = Path(__file__).with_name("stage_a_controls.jsonl")
EXPERIMENT = "stage_a_basic_codec_scope"
VARIANT = "current_basic_head"


def profile_constants() -> dict:
    tree = ast.parse(PROFILE_PATH.read_text(encoding="utf-8"))
    wanted = {"CONTINUOUS", "CATEGORICAL", "PHASE_CONTROL"}
    values = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        names = {target.id for target in node.targets if isinstance(target, ast.Name)}
        for name in names & wanted:
            if name == "PHASE_CONTROL":
                values[name] = {**{f"osc_{index}_random_phase": 0. for index in range(1, 4)}, "sample_random_phase": 0.}
                continue
            if isinstance(node.value, ast.Dict) and any(key is None for key in node.value.keys):
                parsed = {}
                for key, value in zip(node.value.keys, node.value.values, strict=True):
                    if key is None:
                        parsed.update(ast.literal_eval(value))
                    else:
                        parsed[ast.literal_eval(key)] = ast.literal_eval(value)
                values[name] = parsed
            else:
                values[name] = ast.literal_eval(node.value)
    if values.keys() != wanted:
        raise ValueError("basic profile is missing a control inventory")
    return values


def stable_hash(value) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def family_for(name: str) -> tuple[str, int | None]:
    match = re.match(r"^(osc|filter|env|lfo|modulation|random)_(\d+)_(.+)$", name)
    if match:
        prefix, index, field = match.groups()
        return f"{prefix}.{field}", int(index)
    if name.startswith("sample_"):
        return f"sample.{name.removeprefix('sample_')}", None
    return name, None


def structured_record(control_key: str, family: str, instance: int, value, *, data_path: str,
                      value_summary, present: bool = True, reason: str = "fixed by the basic-codec template") -> dict:
    return {"experiment_id": EXPERIMENT, "variant": VARIANT, "control_key": control_key, "vital_key": None,
            "family": family, "instance": instance, "kind": "structured", "changeable": False,
            "head_outputs": 0, "output_type": "none", "update_method": "fixed template",
            "training_signal": "none", "gradient_path": "none", "minimum": None, "maximum": None,
            "choices": None, "default_value": value_summary, "data_path": data_path,
            "data_sha256": stable_hash(value) if present else None, "present_in_template": present,
            "active_condition": None, "fixed_reason": reason}


def build() -> list[dict]:
    inventory = json.loads((SCHEMA_DIR / "parameter_inventory.json").read_text(encoding="utf-8"))["parameters"]
    preset = json.loads((SCHEMA_DIR / "init.vital").read_text(encoding="utf-8"))
    settings = preset["settings"]
    source = profile_constants()
    continuous = {name: (lo, hi, default) for name, lo, hi, default in source["CONTINUOUS"]}
    categorical = {name: tuple(options) for name, options in source["CATEGORICAL"]}
    phases = source["PHASE_CONTROL"]
    scalar_names = set(inventory)
    if not scalar_names <= set(settings) or (set(continuous) | set(categorical)) - scalar_names:
        raise ValueError("basic output inventory does not match the pinned Vital schema")

    rows = []
    for name, spec in sorted(inventory.items()):
        family, instance = family_for(name)
        if name in continuous:
            lo, hi, default = continuous[name]
            output_type, head_outputs, choices = "continuous", 1, None
            minimum, maximum, fixed_value = lo, hi, None
            update_method = "shared-model continuous head"
            training_signal = "source-parameter supervision in the current basic trainer"
            gradient_path = "parameter loss to prediction head; shared audio-loss path is not qualified"
            active_condition = ("masked when oscillator 1 uses one voice" if name == "osc_1_unison_detune" else
                                "masked when filter 1 is off" if name in {"filter_1_cutoff", "filter_1_resonance"} else None)
        elif name in categorical:
            options = categorical[name]
            output_type, head_outputs, choices = "categorical_scores", len(options), list(options)
            minimum = maximum = None
            fixed_value = None
            update_method = "shared-model categorical head"
            training_signal = "source-parameter supervision in the current basic trainer"
            gradient_path = "category loss to prediction head; shared audio-loss path is not qualified"
            active_condition = None
            default = options[0]
        else:
            output_type, head_outputs, choices = "fixed", 0, None
            minimum, maximum = spec.get("min"), spec.get("max")
            fixed_value = phases.get(name, settings[name])
            default = fixed_value
            update_method = "fixed renderer phase" if name in phases else "fixed reviewed template"
            training_signal = gradient_path = "none"
            active_condition = None
        rows.append({"experiment_id": EXPERIMENT, "variant": VARIANT, "control_key": name, "vital_key": name,
                     "family": family, "instance": instance, "kind": "scalar", "changeable": name in continuous or name in categorical,
                     "head_outputs": head_outputs, "output_type": output_type, "update_method": update_method,
                     "training_signal": training_signal, "gradient_path": gradient_path, "minimum": minimum,
                     "maximum": maximum, "choices": choices, "default_value": default, "fixed_value": fixed_value,
                     "data_path": f"settings.{name}", "data_sha256": None, "present_in_template": True,
                     "active_condition": active_condition,
                     "fixed_reason": None if name in continuous or name in categorical else "not represented by the Stage A head"})

    for index, route in enumerate(settings["modulations"], start=1):
        for field in ("source", "destination"):
            path = f"settings.modulations[{index - 1}].{field}"
            rows.append(structured_record(f"modulation_{index:02d}_{field}", f"modulation.{field}", index,
                                          route[field], data_path=path, value_summary=route[field]))
        rows.append(structured_record(f"modulation_{index:02d}_mapping_curve", "modulation.mapping_curve", index,
                                      None, data_path=f"modulation_slot_{index:02d}.mapping_curve",
                                      value_summary="linear default; no custom mapping stored", present=False,
                                      reason="no mapping-curve output in Stage A; retain linear mapping"))

    for index, lfo in enumerate(settings["lfos"], start=1):
        summary = {key: lfo.get(key) for key in ("num_points", "smooth")}
        summary["points_count"] = len(lfo.get("points", []))
        summary["powers_count"] = len(lfo.get("powers", []))
        rows.append(structured_record(f"lfo_{index}_curve", "lfo.curve", index, lfo,
                                      data_path=f"settings.lfos[{index - 1}]", value_summary=summary))

    for index, wavetable in enumerate(settings["wavetables"], start=1):
        summary = {key: wavetable.get(key) for key in ("name", "version", "author")}
        rows.append(structured_record(f"osc_{index}_wavetable_asset", "oscillator.wavetable_asset", index,
                                      wavetable, data_path=f"settings.wavetables[{index - 1}]", value_summary=summary))
        rows.append(structured_record(f"osc_{index}_custom_curve", "oscillator.custom_curve", index,
                                      wavetable, data_path=f"settings.wavetables[{index - 1}]",
                                      value_summary="fixed by the template; no custom-curve head"))

    sample = settings["sample"]
    sample_summary = {"type": type(sample).__name__}
    if isinstance(sample, str):
        sample_summary["length"] = len(sample)
    elif isinstance(sample, (dict, list)):
        sample_summary["items"] = len(sample)
    rows.append(structured_record("sample_asset", "sample.asset", 1, sample,
                                  data_path="settings.sample", value_summary=sample_summary))
    return rows


def main() -> None:
    rows = build()
    scalar_rows = [row for row in rows if row["kind"] == "scalar"]
    active_rows = [row for row in scalar_rows if row["changeable"]]
    if (len(scalar_rows) != 772 or len(active_rows) != 11
            or sum(row["head_outputs"] for row in active_rows) != 14
            or len({row["control_key"] for row in rows}) != len(rows)):
        raise ValueError("Stage A inventory counts or control keys do not reconcile")
    OUTPUT_PATH.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT_PATH.relative_to(ROOT)), "records": len(rows),
                      "scalar_fields": len(scalar_rows), "enabled_fields": len(active_rows),
                      "fixed_scalar_fields": len(scalar_rows) - len(active_rows),
                      "dense_outputs": sum(row["head_outputs"] for row in active_rows)}, indent=2))


if __name__ == "__main__":
    main()
