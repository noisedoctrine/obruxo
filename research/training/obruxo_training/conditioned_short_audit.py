"""Separate inferred input from playback, with one retained case per unique cell."""

import numpy as np
import torch

from obruxo_data.hashing import canonical_sha256
from obruxo_data.midi import Performance
from obruxo_data.render.qa import audio_float32_sha256

from .audit import write_json
from .codec import PilotTarget
from .conditioned_fit_policy import learning_access
from .features import tensor_digest
from .native_fit import audit_case, predict_candidates, record_features
from .paired_native_audit import equal_tree
from .paired_native_evidence import rescore_waveform
from .spectral import SpectralTargets

NAMES = ("initial", "parameter", "spectral")
CELLS = ((0, 0), (0, 1), (0, 2), (1, 1), (1, 0), (2, 2), (2, 0))


def cell_identity(cell):
    return {key: cell[key] for key in ("label", "input_view", "playback_view", "training_index")}


def cached_features(bank, recipe, view):
    row = bank[recipe["source_recipe_ids"][view]]
    if (row["state_id"] != recipe["state_id"] or row["target"] != recipe["target"]
            or row["audio_sha256"] != recipe["source_audio_sha256"][view]
            or tensor_digest(row["features"]) != recipe["source_feature_sha256"][view]):
        raise ValueError("short fit cached input identity changed")
    return row["features"]


def feature_bank(path):
    cache = torch.load(path, map_location="cpu", weights_only=True)
    rows = cache["replay"]
    bank = {row["recipe_id"]: row for row in rows}
    if len(bank) != len(rows):
        raise ValueError("short fit requires unique scheduling recipe IDs")
    return bank


def input_features(cell, records, bank):
    recipe, view, index = cell["recipe"], cell["input_view"], cell["training_index"]
    features = cached_features(bank, recipe, view)
    if index is not None and tensor_digest(record_features(records[index], view)) != tensor_digest(features):
        raise ValueError("short fit trained and scored different input recordings")
    return features


def collect_comparisons(session, output, plan, models, codec, records, cache_path, identities):
    bank, cases = feature_bank(cache_path), []
    for cell in plan["fit_audit_cells"]:
        recipe, playback = cell["recipe"], cell["playback_view"]
        features = input_features(cell, records, bank)
        candidates = predict_candidates(models, codec, features)
        case, _ = audit_case(session, output, cell["label"], recipe, playback, lambda _audio: candidates,
                             identities, expected_audio=recipe["source_audio_sha256"][playback])
        cases.append({**case, **cell_identity(cell)})
        write_json(output / "fit_cases.json", cases)
        records.check_limits()
    return cases


def means_by_state(cases):
    states = sorted({row["state_id"] for row in cases})
    return {name: {metric: float(np.mean([np.mean([row["rows"][name]["scores"][metric] for row in cases if row["state_id"] == state])
                                            for state in states])) for metric in ("spectral", "mrstft")} for name in NAMES}


def short_summary(cases, plan, trace):
    expected = plan["fit_audit_cells"]
    complete = (len(cases) == len(expected) and [cell_identity(row) for row in cases] == [cell_identity(row) for row in expected]
                and all(row["repeat_exact"] and row["view"] == cell["playback_view"] and row["state_id"] == cell["recipe"]["state_id"]
                        for row, cell in zip(cases, expected, strict=True)))
    if not complete:
        raise ValueError("short fit comparison coverage changed")
    groups = {}
    for view, role in enumerate(plan["training_view_roles"]):
        for transfer in (False, True):
            selected = [row for row in cases if row["training_index"] is not None and row["input_view"] == view
                        and (row["input_view"] != row["playback_view"]) == transfer]
            means = means_by_state(selected)
            passed = (means["spectral"]["spectral"] <= .9 * means["initial"]["spectral"]
                      and means["spectral"]["spectral"] <= means["parameter"]["spectral"]
                      and means["spectral"]["mrstft"] < means["initial"]["mrstft"])
            groups[f"{role}_{'transfer' if transfer else 'own'}"] = {"states": len({row["state_id"] for row in selected}),
                "cases": len(selected), "means": means, "pass": bool(passed)}
    original = [row for row in cases if row["input_view"] == row["playback_view"] == 0]
    if {row["state_id"] for row in original} != set(plan["regression_state_ids"]):
        raise ValueError("short fit original regression panel changed")
    means = means_by_state(original)
    p90 = {name: float(np.quantile([row["rows"][name]["scores"]["mrstft"] for row in original], .9)) for name in NAMES}
    regression = {"states": len(original), "means": means, "mrstft_p90": p90,
                  "pass": bool(all(means["spectral"][metric] <= means["initial"][metric] for metric in ("spectral", "mrstft"))
                               and p90["spectral"] <= 1.1 * p90["initial"])}
    access = learning_access(trace)
    return {"complete": True, "groups": groups, "original_regression": regression, "correction_learning_access": access,
            "pass": bool(all(row["pass"] for row in groups.values()) and regression["pass"] and access["pass"])}


def verify_comparisons(source, plan, models, codec, frontend, records, cache_path, cases, identities, require_receipt, require_request):
    bank, compared = feature_bank(cache_path), 0
    if len(cases) != len(plan["fit_audit_cells"]):
        raise ValueError("short fit comparison count changed")
    for cell, case in zip(plan["fit_audit_cells"], cases, strict=True):
        recipe, playback = cell["recipe"], cell["playback_view"]
        if cell_identity(cell) != cell_identity(case) or case["state_id"] != recipe["state_id"] or case["view"] != playback:
            raise ValueError("short fit input/playback identity changed")
        audit = torch.load(source / "audits" / f"{cell['label']}.pt", map_location="cpu", weights_only=True)
        contract = {"plan_id": canonical_sha256(plan), "recipe": recipe, "view": playback, "identity": identities}
        if (audit["contract"] != contract or audit["rows"] != case["rows"] or audit["source_receipt"] != case["source_receipt"]
                or not audit["repeat_exact"] or not case["repeat_exact"]):
            raise ValueError("short fit comparison evidence changed")
        reference = audit["audio"].numpy()
        if audio_float32_sha256(reference) != recipe["source_audio_sha256"][playback]:
            raise ValueError("short fit source audio changed")
        require_receipt(audit["source_receipt"])
        if audit["source_receipt"]["qa"]["audio_float32_sha256"] != audio_float32_sha256(reference):
            raise ValueError("short fit source receipt audio changed")
        performance = Performance.from_dict(recipe["performances"][playback])
        require_request(PilotTarget.from_dict(recipe["target"]), performance, audit["source_receipt"])
        # Reconstruct the playback's features and spectral target from actual retained audio.
        if tensor_digest(frontend(reference)) != recipe["source_feature_sha256"][playback]:
            raise ValueError("short fit playback input features cannot be reconstructed")
        if cell["training_index"] is not None:
            record = records[cell["training_index"]]
            if not equal_tree(SpectralTargets()(audit["audio"].T[None]), record["targets"][playback]):
                raise ValueError("short fit cached sound target cannot be reconstructed")
        candidates = predict_candidates(models, codec, input_features(cell, records, bank))
        if audit["candidates"] != {name: value.to_dict() for name, value in candidates.items()} or set(case["rows"]) != set(candidates):
            raise ValueError("short fit hard predictions differ from their declared input recording")
        for name, candidate in candidates.items():
            row = case["rows"][name]
            if row["candidate"] != candidate.to_dict():
                raise ValueError("short fit hard prediction changed")
            require_receipt(row["receipt"])
            require_request(candidate, performance, row["receipt"])
            if rescore_waveform(source, row["receipt"], reference, codec.contract.tail_seconds) != row["scores"]:
                raise ValueError("short fit sound score cannot be reconstructed")
            compared += 1
    return compared
