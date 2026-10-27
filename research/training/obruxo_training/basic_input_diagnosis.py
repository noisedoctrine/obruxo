"""Inspect saved basic-fit failures and prepare genuinely unanswered listening clips."""

from collections import defaultdict
import hashlib
import json
from statistics import mean, median

from obruxo_data.hashing import canonical_sha256

from .audit import write_json
from .codec import PilotCodec, PilotTarget

INPUTS = ("original", "held_note", "alternate")
SELECTION = "per_group_family_mean_gap_median_and_largest_remaining_gap_first_playback_v1"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def describe(cells, rows, codec):
    predictions, groups = {}, defaultdict(list)
    for row in rows:
        cell = cells[row["index"]]
        if cell["training_index"] is None:
            continue
        key = cell["recipe"]["state_id"], cell["input_view"], row["arm"]
        value = row["candidate"], cell["recipe"]["target"]
        if key in predictions and predictions[key] != value:
            raise ValueError("one input recording produced different patches across playback performances")
        predictions[key] = value
    for (_family, view, arm), value in predictions.items():
        groups[INPUTS[view], arm].append(value)
    result = {}
    for (view, arm), pairs in sorted(groups.items()):
        errors = {}
        for index, (name, *_bounds) in enumerate(codec.continuous):
            values = [abs(predicted["continuous"][index] - target["continuous"][index]) for predicted, target in pairs
                      if codec.continuous_mask(PilotTarget.from_dict(target))[index]]
            errors[name] = mean(values) if values else None
        result[f"{view}:{arm}"] = {"independent_patches": len(pairs), "correct_unison": sum(a["categorical"][0] == b["categorical"][0] for a, b in pairs),
                                  "correct_filter": sum(a["categorical"][1] == b["categorical"][1] for a, b in pairs),
                                  "source_active_normalized_control_mae": errors}
    return result


def select_examples(cells, rows):
    finals = {(row["index"], row["arm"]): row for row in rows}
    if len(finals) != len(rows):
        raise ValueError("duplicate final prediction")
    groups = defaultdict(lambda: defaultdict(list))
    for index, cell in enumerate(cells):
        if cell["training_index"] is None:
            continue
        group = f"{INPUTS[cell['input_view']]}:{'own' if cell['input_view'] == cell['playback_view'] else 'transfer'}"
        delta = finals[index, "spectral"]["scores"]["spectral"] - finals[index, "parameter"]["scores"]["spectral"]
        groups[group][cell["recipe"]["state_id"]].append((index, delta))
    selected = []
    for group, families in sorted(groups.items()):
        if len(families) < 2:
            raise ValueError("each diagnostic group requires two independent patches")
        gaps = {family: mean(delta for _index, delta in values) for family, values in families.items()}
        middle = min(gaps, key=lambda family: (abs(gaps[family] - median(gaps.values())), family))
        worst = max((family for family in gaps if family != middle), key=lambda family: (gaps[family], family))
        for choice, family in (("median_gap", middle), ("largest_remaining_gap", worst)):
            index = min(value[0] for value in families[family])
            selected.append({"group": group, "selection": choice, "family_id": family, "index": index, "family_mean_combined_loss_gap": gaps[family]})
    if len(groups) != 6 or len(selected) != 12:
        raise ValueError("diagnostic listening requires all six groups")
    return selected


def prepare(fit, output, *, implementation_commit, ffmpeg=None):
    if output.exists():
        raise FileExistsError(output)
    from .listening_loudness import resolve_ffmpeg
    ffmpeg = resolve_ffmpeg(ffmpeg)
    plan = json.loads((fit / "plan.json").read_bytes())
    verified = json.loads((fit / "verification.json").read_bytes())
    if not verified["verified"] or verified["plan_id"] != canonical_sha256(plan) or not verified["complete"]:
        raise ValueError("listening preparation requires a complete independently verified fit")
    for name in ("plan.json", "finals.jsonl"):
        if digest(fit / name) != verified["evidence_sha256"].get(name):
            raise ValueError("fit evidence changed after verification")
    rows = [json.loads(line) for line in (fit / "finals.jsonl").read_text(encoding="utf-8").splitlines()]
    cells = plan["fit_audit_cells"]
    if len(rows) != 2 * len(cells):
        raise ValueError("the diagnostic needs both final arms for every planned cell")
    controls, selected = describe(cells, rows, PilotCodec()), select_examples(cells, rows)
    output.mkdir(parents=True)
    frozen = {"fit_plan_id": verified["plan_id"], "implementation_commit": implementation_commit, "verification_sha256": digest(fit / "verification.json"),
              "selection_rule": SELECTION, "examples": selected, "new_native_requests": 0, "optimizer_applications": 0,
              "limits": {"wall_seconds": 300, "process_tree_bytes": 4 * 1024**3, "archive_bytes": 128 * 1024**2}}
    write_json(output / "plan.json", frozen)
    import numpy as np
    from scipy.io import wavfile
    import torch
    from obruxo_data.render.qa import audio_float32_sha256
    from .listening import prepare as prepare_listening
    finals = {(row["index"], row["arm"]): row for row in rows}
    inputs = output / "inputs"
    inputs.mkdir()
    manifest = {"format": "blind_pairwise_listening_inputs_v1", "cases": []}
    for number, selection in enumerate(selected):
        index = selection["index"]
        candidate, control = finals[index, "spectral"], finals[index, "parameter"]
        assets = {"reference": candidate["source_audio"], "candidate": candidate["receipt"]["retained_audio"], "control": control["receipt"]["retained_audio"]}
        case = {"family_id": selection["family_id"], "role": f"{selection['group']}:{selection['selection']}"}
        for role, metadata in assets.items():
            source = (fit / metadata["file"]).resolve()
            if not source.is_relative_to(fit.resolve()) or digest(source) != metadata["sha256"]:
                raise ValueError("retained diagnostic waveform changed or leaves its fit directory")
            audio = torch.load(source, map_location="cpu", weights_only=True)["audio"].numpy()
            if role != "reference":
                row = candidate if role == "candidate" else control
                if audio_float32_sha256(audio) != row["receipt"]["qa"]["audio_float32_sha256"]:
                    raise ValueError("diagnostic waveform differs from its native receipt")
            destination = inputs / f"case-{number:02d}-{role}.wav"
            wavfile.write(destination, 44100, audio)
            rate, reread = wavfile.read(destination)
            if rate != 44100 or not np.array_equal(reread, audio):
                raise ValueError("listening export changed the native float32 samples")
            case[role] = {"path": destination.relative_to(output).as_posix(), "sha256": digest(destination)}
        manifest["cases"].append(case)
    write_json(output / "inputs.json", manifest)
    packet = prepare_listening(output / "inputs.json", output / "packet", ffmpeg=ffmpeg)
    write_json(output / "control_diagnosis.json", {"groups": controls, "claim": "source_parameter_errors_are_diagnostic_not_reconstruction_acceptance"})
    return {"fit_plan_id": verified["plan_id"], "packet_id": packet["packet_id"], "trials": packet["trials"],
            "independent_families": len({row["family_id"] for row in selected}), "listener_page": packet["listener_page"],
            "new_native_requests": 0, "optimizer_applications": 0, "responses_received": False, "qualified_for_release": False}
