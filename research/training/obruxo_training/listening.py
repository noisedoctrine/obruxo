"""Anonymous pairwise listening packets; actual judgments remain a separate input."""

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import secrets
from statistics import mean


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def checked_audio(asset, root):
    import numpy as np
    from scipy.io import wavfile
    path = root / asset["path"]
    if digest(path) != asset["sha256"]:
        raise ValueError("listening input file hash changed")
    rate, audio = wavfile.read(path)
    if (rate != 44100 or audio.dtype != np.float32 or audio.ndim != 2 or audio.shape[1] != 2
            or not len(audio) or not np.isfinite(audio).all() or np.max(np.abs(audio)) > 1):
        raise ValueError("listening input must be finite 44.1-kHz stereo float32 WAV within [-1, 1]")
    return audio


def prepare(manifest_path, output, *, seed=None):
    from scipy.io import wavfile
    manifest_path, output = Path(manifest_path), Path(output)
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest["format"] != "blind_pairwise_listening_inputs_v1" or not manifest["cases"]:
        raise ValueError("expected nonempty blind pairwise listening inputs")
    seed = secrets.randbits(128) if seed is None else seed
    if type(seed) is not int:
        raise ValueError("listening seed must be an integer")
    cases, seen = [], set()
    for row in manifest["cases"]:
        case = (row["family_id"], row["role"])
        if not all(isinstance(value, str) and value for value in case) or case in seen:
            raise ValueError("listening cases require unique nonempty family/role pairs")
        seen.add(case)
        audio = {name: checked_audio(row[name], manifest_path.parent) for name in ("reference", "candidate", "control")}
        if len({value.shape for value in audio.values()}) != 1:
            raise ValueError("paired clips must preserve the same complete observation length")
        cases.append(row)
        del audio
    # Validate every source before creating an output directory. Re-encoding
    # removes identifying WAV metadata, preserving all float32 sample values.
    output.mkdir(parents=True, exist_ok=False)
    public = output / "listen"
    public.mkdir()
    rng = random.Random(seed)
    rng.shuffle(cases)
    assignments = [["candidate", "control"] if index % 2 else ["control", "candidate"] for index in range(len(cases))]
    rng.shuffle(assignments)
    trials, key = [], []
    for index, row in enumerate(cases):
        audio = {name: checked_audio(row[name], manifest_path.parent) for name in ("reference", "candidate", "control")}
        trial_id = f"trial-{index + 1:03d}"
        methods = assignments[index]
        files = {}
        for slot, method in (("reference", "reference"), ("a", methods[0]), ("b", methods[1])):
            name = f"{trial_id}-{slot}.wav"
            wavfile.write(public / name, 44100, audio[method])
            files[slot] = {"file": name, "sha256": digest(public / name)}
        trials.append({"trial_id": trial_id, "audio": files})
        key.append({"trial_id": trial_id, "family_id": row["family_id"], "role": row["role"],
                    "a": methods[0], "b": methods[1]})
        del audio
    private = {"seed": seed, "trials": key, "input_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(), "inputs": manifest}
    packet = {"format": "blind_pairwise_listening_packet_v1", "trials": trials, "assignment_sha256": identity(private)}
    packet["packet_id"] = identity(packet)
    write_json(public / "packet.json", packet)
    template = Path(__file__).with_name("listening.html").read_text(encoding="utf-8")
    (public / "index.html").write_text(template.replace("__PACKET__", json.dumps(packet)), encoding="utf-8")
    write_json(output / "answer-key.json", {"format": "blind_pairwise_listening_key_v1", "packet": packet, **private,
               "files_sha256": {name: digest(public / name) for name in ("index.html", "packet.json")}})
    return {"packet_id": packet["packet_id"], "trials": len(trials), "listener_page": str(public / "index.html")}


def summarize(key_path, responses_path):
    key_path, responses_path = Path(key_path), Path(responses_path)
    key = json.loads(key_path.read_text(encoding="utf-8"))
    responses = json.loads(responses_path.read_text(encoding="utf-8"))
    packet = key["packet"]
    payload = {name: value for name, value in packet.items() if name != "packet_id"}
    private = {name: key[name] for name in ("seed", "trials", "input_manifest_sha256", "inputs")}
    if (key["format"] != "blind_pairwise_listening_key_v1" or identity(payload) != packet["packet_id"]
            or identity(private) != packet["assignment_sha256"]
            or responses.get("format") != "blind_pairwise_listening_responses_v1"
            or responses.get("packet_id") != packet["packet_id"]):
        raise ValueError("listening response belongs to a different or modified packet")
    public = key_path.parent / "listen"
    for name in ("index.html", "packet.json"):
        if digest(public / name) != key["files_sha256"][name]:
            raise ValueError("listening page or packet changed")
    for trial in packet["trials"]:
        for asset in trial["audio"].values():
            if digest(public / asset["file"]) != asset["sha256"]:
                raise ValueError("listening packet audio changed")
    expected = {trial["trial_id"]: trial for trial in key["trials"]}
    rows = responses.get("judgments", [])
    actual = [row.get("trial_id") for row in rows]
    if len(expected) != len(key["trials"]) or len(actual) != len(expected) or set(actual) != set(expected):
        raise ValueError("every planned listening trial must have exactly one judgment")
    counts, families, decoded = Counter(), defaultdict(list), []
    for row in rows:
        if row.get("preferred") not in {"a", "b", "tie", "neither"} or row.get("usable") not in {"a", "b", "both", "neither"}:
            raise ValueError("each trial needs valid preference and usefulness judgments")
        trial = expected[row["trial_id"]]
        preferred = trial[row["preferred"]] if row["preferred"] in {"a", "b"} else row["preferred"]
        usable = [trial[slot] for slot in ("a", "b") if row["usable"] in {slot, "both"}]
        score = 1 if preferred == "candidate" else -1 if preferred == "control" else 0
        value = {"family_id": trial["family_id"], "role": trial["role"], "preferred": preferred,
                 "usable": sorted(usable), "preference_score": score}
        decoded.append(value)
        families[trial["family_id"]].append(value)
        counts[preferred] += 1
    family_scores = [{"family_id": family, "trials": len(values),
                      "preference_score": mean(row["preference_score"] for row in values),
                      "candidate_usable_fraction": mean("candidate" in row["usable"] for row in values),
                      "control_usable_fraction": mean("control" in row["usable"] for row in values)}
                     for family, values in sorted(families.items())]
    return {"format": "blind_pairwise_listening_summary_v1", "packet_id": packet["packet_id"],
            "responses_sha256": digest(responses_path), "answer_key_sha256": digest(key_path),
            "planned_trials": len(expected), "completed_trials": len(rows), "families": family_scores,
            "counts": {name: counts[name] for name in ("candidate", "control", "tie", "neither")},
            "family_mean_preference": mean(row["preference_score"] for row in family_scores),
            "candidate_family_mean_usable_fraction": mean(row["candidate_usable_fraction"] for row in family_scores),
            "control_family_mean_usable_fraction": mean(row["control_usable_fraction"] for row in family_scores),
            "judgments": sorted(decoded, key=lambda row: (row["family_id"], row["role"])),
            "claim": "actual_listener_judgments_required; descriptive_only_not_stage_qualification"}
