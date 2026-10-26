import copy
import json

import numpy as np
import pytest
from scipy.io import wavfile

from obruxo_training.listening import digest, prepare as prepare_listening, refresh_page, summarize, write_json


def original_level_packet(*args, **kwargs):
    return prepare_listening(*args, purpose="level_accuracy", **kwargs)


def fixture_inputs(tmp_path):
    audio = np.column_stack([np.linspace(-.3, .3, 256, dtype=np.float32)] * 2)
    assets = {}
    for index, name in enumerate(("reference", "candidate", "control"), 1):
        path = tmp_path / f"SECRET-MODEL-{name}.wav"
        wavfile.write(path, 44100, audio / index)
        assets[name] = {"path": path.name, "sha256": digest(path)}
    manifest = {"format": "blind_pairwise_listening_inputs_v1", "cases": [
        {"family_id": family, "role": role, **assets} for family, role in
        (("SECRET-FAMILY-1", "reference"), ("SECRET-FAMILY-1", "transfer"), ("SECRET-FAMILY-2", "reference"))]}
    path = tmp_path / "manifest.json"
    write_json(path, manifest)
    return path, manifest


def test_packet_hides_labels_preserves_samples_and_repeats_assignment(tmp_path):
    manifest_path, manifest = fixture_inputs(tmp_path)
    first, second = tmp_path / "first", tmp_path / "second"
    result = original_level_packet(manifest_path, first, seed=17)
    assert result["trials"] == 3
    assert original_level_packet(manifest_path, second, seed=17)["packet_id"] == result["packet_id"]
    randomized = tmp_path / "randomized"
    original_level_packet(manifest_path, randomized)
    assert type(json.loads((randomized / "answer-key.json").read_text())["seed"]) is int
    key = json.loads((first / "answer-key.json").read_text())
    assert {trial["a"] for trial in key["trials"]} == {"candidate", "control"}
    public = first / "listen"
    assert not any("SECRET" in path.name for path in public.iterdir())
    for path in (public / "packet.json", public / "index.html"):
        assert "SECRET" not in path.read_text()
    for trial in key["trials"]:
        for slot in ("a", "b"):
            _, original = wavfile.read(tmp_path / manifest["cases"][0][trial[slot]]["path"])
            _, anonymous = wavfile.read(public / f"{trial['trial_id']}-{slot}.wav")
            assert np.array_equal(original, anonymous)
    with pytest.raises(FileExistsError):
        original_level_packet(manifest_path, first, seed=17)


def test_bad_audio_or_hash_fails_before_packet_is_created(tmp_path):
    path, manifest = fixture_inputs(tmp_path)
    manifest["cases"][0]["candidate"] = {**manifest["cases"][0]["candidate"], "sha256": "changed"}
    write_json(path, manifest)
    output = tmp_path / "packet"
    with pytest.raises(ValueError, match="hash"):
        original_level_packet(path, output, seed=0)
    assert not output.exists()
    _, manifest = fixture_inputs(tmp_path)
    asset = tmp_path / manifest["cases"][0]["candidate"]["path"]
    wavfile.write(asset, 44100, np.full((256, 2), np.nan, dtype=np.float32))
    for row in manifest["cases"]:
        row["candidate"]["sha256"] = digest(asset)
    write_json(path, manifest)
    with pytest.raises(ValueError, match="finite"):
        original_level_packet(path, output, seed=0)
    assert not output.exists()


def test_page_refresh_preserves_choices_packet_assignment_audio_and_valid_judgments(tmp_path):
    manifest_path, _ = fixture_inputs(tmp_path)
    packet = tmp_path / "packet"
    original_level_packet(manifest_path, packet, seed=17)
    key_path = packet / "answer-key.json"
    before = json.loads(key_path.read_bytes())
    audio = {path.name: digest(path) for path in (packet / "listen").glob("*.wav")}
    answers = {"trial-001-preferred": "b", "trial-001-usable": "neither"}
    receipt = refresh_page(key_path, answers)
    after = json.loads(key_path.read_bytes())
    assert after["packet"] == before["packet"] and after["trials"] == before["trials"]
    assert {path.name: digest(path) for path in (packet / "listen").glob("*.wav")} == audio
    assert receipt["restored_answers"] == 2 and receipt["audio_and_assignment_unchanged"]
    assert '"trial-001-preferred": "b"' in (packet / "listen/index.html").read_text(encoding="utf-8")
    responses, _ = make_responses(packet, tmp_path)
    assert summarize(key_path, responses)["completed_trials"] == 3
    assert list((packet / "page-revisions").glob("*.html"))
    with pytest.raises(ValueError, match="restored choices"):
        refresh_page(key_path, {"foreign-preferred": "a"})


def make_responses(packet, tmp_path):
    key = json.loads((packet / "answer-key.json").read_text())
    rows = []
    for trial in key["trials"]:
        method = "candidate" if trial["family_id"] == "SECRET-FAMILY-1" else "control"
        slot = next(slot for slot in ("a", "b") if trial[slot] == method)
        rows.append({"trial_id": trial["trial_id"], "preferred": slot, "usable": slot})
    responses = {"format": "blind_pairwise_listening_responses_v1", "packet_id": key["packet"]["packet_id"], "judgments": rows}
    path = tmp_path / "responses.json"
    write_json(path, responses)
    return path, responses


def test_judgments_decode_both_assignments_and_weight_independent_families(tmp_path):
    manifest_path, _ = fixture_inputs(tmp_path)
    packet = tmp_path / "packet"
    original_level_packet(manifest_path, packet, seed=17)
    responses_path, _ = make_responses(packet, tmp_path)
    report = summarize(packet / "answer-key.json", responses_path)
    assert report["completed_trials"] == report["planned_trials"] == 3
    assert report["counts"] == {"candidate": 2, "control": 1, "tie": 0, "neither": 0}
    assert report["family_mean_preference"] == 0
    assert report["candidate_family_mean_usable_fraction"] == .5
    assert report["control_family_mean_usable_fraction"] == .5
    changed = json.loads(responses_path.read_text())
    changed["judgments"][0].update(preferred="tie", usable="both")
    changed["judgments"][1].update(preferred="neither", usable="neither")
    write_json(responses_path, changed)
    report = summarize(packet / "answer-key.json", responses_path)
    assert report["counts"]["tie"] == report["counts"]["neither"] == 1
    assert report["completed_trials"] == 3


def test_missing_duplicate_foreign_and_invalid_judgments_never_drop_trials(tmp_path):
    manifest_path, _ = fixture_inputs(tmp_path)
    packet = tmp_path / "packet"
    original_level_packet(manifest_path, packet, seed=17)
    path, responses = make_responses(packet, tmp_path)
    for mutation in ("missing", "duplicate", "foreign", "choice", "packet"):
        changed = copy.deepcopy(responses)
        if mutation == "missing":
            changed["judgments"].pop()
        elif mutation == "duplicate":
            changed["judgments"][1] = changed["judgments"][0]
        elif mutation == "foreign":
            changed["judgments"][0]["trial_id"] = "foreign"
        elif mutation == "choice":
            changed["judgments"][0]["usable"] = "maybe"
        else:
            changed["packet_id"] = "foreign"
        write_json(path, changed)
        with pytest.raises(ValueError):
            summarize(packet / "answer-key.json", path)


def test_packet_audio_and_private_assignment_tampering_are_detected(tmp_path):
    manifest_path, _ = fixture_inputs(tmp_path)
    packet = tmp_path / "packet"
    original_level_packet(manifest_path, packet, seed=17)
    path, _ = make_responses(packet, tmp_path)
    key_path = packet / "answer-key.json"
    original = json.loads(key_path.read_text())
    changed = copy.deepcopy(original)
    trial = changed["trials"][0]
    trial["a"], trial["b"] = trial["b"], trial["a"]
    write_json(key_path, changed)
    with pytest.raises(ValueError, match="packet"):
        summarize(key_path, path)
    write_json(key_path, original)
    page = packet / "listen" / "index.html"
    original_page = page.read_bytes()
    page.write_bytes(original_page + b"changed")
    with pytest.raises(ValueError, match="page or packet"):
        summarize(key_path, path)
    page.write_bytes(original_page)
    (packet / "listen" / original["packet"]["trials"][0]["audio"]["a"]["file"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="audio changed"):
        summarize(key_path, path)
