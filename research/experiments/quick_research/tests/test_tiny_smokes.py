"""Tiny synthetic checks only: no actual corpus, Vital plugin, weights, or inference."""

from argparse import Namespace
import json
from pathlib import Path
import subprocess
from unittest.mock import Mock

import numpy as np
import pytest

from quickresearch import cli, fixtures, identity, measurement, source_audit
from quickresearch.common import BudgetExpired, Context, checked_output, file_hash, read_json, write_json
from quickresearch.metrics import distances, separation
from quickresearch.renders import render_one


def test_lossless_identity_and_display_rules():
    first = b'{"preset_name":"a","settings":{"x":1}}'
    formatted = b' { "settings": {"x":1}, "preset_name":"a" } '
    renamed = b'{"preset_name":"b","settings":{"x":1}}'
    a, b, c = map(identity.identities, (first, formatted, renamed))
    assert a["raw"] != b["raw"] and a["canonical"] == b["canonical"]
    assert a["canonical"] != c["canonical"] and a["state"] == c["state"]
    for left, right in [
        ('{"x":9007199254740992}', '{"x":9007199254740993}'),
        ('{"x":0.123456789012345678901}', '{"x":0.123456789012345678902}'),
        ('{"x":true}', '{"x":1}'), ('{"x":[1,2]}', '{"x":[2,1]}'),
        ('{"settings":{"author":"a"}}', '{"settings":{"author":"b"}}'),
        ('{"synth_version":"1"}', '{"synth_version":"2"}'),
        ('{"x":"1.0"}', '{"x":1.0}'), ('{"x":1}', '{"x":1.0}'),
    ]:
        assert identity.identities(left.encode())["state"] != identity.identities(right.encode())["state"]


@pytest.mark.parametrize("raw", [b'{"a":1,"a":2}', b'{"a":NaN}', b'{"a":Infinity}', b'[]'])
def test_identity_rejects_ambiguous_documents(raw):
    with pytest.raises(ValueError):
        identity.identities(raw)


def test_identity_audit_synthetic_four_files(tmp_path):
    corpus, output = tmp_path / "corpus", tmp_path / "output"
    corpus.mkdir()
    output.mkdir()
    for name, value in {"a.vital": '{"preset_name":"a","settings":{"x":1}}',
                        "b.vital": '{"settings":{"x":1},"preset_name":"a"}',
                        "c.vital": '{"preset_name":"b","settings":{"x":1}}', "bad.vital": '{"a":NaN}'}.items():
        (corpus / name).write_text(value)
    before = {path.name: file_hash(path) for path in corpus.iterdir()}
    result = identity.run(Context(output, 10), Namespace(corpus=str(corpus), seed=0))
    assert result["state"] == "complete"
    summary = read_json(output / "identity_summary.json")
    assert summary["valid_files"] == 3 and summary["excluded_files"] == 1
    assert [summary["levels"][level]["groups"] for level in identity.LEVELS] == [3, 2, 1]
    assert {path.name: file_hash(path) for path in corpus.iterdir()} == before
    rows = [json.loads(line) for line in (output / "members.jsonl").read_text().splitlines()]
    for index, row in enumerate(rows):
        row["simulated_split"] = ("train", "validation", "test")[index]
    assert identity.summarize(rows)["state"]["crossing_files"] == 3


def test_deadline_and_output_boundary(tmp_path):
    with pytest.raises(ValueError):
        checked_output(tmp_path)
    with pytest.raises(BudgetExpired):
        Context(tmp_path, 0).check()
    with pytest.raises(SystemExit):
        cli.parser().parse_args(["identity", "--corpus", str(tmp_path), "--output", str(tmp_path), "--budget-seconds", "1801"])


def test_metrics_analytic_signals_and_separation():
    samples = np.sin(np.arange(4096) * .07)
    audio = np.column_stack([samples, samples]).astype(np.float32)
    assert all(value == 0 for value in distances(audio, audio).values())
    changed = distances(audio, audio * .5)
    assert changed["waveform_rmse"] == pytest.approx(float(np.sqrt(np.mean((audio * .5)**2))), rel=1e-6)
    assert changed["envelope_mae"] > 0 and changed["spectral_distance"] > 0
    assert separation([.1, .2, .3], [.4, .5])["interpretation"] == "observed_separation"
    assert separation([.1, .2, .3], [.2, .5])["interpretation"] == "overlap_inconclusive"
    assert separation([.1], [.2])["interpretation"] == "insufficient_valid_comparisons"
    with pytest.raises(ValueError):
        distances(audio, audio[:-1])
    with pytest.raises(ValueError):
        distances(audio, np.full_like(audio, np.nan))


def synthetic_fixture(tmp_path):
    from obruxo_data.vital import VitalPreset
    preset = VitalPreset.init()
    path = tmp_path / "synthetic.vital"
    preset.save(path)
    return {"fixture_id": "synthetic", "source_path": str(path), "source_sha256": file_hash(path), "category": "Bass",
            "group_id": identity.identities(path.read_bytes())["state"], "controls": [fixtures.perturb(preset, "osc_1_level")[1]]}, preset


def test_performances_and_single_control_mutation(tmp_path):
    row, preset = synthetic_fixture(tmp_path)
    changed, info = fixtures.perturb(preset, "osc_1_level")
    a, b = preset.to_dict(), changed.to_dict()
    assert a != b
    b["settings"]["osc_1_level"] = info["original_raw"]
    assert a == b
    preset.set_raw("osc_1_level", 1)
    assert fixtures.perturb(preset, "osc_1_level")[1]["changed_raw"] == .95
    notes = fixtures.performance("staccato_c4")
    assert [(span.start_tick, span.end_tick) for span in notes.note_spans()] == [(0, 240), (480, 720), (960, 1200), (1440, 1680)]
    assert notes.end_tick == 1920
    assert fixtures.performance("held_c3").note_spans()[0].pitch == 48
    assert len(measurement.jobs_for([row])) == 5


def test_static_schema_failure_is_excluded_not_stripped(tmp_path):
    # No real corpus: the only candidate deliberately has a newer unsupported field.
    row, _ = synthetic_fixture(tmp_path)
    source = Path(row["source_path"])
    document = json.loads(source.read_text())
    document["settings"]["unsupported_future_field"] = 1
    source.write_text(json.dumps(document))
    original = source.read_bytes()
    audit = tmp_path / "audit"
    audit.mkdir()
    members = audit / "members.jsonl"
    ids = identity.identities(original)
    members.write_text(json.dumps({"relative_path": source.name, **ids}) + "\n")
    write_json(audit / "identity_manifest.json", {"rule": identity.RULE, "complete": True, "corpus_root": str(tmp_path),
               "members_file": "members.jsonl", "members_sha256": file_hash(members)})
    metadata = tmp_path / "metadata.csv"
    metadata.write_text(f"type,preset_file,preset_id\nBass,{source.name},synthetic\n")
    output = tmp_path / "fixtures"
    output.mkdir()
    result = fixtures.run(Context(output, 10), Namespace(identity=str(audit / "identity_manifest.json"), metadata=str(metadata)))
    assert result["state"] == "insufficient_fixtures"
    assert source.read_bytes() == original
    assert "unsupported_future_field" not in (output / "fixtures.json").read_text()
    assert len((output / "static_exclusions.csv").read_text().splitlines()) == 2


def test_independent_repeat_cache_hash_and_source_guard(tmp_path):
    from obruxo_data.render import RenderProvenance, RenderResult
    from obruxo_data.render.qa import analyze_audio
    row, _ = synthetic_fixture(tmp_path)
    class FakeRenderer:
        renderer_id = "synthetic-renderer"
        calls = 0

        def render(self, request):
            self.calls += 1
            audio = np.full((4096, 2), .01 * self.calls, dtype=np.float32)
            qa, diagnostics = analyze_audio(audio, sample_rate=44100, expected_frames=4096, expected_channels=2)
            return RenderResult(audio, 44100, diagnostics, RenderProvenance(request.request_id, self.renderer_id, "fake", "fake", {}), qa)

    renderer = FakeRenderer()
    directory = tmp_path / "renders"
    context = Context(tmp_path, 20)
    first = render_one(context, renderer, row, "held_c4", "baseline", 0, directory, "test")
    same = render_one(context, renderer, row, "held_c4", "baseline", 0, directory, "test")
    second = render_one(context, renderer, row, "held_c4", "baseline", 1, directory, "test")
    assert renderer.calls == 2 and same["cache_hit"]
    assert first["identity"]["request_id"] == second["identity"]["request_id"]
    assert first["audio_sha256"] != second["audio_sha256"]
    Path(first["wav"]).write_bytes(b"corrupt cache")
    repaired = render_one(context, renderer, row, "held_c4", "baseline", 0, directory, "test")
    assert renderer.calls == 3 and not repaired["cache_hit"]
    source = Path(row["source_path"])
    source.write_text(source.read_text() + " ")
    with pytest.raises(ValueError, match="source changed"):
        render_one(context, renderer, row, "held_c4", "baseline", 0, directory, "test")


def test_source_packet_requires_review_and_verifies_citations(tmp_path, monkeypatch):
    source, output = tmp_path / "source", tmp_path / "packet"
    source.mkdir()
    output.mkdir()
    commit = "a" * 40
    lines = "// synthetic source\nint modulation_amount;\n// end"
    def fake_git(context, root, *arguments, **kwargs):
        if arguments[0] == "rev-parse":
            return commit
        if arguments[0] == "config":
            return "https://github.com/example/synthetic.git"
        if arguments[0] == "grep":
            return f"{commit}:src/example.cpp:2:int modulation_amount;"
        return lines
    monkeypatch.setattr(source_audit, "git", fake_git)
    result = source_audit.run(Context(output, 10), Namespace(source=str(source), revision=commit, max_hits=1))
    assert result["state"] == "evidence_collected_review_required"
    answers_path = output / "review_answers.json"
    answers = read_json(answers_path)
    for case in answers["cases"]:
        case.update(status="supported", answer="Synthetic smoke assertion, not a Vital finding.",
                    citations=[{"path": "src/example.cpp", "start": 2, "end": 2}])
    write_json(answers_path, answers)
    reviewed = tmp_path / "review"
    reviewed.mkdir()
    result = source_audit.review(Context(reviewed, 10), Namespace(packet=str(output / "source_packet.json"), answers=str(answers_path)))
    assert result["state"] == "review_recorded"
    assert "#L2-L2" in (reviewed / "MODULATION_SLOT_SEMANTICS_AUDIT.md").read_text()
    answers["cases"][0]["citations"][0]["end"] = 99
    write_json(answers_path, answers)
    with pytest.raises(ValueError, match="citation range"):
        source_audit.review(Context(reviewed, 10), Namespace(packet=str(output / "source_packet.json"), answers=str(answers_path)))


def test_supervisor_timeout_does_not_claim_completion(tmp_path, monkeypatch):
    output = tmp_path / "supervised"
    monkeypatch.setattr(cli, "checked_output", lambda path: path)
    monkeypatch.setattr(cli, "code_hash", lambda *paths: "synthetic")
    child = Mock()
    child.wait.side_effect = subprocess.TimeoutExpired("synthetic", 1)
    monkeypatch.setattr(cli.subprocess, "Popen", lambda *a, **k: child)
    stop = Mock()
    monkeypatch.setattr(cli, "stop_worker", stop)
    code = cli.supervise({"command": "identity", "output": str(output), "corpus": str(tmp_path), "seed": 0,
                          "budget_seconds": 1, "resume": False})
    assert code == 2
    stop.assert_called_once_with(child)
    assert read_json(output / "status.json")["state"] == "deadline_terminated"
    assert not (output / "run.lock").exists()


def test_feature_shape_guard_without_model_or_torch_import():
    from quickresearch.encoder import FrozenFeatures
    arrays = {name: np.zeros((4, width), dtype=np.float32) for name, width in (("note", 88), ("onset", 88), ("contour", 264))}
    FrozenFeatures.validate(arrays)
    arrays["note"][0, 0] = np.nan
    with pytest.raises(ValueError):
        FrozenFeatures.validate(arrays)


def test_report_plots_with_synthetic_arrays(tmp_path):
    from quickresearch.encoder import plot_fixture, posterior_diagnostics
    from quickresearch.measurement import plot_ranges
    from obruxo_basic_pitch.evaluation.labels import ReferenceNote
    features = tmp_path / "features.npz"
    arrays = {name: np.zeros((172, width), dtype=np.float32) for name, width in (("note", 88), ("onset", 88), ("contour", 264))}
    arrays["note"][:, 39] = .8
    arrays["onset"][0, 39] = .9
    np.savez_compressed(features, **arrays)
    reference = [ReferenceNote(0, 1, 60, 100)]
    diagnostics = posterior_diagnostics(reference, arrays)
    assert diagnostics[0]["correct_pitch_mean"] == pytest.approx(.8)
    case = {"performance": "held_c4", "features": {"arrays_path": str(features)},
            "reference": [reference[0].as_dict()], "events": []}
    assert (tmp_path / plot_fixture(tmp_path, "fixture", [case])).is_file()
    rows = []
    for metric in ("waveform_rmse", "spectral_distance", "envelope_mae"):
        rows.append({"fixture_id": "fixture", "control": "osc_1_level", "metric": metric,
                     "unchanged_min": .01, "unchanged_max": .02, "changed_min": .03, "changed_max": .04})
    assert (tmp_path / plot_ranges(tmp_path, rows)).is_file()
