import json

import pytest

from obruxo_data.hashing import canonical_sha256
from obruxo_training.audit import audit_corpus, load_snapshot
from obruxo_training.cli import main
from obruxo_training.identity import PresetIdentity


def corpus_fixture(root, init_raw):
    root.mkdir()
    (root / "private-name-a.vital").write_bytes(init_raw)
    document = json.loads(init_raw)
    document["preset_name"] = "private-name-b"
    (root / "private-name-b.vital").write_text(json.dumps(document))
    (root / "broken.vital").write_bytes(b"{broken")


def test_audit_is_read_only_grouped_and_loadable(tmp_path, init_raw, schema, catalog):
    corpus, output = tmp_path / "corpus", tmp_path / "snapshot"
    corpus_fixture(corpus, init_raw)
    before = {path.name: path.read_bytes() for path in corpus.iterdir()}
    summary = audit_corpus(corpus, output, schema=schema, catalog=catalog)
    assert summary["population_complete"]
    assert summary["strictly_parsed_files"] == 2 and summary["excluded_files"] == 1
    assert summary["identity_groups"]["state"] == 1
    assert summary["static_eligible_states"] == 1
    assert before == {path.name: path.read_bytes() for path in corpus.iterdir()}
    assert "private-name" not in (output / "REPORT.md").read_text()
    contract, loaded_catalog, manifest = load_snapshot(output, schema)
    assert loaded_catalog == catalog and len(manifest.assignments) == 1
    assert contract.contract_id == summary["contract_id"]
    with pytest.raises(FileExistsError):
        audit_corpus(corpus, output, schema=schema, catalog=catalog)


def test_partial_population_cannot_be_loaded(tmp_path, init_raw, schema, catalog):
    corpus, output = tmp_path / "corpus", tmp_path / "snapshot"
    corpus_fixture(corpus, init_raw)
    summary = audit_corpus(corpus, output, schema=schema, catalog=catalog, limit=1)
    assert not summary["population_complete"] and summary["inspected_files"] == 1
    with pytest.raises(ValueError, match="incomplete population"):
        load_snapshot(output, schema)


def test_changed_artifact_and_output_inside_corpus_are_rejected(tmp_path, init_raw, schema, catalog):
    corpus, output = tmp_path / "corpus", tmp_path / "snapshot"
    corpus_fixture(corpus, init_raw)
    with pytest.raises(ValueError, match="outside"):
        audit_corpus(corpus, corpus / "output", schema=schema, catalog=catalog)
    audit_corpus(corpus, output, schema=schema, catalog=catalog)
    (output / "splits.json").write_text("{}")
    with pytest.raises(ValueError, match="artifact changed"):
        load_snapshot(output, schema)


def test_malformed_nested_state_is_excluded_but_identity_is_grouped(tmp_path, init_raw, schema, catalog):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    document = json.loads(init_raw)
    document["settings"]["modulations"] = [{"source": [], "destination": "osc_1_level"}] * 64
    (corpus / "bad.vital").write_text(json.dumps(document))
    result = audit_corpus(corpus, tmp_path / "audit", schema=schema, catalog=catalog)
    assert result["strictly_parsed_files"] == 1
    assert result["static_eligible_files"] == 0


def test_cli_snapshot_verification_and_partial_exit(tmp_path, init_raw, capsys):
    corpus, output = tmp_path / "corpus", tmp_path / "snapshot"
    corpus_fixture(corpus, init_raw)
    assert main(["audit", "--corpus", str(corpus), "--output", str(output)]) == 0
    assert main(["verify", "--snapshot", str(output)]) == 0
    assert '"verified": true' in capsys.readouterr().out
    assert main(["audit", "--corpus", str(corpus), "--output", str(tmp_path / "partial"), "--limit", "1"]) == 2


def test_parent_provenance_is_saved_and_verified(tmp_path, init_raw, schema, catalog):
    corpus, output = tmp_path / "corpus", tmp_path / "snapshot"
    corpus_fixture(corpus, init_raw)
    document = json.loads(init_raw)
    document["settings"]["osc_1_level"] = 0.25
    variant = json.dumps(document).encode()
    (corpus / "variant.vital").write_bytes(variant)
    parent, child = PresetIdentity.from_bytes(init_raw).state, PresetIdentity.from_bytes(variant).state
    parents = {child: [parent]}
    summary = audit_corpus(corpus, output, schema=schema, catalog=catalog, parents=parents)
    assert summary["lineage_groups"] == 1
    members = json.loads((output / "members.json").read_text())
    assert members["parents"] == parents
    assert len(load_snapshot(output, schema)[2].assignments) == 2
    members["parents"] = {}
    (output / "members.json").write_text(json.dumps(members))
    status = json.loads((output / "status.json").read_text())
    status["artifact_ids"]["members.json"] = canonical_sha256(members)
    (output / "status.json").write_text(json.dumps(status))
    with pytest.raises(ValueError, match="ancestry disagrees"):
        load_snapshot(output, schema)


def test_discovery_errors_prevent_complete_population(tmp_path, init_raw, schema, catalog, monkeypatch):
    corpus, output = tmp_path / "corpus", tmp_path / "snapshot"
    corpus_fixture(corpus, init_raw)
    monkeypatch.setattr("obruxo_training.audit._discover", lambda root: ([corpus / "private-name-a.vital"], ["PermissionError"]))
    summary = audit_corpus(corpus, output, schema=schema, catalog=catalog)
    assert not summary["population_complete"] and summary["discovery_error_counts"] == {"PermissionError": 1}
    with pytest.raises(ValueError, match="incomplete population"):
        load_snapshot(output, schema)
