import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from obruxo_training.bundle_runner import bundle_file, identity, verify_bundle
from obruxo_training.cli import main
from obruxo_training.local_bundle import create_bundle
from test_prediction import artifacts  # noqa: F401


@pytest.fixture(scope="module")
def snapshots(artifacts, tmp_path_factory):  # noqa: F811
    root, _base, _frontend, _codec, _audio, _saved = artifacts
    output = tmp_path_factory.mktemp("local-model-bundles")
    for name in ("legacy", "direct", "inputs"):
        assert main(["bundle-basic", "--model", str(root / f"{name}.pt"), "--output", str(output / name)]) == 0
    return root, output


@pytest.mark.parametrize("name", ["legacy", "direct", "inputs"])
def test_packaged_model_runs_in_isolated_process_outside_checkout(snapshots, tmp_path, name):
    source, root = snapshots
    bundle = tmp_path / "relocated bundle"
    shutil.copytree(root / name, bundle)
    expected = tmp_path / "expected.vital"
    assert main(["infer", "--model", str(source / f"{name}.pt"), "--audio", str(source / "input.wav"), "--output", str(expected)]) == 0
    process = subprocess.run([sys.executable, "-I", str(bundle / "run.py"), "--audio", str(source / "input.wav"),
                              "--output", str(tmp_path / "actual.vital"), "--report", str(tmp_path / "origins.json")],
                             cwd=tmp_path, capture_output=True, text=True, timeout=60)
    assert process.returncode == 0, process.stderr
    assert (tmp_path / "actual.vital").read_bytes() == expected.read_bytes()
    report = json.loads((tmp_path / "origins.json").read_bytes())
    assert report["source_isolated"] and report["qualified_for_release"] is False
    assert report["runtime_modules"]["obruxo_training.prediction"] == "research/training/obruxo_training/prediction.py"
    assert {name.split(".", 1)[0] for name in report["runtime_modules"]} == {"obruxo_training", "obruxo_data", "obruxo_basic_pitch"}
    assert not (bundle / "research/modelling/basic_pitch/artifacts").exists()


@pytest.mark.parametrize("file", ["model.pt", "research/training/obruxo_training/prediction.py",
                                  "research/data_generation/obruxo_data/vital/schema/vital-1.0.8-vita-0.1.0/init.vital"])
def test_bundle_rejects_changed_model_code_and_schema(snapshots, tmp_path, file):
    _source, root = snapshots
    bundle = tmp_path / "changed"
    shutil.copytree(root / "direct", bundle)
    path = bundle / file
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="changed or is missing"):
        verify_bundle(bundle)


def test_bundle_refuses_existing_output_and_source_overlap(snapshots):
    source, root = snapshots
    before = (root / "direct/bundle.json").read_bytes()
    with pytest.raises(FileExistsError):
        create_bundle(source / "direct.pt", root / "direct")
    assert (root / "direct/bundle.json").read_bytes() == before
    repository = Path(__file__).resolve().parents[3]
    with pytest.raises(ValueError, match="outside the runtime source"):
        create_bundle(source / "direct.pt", repository / "research/training/obruxo_training/bad-bundle")


@pytest.mark.parametrize("name", ["../outside", "/absolute", "C:/outside", "a\\outside"])
def test_manifest_paths_cannot_leave_the_bundle(tmp_path, name):
    with pytest.raises(ValueError, match="relative path"):
        bundle_file(tmp_path, name)


def test_runtime_mismatch_is_rejected_before_importing_model(snapshots, tmp_path):
    _source, root = snapshots
    bundle = tmp_path / "wrong-runtime"
    shutil.copytree(root / "direct", bundle)
    path = bundle / "bundle.json"
    manifest = json.loads(path.read_bytes())
    manifest["runtime"]["dependencies"]["torch"] = "0.0.0"
    manifest["bundle_id"] = identity(manifest)
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="dependency version differs"):
        verify_bundle(bundle)


def test_packaged_verify_command_uses_only_manifest_and_runtime(snapshots, tmp_path):
    _source, root = snapshots
    process = subprocess.run([sys.executable, "-I", str(root / "direct/run.py"), "--verify-only"], cwd=tmp_path,
                             capture_output=True, text=True, timeout=30)
    assert process.returncode == 0, process.stderr
    assert json.loads(process.stdout)["qualified_for_release"] is False
