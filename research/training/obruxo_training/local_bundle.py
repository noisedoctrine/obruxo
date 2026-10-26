"""Create a source-isolated local snapshot of a basic inference artifact."""

import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil

from .bundle_runner import FORMAT, bundle_file, identity, verify_bundle
from .codec import PilotCodec
from .inference_profile import model_rendering
from .prediction import load_predictor
from .runtime_notices import verify as verify_notice_inventory

SOURCE_DIRS = ("research/training/obruxo_training", "research/data_generation/obruxo_data", "research/modelling/basic_pitch/obruxo_basic_pitch")
DEPENDENCIES = ("torch", "numpy", "scipy", "PyYAML", "mido", "psutil")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def create_bundle(model_path, output, *, repository=None, notice_archive=None):
    repository = Path(repository) if repository else Path(__file__).resolve().parents[3]
    sources = [repository / name for name in SOURCE_DIRS]
    if any(not source.is_dir() for source in sources):
        raise ValueError("bundle source package is missing")
    if any(output.resolve().is_relative_to(source.resolve()) for source in sources):
        raise ValueError("bundle output must be outside the runtime source directories")
    upstream = repository / "research/training/notices/upstream"
    attribution = json.loads((upstream / "manifest.json").read_bytes())
    if attribution["format"] != "obruxo_frontend_attributions_v1" or len(attribution["files"]) != 3:
        raise ValueError("frontend attribution manifest is incomplete")
    for name, row in attribution["files"].items():
        if digest(bundle_file(upstream, name)) != row["sha256"]:
            raise ValueError("frontend license or notice bytes changed")
    inventory = verify_notice_inventory(notice_archive) if notice_archive is not None else None
    model, frontend, codec, settings = load_predictor(model_path)
    if type(codec) is not PilotCodec:
        raise ValueError("local basic bundles require the 14-output basic codec")
    frontend.require_frozen()
    runtime = {"python": platform.python_version(), "dependencies": {name: importlib.metadata.version(name) for name in DEPENDENCIES}}
    if inventory:
        from .runtime_notices import canonical_name
        if any(inventory["packages"].get(canonical_name(name), {}).get("version") != version for name, version in runtime["dependencies"].items()):
            raise ValueError("notice inventory differs from the bundle's direct runtime versions")
    model_sha = digest(model_path)
    output.mkdir(parents=True, exist_ok=False)
    for source in sources:
        for path in sorted(source.rglob("*")):
            if not path.is_file() or "__pycache__" in path.parts or path.suffix not in {".py", ".json", ".vital"}:
                continue
            if path.is_symlink() or not path.resolve().is_relative_to(repository.resolve()):
                raise ValueError("runtime source leaves its repository")
            target = output / path.relative_to(repository)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
    shutil.copyfile(repository / "research/training/obruxo_training/bundle_runner.py", output / "run.py")
    shutil.copyfile(model_path, output / "model.pt")
    notices = {}
    for name, relative in (("project", "README.md"), ("basic_pitch", "research/modelling/basic_pitch/README.md")):
        source = repository / relative
        if source.is_file():
            target = output / "notices" / f"{name}-README.md"
            target.parent.mkdir(exist_ok=True)
            shutil.copyfile(source, target)
            notices[name] = target.relative_to(output).as_posix()
    for name in (*attribution["files"], "manifest.json"):
        target = output / "notices/upstream" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(bundle_file(upstream, name), target)
    notices["frontend_attributions"] = "notices/upstream/manifest.json"
    if inventory:
        for name in (*inventory["files"], "inventory.json"):
            target = output / "notices/runtime" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(bundle_file(notice_archive, name), target)
        notices["runtime_inventory"] = "notices/runtime/inventory.json"
    lock = "\n".join(f"{name}=={version}" for name, version in runtime["dependencies"].items()) + "\n"
    (output / "requirements.lock").write_text(lock, encoding="utf-8", newline="\n")
    (output / "README.md").write_text(
        "# OBRUXO local basic inference snapshot\n\n"
        "Use the Python and dependency versions recorded in bundle.json and requirements.lock. Runtime packages are separate from this snapshot.\n\n"
        "Verify files and dependencies: `python -I run.py --verify-only`\n\n"
        "Predict a patch: `python -I run.py --audio input.wav --output predicted.vital`\n\n"
        "Input is a 44.1 kHz WAV; inference receives audio only. The model, embedded frontend, codec defaults, assets and project source are included. "
        "Existing preset files are preserved. The JSON result retains the source model's qualification status.\n\n"
        "This is a local snapshot, with bundle qualification false. Model-quality confirmation, listening, clean dependency installation and release notices remain separate requirements.\n",
        encoding="utf-8", newline="\n")
    files = {path.relative_to(output).as_posix(): digest(path) for path in sorted(output.rglob("*")) if path.is_file()}
    if digest(output / "model.pt") != model_sha or digest(model_path) != model_sha:
        raise ValueError("source model changed while packaging")
    manifest = {"format": FORMAT, "qualified_for_release": False, "source_model_qualified_for_release": settings["qualified_for_release"],
                "model_format": settings["format"], "model_file_sha256": model_sha, "dense_outputs": 14, "codec_values": 11,
                "input": "audio_only", "frontend": frontend.describe(), "rendering": model_rendering(model),
                "runtime": runtime, "files": files, "notices": notices, "release_notices_complete": False,
                "runtime_notice_inventory_id": inventory["inventory_id"] if inventory else None}
    manifest["bundle_id"] = identity(manifest)
    (output / "bundle.json").write_text(json.dumps(manifest, sort_keys=True, indent=2, allow_nan=False), encoding="utf-8", newline="\n")
    verify_bundle(output)
    return {"bundle_id": manifest["bundle_id"], "files": len(files), "model_file_sha256": model_sha,
            "qualified_for_release": False, "source_model_qualified_for_release": settings["qualified_for_release"], "output": str(output)}
