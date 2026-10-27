"""Standalone launcher copied into a local inference snapshot."""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path, PurePosixPath
import platform
import sys

FORMAT = "obruxo_local_basic_bundle_v1"
PACKAGES = ("obruxo_training", "obruxo_data", "obruxo_basic_pitch")


def identity(value):
    document = {key: item for key, item in value.items() if key != "bundle_id"}
    return hashlib.sha256(json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def bundle_file(root, name):
    path = PurePosixPath(name)
    if not name or path.is_absolute() or path.as_posix() != name or ".." in path.parts or ":" in name or "\\" in name:
        raise ValueError("bundle contains an invalid relative path")
    target = (root / name).resolve()
    if not target.is_relative_to(root.resolve()):
        raise ValueError("bundle path leaves its directory")
    return target


def verify_bundle(root, *, check_dependencies=True):
    manifest = json.loads((root / "bundle.json").read_bytes())
    if manifest["format"] != FORMAT or manifest["bundle_id"] != identity(manifest) or manifest["qualified_for_release"] is not False:
        raise ValueError("unsupported or changed local bundle manifest")
    if not manifest["files"].get("run.py") or manifest["files"].get("model.pt") != manifest["model_file_sha256"]:
        raise ValueError("bundle is missing its model or launcher identity")
    for name, expected in manifest["files"].items():
        path = bundle_file(root, name)
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"bundle file changed or is missing: {name}")
    if check_dependencies:
        if platform.python_version() != manifest["runtime"]["python"]:
            raise ValueError("bundle Python version differs from its recorded runtime")
        for name, expected in manifest["runtime"]["dependencies"].items():
            try:
                actual = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError as error:
                raise ValueError(f"bundle dependency is missing: {name}") from error
            if actual != expected:
                raise ValueError(f"bundle dependency version differs: {name}")
    return manifest


def runtime_origins(root):
    result = {}
    for name, module in tuple(sys.modules.items()):
        if name.split(".", 1)[0] not in PACKAGES or not getattr(module, "__file__", None):
            continue
        path = Path(module.__file__).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError(f"inference imported project code outside its bundle: {name}")
        result[name] = path.relative_to(root.resolve()).as_posix()
    return result


def windows_runtime_libraries():
    """Keep a venv's oneMKL libraries ahead of its Conda base interpreter."""
    if sys.platform != "win32" or sys.prefix == sys.base_prefix:
        return []
    native = Path(sys.prefix) / "Library/bin"
    if not native.is_dir():
        return []
    import ctypes
    import os
    torch_lib = Path(importlib.metadata.distribution("torch").locate_file("torch/lib"))
    handles = [os.add_dll_directory(str(path)) for path in (native, torch_lib) if path.is_dir()]
    os.environ["PATH"] = os.pathsep.join((str(native), str(torch_lib), os.environ.get("PATH", "")))
    # Torch adds both base and venv directories. Windows does not order those
    # directories, so pin the installed venv copies before Torch loads them.
    handles.extend(ctypes.WinDLL(str(path)) for path in sorted(native.glob("mkl_sycl_*.dll")))
    return handles


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run a local basic model using only its packaged project source and data.")
    parser.add_argument("--audio", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parent
    try:
        manifest = verify_bundle(root)
        if args.verify_only:
            print(json.dumps({"bundle_id": manifest["bundle_id"], "files_verified": len(manifest["files"]), "qualified_for_release": False}))
            return 0
        if args.audio is None or args.output is None:
            parser.error("--audio and --output are required unless --verify-only is set")
        if args.report and args.report.exists():
            raise FileExistsError(f"report already exists: {args.report}")
        sys.path[:0] = [str(root / "research/training"), str(root / "research/data_generation"), str(root / "research/modelling/basic_pitch")]
        _runtime_libraries = windows_runtime_libraries()
        import torch
        from obruxo_training.cli import _read_inference_audio
        from obruxo_training.inference_profile import prediction_preset
        from obruxo_training.prediction import load_predictor
        from obruxo_training.trainer import predict_audio
        model, frontend, codec, settings = load_predictor(root / "model.pt", args.device)
        torch.set_num_threads(settings["cpu_threads"])
        runtime_origins(root)
        sample_rate, audio = _read_inference_audio(args.audio)
        target = predict_audio(model, frontend, codec, audio, sample_rate)
        preset = prediction_preset(model, codec, target)
        origins = runtime_origins(root)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("xb") as handle:
            handle.write(preset)
        result = {"bundle_id": manifest["bundle_id"], "output": str(args.output), "target": target.to_dict(),
                  "qualified_for_release": False, "source_model_qualified_for_release": settings["qualified_for_release"],
                  "source_isolated": True, "runtime_modules": origins}
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            with args.report.open("x", encoding="utf-8") as handle:
                json.dump(result, handle, sort_keys=True, indent=2, allow_nan=False)
        print(json.dumps({key: value for key, value in result.items() if key != "runtime_modules"}))
        return 0
    except (OSError, ValueError, TypeError, KeyError, RuntimeError) as error:
        parser.exit(1, f"Inference failed: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
