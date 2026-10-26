"""Verify one snapshot prediction in a fresh dependency environment."""

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import runpy
import site
import sys


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--expected-preset-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not sys.flags.isolated or sys.prefix == sys.base_prefix:
        parser.error("use the new virtual environment's Python with -I")
    prefix, bundle = Path(sys.prefix).resolve(), args.bundle.resolve()
    if any(not Path(path).resolve().is_relative_to(prefix) for path in site.getsitepackages()):
        parser.error("the environment must not share site-packages")
    if len(args.expected_preset_sha256) != 64 or any(char not in "0123456789abcdef" for char in args.expected_preset_sha256):
        parser.error("expected preset identity must be a lowercase SHA-256")
    args.output.mkdir(parents=True, exist_ok=False)
    preset, origins = args.output / "prediction.vital", args.output / "origins.json"
    sys.argv = [str(bundle / "run.py"), "--audio", str(args.audio.resolve()), "--output", str(preset.resolve()), "--report", str(origins.resolve())]
    status = 0
    try:
        runpy.run_path(str(bundle / "run.py"), run_name="__main__")
    except SystemExit as error:
        status = error.code
    modules = {name: str(Path(importlib.import_module(name).__file__).resolve()) for name in ("torch", "numpy", "scipy", "yaml", "mido", "psutil")}
    external_modules = {name: path for name, path in modules.items() if not Path(path).is_relative_to(prefix)}
    psutil = importlib.import_module("psutil")
    tokens = ("torch", "mkl", "sycl", "libiomp", "tbb", "tcmlib", "umf", "ur_loader", "openblas")
    libraries = sorted({str(Path(item.path).resolve()) for item in psutil.Process().memory_maps()
                        if any(token in Path(item.path).name.lower() for token in tokens) and Path(item.path).is_file()})
    external_libraries = [path for path in libraries if not Path(path).is_relative_to(prefix)]
    actual = digest(preset) if preset.is_file() else None
    report = {"format": "obruxo_clean_runtime_check_v1", "passed": bool(status == 0 and actual == args.expected_preset_sha256 and not external_modules and not external_libraries),
              "inference_exit_code": status, "preset_sha256": actual, "expected_preset_sha256": args.expected_preset_sha256,
              "audio_sha256": digest(args.audio), "bundle_manifest_sha256": digest(bundle / "bundle.json"),
              "prefix": sys.prefix, "base_prefix": sys.base_prefix, "isolated": sys.flags.isolated,
              "site_packages": site.getsitepackages(), "user_site_enabled": site.ENABLE_USER_SITE,
              "dependency_modules": modules, "external_dependency_modules": external_modules,
              "model_native_libraries": libraries, "external_model_native_libraries": external_libraries,
              "new_native_requests": 0, "optimizer_applications": 0, "qualified_for_release": False}
    (args.output / "runtime.json").write_text(json.dumps(report, sort_keys=True, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("passed", "preset_sha256", "external_dependency_modules", "external_model_native_libraries")}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
