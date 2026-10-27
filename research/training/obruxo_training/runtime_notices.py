"""Collect original license/notice bytes from an explicitly locked local runtime."""

import base64
import hashlib
import importlib.metadata
import json
from pathlib import Path, PurePosixPath
import re

FORMAT = "obruxo_runtime_notice_inventory_v1"


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def identity(manifest):
    return digest(json.dumps({key: value for key, value in manifest.items() if key != "inventory_id"}, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())


def canonical_name(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def locked_versions(lock):
    result = {}
    for line in lock.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        match = re.fullmatch(r"([A-Za-z0-9_.-]+)==([^\s]+) --hash=sha256:([0-9a-f]{64})", line)
        if not match or canonical_name(match[1]) in result:
            raise ValueError("notice inventory requires unique exact versions and wheel hashes")
        result[canonical_name(match[1])] = {"version": match[2], "wheel_sha256": match[3]}
    if not result:
        raise ValueError("notice runtime lock is empty")
    return result


def notice_path(path):
    pattern = r"(^|[_-])(licen[cs]e|notice|copying|authors)([._-]|$)"
    in_metadata_licenses = any(part.lower() in {"licenses", "licences"} and any(parent.endswith(".dist-info") for parent in path.parts[:index])
                               for index, part in enumerate(path.parts[:-1]))
    return bool(re.search(pattern, path.name, re.IGNORECASE) or in_metadata_licenses)


def collect(lock, environment, site_packages, output):
    if output.exists():
        raise FileExistsError(output)
    environment, site_packages = environment.resolve(), site_packages.resolve()
    if not site_packages.is_relative_to(environment) or output.resolve().is_relative_to(environment):
        raise ValueError("notice input or output overlaps the installed runtime")
    installed = {}
    for distribution in importlib.metadata.distributions(path=[str(site_packages)]):
        name = canonical_name(distribution.metadata["Name"])
        if name in installed:
            raise ValueError("ambiguous installed notice distribution")
        installed[name] = distribution
    locked, packages, contents = locked_versions(lock), {}, {}
    for name, pin in sorted(locked.items()):
        distribution = installed.get(name)
        if distribution is None or distribution.version != pin["version"]:
            raise ValueError(f"locked notice package missing or different: {name}")
        entries = sorted((path for path in distribution.files or [] if notice_path(path)), key=str)
        declared = distribution.metadata.get_all("License-File", [])
        if not entries or any(not any(str(path).endswith("/" + value) for path in entries) for value in declared):
            raise ValueError(f"installed package lacks recorded or declared license files: {name}")
        files = []
        for index, entry in enumerate(entries):
            source = Path(distribution.locate_file(entry)).resolve()
            if not source.is_relative_to(environment) or not source.is_file():
                raise ValueError("installed notice path leaves its runtime or is missing")
            raw = source.read_bytes()
            if not raw or len(raw) > 8 * 1024**2:
                raise ValueError("installed notice is empty or exceeds its size bound")
            recorded = entry.hash
            if recorded is None or recorded.mode != "sha256" or base64.urlsafe_b64encode(hashlib.sha256(raw).digest()).rstrip(b"=").decode() != recorded.value:
                raise ValueError(f"installed notice differs from its distribution RECORD: {name}/{entry}")
            basename = re.sub(r"[^A-Za-z0-9_.-]", "_", entry.name)
            target = f"packages/{name}/{index:03d}-{basename}"
            contents[target] = raw
            files.append({"installed_path": str(entry), "file": target, "sha256": digest(raw), "bytes": len(raw)})
        packages[name] = {"name": distribution.metadata["Name"], **pin, "license_expression": distribution.metadata.get("License-Expression"),
                          "declared_license_files": declared, "files": files}
    if sum(map(len, contents.values())) > 64 * 1024**2:
        raise ValueError("runtime notices exceed the 64 MiB allocation")
    output.mkdir(parents=True, exist_ok=False)
    for name, raw in contents.items():
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    (output / "requirements.lock").write_bytes(lock.read_bytes())
    manifest = {"format": FORMAT, "lock_sha256": digest(lock.read_bytes()), "packages": packages,
                "files": {name: digest(raw) for name, raw in contents.items()}, "release_notices_complete": False,
                "scope": "original_installed_runtime_notice_bytes; wheel_pins_recorded; installed_files_checked_against_RECORD"}
    manifest["files"]["requirements.lock"] = digest(lock.read_bytes())
    manifest["inventory_id"] = identity(manifest)
    (output / "inventory.json").write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    verify(output)
    return manifest


def verify(root):
    manifest = json.loads((root / "inventory.json").read_bytes())
    if manifest["format"] != FORMAT or manifest["inventory_id"] != identity(manifest) or manifest["release_notices_complete"] is not False:
        raise ValueError("runtime notice inventory identity changed")
    for name, expected in manifest["files"].items():
        relative = PurePosixPath(name)
        target = (root / name).resolve()
        if (relative.is_absolute() or relative.as_posix() != name or ".." in relative.parts or ":" in name or "\\" in name
                or not target.is_relative_to(root.resolve()) or not target.is_file() or digest(target.read_bytes()) != expected):
            raise ValueError("runtime notice file changed, is missing or leaves its inventory")
    locked = locked_versions(root / "requirements.lock")
    if digest((root / "requirements.lock").read_bytes()) != manifest["lock_sha256"] or locked.keys() != manifest["packages"].keys():
        raise ValueError("runtime notice package coverage differs from its lock")
    declared = {"requirements.lock": manifest["lock_sha256"]}
    for name, package in manifest["packages"].items():
        if package["version"] != locked[name]["version"] or package["wheel_sha256"] != locked[name]["wheel_sha256"] or not package["files"]:
            raise ValueError("runtime notice package pin or coverage changed")
        declared.update({row["file"]: row["sha256"] for row in package["files"]})
    if declared != manifest["files"]:
        raise ValueError("runtime notice file coverage changed")
    return manifest
