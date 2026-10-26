import base64
import csv
import hashlib
import json
from pathlib import Path

import pytest

from obruxo_training.runtime_notices import collect, identity, notice_path, verify


def installed(tmp_path, packages=None):
    environment = tmp_path / "environment"
    site = environment / "site-packages"
    lock = tmp_path / "runtime.lock"
    pins = []
    for name, version in packages or [("Example_Package", "1.0")]:
        metadata = site / f"{name}-{version}.dist-info"
        metadata.mkdir(parents=True)
        (metadata / "licenses").mkdir()
        (metadata / "licenses/LICENSE.txt").write_bytes(b"Synthetic license fixture\n")
        (metadata / "licenses/NOTICE").write_bytes(b"Synthetic notice fixture\n")
        (metadata / "METADATA").write_text(f"Name: {name}\nVersion: {version}\nLicense-Expression: MIT\nLicense-File: LICENSE.txt\nLicense-File: NOTICE\n", encoding="utf-8")
        with (metadata / "RECORD").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            for file in sorted(metadata.rglob("*")):
                if not file.is_file() or file.name == "RECORD":
                    continue
                raw = file.read_bytes()
                digest = base64.urlsafe_b64encode(hashlib.sha256(raw).digest()).rstrip(b"=").decode()
                writer.writerow([file.relative_to(site).as_posix(), "sha256=" + digest, len(raw)])
        pins.append(f"{name}=={version} --hash=sha256:{'a' * 64}\n")
    lock.write_text("".join(pins), encoding="utf-8")
    return environment, site, lock


def test_collects_original_texts_for_every_locked_distribution_and_checks_all_files(tmp_path):
    env, site, lock = installed(tmp_path)
    root = tmp_path / "notices"
    result = collect(lock, env, site, root)
    assert set(result["packages"]) == {"example-package"}
    assert result["packages"]["example-package"]["license_expression"] == "MIT"
    assert len(result["files"]) == 3 and not result["release_notices_complete"]
    assert verify(root) == result
    assert (root / result["packages"]["example-package"]["files"][0]["file"]).read_bytes() == b"Synthetic license fixture\n"


def test_license_library_code_and_generated_caches_are_not_notice_texts():
    assert not notice_path(Path("packaging/licenses/__init__.py"))
    assert not notice_path(Path("packaging/licenses/__pycache__/_spdx.cpython-312.pyc"))
    assert notice_path(Path("example.dist-info/licenses/vendor/README"))
    assert notice_path(Path("numpy/random/dragon4_LICENSE.txt"))


def test_changed_record_bytes_or_missing_declared_notice_fail_before_output(tmp_path):
    env, site, lock = installed(tmp_path)
    license_file = next(site.rglob("LICENSE.txt"))
    license_file.write_bytes(b"Changed license fixture")
    output = tmp_path / "notices"
    with pytest.raises(ValueError, match="distribution RECORD"):
        collect(lock, env, site, output)
    assert not output.exists()
    license_file.unlink()
    with pytest.raises(ValueError, match="declared license files"):
        collect(lock, env, site, output)
    assert not output.exists()


def test_missing_or_different_locked_distribution_is_not_silently_skipped(tmp_path):
    env, site, lock = installed(tmp_path)
    lock.write_text(f"absent==1.0 --hash=sha256:{'a' * 64}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing or different"):
        collect(lock, env, site, tmp_path / "notices")


def test_inventory_detects_missing_file_coverage_even_if_identity_is_recomputed(tmp_path):
    env, site, lock = installed(tmp_path)
    output = tmp_path / "notices"
    data = collect(lock, env, site, output)
    target = data["packages"]["example-package"]["files"][0]["file"]
    data["files"].pop(target)
    data["inventory_id"] = identity(data)
    (output / "inventory.json").write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="file coverage changed"):
        verify(output)


def test_inventory_detects_text_changes_and_preserves_existing_output(tmp_path):
    env, site, lock = installed(tmp_path)
    output = tmp_path / "notices"
    data = collect(lock, env, site, output)
    with pytest.raises(FileExistsError):
        collect(lock, env, site, output)
    target = data["packages"]["example-package"]["files"][0]["file"]
    (output / target).write_bytes(b"Changed text")
    with pytest.raises(ValueError, match="file changed"):
        verify(output)
