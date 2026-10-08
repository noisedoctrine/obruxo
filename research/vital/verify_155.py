"""Validate one preset or audit every file in a corpus against the 1.5.5 contract."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from full_contract import FullPresetContract, strict_json


def audit(corpus):
    contract = FullPresetContract()
    counts, versions, errors, fields = Counter(), Counter(), Counter(), Counter()
    examples, identities = [], []
    for index, path in enumerate(sorted(corpus.rglob("*.vital"))):
        if index % 1000 == 0:
            print(f"Verified {index} files", flush=True)
        counts["discovered"] += 1
        try:
            with path.open("rb") as stream:
                raw = stream.read(contract.spec["max_file_bytes"] + 1)
            if len(raw) > contract.spec["max_file_bytes"]:
                raise ValueError("file size limit exceeded")
            document = strict_json(raw.decode("utf-8-sig"))
            if not isinstance(document, dict):
                raise ValueError("preset is not an object")
        except (OSError, ValueError, RecursionError):
            counts["unreadable_or_invalid_json"] += 1
            continue
        version = document.get("synth_version")
        versions[str(version)] += 1
        if version != "1.5.5":
            counts["other_version_not_validated"] += 1
            continue
        digest = hashlib.sha256(raw).hexdigest()
        identities.append(digest)
        counts["target_1_5_5"] += 1
        structure = contract.validate(document, authoring=False)
        authoring = structure.merge(contract.validate_ranges(document)) if isinstance(document.get("settings"), dict) else structure
        counts["structural_pass" if structure.valid else "structural_fail"] += 1
        counts["authoring_pass" if authoring.valid else "authoring_fail"] += 1
        for item in {d.code for d in authoring.diagnostics}:
            errors[item] += 1
        for item in {d.pointer for d in authoring.diagnostics}:
            fields[str(item)] += 1
        if not authoring.valid and len(examples) < 12:
            examples.append({"sha256": digest, "diagnostics": authoring.to_dict()["diagnostics"][:8]})
    digest = hashlib.sha256("\n".join(sorted(identities)).encode()).hexdigest()
    return {"contract_id": contract.spec["id"], "counts": dict(counts), "versions": dict(sorted(versions.items())),
            "corpus_content_digest": digest, "matches_build_corpus": digest == contract.spec["evidence"]["corpus_content_digest"],
            "files_by_error_code": dict(errors.most_common()), "files_by_error_pointer": dict(fields.most_common()),
            "examples": examples, "native_rendering_performed": False,
            "validator_sha256": hashlib.sha256(Path(__file__).with_name("full_contract.py").read_bytes().replace(b"\r\n", b"\n")).hexdigest(),
            "interpretation": "Structural checks include payload lengths and relationships. Authoring additionally enforces measured renderer scalar bounds. A rejection can mean unsupported authoring data, not a broken Vital preset."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--corpus", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--structure-only", action="store_true")
    args = parser.parse_args()
    if args.corpus:
        result = audit(args.path)
        code = 0  # An audit completes even when individual corpus files are rejected.
    else:
        _, report = FullPresetContract().read(args.path, authoring=not args.structure_only)
        result, code = report.to_dict(), 0 if report.valid else 1
    text = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(text)
    print(text)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
