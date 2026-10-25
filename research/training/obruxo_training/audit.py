from __future__ import annotations

from collections import Counter
import json
import os
from pathlib import Path
import re

from obruxo_data.hashing import canonical_sha256
from obruxo_data.vital import VitalSchema

from .assets import AssetCatalog, AssetReference
from .atomic_io import replace_file
from .contracts import TrainingContract
from .eligibility import inspect_preset
from .identity import PresetIdentity
from .performances import bank_document_for
from .splits import SplitManifest, build_splits, normalize_parents


def write_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=True, allow_nan=False, sort_keys=True, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    replace_file(temporary, path)


def _discover(root: Path) -> tuple[list[Path], list[str]]:
    paths, errors = [], []
    for directory, directories, files in os.walk(root, followlinks=False, onerror=lambda error: errors.append(type(error).__name__)):
        linked = {name for name in directories if (Path(directory) / name).is_symlink() or (Path(directory) / name).is_junction()}
        errors.extend("linked_directory_excluded" for _ in linked)
        directories[:] = sorted(set(directories) - linked)
        paths.extend(Path(directory) / name for name in sorted(files) if Path(name).suffix.lower() == ".vital")
    return paths, errors


def _exclusion(error: Exception) -> str:
    if isinstance(error, UnicodeDecodeError):
        return "invalid_utf8"
    if isinstance(error, json.JSONDecodeError):
        return "invalid_json"
    if isinstance(error, OSError):
        return "read_error"
    for text, reason in (("duplicate JSON key", "duplicate_json_key"), ("nonfinite JSON constant", "nonfinite_json"),
                         ("preset root must be an object", "non_object_root")):
        if text in str(error):
            return reason
    return "invalid_document"


def audit_corpus(corpus: Path, output: Path, *, schema: VitalSchema, catalog: AssetCatalog, seed: int = 0,
                 parents: dict[str, list[str]] | None = None, limit: int | None = None, progress=None) -> dict:
    root, output = corpus.resolve(strict=True), output.resolve()
    if not root.is_dir() or output.is_relative_to(root):
        raise ValueError("output must be outside the source corpus directory")
    if limit is not None and (type(limit) is not int or limit <= 0):
        raise ValueError("limit must be a positive integer")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    parents = normalize_parents(parents)
    contract = TrainingContract.create(schema, catalog)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "status.json", {"complete": False, "stage": "audit"})
    paths, discovery_errors = _discover(root)
    selected = paths if limit is None else paths[:limit]
    rows, exclusions = [], []
    for index, path in enumerate(selected, 1):
        relative = path.relative_to(root).as_posix()
        raw = b""
        try:
            if path.is_symlink():
                raise ValueError("symbolic preset links are excluded")
            raw = path.read_bytes()
            row = inspect_preset(raw, schema, catalog)
            row["relative_path"] = relative
            rows.append(row)
        except (OSError, ValueError, TypeError, KeyError, OverflowError, RecursionError) as error:
            reason = "symlink" if path.is_symlink() else _exclusion(error)
            # Preserve strict parse identity even when a malformed nested structure defeats validation.
            try:
                identity = PresetIdentity.from_bytes(raw).to_dict() if reason not in {"read_error", "symlink"} else None
            except (ValueError, TypeError, RecursionError):
                identity = None
            exclusions.append({"relative_path": relative, "reason": reason, "identity": identity})
        if progress and (index % 250 == 0 or index == len(selected)):
            progress(index, len(selected))
    all_identities = [row["identity"] for row in rows + exclusions if row.get("identity")]
    manifest = build_splits((identity["state"] for identity in all_identities), parents=parents, seed=seed)
    for row in rows:
        entry = manifest.by_state[row["identity"]["state"]]
        row.update(lineage_id=entry.lineage_id, split=entry.split)
    contract.require_match(TrainingContract.create(schema, catalog).to_dict())
    complete = not discovery_errors and len(selected) == len(paths)
    summary = summarize(rows, exclusions, manifest, len(paths), complete, discovery_errors)
    summary.update(contract_id=contract.contract_id, split_manifest_id=manifest.manifest_id, catalog_id=catalog.catalog_id)
    write_json(output / "contract.json", contract.to_dict())
    write_json(output / "assets.json", catalog.to_dict())
    write_json(output / "performances.json", bank_document_for(contract.renderer_id))
    write_json(output / "splits.json", manifest.to_dict())
    write_json(output / "members.json", {"version": 1, "parents": parents, "records": rows, "exclusions": exclusions})
    write_json(output / "summary.json", summary)
    (output / "REPORT.md").write_text(render_report(summary), encoding="utf-8")
    artifacts = ("contract.json", "assets.json", "performances.json", "splits.json", "members.json", "summary.json")
    write_json(output / "status.json", {"complete": complete, "stage": "complete" if complete else "partial",
                                        "artifact_ids": {name: canonical_sha256(json.loads((output / name).read_text())) for name in artifacts}})
    return summary


def load_snapshot(root: Path, schema: VitalSchema) -> tuple[TrainingContract, AssetCatalog, SplitManifest]:
    status = json.loads((root / "status.json").read_text(encoding="utf-8"))
    if status.get("complete") is not True or status.get("stage") != "complete":
        raise ValueError("cannot load an incomplete population snapshot")
    names = ("contract.json", "assets.json", "performances.json", "splits.json", "members.json", "summary.json")
    if set(status.get("artifact_ids", {})) != set(names):
        raise ValueError("snapshot artifact inventory is incomplete")
    documents = {}
    for name in names:
        value = json.loads((root / name).read_text(encoding="utf-8"))
        if canonical_sha256(value) != status["artifact_ids"][name]:
            raise ValueError(f"snapshot artifact changed: {name}")
        documents[name] = value
    catalog = AssetCatalog(tuple(AssetReference(**entry) for entry in documents["assets.json"]["references"]))
    if documents["assets.json"] != catalog.to_dict():
        raise ValueError("unsupported asset catalog contract")
    contract = TrainingContract.create(schema, catalog)
    contract.require_match(documents["contract.json"])
    if documents["performances.json"] != bank_document_for(contract.renderer_id):
        raise ValueError("performance bank changed")
    manifest = SplitManifest.from_dict(documents["splits.json"])
    members = documents["members.json"]
    states = [row["identity"]["state"] for row in members["records"] + members["exclusions"] if row.get("identity")]
    rebuilt = build_splits(states, parents=members["parents"], seed=manifest.seed)
    if members.get("version") != 1 or rebuilt.manifest_id != manifest.manifest_id:
        raise ValueError("snapshot membership or ancestry disagrees with its split manifest")
    for row in members["records"]:
        assignment = manifest.by_state[row["identity"]["state"]]
        if (row["lineage_id"], row["split"]) != (assignment.lineage_id, assignment.split):
            raise ValueError("snapshot member has an invalid split assignment")
    summary = documents["summary.json"]
    if (summary.get("contract_id"), summary.get("split_manifest_id"), summary.get("catalog_id")) != (
        contract.contract_id, manifest.manifest_id, catalog.catalog_id
    ):
        raise ValueError("snapshot identities disagree")
    return contract, catalog, manifest


def summarize(rows, exclusions, manifest, discovered, complete, discovery_errors) -> dict:
    identities = [row["identity"] for row in rows + exclusions if row.get("identity")]
    eligible = [row for row in rows if row["static_eligible"]]
    version_counts = Counter(row["synth_version"] if re.fullmatch(r"\d{1,5}\.\d{1,3}\.\d{1,3}", row["synth_version"])
                             else "unrecognized" for row in rows)
    unsupported = Counter(name if re.fullmatch(r"[a-z][a-z_0-9]{0,79}", name) else "unrecognized_field"
                          for row in rows for name in row["unsupported_fields"])
    return {
        "version": 1, "population_complete": complete, "discovered_files": discovered,
        "inspected_files": len(rows) + len(exclusions), "strictly_parsed_files": len(identities),
        "validated_records": len(rows), "excluded_files": len(exclusions),
        "exclusion_counts": dict(sorted(Counter(row["reason"] for row in exclusions).items())),
        "discovery_error_counts": dict(sorted(Counter(discovery_errors).items())),
        "identity_groups": {level: len({identity[level] for identity in identities}) for level in ("raw", "canonical", "state")},
        "lineage_groups": len({entry.lineage_id for entry in manifest.assignments}),
        "static_eligible_files": len(eligible), "static_eligible_states": len({row["identity"]["state"] for row in eligible}),
        "eligible_states_by_split": {split: len({row["identity"]["state"] for row in eligible if row["split"] == split})
                                     for split in ("train", "validation", "test")},
        "ineligibility_counts": dict(sorted(Counter(reason for row in rows for reason in row["reasons"]).items())),
        "versions": dict(sorted(version_counts.items())), "unsupported_field_counts": dict(sorted(unsupported.items())),
        "audio_qa": "not_run", "near_duplicate_ancestry": "not_inferred",
    }


def render_report(summary: dict) -> str:
    rows = "\n".join(f"| {key} | {summary[key]} |" for key in (
        "discovered_files", "inspected_files", "strictly_parsed_files", "excluded_files",
        "lineage_groups", "static_eligible_files", "static_eligible_states"))
    reasons = "\n".join(f"| {key} | {count} |" for key, count in summary["ineligibility_counts"].items())
    return ("# Training contract eligibility audit\n\n"
            f"Population complete: `{str(summary['population_complete']).lower()}`. Audio QA: **not run**.\n\n"
            f"| Measure | Count |\n|---|---:|\n{rows}\n\n"
            f"| Static ineligibility reason (overlapping) | Files |\n|---|---:|\n{reasons}\n\n"
            "Eligibility uses exact registered asset content and the reviewed schema; it does not certify audibility, "
            "factory origin of caller-declared references, or a finished codec. The default catalog covers the reviewed "
            "init assets only. Unresolved content is not automatically imported or edited. Near-duplicate ancestry "
            "is not inferred. Declared parents and exact stored-state duplicates share one split.\n\n"
            f"Eligible independent states by split: `{json.dumps(summary['eligible_states_by_split'], sort_keys=True)}`.\n\n"
            "Private membership paths stay in `members.json`; this report contains aggregate counts only. "
            "A partial audit must not freeze a training population.\n")
