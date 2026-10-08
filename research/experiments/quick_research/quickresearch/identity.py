from __future__ import annotations

from collections import Counter, defaultdict
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path

from .common import BudgetExpired, Context, file_hash, finish_report, write_csv, write_json

RULE = "patch_identity_v1"
DISPLAY_FIELDS = frozenset({"author", "comments", "preset_name"})
LEVELS = ("raw", "canonical", "state")


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def reject_constant(value):
    raise ValueError("nonfinite JSON constant")


def parse_document(raw: bytes) -> dict:
    value = json.loads(raw.decode("utf-8-sig"), parse_float=Decimal,
                       object_pairs_hook=unique_object, parse_constant=reject_constant)
    if not isinstance(value, dict):
        raise ValueError("preset root must be an object")
    return value


def typed_tree(value):
    if isinstance(value, dict):
        return ["object", [[key, typed_tree(value[key])] for key in sorted(value)]]
    if isinstance(value, list):
        return ["array", [typed_tree(item) for item in value]]
    if isinstance(value, Decimal):
        return ["decimal", str(value)]
    if value is None:
        return ["null"]
    if isinstance(value, bool):
        return ["bool", value]
    if isinstance(value, int):
        return ["integer", str(value)]
    if isinstance(value, str):
        return ["string", value]
    raise TypeError(type(value).__name__)


def document_hash(document: dict, *, ignore_display: bool = False) -> str:
    if ignore_display:
        document = {key: value for key, value in document.items() if key not in DISPLAY_FIELDS}
    payload = json.dumps(typed_tree(document), ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(f"{RULE}:ignore_display={ignore_display}:{payload}".encode()).hexdigest()


def identities(raw: bytes) -> dict[str, str]:
    document = parse_document(raw)
    return {"raw": hashlib.sha256(raw).hexdigest(), "canonical": document_hash(document),
            "state": document_hash(document, ignore_display=True)}


def simulated_split(identifier: str, seed: int = 0) -> str:
    value = int(hashlib.sha256(f"{seed}:{identifier}".encode()).hexdigest()[:16], 16) / 2**64
    return "train" if value < 0.8 else "validation" if value < 0.9 else "test"


def summarize(rows: list[dict]) -> dict:
    result = {}
    for level in LEVELS:
        groups = defaultdict(list)
        for row in rows:
            groups[row[level]].append(row)
        crossing = [members for members in groups.values() if len({row["simulated_split"] for row in members}) > 1]
        previous = "raw" if level == "canonical" else "canonical"
        introduced = [members for members in groups.values() if len({row[previous] for row in members}) > 1] if level != "raw" else []
        result[level] = {"groups": len(groups), "duplicate_excess": len(rows) - len(groups),
                         "group_size_counts": dict(Counter(str(len(members)) for members in groups.values())),
                         "crossing_groups": len(crossing), "crossing_files": sum(map(len, crossing)),
                         "introduced_groups": len(introduced)}
    return result


def diff_paths(left, right, prefix: str = "", limit: int = 40) -> list[str]:
    if typed_tree(left) == typed_tree(right):
        return []
    if isinstance(left, dict) and isinstance(right, dict):
        result = []
        for key in sorted(left.keys() | right.keys()):
            pointer = prefix + "/" + key.replace("~", "~0").replace("/", "~1")
            result.extend([pointer] if key not in left or key not in right else diff_paths(left[key], right[key], pointer))
            if len(result) >= limit:
                break
        return result[:limit]
    return [prefix or "/"]


def run(context: Context, args) -> dict:
    root = Path(args.corpus).resolve(strict=True)
    if context.output.is_relative_to(root):
        raise ValueError("output cannot be inside the corpus")
    if not root.is_dir():
        raise ValueError("corpus must be a directory")
    members_path = context.output / "members.jsonl"
    previous = {}
    if members_path.exists():
        previous = {row["relative_path"]: row for row in map(json.loads, members_path.read_text().splitlines())}
    rows, exclusions, examples = [], [], []
    representatives = {level: {} for level in ("canonical", "state")}
    complete = True
    temporary = members_path.with_suffix(".jsonl.partial")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            for directory, dirs, files in os.walk(root, followlinks=False):
                dirs[:] = sorted(name for name in dirs if not (Path(directory) / name).is_symlink())
                for name in sorted(files):
                    context.check(5)
                    if Path(name).suffix.lower() != ".vital":
                        continue
                    path = Path(directory) / name
                    relative = path.relative_to(root).as_posix()
                    if path.is_symlink():
                        exclusions.append({"relative_path": relative, "reason": "symlink_excluded"})
                        continue
                    try:
                        raw = path.read_bytes()
                        raw_hash = hashlib.sha256(raw).hexdigest()
                        saved = previous.get(relative)
                        ids = {key: saved[key] for key in LEVELS} if saved and saved["raw"] == raw_hash else identities(raw)
                        row = {"relative_path": relative, **ids, "simulated_split": simulated_split(relative, args.seed)}
                        rows.append(row)
                        stream.write(json.dumps(row) + "\n")
                        stream.flush()
                        for level, parent in (("canonical", "raw"), ("state", "canonical")):
                            other = representatives[level].setdefault(row[level], row)
                            if other[parent] != row[parent] and len(examples) < 20:
                                changed = diff_paths(parse_document((root / other["relative_path"]).read_bytes()), parse_document(raw))
                                examples.append({"level": level, "group_id": row[level], "changed_json_pointers": changed,
                                                 "interpretation": "formatting-only" if not changed else "declared display fields only"})
                    except (OSError, ValueError, TypeError, UnicodeError) as error:
                        exclusions.append({"relative_path": relative, "reason": type(error).__name__})
                    context.progress(completed_files=len(rows), excluded_files=len(exclusions))
    except BudgetExpired:
        complete = False
    os.replace(temporary, members_path)
    summary = {"rule": RULE, "complete": complete, "valid_files": len(rows), "excluded_files": len(exclusions),
               "seed": args.seed, "split_policy": "simulated file-relative-path hash, expected 80/10/10; not an existing training split",
               "levels": summarize(rows), "introduced_examples": examples,
               "reference_census_valid_files": 9618, "difference_from_reference": len(rows) - 9618 if complete else None}
    write_json(context.output / "identity_summary.json", summary)
    write_json(context.output / "identity_manifest.json", {"rule": RULE, "corpus_root": str(root), "complete": complete,
               "members_sha256": file_hash(members_path), "members_file": "members.jsonl", "valid_files": len(rows)})
    write_csv(context.output / "exclusions.csv", exclusions)
    table = "\n".join(f"| {level} | {value['groups']} | {value['duplicate_excess']} | {value['crossing_groups']} | {value['crossing_files']} |"
                      for level, value in summary["levels"].items())
    report = (f"# Dataset identity\n\nStatus: {'complete' if complete else 'partial / budget reached'}. "
              f"Parsed {len(rows)} files; excluded {len(exclusions)}.\n\n"
              "| Identity | Groups | Extra duplicates | Simulated crossing groups | Affected files |\n|---|---:|---:|---:|---:|\n" + table +
              "\n\nFormatting and the three declared top-level display fields are the only normalization axes. "
              "These are stored-state groups, not ancestry or perceptual-equivalence claims. "
              "Split crossings concern a deterministic simulated file split.\n\n"
              "The historical census had 9,618 parsed files; any difference is a corpus/parser population difference requiring inspection of the local exclusions. "
              "Group membership and source paths are private local artifacts. Introduced-group examples contain only field pointers.\n")
    return finish_report(context, "DATASET_IDENTITY_REPORT.md", report, "complete" if complete else "budget_exhausted", valid_files=len(rows))
