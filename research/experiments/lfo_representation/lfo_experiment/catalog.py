"""Corpus extraction and provisional stock-codebook construction."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .constants import LFO_SCALAR_SUFFIXES, STOCK_LFO_NAMES
from .model import LfoShape


def _json(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False)


def _bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)


def _preset_path(metadata_path: Path, relative: str) -> Path:
    return metadata_path.parent / Path(relative.replace("\\", "/"))


def build_catalog(
    metadata_path: Path,
    output_path: Path,
    *,
    limit: int | None = None,
) -> pd.DataFrame:
    """Extract one row for every LFO referenced by a valid modulation route."""
    metadata_path = metadata_path.resolve()
    metadata = pd.read_csv(metadata_path, dtype=str, keep_default_na=False)
    if limit is not None:
        metadata = metadata.head(limit)

    rows: list[dict[str, Any]] = []
    errors = 0
    for ordinal, record in enumerate(metadata.to_dict("records"), 1):
        relative = record.get("preset_file", "")
        if not relative:
            continue
        path = _preset_path(metadata_path, relative)
        try:
            preset = json.loads(path.read_bytes())
            settings = preset["settings"]
            lfos = settings.get("lfos", [])
        except Exception:
            errors += 1
            continue

        routes: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for slot, route in enumerate(settings.get("modulations", []), 1):
            if not isinstance(route, dict):
                continue
            source = str(route.get("source", ""))
            destination = str(route.get("destination", ""))
            if not destination or not source.startswith("lfo_"):
                continue
            suffix = source[4:]
            if not suffix.isdigit():
                continue
            amount = float(settings.get(f"modulation_{slot}_amount", 0.0) or 0.0)
            bypass = bool(float(settings.get(f"modulation_{slot}_bypass", 0.0) or 0.0))
            routes[int(suffix)].append(
                {
                    "slot": slot,
                    "destination": destination,
                    "amount": amount,
                    "bypass": bypass,
                    "materially_active": not bypass and abs(amount) > 1e-8,
                }
            )

        for lfo_index, lfo_routes in routes.items():
            if not (1 <= lfo_index <= len(lfos)) or not isinstance(
                lfos[lfo_index - 1], dict
            ):
                continue
            try:
                shape = LfoShape.from_json(lfos[lfo_index - 1])
            except ValueError:
                errors += 1
                continue

            scalar_controls = {
                suffix: settings.get(f"lfo_{lfo_index}_{suffix}")
                for suffix in LFO_SCALAR_SUFFIXES
            }
            rows.append(
                {
                    "preset_id": record.get("preset_id", ""),
                    "preset_file": relative,
                    "author_id": record.get("author_id", ""),
                    "author": record.get("author", ""),
                    "synth_version": str(preset.get("synth_version", "")),
                    "lfo_index": lfo_index,
                    "shape_name": shape.name,
                    "shape_signature": shape.signature(),
                    "num_points": len(shape.points),
                    "smooth": shape.smooth,
                    "points": _json(shape.points.reshape(-1).tolist()),
                    "powers": _json(shape.powers.tolist()),
                    "route_count": len(lfo_routes),
                    "active_route_count": sum(
                        route["materially_active"] for route in lfo_routes
                    ),
                    "materially_active": any(
                        route["materially_active"] for route in lfo_routes
                    ),
                    "routes": _json(lfo_routes),
                    "scalar_controls": _json(scalar_controls),
                    "stock_name_hint": shape.name in STOCK_LFO_NAMES,
                }
            )

        if ordinal % 1000 == 0:
            print(f"Scanned {ordinal:,}/{len(metadata):,} presets", flush=True)

    catalog = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    catalog.to_csv(output_path, index=False)
    print(
        f"Wrote {len(catalog):,} routed LFOs to {output_path} "
        f"({errors:,} unreadable or invalid records)"
    )
    return catalog


def build_provisional_codebook(catalog_path: Path, output_path: Path) -> list[dict[str, Any]]:
    """Choose modal geometry for each known stock name.

    This bootstraps the benchmark. Controlled Vital saves should replace these
    entries because an LFO can retain a stock name after later editing.
    """
    catalog = pd.read_csv(catalog_path, keep_default_na=False)
    entries: list[dict[str, Any]] = []
    for name in STOCK_LFO_NAMES:
        named = catalog[catalog["shape_name"] == name]
        if named.empty:
            print(f"Warning: no corpus examples named {name!r}")
            continue
        counts = named["shape_signature"].value_counts()
        signature = str(counts.index[0])
        representative = named[named["shape_signature"] == signature].iloc[0]
        shape = LfoShape.from_serialized(
            representative["points"],
            representative["powers"],
            name=name,
            smooth=_bool(representative["smooth"]),
        )
        entries.append(
            {
                "name": name,
                "provenance": "corpus_modal_name",
                "support": int(counts.iloc[0]),
                "named_total": int(len(named)),
                "modal_share": float(counts.iloc[0] / len(named)),
                "shape": shape.to_json(),
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(entries, indent=2), encoding="utf-8")
    print(f"Wrote {len(entries)} provisional stock shapes to {output_path}")
    return entries


def load_codebook(path: Path) -> list[tuple[str, LfoShape]]:
    entries = json.loads(path.read_text(encoding="utf-8"))
    return [(entry["name"], LfoShape.from_json(entry["shape"])) for entry in entries]
