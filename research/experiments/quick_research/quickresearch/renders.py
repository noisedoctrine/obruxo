from __future__ import annotations

import time
from pathlib import Path

from .common import Context, digest, file_hash, read_json, write_json
from .fixtures import load_fixture, performance, perturb


def render_key(fixture: dict, performance_name: str, treatment: str, repeat: int) -> str:
    return f"{fixture['fixture_id']}_{performance_name}_{treatment}_repeat{repeat}"


def usable(row: dict) -> bool:
    return row.get("state") == "ok"


def validated_cache(directory: Path, key: str, identity: dict) -> dict | None:
    from scipy.io import wavfile
    from obruxo_data.render.qa import audio_float32_sha256
    sidecar, wav = directory / f"{key}.json", directory / f"{key}.wav"
    if not sidecar.exists():
        return None
    row = read_json(sidecar)
    if row.get("identity") != identity:
        raise ValueError("render cache identity changed; use a new output directory")
    if row["state"] == "render_error":
        return row  # Stable failed outcome; a new run is required to retry.
    if not wav.exists():
        return None
    try:
        rate, audio = wavfile.read(wav)
        if (rate != 44100 or list(audio.shape) != row.get("shape") or str(audio.dtype) != "float32"
                or audio_float32_sha256(audio) != row["audio_sha256"] or file_hash(wav) != row["wav_sha256"]):
            return None
    except (OSError, ValueError, KeyError):
        return None
    return row


def render_one(context: Context, renderer, fixture: dict, performance_name: str, treatment: str, repeat: int,
               directory: Path, implementation_hash: str) -> dict:
    from obruxo_data.render import RenderRequest
    context.check(10)
    preset = load_fixture(fixture)
    change = None
    if treatment != "baseline":
        preset, change = perturb(preset, treatment)
    notes = performance(performance_name)
    request = RenderRequest(preset=preset, performance=notes, tail_seconds=2.0, renderer_id=renderer.renderer_id)
    key = render_key(fixture, performance_name, treatment, repeat)
    identity = {"request_id": request.request_id, "source_sha256": fixture["source_sha256"], "repeat": repeat,
                "implementation": implementation_hash, "schema_id": preset.schema.schema_id}
    directory.mkdir(parents=True, exist_ok=True)
    cached = validated_cache(directory, key, identity)
    if cached is not None:
        return {**cached, "cache_hit": True}
    wav, sidecar = directory / f"{key}.wav", directory / f"{key}.json"
    row = {"key": key, "identity": identity, "fixture_id": fixture["fixture_id"], "category": fixture["category"],
           "performance": performance_name, "performance_data": notes.to_dict(), "treatment": treatment,
           "change": change, "repeat": repeat, "wav": str(wav.resolve()), "cache_hit": False}
    started = time.monotonic()
    context.progress(current_render=key)
    try:
        result = renderer.render(request)  # Independent call, never deduplicated by request ID.
        diagnostics = [item.to_dict() for item in result.diagnostics]
        qa = result.qa
        invalid = (not qa["finite"] or qa["clipping_count"] > 0 or qa["rms"] <= qa["silence_threshold"]
                   or any(item["severity"] == "error" for item in diagnostics))
        result.write_wav(wav, force=True)
        row.update(state="qa_failed" if invalid else "ok", qa=qa, diagnostics=diagnostics,
                   provenance=result.provenance.to_dict(), audio_sha256=qa["audio_float32_sha256"],
                   wav_sha256=file_hash(wav), shape=list(result.audio.shape))
    except Exception as error:
        row.update(state="render_error", error_type=type(error).__name__, error=str(error))
    row["render_seconds"] = time.monotonic() - started
    write_json(sidecar, row)
    return row


def renderer_for(args):
    import importlib.util
    from obruxo_data.render import VitalRenderer
    missing = [name for name in ("vita", "dawdreamer") if importlib.util.find_spec(name) is None]
    if missing:
        raise ValueError(f"missing native dependencies: {', '.join(missing)}; see README setup before scheduling renders")
    return VitalRenderer.from_config(Path(args.renderer_config), plugin_path=args.plugin)


def bank_identity(fixtures: dict, renderer, implementation: str) -> str:
    return digest({"fixtures": fixtures["fixtures_hash"], "renderer": renderer.renderer_id, "implementation": implementation})
