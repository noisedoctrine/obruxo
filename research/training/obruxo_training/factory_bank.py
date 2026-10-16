"""Hash-pinned factory imports qualified against the installed native renderer."""
import base64
from copy import deepcopy
import hashlib
import importlib.metadata
import multiprocessing as mp
from pathlib import Path
import tempfile
from time import perf_counter
import wave

import numpy as np

from obruxo_data.hashing import canonical_json, canonical_sha256
from obruxo_data.render import VitalRenderer
from obruxo_data.render.base import FULL_PRESET_RENDERER_ID
from obruxo_data.render.full_state import verify_full_nested_state
from obruxo_data.render.qa import analyze_audio
from obruxo_data.render.vita import VitalVst3StateTemplate, verify_loaded_scalars
from obruxo_data.vital.full_contract import FullPresetContract, strict_json

from .audit import write_json
from .benchmark import MemoryMonitor


IMPORT_POLICY = "native_reexport_then_explicit_155_header_with_audio_equivalence_v1"
RELATIVE_RMSE_LIMIT = .001
ABSOLUTE_PEAK_ERROR_LIMIT = .0001
MANIFEST = Path(__file__).resolve().parents[1] / "configs/factory-assets-v1.json"
_NATIVE_ATTEMPTS = 0
_NATIVE_ACCEPTED = 0


def reference_sources(manifest, root):
    document = strict_json(Path(manifest).read_bytes())
    if document.get("version") != 1 or not isinstance(document.get("references"), list) or not document["references"]:
        raise ValueError("expected a nonempty version-one factory reference manifest")
    root = Path(root).resolve(strict=True)
    ids, paths, entries = set(), set(), []
    for entry in document["references"]:
        if entry["kind"] not in {"wavetable", "sample", "lfo"} or entry["provenance"] != "factory_export":
            raise ValueError("factory references need an explicit supported kind and factory_export provenance")
        path = (root / entry["path"]).resolve(strict=True)
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError("factory reference path escapes the selected asset root")
        if path.stat().st_size > 64 * 1024**2:
            raise ValueError("factory reference exceeds the 64 MiB safety limit")
        if entry["asset_id"] in ids or path in paths:
            raise ValueError("factory reference IDs and paths must be unique")
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry["source_sha256"]:
            raise ValueError("factory reference source hash changed")
        ids.add(entry["asset_id"])
        paths.add(path)
        entries.append(dict(entry))
    return document, entries


def sample_from_wav(path):
    with wave.open(str(path), "rb") as source:
        channels, width, rate, frames = source.getnchannels(), source.getsampwidth(), source.getframerate(), source.getnframes()
        if channels not in {1, 2} or width != 2 or not 0 < frames <= 16_777_216 or not 0 < rate <= 768000:
            raise ValueError("factory samples require mono/stereo PCM16 within the full contract bounds")
        raw = source.readframes(frames)
    if len(raw) != frames * channels * 2:
        raise ValueError("truncated PCM16 sample")
    pcm = np.frombuffer(raw, dtype="<i2").reshape(frames, channels)
    sample = {"name": Path(path).stem, "length": frames, "sample_rate": rate,
              "samples": base64.b64encode(pcm[:, 0].copy().tobytes()).decode()}
    if channels == 2:
        sample["samples_stereo"] = base64.b64encode(pcm[:, 1].copy().tobytes()).decode()
    return sample


def _base_document():
    document = FullPresetContract().template()
    document["settings"].update({f"osc_{index}_random_phase": 0. for index in range(1, 4)})
    document["settings"].update(osc_1_level=.15, sample_random_phase=0.)
    return document


def _native(document, *, pitch=60, imported_wave=False, allow_silence=False):
    """Diagnostic importer: fresh plugin, pinned binary, no relaxed production readback."""
    import dawdreamer as daw
    global _NATIVE_ATTEMPTS, _NATIVE_ACCEPTED
    _NATIVE_ATTEMPTS += 1
    renderer = VitalRenderer(renderer_id=FULL_PRESET_RENDERER_ID)
    if importlib.metadata.version("dawdreamer") != "0.8.3":
        raise ValueError("factory import requires DawDreamer 0.8.3")
    started = perf_counter()
    with tempfile.TemporaryDirectory(prefix="obruxo-factory-import-") as directory:
        directory = Path(directory)
        engine = daw.RenderEngine(44100, 128)
        engine.set_bpm(120.)
        plugin = engine.make_plugin_processor("vital", str(renderer.plugin_path))
        plugin.save_state(str(directory / "initial.state"))
        template = VitalVst3StateTemplate((directory / "initial.state").read_bytes())
        (directory / "requested.state").write_bytes(template.build(canonical_json(document)))
        plugin.load_state(str(directory / "requested.state"))
        plugin.save_state(str(directory / "loaded.state"))
        loaded = VitalVst3StateTemplate((directory / "loaded.state").read_bytes()).preset_document
        count = verify_loaded_scalars(document, loaded)
        expected = deepcopy(document)
        if imported_wave:
            # Only the explicitly requested source asset is being imported. Capture it;
            # every other nested field still receives the production readback check.
            expected["settings"]["wavetables"][0] = deepcopy(loaded["settings"]["wavetables"][0])
        nested = verify_full_nested_state(expected, loaded, FullPresetContract().spec["fixed_native_ramps"])
        plugin.add_midi_note(pitch, 96, 0., .5)
        engine.load_graph([(plugin, [])])
        if not engine.render(2.):
            raise RuntimeError("factory import native render failed")
        audio = np.ascontiguousarray(np.asarray(engine.get_audio(), dtype=np.float32).T)
    qa, diagnostics = analyze_audio(audio, sample_rate=44100, expected_frames=88200, expected_channels=2)
    rejected = {"audio.silence", "audio.clipping", "audio.tail_truncation", "audio.non_finite"}
    if allow_silence:
        rejected.remove("audio.silence")
    if any(item.code in rejected or item.severity.value == "error" for item in diagnostics):
        raise RuntimeError("factory reference failed audio QA: " + ", ".join(item.code for item in diagnostics))
    _NATIVE_ACCEPTED += 1
    return audio, loaded, {"request_sha256": canonical_sha256({"document": document, "pitch": pitch, "sample_rate": 44100,
                                                               "note_seconds": .5, "render_seconds": 2., "phase": "fixed"}),
                           "audio_float32_sha256": qa["audio_float32_sha256"], "qa": qa,
                           "verified_loaded_scalars": count, "nested_readback": nested, "seconds": perf_counter() - started}


def _wave_document(wave_asset, frame):
    document = _base_document()
    document["settings"]["wavetables"][0] = deepcopy(wave_asset)
    document["settings"]["osc_1_wave_frame"] = float(frame)
    return document


def _canonical_wave(loaded):
    wave_asset = deepcopy(loaded["settings"]["wavetables"][0])
    wave_asset["version"] = "1.5.5"  # Explicit import policy, validated and rendered again below.
    FullPresetContract().validate(_wave_document(wave_asset, 0)).require_valid()
    return wave_asset


def _equivalence(left, right):
    scale = max(float(np.sqrt(np.mean(left.astype(np.float64)**2))), 1e-8)
    difference = left.astype(np.float64) - right
    relative_rmse = float(np.sqrt(np.mean(difference**2))) / scale
    peak_error = float(np.max(np.abs(difference)))
    return {"relative_rmse": relative_rmse, "absolute_peak_error": peak_error,
            "exact": np.array_equal(left, right), "passed": relative_rmse <= RELATIVE_RMSE_LIMIT and peak_error <= ABSOLUTE_PEAK_ERROR_LIMIT}


def _qualify_reference(entry, root):
    path = Path(root) / entry["path"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != entry["source_sha256"]:
        raise ValueError("factory source changed after planning")
    kind, renders, comparisons = entry["kind"], [], []
    if kind == "wavetable":
        source = strict_json(path.read_bytes())
        originals = []
        for frame in (0, 128, 256):
            audio, loaded, metadata = _native(_wave_document(source, frame), imported_wave=True, allow_silence=True)
            originals.append(audio)
            renders.append(metadata)
        asset = _canonical_wave(loaded)
        stable = False
        for _ in range(4):
            _, loaded, metadata = _native(_wave_document(asset, 0), imported_wave=True, allow_silence=True)
            renders.append(metadata)
            next_asset = _canonical_wave(loaded)
            if next_asset == asset:
                stable = True
                break
            asset = next_asset
        if not stable:
            raise RuntimeError("factory wavetable re-export did not stabilize in four attempts")
        init = _base_document()["settings"]["wavetables"][0]
        def payload(value):
            return {key: item for key, item in value.items() if key not in {"name", "author", "version"}}
        if payload(asset) == payload(init):
            raise RuntimeError("factory wavetable is indistinguishable from default readback; import needs review")
        for frame, original in zip((0, 128, 256), originals, strict=True):
            audio, _, metadata = _native(_wave_document(asset, frame), allow_silence=True)
            renders.append(metadata)
            comparisons.append({"frame": frame, **_equivalence(original, audio)})
        if not any(float(np.max(np.abs(audio))) > 1e-6 for audio in originals):
            raise RuntimeError("factory wavetable was silent at every declared probe frame")
    elif kind == "sample":
        asset = sample_from_wav(path)
        document = _base_document()
        document["settings"].update(osc_1_on=0., sample_on=1., sample_level=.15, sample_keytrack=1., sample_loop=0.)
        document["settings"]["sample"] = asset
        FullPresetContract().validate(document).require_valid()
        reference = None
        for pitch in (48, 60, 72, 60):
            audio, _, metadata = _native(document, pitch=pitch)
            renders.append(metadata)
            if pitch == 60:
                if reference is not None:
                    comparisons.append({"pitch": pitch, **_equivalence(reference, audio)})
                reference = audio
    else:
        asset = strict_json(path.read_bytes())
        asset.setdefault("name", path.stem)
        asset["num_points"] = int(asset["num_points"])
        for key in ("points", "powers"):
            asset[key] = [float(np.float32(value)) for value in asset[key]]
        document = _base_document()
        document["settings"]["lfos"][0] = asset
        document["settings"]["modulations"][0] = {"source": "lfo_1", "destination": "osc_1_level"}
        document["settings"].update(lfo_1_frequency=0., lfo_1_sync=0., modulation_1_amount=.2)
        FullPresetContract().validate(document).require_valid()
        first, _, metadata = _native(document)
        renders.append(metadata)
        repeat, _, metadata = _native(document)
        renders.append(metadata)
        comparisons.append(_equivalence(first, repeat))
    passed = all(row["passed"] for row in comparisons)
    if kind != "wavetable":
        passed &= all(row["exact"] for row in comparisons)
    return {**entry, "asset": asset, "asset_sha256": canonical_sha256(asset)}, {"asset_id": entry["asset_id"], "kind": kind,
               "passed": passed, "native_attempts": len(renders), "comparisons": comparisons, "renders": renders}


def _import_child(connection, entry, root):
    try:
        connection.send({"status": "complete", "result": _qualify_reference(entry, root)})
    except Exception as error:
        connection.send({"status": "error", "error_type": type(error).__name__, "message": str(error)[:1000],
                         "native_attempts": _NATIVE_ATTEMPTS, "native_accepted": _NATIVE_ACCEPTED})
    finally:
        connection.close()


def _import_isolated(entry, root, timeout):
    context = mp.get_context("spawn")
    receive, send = context.Pipe(duplex=False)
    process = context.Process(target=_import_child, args=(send, entry, str(root)), daemon=True)
    try:
        process.start()
        send.close()
        if not receive.poll(timeout):
            return {"status": "error", "error_type": "ImportTimeout", "message": "factory asset exceeded its fixed deadline"}
        try:
            return receive.recv()
        except EOFError:
            return {"status": "error", "error_type": "ImportWorkerExited", "exit_code": process.exitcode}
    finally:
        send.close()
        receive.close()
        if process.pid is not None:
            process.join(.2)
            if process.is_alive():
                process.terminate()
                process.join(1)
            if process.is_alive():
                process.kill()
                process.join(1)
        process.close()


def qualify_factory_bank(root, output, *, manifest=MANIFEST, timeout=120., progress=None):
    if type(timeout) not in {float, int} or not np.isfinite(timeout) or timeout <= 0:
        raise ValueError("factory import timeout must be positive and finite")
    document, references = reference_sources(manifest, root)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    source = {str(path.relative_to(Path(__file__).parents[2])): hashlib.sha256(path.read_bytes()).hexdigest() for path in
              (Path(__file__), Path(__file__).parents[2] / "data_generation/obruxo_data/render/vita.py",
               Path(__file__).parents[2] / "data_generation/obruxo_data/render/full_state.py")}
    counts = {kind: sum(entry["kind"] == kind for entry in references) for kind in ("wavetable", "sample", "lfo")}
    plan = {"policy": IMPORT_POLICY, "references": document, "source_sha256": source, "counts": counts,
            "renderer_id": FULL_PRESET_RENDERER_ID, "timeout_per_reference_seconds": timeout,
            "maximum_native_attempts": counts["wavetable"] * 10 + counts["sample"] * 4 + counts["lfo"] * 2,
            "relative_rmse_limit": RELATIVE_RMSE_LIMIT, "absolute_peak_error_limit": ABSOLUTE_PEAK_ERROR_LIMIT,
            "phase_control": "all_three_oscillator_random_phases_and_sampler_random_phase_zero", "rss_limit_bytes": 8 * 1024**3,
            "test_reserve_used": False, "audio_archive": False}
    write_json(output / "plan.json", plan)
    assets, checks = [], []
    monitor, started = MemoryMonitor(), perf_counter()
    monitor.start()
    try:
        for entry in references:
            result = _import_isolated(entry, root, timeout)
            if result["status"] == "complete":
                asset, check = result["result"]
                assets.append(asset)
                checks.append(check)
            else:
                checks.append({"asset_id": entry["asset_id"], "kind": entry["kind"], "passed": False, **result})
            write_json(output / "checks.json", checks)
            if progress:
                progress({"asset": entry["path"], "passed": checks[-1]["passed"], "done": len(checks), "total": len(references),
                          **({} if checks[-1]["passed"] else {"failure": checks[-1].get("message", "audio equivalence failed")})})
    finally:
        monitor.close()
    unchanged = source == {name: hashlib.sha256((Path(__file__).parents[2] / name).read_bytes()).hexdigest() for name in source}
    bank = {"version": 1, "policy": IMPORT_POLICY, "source_manifest_sha256": canonical_sha256(document), "references": assets}
    bank_id = canonical_sha256(bank)
    qualified = (len(assets) == len(references) and all(check["passed"] for check in checks) and unchanged
                 and monitor.errors == 0 and monitor.peak_bytes is not None and monitor.peak_bytes <= plan["rss_limit_bytes"])
    qualification = {"qualified": qualified, "bank_id": bank_id, "renderer_id": FULL_PRESET_RENDERER_ID,
                     "plan_sha256": canonical_sha256(plan), "source_unchanged": unchanged, "references_passed": sum(check["passed"] for check in checks),
                     "references_requested": len(references), "peak_process_rss_bytes": monitor.peak_bytes, "memory_sampling_errors": monitor.errors,
                     "elapsed_seconds": perf_counter() - started, "native_attempts": sum(check.get("native_attempts", 0) for check in checks),
                     "native_accepted": sum(check.get("native_accepted", check.get("native_attempts", 0)) for check in checks)}
    result = {"format": "obruxo_factory_asset_bank_v1", "bank_id": bank_id, "bank": bank, "qualification": qualification}
    write_json(output / "bank.json", result)
    return result
