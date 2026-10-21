"""Re-score retained pre-roll waveforms and audit the renderer source and receipts."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

RESEARCH = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(RESEARCH / "training"), str(RESEARCH / "data_generation")]


def main():
    import numpy as np
    from obruxo_data.hashing import canonical_sha256, sha256_bytes
    from obruxo_data.midi import Performance
    from obruxo_data.render.base import RenderRequest
    from obruxo_data.render.qa import analyze_audio, audio_float32_sha256
    from obruxo_training.audit import write_json
    from obruxo_training.codec import PilotTarget, load_codec
    from obruxo_training.reproducibility import fixed_generator_phases
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--source-ref", required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("refusing to replace audited evidence")
    plan = json.loads((args.source / "plan.json").read_text())
    summary = json.loads((args.source / "summary.json").read_text())
    if canonical_sha256(plan) != summary["plan_id"] or summary["plan"] != plan:
        raise ValueError("frozen pre-roll plan changed")
    for name, expected in plan["source"].items():
        path = (RESEARCH / "data_generation/obruxo_data/render" / name if name in {"base.py", "vita.py", "midi_transport.py"} else
                Path(__file__).with_name(name) if name == "check_native_preroll.py" else RESEARCH / "training/obruxo_training" / name)
        relative = path.relative_to(RESEARCH.parent).as_posix()
        raw = subprocess.check_output(["git", "show", f"{args.source_ref}:{relative}"], cwd=RESEARCH.parent)
        if sha256_bytes(raw.replace(b"\r\n", b"\n")) != expected:
            raise ValueError(f"frozen renderer source mismatch: {name}")
    codec, performance, audio = load_codec(plan["codec"]), Performance.from_dict(plan["performance"]), {}
    if len(summary["cases"]) != len(plan["cases"]) or summary["cases"] != json.loads((args.source / "cases.json").read_text()):
        raise ValueError("pre-roll cases incomplete or changed")
    total_attempts = 0
    for directory in args.source.glob("profile-*"):
        events = [json.loads(line) for line in (directory / "renders.jsonl").read_text().splitlines()]
        attempts = [row for row in events if row["event"] == "attempt"]
        outcomes = [row for row in events if row["event"] == "outcome"]
        if len(attempts) != len(outcomes) or any(a["attempt"] != i or b["attempt"] != i or a["request_id"] != b["request_id"]
                                              for i, (a, b) in enumerate(zip(attempts, outcomes, strict=True))):
            raise ValueError("pre-roll native accounting changed")
        total_attempts += len(attempts)
    for planned, row in zip(plan["cases"], summary["cases"], strict=True):
        if any(row[key] != planned[key] for key in ("name", "target", "preroll_frames")):
            raise ValueError("pre-roll candidate changed")
        receipt = row["receipt"]
        directory = args.source / f"profile-{row['preroll_frames']}"
        outcomes = [json.loads(line) for line in (directory / "renders.jsonl").read_text().splitlines()]
        recorded = next(value for value in outcomes if value["event"] == "outcome" and value["attempt"] == receipt["attempt"])
        if recorded != receipt:
            raise ValueError("pre-roll receipt changed")
        raw = codec.decode(PilotTarget.from_dict(row["target"]))
        request = RenderRequest(preset_json=fixed_generator_phases(raw).decode(), performance=performance,
                                renderer_id=codec.contract.renderer_id, tail_seconds=codec.contract.tail_seconds,
                                preroll_frames=row["preroll_frames"])
        if (request.request_id != receipt["request_id"] or sha256_bytes(raw) != receipt["source_sha256"]
                or sha256_bytes(request.preset_json.encode()) != receipt["controlled_sha256"]):
            raise ValueError("pre-roll native request cannot be reconstructed")
        measured = np.load(args.source / "audio" / f"{request.request_id}.npy", allow_pickle=False)
        qa, _ = analyze_audio(measured, sample_rate=44100, expected_frames=receipt["qa"]["expected_frames"], expected_channels=2)
        if qa != receipt["qa"] or audio_float32_sha256(measured) != qa["audio_float32_sha256"]:
            raise ValueError("retained native waveform or measurements changed")
        if row["preroll_frames"] and (receipt["status"] != "accepted" or qa["clipping_count"]):
            raise ValueError("warm waveform failed strict audio QA")
        audio[row["name"]] = measured.astype(np.float64)
    high, low, longer = (audio[name] for name in ("bad", "lower", "long_preroll"))
    targets = {row["name"]: row["target"]["continuous"] for row in plan["cases"]}
    ratio = ((.15 + .2 * targets["lower"][0]) / (.15 + .2 * targets["bad"][0])) ** 2
    level = float(np.linalg.norm(low - ratio * high) / np.linalg.norm(low))
    settling = float(np.linalg.norm(longer - high) / np.linalg.norm(high))
    repeats = all(np.array_equal(audio[name], audio[f"{name}_repeat"]) for name in ("source", "bad"))
    settling_key = "short_vs_long_relative_wave_error" if "short_vs_long_relative_wave_error" in summary else "8192_vs_16384_relative_wave_error"
    if (not np.isclose(level, summary["level_relative_wave_error"], rtol=1e-10, atol=1e-12)
            or not np.isclose(settling, summary[settling_key], rtol=1e-10, atol=1e-12) or repeats != summary["repeats_sample_exact"]
            or total_attempts != summary["native_attempts"]):
        raise ValueError("pre-roll measurements do not reproduce saved evidence")
    archive = sum(path.stat().st_size for path in args.source.rglob("*") if path.is_file())
    limits = plan["limits"]
    resources = (summary["rss_sampling_errors"] == 0 and 0 < summary["peak_process_tree_rss_bytes"] <= limits["process_tree_bytes"]
                 and summary["elapsed_seconds"] <= limits["wall_seconds"] and archive <= limits["archive_bytes"]
                 and total_attempts <= limits["qualification_attempts"])
    passed = repeats and level <= 1e-5 and settling <= 1e-5 and resources
    if passed != summary["preflight_passed"] or resources != summary["within_resource_limits"]:
        raise ValueError("published pre-roll gates differ from native evidence")
    result = {**summary, "frozen_source_ref": args.source_ref, "retained_archive_bytes": archive,
              "all_native_waveforms_rescored": True, "all_requests_reconstructed": True,
              "evidence_sha256": {name: sha256_bytes((args.source / name).read_bytes()) for name in ("plan.json", "summary.json", "cases.json")}}
    write_json(args.output, result)
    print(json.dumps({"preflight_passed": passed, "native_attempts": total_attempts, "all_waveforms_rescored": True}))


if __name__ == "__main__":
    main()
