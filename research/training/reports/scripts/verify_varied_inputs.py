"""Reconstruct source requests and independently check saved native-check receipts."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

RESEARCH = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(RESEARCH / "training"), str(RESEARCH / "data_generation")]

from obruxo_data.hashing import canonical_sha256, sha256_bytes  # noqa: E402
from obruxo_data.render.base import RenderRequest  # noqa: E402
from obruxo_training.input_performances import VariedSampler  # noqa: E402
from obruxo_training.pilot import PilotPool  # noqa: E402
from obruxo_training.reproducibility import FULL_PHASE_CONTROL  # noqa: E402
from obruxo_training.stream import request_frames  # noqa: E402


def verify(root, pool_path, output):
    plan, summary, status = [json.loads((root / name).read_text()) for name in ("plan.json", "summary.json", "status.json")]
    attempts, renders = [[json.loads(line) for line in (root / name).read_text().splitlines()] for name in ("attempts.jsonl", "renders.jsonl")]
    pool = PilotPool.load(pool_path)
    assert pool.pool_id == plan["pool_id"] and pool.manifest.manifest_id == plan["manifest_id"]
    sampler = VariedSampler(pool)
    assert sampler.observation_contract == plan["observation_contract"]
    assert summary["plan_id"] == canonical_sha256(plan) and summary["complete"] and status["complete"]
    assert len(attempts) == len(renders) == summary["accepted"] == plan["planned_native_requests"] == 76
    assert summary["optimizer_updates"] == plan["optimizer_updates"] == 0
    assert not summary["qualified_for_release"] and not summary["test_split_used"] and not plan["test_split_used"]
    repeats, parity, wall, native, audio_seconds = defaultdict(list), 0, 0., 0., 0.
    for index, (spec, attempt, row) in enumerate(zip(plan["requests"], attempts, renders, strict=True)):
        request = RenderRequest.from_dict(spec["request"])
        assert index == attempt["index"] == row["index"]
        assert spec["request_id"] == attempt["request_id"] == row["request_id"] == row["provenance"]["request_id"] == request.request_id
        assert row["status"] == "accepted" and row["rss_sampling_errors"] == 0
        assert {key: row["provenance"][key] for key in plan["renderer_identity"]} == plan["renderer_identity"]
        assert request.preroll_frames == 32768
        settings = json.loads(request.preset_json)["settings"]
        assert all(settings[key] == value for key, value in FULL_PHASE_CONTROL.items())
        qa = row["qa"]
        assert qa["finite"] and qa["clipping_count"] == 0
        assert qa["actual_frames"] == qa["expected_frames"] == request_frames(request)
        assert qa["actual_channels"] == request.channels == 2
        assert not {"audio.silence", "audio.clipping", "audio.tail_truncation"} & set(row["diagnostic_codes"])
        if spec["kind"] == "varied":
            recipe = sampler.recipe_at(spec["sequence_id"])
            assert recipe.state_id == spec["state_id"] and recipe.recipe_id == spec["recipe_id"]
            assert recipe.split == "train" and recipe.input_pattern == spec["pattern"]
            assert recipe.render_request().to_dict() == request.to_dict()
            repeats[request.request_id].append(qa["audio_float32_sha256"])
        else:
            assert spec["kind"] == "historical_parity" and qa["audio_float32_sha256"] == spec["expected_audio_sha256"]
            parity += 1
        wall += row["wall_seconds"]
        native += row["render_seconds"]
        audio_seconds += request_frames(request) / request.sample_rate
    assert len(repeats) == 36 and parity == 4
    assert all(len(values) == 2 and values[0] == values[1] for values in repeats.values())
    assert summary["passed"] and summary["all_repeat_hashes_exact"] and summary["all_historical_hashes_exact"]
    assert summary["elapsed_seconds"] < plan["limits"]["wall_seconds"]
    assert summary["peak_process_tree_rss_bytes"] >= max(row["peak_process_tree_rss_bytes"] for row in renders)
    assert summary["peak_process_tree_rss_bytes"] <= plan["limits"]["process_tree_bytes"] and summary["rss_sampling_errors"] == 0
    assert summary["archive_bytes"] < plan["limits"]["archive_bytes"]
    for name, digest in plan["source"].items():
        assert sha256_bytes((RESEARCH / "training/obruxo_training" / name).read_bytes().replace(b"\r\n", b"\n")) == digest
    names = ("plan.json", "run.json", "attempts.jsonl", "renders.jsonl", "summary.json", "status.json")
    audit = {"plan_id": canonical_sha256(plan), "all_varied_source_requests_reconstructed": True,
             "all_attempt_outcome_request_identities_exact": True, "exact_repeat_pairs": len(repeats), "historical_hashes_matched": parity,
             "native_renders_added": 0, "optimizer_updates_added": 0, "waveforms_retained_or_rescored": False,
             "native_call_seconds": native, "complete_worker_wall_seconds": wall, "output_audio_seconds": audio_seconds,
             "output_audio_per_native_second": audio_seconds / native, "output_audio_per_worker_wall_second": audio_seconds / wall,
             "evidence_sha256": {name: sha256_bytes((root / name).read_bytes()) for name in names},
             "verifier_sha256_lf": sha256_bytes(Path(__file__).read_bytes().replace(b"\r\n", b"\n"))}
    output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    return audit


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "pool", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.root, args.pool, args.output)), flush=True)
