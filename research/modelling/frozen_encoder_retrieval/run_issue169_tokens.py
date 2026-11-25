"""Verify the existing Issue 167 WAVs and cache V7's full online encoder output."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from time import sleep

import numpy as np

from frozen_retrieval.core import digest
from frozen_retrieval.encoders import file_sha256, load_adapter, load_checkpoint_lock, resolve_checkpoint
from frozen_retrieval.rendering import accepted_receipts
from run_issue167_v7 import CHECKPOINTS, DATASET, OUTPUTS, ROOT, read_json, read_rows, write_json


RUN = OUTPUTS / "issue169-v7-retrieval"
LOCK = ROOT / "issue167-checkpoint-lock.json"
MODEL = "vital_jepa_v7"
SHARD_SIZE = 100
TOKEN_SHAPE = (260, 384)


def status(phase, completed, total, *, state="running"):
    updated = datetime.now(timezone.utc).isoformat(timespec="seconds")
    path = RUN / "status.json"
    temporary = RUN / f"status-{os.getpid()}.tmp"
    payload = {"phase": phase, "completed": completed, "total": total, "failures": int(state == "failed"),
               "updated_at": updated, "state": state, "pid": os.getpid()}
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    for attempt in range(20):
        try:
            temporary.replace(path)
            break
        except PermissionError:
            if attempt == 19:
                raise
            sleep(0.1)
    if state != "running" or completed in (0, total) or completed % 500 == 0:
        print(f"{updated} {phase}: {completed:,}/{total:,} ({state})", flush=True)


def cohort():
    episodes_path = DATASET / "episodes_qa_qualified.private.jsonl"
    candidates_path = DATASET / "candidate_sets.render_qualified.private.json"
    episodes = read_rows(episodes_path)
    pairs = {(row["target_state_id"], row["query_midi_id"]) for row in episodes
             if row["condition"] == "cross_midi_common_gallery"}
    requests = {(row["state_id"], row["midi_id"]): row["request_id"]
                for row in read_rows(DATASET / "render_plan.private.jsonl") if (row["state_id"], row["midi_id"]) in pairs}
    receipts = accepted_receipts(DATASET)
    if (len(pairs), len({p for p, _ in pairs}), len({m for _, m in pairs})) != (6450, 129, 50):
        raise ValueError("Issue 167 cohort no longer matches the 129 by 50 assignments")
    if set(requests) != pairs or set(requests.values()) != set(receipts) or len(receipts) != len(pairs):
        raise ValueError("Issue 167 retained receipts do not cover exactly the assigned recordings")
    return episodes, read_json(candidates_path)["candidate_sets"], requests, receipts


def checkpoint():
    return resolve_checkpoint(MODEL, CHECKPOINTS, load_checkpoint_lock(LOCK))


def array_to_file(path, array):
    temp = path.with_suffix(".tmp")
    with temp.open("wb") as stream:
        np.save(stream, array, allow_pickle=False)
    temp.replace(path)


def load_shard(path, size):
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    if array.shape != (size, *TOKEN_SHAPE) or array.dtype != np.float32 or not np.isfinite(array).all():
        raise ValueError(f"Invalid token shard {path.name}")
    return array


def verify_wav(receipt):
    path = (DATASET / receipt["wav_relative_path"]).resolve()
    if not path.is_relative_to(DATASET.resolve()) or not path.is_file():
        return f"missing or invalid path: {receipt['wav_relative_path']}"
    if file_sha256(path) != receipt["audio_sha256"]:
        return f"SHA-256 mismatch: {receipt['wav_relative_path']}"
    return None


def extract(device):
    from frozen_retrieval.vital_jepa_v7 import PREPROCESSING

    _, _, _, receipts = cohort()
    request_ids = sorted(receipts)
    status("Verify Issue 167 WAVs", 0, len(request_ids))
    failures = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for index, failure in enumerate(pool.map(verify_wav, (receipts[key] for key in request_ids)), 1):
            if failure:
                failures.append(failure)
            if index % 100 == 0 or index == len(request_ids):
                status("Verify Issue 167 WAVs", index, len(request_ids))
    if failures:
        write_json(RUN / "wav-audit-failures.private.json", failures)
        raise ValueError(f"{len(failures)} retained WAVs failed verification. See the private audit file.")

    path, entry, checkpoint_sha = checkpoint()
    identity = {"checkpoint_sha256": checkpoint_sha, "preprocessing": PREPROCESSING, "tap": "online_final_norm",
                "dtype": "float32", "device": device, "shard_size": SHARD_SIZE, "token_shape": list(TOKEN_SHAPE),
                "rows": [[key, receipts[key]["audio_sha256"]] for key in request_ids]}
    cache = RUN / "tokens" / digest(identity)
    cache.mkdir(parents=True, exist_ok=True)
    write_json(cache / "identity.json", identity)
    write_json(RUN / "token-cache.json", {"cache_id": cache.name, "recordings": len(request_ids)})
    adapter = None
    status("Cache online encoder tokens", 0, len(request_ids))
    for start in range(0, len(request_ids), SHARD_SIZE):
        batch = request_ids[start:start + SHARD_SIZE]
        shard = cache / f"{start:05d}.npy"
        if shard.exists():
            load_shard(shard, len(batch))
        else:
            if adapter is None:
                adapter = load_adapter(MODEL, path, entry, checkpoint_sha, checkpoint_root=CHECKPOINTS, device=device)
            array = np.empty((len(batch), *TOKEN_SHAPE), dtype=np.float32)
            with adapter.torch.inference_mode():
                for index, key in enumerate(batch):
                    waveform, _ = adapter._audio(DATASET / receipts[key]["wav_relative_path"])
                    tensor = adapter.torch.from_numpy(waveform)
                    array[index] = adapter.model.encode_tokens(tensor)[0].cpu().numpy()
            if not np.isfinite(array).all():
                raise ValueError(f"Nonfinite token output in shard starting at {start}")
            array_to_file(shard, array)
        status("Cache online encoder tokens", start + len(batch), len(request_ids))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="xpu", choices=("xpu", "cpu"))
    args = parser.parse_args()
    RUN.mkdir(parents=True, exist_ok=True)
    try:
        extract(args.device)
    except Exception:
        status("Token extraction failed", 0, 1, state="failed")
        raise
    status("Token cache ready", 6450, 6450, state="complete")


if __name__ == "__main__":
    main()
