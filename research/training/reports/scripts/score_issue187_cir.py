"""Score original/reconstruction CIR cosine with the locked retrieval runtime."""

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.io import wavfile
import torch


TRAINING = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(TRAINING), str(TRAINING.parent / "modelling/frozen_encoder_retrieval")]

from frozen_retrieval.encoders import (load_adapter, load_checkpoint_lock, resolve_checkpoint,
                                       resolve_source_checkout)
from obruxo_training.reconstruction_metrics import SAMPLE_RATE, cosine_similarity


def score(adapter, manifest, output, references):
    originals = output.parent / "cir-reference-wavs"
    originals.mkdir(exist_ok=True)
    cases = {}
    for row in manifest:
        case = row["case"]
        if case not in references:
            path = originals / f"{case}.wav"
            wavfile.write(path, SAMPLE_RATE, np.load(row["original_path"], allow_pickle=False))
            references[case] = adapter.extract(path)["official_clip"]
        vector = adapter.extract(Path(row["wave_path"]))["official_clip"]
        cases[case] = cosine_similarity(references[case], vector)
    return cases


def write(path, record):
    temporary = path.with_suffix(".cir.tmp")
    temporary.write_text(json.dumps(record, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def merge_validation(path, cases):
    if not path.is_file():
        return
    validation = json.loads(path.read_text(encoding="utf-8"))
    for row in validation["cases"]:
        row["cir_cosine"] = cases[f"{row['pair_id']}-{row['clip_role']}"]
    valid = [value for value in cases.values() if value is not None]
    validation["metrics"].update(cir_cosine=float(np.mean(valid)) if valid else None,
                                 cir_cosine_valid=len(valid))
    write(path, validation)


def follow(run):
    seen = set()
    while True:
        for path in sorted(run.glob("waveforms-*.json"), key=lambda item: item.stat().st_mtime):
            name = path.stem.removeprefix("waveforms-")
            if name not in seen:
                seen.add(name)
                yield name
        status = run / "status.json"
        if status.is_file() and json.loads(status.read_text(encoding="utf-8"))["state"] in {
            "analysis_required", "incomplete", "blocked", "failed", "complete",
        }:
            return
        time.sleep(5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--models", nargs="+")
    mode.add_argument("--follow", action="store_true", help="score validation manifests as training produces them")
    parser.add_argument("--wait-seconds", type=int, default=0)
    args = parser.parse_args()
    torch.set_num_threads(3)
    model = "cir_ast"
    root = args.checkpoint_root.resolve()
    lock = load_checkpoint_lock(root / "checkpoint-lock.json")
    checkpoint, entry, digest = resolve_checkpoint(model, root, lock)
    adapter = load_adapter(model, checkpoint, entry, digest, checkpoint_root=root, device=args.device,
                           source_root=resolve_source_checkout(root, entry))
    references = {}
    for name in follow(args.run) if args.follow else args.models:
        output = args.run / f"cir-{name}.json"
        if output.exists():
            merge_validation(args.run / f"validation-{name}.json",
                             json.loads(output.read_text(encoding="utf-8"))["cases"])
            continue
        manifest_path = args.run / f"waveforms-{name}.json"
        deadline = time.monotonic() + args.wait_seconds
        while not manifest_path.is_file() and time.monotonic() < deadline:
            time.sleep(2)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        cases = score(adapter, manifest, output, references)
        write(output, {"checkpoint_sha256": digest, "cases": cases})
        merge_validation(args.run / f"validation-{name}.json", cases)
        print(f"{name}: {sum(value is not None for value in cases.values())}/{len(cases)} CIR scores", flush=True)


if __name__ == "__main__":
    main()
