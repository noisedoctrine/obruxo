"""Fetch the pinned public SpanSynth checkpoint and extract its frozen MIDI encoder."""

import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import snapshot_download

from obruxo_training.issue187_stage2 import MIDI_ENCODER_REVISION, extract_midi_encoder


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--encoder-output", type=Path, required=True)
    parser.add_argument("--provenance-output", type=Path, required=True)
    args = parser.parse_args()
    local = Path(snapshot_download(
        repo_id="mimbres/spansynth-edit", revision=MIDI_ENCODER_REVISION,
        allow_patterns=("v3-sq-v8-127750step/config.json", "v3-sq-v8-127750step/model.safetensors"),
        local_dir=args.checkpoint_dir,
    ))
    config = local / "v3-sq-v8-127750step/config.json"
    weights = local / "v3-sq-v8-127750step/model.safetensors"
    extracted = extract_midi_encoder(weights, args.encoder_output)
    provenance = {
        "repository": "https://huggingface.co/mimbres/spansynth-edit",
        "revision": MIDI_ENCODER_REVISION,
        "model_subdirectory": "v3-sq-v8-127750step",
        "license": "Apache-2.0",
        "upstream_source": "https://github.com/mimbres/spansynth-edit",
        "upstream_source_revision": "f593bf21e48cc4aa836002a3834cfae42e8d4254",
        "config_sha256": sha256(config),
        "full_checkpoint_sha256": sha256(weights),
        "extracted_encoder_sha256": sha256(args.encoder_output),
        "extracted_encoder": extracted,
    }
    args.provenance_output.parent.mkdir(parents=True, exist_ok=True)
    args.provenance_output.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(provenance, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
