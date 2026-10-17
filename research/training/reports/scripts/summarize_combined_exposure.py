"""Summarize supervised control exposure in a retained training checkpoint."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

RESEARCH = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(RESEARCH / "training"), str(RESEARCH / "data_generation")]


def main():
    import torch
    from obruxo_training.audit import write_json
    from obruxo_training.capability_audit import _scalar_view
    from obruxo_training.codec import PilotTarget, load_codec
    from obruxo_training.full_scalar import EFFECTS, scalar_family
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if saved.get("format") != "obruxo_pilot_training_v1" or not saved.get("replay"):
        raise ValueError("exposure needs a retained training checkpoint")
    codec = load_codec(saved["run_contract"]["codec"])
    values, counts, effects = defaultdict(set), Counter(), Counter()
    choices = [set() for _ in range(4)]
    offset = len(codec.base.scalar.categorical)
    for item in saved["replay"]:
        target = PilotTarget.from_dict(item["target"])
        settings, mask = _scalar_view(codec, target)
        for name, enabled in mask.items():
            if enabled:
                counts[name] += 1
                values[name].add(settings[name])
        effects.update(effect for effect in EFFECTS if settings[effect + "_on"])
        for seen, choice, enabled in zip(choices, target.categorical[offset:offset + 4], codec.categorical_mask(target)[offset:offset + 4], strict=True):
            if enabled:
                seen.add(choice)
    status = {name: "masked" if not counts[name] else "constant" if len(values[name]) == 1 else "varied" for name in codec.schema.scalar_names}
    families = {family: dict(Counter(value for name, value in status.items() if scalar_family(name) == family)) for family in sorted({scalar_family(name) for name in status})}
    result = {"analysis": "retained_supervision_exposure_not_learning_v1", "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
              "script_sha256_lf": hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest(),
              "pool_id": saved["run_contract"]["pool_id"], "codec_id": codec.codec_id, "examples": len(saved["replay"]),
              "distinct_states": len({item["state_id"] for item in saved["replay"]}), "scalar_status_counts": dict(Counter(status.values())),
              "scalar_families": families, "active_asset_choices_by_slot": [len(seen) for seen in choices], "enabled_effect_state_counts": dict(effects),
              "render_attempts": 0, "optimizer_steps": 0, "final_test_audio": False, "model_promotion": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, result)
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
