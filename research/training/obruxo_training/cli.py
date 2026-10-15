from __future__ import annotations

import argparse
import json
from pathlib import Path

from obruxo_data.vital import VitalSchema

from .assets import AssetCatalog
from .audit import audit_corpus, load_snapshot, write_json
from .contracts import TrainingContract
from .performances import bank_document


def _named_paths(entries: list[str], label: str) -> dict[str, Path]:
    result = {}
    for entry in entries:
        name, separator, path = entry.partition("=")
        if not separator or not name or not path or name in result:
            raise ValueError(f"comparison {label} need unique NAME=PATH entries")
        result[name] = Path(path)
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="OBRUXO training contracts, authored pilot and on-demand rendering.")
    parser.add_argument("--asset-references", type=Path, help="Hash-pinned, caller-reviewed reference export manifest")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("describe", help="Print the initial schema, stream, asset and performance contracts")
    verify = commands.add_parser("verify", help="Verify a complete snapshot and its current executable contracts")
    verify.add_argument("--snapshot", type=Path, required=True)
    audit = commands.add_parser("audit", help="Inspect a corpus read-only and write a new contract snapshot")
    audit.add_argument("--corpus", type=Path, required=True)
    audit.add_argument("--output", type=Path, required=True)
    audit.add_argument("--seed", type=int, default=0)
    audit.add_argument("--parents", type=Path, help="JSON object mapping child state IDs to parent state-ID lists")
    audit.add_argument("--limit", type=int, help="Bounded diagnostic only; a truncated population is marked partial")
    pilot = commands.add_parser("pilot", help="Author a compact statically checked recipe pool without rendering audio")
    pilot.add_argument("--output", type=Path, required=True)
    pilot.add_argument("--count", type=int, default=256)
    pilot.add_argument("--seed", type=int, default=0)
    alpha = commands.add_parser("alpha-pool", help="Author a broader pool from explicitly pinned wavetable exports")
    alpha.add_argument("--wavetable-manifest", type=Path, required=True)
    alpha.add_argument("--wavetable-root", type=Path, required=True)
    alpha.add_argument("--output", type=Path, required=True)
    alpha.add_argument("--count", type=int, default=256)
    alpha.add_argument("--seed", type=int, default=0)
    expansion = commands.add_parser("expand-pilot", help="Append authored states while verifying every frozen parent split")
    expansion.add_argument("--pool", type=Path, required=True, help="Existing parent pool; its seed and vocabulary remain fixed")
    expansion.add_argument("--output", type=Path, required=True)
    expansion.add_argument("--count", type=int, required=True, help="Total expanded population, at most 4096")
    benchmark = commands.add_parser("benchmark", help="Profile native on-demand rendering and a CPU spectral consumer")
    benchmark.add_argument("--pool", type=Path, required=True)
    benchmark.add_argument("--output", type=Path, required=True)
    benchmark.add_argument("--count", type=int, default=32)
    benchmark.add_argument("--split", choices=("train", "validation", "test"), default="train")
    benchmark.add_argument("--role", choices=("reference", "transfer"), default="reference")
    benchmark.add_argument("--prefetch-clips", type=int, default=4)
    benchmark.add_argument("--prefetch-mib", type=int, default=32)
    benchmark.add_argument("--timeout", type=float, default=60.0)
    benchmark.add_argument("--max-retries", type=int, default=0)
    benchmark.add_argument("--plugin-path", type=str)
    benchmark.add_argument("--resume", type=Path, help="Acknowledged stream cursor; queued audio is regenerated")
    codec_check = commands.add_parser("codec-check", help="Qualify native round trips and, for alpha, audible feature interventions")
    codec_check.add_argument("--pool", type=Path, required=True)
    codec_check.add_argument("--output", type=Path, required=True)
    codec_check.add_argument("--timeout", type=float, default=60.0)
    codec_check.add_argument("--plugin-path", type=str)
    train = commands.add_parser("train", help="Train the audio-only pilot with bounded online rendering or tiny replay overfit")
    train.add_argument("--pool", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--mode", choices=("overfit", "online"), default="overfit")
    train.add_argument("--steps", type=int, default=400)
    train.add_argument("--replay-capacity", type=int, default=32)
    train.add_argument("--replay-mib", type=int, default=8)
    train.add_argument("--batch-size", type=int, default=16)
    train.add_argument("--learning-rate", type=float, default=0.003)
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--learning-seed-override", type=int,
                       help="Reseed model initialization and minibatches while preserving the --seed replay sequence")
    train.add_argument("--checkpoint-every", type=int, default=50)
    train.add_argument("--render-budget", type=int, default=512)
    train.add_argument("--cpu-threads", type=int, default=4)
    train.add_argument("--sampling-policy", choices=("pairs", "state_round_robin"), default="pairs")
    train.add_argument("--updates-per-render", type=int, default=1)
    train.add_argument("--fresh-examples", type=int, help="Stop adding fresh examples at this count, then continue bounded replay")
    train.add_argument("--device", default="cpu")
    train.add_argument("--timeout", type=float, default=60)
    train.add_argument("--plugin-path", type=str)
    train.add_argument("--resume", type=Path)
    train.add_argument("--seed-replay", type=Path, help="Seed a new model from a verified training feature prefix; weights and optimizer reset")
    train.add_argument("--stop-after", type=int, help="Preempt at this optimizer step, retaining the original total-step contract")
    evaluate = commands.add_parser("evaluate", help="Compare real renders with fixed and train-only retrieval baselines")
    evaluate.add_argument("--pool", type=Path, required=True)
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--count", type=int, default=4)
    evaluate.add_argument("--split", choices=("train", "validation"), default="validation")
    evaluate.add_argument("--timeout", type=float, default=60)
    evaluate.add_argument("--plugin-path", type=str)
    curve = commands.add_parser("evaluate-curve", help="Compare checkpoints with shared validation sources and training retrieval")
    curve.add_argument("--pool", type=Path, required=True)
    curve.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    curve.add_argument("--output", type=Path, required=True)
    curve.add_argument("--count", type=int, default=21)
    curve.add_argument("--timeout", type=float, default=60)
    curve.add_argument("--plugin-path", type=str)
    curve.add_argument("--resume", action="store_true", help="Resume the same output directory from complete, hash-verified cases")
    compare = commands.add_parser("compare", help="Compare named training runs against an explicit fixed training retrieval bank")
    compare.add_argument("--pool", type=Path, required=True)
    compare.add_argument("--checkpoints", nargs="+", required=True, help="Unique NAME=PATH checkpoint entries in tie-break order")
    compare.add_argument("--retrieval-checkpoint", type=Path, required=True)
    compare.add_argument("--checkpoint-pools", nargs="+", default=[], help="NAME=PATH training pools for expanded candidates; --pool pins retrieval")
    compare.add_argument("--validation-pool", type=Path,
                         help="Verified parent-pool expansion; evaluate all its newly added validation states, never parent or test states")
    compare.add_argument("--output", type=Path, required=True)
    compare.add_argument("--count", type=int, default=21)
    compare.add_argument("--timeout", type=float, default=60)
    compare.add_argument("--plugin-path", type=str)
    compare.add_argument("--resume", action="store_true")
    infer = commands.add_parser("infer", help="Export a pilot .vital prediction from a 44.1 kHz stereo WAV")
    infer.add_argument("--model", type=Path, required=True)
    infer.add_argument("--audio", type=Path, required=True)
    infer.add_argument("--output", type=Path, required=True)
    infer.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    try:
        if args.command == "infer":
            import numpy as np
            import torch
            from scipy.io import wavfile
            from .trainer import load_model, predict_audio
            if args.asset_references:
                raise ValueError("inference uses the bundle's embedded asset catalog")
            torch.set_num_threads(4)
            sample_rate, audio = wavfile.read(args.audio)
            if audio.dtype == np.uint8:
                audio = (audio.astype(np.float32) - 128) / 128
            elif np.issubdtype(audio.dtype, np.signedinteger):
                audio = audio.astype(np.float32) / (float(np.iinfo(audio.dtype).max) + 1)
            model, frontend, codec = load_model(args.model, args.device)
            target = predict_audio(model, frontend, codec, audio.astype(np.float32), sample_rate)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("xb") as handle:
                handle.write(codec.decode(target))
            print(json.dumps({"output": str(args.output), "scope": codec.to_dict()["scope"], "target": target.to_dict()}))
            return 0
        if args.command in {"pilot", "alpha-pool", "expand-pilot", "benchmark", "codec-check", "train", "evaluate", "evaluate-curve", "compare"}:
            from .pilot import PilotPool
            if args.asset_references:
                raise ValueError("authored pools use their embedded catalog; alpha-pool takes --wavetable-manifest")
            if args.command in {"pilot", "alpha-pool", "expand-pilot"}:
                parent = PilotPool.load(args.pool) if args.command == "expand-pilot" else None
                codec = None
                if args.command == "alpha-pool":
                    from .alpha import AlphaCodec, read_wavetable_bank
                    codec = AlphaCodec(read_wavetable_bank(args.wavetable_manifest, args.wavetable_root))
                pool = parent.expand(args.count) if parent is not None else PilotPool.create(args.count, args.seed, codec=codec)
                args.output.mkdir(parents=True, exist_ok=False)
                pool.save(args.output / "pool.json")
                write_json(args.output / "summary.json", pool.summary())
                if parent is not None:
                    write_json(args.output / "expansion.json", pool.expansion_from(parent))
                print(json.dumps(pool.summary(), indent=2))
                return 0
            from .benchmark import benchmark_stream
            from .stream import NativeRendererFactory, StreamConfig
            pool = PilotPool.load(args.pool)
            if args.command == "compare":
                from .evaluate import evaluate_comparison
                checkpoints = _named_paths(args.checkpoints, "checkpoints")
                checkpoint_pools = {name: PilotPool.load(path) for name, path in _named_paths(args.checkpoint_pools, "pools").items()}
                result = evaluate_comparison(pool, checkpoints, args.retrieval_checkpoint, args.output, count=args.count,
                                             checkpoint_pools=checkpoint_pools,
                                             validation_pool=PilotPool.load(args.validation_pool) if args.validation_pool else None,
                                             renderer_factory=NativeRendererFactory(args.plugin_path), timeout=args.timeout,
                                             progress=lambda value: print(json.dumps(value), flush=True), resume=args.resume)
                print(json.dumps(result["comparison"], indent=2))
                return 0 if result["renders_accepted"] == result["renders_attempted"] else 2
            if args.command == "evaluate-curve":
                from .evaluate import evaluate_learning_curve
                result = evaluate_learning_curve(pool, args.checkpoints, args.output, count=args.count,
                                                 renderer_factory=NativeRendererFactory(args.plugin_path), timeout=args.timeout,
                                                 progress=lambda value: print(json.dumps(value), flush=True), resume=args.resume)
                print(json.dumps(result["learning_curve"], indent=2))
                return 0 if result["renders_accepted"] == result["renders_attempted"] else 2
            if args.command == "evaluate":
                from .evaluate import evaluate_pilot
                result = evaluate_pilot(pool, args.checkpoint, args.output, count=args.count, split=args.split,
                                        renderer_factory=NativeRendererFactory(args.plugin_path), timeout=args.timeout,
                                        progress=lambda value: print(json.dumps(value), flush=True))
                print(json.dumps(result["aggregates"], indent=2))
                return 0 if result["renders_accepted"] == result["renders_attempted"] else 2
            if args.command == "train":
                from .trainer import TrainConfig, train_pilot
                config = TrainConfig(**{key: getattr(args, key) for key in TrainConfig.__dataclass_fields__})
                result = train_pilot(pool, args.output, config=config, device=args.device,
                                     stream_config=StreamConfig(prefetch_clips=1, timeout_seconds=args.timeout),
                                     renderer_factory=NativeRendererFactory(args.plugin_path), resume=args.resume,
                                     seed_replay=args.seed_replay,
                                     stop_after=args.stop_after, progress=lambda value: print(json.dumps(value), flush=True))
                print(json.dumps(result, indent=2))
                return 2 if result["complete"] and config.mode == "overfit" and not result["tiny_overfit_gate"] else 0
            if args.command == "codec-check":
                from .codec_check import check_codec_audio
                result = check_codec_audio(pool, args.output, renderer_factory=NativeRendererFactory(args.plugin_path),
                                           timeout=args.timeout,
                                           progress=lambda done, total, status: print(f"Checked {done}/{total}: {status}", flush=True))
                print(json.dumps({key: value for key, value in result.items() if key != "cases"}, indent=2))
                return 0 if result["passed"] else 2
            config = StreamConfig(args.prefetch_clips, args.prefetch_mib * 1024 * 1024, args.timeout, args.max_retries)
            resume = json.loads(args.resume.read_text(encoding="utf-8")) if args.resume else None
            result = benchmark_stream(pool, args.output, count=args.count, split=args.split, role=args.role,
                                      config=config, renderer_factory=NativeRendererFactory(args.plugin_path), resume=resume,
                                      progress=lambda done, total, status: print(f"Handled {done}/{total}: {status}", flush=True))
            print(json.dumps({key: result[key] for key in ("requested", "accepted", "rejected", "errors", "elapsed_seconds",
                             "accepted_clips_per_second", "peak_reserved_bytes", "peak_process_tree_rss_bytes")}, indent=2))
            return 0 if result["accepted"] == args.count else 2
        schema = VitalSchema.load()
        if args.command == "verify":
            contract, catalog, manifest = load_snapshot(args.snapshot, schema)
            print(json.dumps({"verified": True, "contract_id": contract.contract_id, "catalog_id": catalog.catalog_id,
                              "split_manifest_id": manifest.manifest_id, "states": len(manifest.assignments)}))
            return 0
        catalog = AssetCatalog.from_reference_manifest(args.asset_references) if args.asset_references else AssetCatalog.reviewed_init(schema)
        contract = TrainingContract.create(schema, catalog)
        if args.command == "describe":
            print(json.dumps({"contract_id": contract.contract_id, "contract": contract.to_dict(),
                              "assets": catalog.to_dict(), "performances": bank_document()}, indent=2))
            return 0
        parents = json.loads(args.parents.read_text(encoding="utf-8")) if args.parents else None
        summary = audit_corpus(args.corpus, args.output, schema=schema, catalog=catalog, seed=args.seed,
                               parents=parents, limit=args.limit,
                               progress=lambda done, total: print(f"Inspected {done}/{total}", flush=True))
        print(json.dumps({key: summary[key] for key in ("population_complete", "inspected_files", "strictly_parsed_files",
                         "excluded_files", "static_eligible_files", "static_eligible_states", "contract_id", "split_manifest_id")}, indent=2))
        return 0 if summary["population_complete"] else 2
    except (OSError, ValueError, TypeError, KeyError, RuntimeError) as error:
        parser.exit(1, f"Training command failed: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
