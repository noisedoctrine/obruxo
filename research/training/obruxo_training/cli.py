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


def _read_inference_audio(path):
    import numpy as np
    from scipy.io import wavfile
    sample_rate, audio = wavfile.read(path)
    if audio.dtype == np.uint8:
        audio = (audio.astype(np.float32) - 128) / 128
    elif np.issubdtype(audio.dtype, np.signedinteger):
        audio = audio.astype(np.float32) / (float(np.iinfo(audio.dtype).max) + 1)
    if audio.ndim == 1:
        audio = np.stack((audio, audio), axis=1)
    return sample_rate, audio.astype(np.float32)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="OBRUXO training contracts, authored pilot and on-demand rendering.")
    parser.add_argument("--asset-references", type=Path, help="Hash-pinned, caller-reviewed reference export manifest")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("describe", help="Print the initial schema, stream, asset and performance contracts")
    corpus = commands.add_parser("create-corpus-pool", help="Connect frozen training labels to original sources for on-demand rendering")
    for name in ("snapshot", "corpus", "candidates", "output"):
        corpus.add_argument("--" + name, type=Path, required=True)
    observations = commands.add_parser("extend-observations", help="Add bounded long-note observations to an original corpus source pool")
    observations.add_argument("--pool", type=Path, required=True)
    observations.add_argument("--output", type=Path, required=True)
    mixed = commands.add_parser("mix-sources", help="Combine original corpus and authored coverage under frozen source splits")
    for name in ("corpus-pool", "authored-pool", "snapshot", "output"):
        mixed.add_argument("--" + name, type=Path, required=True)
    proxy = commands.add_parser("proxy-screen", help="Train and qualify a small patch-to-sound helper against native audio")
    proxy.add_argument("--pool", type=Path, required=True)
    proxy.add_argument("--output", type=Path, required=True)
    proxy.add_argument("--training-states", type=int, default=64)
    proxy.add_argument("--validation-states", type=int, default=10)
    proxy.add_argument("--steps", type=int, default=4000)
    proxy.add_argument("--batch-size", type=int, default=32)
    proxy.add_argument("--learning-rate", type=float, default=.001)
    proxy.add_argument("--seed", type=int, default=0)
    proxy.add_argument("--checkpoint-every", type=int, default=250)
    proxy.add_argument("--replay-mib", type=int, default=8)
    proxy.add_argument("--cpu-threads", type=int, default=4)
    proxy.add_argument("--timeout", type=float, default=30.)
    proxy.add_argument("--plugin-path")
    proxy.add_argument("--resume", action="store_true")
    candidate = commands.add_parser("candidate-screen", help="Improve training targets with native renders and compare matched learning arms")
    candidate.add_argument("--pool", type=Path, required=True)
    candidate.add_argument("--checkpoint", type=Path, required=True)
    candidate.add_argument("--output", type=Path, required=True)
    candidate.add_argument("--training-states", type=int, default=32)
    candidate.add_argument("--validation-states", type=int, default=10)
    candidate.add_argument("--steps", type=int, default=1500)
    candidate.add_argument("--batch-size", type=int, default=16)
    candidate.add_argument("--learning-rate", type=float, default=.0001)
    candidate.add_argument("--candidate-weight", type=float, default=.5)
    candidate.add_argument("--confirmation-margin", type=float, default=.05)
    candidate.add_argument("--checkpoint-every", type=int, default=250)
    candidate.add_argument("--replay-mib", type=int, default=8)
    candidate.add_argument("--cpu-threads", type=int, default=4)
    candidate.add_argument("--seed", type=int, default=0)
    candidate.add_argument("--timeout", type=float, default=30.)
    candidate.add_argument("--plugin-path")
    candidate.add_argument("--resume", action="store_true")
    candidate_export = commands.add_parser("candidate-export", help="Export a diagnostic audio-only candidate bundle with a train-only bank")
    candidate_export.add_argument("--pool", type=Path, required=True)
    candidate_export.add_argument("--checkpoint", type=Path, required=True)
    candidate_export.add_argument("--output", type=Path, required=True)
    candidate_check = commands.add_parser("candidate-check", help="Qualify deployed audio-only ranking separately from evaluator best-of-four")
    candidate_check.add_argument("--pool", type=Path, required=True)
    candidate_check.add_argument("--checkpoint", type=Path, required=True)
    candidate_check.add_argument("--output", type=Path, required=True)
    candidate_check.add_argument("--count", type=int, default=10)
    candidate_check.add_argument("--timeout", type=float, default=30.)
    candidate_check.add_argument("--plugin-path")
    candidate_check.add_argument("--resume", action="store_true")
    candidate_infer = commands.add_parser("infer-candidates", help="Optionally rank four legal guesses from audio-derived notes and native renders")
    candidate_infer.add_argument("--model", type=Path, required=True)
    candidate_infer.add_argument("--audio", type=Path, required=True)
    candidate_infer.add_argument("--output", type=Path, required=True)
    candidate_infer.add_argument("--report", type=Path)
    candidate_infer.add_argument("--timeout", type=float, default=30.)
    candidate_infer.add_argument("--plugin-path")
    candidate_infer.add_argument("--allow-unqualified", action="store_true", help="Explicit diagnostic use of an unqualified ranker")
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
    benchmark.add_argument("--worker-jobs", type=int, default=1, help="Jobs per isolated process, 1..128; each job still creates a fresh plugin")
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
    train.add_argument("--sampling-policy", choices=("pairs", "state_round_robin", "paired_views", "performance_round_robin", "attack_compatible", "source_balanced"), default="pairs")
    train.add_argument("--identity-weight", type=float, default=0., help="Weight of paired identity contrastive loss; requires paired_views")
    train.add_argument("--identity-temperature", type=float, default=.1)
    train.add_argument("--conditioning", choices=("concat", "cross_attention"), default="concat")
    train.add_argument("--cyclic-head", choices=("sigmoid", "periodic"), default="sigmoid")
    train.add_argument("--prediction-scope", choices=("codec", "basic"), default="codec",
                       help="basic enforces the nine-control/two-choice pilot; codec preserves historical experiments")
    train.add_argument("--preroll-frames", type=int, choices=(0, 32768),
                       help="Basic defaults to the qualified lead with fixed random phase; explicit zero preserves legacy rendering")
    train.add_argument("--input-performances", choices=("initial", "varied", "mixed"), default="initial",
                       help="varied gives basic patches paired broad-pitch observations with category-independent pattern assignment")
    train.add_argument("--initial-model", type=Path, help="Verified same-pool inference weights; start a fresh optimizer and learning RNG")
    train.add_argument("--updates-per-render", type=int, default=1)
    train.add_argument("--fresh-examples", type=int, help="Stop adding fresh examples at this count, then continue bounded replay")
    train.add_argument("--device", default="cpu")
    train.add_argument("--timeout", type=float, default=60)
    train.add_argument("--worker-jobs", type=int, default=1, help="Jobs per isolated process, 1..128; frozen in the training contract")
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
    compare.add_argument("--identity-diagnostics", action="store_true", help="Measure representations across reference and reserved transfer audio")
    compare.add_argument("--validation-states", type=Path, help="JSON list of explicit validation state IDs in evaluation order")
    compare.add_argument("--allow-architecture-comparison", action="store_true", help="Explicitly compare supported conditioning variants")
    bundle = commands.add_parser("bundle-basic", help="Package a basic model and project runtime for local audio-only inference")
    bundle.add_argument("--model", type=Path, required=True)
    bundle.add_argument("--output", type=Path, required=True)
    infer = commands.add_parser("infer", help="Export a .vital prediction from an audio-only inference model and 44.1 kHz WAV")
    infer.add_argument("--model", type=Path, required=True)
    infer.add_argument("--audio", type=Path, required=True)
    infer.add_argument("--output", type=Path, required=True)
    infer.add_argument("--device", default="cpu")
    full_check = commands.add_parser("full-check", help="Qualify all 775 controls and six nested structures with authored native fixtures")
    full_check.add_argument("--output", type=Path, required=True)
    full_check.add_argument("--plugin-path")
    full_check.add_argument("--timeout", type=float, default=30.)
    scalar_pool = commands.add_parser("full-scalar-pool", help="Author full 775-scalar targets; nested structures remain explicitly fixed")
    scalar_pool.add_argument("--output", type=Path, required=True)
    scalar_pool.add_argument("--count", type=int, default=256)
    scalar_pool.add_argument("--seed", type=int, default=0)
    factory_bank = commands.add_parser("factory-bank", help="Import hash-pinned factory assets and qualify their native sound/readback")
    factory_bank.add_argument("--factory-root", type=Path, required=True)
    factory_bank.add_argument("--output", type=Path, required=True)
    factory_bank.add_argument("--manifest", type=Path)
    factory_bank.add_argument("--timeout", type=float, default=120.)
    factory_pool = commands.add_parser("factory-pool", help="Author full-scalar states with learned qualified wavetable and sample choices")
    factory_pool.add_argument("--bank", type=Path, required=True)
    factory_pool.add_argument("--output", type=Path, required=True)
    factory_pool.add_argument("--count", type=int, default=256)
    factory_pool.add_argument("--seed", type=int, default=0)
    lfo_fit = commands.add_parser("lfo-fit", help="Fit the existing Era2 curve dictionaries from registered training presets only")
    lfo_fit.add_argument("--presets", type=Path, nargs="+", required=True)
    lfo_fit.add_argument("--split-manifest", type=Path, required=True)
    lfo_fit.add_argument("--output", type=Path, required=True)
    lfo_fit.add_argument("--maximum-curves", type=int, default=256)
    lfo_pool = commands.add_parser("lfo-pool", help="Author registered factory choices and eight train-only Era2 curve outputs")
    lfo_pool.add_argument("--asset-bank", type=Path, required=True)
    lfo_pool.add_argument("--lfo-bank", type=Path, required=True)
    lfo_pool.add_argument("--output", type=Path, required=True)
    lfo_pool.add_argument("--count", type=int, default=256)
    lfo_pool.add_argument("--seed", type=int, default=0)
    routing_pool = commands.add_parser("routing-pool", help="Author all 64 learned ordered connection slots and line-map outputs")
    routing_pool.add_argument("--asset-bank", type=Path, required=True)
    routing_pool.add_argument("--lfo-bank", type=Path, required=True)
    routing_pool.add_argument("--output", type=Path, required=True)
    routing_pool.add_argument("--count", type=int, default=256)
    routing_pool.add_argument("--seed", type=int, default=0)
    nested_pool = commands.add_parser("nested-pool", help="Author custom oscillator curves and uint32 seed outputs over all ordered connections")
    nested_pool.add_argument("--asset-bank", type=Path, required=True)
    nested_pool.add_argument("--lfo-bank", type=Path, required=True)
    nested_pool.add_argument("--output", type=Path, required=True)
    nested_pool.add_argument("--count", type=int, default=256)
    nested_pool.add_argument("--seed", type=int, default=0)
    expression_pool = commands.add_parser("expression-pool", help="Author nested-state patches with expressive reference/transfer performances")
    expression_pool.add_argument("--asset-bank", type=Path, required=True)
    expression_pool.add_argument("--lfo-bank", type=Path, required=True)
    expression_pool.add_argument("--output", type=Path, required=True)
    expression_pool.add_argument("--count", type=int, default=256)
    expression_pool.add_argument("--seed", type=int, default=0)
    combined_pool = commands.add_parser("combined-pool", help="Author combined scalar, asset, curve, connection and expression recipes")
    combined_pool.add_argument("--asset-bank", type=Path, required=True)
    combined_pool.add_argument("--lfo-bank", type=Path, required=True)
    combined_pool.add_argument("--output", type=Path, required=True)
    combined_pool.add_argument("--count", type=int, default=256)
    combined_pool.add_argument("--seed", type=int, default=0)
    curve_pool = commands.add_parser("curve-pool", help="Author combined sources with original one-point curves and explicit native import")
    curve_pool.add_argument("--asset-bank", type=Path, required=True)
    curve_pool.add_argument("--lfo-bank", type=Path, required=True)
    curve_pool.add_argument("--output", type=Path, required=True)
    curve_pool.add_argument("--count", type=int, default=256)
    curve_pool.add_argument("--seed", type=int, default=0)
    graph_pool = commands.add_parser("graph-pool", help="Author paired variants with every connection destination scheduled under frozen roots")
    graph_pool.add_argument("--asset-bank", type=Path, required=True)
    graph_pool.add_argument("--lfo-bank", type=Path, required=True)
    graph_pool.add_argument("--output", type=Path, required=True)
    graph_pool.add_argument("--count", type=int)
    graph_pool.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    try:
        if args.command == "mix-sources":
            from .mixed_pool import MixedPool
            if args.asset_references:
                raise ValueError("mixed sources retain their hash-pinned source assets")
            pool = MixedPool.create(args.corpus_pool, args.authored_pool, args.snapshot, progress=lambda row: print(json.dumps(row), flush=True))
            pool.save(args.output)
            print(json.dumps(pool.summary(), indent=2))
            return 0
        if args.command == "extend-observations":
            from .observation_codec import extend_observations
            from .pilot import PilotPool
            if args.asset_references:
                raise ValueError("observation extension retains the source pool's registered assets")
            pool = extend_observations(PilotPool.load(args.pool))
            pool.save(args.output)
            print(json.dumps(pool.summary(), indent=2))
            return 0
        if args.command == "create-corpus-pool":
            from .corpus_pool import CorpusPool
            if args.asset_references:
                raise ValueError("corpus pools use the qualified candidate codec's registered assets")
            pool = CorpusPool.create(args.snapshot, args.corpus, args.candidates)
            pool.save(args.output)
            print(json.dumps(pool.summary(), indent=2))
            return 0
        if args.command == "lfo-fit":
            from .lfo_bank import fit_lfo_bank
            from .splits import SplitManifest
            if args.asset_references or args.output.exists():
                raise ValueError("lfo-fit uses registered source states and requires a new output directory")
            manifest = SplitManifest.from_dict(json.loads(args.split_manifest.read_text(encoding="utf-8")))
            bank = fit_lfo_bank((path.read_bytes() for path in args.presets), manifest, maximum_curves=args.maximum_curves,
                                progress=lambda event: print(event, flush=True))
            args.output.mkdir(parents=True, exist_ok=False)
            write_json(args.output / "bank.json", bank.to_dict())
            print(json.dumps({"bank_id": bank.bank_id, "fitted_curves": len(bank.to_dict()["bank"]["fitted_rows"])}))
            return 0
        if args.command == "factory-bank":
            from .factory_bank import MANIFEST, qualify_factory_bank
            if args.asset_references:
                raise ValueError("factory-bank uses its explicitly hash-pinned source manifest")
            result = qualify_factory_bank(args.factory_root, args.output, manifest=args.manifest or MANIFEST, timeout=args.timeout,
                                          progress=lambda event: print(json.dumps(event), flush=True))
            print(json.dumps(result["qualification"], indent=2))
            return 0 if result["qualification"]["qualified"] else 2
        if args.command == "full-check":
            from .full_check import check_full_render
            if args.asset_references:
                raise ValueError("full correctness fixtures use the pinned factory template")
            result = check_full_render(args.output, plugin_path=args.plugin_path, timeout=args.timeout,
                                       progress=lambda event: print(json.dumps(event), flush=True))
            print(json.dumps(result, indent=2))
            return 0 if result["qualified"] else 2
        if args.command in {"candidate-export", "candidate-check", "infer-candidates"}:
            import torch
            from .audio_candidates import export_candidate_model, load_candidate_model, rank_audio
            from .pilot import PilotPool
            from .stream import NativeRendererFactory
            if args.asset_references:
                raise ValueError("candidate commands use the embedded source assets")
            torch.set_num_threads(4)
            if args.command == "candidate-export":
                if args.output.exists():
                    raise ValueError("candidate export output already exists")
                args.output.parent.mkdir(parents=True, exist_ok=True)
                identity = export_candidate_model(PilotPool.load(args.pool), args.checkpoint, args.output)
                print(json.dumps({"output": str(args.output), "qualified": False, "identity": identity}, indent=2))
                return 0
            if args.command == "candidate-check":
                from .ranking_check import check_audio_ranking
                result = check_audio_ranking(PilotPool.load(args.pool), args.checkpoint, args.output, count=args.count,
                                             renderer_factory=NativeRendererFactory(args.plugin_path), timeout=args.timeout, resume=args.resume,
                                             progress=lambda event: print(json.dumps(event), flush=True))
                print(json.dumps(result, indent=2))
                return 0 if result["qualification"]["qualified"] else 2
            report = args.report or args.output.with_suffix(".ranking.json")
            if args.output.exists() or report.exists() or report.resolve() == args.output.resolve():
                raise ValueError("candidate prediction/report output already exists or collides")
            model, frontend, codec, bank, renderer = load_candidate_model(args.model, allow_unqualified=args.allow_unqualified)
            sample_rate, audio = _read_inference_audio(args.audio)
            target, result = rank_audio(model, frontend, codec, bank, audio, sample_rate=sample_rate,
                                        renderer_factory=NativeRendererFactory(args.plugin_path), timeout=args.timeout, expected_renderer=renderer)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            from .inference_profile import prediction_preset
            with args.output.open("xb") as handle:
                handle.write(prediction_preset(model, codec, target))
            report.parent.mkdir(parents=True, exist_ok=True)
            write_json(report, result)
            print(json.dumps({"output": str(args.output), "report": str(report), "ranked": result["ranked"], "selected": result["selected"]}))
            return 0 if result["ranked"] else 2
        if args.command == "candidate-screen":
            from .candidate_data import CandidateConfig
            from .candidate_screen import run_candidate_screen
            from .pilot import PilotPool
            from .stream import NativeRendererFactory
            if args.asset_references:
                raise ValueError("candidate training uses the pool's embedded asset catalog")
            config = CandidateConfig(**{name: getattr(args, name) for name in CandidateConfig.__dataclass_fields__})
            def candidate_progress(event):
                if event["stage"] != "evaluation" or event["method"] == "source_repeat":
                    print(json.dumps(event), flush=True)
            result = run_candidate_screen(PilotPool.load(args.pool), args.checkpoint, args.output, config=config,
                                          renderer_factory=NativeRendererFactory(args.plugin_path), resume=args.resume, progress=candidate_progress)
            print(json.dumps(result, indent=2))
            return 0 if result["validation_renders_accepted"] == result["validation_renders_attempted"] else 2
        if args.command == "proxy-screen":
            from .pilot import PilotPool
            from .proxy_screen import ProxyConfig, run_proxy_screen
            from .stream import NativeRendererFactory
            if args.asset_references:
                raise ValueError("proxy uses the pool's embedded asset catalog")
            config = ProxyConfig(**{name: getattr(args, name) for name in ProxyConfig.__dataclass_fields__})
            def proxy_progress(event):
                if event["stage"] != "collect" or event["attempted"] % 16 == 0 or event["status"] != "accepted":
                    print(json.dumps(event), flush=True)
            result = run_proxy_screen(PilotPool.load(args.pool), args.output, config=config,
                                      renderer_factory=NativeRendererFactory(args.plugin_path), resume=args.resume, progress=proxy_progress)
            print(json.dumps(result, indent=2))
            return 0 if result["qualified_for_inverse_training"] else 2
        if args.command == "bundle-basic":
            from .local_bundle import create_bundle
            if args.asset_references:
                raise ValueError("basic bundles use the model's embedded codec and assets")
            print(json.dumps(create_bundle(args.model, args.output)))
            return 0
        if args.command == "infer":
            import torch
            from .inference_profile import prediction_preset
            from .prediction import load_predictor
            from .trainer import predict_audio
            if args.asset_references:
                raise ValueError("inference uses the bundle's embedded asset catalog")
            sample_rate, audio = _read_inference_audio(args.audio)
            model, frontend, codec, settings = load_predictor(args.model, args.device)
            torch.set_num_threads(settings["cpu_threads"])
            target = predict_audio(model, frontend, codec, audio, sample_rate)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("xb") as handle:
                handle.write(prediction_preset(model, codec, target))
            report = {"output": str(args.output), "scope": codec.to_dict()["scope"], "target": target.to_dict()}
            if settings["qualified_for_release"] is not None:
                report["qualified_for_release"] = settings["qualified_for_release"]
            print(json.dumps(report))
            return 0
        if args.command in {"pilot", "alpha-pool", "full-scalar-pool", "factory-pool", "lfo-pool", "routing-pool", "nested-pool", "expression-pool", "combined-pool", "curve-pool", "graph-pool", "expand-pilot", "benchmark", "codec-check", "train", "evaluate", "evaluate-curve", "compare"}:
            from .pilot import PilotPool
            if args.asset_references:
                raise ValueError("authored pools use their embedded catalog; alpha-pool takes --wavetable-manifest")
            if args.command in {"pilot", "alpha-pool", "full-scalar-pool", "factory-pool", "lfo-pool", "routing-pool", "nested-pool", "expression-pool", "combined-pool", "curve-pool", "graph-pool", "expand-pilot"}:
                parent = PilotPool.load(args.pool) if args.command == "expand-pilot" else None
                codec = None
                if args.command == "alpha-pool":
                    from .alpha import AlphaCodec, read_wavetable_bank
                    codec = AlphaCodec(read_wavetable_bank(args.wavetable_manifest, args.wavetable_root))
                if args.command == "full-scalar-pool":
                    from .full_scalar import FullScalarCodec
                    codec = FullScalarCodec()
                if args.command == "factory-pool":
                    from .asset_codec import FactoryAssetCodec
                    codec = FactoryAssetCodec(json.loads(args.bank.read_text(encoding="utf-8")))
                if args.command == "lfo-pool":
                    from .lfo_codec import LfoCurveCodec
                    codec = LfoCurveCodec(json.loads(args.asset_bank.read_text(encoding="utf-8")), json.loads(args.lfo_bank.read_text(encoding="utf-8")))
                if args.command == "routing-pool":
                    from .routing import OrderedRoutingCodec
                    codec = OrderedRoutingCodec(json.loads(args.asset_bank.read_text(encoding="utf-8")), json.loads(args.lfo_bank.read_text(encoding="utf-8")))
                if args.command == "nested-pool":
                    from .nested_codec import NestedStateCodec
                    codec = NestedStateCodec(json.loads(args.asset_bank.read_text(encoding="utf-8")), json.loads(args.lfo_bank.read_text(encoding="utf-8")))
                if args.command == "expression-pool":
                    from .expression_codec import ExpressionCodec
                    codec = ExpressionCodec(json.loads(args.asset_bank.read_text(encoding="utf-8")), json.loads(args.lfo_bank.read_text(encoding="utf-8")))
                if args.command == "combined-pool":
                    from .combined_codec import CombinedCodec
                    codec = CombinedCodec(json.loads(args.asset_bank.read_text(encoding="utf-8")), json.loads(args.lfo_bank.read_text(encoding="utf-8")))
                if args.command == "curve-pool":
                    from .curve_codec import CurveInputCodec
                    codec = CurveInputCodec(json.loads(args.asset_bank.read_text(encoding="utf-8")), json.loads(args.lfo_bank.read_text(encoding="utf-8")))
                if args.command == "graph-pool":
                    from .graph_codec import GraphCoverageCodec
                    codec = GraphCoverageCodec(json.loads(args.asset_bank.read_text(encoding="utf-8")), json.loads(args.lfo_bank.read_text(encoding="utf-8")))
                    args.count = codec.required_count if args.count is None else args.count
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
            factory = NativeRendererFactory(args.plugin_path, renderer_id=pool.codec.contract.renderer_id)
            if args.command == "compare":
                from .evaluate import evaluate_comparison
                checkpoints = _named_paths(args.checkpoints, "checkpoints")
                checkpoint_pools = {name: PilotPool.load(path) for name, path in _named_paths(args.checkpoint_pools, "pools").items()}
                result = evaluate_comparison(pool, checkpoints, args.retrieval_checkpoint, args.output, count=args.count,
                                             checkpoint_pools=checkpoint_pools,
                                             identity_diagnostics=args.identity_diagnostics,
                                             allow_architecture_comparison=args.allow_architecture_comparison,
                                             validation_states=json.loads(args.validation_states.read_text(encoding="utf-8")) if args.validation_states else None,
                                             validation_pool=PilotPool.load(args.validation_pool) if args.validation_pool else None,
                                             renderer_factory=factory, timeout=args.timeout,
                                             progress=lambda value: print(json.dumps(value), flush=True), resume=args.resume)
                print(json.dumps(result["comparison"], indent=2))
                return 0 if result["renders_accepted"] == result["renders_attempted"] else 2
            if args.command == "evaluate-curve":
                from .evaluate import evaluate_learning_curve
                result = evaluate_learning_curve(pool, args.checkpoints, args.output, count=args.count,
                                                 renderer_factory=factory, timeout=args.timeout,
                                                 progress=lambda value: print(json.dumps(value), flush=True), resume=args.resume)
                print(json.dumps(result["learning_curve"], indent=2))
                return 0 if result["renders_accepted"] == result["renders_attempted"] else 2
            if args.command == "evaluate":
                from .evaluate import evaluate_pilot
                result = evaluate_pilot(pool, args.checkpoint, args.output, count=args.count, split=args.split,
                                        renderer_factory=factory, timeout=args.timeout,
                                        progress=lambda value: print(json.dumps(value), flush=True))
                print(json.dumps(result["aggregates"], indent=2))
                return 0 if result["renders_accepted"] == result["renders_attempted"] else 2
            if args.command == "train":
                from .trainer import TrainConfig, train_pilot
                if args.preroll_frames is None:
                    args.preroll_frames = 32768 if args.prediction_scope == "basic" else 0
                config = TrainConfig(**{key: getattr(args, key) for key in TrainConfig.__dataclass_fields__})
                result = train_pilot(pool, args.output, config=config, device=args.device,
                                     stream_config=StreamConfig(prefetch_clips=1, timeout_seconds=args.timeout, worker_jobs=args.worker_jobs),
                                     renderer_factory=factory, resume=args.resume,
                                     seed_replay=args.seed_replay,
                                     initial_model=args.initial_model,
                                     stop_after=args.stop_after, progress=lambda value: print(json.dumps(value), flush=True))
                print(json.dumps(result, indent=2))
                return 2 if result["complete"] and config.mode == "overfit" and not result["tiny_overfit_gate"] else 0
            if args.command == "codec-check":
                from .codec_check import check_codec_audio
                result = check_codec_audio(pool, args.output, renderer_factory=factory,
                                           timeout=args.timeout,
                                           progress=lambda done, total, status: print(f"Checked {done}/{total}: {status}", flush=True))
                print(json.dumps({key: value for key, value in result.items() if key != "cases"}, indent=2))
                return 0 if result["passed"] else 2
            config = StreamConfig(args.prefetch_clips, args.prefetch_mib * 1024 * 1024, args.timeout, args.max_retries, args.worker_jobs)
            resume = json.loads(args.resume.read_text(encoding="utf-8")) if args.resume else None
            result = benchmark_stream(pool, args.output, count=args.count, split=args.split, role=args.role,
                                      config=config, renderer_factory=factory, resume=resume,
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
