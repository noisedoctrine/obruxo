"""Local A0: hard codec/native agreement, phase-fixed repeats and unpooled loss profiling."""

import argparse
import itertools
import json
from pathlib import Path
import sys
from time import perf_counter

RESEARCH = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(RESEARCH / "training"), str(RESEARCH / "data_generation")]


def main():
    import numpy as np
    import torch
    from obruxo_data.hashing import canonical_sha256, sha256_bytes
    from obruxo_data.render.base import RenderRequest
    from obruxo_training.audit import write_json
    from obruxo_training.benchmark import MemoryMonitor
    from obruxo_training.codec import PilotTarget
    from obruxo_training.performances import initial_bank
    from obruxo_training.reproducibility import fixed_generator_phases
    from obruxo_training.spectral import SpectralTargets, spectral_loss, target_bytes
    from obruxo_training.spectral_controls import BasicControlAdapter
    from obruxo_training.stream import NativeRendererFactory, RenderWorker

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    adapter, measure = BasicControlAdapter(), SpectralTargets()
    codec, performance = adapter.codec, initial_bank()[0].performance()
    jobs = []
    for voices, filtered in itertools.product(range(3), range(2)):
        target = codec.canonical_target(PilotTarget(codec.default_target().continuous, (voices, filtered)))
        values = torch.tensor([target.continuous], dtype=torch.float32)
        scores = [torch.nn.functional.one_hot(torch.tensor([voices]), 3).float(),
                  torch.nn.functional.one_hot(torch.tensor([filtered]), 2).float()]
        exported = adapter.export(values, scores)[0]
        # Float32 adapter values are its declared numerical representation, not a byte claim about float64 labels.
        assert exported.categorical == target.categorical and np.allclose(exported.continuous, target.continuous, rtol=0, atol=1e-7)
        source = codec.decode(exported)
        request = RenderRequest(preset_json=fixed_generator_phases(source).decode(), performance=performance,
                                renderer_id=codec.contract.renderer_id, tail_seconds=codec.contract.tail_seconds)
        for repeat in range(2):
            jobs.append(({"voices_index": voices, "filter_index": filtered, "repeat": repeat,
                          "source_sha256": sha256_bytes(source), "request_id": request.request_id}, request))
    plan = {"format": "basic_spectral_a0_v1", "contract": adapter.contract(), "contract_id": canonical_sha256(adapter.contract()),
            "source_sha256": {str(path.relative_to(RESEARCH)): sha256_bytes(path.read_bytes().replace(b"\r\n", b"\n"))
                              for path in (Path(__file__), RESEARCH / "training/obruxo_training/spectral.py",
                                           RESEARCH / "training/obruxo_training/spectral_controls.py")},
            "jobs": [identity for identity, _ in jobs], "planned_requests": len(jobs), "maximum_requests": 12,
            "maximum_seconds": 1800, "maximum_process_tree_bytes": 8 * 1024**3, "cpu_threads": 2,
            "device": "cpu", "precision": "float32", "optimizer_updates": 0, "audio_archive": False,
            "quality_claim": "contract_and_resource_profile_only", "test_split_used": False,
            "fixtures": "six_default_control_category_combinations_not_training_population"}
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "plan.json", plan)
    factory = NativeRendererFactory(renderer_id=codec.contract.renderer_id)
    monitor, rows, profiles, failure = MemoryMonitor(), [], [], None
    monitor.start()
    started, source, renderer_identity = perf_counter(), None, None
    try:
        for identity, request in jobs:
            remaining = plan["maximum_seconds"] - (perf_counter() - started)
            if remaining <= 0 or (monitor.peak_bytes or 0) > plan["maximum_process_tree_bytes"]:
                raise RuntimeError("A0 resource limit reached")
            with RenderWorker(renderer_factory=factory, max_jobs=1) as worker:
                audio, metadata = worker.render(request, timeout=min(60., remaining))
            row = {**identity, **metadata}
            rows.append(row)
            write_json(args.output / "events.json", rows)
            if audio is None or metadata["status"] != "accepted":
                raise RuntimeError("A0 native render failed; no automatic replacement")
            provenance = {key: value for key, value in metadata["provenance"].items() if key != "request_id"}
            if renderer_identity is not None and provenance != renderer_identity:
                raise RuntimeError("renderer identity changed")
            renderer_identity = provenance
            tensor = torch.from_numpy(audio.copy()).T[None]
            measured_at = perf_counter()
            target = measure(tensor)
            row.update(target_bytes=target_bytes(target), measurement_seconds=perf_counter() - measured_at,
                       spectral_shapes=[list(value.shape) for value in target["magnitudes"]])
            if identity["repeat"] == 0:
                source = audio.copy()
                row["identity_loss"] = float(spectral_loss(target, target)["loss"].item())
                for label, altered in (("gain", tensor * .5), ("pan", torch.cat((tensor[:, :1], tensor[:, 1:] * .5), 1)),
                                       ("timing", torch.cat((torch.zeros_like(tensor[..., :4410]), tensor[..., :-4410]), -1))):
                    row[f"{label}_loss"] = float(spectral_loss(measure(altered), target)["loss"].item())
                    if row[f"{label}_loss"] <= 0:
                        raise RuntimeError(f"measurement lost {label} sensitivity")
            else:
                row["repeat_exact"] = bool(np.array_equal(source, audio))
                if not row["repeat_exact"]:
                    raise RuntimeError("phase-fixed repeat differs")
            write_json(args.output / "events.json", rows)
            print(json.dumps({"completed": len(rows), **identity, "target_bytes": row["target_bytes"]}), flush=True)
        # Profile the actual objective/backward allocation, not a dense hypothetical helper output layer.
        for seconds, batch in ((4, 1), (4, 2), (4, 4), (4, 8), (16, 1)):
            if perf_counter() - started >= plan["maximum_seconds"] or (monitor.peak_bytes or 0) > plan["maximum_process_tree_bytes"]:
                raise RuntimeError("A0 resource limit reached")
            tensor = torch.zeros(batch, 2, seconds * 44100)
            tensor[:, :, :min(len(source), tensor.shape[-1])] = torch.from_numpy(source[:tensor.shape[-1]].copy()).T
            tick = perf_counter()
            target = measure(tensor)
            prediction = {"magnitudes": tuple((value * .9).requires_grad_() for value in target["magnitudes"]),
                          "envelope": (target["envelope"] * .9).requires_grad_()}
            loss = spectral_loss(prediction, target)["loss"].mean()
            loss.backward()
            gradients = [value.grad for value in (*prediction["magnitudes"], prediction["envelope"])]
            if not all(value is not None and torch.isfinite(value).all() for value in gradients):
                raise RuntimeError("objective produced missing/nonfinite gradients")
            size = target_bytes(target)
            profiles.append({"seconds": seconds, "batch": batch, "target_bytes": size,
                             "measure_and_backward_seconds": perf_counter() - tick,
                             "records_in_512_mib": min(512, (512 * 1024**2) // (size // batch))})
            del tensor, target, prediction, loss, gradients
    except Exception as error:
        failure = f"{type(error).__name__}: {error}"
        raise
    finally:
        monitor.close()
        elapsed = perf_counter() - started
        summary = {"plan": plan, "events": rows, "profiles": profiles, "failure": failure,
                   "complete": failure is None and len(rows) == len(jobs) and len(profiles) == 5,
                   "elapsed_seconds": elapsed, "peak_process_tree_rss_bytes": monitor.peak_bytes,
                   "rss_sampling_errors": monitor.errors, "renderer_identity": renderer_identity,
                   "within_resource_limits": monitor.errors == 0 and monitor.peak_bytes is not None
                       and monitor.peak_bytes <= plan["maximum_process_tree_bytes"] and elapsed <= plan["maximum_seconds"],
                   "profile_scope": "targets_and_loss_backward_excludes_future_helper_and_inverse_optimizer"}
        write_json(args.output / "summary.json", summary)
    return 0 if summary["complete"] and summary["within_resource_limits"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
