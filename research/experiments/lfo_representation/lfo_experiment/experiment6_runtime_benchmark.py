"""Actual Experiment 6 runtime benchmark for CPU/XPU/backend parallelism.

This intentionally runs real candidate evaluation on a held-out corpus slice.
It writes only under the device_benchmark scratch tree, not the production
Experiment 6 checkpoint directory.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import time

from .experiment6 import CandidateJob, _run_candidate_job, _truncate_chain
from .phase4 import PhaseChain, compose_additive, compose_switch


def _now_id() -> str:
    return datetime.now(timezone.utc).astimezone().strftime("%Y%m%d_%H%M%S")


def _candidate_jobs(experiment4_dir: Path, *, beam_width: int) -> list[CandidateJob]:
    codebook_dir = experiment4_dir / "codebooks"
    shared = PhaseChain.load(codebook_dir / "phase_shared")
    topology = PhaseChain.load(codebook_dir / "phase_topology")
    chains = [
        _truncate_chain(compose_additive(shared, topology, 16), 8, "phase_additive_k16_d4"),
        _truncate_chain(compose_additive(shared, topology, 12), 8, "phase_additive_k12_d4"),
        _truncate_chain(compose_switch(shared, topology, 1), 4, "phase_switch_1_d4"),
    ]
    jobs: list[CandidateJob] = []
    for chain in chains:
        jobs.append(
            CandidateJob(
                job_id=f"{chain.name}_bw{beam_width}_eval1920",
                kind="phase_residual",
                eval_resolution=1920,
                beam_width=beam_width,
                batch_size=1,
                training_feature_grid=128,
                chain=chain,
                weight=1.0,
            )
        )
    return jobs


def _run_one(args: tuple[str, str, Path, Path, CandidateJob, int]) -> dict[str, object]:
    route, align_device, catalog_path, output_dir, job, max_shapes = args
    os.environ["LFO_ALIGN_DEVICE"] = align_device
    os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir.parent / "mpl"))
    started = time.perf_counter()
    _run_candidate_job(catalog_path, output_dir / route, job, max_shapes=max_shapes, quick=False)
    elapsed = time.perf_counter() - started
    return {
        "route": route,
        "align_device": align_device,
        "job_id": job.job_id,
        "elapsed_seconds": elapsed,
        "max_shapes": max_shapes,
        "beam_width": job.beam_width,
        "eval_resolution": job.eval_resolution,
    }


def _clean(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def run_runtime_benchmark(
    catalog_path: Path,
    experiment4_dir: Path,
    output_root: Path,
    *,
    beam_width: int = 64,
    max_shapes: int = 8,
    parallel: int = 2,
) -> Path:
    run_dir = output_root / f"runtime_{_now_id()}_bw{beam_width}_n{max_shapes}"
    _clean(run_dir)
    jobs = _candidate_jobs(experiment4_dir, beam_width=beam_width)
    heavy = jobs[0]
    second = jobs[1]
    rows: list[dict[str, object]] = []

    # Single worst-case candidate, clean A/B.
    for device in ("cpu", "xpu"):
        rows.append(_run_one((f"{device}_serial_one", device, catalog_path, run_dir, heavy, max_shapes)))

    # Two-candidate sequential vs process-parallel throughput. This is the part
    # that exposes XPU copy contention under candidate-level parallelism.
    for device in ("cpu", "xpu"):
        started = time.perf_counter()
        first = _run_one((f"{device}_sequential_two", device, catalog_path, run_dir, heavy, max_shapes))
        second_result = _run_one((f"{device}_sequential_two", device, catalog_path, run_dir, second, max_shapes))
        rows.append({
            "route": f"{device}_sequential_two_total",
            "align_device": device,
            "job_id": f"{heavy.job_id}+{second.job_id}",
            "elapsed_seconds": time.perf_counter() - started,
            "component_seconds": [first["elapsed_seconds"], second_result["elapsed_seconds"]],
            "max_shapes": max_shapes,
            "beam_width": beam_width,
            "eval_resolution": 1920,
        })

    for device in ("cpu", "xpu"):
        route = f"{device}_parallel{parallel}_two"
        args = [
            (route, device, catalog_path, run_dir, job, max_shapes)
            for job in (heavy, second)
        ]
        started = time.perf_counter()
        with ProcessPoolExecutor(max_workers=parallel) as executor:
            component = list(executor.map(_run_one, args))
        rows.append({
            "route": f"{route}_total",
            "align_device": device,
            "job_id": f"{heavy.job_id}+{second.job_id}",
            "elapsed_seconds": time.perf_counter() - started,
            "component_seconds": [row["elapsed_seconds"] for row in component],
            "max_shapes": max_shapes,
            "beam_width": beam_width,
            "eval_resolution": 1920,
        })

    report = {
        "created_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "run_dir": str(run_dir),
        "settings": {
            "beam_width": beam_width,
            "max_shapes": max_shapes,
            "parallel": parallel,
            "jobs": [
                {
                    "job_id": job.job_id,
                    "kind": job.kind,
                    "chain": job.chain.name if job.chain is not None else None,
                    "eval_resolution": job.eval_resolution,
                    "beam_width": job.beam_width,
                    "batch_size": job.batch_size,
                }
                for job in jobs
            ],
        },
        "results": rows,
    }
    path = run_dir / "runtime_benchmark.json"
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("catalog", type=Path)
    parser.add_argument("experiment4", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--beam-width", type=int, default=64)
    parser.add_argument("--max-shapes", type=int, default=8)
    parser.add_argument("--parallel", type=int, default=2)
    args = parser.parse_args()
    os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
    os.environ.setdefault("MPLCONFIGDIR", str(args.output / "mpl"))
    path = run_runtime_benchmark(
        args.catalog,
        args.experiment4,
        args.output,
        beam_width=args.beam_width,
        max_shapes=args.max_shapes,
        parallel=args.parallel,
    )
    print(path)


if __name__ == "__main__":
    main()
