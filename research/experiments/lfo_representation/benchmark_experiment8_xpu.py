#!/usr/bin/env python3
"""Synthetic CPU/XPU timing check for Experiment 8 kernels.

This intentionally does not read or write Experiment 8 artifacts. Run it only
when no production experiment is using the XPU, otherwise the timings will be
contaminated and the live run will slow down.
"""

from __future__ import annotations

import argparse
import os
import time
from contextlib import contextmanager

import numpy as np

from lfo_experiment.experiment7 import (
    _apply_training_stage,
    _apply_training_stage_torch,
    _encode_final_clip_beam_torch,
    encode_final_clip_beam,
)
from lfo_experiment.phase4 import PhaseChain
from lfo_experiment.stacked import TOPOLOGY_NAMES


@contextmanager
def _temporary_env(**values: str):
    old = {key: os.environ.get(key) for key in values}
    try:
        for key, value in values.items():
            os.environ[key] = value
        yield
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _sync_xpu() -> None:
    try:
        import torch

        if hasattr(torch, "xpu") and torch.xpu.is_available():
            torch.xpu.synchronize()
    except Exception:
        return


def _time_call(label: str, fn, *, sync: bool = False) -> float:
    if sync:
        _sync_xpu()
    started = time.perf_counter()
    fn()
    if sync:
        _sync_xpu()
    elapsed = time.perf_counter() - started
    print(f"{label:28s} {elapsed:8.3f}s", flush=True)
    return elapsed


def _synthetic_chain(*, rng: np.random.Generator, bases: int, width: int, depth: int, resolution: int) -> PhaseChain:
    base_codes = rng.normal(0.5, 0.2, (bases, resolution)).astype(np.float32)
    stages = tuple(
        rng.normal(0.0, 0.05, (len(TOPOLOGY_NAMES), width, resolution)).astype(np.float32)
        for _ in range(depth)
    )
    return PhaseChain(
        name="experiment8_synthetic_benchmark",
        bases=base_codes,
        stages=stages,
        base_sources=np.arange(bases, dtype=np.int32),
        stage_sources=tuple(np.arange(width, dtype=np.int32) for _ in stages),
        stage_labels=tuple(f"stage_{index + 1}" for index in range(depth)),
        topology_conditioned=True,
        stage_layers=tuple(range(1, depth + 1)),
        stage_branches=tuple("shared" if index % 2 == 0 else "topology" for index in range(depth)),
        canonical_rotations=tuple(np.zeros(width, dtype=np.float32) for _ in stages),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-rows", type=int, default=1024)
    parser.add_argument("--eval-rows", type=int, default=96)
    parser.add_argument("--resolution", type=int, default=120)
    parser.add_argument("--bases", type=int, default=32)
    parser.add_argument("--width", type=int, default=8)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--beam-width", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--train-stage-batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=8675309)
    args = parser.parse_args()

    try:
        import torch

        has_xpu = hasattr(torch, "xpu") and torch.xpu.is_available()
        print(f"torch={torch.__version__} xpu_available={has_xpu}")
        if not has_xpu:
            raise SystemExit("XPU is unavailable in this environment")
    except ImportError as exc:
        raise SystemExit(f"PyTorch import failed: {exc}") from exc

    rng = np.random.default_rng(args.seed)
    prefix = rng.random((args.train_rows, args.resolution), dtype=np.float32)
    train_targets = rng.random((args.train_rows, args.resolution), dtype=np.float32)
    train_codes = rng.normal(0.0, 0.1, (args.width, args.resolution)).astype(np.float32)
    chain = _synthetic_chain(
        rng=rng,
        bases=args.bases,
        width=args.width,
        depth=args.depth,
        resolution=args.resolution,
    )
    eval_targets = rng.random((args.eval_rows, args.resolution), dtype=np.float32)
    conditions = rng.integers(0, len(TOPOLOGY_NAMES), size=args.eval_rows, dtype=np.int64)

    with _temporary_env(LFO_TRAIN_STAGE_BATCH_SIZE=str(args.train_stage_batch_size), LFO_ALIGN_DEVICE="cpu"):
        cpu_train = _time_call(
            "training apply cpu",
            lambda: _apply_training_stage(prefix, train_targets, train_codes),
        )
        cpu_eval = _time_call(
            "beam eval cpu",
            lambda: encode_final_clip_beam(
                eval_targets,
                chain,
                conditions,
                beam_width=args.beam_width,
                batch_size=args.batch_size,
            ),
        )

    with _temporary_env(LFO_TRAIN_STAGE_BATCH_SIZE=str(args.train_stage_batch_size), LFO_XPU_SHIFT_IMPL="roll_bank"):
        xpu_train = _time_call(
            "training apply xpu",
            lambda: _apply_training_stage_torch(prefix, train_targets, train_codes, device="xpu:0"),
            sync=True,
        )
        xpu_eval = _time_call(
            "beam eval xpu",
            lambda: _encode_final_clip_beam_torch(
                eval_targets,
                chain,
                conditions,
                beam_width=args.beam_width,
                batch_size=args.batch_size,
                device="xpu:0",
            ),
            sync=True,
        )

    print()
    print(f"training speedup: {cpu_train / xpu_train:0.2f}x")
    print(f"eval speedup:     {cpu_eval / xpu_eval:0.2f}x")


if __name__ == "__main__":
    main()
