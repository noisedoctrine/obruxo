#!/usr/bin/env python
"""Convenience wrapper for running the Era 2 framework CLI from the code folder."""

import os
import sys


def _pop_runtime_env_args(argv: list[str]) -> dict[str, str]:
    env: dict[str, str] = {}
    value_flags = {
        "--mkl-threading-layer": "MKL_THREADING_LAYER",
        "--openblas-threads": "OPENBLAS_NUM_THREADS",
        "--omp-threads": "OMP_NUM_THREADS",
        "--mkl-threads": "MKL_NUM_THREADS",
    }
    index = 1
    while index < len(argv):
        arg = argv[index]
        if arg == "--native-threads":
            value = _require_flag_value(argv, index)
            env.update(
                {
                    "OPENBLAS_NUM_THREADS": value,
                    "OMP_NUM_THREADS": value,
                    "MKL_NUM_THREADS": value,
                }
            )
            del argv[index : index + 2]
            continue
        if arg.startswith("--native-threads="):
            value = arg.split("=", 1)[1]
            env.update(
                {
                    "OPENBLAS_NUM_THREADS": value,
                    "OMP_NUM_THREADS": value,
                    "MKL_NUM_THREADS": value,
                }
            )
            del argv[index]
            continue
        matched = False
        for flag, name in value_flags.items():
            if arg == flag:
                env[name] = _require_flag_value(argv, index)
                del argv[index : index + 2]
                matched = True
                break
            if arg.startswith(f"{flag}="):
                env[name] = arg.split("=", 1)[1]
                del argv[index]
                matched = True
                break
        if not matched:
            index += 1
    return env


def _require_flag_value(argv: list[str], index: int) -> str:
    if index + 1 >= len(argv) or argv[index + 1].startswith("--"):
        raise SystemExit(f"{argv[index]} requires a value")
    return argv[index + 1]


if __name__ == "__main__":
    runtime_env = _pop_runtime_env_args(sys.argv)
    os.environ.update(runtime_env)
    from lfo_era2.cli import main

    main()
