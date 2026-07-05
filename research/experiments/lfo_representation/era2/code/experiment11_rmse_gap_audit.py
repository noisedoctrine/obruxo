#!/usr/bin/env python3
"""Run the Experiment 11 RMSE gap forensic audit."""

from __future__ import annotations

import os
import sys


def _preimport_runtime_flags(argv: list[str]) -> list[str]:
    cleaned = [argv[0]]
    index = 1
    while index < len(argv):
        item = argv[index]
        if item == "--mkl-threading-layer" and index + 1 < len(argv):
            os.environ["MKL_THREADING_LAYER"] = argv[index + 1]
            index += 2
            continue
        if item == "--native-threads" and index + 1 < len(argv):
            value = argv[index + 1]
            os.environ["OPENBLAS_NUM_THREADS"] = value
            os.environ["OMP_NUM_THREADS"] = value
            os.environ["MKL_NUM_THREADS"] = value
            index += 2
            continue
        cleaned.append(item)
        index += 1
    return cleaned


sys.argv = _preimport_runtime_flags(sys.argv)

from lfo_era2.rmse_gap_audit import main  # noqa: E402


if __name__ == "__main__":
    main()
