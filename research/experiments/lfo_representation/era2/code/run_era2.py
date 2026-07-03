#!/usr/bin/env python
"""Convenience wrapper for running the Era 2 framework CLI from the code folder."""

import sys


if __name__ == "__main__":
    if "grid-ceiling" in sys.argv[1:]:
        print("run_era2: starting grid-ceiling command; importing Era 2 CLI...", flush=True)
    from lfo_era2.cli import main

    main()
