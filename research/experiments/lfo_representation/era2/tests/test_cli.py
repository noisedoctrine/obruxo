from __future__ import annotations

from contextlib import redirect_stderr
import io
from pathlib import Path
import sys
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from lfo_era2.cli import parser  # noqa: E402


class CliTests(unittest.TestCase):
    def test_framework_cli_does_not_expose_experiment10_grid_audit(self) -> None:
        cli = parser()
        self.assertNotIn("grid-ceiling", cli.format_help())
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            cli.parse_args(["grid-ceiling"])


if __name__ == "__main__":
    unittest.main()
