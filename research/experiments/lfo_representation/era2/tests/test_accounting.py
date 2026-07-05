from __future__ import annotations

from pathlib import Path
import sys
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from lfo_era2.accounting import (  # noqa: E402
    basis_coefficients_budget,
    continuous_address_budget,
    flat_categorical_budget,
    path_address_budget,
)


class AccountingTests(unittest.TestCase):
    def test_flat_categorical_phase_only_formula(self) -> None:
        budget = flat_categorical_budget(residual_layer_count=3, width=4)
        self.assertEqual(budget.residual_atom_selection_outputs, 12)
        self.assertEqual(budget.categorical_outputs, 44)
        self.assertEqual(budget.scalar_outputs, 4)
        self.assertEqual(budget.head_outputs_actual, 48)
        self.assertEqual(budget.head_outputs_formula, "32 + D * W + (D + 1)")

    def test_residual_gain_budget_is_model_facing_only_when_counted(self) -> None:
        phase_only = flat_categorical_budget(residual_layer_count=16, width=8)
        residual_gain = flat_categorical_budget(residual_layer_count=16, width=8, scalar_outputs=33)
        self.assertEqual(phase_only.head_outputs_actual, 177)
        self.assertEqual(residual_gain.head_outputs_actual, 193)
        self.assertEqual(residual_gain.head_outputs_actual - phase_only.head_outputs_actual, 16)

    def test_basis_coefficients_budget(self) -> None:
        budget = basis_coefficients_budget(residual_layer_count=5, basis_count=6)
        self.assertEqual(budget.categorical_outputs, 32)
        self.assertEqual(budget.continuous_outputs, 30)
        self.assertEqual(budget.scalar_outputs, 6)
        self.assertEqual(budget.head_outputs_actual, 68)

    def test_path_address_budget(self) -> None:
        budget = path_address_budget(residual_layer_count=4, branch_factors=[2, 2, 2])
        self.assertEqual(budget.residual_atom_selection_outputs, 24)
        self.assertEqual(budget.categorical_outputs, 56)
        self.assertEqual(budget.scalar_outputs, 5)
        self.assertEqual(budget.head_outputs_actual, 61)
        self.assertEqual(budget.parameters["leaf_capacity"], 8)

    def test_continuous_address_budget(self) -> None:
        budget = continuous_address_budget(residual_layer_count=7, address_dim=3)
        self.assertEqual(budget.categorical_outputs, 32)
        self.assertEqual(budget.continuous_outputs, 21)
        self.assertEqual(budget.scalar_outputs, 8)
        self.assertEqual(budget.head_outputs_actual, 61)


if __name__ == "__main__":
    unittest.main()
