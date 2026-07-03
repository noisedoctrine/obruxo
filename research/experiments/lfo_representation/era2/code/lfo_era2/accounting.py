"""Model prediction head budget accounting for Era 2 runtime interfaces."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any


@dataclass(frozen=True)
class BudgetBreakdown:
    addressing_scheme: str
    base_selection_outputs: int
    residual_atom_selection_outputs: int
    categorical_outputs: int
    continuous_outputs: int
    scalar_outputs: int
    head_outputs_actual: int
    head_outputs_formula: str
    parameters: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "addressing_scheme": self.addressing_scheme,
            "base_selection_outputs": self.base_selection_outputs,
            "residual_atom_selection_outputs": self.residual_atom_selection_outputs,
            "categorical_outputs": self.categorical_outputs,
            "continuous_outputs": self.continuous_outputs,
            "scalar_outputs": self.scalar_outputs,
            "head_outputs_actual": self.head_outputs_actual,
            "head_outputs_formula": self.head_outputs_formula,
            **self.parameters,
        }


@dataclass(frozen=True)
class RuntimeInterfaceSpec:
    addressing_scheme: str
    residual_layer_count: int
    dictionary_scope: str
    parameters: dict[str, Any] = field(default_factory=dict)

    def budget(self, *, base_dictionary_size: int = 32, scalar_outputs: int | None = None) -> BudgetBreakdown:
        if self.addressing_scheme == "flat_categorical":
            width = self.parameters.get("width")
            widths = self.parameters.get("widths_by_residual_layer")
            return flat_categorical_budget(
                base_dictionary_size=base_dictionary_size,
                residual_layer_count=self.residual_layer_count,
                width=width,
                widths_by_residual_layer=widths,
                scalar_outputs=scalar_outputs,
            )
        if self.addressing_scheme == "basis_coefficients":
            return basis_coefficients_budget(
                base_dictionary_size=base_dictionary_size,
                residual_layer_count=self.residual_layer_count,
                basis_count=int(self.parameters["basis_count"]),
                scalar_outputs=scalar_outputs,
            )
        if self.addressing_scheme == "path_address":
            return path_address_budget(
                base_dictionary_size=base_dictionary_size,
                residual_layer_count=self.residual_layer_count,
                branch_factors=list(self.parameters["branch_factors"]),
                scalar_outputs=scalar_outputs,
            )
        if self.addressing_scheme == "continuous_address":
            return continuous_address_budget(
                base_dictionary_size=base_dictionary_size,
                residual_layer_count=self.residual_layer_count,
                address_dim=int(self.parameters["address_dim"]),
                scalar_outputs=scalar_outputs,
            )
        raise ValueError(f"unsupported addressing_scheme: {self.addressing_scheme}")


def default_phase_scalar_outputs(residual_layer_count: int) -> int:
    return int(residual_layer_count) + 1


def flat_categorical_budget(
    *,
    base_dictionary_size: int = 32,
    residual_layer_count: int,
    width: int | None = None,
    widths_by_residual_layer: list[int] | tuple[int, ...] | None = None,
    scalar_outputs: int | None = None,
) -> BudgetBreakdown:
    D = int(residual_layer_count)
    S = default_phase_scalar_outputs(D) if scalar_outputs is None else int(scalar_outputs)
    if width is None and widths_by_residual_layer is None:
        raise ValueError("flat categorical budget needs width or widths_by_residual_layer")
    if widths_by_residual_layer is None:
        W = int(width)  # type: ignore[arg-type]
        widths = [W] * D
        residual_outputs = D * W
        formula = f"{base_dictionary_size} + D * W + (D + 1)"
        params: dict[str, Any] = {"D": D, "W": W, "W_by_residual_layer": widths}
    else:
        widths = [int(value) for value in widths_by_residual_layer]
        if len(widths) != D:
            raise ValueError("widths_by_residual_layer length must equal residual_layer_count")
        residual_outputs = sum(widths)
        formula = f"{base_dictionary_size} + sum(W_by_residual_layer) + {S}"
        params = {"D": D, "W_by_residual_layer": widths}
    categorical = int(base_dictionary_size) + residual_outputs
    total = categorical + S
    return BudgetBreakdown(
        addressing_scheme="flat_categorical",
        base_selection_outputs=int(base_dictionary_size),
        residual_atom_selection_outputs=residual_outputs,
        categorical_outputs=categorical,
        continuous_outputs=0,
        scalar_outputs=S,
        head_outputs_actual=total,
        head_outputs_formula=formula,
        parameters=params,
    )


def basis_coefficients_budget(
    *,
    base_dictionary_size: int = 32,
    residual_layer_count: int,
    basis_count: int,
    scalar_outputs: int | None = None,
) -> BudgetBreakdown:
    D = int(residual_layer_count)
    P = int(basis_count)
    S = default_phase_scalar_outputs(D) if scalar_outputs is None else int(scalar_outputs)
    continuous = D * P
    categorical = int(base_dictionary_size)
    return BudgetBreakdown(
        addressing_scheme="basis_coefficients",
        base_selection_outputs=int(base_dictionary_size),
        residual_atom_selection_outputs=continuous,
        categorical_outputs=categorical,
        continuous_outputs=continuous,
        scalar_outputs=S,
        head_outputs_actual=categorical + continuous + S,
        head_outputs_formula=f"{base_dictionary_size} + D * P + (D + 1)",
        parameters={"D": D, "P": P, "basis_count": P},
    )


def path_address_budget(
    *,
    base_dictionary_size: int = 32,
    residual_layer_count: int,
    branch_factors: list[int] | tuple[int, ...],
    scalar_outputs: int | None = None,
) -> BudgetBreakdown:
    D = int(residual_layer_count)
    factors = [int(value) for value in branch_factors]
    if not factors or any(value < 2 for value in factors):
        raise ValueError("branch_factors must contain values >= 2")
    S = default_phase_scalar_outputs(D) if scalar_outputs is None else int(scalar_outputs)
    branch_outputs = sum(factors)
    residual_outputs = D * branch_outputs
    categorical = int(base_dictionary_size) + residual_outputs
    leaf_capacity = math.prod(factors)
    return BudgetBreakdown(
        addressing_scheme="path_address",
        base_selection_outputs=int(base_dictionary_size),
        residual_atom_selection_outputs=residual_outputs,
        categorical_outputs=categorical,
        continuous_outputs=0,
        scalar_outputs=S,
        head_outputs_actual=categorical + S,
        head_outputs_formula=f"{base_dictionary_size} + D * sum(branch_factors) + (D + 1)",
        parameters={
            "D": D,
            "branch_factors": factors,
            "path_length": len(factors),
            "leaf_capacity": leaf_capacity,
        },
    )


def path_address_budget_for_width(
    *,
    base_dictionary_size: int = 32,
    residual_layer_count: int,
    width: int,
    branching_factor: int,
    scalar_outputs: int | None = None,
) -> BudgetBreakdown:
    if width < 1:
        raise ValueError("width must be positive")
    if branching_factor < 2:
        raise ValueError("branching_factor must be >= 2")
    length = max(1, math.ceil(math.log(width, branching_factor)))
    return path_address_budget(
        base_dictionary_size=base_dictionary_size,
        residual_layer_count=residual_layer_count,
        branch_factors=[branching_factor] * length,
        scalar_outputs=scalar_outputs,
    )


def continuous_address_budget(
    *,
    base_dictionary_size: int = 32,
    residual_layer_count: int,
    address_dim: int,
    scalar_outputs: int | None = None,
) -> BudgetBreakdown:
    D = int(residual_layer_count)
    E = int(address_dim)
    S = default_phase_scalar_outputs(D) if scalar_outputs is None else int(scalar_outputs)
    continuous = D * E
    categorical = int(base_dictionary_size)
    return BudgetBreakdown(
        addressing_scheme="continuous_address",
        base_selection_outputs=int(base_dictionary_size),
        residual_atom_selection_outputs=continuous,
        categorical_outputs=categorical,
        continuous_outputs=continuous,
        scalar_outputs=S,
        head_outputs_actual=categorical + continuous + S,
        head_outputs_formula=f"{base_dictionary_size} + D * E + (D + 1)",
        parameters={"D": D, "E": E, "address_dim": E},
    )

