"""Frequency-first residual codebooks and compact envelopes for Experiment 3."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.cluster.vq import kmeans2

from .stacked import (
    FEATURE_RESOLUTION,
    SEED,
    TOPOLOGY_NAMES,
    CurveDataset,
    StackedChain,
    _features,
    nearest_indices,
)


def _candidate_medoids(
    targets: np.ndarray,
    count: int,
    rng: np.random.Generator,
    *,
    exclude_zero: bool = True,
) -> np.ndarray:
    """Return an overcomplete observed candidate set; duplicates retain their weight."""
    energy = np.mean(targets * targets, axis=1)
    eligible = np.flatnonzero(energy > 1e-12) if exclude_zero else np.arange(len(targets))
    if len(eligible) < count:
        raise ValueError(f"need {count} eligible observations, found {len(eligible)}")
    values = targets[eligible]
    features = _features(values).astype(np.float32)
    candidate_count = min(len(values), max(64, count * 8))
    centers, _ = kmeans2(
        features,
        candidate_count,
        iter=25,
        minit="points",
        missing="warn",
        rng=rng,
    )
    # Snap every center to a distinct observed vector.
    selected: list[int] = []
    for center in centers:
        distances = np.mean((features - center) ** 2, axis=1)
        choice = next((int(i) for i in np.argsort(distances) if int(i) not in selected), None)
        if choice is not None:
            selected.append(choice)
    # Include high-energy observations so rare, large misses remain candidates.
    for choice in np.argsort(energy[eligible])[::-1]:
        value = int(choice)
        if value not in selected:
            selected.append(value)
        if len(selected) >= candidate_count:
            break
    return eligible[np.asarray(selected, dtype=np.int32)]


def fit_frequency_first_codewords(
    targets: np.ndarray,
    count: int,
    *,
    rng: np.random.Generator,
    initial_codes: np.ndarray | None = None,
    include_zero: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Greedily maximize total corpus error reduction over observed candidates.

    Every occurrence contributes. Consequently repeated/common configurations have
    proportionally more influence in early layers instead of being signature-capped.
    """
    learned_count = count - int(include_zero)
    if learned_count < 0:
        raise ValueError("count must include the reserved zero code")
    dimension = targets.shape[1]
    seed_codes = np.zeros((1, dimension), dtype=np.float32) if include_zero else initial_codes
    if seed_codes is None or len(seed_codes) == 0:
        raise ValueError("an initial code or zero code is required")

    target_features = _features(targets).astype(np.float32)
    seed_features = _features(seed_codes).astype(np.float32)
    _, current_loss = nearest_indices(target_features, seed_features)
    candidate_sources = _candidate_medoids(targets, learned_count, rng)
    candidate_features = target_features[candidate_sources]

    # Pairwise MSE through dot products: [observations, candidates].
    target_norm = np.sum(target_features * target_features, axis=1, keepdims=True)
    candidate_norm = np.sum(candidate_features * candidate_features, axis=1)[None, :]
    distances = np.maximum(
        (target_norm + candidate_norm - 2.0 * target_features @ candidate_features.T)
        / target_features.shape[1],
        0.0,
    )
    selected: list[int] = []
    gains: list[float] = []
    for _ in range(learned_count):
        utility = np.sum(np.maximum(current_loss[:, None] - distances, 0.0), axis=0)
        if selected:
            utility[np.asarray(selected)] = -np.inf
        choice = int(np.argmax(utility))
        if not np.isfinite(utility[choice]):
            raise ValueError("candidate pool exhausted while fitting codewords")
        selected.append(choice)
        gains.append(float(utility[choice]))
        current_loss = np.minimum(current_loss, distances[:, choice])

    sources = candidate_sources[np.asarray(selected, dtype=np.int32)]
    learned = targets[sources].astype(np.float32)
    if include_zero:
        codes = np.concatenate([np.zeros((1, dimension), dtype=np.float32), learned])
        sources = np.concatenate([np.asarray([-1], dtype=np.int32), sources])
        gains = [0.0, *gains]
    else:
        codes = learned
    return codes, sources.astype(np.int32), np.asarray(gains, dtype=np.float64)


def _fit_layers(
    values: np.ndarray,
    bases: np.ndarray,
    width: int,
    depth: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    _, reconstruction = _assign(values, bases)
    layers: list[np.ndarray] = []
    sources: list[np.ndarray] = []
    utilities: list[np.ndarray] = []
    for _ in range(depth):
        residual = values - reconstruction
        codes, source, utility = fit_frequency_first_codewords(
            residual, width, rng=rng, include_zero=True
        )
        _, addition = _assign(residual, codes)
        reconstruction = np.clip(reconstruction + addition, 0.0, 1.0)
        layers.append(codes)
        sources.append(source)
        utilities.append(utility)
    return np.asarray(layers), np.asarray(sources), np.asarray(utilities)


def _assign(values: np.ndarray, codes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    indices, _ = nearest_indices(values, codes)
    return indices, codes[indices]


def train_frequency_first_chain(
    dataset: CurveDataset,
    stock_bases: np.ndarray,
    *,
    residual_width: int = 16,
    max_depth: int = 4,
    seed: int = SEED + 3000,
) -> tuple[StackedChain, np.ndarray]:
    """Fit exactly 15 stock bases + 17 frequency-weighted corpus medoids."""
    if len(stock_bases) != 15:
        raise ValueError("Experiment 3 requires exactly 15 stock bases")
    rng = np.random.default_rng(seed)
    train = dataset.train_curves
    extra, local_sources, base_utility = fit_frequency_first_codewords(
        train, 17, rng=rng, initial_codes=stock_bases
    )
    bases = np.concatenate([stock_bases, extra]).astype(np.float32)
    layers, local_residual_sources, residual_utility = _fit_layers(
        train, bases, residual_width, max_depth, rng
    )
    global_base_sources = dataset.train_indices[local_sources]
    global_residual_sources = np.where(
        local_residual_sources < 0,
        -1,
        dataset.train_indices[np.maximum(local_residual_sources, 0)],
    )
    chain = StackedChain(
        base_width=32,
        residual_width=residual_width,
        max_depth=max_depth,
        strategy="frequency_first_shared",
        bases=bases,
        residuals=layers[:, None].astype(np.float32),
        base_source_indices=np.concatenate(
            [np.full(15, -1, dtype=np.int32), global_base_sources]
        ),
        residual_source_indices=global_residual_sources[:, None].astype(np.int32),
    )
    utility = np.full((max_depth + 1, 17), np.nan, dtype=np.float64)
    utility[0] = base_utility
    utility[1:, :residual_width] = residual_utility
    return chain, utility


def train_frequency_topology_chain(
    dataset: CurveDataset,
    shared: StackedChain,
    *,
    seed: int = SEED + 4000,
) -> tuple[StackedChain, np.ndarray]:
    train = dataset.train_curves
    residuals = np.repeat(shared.residuals, len(TOPOLOGY_NAMES), axis=1)
    sources = np.repeat(shared.residual_source_indices, len(TOPOLOGY_NAMES), axis=1)
    utilities = np.zeros((shared.max_depth, len(TOPOLOGY_NAMES), shared.residual_width))
    train_topology = dataset.topology[dataset.train_indices]
    for condition in range(len(TOPOLOGY_NAMES)):
        members = np.flatnonzero(train_topology == condition)
        if len(members) < shared.residual_width:
            continue
        rng = np.random.default_rng(seed + condition)
        layers, local_sources, local_utility = _fit_layers(
            train[members], shared.bases, shared.residual_width, shared.max_depth, rng
        )
        residuals[:, condition] = layers
        sources[:, condition] = np.where(
            local_sources < 0,
            -1,
            dataset.train_indices[members[np.maximum(local_sources, 0)]],
        )
        utilities[:, condition] = local_utility
    chain = StackedChain(
        base_width=shared.base_width,
        residual_width=shared.residual_width,
        max_depth=shared.max_depth,
        strategy="frequency_first_topology",
        bases=shared.bases,
        residuals=residuals,
        base_source_indices=shared.base_source_indices,
        residual_source_indices=sources.astype(np.int32),
        condition_kind="topology",
        condition_labels=TOPOLOGY_NAMES,
    )
    return chain, utilities


def envelope_basis(mode: str, resolution: int) -> np.ndarray:
    if mode == "scalar":
        return np.ones((1, resolution), dtype=np.float32)
    if mode == "linear":
        phase = np.linspace(-1.0, 1.0, resolution, endpoint=False, dtype=np.float32)
        return np.stack([np.ones(resolution, dtype=np.float32), phase])
    if mode == "step2":
        first = np.zeros(resolution, dtype=np.float32)
        first[: resolution // 2] = 1.0
        return np.stack([first, 1.0 - first])
    if mode == "none":
        return np.empty((0, resolution), dtype=np.float32)
    raise ValueError(f"unsupported envelope mode: {mode}")


@dataclass
class FlexibleEncoding:
    base_indices: np.ndarray
    residual_indices: list[np.ndarray]
    coefficients: list[np.ndarray]
    shifts: list[np.ndarray]


def _shifted_codes(codes: np.ndarray, shifts: int) -> np.ndarray:
    if shifts == 1:
        return codes[:, :, None, :]
    width = codes.shape[-1]
    return np.stack(
        [np.roll(codes, (index * width) // shifts, axis=-1) for index in range(shifts)],
        axis=2,
    )


def flexible_beam_encode(
    targets: np.ndarray,
    chain: StackedChain,
    conditions: np.ndarray,
    *,
    depth: int,
    mode: str,
    shifts: int = 1,
    beam_width: int = 32,
) -> FlexibleEncoding:
    target = _features(targets)
    bases = _features(chain.bases)
    layer_values = np.stack(
        [_features(layer.reshape(-1, layer.shape[-1])).reshape(layer.shape[0], layer.shape[1], -1)
         for layer in chain.residuals]
    )
    shifted = _shifted_codes(layer_values, shifts)
    basis = envelope_basis(mode, FEATURE_RESOLUTION)
    parameter_count = len(basis)
    n = len(target)
    out_base = np.empty(n, dtype=np.int32)
    out_codes = [np.empty(n, dtype=np.int16) for _ in range(depth)]
    out_coeff = [np.empty((n, parameter_count), dtype=np.float32) for _ in range(depth)]
    out_shifts = [np.empty(n, dtype=np.int8) for _ in range(depth)]

    for start in range(0, n, 48):
        stop = min(start + 48, n)
        batch_target = target[start:stop]
        batch_conditions = conditions[start:stop]
        base_mse = np.mean((batch_target[:, None] - bases[None]) ** 2, axis=2)
        width = min(beam_width, len(bases))
        choice = np.argpartition(base_mse, width - 1, axis=1)[:, :width]
        scores = np.take_along_axis(base_mse, choice, axis=1)
        order = np.argsort(scores, axis=1)
        base_paths = np.take_along_axis(choice, order, axis=1)
        rows = np.arange(stop - start)[:, None]
        prefix = bases[base_paths]
        code_paths = np.empty((stop - start, width, 0), dtype=np.int16)
        shift_paths = np.empty((stop - start, width, 0), dtype=np.int8)
        coeff_paths = np.empty((stop - start, width, 0, parameter_count), dtype=np.float32)

        for layer in range(depth):
            codes = shifted[layer, batch_conditions].reshape(stop - start, -1, FEATURE_RESOLUTION)
            remaining = batch_target[:, None, :] - prefix
            if mode == "none":
                additions = codes[:, None]
                coefficients = np.empty(
                    (stop - start, prefix.shape[1], len(codes[0]), 0), dtype=np.float32
                )
            else:
                design = codes[:, :, None, :] * basis[None, None, :, :]
                rhs = np.einsum("bwf,bkpf->bwkp", remaining, design)
                gram = np.einsum("bkpf,bkqf->bkpq", design, design)
                gram = gram[:, None] + np.eye(parameter_count, dtype=np.float32) * 1e-8
                coefficients = np.linalg.solve(gram, rhs[..., None])[..., 0]
                coefficients = np.clip(coefficients, -2.0, 2.0)
                additions = np.einsum("bwkp,bkpf->bwkf", coefficients, design)
            candidates = np.clip(prefix[:, :, None] + additions, 0.0, 1.0)
            mse = np.mean((batch_target[:, None, None] - candidates) ** 2, axis=3)
            flat = mse.reshape(stop - start, -1)
            next_width = min(beam_width, flat.shape[1])
            flat_choice = np.argpartition(flat, next_width - 1, axis=1)[:, :next_width]
            flat_scores = np.take_along_axis(flat, flat_choice, axis=1)
            flat_choice = np.take_along_axis(
                flat_choice, np.argsort(flat_scores, axis=1), axis=1
            )
            atom_count = codes.shape[1]
            parent = flat_choice // atom_count
            atom = flat_choice % atom_count
            code = atom // shifts
            shift = atom % shifts
            prefix = candidates[rows, parent, atom]
            base_paths = base_paths[rows, parent]
            code_paths = np.concatenate([code_paths[rows, parent], code[..., None]], axis=2)
            shift_paths = np.concatenate([shift_paths[rows, parent], shift[..., None]], axis=2)
            if parameter_count:
                selected_coeff = coefficients[rows, parent, atom]
                coeff_paths = np.concatenate(
                    [coeff_paths[rows, parent], selected_coeff[:, :, None]], axis=2
                )

        out_base[start:stop] = base_paths[:, 0]
        for layer in range(depth):
            out_codes[layer][start:stop] = code_paths[:, 0, layer]
            out_shifts[layer][start:stop] = shift_paths[:, 0, layer]
            if parameter_count:
                out_coeff[layer][start:stop] = coeff_paths[:, 0, layer]
    return FlexibleEncoding(out_base, out_codes, out_coeff, out_shifts)


def flexible_decode(
    chain: StackedChain,
    encoding: FlexibleEncoding,
    conditions: np.ndarray,
    *,
    mode: str,
    shifts: int = 1,
) -> np.ndarray:
    result = chain.bases[encoding.base_indices].copy()
    basis = envelope_basis(mode, result.shape[1])
    rows = np.arange(len(result))
    for layer, indices in enumerate(encoding.residual_indices):
        additions = chain.residuals[layer, conditions, indices].copy()
        if shifts > 1:
            for shift in range(shifts):
                members = encoding.shifts[layer] == shift
                additions[members] = np.roll(
                    additions[members], (shift * result.shape[1]) // shifts, axis=1
                )
        if mode != "none":
            envelope = encoding.coefficients[layer] @ basis
            additions *= envelope
        result = np.clip(result + additions, 0.0, 1.0)
    return result
