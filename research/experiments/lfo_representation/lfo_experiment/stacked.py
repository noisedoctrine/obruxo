"""Stacked residual-vector-quantization utilities for Experiment 2."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .catalog import load_codebook
from .curve import sample_shape
from .model import LfoShape


SEED = 20260621
FEATURE_RESOLUTION = 128
TOPOLOGY_NAMES = ("smooth", "continuous", "discontinuous")


def _bool(value: object) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)


def _author_is_validation(key: str) -> bool:
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % 100 < 20


def _topology(row: pd.Series) -> int:
    if _bool(row["smooth"]):
        return 0
    x = np.asarray(json.loads(row["points"]), dtype=np.float32)[::2]
    return 2 if np.any(np.diff(x) == 0.0) else 1


def _features(values: np.ndarray) -> np.ndarray:
    if values.shape[1] % FEATURE_RESOLUTION == 0:
        return values[:, :: values.shape[1] // FEATURE_RESOLUTION]
    source = np.linspace(0.0, 1.0, values.shape[1], endpoint=False)
    target = np.linspace(0.0, 1.0, FEATURE_RESOLUTION, endpoint=False)
    return np.stack([np.interp(target, source, row) for row in values])


@dataclass
class CurveDataset:
    frame: pd.DataFrame
    curves: np.ndarray
    features: np.ndarray
    topology: np.ndarray
    train_indices: np.ndarray
    validation_indices: np.ndarray

    @property
    def train_curves(self) -> np.ndarray:
        return self.curves[self.train_indices]

    @property
    def train_features(self) -> np.ndarray:
        return self.features[self.train_indices]

    @property
    def validation_curves(self) -> np.ndarray:
        return self.curves[self.validation_indices]

    @property
    def validation_features(self) -> np.ndarray:
        return self.features[self.validation_indices]


def load_curve_dataset(
    catalog_path: Path,
    *,
    resolution: int = 1024,
    active_only: bool = True,
) -> CurveDataset:
    frame = pd.read_csv(catalog_path, dtype={"author_id": str}, keep_default_na=False)
    if "author_id" not in frame:
        raise ValueError("catalog lacks author_id; regenerate it before Experiment 2")
    if active_only:
        active = frame["materially_active"].astype(str).str.lower() == "true"
        frame = frame[active].copy()
    frame.reset_index(drop=True, inplace=True)

    curves: list[np.ndarray] = []
    for row in frame.itertuples(index=False):
        shape = LfoShape.from_serialized(
            row.points, row.powers, name=row.shape_name, smooth=_bool(row.smooth)
        )
        curves.append(sample_shape(shape, resolution).astype(np.float32))
    curve_array = np.stack(curves)

    author_keys = []
    for row in frame.itertuples(index=False):
        key = str(row.author_id or row.author or f"preset:{row.preset_id}")
        author_keys.append(key)
    validation_mask = np.asarray([_author_is_validation(key) for key in author_keys])
    if not np.any(validation_mask) or np.all(validation_mask):
        raise ValueError("deterministic author split produced an empty partition")

    topology = np.asarray([_topology(row) for _, row in frame.iterrows()], dtype=np.int8)
    return CurveDataset(
        frame=frame,
        curves=curve_array,
        features=_features(curve_array).astype(np.float32),
        topology=topology,
        train_indices=np.flatnonzero(~validation_mask),
        validation_indices=np.flatnonzero(validation_mask),
    )


def load_stock_curves(codebook_path: Path, resolution: int = 1024) -> tuple[list[str], np.ndarray]:
    entries = load_codebook(codebook_path)
    names = [name for name, _ in entries]
    curves = np.stack([sample_shape(shape, resolution) for _, shape in entries])
    return names, curves.astype(np.float32)


def nearest_indices(values: np.ndarray, codes: np.ndarray, chunk_size: int = 512) -> tuple[np.ndarray, np.ndarray]:
    indices = np.empty(len(values), dtype=np.int32)
    losses = np.empty(len(values), dtype=np.float32)
    for start in range(0, len(values), chunk_size):
        stop = min(start + chunk_size, len(values))
        difference = values[start:stop, None, :] - codes[None, :, :]
        mse = np.mean(difference * difference, axis=2)
        choice = np.argmin(mse, axis=1)
        indices[start:stop] = choice
        losses[start:stop] = mse[np.arange(stop - start), choice]
    return indices, losses


def capped_training_indices(frame: pd.DataFrame, indices: np.ndarray, cap: int = 8) -> np.ndarray:
    selected: list[int] = []
    counts: dict[str, int] = {}
    for index in indices:
        signature = str(frame.iloc[index]["shape_signature"])
        count = counts.get(signature, 0)
        if count < cap:
            selected.append(int(index))
            counts[signature] = count + 1
    return np.asarray(selected, dtype=np.int32)


def _snap_centers(
    targets: np.ndarray,
    target_features: np.ndarray,
    centers: np.ndarray,
    *,
    forbidden: set[int] | None = None,
) -> np.ndarray:
    forbidden = set() if forbidden is None else set(forbidden)
    sources: list[int] = []
    for center in centers:
        distances = np.mean((target_features - center[None, :]) ** 2, axis=1)
        for candidate in np.argsort(distances):
            value = int(candidate)
            if value not in forbidden and value not in sources:
                sources.append(value)
                break
    return np.asarray(sources, dtype=np.int32)


def fit_observed_codewords(
    targets: np.ndarray,
    count: int,
    *,
    rng: np.random.Generator,
    include_zero: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Cluster for grouping, then store only observed vectors as codewords."""
    from scipy.cluster.vq import kmeans2
    if count < 1:
        raise ValueError("codeword count must be positive")
    nonzero_count = count - int(include_zero)
    dimension = targets.shape[1]
    if nonzero_count == 0:
        return np.zeros((1, dimension), dtype=np.float32), np.asarray([-1], dtype=np.int32)

    nontrivial = np.flatnonzero(np.mean(targets * targets, axis=1) > 1e-12)
    if len(nontrivial) < nonzero_count:
        raise ValueError(f"need {nonzero_count} non-zero observations, found {len(nontrivial)}")
    pool = targets[nontrivial]
    pool_features = _features(pool).astype(np.float32)

    centers, _ = kmeans2(
        pool_features,
        nonzero_count,
        iter=20,
        minit="points",
        missing="warn",
        rng=rng,
    )
    local_sources = _snap_centers(pool, pool_features, centers)

    # Three snap-refinement rounds: means group observations, but only observed
    # vectors are retained as decodable codewords.
    for _ in range(3):
        snapped_features = pool_features[local_sources]
        assignments, losses = nearest_indices(pool_features, snapped_features)
        updated: list[int] = []
        for code in range(nonzero_count):
            members = np.flatnonzero(assignments == code)
            if len(members):
                center = np.mean(pool_features[members], axis=0)
                distances = np.mean((pool_features[members] - center) ** 2, axis=1)
                ordered = members[np.argsort(distances)]
            else:
                ordered = np.argsort(losses)[::-1]
            choice = next((int(item) for item in ordered if int(item) not in updated), None)
            if choice is None:
                choice = next(item for item in range(len(pool)) if item not in updated)
            updated.append(choice)
        local_sources = np.asarray(updated, dtype=np.int32)

    source_indices = nontrivial[local_sources]
    observed = targets[source_indices].astype(np.float32)
    if include_zero:
        codes = np.concatenate([np.zeros((1, dimension), dtype=np.float32), observed])
        sources = np.concatenate([np.asarray([-1], dtype=np.int32), source_indices])
    else:
        codes = observed
        sources = source_indices

    # Zero can legitimately absorb tiny residuals. Ensure every learned non-zero
    # code still owns at least one training observation, replacing dead entries
    # with the worst represented observed target when necessary.
    first_learned = int(include_zero)
    for _ in range(count * 2):
        assignments, losses = nearest_indices(targets, codes)
        usage = np.bincount(assignments, minlength=count)
        dead = [code for code in range(first_learned, count) if usage[code] == 0]
        if not dead:
            break
        used_sources = {int(value) for value in sources if value >= 0}
        candidates = np.argsort(losses)[::-1]
        for code in dead:
            replacement = next(
                (int(item) for item in candidates if int(item) not in used_sources), None
            )
            if replacement is None:
                raise ValueError("could not replace a dead learned codeword")
            codes[code] = targets[replacement]
            sources[code] = replacement
            used_sources.add(replacement)
    return codes, sources


@dataclass
class StackedChain:
    base_width: int
    residual_width: int
    max_depth: int
    strategy: str
    bases: np.ndarray
    residuals: np.ndarray  # [layer, condition, code, phase]
    base_source_indices: np.ndarray
    residual_source_indices: np.ndarray  # [layer, condition, code]
    condition_kind: str = "shared"
    condition_labels: tuple[str, ...] = ("shared",)

    @property
    def conditions(self) -> int:
        return self.residuals.shape[1]

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            directory / "codebook.npz",
            bases=self.bases,
            residuals=self.residuals,
            base_source_indices=self.base_source_indices,
            residual_source_indices=self.residual_source_indices,
        )
        manifest = {
            "base_width": self.base_width,
            "residual_width": self.residual_width,
            "max_depth": self.max_depth,
            "strategy": self.strategy,
            "condition_kind": self.condition_kind,
            "condition_labels": list(self.condition_labels),
            "provisional_stock_bases": 15,
        }
        (directory / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _assign_and_reconstruct(values: np.ndarray, codes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    assignments, _ = nearest_indices(values, codes)
    return assignments, codes[assignments]


def train_shared_chain(
    dataset: CurveDataset,
    stock_bases: np.ndarray,
    *,
    base_width: int,
    residual_width: int,
    max_depth: int = 4,
    seed: int = SEED,
) -> StackedChain:
    if base_width < len(stock_bases):
        raise ValueError("base width cannot be smaller than the fixed stock codebook")
    rng = np.random.default_rng(seed + base_width * 100 + residual_width)
    train = dataset.train_curves
    capped_global = capped_training_indices(dataset.frame, dataset.train_indices)
    capped_local = np.searchsorted(dataset.train_indices, capped_global)

    bases = stock_bases.copy()
    base_sources = np.full(len(stock_bases), -1, dtype=np.int32)
    extra = base_width - len(stock_bases)
    if extra:
        _, current_loss = nearest_indices(train, bases)
        candidates = capped_local[current_loss[capped_local] > 1e-10]
        learned, sources = fit_observed_codewords(
            train[candidates], extra, rng=rng, include_zero=False
        )
        bases = np.concatenate([bases, learned])
        base_sources = np.concatenate([base_sources, dataset.train_indices[candidates[sources]]])

    base_assignments, reconstruction = _assign_and_reconstruct(train, bases)
    residual_layers: list[np.ndarray] = []
    source_layers: list[np.ndarray] = []
    for layer in range(max_depth):
        targets = train - reconstruction
        learned, sources = fit_observed_codewords(
            targets[capped_local], residual_width, rng=rng, include_zero=True
        )
        global_sources = np.where(
            sources < 0, -1, dataset.train_indices[capped_local[np.maximum(sources, 0)]]
        ).astype(np.int32)
        assignments, addition = _assign_and_reconstruct(targets, learned)
        reconstruction = np.clip(reconstruction + addition, 0.0, 1.0)
        residual_layers.append(learned)
        source_layers.append(global_sources)

    return StackedChain(
        base_width=base_width,
        residual_width=residual_width,
        max_depth=max_depth,
        strategy="shared",
        bases=bases.astype(np.float32),
        residuals=np.asarray(residual_layers, dtype=np.float32)[:, None, :, :],
        base_source_indices=base_sources,
        residual_source_indices=np.asarray(source_layers, dtype=np.int32)[:, None, :],
    )


def training_conditions(dataset: CurveDataset, chain: StackedChain, kind: str) -> tuple[np.ndarray, tuple[str, ...]]:
    if kind == "topology":
        return dataset.topology[dataset.train_indices], TOPOLOGY_NAMES
    if kind == "base":
        assignments, _ = nearest_indices(dataset.train_curves, chain.bases)
        return assignments.astype(np.int32), tuple(f"base_{i}" for i in range(len(chain.bases)))
    raise ValueError(f"unsupported condition kind: {kind}")


def train_conditional_chain(
    dataset: CurveDataset,
    shared: StackedChain,
    *,
    kind: str,
    min_support: int = 64,
    seed: int = SEED,
) -> StackedChain:
    conditions, labels = training_conditions(dataset, shared, kind)
    count = len(labels)
    residuals = np.repeat(shared.residuals, count, axis=1)
    sources = np.repeat(shared.residual_source_indices, count, axis=1)
    train = dataset.train_curves
    base_assignments, _ = nearest_indices(train, shared.bases)

    for condition in range(count):
        members = np.flatnonzero(conditions == condition)
        if len(members) < min_support:
            continue
        rng = np.random.default_rng(seed + condition * 1000 + shared.base_width * 100 + shared.residual_width)
        reconstruction = shared.bases[base_assignments[members]].copy()
        local_global = dataset.train_indices[members]
        capped_global = capped_training_indices(dataset.frame, local_global)
        position = {int(value): i for i, value in enumerate(local_global)}
        capped_local = np.asarray([position[int(value)] for value in capped_global], dtype=np.int32)
        for layer in range(shared.max_depth):
            targets = train[members] - reconstruction
            available = np.count_nonzero(
                np.mean(targets[capped_local] * targets[capped_local], axis=1) > 1e-12
            )
            if available < shared.residual_width - 1:
                learned = shared.residuals[layer, 0]
                mapped = shared.residual_source_indices[layer, 0]
            else:
                learned, local_sources = fit_observed_codewords(
                    targets[capped_local], shared.residual_width, rng=rng, include_zero=True
                )
                mapped = np.where(
                    local_sources < 0,
                    -1,
                    local_global[capped_local[np.maximum(local_sources, 0)]],
                )
            assignments, additions = _assign_and_reconstruct(targets, learned)
            reconstruction = np.clip(reconstruction + additions, 0.0, 1.0)
            residuals[layer, condition] = learned
            sources[layer, condition] = mapped

    return StackedChain(
        base_width=shared.base_width,
        residual_width=shared.residual_width,
        max_depth=shared.max_depth,
        strategy=f"{kind}_conditioned",
        bases=shared.bases,
        residuals=residuals,
        base_source_indices=shared.base_source_indices,
        residual_source_indices=sources,
        condition_kind=kind,
        condition_labels=labels,
    )


def validation_conditions(dataset: CurveDataset, chain: StackedChain) -> np.ndarray:
    if chain.condition_kind == "shared":
        return np.zeros(len(dataset.validation_indices), dtype=np.int32)
    if chain.condition_kind == "topology":
        return dataset.topology[dataset.validation_indices].astype(np.int32)
    assignments, _ = nearest_indices(dataset.validation_curves, chain.bases)
    return assignments.astype(np.int32)


@dataclass
class BeamEncoding:
    base_indices: np.ndarray
    residual_indices: list[np.ndarray]
    gains: list[np.ndarray]


def beam_encode(
    targets: np.ndarray,
    chain: StackedChain,
    conditions: np.ndarray,
    *,
    depth: int,
    beam_width: int = 32,
    use_gains: bool = False,
    feature_only: bool = True,
) -> BeamEncoding:
    target_values = _features(targets) if feature_only else targets
    bases = _features(chain.bases) if feature_only else chain.bases
    residuals = (
        np.stack([_features(layer.reshape(-1, layer.shape[-1])).reshape(layer.shape[0], layer.shape[1], -1) for layer in chain.residuals])
        if feature_only
        else chain.residuals
    )

    all_bases = np.empty(len(targets), dtype=np.int32)
    all_residuals = [np.empty(len(targets), dtype=np.int16) for _ in range(depth)]
    all_gains = [np.ones(len(targets), dtype=np.float32) for _ in range(depth)]
    chunk_size = 64
    for start in range(0, len(targets), chunk_size):
        stop = min(start + chunk_size, len(targets))
        target = target_values[start:stop]
        condition = conditions[start:stop]
        base_mse = np.mean((target[:, None, :] - bases[None, :, :]) ** 2, axis=2)
        width = min(beam_width, len(bases))
        selected = np.argpartition(base_mse, width - 1, axis=1)[:, :width]
        selected_scores = np.take_along_axis(base_mse, selected, axis=1)
        order = np.argsort(selected_scores, axis=1)
        selected = np.take_along_axis(selected, order, axis=1)
        prefix = bases[selected]
        base_paths = selected.copy()
        residual_paths = np.empty((stop - start, width, 0), dtype=np.int16)
        gain_paths = np.empty((stop - start, width, 0), dtype=np.float32)

        for layer in range(depth):
            layer_codes = residuals[layer, condition]  # [batch, code, feature]
            candidate_addition = layer_codes[:, None, :, :]
            if use_gains:
                remaining = target[:, None, None, :] - prefix[:, :, None, :]
                denominator = np.sum(candidate_addition * candidate_addition, axis=3)
                numerator = np.sum(remaining * candidate_addition, axis=3)
                gains = np.divide(
                    numerator,
                    denominator,
                    out=np.zeros_like(numerator),
                    where=denominator > 1e-12,
                )
                gains = np.clip(gains, -2.0, 2.0)
            else:
                gains = np.ones(
                    (stop - start, prefix.shape[1], layer_codes.shape[1]),
                    dtype=np.float32,
                )
            candidates = prefix[:, :, None, :] + gains[:, :, :, None] * candidate_addition
            candidates = np.clip(candidates, 0.0, 1.0)
            mse = np.mean((target[:, None, None, :] - candidates) ** 2, axis=3)
            flat_mse = mse.reshape(stop - start, -1)
            next_width = min(beam_width, flat_mse.shape[1])
            flat_choice = np.argpartition(flat_mse, next_width - 1, axis=1)[:, :next_width]
            choice_scores = np.take_along_axis(flat_mse, flat_choice, axis=1)
            choice_order = np.argsort(choice_scores, axis=1)
            flat_choice = np.take_along_axis(flat_choice, choice_order, axis=1)
            parent = flat_choice // layer_codes.shape[1]
            code = flat_choice % layer_codes.shape[1]
            batch = np.arange(stop - start)[:, None]
            prefix = candidates[batch, parent, code]
            base_paths = base_paths[batch, parent]
            residual_paths = np.concatenate(
                [residual_paths[batch, parent], code[:, :, None].astype(np.int16)], axis=2
            )
            selected_gain = gains[batch, parent, code]
            gain_paths = np.concatenate(
                [gain_paths[batch, parent], selected_gain[:, :, None]], axis=2
            )

        all_bases[start:stop] = base_paths[:, 0]
        for layer in range(depth):
            all_residuals[layer][start:stop] = residual_paths[:, 0, layer]
            all_gains[layer][start:stop] = gain_paths[:, 0, layer]

    return BeamEncoding(all_bases, all_residuals, all_gains)


def decode_encoding(
    chain: StackedChain,
    encoding: BeamEncoding,
    conditions: np.ndarray,
) -> np.ndarray:
    reconstructed = chain.bases[encoding.base_indices].copy()
    rows = np.arange(len(reconstructed))
    for layer, indices in enumerate(encoding.residual_indices):
        additions = chain.residuals[layer, conditions, indices]
        reconstructed = np.clip(
            reconstructed + encoding.gains[layer][:, None] * additions, 0.0, 1.0
        )
    return reconstructed


def metric_arrays(reference: np.ndarray, reconstructed: np.ndarray) -> dict[str, np.ndarray]:
    difference = reconstructed - reference
    reference_delta = np.diff(np.concatenate([reference, reference[:, :1]], axis=1), axis=1)
    reconstructed_delta = np.diff(
        np.concatenate([reconstructed, reconstructed[:, :1]], axis=1), axis=1
    )
    return {
        "rmse": np.sqrt(np.mean(difference * difference, axis=1)),
        "max_abs_error": np.max(np.abs(difference), axis=1),
        "derivative_rmse": np.sqrt(
            np.mean((reconstructed_delta - reference_delta) ** 2, axis=1)
        ),
    }
