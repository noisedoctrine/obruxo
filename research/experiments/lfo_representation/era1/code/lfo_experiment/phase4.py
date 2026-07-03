"""Circular phase-factorized residual codebooks used by Experiment 4."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from .stacked import FEATURE_RESOLUTION, SEED, TOPOLOGY_NAMES, CurveDataset, _features


EPS = 1e-10


def circular_shift(values: np.ndarray, phase: np.ndarray | float) -> np.ndarray:
    """Fractionally roll periodic rows by phase cycles using linear interpolation."""
    array = np.asarray(values, dtype=np.float32)
    was_vector = array.ndim == 1
    if was_vector:
        array = array[None, :]
    phases = np.broadcast_to(np.asarray(phase, dtype=np.float64), (len(array),))
    width = array.shape[1]
    position = (np.arange(width)[None, :] - phases[:, None] * width) % width
    left = np.floor(position).astype(np.int64) % width
    fraction = (position - left).astype(np.float32)
    rows = np.arange(len(array))[:, None]
    shifted = array[rows, left] * (1.0 - fraction) + array[rows, (left + 1) % width] * fraction
    return shifted[0] if was_vector else shifted


def _correlations(targets: np.ndarray, codes: np.ndarray) -> np.ndarray:
    """Circular correlations where index s corresponds to circular_shift(code, s/F)."""
    target_fft = np.fft.rfft(targets, axis=-1)
    code_fft = np.fft.rfft(codes, axis=-1)
    return np.fft.irfft(
        target_fft[:, None, :] * np.conj(code_fft[None, :, :]),
        n=targets.shape[1],
        axis=-1,
    ).real.astype(np.float32)


def phase_distances(
    targets: np.ndarray,
    codes: np.ndarray,
    *,
    gains: bool,
    allow_phase: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return best MSE, phase, and gain for every target/code pair."""
    correlations = _correlations(targets, codes)
    if not allow_phase:
        correlations = correlations[..., :1]
    denominator = np.sum(codes * codes, axis=1)[None, :, None]
    if gains:
        scale = np.divide(
            correlations,
            denominator,
            out=np.zeros_like(correlations),
            where=denominator > EPS,
        )
        scale = np.clip(scale, -2.0, 2.0)
    else:
        scale = np.ones_like(correlations)
    target_energy = np.sum(targets * targets, axis=1)[:, None, None]
    mse = (target_energy - 2.0 * scale * correlations + scale * scale * denominator) / targets.shape[1]
    zero = np.sum(codes * codes, axis=1) <= EPS
    if np.any(zero):
        mse[:, zero, 1:] = np.inf
        scale[:, zero] = 0.0
    choice = np.argmin(mse, axis=2)
    rows = np.arange(len(targets))[:, None]
    cols = np.arange(len(codes))[None, :]
    best_mse = np.maximum(mse[rows, cols, choice], 0.0)
    best_gain = scale[rows, cols, choice]
    phase = choice.astype(np.float32) / targets.shape[1]
    return best_mse.astype(np.float32), phase, best_gain.astype(np.float32)


def sample_phase_distances(
    targets: np.ndarray,
    codes: np.ndarray,
    *,
    gains: bool,
    allow_phase: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized phase search for one code dictionary per target row."""
    target_fft = np.fft.rfft(targets, axis=-1)
    code_fft = np.fft.rfft(codes, axis=-1)
    correlations = np.fft.irfft(
        target_fft[:, None] * np.conj(code_fft), n=targets.shape[1], axis=-1
    ).real.astype(np.float32)
    if not allow_phase:
        correlations = correlations[..., :1]
    denominator = np.sum(codes * codes, axis=2)[..., None]
    if gains:
        scale = np.divide(correlations, denominator, out=np.zeros_like(correlations), where=denominator > EPS)
        scale = np.clip(scale, -2.0, 2.0)
    else:
        scale = np.ones_like(correlations)
    target_energy = np.sum(targets * targets, axis=1)[:, None, None]
    mse = (target_energy - 2 * scale * correlations + scale * scale * denominator) / targets.shape[1]
    zero = denominator[..., 0] <= EPS
    flat_mse = mse.reshape(-1, mse.shape[-1])
    flat_scale = scale.reshape(-1, scale.shape[-1])
    flat_mse[zero.ravel(), 1:] = np.inf
    flat_scale[zero.ravel()] = 0.0
    choice = np.argmin(mse, axis=2)
    rows = np.arange(len(targets))[:, None]
    cols = np.arange(codes.shape[1])[None]
    return (
        np.maximum(mse[rows, cols, choice], 0).astype(np.float32),
        choice.astype(np.float32) / targets.shape[1],
        scale[rows, cols, choice].astype(np.float32),
    )


def canonical_orientation(
    code: np.ndarray,
    targets: np.ndarray,
    current_loss: np.ndarray,
    *,
    gains: bool,
) -> tuple[np.ndarray, float, float]:
    """Orient a code so phase zero gives its largest frequency-weighted global effect."""
    feature = _features(code[None])[0]
    correlations = _correlations(_features(targets), feature[None])[:, 0]
    denominator = float(np.sum(feature * feature))
    if gains and denominator > EPS:
        scale = np.clip(correlations / denominator, -2.0, 2.0)
    else:
        scale = np.ones_like(correlations)
    energy = np.sum(_features(targets) ** 2, axis=1)[:, None]
    mse = (energy - 2 * scale * correlations + scale * scale * denominator) / FEATURE_RESOLUTION
    utility = np.sum(np.maximum(current_loss[:, None] - mse, 0.0), axis=0)
    rotation = int(np.argmax(utility))
    return circular_shift(code, rotation / FEATURE_RESOLUTION), rotation / FEATURE_RESOLUTION, float(utility[rotation])


def fit_phase_codewords(
    targets: np.ndarray,
    count: int,
    *,
    rng: np.random.Generator,
    initial_codes: np.ndarray | None = None,
    include_zero: bool,
    gains: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Frequency-first facility selection under phase-invariant distance."""
    from .residual3 import _candidate_medoids
    learned_count = count - int(include_zero)
    feature_targets = _features(targets).astype(np.float32)
    if include_zero:
        current_loss = np.mean(feature_targets * feature_targets, axis=1)
    elif initial_codes is not None:
        distance, _, _ = phase_distances(
            feature_targets, _features(initial_codes), gains=False, allow_phase=True
        )
        current_loss = np.min(distance, axis=1)
    else:
        raise ValueError("base fitting requires initial codes")

    candidates = _candidate_medoids(targets, max(learned_count, 1), rng)
    candidate_values = targets[candidates]
    candidate_features = feature_targets[candidates]
    distances, _, _ = phase_distances(feature_targets, candidate_features, gains=gains)
    selected: list[int] = []
    utilities: list[float] = []
    orientations: list[float] = []
    codes: list[np.ndarray] = []
    for _ in range(learned_count):
        utility = np.sum(np.maximum(current_loss[:, None] - distances, 0.0), axis=0)
        if selected:
            utility[np.asarray(selected)] = -np.inf
        chosen = int(np.argmax(utility))
        if not np.isfinite(utility[chosen]):
            raise ValueError("phase candidate pool exhausted")
        oriented, rotation, zero_utility = canonical_orientation(
            candidate_values[chosen], targets, current_loss, gains=gains
        )
        selected.append(chosen)
        codes.append(oriented)
        orientations.append(rotation)
        utilities.append(zero_utility)
        current_loss = np.minimum(current_loss, distances[:, chosen])

    source = candidates[np.asarray(selected, dtype=np.int32)]
    if include_zero:
        code_array = np.concatenate([np.zeros((1, targets.shape[1]), np.float32), np.asarray(codes)])
        source = np.concatenate([np.asarray([-1], np.int32), source])
        utilities = [0.0, *utilities]
        orientations = [0.0, *orientations]
    else:
        code_array = np.asarray(codes)
    return (
        code_array.astype(np.float32),
        source.astype(np.int32),
        np.asarray(utilities, np.float64),
        np.asarray(orientations, np.float32),
    )


@dataclass
class PhaseChain:
    name: str
    bases: np.ndarray
    stages: tuple[np.ndarray, ...]  # each [condition, code, phase]
    base_sources: np.ndarray
    stage_sources: tuple[np.ndarray, ...]
    stage_labels: tuple[str, ...]
    topology_conditioned: bool
    stage_layers: tuple[int, ...]
    stage_branches: tuple[str, ...]
    canonical_rotations: tuple[np.ndarray, ...]

    @property
    def stage_widths(self) -> tuple[int, ...]:
        return tuple(stage.shape[1] for stage in self.stages)

    @property
    def stored_floats(self) -> int:
        return int(self.bases.size + sum(stage.size for stage in self.stages))

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        payload = {"bases": self.bases, "base_sources": self.base_sources}
        for index, (stage, source, rotation) in enumerate(
            zip(self.stages, self.stage_sources, self.canonical_rotations)
        ):
            payload[f"stage_{index}"] = stage
            payload[f"stage_source_{index}"] = source
            payload[f"stage_rotation_{index}"] = rotation
        np.savez_compressed(directory / "codebook.npz", **payload)
        manifest = {
            "name": self.name,
            "stage_widths": self.stage_widths,
            "stage_labels": self.stage_labels,
            "stage_layers": self.stage_layers,
            "stage_branches": self.stage_branches,
            "topology_conditioned": self.topology_conditioned,
            "canonical_rule": "maximum frequency-weighted zero-offset utility",
        }
        (directory / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, directory: Path) -> "PhaseChain":
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        payload = np.load(directory / "codebook.npz")
        count = len(manifest["stage_widths"])
        return cls(
            manifest["name"], payload["bases"],
            tuple(payload[f"stage_{i}"] for i in range(count)),
            payload["base_sources"],
            tuple(payload[f"stage_source_{i}"] for i in range(count)),
            tuple(manifest["stage_labels"]), bool(manifest["topology_conditioned"]),
            tuple(manifest["stage_layers"]), tuple(manifest["stage_branches"]),
            tuple(payload[f"stage_rotation_{i}"] for i in range(count)),
        )


@dataclass
class PhaseEncoding:
    base_indices: np.ndarray
    base_phases: np.ndarray
    stage_indices: list[np.ndarray]
    stage_phases: list[np.ndarray]
    stage_gains: list[np.ndarray]


def refine_encoding(
    targets: np.ndarray,
    chain: PhaseChain,
    encoding: PhaseEncoding,
    conditions: np.ndarray,
    *,
    base_phase: bool,
    residual_phase: bool,
    gains: bool,
    steps: int = 9,
) -> PhaseEncoding:
    """Refine coarse phases locally against full-resolution prefixes."""
    result = np.empty_like(targets)
    offsets = np.linspace(-1.0 / FEATURE_RESOLUTION, 1.0 / FEATURE_RESOLUTION, steps)
    for row in range(len(targets)):
        base = chain.bases[encoding.base_indices[row]]
        candidates = encoding.base_phases[row] + (offsets if base_phase else np.asarray([0.0]))
        shifted = circular_shift(np.repeat(base[None], len(candidates), axis=0), candidates)
        error = np.mean((shifted - targets[row]) ** 2, axis=1)
        choice = int(np.argmin(error))
        encoding.base_phases[row] = candidates[choice] % 1.0
        result[row] = shifted[choice]
    for stage_index, stage in enumerate(chain.stages):
        for row in range(len(targets)):
            index = int(encoding.stage_indices[stage_index][row])
            if index == 0:
                encoding.stage_phases[stage_index][row] = 0.0
                encoding.stage_gains[stage_index][row] = 0.0
                continue
            code = stage[conditions[row], index]
            offsets_here = offsets if residual_phase else np.asarray([0.0])
            phases = encoding.stage_phases[stage_index][row] + offsets_here
            shifted = circular_shift(np.repeat(code[None], len(phases), axis=0), phases)
            remaining = targets[row] - result[row]
            if gains:
                denominator = np.sum(shifted * shifted, axis=1)
                scale = np.divide(
                    shifted @ remaining,
                    denominator,
                    out=np.zeros(len(phases), dtype=np.float32),
                    where=denominator > EPS,
                )
                scale = np.clip(scale, -2.0, 2.0)
            else:
                scale = np.ones(len(phases), np.float32)
            candidates = np.clip(result[row][None] + scale[:, None] * shifted, 0.0, 1.0)
            error = np.mean((candidates - targets[row]) ** 2, axis=1)
            choice = int(np.argmin(error))
            baseline_error = float(np.mean((result[row] - targets[row]) ** 2))
            if error[choice] >= baseline_error - 1e-12:
                encoding.stage_indices[stage_index][row] = 0
                encoding.stage_phases[stage_index][row] = 0.0
                encoding.stage_gains[stage_index][row] = 0.0
                continue
            encoding.stage_phases[stage_index][row] = phases[choice] % 1.0
            encoding.stage_gains[stage_index][row] = scale[choice]
            result[row] = candidates[choice]
    return encoding


def quantize_phases(encoding: PhaseEncoding, bins: int) -> PhaseEncoding:
    return PhaseEncoding(
        encoding.base_indices.copy(),
        (np.round(encoding.base_phases * bins) / bins) % 1.0,
        [value.copy() for value in encoding.stage_indices],
        [(np.round(value * bins) / bins) % 1.0 for value in encoding.stage_phases],
        [value.copy() for value in encoding.stage_gains],
    )


def _condition_codes(stage: np.ndarray, conditions: np.ndarray) -> np.ndarray:
    return stage[conditions]


def encode_phase_chain(
    targets: np.ndarray,
    chain: PhaseChain,
    conditions: np.ndarray,
    *,
    base_phase: bool,
    residual_phase: bool,
    gains: bool,
    beam_width: int = 64,
) -> PhaseEncoding:
    """Beam encode on the 128-point search grid with phase/gain factorization."""
    target = _features(targets).astype(np.float32)
    bases = _features(chain.bases).astype(np.float32)
    n = len(target)
    out_base = np.empty(n, np.int16)
    out_base_phase = np.zeros(n, np.float32)
    out_indices = [np.empty(n, np.int16) for _ in chain.stages]
    out_phases = [np.zeros(n, np.float32) for _ in chain.stages]
    out_gains = [np.zeros(n, np.float32) for _ in chain.stages]

    for start in range(0, n, 24):
        stop = min(start + 24, n)
        batch = target[start:stop]
        b = len(batch)
        distance, phase, _ = phase_distances(batch, bases, gains=False, allow_phase=base_phase)
        flat = distance.reshape(b, -1)
        width = min(beam_width, flat.shape[1])
        chosen = np.argpartition(flat, width - 1, axis=1)[:, :width]
        score = np.take_along_axis(flat, chosen, axis=1)
        chosen = np.take_along_axis(chosen, np.argsort(score, axis=1), axis=1)
        base_code = chosen
        base_phase_values = np.take_along_axis(phase, base_code, axis=1)
        prefix = np.stack(
            [circular_shift(bases[base_code[row]], base_phase_values[row]) for row in range(b)]
        )
        base_paths = base_code.astype(np.int16)
        base_phase_paths = base_phase_values.astype(np.float32)
        index_paths = np.empty((b, width, 0), np.int16)
        phase_paths = np.empty((b, width, 0), np.float32)
        gain_paths = np.empty((b, width, 0), np.float32)

        for stage_index, stage in enumerate(chain.stages):
            codes = _features(stage.reshape(-1, stage.shape[-1])).reshape(
                stage.shape[0], stage.shape[1], -1
            )
            selected_codes = _condition_codes(codes, conditions[start:stop])
            remaining = batch[:, None, :] - prefix
            bw = remaining.shape[1]
            flat_remaining = remaining.reshape(b * bw, -1)
            repeated_codes = np.repeat(selected_codes, bw, axis=0)
            d, p, g = sample_phase_distances(
                flat_remaining, repeated_codes, gains=gains, allow_phase=residual_phase
            )
            d = d.reshape(b, bw, -1)
            p = p.reshape(b, bw, -1)
            g = g.reshape(b, bw, -1)
            flat_d = d.reshape(b, -1)
            next_width = min(beam_width, flat_d.shape[1])
            choice = np.argpartition(flat_d, next_width - 1, axis=1)[:, :next_width]
            choice_score = np.take_along_axis(flat_d, choice, axis=1)
            choice = np.take_along_axis(choice, np.argsort(choice_score, axis=1), axis=1)
            code_count = selected_codes.shape[1]
            parent = choice // code_count
            code = choice % code_count
            rows = np.arange(b)[:, None]
            selected_phase = p[rows, parent, code]
            selected_gain = g[rows, parent, code]
            additions = np.stack(
                [
                    circular_shift(selected_codes[row, code[row]], selected_phase[row])
                    * selected_gain[row, :, None]
                    for row in range(b)
                ]
            )
            prefix = np.clip(prefix[rows, parent] + additions, 0.0, 1.0)
            base_paths = base_paths[rows, parent]
            base_phase_paths = base_phase_paths[rows, parent]
            index_paths = np.concatenate([index_paths[rows, parent], code[..., None]], axis=2)
            phase_paths = np.concatenate([phase_paths[rows, parent], selected_phase[..., None]], axis=2)
            gain_paths = np.concatenate([gain_paths[rows, parent], selected_gain[..., None]], axis=2)

        out_base[start:stop] = base_paths[:, 0]
        out_base_phase[start:stop] = base_phase_paths[:, 0]
        for stage_index in range(len(chain.stages)):
            out_indices[stage_index][start:stop] = index_paths[:, 0, stage_index]
            out_phases[stage_index][start:stop] = phase_paths[:, 0, stage_index]
            out_gains[stage_index][start:stop] = gain_paths[:, 0, stage_index]
    return PhaseEncoding(out_base, out_base_phase, out_indices, out_phases, out_gains)


def decode_phase_chain(
    chain: PhaseChain,
    encoding: PhaseEncoding,
    conditions: np.ndarray,
) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
    base_canonical = chain.bases[encoding.base_indices]
    result = circular_shift(base_canonical, encoding.base_phases)
    cumulative = [result.copy()]
    transformed: list[np.ndarray] = []
    rows = np.arange(len(result))
    for stage_index, stage in enumerate(chain.stages):
        code = stage[conditions, encoding.stage_indices[stage_index]]
        addition = circular_shift(code, encoding.stage_phases[stage_index])
        addition *= encoding.stage_gains[stage_index][:, None]
        result = np.clip(result + addition, 0.0, 1.0)
        transformed.append(addition)
        cumulative.append(result.copy())
    return result, cumulative, transformed


def train_phase_endpoints(
    dataset: CurveDataset,
    stock: np.ndarray,
    *,
    residual_width: int = 16,
    depth: int = 4,
    seed: int = SEED + 5000,
) -> tuple[PhaseChain, PhaseChain, dict[str, np.ndarray]]:
    """Train phase-aware shared and topology endpoint dictionaries."""
    rng = np.random.default_rng(seed)
    train = dataset.train_curves
    base_zero_loss = np.mean(_features(train) ** 2, axis=1)
    oriented_stock = []
    stock_rotations = []
    for code in stock:
        oriented, rotation, _ = canonical_orientation(
            code, train, base_zero_loss, gains=False
        )
        oriented_stock.append(oriented)
        stock_rotations.append(rotation)
    stock = np.asarray(oriented_stock, dtype=np.float32)
    extra, extra_source, base_utility, base_rotation = fit_phase_codewords(
        train, 17, rng=rng, initial_codes=stock, include_zero=False, gains=False
    )
    bases = np.concatenate([stock, extra]).astype(np.float32)
    base_sources = np.concatenate([
        np.full(len(stock), -1, np.int32), dataset.train_indices[extra_source]
    ])
    base_rotation_full = np.concatenate([np.asarray(stock_rotations, np.float32), base_rotation])
    base_distance, base_phase, _ = phase_distances(
        _features(train), _features(bases), gains=False
    )
    base_choice = np.argmin(base_distance, axis=1)
    reconstruction = circular_shift(bases[base_choice], base_phase[np.arange(len(train)), base_choice])

    shared_stages: list[np.ndarray] = []
    shared_sources: list[np.ndarray] = []
    shared_rotations: list[np.ndarray] = []
    shared_utility: list[np.ndarray] = []
    for _ in range(depth):
        residual = train - reconstruction
        codes, source, utility, rotation = fit_phase_codewords(
            residual, residual_width, rng=rng, include_zero=True, gains=True
        )
        distance, phase, gain = phase_distances(_features(residual), _features(codes), gains=True)
        choice = np.argmin(distance, axis=1)
        addition = circular_shift(codes[choice], phase[np.arange(len(train)), choice])
        addition *= gain[np.arange(len(train)), choice, None]
        reconstruction = np.clip(reconstruction + addition, 0.0, 1.0)
        shared_stages.append(codes[None])
        shared_sources.append(np.where(source < 0, -1, dataset.train_indices[np.maximum(source, 0)])[None])
        shared_rotations.append(rotation[None])
        shared_utility.append(utility)

    shared = PhaseChain(
        "phase_shared", bases, tuple(shared_stages), base_sources,
        tuple(shared_sources), tuple(f"layer_{i+1}" for i in range(depth)), False,
        tuple(range(1, depth + 1)), tuple("shared" for _ in range(depth)),
        tuple(shared_rotations),
    )

    topology_stages: list[np.ndarray] = []
    topology_sources: list[np.ndarray] = []
    topology_rotations: list[np.ndarray] = []
    train_topology = dataset.topology[dataset.train_indices]
    topology_reconstruction = reconstruction * 0 + circular_shift(
        bases[base_choice], base_phase[np.arange(len(train)), base_choice]
    )
    for layer in range(depth):
        layer_codes = []
        layer_sources = []
        layer_rotations = []
        additions = np.zeros_like(train)
        for condition in range(len(TOPOLOGY_NAMES)):
            members = np.flatnonzero(train_topology == condition)
            residual = train[members] - topology_reconstruction[members]
            codes, source, _, rotation = fit_phase_codewords(
                residual, residual_width, rng=np.random.default_rng(seed + 1000 + layer * 10 + condition),
                include_zero=True, gains=True,
            )
            distance, phase, gain = phase_distances(_features(residual), _features(codes), gains=True)
            choice = np.argmin(distance, axis=1)
            selected = circular_shift(codes[choice], phase[np.arange(len(members)), choice])
            additions[members] = selected * gain[np.arange(len(members)), choice, None]
            layer_codes.append(codes)
            layer_sources.append(np.where(source < 0, -1, dataset.train_indices[members[np.maximum(source, 0)]]))
            layer_rotations.append(rotation)
        topology_reconstruction = np.clip(topology_reconstruction + additions, 0.0, 1.0)
        topology_stages.append(np.asarray(layer_codes))
        topology_sources.append(np.asarray(layer_sources))
        topology_rotations.append(np.asarray(layer_rotations))

    topology = PhaseChain(
        "phase_topology", bases, tuple(topology_stages), base_sources,
        tuple(topology_sources), tuple(f"layer_{i+1}" for i in range(depth)), True,
        tuple(range(1, depth + 1)), tuple("topology" for _ in range(depth)),
        tuple(topology_rotations),
    )
    audit = {"base_utility": base_utility, "base_rotation": base_rotation_full,
             "shared_utility": np.asarray(shared_utility)}
    return shared, topology, audit


def compose_partitioned(
    shared: PhaseChain,
    topology: PhaseChain,
    shared_counts: Sequence[int],
    *,
    name: str,
) -> PhaseChain:
    stages = []
    sources = []
    rotations = []
    for layer, shared_nonzero in enumerate(shared_counts):
        topology_nonzero = 15 - shared_nonzero
        condition_codes = []
        condition_sources = []
        condition_rotations = []
        for condition in range(3):
            codes = [np.zeros_like(shared.stages[layer][0, :1])]
            src = [np.asarray([-1], np.int32)]
            rot = [np.asarray([0.0], np.float32)]
            if shared_nonzero:
                codes.append(shared.stages[layer][0, 1 : 1 + shared_nonzero])
                src.append(shared.stage_sources[layer][0, 1 : 1 + shared_nonzero])
                rot.append(shared.canonical_rotations[layer][0, 1 : 1 + shared_nonzero])
            if topology_nonzero:
                codes.append(topology.stages[layer][condition, 1 : 1 + topology_nonzero])
                src.append(topology.stage_sources[layer][condition, 1 : 1 + topology_nonzero])
                rot.append(topology.canonical_rotations[layer][condition, 1 : 1 + topology_nonzero])
            condition_codes.append(np.concatenate(codes))
            condition_sources.append(np.concatenate(src))
            condition_rotations.append(np.concatenate(rot))
        stages.append(np.asarray(condition_codes))
        sources.append(np.asarray(condition_sources))
        rotations.append(np.asarray(condition_rotations))
    return PhaseChain(
        name, shared.bases, tuple(stages), shared.base_sources, tuple(sources),
        tuple(f"layer_{i+1}" for i in range(4)), True, (1, 2, 3, 4),
        tuple("partitioned" for _ in range(4)), tuple(rotations),
    )


def compose_switch(shared: PhaseChain, topology: PhaseChain, switch: int) -> PhaseChain:
    stages = []
    sources = []
    rotations = []
    for layer in range(4):
        if layer < switch:
            stages.append(np.repeat(shared.stages[layer], 3, axis=0))
            sources.append(np.repeat(shared.stage_sources[layer], 3, axis=0))
            rotations.append(np.repeat(shared.canonical_rotations[layer], 3, axis=0))
        else:
            stages.append(topology.stages[layer])
            sources.append(topology.stage_sources[layer])
            rotations.append(topology.canonical_rotations[layer])
    return PhaseChain(
        f"phase_switch_{switch}", shared.bases, tuple(stages), shared.base_sources,
        tuple(sources), tuple(f"layer_{i+1}" for i in range(4)), True,
        (1, 2, 3, 4), tuple("shared" if i < switch else "topology" for i in range(4)),
        tuple(rotations),
    )


def compose_additive(shared: PhaseChain, topology: PhaseChain, width: int) -> PhaseChain:
    stages = []
    sources = []
    rotations = []
    labels = []
    layers = []
    branches = []
    for layer in range(4):
        stages.append(np.repeat(shared.stages[layer][:, :width], 3, axis=0))
        sources.append(np.repeat(shared.stage_sources[layer][:, :width], 3, axis=0))
        rotations.append(np.repeat(shared.canonical_rotations[layer][:, :width], 3, axis=0))
        labels.append(f"layer_{layer+1}_shared")
        layers.append(layer + 1)
        branches.append("shared")
        stages.append(topology.stages[layer][:, :width])
        sources.append(topology.stage_sources[layer][:, :width])
        rotations.append(topology.canonical_rotations[layer][:, :width])
        labels.append(f"layer_{layer+1}_topology")
        layers.append(layer + 1)
        branches.append("topology")
    return PhaseChain(
        f"phase_additive_k{width}", shared.bases, tuple(stages), shared.base_sources,
        tuple(sources), tuple(labels), True, tuple(layers), tuple(branches), tuple(rotations),
    )
