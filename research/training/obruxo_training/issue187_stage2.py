# SPDX-FileCopyrightText: 2026 SpanSynth contributors (checkpoint-compatible MIDI encoder)
# SPDX-License-Identifier: Apache-2.0

"""Frozen SpanSynth MIDI features and matched Issue 187 Stage 2 adapters."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

import torch
from torch import Tensor, nn
from torch.nn import functional as F

SAMPLE_RATE = 48_000
MIDI_HZ = 25
HOP_LENGTH = 1_920
MIDI_FRAMES = 512
EVENT_CAPACITY = 128
MIDI_WIDTH = 768
WIDTH = 64
MIDI_ENCODER_REVISION = "9a3eff857e821c91f21a05bda8048b7541f4d9e4"
MIDI_SOURCE_REVISION = "f593bf21e48cc4aa836002a3834cfae42e8d4254"
PREPROCESSING_VERSION = "spansynth-v8-fine40-sample-rows-v1"

# SpanSynth-Edit's Fine40 GM program vocabulary (Apache-2.0; see NOTICE).
_PROGRAM_GROUPS = (
    (0, 1, 3, 6, 7), (2, 4, 5), tuple(range(8, 16)), tuple(range(16, 24)),
    (24, 25), (26, 27, 28), (29, 30, 31), (32, 35), (33, 34, 36, 37, 38, 39),
    (40,), (41,), (42,), (43,), (46,), (47,), (48, 49, 44, 45), (50, 51), (52, 53, 54),
    (55,), (56, 59), (57,), (58,), (60,), (61, 62, 63), (64, 65), (66,), (67,),
    (68,), (69,), (70,), (71,), (73, 72, 74, 75, 76, 77, 78, 79), tuple(range(80, 88)),
    tuple(range(88, 96)), (100,), (101,), (128,), (104,), (105,), (110,),
)
_PROGRAM_CATEGORY = {program: group for group, values in enumerate(_PROGRAM_GROUPS) for program in values}


@dataclass(frozen=True)
class EventSetConfig:
    """The released V8 event encoder geometry, kept key-compatible with upstream."""

    capacity: int = 128
    kind_embedding_dim: int = 16
    pitch_embedding_dim: int = 32
    numeric_dim: int = 7
    row_hidden_dim: int = 256
    row_dim: int = 128
    learned_pool_tokens: int = 5
    output_dim: int = 768
    row_chunk_size: int = 16
    norm_eps: float = 1e-6

    @property
    def row_input_dim(self) -> int:
        return self.kind_embedding_dim + self.pitch_embedding_dim + self.numeric_dim


class EventSetEncoder(nn.Module):
    """Checkpoint-compatible copy of SpanSynth V8's frozen MIDI event encoder."""

    def __init__(self, config: EventSetConfig | None = None):
        super().__init__()
        self.config = config or EventSetConfig()
        config = self.config
        self.kind_embedding = nn.Embedding(4, config.kind_embedding_dim, padding_idx=0)
        self.pitch_embedding = nn.Embedding(129, config.pitch_embedding_dim, padding_idx=0)
        self.row_mlp = nn.Sequential(
            nn.Linear(config.row_input_dim, config.row_hidden_dim), nn.SiLU(),
            nn.Linear(config.row_hidden_dim, config.row_dim), nn.RMSNorm(config.row_dim, eps=config.norm_eps),
        )
        self.pool_queries = nn.Parameter(torch.empty(config.learned_pool_tokens, config.row_dim))
        nn.init.normal_(self.pool_queries, mean=0.0, std=0.02)
        self.pool_norm = nn.RMSNorm(config.row_dim, eps=config.norm_eps)
        self.count_projection = nn.Linear(5, config.output_dim, bias=False)
        self.row_chunk_size = config.row_chunk_size

    def _counts(self, kind_id: Tensor, numeric: Tensor, event_valid: Tensor, dtype: torch.dtype) -> Tensor:
        drum = event_valid & (numeric[..., 0] > 0.0)
        melodic = event_valid & ~drum
        values = torch.stack((melodic.sum(2), drum.sum(2), (event_valid & (kind_id == 1)).sum(2),
                              (event_valid & (kind_id == 2)).sum(2), (event_valid & (kind_id == 3)).sum(2)), -1)
        return torch.log1p(values.to(dtype=dtype)) / math.log1p(self.config.capacity)

    def _encode_chunk(self, kind_id: Tensor, pitch_id: Tensor, numeric: Tensor,
                      event_valid: Tensor) -> tuple[Tensor, Tensor]:
        batch, frames, _ = kind_id.shape
        positions = event_valid.nonzero(as_tuple=False)
        frame_index = positions[:, 0] * frames + positions[:, 1]
        kind = self.kind_embedding(kind_id[event_valid].to(torch.int64))
        pitch = self.pitch_embedding(pitch_id[event_valid].to(torch.int64))
        encoded = self.row_mlp(torch.cat((kind, pitch, numeric[event_valid].to(kind.dtype)), -1))
        all_sum = encoded.new_zeros(batch * frames, self.config.row_dim)
        gated_sums = encoded.new_zeros(batch * frames, self.config.learned_pool_tokens, self.config.row_dim)
        if frame_index.numel():
            all_sum = all_sum.index_add(0, frame_index, encoded)
            gates = (encoded @ self.pool_queries.transpose(0, 1) * (1.0 / math.sqrt(self.config.row_dim))).sigmoid()
            gated_sums = gated_sums.index_add(0, frame_index, gates[..., None] * encoded[:, None, :])
        return (all_sum.reshape(batch, frames, self.config.row_dim),
                gated_sums.reshape(batch, frames, self.config.learned_pool_tokens, self.config.row_dim))

    def forward(self, kind_id: Tensor, pitch_id: Tensor, numeric: Tensor,
                event_valid: Tensor, *, category_id: Tensor | None = None) -> Tensor:
        del category_id  # present in upstream signature; categories are already numeric features
        if kind_id.ndim != 3 or kind_id.shape[-1] != self.config.capacity:
            raise ValueError("kind_id must have shape [batch, frames, 128]")
        if pitch_id.shape != kind_id.shape or event_valid.shape != kind_id.shape:
            raise ValueError("pitch_id and event_valid must match kind_id")
        if numeric.shape != (*kind_id.shape, 7) or event_valid.dtype is not torch.bool:
            raise ValueError("numeric must be [batch, frames, 128, 7] and event_valid boolean")
        if not (kind_id.device == pitch_id.device == numeric.device == event_valid.device):
            raise ValueError("all MIDI event tensors must share a device")
        all_sum = gated_sums = None
        for start in range(0, self.config.capacity, self.config.row_chunk_size):
            stop = min(start + self.config.row_chunk_size, self.config.capacity)
            current_all, current_gated = self._encode_chunk(kind_id[:, :, start:stop], pitch_id[:, :, start:stop],
                                                            numeric[:, :, start:stop], event_valid[:, :, start:stop])
            all_sum = current_all if all_sum is None else all_sum + current_all
            gated_sums = current_gated if gated_sums is None else gated_sums + current_gated
        pooled = torch.cat((all_sum[:, :, None, :], gated_sums), dim=2)
        nonempty = event_valid.any(dim=2)
        pooled = self.pool_norm(pooled).masked_fill(~nonempty[:, :, None, None], 0.)
        condition = pooled.reshape(*kind_id.shape[:2], self.config.output_dim)
        condition = condition + self.count_projection(self._counts(kind_id, numeric, event_valid, condition.dtype))
        return condition.masked_fill(~nonempty[..., None], 0.)


def extract_midi_encoder(full_checkpoint: Path, output_checkpoint: Path) -> dict[str, int | str]:
    """Copy only midi_encoder.* tensors from a pinned SpanSynth safetensors file."""
    from safetensors import safe_open
    from safetensors.torch import save_file

    output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    tensors = {}
    with safe_open(str(full_checkpoint), framework="pt", device="cpu") as source:
        for key in source.keys():
            if key.startswith("midi_encoder."):
                tensors[key.removeprefix("midi_encoder.")] = source.get_tensor(key).contiguous()
    expected = set(EventSetEncoder().state_dict())
    if set(tensors) != expected:
        missing, extra = sorted(expected - set(tensors)), sorted(set(tensors) - expected)
        raise ValueError(f"SpanSynth MIDI encoder state mismatch (missing={missing}, extra={extra})")
    save_file(tensors, str(output_checkpoint), metadata={"source": "mimbres/spansynth-edit",
                                                          "revision": MIDI_ENCODER_REVISION,
                                                          "component": "midi_encoder"})
    return {"revision": MIDI_ENCODER_REVISION, "tensor_count": len(tensors),
            "parameter_count": sum(t.numel() for t in tensors.values())}


def load_midi_encoder(checkpoint: Path, *, device: str | torch.device = "cpu") -> EventSetEncoder:
    from safetensors.torch import load_file

    encoder = EventSetEncoder()
    state = load_file(str(checkpoint), device="cpu")
    encoder.load_state_dict(state, strict=True)
    return encoder.to(device=device).eval().requires_grad_(False)


def encode_midi_spans(notes: list[dict[str, int]], *, duration_seconds: float) -> dict[str, Tensor]:
    """Encode absolute sample note spans into the upstream V8 25 Hz K128 rows."""
    if not math.isfinite(duration_seconds) or not 0 < duration_seconds <= MIDI_FRAMES / MIDI_HZ:
        raise ValueError("duration must lie in (0, 20.48] seconds")
    valid_count = min(MIDI_FRAMES, math.ceil(duration_seconds * MIDI_HZ))
    valid = torch.arange(MIDI_FRAMES) < valid_count
    frames: list[list[tuple[int, int, int, int, int, int, int, int]]] = [[] for _ in range(MIDI_FRAMES)]
    horizon = MIDI_FRAMES * HOP_LENGTH
    for note in notes:
        onset, offset = note["onset"], note["offset"]
        pitch, velocity, program, physical = (note[k] for k in ("pitch", "velocity", "program", "physical_id"))
        if not (0 <= onset <= offset and 0 <= pitch <= 127 and 1 <= velocity <= 127 and
                program in _PROGRAM_CATEGORY and physical >= 0):
            raise ValueError("MIDI span has unsupported timing, pitch, velocity, program or identity")
        category = _PROGRAM_CATEGORY[program]
        if program == 128:
            if onset < horizon and valid[onset // HOP_LENGTH]:
                f = onset // HOP_LENGTH
                frames[f].append((1, pitch, velocity, program, category, onset - f * HOP_LENGTH, 0, physical))
            continue
        if offset <= onset:
            raise ValueError("melodic notes must have positive duration")
        for frame in range(max(0, onset // HOP_LENGTH), min(MIDI_FRAMES, math.ceil(offset / HOP_LENGTH))):
            if not valid[frame]:
                continue
            start, stop = frame * HOP_LENGTH, (frame + 1) * HOP_LENGTH
            if not (start < offset and onset < stop):
                continue
            if start <= onset < stop:
                kind, boundary, remaining = 1, onset - start, offset - onset
            elif start < offset <= stop:
                kind, boundary, remaining = 3, offset - start, offset - start
            else:
                kind, boundary, remaining = 2, 0, offset - start
            frames[frame].append((kind, pitch, velocity, program, category, boundary, remaining, physical))
    if max(map(len, frames), default=0) > EVENT_CAPACITY:
        raise ValueError("SpanSynth V8 MIDI frame exceeds the 128-event checkpoint capacity")
    kind_id = torch.zeros(1, MIDI_FRAMES, EVENT_CAPACITY, dtype=torch.int64)
    pitch_id = torch.zeros_like(kind_id)
    numeric = torch.zeros(1, MIDI_FRAMES, EVENT_CAPACITY, 7, dtype=torch.float32)
    event_valid = torch.zeros_like(kind_id, dtype=torch.bool)
    for frame, rows in enumerate(frames):
        rows.sort(key=lambda row: (row[3] == 128, row[4], row[1], row[0], row[7]))
        for slot, (kind, pitch, velocity, program, category, boundary, remaining, _) in enumerate(rows):
            kind_id[0, frame, slot] = kind
            pitch_id[0, frame, slot] = pitch + 1
            event_valid[0, frame, slot] = True
            angle = 2.0 * math.pi * category / len(_PROGRAM_GROUPS)
            numeric[0, frame, slot] = torch.tensor((1.0 if program == 128 else -1.0, math.sin(angle),
                                                    math.cos(angle), 2.0 * pitch / 127.0 - 1.0,
                                                    2.0 * velocity / 127.0 - 1.0,
                                                    0.0 if kind == 2 else 2.0 * boundary / HOP_LENGTH - 1.0,
                                                    2.0 * math.log1p(min(remaining, horizon) / SAMPLE_RATE) /
                                                    math.log1p(horizon / SAMPLE_RATE) - 1.0))
    return {"kind_id": kind_id, "pitch_id": pitch_id, "numeric": numeric, "event_valid": event_valid,
            "frame_valid": valid[None]}


def performance_to_spans(performance, *, default_program: int, sample_rate: int = SAMPLE_RATE) -> list[dict[str, int]]:
    """Convert OBRUXO's tick-domain Performance without losing polyphony or note lengths."""
    from obruxo_data.midi.timing import TempoMap

    if default_program not in _PROGRAM_CATEGORY or default_program == 128:
        raise ValueError("default_program must be a supported melodic Fine40 GM program")
    tempo_map = TempoMap.from_performance(performance)
    notes = []
    for physical_id, note in enumerate(performance.note_spans()):
        program = 128 if note.channel == 9 else default_program
        onset = math.floor(float(tempo_map.tick_to_seconds(note.start_tick)) * sample_rate + 0.5)
        offset = math.floor(float(tempo_map.tick_to_seconds(note.end_tick)) * sample_rate + 0.5)
        notes.append({"onset": onset, "offset": offset, "pitch": note.pitch, "velocity": note.velocity,
                      "program": program, "physical_id": physical_id})
    return notes


def midi_cache_key(notes: list[dict[str, int]], duration_seconds: float, revision: str = MIDI_ENCODER_REVISION) -> str:
    payload = {"version": PREPROCESSING_VERSION, "revision": revision,
               "duration_seconds": float(duration_seconds), "notes": notes}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class MidiEmbeddingCache:
    """Atomic cache of detached, frozen 25 Hz MIDI encoder output."""

    def __init__(self, directory: Path, revision: str = MIDI_ENCODER_REVISION):
        self.directory, self.revision = Path(directory), revision

    def get_or_encode(self, notes: list[dict[str, int]], duration_seconds: float,
                      encoder: EventSetEncoder, *, device: str | torch.device) -> Tensor:
        key = midi_cache_key(notes, duration_seconds, self.revision)
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / f"{key}.pt"
        if path.is_file():
            cached = torch.load(path, map_location="cpu", weights_only=True)
            if cached["key"] != key:
                raise ValueError("MIDI cache key mismatch")
            return cached["embedding"].to(device)
        rows = encode_midi_spans(notes, duration_seconds=duration_seconds)
        with torch.inference_mode():
            embedding = encoder(rows["kind_id"].to(device), rows["pitch_id"].to(device),
                                rows["numeric"].to(device), rows["event_valid"].to(device)).float()[0]
            embedding = embedding[:int(rows["frame_valid"].sum().item())]
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=self.directory, suffix=".tmp", delete=False) as stream:
                temporary = Path(stream.name)
            torch.save({"key": key, "embedding": embedding.cpu()}, temporary)
            os.replace(temporary, path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        return embedding


def align_midi_tokens(tokens: Tensor, target_frames: int, target_hz: float, *, source_hz: float = MIDI_HZ) -> Tensor:
    """Linearly sample 25 Hz frame-start features at a decoder/generator time grid."""
    if tokens.ndim != 3 or target_frames < 1 or source_hz <= 0 or target_hz <= 0:
        raise ValueError("tokens must be [batch, frames, channels] and rates/frame count positive")
    locations = torch.arange(target_frames, device=tokens.device, dtype=torch.float32) * (source_hz / target_hz)
    left = locations.floor().long().clamp_max(tokens.shape[1] - 1)
    right = (left + 1).clamp_max(tokens.shape[1] - 1)
    fraction = (locations - left).to(tokens.dtype)[None, :, None]
    return torch.lerp(tokens[:, left], tokens[:, right], fraction)


def align_mask(mask: Tensor, target_frames: int, target_hz: float, *, source_hz: float = MIDI_HZ) -> Tensor:
    locations = torch.arange(target_frames, device=mask.device, dtype=torch.float32) * (source_hz / target_hz)
    index = locations.floor().long().clamp_max(mask.shape[1] - 1)
    return mask[:, index]


class RMSNorm(nn.Module):
    def __init__(self, width: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, value: Tensor) -> Tensor:
        return F.rms_norm(value.float(), (value.shape[-1],), self.weight.float(), self.eps).to(value.dtype)


class ResidualMLPBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(WIDTH)
        self.up = nn.Linear(WIDTH, 320)
        self.down = nn.Linear(320, WIDTH)

    def forward(self, value: Tensor) -> Tensor:
        return value + self.down(F.gelu(self.up(self.norm(value))))


class RotarySelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.heads, self.head_dim = 4, WIDTH // 4
        self.q_proj, self.k_proj, self.v_proj = (nn.Linear(WIDTH, WIDTH) for _ in range(3))
        self.out_proj = nn.Linear(WIDTH, WIDTH)
        self.q_norm, self.k_norm = RMSNorm(self.head_dim), RMSNorm(self.head_dim)

    def forward(self, value: Tensor) -> Tensor:
        batch, frames, _ = value.shape
        def split(layer):
            return layer(value).view(batch, frames, self.heads, self.head_dim)
        q, k, v = split(self.q_proj), split(self.k_proj), split(self.v_proj)
        positions = torch.arange(frames, device=value.device, dtype=torch.float32)
        inverse = torch.exp(-math.log(10_000.0) * torch.arange(0, self.head_dim, 2, device=value.device) / self.head_dim)
        angles = positions[:, None] * inverse[None, :]
        cosine, sine = angles.cos()[None, :, None, :], angles.sin()[None, :, None, :]
        def rotate(tensor):
            even, odd = tensor.float()[..., 0::2], tensor.float()[..., 1::2]
            return torch.stack((even * cosine - odd * sine, even * sine + odd * cosine), -1).flatten(-2).to(tensor.dtype)
        q, k = self.q_norm(rotate(q)), self.k_norm(rotate(k))
        attended = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
        return self.out_proj(attended.transpose(1, 2).reshape(batch, frames, WIDTH))


class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn_norm, self.ff_norm = RMSNorm(WIDTH), RMSNorm(WIDTH)
        self.attention = RotarySelfAttention()
        self.gate, self.up, self.down = nn.Linear(WIDTH, 128), nn.Linear(WIDTH, 128), nn.Linear(128, WIDTH)

    def forward(self, value: Tensor) -> Tensor:
        value = value + self.attention(self.attn_norm(value))
        normalized = self.ff_norm(value)
        hidden = F.silu(self.gate(normalized)) * self.up(normalized)
        return value + self.down(hidden)


class DirectLatentConnector(nn.Module):
    def __init__(self, width: int, latent_channels: int):
        super().__init__()
        self.output = nn.Linear(width, latent_channels)

    def forward(self, hidden: Tensor) -> Tensor:
        return self.output(hidden).transpose(1, 2)


class GeneratorConditioningConnector(nn.Module):
    def __init__(self, width: int, conditioning_width: int):
        super().__init__()
        self.output = nn.Linear(width, conditioning_width)

    def forward(self, hidden: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        if mask.shape != hidden.shape[:2]:
            raise ValueError("generator conditioning mask must match [batch, frames]")
        return self.output(hidden), mask


class Issue187Adapter(nn.Module):
    """Shared patch/MIDI/time fusion with parameter-matched two-block backbones."""

    def __init__(self, backbone: str, control_width: int, output_width: int, *, generator: bool = False):
        super().__init__()
        if backbone not in {"residual_mlp", "transformer"}:
            raise ValueError(f"unsupported Issue 187 backbone: {backbone}")
        self.backbone_name = backbone
        self.patch = nn.Linear(control_width, WIDTH)
        self.midi = nn.Linear(MIDI_WIDTH, WIDTH)
        self.timing = nn.Linear(3, WIDTH)
        self.fusion_norm = nn.LayerNorm(WIDTH)
        block = ResidualMLPBlock if backbone == "residual_mlp" else TransformerBlock
        self.blocks = nn.ModuleList((block(), block()))
        self.final_norm = nn.LayerNorm(WIDTH)
        self.connector = (GeneratorConditioningConnector(WIDTH, output_width) if generator
                          else DirectLatentConnector(WIDTH, output_width))

    def forward(self, controls: Tensor, midi_tokens: Tensor, midi_mask: Tensor, *,
                target_frames: int, target_hz: float) -> Tensor | tuple[Tensor, Tensor]:
        if controls.ndim != 2 or midi_tokens.ndim != 3 or len(controls) != len(midi_tokens):
            raise ValueError("controls must be [batch, controls], MIDI tokens [batch, frames, 768]")
        aligned = align_midi_tokens(midi_tokens, target_frames, target_hz)
        mask = align_mask(midi_mask, target_frames, target_hz)
        times = torch.arange(target_frames, device=controls.device, dtype=aligned.dtype) / target_hz
        time_features = torch.stack((times, torch.sin(2 * math.pi * times), torch.cos(2 * math.pi * times)), -1)
        hidden = self.fusion_norm(self.patch(controls)[:, None] + self.midi(aligned) + self.timing(time_features)[None])
        for block in self.blocks:
            hidden = block(hidden)
        hidden = self.final_norm(hidden)
        if isinstance(self.connector, GeneratorConditioningConnector):
            return self.connector(hidden, mask)
        return self.connector(hidden)


def trainable_parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)
