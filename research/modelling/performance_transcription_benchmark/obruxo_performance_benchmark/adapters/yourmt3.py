"""Lossless YourMT3+ event normalization and fixed stock inference settings."""

from __future__ import annotations

import importlib.util
import inspect
import os
import sys
import tempfile
import types
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

from ..artifacts import (
    ArtifactUnavailable,
    ModelSpec,
    verify_checkout,
    verify_checkpoint,
)
from ..types import (
    NormalizedNote,
    TranscriptionOutput,
    common_frame_count,
    rasterize_notes,
)

YOURMT3_IDS = ("ymt3_plus", "yptf_multi", "yptf_moe_multi")


def _module_available(name: str) -> bool:
    if name in sys.modules:
        return True
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _install_inference_compatibility_shims() -> None:
    """Provide only absent training/logging imports required by the stock loader."""
    if not _module_available("pytorch_lightning"):
        import torch

        class LightningModule(torch.nn.Module):
            global_rank = 0

            @property
            def device(self):
                try:
                    return next(self.parameters()).device
                except StopIteration:
                    return torch.device("cpu")

            def save_hyperparameters(self, *args: Any, **kwargs: Any) -> None:
                frame = inspect.currentframe()
                caller = frame.f_back if frame is not None else None
                values = (
                    {}
                    if caller is None
                    else {
                        key: value
                        for key, value in caller.f_locals.items()
                        if key != "self"
                    }
                )
                values.update(kwargs)
                self.hparams = types.SimpleNamespace(**values)

            def log(self, *args: Any, **kwargs: Any) -> None:
                return None

            def log_dict(self, *args: Any, **kwargs: Any) -> None:
                return None

        lightning = types.ModuleType("pytorch_lightning")
        lightning.__path__ = []
        lightning.LightningModule = LightningModule
        lightning.__version__ = "inference-compatibility-shim"
        loggers = types.ModuleType("pytorch_lightning.loggers")
        callbacks = types.ModuleType("pytorch_lightning.callbacks")
        utilities = types.ModuleType("pytorch_lightning.utilities")

        class _Placeholder:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                pass

        lightning.Trainer = _Placeholder

        def rank_zero_only(function=None, *args: Any, **kwargs: Any):
            if function is None:
                return lambda wrapped: wrapped
            return function

        loggers.WandbLogger = _Placeholder
        callbacks.ModelCheckpoint = _Placeholder
        callbacks.LearningRateMonitor = _Placeholder
        utilities.rank_zero_only = rank_zero_only
        sys.modules.update(
            {
                "pytorch_lightning": lightning,
                "pytorch_lightning.loggers": loggers,
                "pytorch_lightning.callbacks": callbacks,
                "pytorch_lightning.utilities": utilities,
            }
        )

    if not _module_available("wandb"):
        wandb = types.ModuleType("wandb")

        class Table:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                self.columns = kwargs.get("columns", [])
                self.data = []

        wandb.Table = Table
        sys.modules["wandb"] = wandb

    if not _module_available("torchmetrics"):
        import torch

        class _ScalarMetric(torch.nn.Module):
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                super().__init__()
                self.reset()

            def update(self, value: Any, *args: Any, **kwargs: Any) -> None:
                self._values.append(torch.as_tensor(value).detach())

            def compute(self):
                if not self._values:
                    return torch.tensor(0.0)
                return torch.stack([value.float() for value in self._values]).mean()

            def reset(self) -> None:
                self._values = []

            def forward(self, value: Any, *args: Any, **kwargs: Any):
                self.update(value, *args, **kwargs)
                return self.compute()

        metrics = types.ModuleType("torchmetrics")
        metrics.MeanMetric = _ScalarMetric
        metrics.SumMetric = _ScalarMetric
        sys.modules["torchmetrics"] = metrics


def _patch_transformers_t5_cache_compatibility() -> None:
    """Bridge the installed Transformers cache-position argument for stock YourMT3 calls."""
    import torch
    from transformers.models.t5.modeling_t5 import T5Attention

    forward = T5Attention.forward
    if "cache_position" not in inspect.signature(forward).parameters:
        return
    if getattr(forward, "_obruxo_cache_compatibility", False):
        return

    def compatible_forward(
        self: Any,
        hidden_states: Any,
        mask: Any = None,
        key_value_states: Any = None,
        position_bias: Any = None,
        past_key_values: Any = None,
        layer_head_mask: Any = None,
        query_length: int | None = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        cache_position: Any = None,
    ):
        """Keep the legacy tuple/cache contract used by the pinned YourMT3 source."""
        batch_size, seq_length = hidden_states.shape[:2]
        is_cross_attention = key_value_states is not None
        query_states = (
            self.q(hidden_states)
            .view(batch_size, -1, self.n_heads, self.key_value_proj_dim)
            .transpose(1, 2)
        )
        current_states = key_value_states if is_cross_attention else hidden_states
        key_states = (
            self.k(current_states)
            .view(batch_size, -1, self.n_heads, self.key_value_proj_dim)
            .transpose(1, 2)
        )
        value_states = (
            self.v(current_states)
            .view(batch_size, -1, self.n_heads, self.key_value_proj_dim)
            .transpose(1, 2)
        )

        past_length = 0
        if past_key_values is not None:
            if len(past_key_values) != 2:
                raise ValueError(
                    "YourMT3 compatibility cache must contain self or cross key/value tensors"
                )
            past_key, past_value = past_key_values
            if is_cross_attention:
                key_states, value_states = past_key, past_value
            else:
                past_length = int(past_key.shape[-2])
                key_states = torch.cat((past_key, key_states), dim=2)
                value_states = torch.cat((past_value, value_states), dim=2)

        scores = torch.matmul(query_states, key_states.transpose(3, 2))
        key_length = key_states.shape[-2]
        if position_bias is None:
            real_seq_length = (
                query_length if query_length is not None else past_length + seq_length
            )
            if cache_position is None:
                cache_position = torch.arange(
                    past_length,
                    past_length + seq_length,
                    dtype=torch.long,
                    device=scores.device,
                )
            if not self.has_relative_attention_bias:
                position_bias = torch.zeros(
                    (1, self.n_heads, seq_length, key_length),
                    device=scores.device,
                    dtype=scores.dtype,
                )
            else:
                position_bias = self.compute_bias(
                    real_seq_length,
                    key_length,
                    device=scores.device,
                    cache_position=cache_position,
                )[:, :, -seq_length:, :]
            if mask is not None:
                position_bias = position_bias + mask[:, :, :, :key_length]

        if self.pruned_heads:
            head_mask = torch.ones(position_bias.shape[1], device=position_bias.device)
            head_mask[list(self.pruned_heads)] = 0
            position_bias_masked = position_bias[:, head_mask.bool()]
        else:
            position_bias_masked = position_bias
        scores += position_bias_masked
        attn_weights = torch.nn.functional.softmax(scores.float(), dim=-1).type_as(
            scores
        )
        attn_weights = torch.nn.functional.dropout(
            attn_weights, p=self.dropout, training=self.training
        )
        if layer_head_mask is not None:
            attn_weights = attn_weights * layer_head_mask
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, -1, self.inner_dim)
        )
        attn_output = self.o(attn_output)
        outputs = (
            attn_output,
            (key_states, value_states) if use_cache else None,
            position_bias,
        )
        if output_attentions:
            outputs = outputs + (attn_weights,)
        return outputs

    compatible_forward._obruxo_cache_compatibility = True
    T5Attention.forward = compatible_forward


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def stock_inference_config(
    model_id: str, overrides: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    if model_id not in YOURMT3_IDS:
        raise ValueError(f"not a YourMT3+ model: {model_id}")
    config = {
        "project": "2024",
        "precision": "float32",
        "deterministic": True,
        "decode": "stock_official_release",
        "checkpoint_selection": "locked_before_corpus_results",
    }
    if overrides:
        changed = {
            key: value for key, value in overrides.items() if config.get(key) != value
        }
        if changed:
            raise ValueError("YourMT3+ stock inference settings cannot be overridden")
    return config


def normalize_note_events(events: Any) -> tuple[NormalizedNote, ...]:
    """Normalize upstream event objects/dicts without adding an alternate decoder."""
    if events is None:
        return ()
    rows = events if isinstance(events, (list, tuple)) else list(events)
    starts: dict[int, dict[str, Any]] = {}
    result: list[NormalizedNote] = []
    for event in rows:
        kind = str(
            _field(event, "event_type", _field(event, "type", event.__class__.__name__))
        ).casefold()
        if "progress" in kind or "tempo" in kind:
            continue
        onset_value = _field(
            event, "onset_seconds", _field(event, "onset", _field(event, "start"))
        )
        offset_value = _field(
            event, "offset_seconds", _field(event, "offset", _field(event, "end"))
        )
        pitch_value = _field(event, "midi_pitch", _field(event, "pitch"))
        if (
            onset_value is not None
            and offset_value is not None
            and pitch_value is not None
        ):
            onset = onset_value
            offset = offset_value
            pitch = _field(event, "midi_pitch", _field(event, "pitch"))
            result.append(
                NormalizedNote(
                    float(onset),
                    float(offset),
                    int(pitch),
                    None
                    if _field(event, "velocity") is None
                    else int(_field(event, "velocity")),
                    None
                    if _field(event, "confidence") is None
                    else float(_field(event, "confidence")),
                    _field(event, "instrument_or_program", _field(event, "instrument")),
                )
            )
            continue
        if "start" in kind or _field(event, "start_time") is not None:
            index = _field(event, "index", len(starts))
            starts[int(index)] = {
                "onset": _field(event, "start_time", _field(event, "onset")),
                "pitch": _field(event, "pitch", _field(event, "midi_pitch")),
                "instrument": _field(event, "instrument", _field(event, "program")),
                "velocity": _field(event, "velocity"),
            }
            continue
        if "end" in kind or _field(event, "end_time") is not None:
            start_event = _field(event, "start_event")
            index = _field(
                event,
                "start_event_index",
                _field(start_event, "index") if start_event is not None else None,
            )
            if index is None or int(index) not in starts:
                raise ValueError("YourMT3+ end event does not reference a start event")
            start = starts[int(index)]
            result.append(
                NormalizedNote(
                    float(start["onset"]),
                    float(_field(event, "end_time")),
                    int(start["pitch"]),
                    None if start["velocity"] is None else int(start["velocity"]),
                    None,
                    start["instrument"],
                )
            )
    return tuple(result)


def write_temporary_midi(
    data: bytes, output_dir: Path, name: str = "upstream.mid"
) -> Path:
    root = Path(output_dir).resolve(strict=False)
    root.mkdir(parents=True, exist_ok=True)
    destination = (root / name).resolve(strict=False)
    if destination.parent != root or destination.suffix.casefold() != ".mid":
        raise ValueError(
            "temporary MIDI must remain directly under the provided output directory"
        )
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.stem}.", suffix=".tmp", dir=root
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        return destination
    finally:
        temporary.unlink(missing_ok=True)


class YourMT3Adapter:
    def __init__(
        self,
        spec: ModelSpec,
        source_root: Path | None,
        checkpoint: Path | None,
        *,
        segment_batch_size: int | None = None,
    ) -> None:
        self.spec = spec
        self.source_root = None if source_root is None else Path(source_root)
        self.checkpoint = None if checkpoint is None else Path(checkpoint)
        self.inference = stock_inference_config(spec.model_id)
        self.segment_batch_size = (
            8 if segment_batch_size is None else int(segment_batch_size)
        )
        if self.segment_batch_size <= 0:
            raise ValueError("YourMT3 segment batch size must be positive")
        self.model: Any | None = None
        self.bound_model: Any | None = None
        self._helper: Any | None = None

    @property
    def inference_configuration(self) -> dict[str, int]:
        return {"segment_batch_size": self.segment_batch_size}

    def preflight(self) -> None:
        if self.source_root is None or self.checkpoint is None:
            raise ArtifactUnavailable("dependency_unavailable")
        verify_checkout(self.spec, self.source_root)
        verify_checkpoint(self.spec, self.checkpoint)

    @property
    def active_model(self) -> Any | None:
        return self.bound_model if self.bound_model is not None else self.model

    def _official_args(self) -> list[str]:
        checkpoint_path = Path(self.spec.checkpoint_path)
        checkpoint_name = checkpoint_path.name
        experiment = checkpoint_path.parent.parent.name
        args = [
            f"{experiment}@{checkpoint_name}",
            "-p",
            "2024",
            "-pr",
            "32",
            "-w",
            "false",
        ]
        if self.spec.model_id == "yptf_multi":
            args += [
                "-tk",
                "mc13_full_plus_256",
                "-dec",
                "multi-t5",
                "-nl",
                "26",
                "-enc",
                "perceiver-tf",
                "-ac",
                "spec",
                "-hop",
                "300",
                "-atc",
                "1",
            ]
        elif self.spec.model_id == "yptf_moe_multi":
            args += [
                "-tk",
                "mc13_full_plus_256",
                "-dec",
                "multi-t5",
                "-nl",
                "26",
                "-enc",
                "perceiver-tf",
                "-sqr",
                "1",
                "-ff",
                "moe",
                "-wf",
                "4",
                "-nmoe",
                "8",
                "-kmoe",
                "2",
                "-act",
                "silu",
                "-epe",
                "rope",
                "-rp",
                "1",
                "-ac",
                "spec",
                "-hop",
                "300",
                "-atc",
                "1",
            ]
        return args

    def load(self, device: str = "cpu") -> None:
        self.preflight()
        if device not in {"cpu", "xpu"}:
            raise ArtifactUnavailable(
                f"unsupported YourMT3+ device requested: {device}"
            )
        try:
            import torch

            if device == "xpu" and (
                not hasattr(torch, "xpu") or not torch.xpu.is_available()
            ):
                raise ArtifactUnavailable("xpu_unavailable")

            root = self.source_root.resolve(strict=True)
            amt_src = root / "amt" / "src"
            if not amt_src.is_dir():
                raise ArtifactUnavailable(
                    "official YourMT3+ amt/src checkout is missing"
                )
            for path in (root, amt_src):
                if str(path) not in sys.path:
                    sys.path.insert(0, str(path))
            _install_inference_compatibility_shims()
            import model_helper

            _patch_transformers_t5_cache_compatibility()
        except (ImportError, OSError) as exc:
            raise ArtifactUnavailable("dependency_unavailable") from exc
        if not Path(self.checkpoint).resolve().is_relative_to(root):
            raise ArtifactUnavailable(
                "checkpoint must be inside the pinned official checkout for stock loading"
            )
        args = self._official_args()
        loader = model_helper.load_model_checkpoint

        from config.config import shared_cfg as default_shared_cfg

        checkpoint = Path(self.checkpoint).resolve(strict=True)

        def initialize_inference_loader(loader_args: Any, stage: str = "test"):
            del loader_args, stage
            return (
                None,
                None,
                {
                    "lightning_dir": str(checkpoint.parent.parent),
                    "last_ckpt_path": str(checkpoint),
                },
                deepcopy(default_shared_cfg),
            )

        model_helper.initialize_trainer = initialize_inference_loader
        try:
            target_device = torch.device(device)
            if "device" in inspect.signature(loader).parameters:
                model = loader(args=args, device=target_device)
            else:
                model = loader(args=args).to(target_device)
        except (ImportError, ModuleNotFoundError) as exc:
            raise ArtifactUnavailable("dependency_unavailable") from exc
        self.model = model
        self.bound_model = model
        self._helper = model_helper

    def bind_model(self, model: Any) -> None:
        self.bound_model = model
        if hasattr(model, "eval"):
            model.eval()

    @staticmethod
    def _frame_count(audio: Path) -> int:
        import sys

        root = Path(__file__).resolve().parents[3] / "basic_pitch"
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from obruxo_basic_pitch.inference import prepare_wav

        return common_frame_count(prepare_wav(Path(audio)).original_sample_count)

    @staticmethod
    def _read_wav(path: Path) -> tuple[Any, int]:
        """Read WAV PCM using the landed workspace convention for this runtime."""
        import torch
        from scipy.io import wavfile

        sample_rate, values = wavfile.read(Path(path).resolve(strict=True))
        decoded = np.asarray(values)
        waveform = decoded.astype(np.float32)
        if decoded.ndim == 2:
            waveform = waveform.mean(axis=1)
        if np.issubdtype(decoded.dtype, np.integer):
            waveform /= np.iinfo(decoded.dtype).max
        if waveform.ndim != 1 or not np.isfinite(waveform).all():
            raise ValueError("audio must be finite mono samples")
        return torch.from_numpy(np.ascontiguousarray(waveform)).unsqueeze(0), int(
            sample_rate
        )

    def transcribe(self, audio: Path) -> TranscriptionOutput:
        if self.active_model is None or self._helper is None:
            self.load()
        import torch
        import torchaudio

        model = self.active_model
        helper = self._helper
        waveform, sample_rate = self._read_wav(Path(audio))
        target_rate = int(model.audio_cfg["sample_rate"])
        waveform = torchaudio.functional.resample(waveform, sample_rate, target_rate)
        segments = helper.slice_padded_array(
            waveform, model.audio_cfg["input_frames"], model.audio_cfg["input_frames"]
        )
        device = next(model.parameters()).device
        segments = torch.from_numpy(segments.astype("float32")).to(device).unsqueeze(1)
        with torch.inference_mode():
            predicted, _ = model.inference_file(
                bsz=self.segment_batch_size, audio_segments=segments
            )
        channels = int(model.task_manager.num_decoding_channels)
        starts = [
            model.audio_cfg["input_frames"] * i / target_rate
            for i in range(segments.shape[0])
        ]
        notes_by_channel = []
        for channel in range(channels):
            channel_tokens = [array[:, channel, :] for array in predicted]
            zipped, _, _ = model.task_manager.detokenize_list_batches(
                channel_tokens, starts, return_events=True
            )
            notes, _ = helper.merge_zipped_note_events_and_ties_to_notes(zipped)
            notes_by_channel.append(notes)
        notes = normalize_note_events(helper.mix_notes(notes_by_channel))
        return TranscriptionOutput(
            notes, rasterize_notes(notes, self._frame_count(Path(audio)))
        )
