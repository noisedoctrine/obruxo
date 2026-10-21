"""Train-only recipe selection and bounded native/feature persistence for the local helper."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import platform
from time import time

import numpy as np
import torch

from obruxo_data.hashing import canonical_sha256, sha256_bytes
from obruxo_data.midi import Performance
from obruxo_data.render.base import RenderRequest

from .audit import write_json
from .codec import PilotCodec, PilotTarget
from .performances import CAPABILITIES, initial_bank
from .pilot import seeded_random
from .proxy import performance_inputs
from .reproducibility import fixed_generator_phases
from .spectral import SpectralTargets, target_bytes
from .spectral_controls import BasicControlAdapter
from .stream import NativeRendererFactory, RenderWorker
from .temporal_proxy import sample_frames, temporal_proxy_spec
from .trainer import atomic_torch_save


@dataclass(frozen=True)
class TemporalProxyConfig:
    training_states: int = 128
    views: int = 8
    validation_states: int = 12
    steps: int = 12000
    batch_size: int = 8
    learning_rate: float = .001
    seed: int = 0
    checkpoint_every: int = 250
    paired_changes: bool = False
    event_frames: bool = False

    def __post_init__(self):
        counts = (self.training_states, self.views, self.validation_states, self.steps, self.batch_size, self.checkpoint_every)
        if any(type(x) is not int or x < 1 for x in counts) or not 6 <= self.training_states <= 256:
            raise ValueError("invalid temporal helper counts")
        if (self.views > 8 or self.training_states * self.views > 2048 or self.validation_states * 20 > 256 or self.steps > 12000
                or type(self.seed) is not int or not 0 < self.learning_rate <= .01):
            raise ValueError("temporal helper exceeds its fixed experiment limits")
        if type(self.paired_changes) is not bool or self.paired_changes and (self.views != 8 or self.batch_size % 2):
            raise ValueError("paired changes require eight views and an even recording batch")
        if type(self.event_frames) is not bool or self.event_frames and not self.paired_changes:
            raise ValueError("event frames require paired-change training")


def category_balanced_entries(pool, split, count):
    if type(pool.codec) is not PilotCodec or split not in {"train", "validation"}:
        raise ValueError("basic helper requires the exact pilot and keeps test states reserved")
    groups = {}
    for entry in pool.entries:
        if pool.manifest.by_state[entry.state_id].split == split:
            groups.setdefault(entry.target.categorical, []).append(entry)
    selected = [group[i] for i in range(max(map(len, groups.values()), default=0))
                for _, group in sorted(groups.items()) if i < len(group)]
    if len(selected) < count:
        raise ValueError("not enough independent helper states")
    return selected[:count]


def training_performance(seed, index, view):
    rng = seeded_random("basic_temporal_performance_v1", seed, index, view)
    performance = Performance(bpm=rng.choice((90, 110, 130, 150)))
    velocity, base = rng.choice((64, 80, 96, 112)), rng.randrange(36, 78)
    mode = view % 4
    if mode == 0:
        notes = [(base, 0, rng.choice((720, 960, 1200, 1440)))]
    elif mode == 1:
        notes = [(base, i * 360, rng.choice((180, 240, 300))) for i in range(4)]
    elif mode == 2:
        notes = [(base + offset, i * 360, rng.choice((300, 420))) for i, offset in enumerate((0, 4, 7, 2))]
    else:
        notes = [(base + offset, 0, rng.choice((960, 1200))) for offset in (0, 4, 7)]
    for pitch, start, duration in notes:
        performance.add_note(pitch=pitch, velocity=velocity, start_tick=start, duration_ticks=duration)
    performance.end_tick = max(960, performance.end_tick + 120)
    performance.validate(CAPABILITIES).require_valid()
    return performance


def make_plan(pool, config):
    training = category_balanced_entries(pool, "train", config.training_states)
    validation = category_balanced_entries(pool, "validation", config.validation_states)
    reserved = {entry.performance_id for entry in initial_bank()}
    recipes = []
    for index, entry in enumerate(training):
        for view in range(config.views):
            performance = training_performance(config.seed, index, view % 4)
            identity = canonical_sha256(performance.to_dict())
            if identity in reserved:
                raise ValueError("generated training performance matches a reserved recipe")
            values, choices = list(entry.target.continuous), list(entry.target.categorical)
            if view in (4, 5):
                if config.paired_changes:
                    active = pool.codec.continuous_mask(entry.target)
                    position = next((index * 2 + view - 4 + offset) % 9 for offset in range(9)
                                    if active[(index * 2 + view - 4 + offset) % 9])
                    values[position] = min(1., max(0., values[position] + (.05 if view == 4 else -.05)))
                else:
                    values = [min(1., max(0., value + (.05 if view == 4 else -.05))) for value in values]
            if view in (6, 7):
                category = view - 6
                choices[category] = (choices[category] + 1) % (3 if category == 0 else 2)
            target = pool.codec.canonical_target(PilotTarget(tuple(values), tuple(choices)))
            recipes.append({"sequence": len(recipes), "source_state_id": entry.state_id, "view": view,
                            "target": target.to_dict(), "performance": performance.to_dict(), "performance_id": identity})
            if config.paired_changes:
                recipes[-1]["pair_id"] = index * 4 + view % 4
    paths = [Path(__file__).parent / name for name in ("temporal_proxy.py", "temporal_proxy_data.py", "temporal_proxy_screen.py",
                                                     "spectral.py", "spectral_controls.py")]
    if config.paired_changes:
        paths.extend((Path(__file__).with_name("paired_spectral.py"), Path(__file__).parents[2] / "data_generation/obruxo_data/render/vita.py"))
    if config.event_frames:
        from .event_frames import note_event_seconds
        paths.append(Path(__file__).with_name("event_frames.py"))
        for recipe in recipes:
            note_event_seconds(Performance.from_dict(recipe["performance"]))
    plan = {"format": "local_basic_temporal_proxy_screen_v1", "pool_id": pool.pool_id, "manifest_id": pool.manifest.manifest_id,
            "codec": pool.codec.to_dict(), "contract": BasicControlAdapter(pool.codec).contract(), "config": asdict(config),
            "model": temporal_proxy_spec(config.event_frames), "training_recipes": recipes, "validation_states": [entry.state_id for entry in validation],
            "control_coverage": "four_original_performances_plus_paired_continuous_plus_minus_unison_and_filter_changes",
            "validation_performances": [{"reference": initial_bank()[i % 4].to_dict(), "transfer": initial_bank()[4 + i % 4].to_dict()}
                                        for i in range(config.validation_states)],
            "source": {path.name: sha256_bytes(path.read_bytes().replace(b"\r\n", b"\n")) for path in paths},
            "runtime": {"python": platform.python_version(), "torch": str(torch.__version__), "numpy": np.__version__,
                        "device": "cpu", "cpu_threads": 2, "precision": "float32"},
            "limits": {"training_attempts": 2048, "qualification_attempts": 256, "wall_seconds": 14400,
                       "process_tree_bytes": 8 * 1024**3, "archive_bytes": 2 * 1024**3, "replay_bytes": 512 * 1024**2,
                       "replay_records": 512},
            "qualification": {"reference_vs_best_baseline_ratio": .9, "transfer_vs_best_baseline_ratio": 1.,
                              "continuous_rank_agreement": .7, "category_rank_agreement": .7, "pan_and_active_detune_agreement": .7,
                              "guided_native_improvement_fraction": .6, "guided_native_mean_delta_max": 0.,
                              "all_qualification_renders_accepted": True, "minimum_training_acceptance_fraction": .9},
            "baselines": ["training_mean", "nearest_training_record", "five_neighbor_interpolation"],
            "gradient_probe": {"steps": 16, "learning_rate": .03, "continuous_radius": .15, "joint_categories": True},
            "test_split_used": False, "helper_refresh_during_inverse_training": False}
    if config.paired_changes:
        from .paired_spectral import pair_spec
        plan.update(format="local_basic_paired_temporal_proxy_v1", paired_objective=pair_spec(),
                    control_coverage="four_original_performances_plus_two_isolated_active_controls_unison_and_filter_pairs")
    return plan, recipes, validation


class NativeSession:
    def __init__(self, output, plan, codec, monitor, renderer_factory=None):
        self.output, self.plan, self.codec, self.monitor = output, plan, codec, monitor
        self.factory = renderer_factory or NativeRendererFactory(renderer_id=codec.contract.renderer_id)
        self.path = output / "renders.jsonl"
        self.events = [json.loads(line) for line in self.path.read_text().splitlines()] if self.path.exists() else []
        self.renderer_identity = next((row["renderer_identity"] for row in self.events if row.get("renderer_identity") is not None), None)
        self.started = json.loads((output / "run.json").read_text())["started_unix"]

    def check_limits(self):
        if time() - self.started >= self.plan["limits"]["wall_seconds"]:
            raise RuntimeError("helper wall-time budget exhausted")
        if (self.monitor.peak_bytes or 0) > self.plan["limits"]["process_tree_bytes"]:
            raise RuntimeError("helper process-tree memory budget exhausted")

    def _append(self, value):
        import os
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(value, allow_nan=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self.events.append(value)

    def render(self, target, performance, phase, label):
        self.check_limits()
        count = sum(row["event"] == "attempt" and row["phase"] == phase for row in self.events)
        if count >= self.plan["limits"][f"{phase}_attempts"]:
            raise RuntimeError("helper native-attempt budget exhausted")
        source = self.codec.decode(target)
        request = RenderRequest(preset_json=fixed_generator_phases(source).decode(), performance=performance,
                                renderer_id=self.codec.contract.renderer_id, tail_seconds=self.codec.contract.tail_seconds,
                                preroll_frames=self.plan.get("render_profile", {}).get("preroll_frames", 0))
        identity = {"phase": phase, "label": label, "request_id": request.request_id, "attempt": count,
                    "source_sha256": sha256_bytes(source), "controlled_sha256": sha256_bytes(request.preset_json.encode())}
        self._append({"event": "attempt", **identity})
        remaining = self.plan["limits"]["wall_seconds"] - (time() - self.started)
        with RenderWorker(renderer_factory=self.factory, max_jobs=1) as worker:
            audio, metadata = worker.render(request, timeout=min(60., max(.1, remaining)))
        if audio is not None:
            current = {key: value for key, value in metadata["provenance"].items() if key != "request_id"}
            if self.renderer_identity is not None and current != self.renderer_identity:
                raise ValueError("renderer identity changed")
            self.renderer_identity = current
        result = {"event": "outcome", **identity, **metadata, "renderer_identity": self.renderer_identity}
        self._append(result)
        return audio, result


def collect_frames(pool, output, recipes, session, progress=None):
    from .codec import PilotTarget
    measured = SpectralTargets()
    records = output / "records"
    records.mkdir(exist_ok=True)
    index_path = output / "records.json"
    committed = json.loads(index_path.read_text()) if index_path.exists() else []
    archive_bytes = sum(row.get("bytes", 0) for row in committed)
    for i, recipe in enumerate(recipes):
        if i < len(committed):
            row = committed[i]
            if row["recipe_id"] != canonical_sha256(recipe):
                raise ValueError("record recipe changed")
            if row["accepted"] and sha256_bytes((records / row["file"]).read_bytes()) != row["sha256"]:
                raise ValueError("record bytes changed")
            continue
        target = PilotTarget.from_dict(recipe["target"])
        audio, receipt = session.render(target, Performance.from_dict(recipe["performance"]), "training", str(i))
        row = {"recipe_id": canonical_sha256(recipe), "sequence": i, "accepted": audio is not None, "receipt": receipt}
        if audio is not None:
            full = measured(torch.from_numpy(audio.copy()).T[None])
            frame_seed = recipe.get("pair_id", i)
            generator = torch.Generator().manual_seed(session.plan["config"]["seed"] + frame_seed)
            if session.plan["config"].get("event_frames"):
                from .event_frames import sample_event_frames
                values, times = sample_event_frames(full, Performance.from_dict(recipe["performance"]), generator)
            else:
                values, times = sample_frames(full, generator)
            record = {"measurements": values, "times": times, "target": recipe["target"],
                      "performance": performance_inputs(Performance.from_dict(recipe["performance"])), "recipe_id": row["recipe_id"]}
            if "pair_id" in recipe:
                record.update(pair_id=recipe["pair_id"], source_state_id=recipe["source_state_id"])
            path = records / f"{i:05d}.pt"
            atomic_torch_save(path, record)
            size = path.stat().st_size
            archive_bytes = sum(path.stat().st_size for path in records.iterdir() if path.is_file())
            if archive_bytes > session.plan["limits"]["archive_bytes"]:
                raise RuntimeError("helper feature archive budget exceeded")
            row.update(file=path.name, bytes=size, sha256=sha256_bytes(path.read_bytes()), target_bytes=target_bytes(values))
        committed.append(row)
        write_json(index_path, committed)
        if progress:
            progress({"stage": "collect", "completed": len(committed), "planned": len(recipes), "accepted": audio is not None})
    if sum(row["accepted"] for row in committed) < .9 * len(recipes):
        raise RuntimeError("helper training audio failed the frozen acceptance floor")
    return committed


class RecordCache:
    def __init__(self, output, rows):
        self.directory, self.rows = output / "records", [row for row in rows if row["accepted"]]
        self.cache, self.bytes = OrderedDict(), 0

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        if index in self.cache:
            self.cache.move_to_end(index)
            return self.cache[index][0]
        row = self.rows[index]
        path = self.directory / row["file"]
        if sha256_bytes(path.read_bytes()) != row["sha256"]:
            raise ValueError("helper record changed")
        value = torch.load(path, map_location="cpu", weights_only=True)
        if value["recipe_id"] != row["recipe_id"]:
            raise ValueError("helper record identity changed")
        size = target_bytes(value["measurements"]) + sum(t.numel() * t.element_size() for t in value["times"]) + value["performance"].numel() * 4
        if size > 512 * 1024**2:
            raise ValueError("one helper record exceeds replay limit")
        while self.cache and (len(self.cache) >= 512 or self.bytes + size > 512 * 1024**2):
            _, (_, removed) = self.cache.popitem(last=False)
            self.bytes -= removed
        self.cache[index] = (value, size)
        self.bytes += size
        return value
