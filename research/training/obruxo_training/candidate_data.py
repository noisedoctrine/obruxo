"""Train-only native-render selection, retaining bounded measurements and full QA outcomes."""

from collections import Counter
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import platform
from time import perf_counter

import torch

from obruxo_data.hashing import canonical_sha256, sha256_bytes
from obruxo_data.render.base import RenderRequest

from .audit import write_json
from .benchmark import MemoryMonitor
from .candidates import PROPOSAL_SPEC, confirmed_improvement, propose_candidates
from .codec import PilotTarget
from .contracts import RenderRecipe
from .evaluate import audio_distance, load_evaluation_checkpoint, retrieval_digest
from .features import tensor_digest, validate_features
from .model import batch_features
from .performances import initial_bank
from .proxy_screen import source_identity
from .splits import SplitLedger
from .stream import NativeRendererFactory, render_isolated
from .trainer import atomic_torch_save, replay_bytes


FORMAT = "obruxo_render_selected_data_v1"


@dataclass(frozen=True)
class CandidateConfig:
    training_states: int = 32
    validation_states: int = 10
    steps: int = 1500
    batch_size: int = 16
    learning_rate: float = .0001
    candidate_weight: float = .5
    confirmation_margin: float = .05
    checkpoint_every: int = 250
    replay_mib: int = 8
    cpu_threads: int = 4
    seed: int = 0
    timeout: float = 30.

    def __post_init__(self):
        integers = (self.training_states, self.validation_states, self.steps, self.batch_size, self.checkpoint_every,
                    self.replay_mib, self.cpu_threads)
        if any(type(value) is not int or value <= 0 for value in integers) or not 8 <= self.training_states <= 128:
            raise ValueError("candidate budgets must be positive integers with 8..128 training states")
        if self.batch_size % 2 or self.batch_size < 2 or type(self.seed) is not int:
            raise ValueError("candidate learning needs an even batch and integer seed")
        if not math.isfinite(self.learning_rate) or not 0 < self.learning_rate <= .01:
            raise ValueError("invalid candidate learning rate")
        if not math.isfinite(self.candidate_weight) or not 0 <= self.candidate_weight <= 1:
            raise ValueError("candidate weight must be finite and between zero and one")
        if not math.isfinite(self.confirmation_margin) or not 0 <= self.confirmation_margin < 1:
            raise ValueError("invalid confirmation margin")
        if not math.isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError("invalid candidate render timeout")


def training_panel(pool, source, count):
    excluded = {item["state_id"] for item in source["replay"]}
    groups = {}
    for entry in pool.entries:
        if pool.manifest.by_state[entry.state_id].split != "train" or entry.state_id in excluded:
            continue
        family = pool.codec.family(entry.target) if hasattr(pool.codec, "family") else "pilot"
        groups.setdefault(family, []).append(entry)
    ordered = [entries[index] for index in range(max(map(len, groups.values()), default=0))
               for entries in groups.values() if index < len(entries)]
    if len(ordered) < count:
        raise ValueError("insufficient fresh training states outside the source replay")
    return ordered[:count]


def dataset_digest(data):
    records = [{**{key: value for key, value in item.items() if key != "features"},
                "features_sha256": tensor_digest(item["features"])} for item in data["records"]]
    return canonical_sha256({"plan": data["plan"], "cases": data["cases"], "history": data["history"], "records": records})


def validate_dataset(data, pool, source, frontend, model):
    plan = data["plan"]
    config = CandidateConfig(**plan["config"])
    torch.set_num_threads(config.cpu_threads)
    if (data["format"] != FORMAT or data["sha256"] != dataset_digest(data) or plan["pool_id"] != pool.pool_id
            or plan["source_model_sha256"] != tensor_digest(source["model"])
            or plan["source_retrieval_sha256"] != retrieval_digest(source["replay"])
            or plan["frontend"] != frontend.describe() or plan["proposal"] != PROPOSAL_SPEC):
        raise ValueError("candidate dataset or source contracts changed")
    panel = training_panel(pool, source, config.training_states)
    if plan["training_states"] != [entry.state_id for entry in panel] or len(data["cases"]) > len(panel):
        raise ValueError("candidate data differs from its fresh train-only panel")
    bank, ledger = initial_bank()[:4], SplitLedger(pool.manifest)
    records = {item["case"]: item for item in data["records"]}
    if len(records) != len(data["records"]) or any(not 0 <= case < len(data["cases"]) for case in records):
        raise ValueError("candidate records lie outside completed cases")
    for index, row in enumerate(data["cases"]):
        entry, performance = panel[index], bank[index % 4]
        recipe = RenderRecipe.create(pool.codec.decode(entry.target), performance.performance(), schema=pool.codec.schema,
                                     catalog=pool.codec.catalog, contract=pool.codec.contract, ledger=ledger, purpose="train", sequence_id=index)
        if row["case"] != index or row["recipe_id"] != recipe.recipe_id:
            raise ValueError("candidate case is not its declared training recipe")
        if not row["source_accepted"]:
            if index in records:
                raise ValueError("rejected native source became a training target")
            continue
        item = records[index]
        if (item["state_id"] != entry.state_id or item["target"] != entry.target.to_dict()
                or item["performance_id"] != performance.performance_id):
            raise ValueError("candidate record contains held-out or changed labels")
        validate_features(item["features"])
        events = [event for event in data["history"] if event["case"] == index and event["method"] == "source"]
        if not any(event["status"] == "accepted" and event["request_id"] == recipe.render_request().request_id
                   and event["qa"]["audio_float32_sha256"] == item["audio_sha256"] for event in events):
            raise ValueError("candidate features have no matching accepted native source")
        with torch.no_grad():
            prediction = model(batch_features([item]))
            candidates = propose_candidates(prediction, item["features"], source["replay"], pool.codec, seed=config.seed + index)
        if row["proposals"] != {name: target.to_dict() for name, target in candidates.items()}:
            raise ValueError("candidate proposal differs from its audio-only source model")
        if set(row["scores"]) != set(candidates):
            raise ValueError("candidate scores do not cover the declared proposals")
        source_attempt = max(event["attempt"] for event in events if event["status"] == "accepted"
                             and event["qa"]["audio_float32_sha256"] == item["audio_sha256"])
        current = {event["method"]: event for event in data["history"] if event["case"] == index and event["attempt"] >= source_attempt}
        for name, target in candidates.items():
            request = RenderRequest(preset_json=pool.codec.decode(target).decode(), performance=performance.performance(),
                                    sample_rate=pool.codec.contract.sample_rate, channels=2, tail_seconds=pool.codec.contract.tail_seconds,
                                    renderer_id=pool.codec.contract.renderer_id)
            methods = [name] + ([name + "_confirmation"] if name + "_confirmation" in current else [])
            if len(row["scores"][name]) != len(methods):
                raise ValueError("candidate confirmation has no matching native attempt")
            for score, method in zip(row["scores"][name], methods, strict=True):
                event = current.get(method, {})
                if event.get("request_id") != request.request_id or (score is not None) != (event.get("status") == "accepted"):
                    raise ValueError("candidate score differs from its native QA outcome")
        selected = confirmed_improvement(row["scores"], relative_margin=config.confirmation_margin)
        expected = candidates[selected].to_dict() if selected else None
        if row["selected"] != selected or item["improved_target"] != expected:
            raise ValueError("candidate target lacks a confirmed native improvement")
    if replay_bytes([*source["replay"], *data["records"]]) > config.replay_mib * 1024 * 1024:
        raise ValueError("candidate feature replay exceeds its byte budget")
    if len(data["history"]) > plan["maximum_training_attempts"]:
        raise ValueError("candidate native-attempt budget exceeded")
    for index, event in enumerate(data["history"]):
        if event["attempt"] != index or not 0 <= event["case"] < config.training_states:
            raise ValueError("invalid candidate native history")
        if event["status"] != "error":
            identity = {key: value for key, value in event["provenance"].items() if key != "request_id"}
            if identity != source["renderer_identity"]:
                raise ValueError("candidate renderer differs from its source renderer")


def collect_candidates(pool, checkpoint, output, *, config=None, renderer_factory=None, resume=False, progress=None):
    config = config or CandidateConfig()
    torch.set_num_threads(config.cpu_threads)
    source, model, frontend, _ = load_evaluation_checkpoint(checkpoint, pool)
    if source["format"] != "obruxo_pilot_training_v1":
        raise ValueError("this candidate screen needs an original supervised source checkpoint")
    maximum_features = replay_bytes(source["replay"]) + config.training_states * replay_bytes(source["replay"][:1])
    if maximum_features > config.replay_mib * 1024 * 1024:
        raise ValueError("declared candidate feature budget is too small for the planned training panel")
    panel = training_panel(pool, source, config.training_states)
    plan = {"format": FORMAT, "pool_id": pool.pool_id, "config": asdict(config), "source": source_identity(),
            "runtime": {"python": platform.python_version(), "torch": str(torch.__version__), "device": "cpu", "precision": "float32"},
            "source_checkpoint_sha256": sha256_bytes(Path(checkpoint).read_bytes()),
            "source_model_sha256": tensor_digest(model.state_dict()), "source_retrieval_sha256": retrieval_digest(source["replay"]),
            "frontend": frontend.describe(), "proposal": PROPOSAL_SPEC, "training_states": [entry.state_id for entry in panel],
            "training_performances": [initial_bank()[index % 4].performance_id for index in range(len(panel))],
            "planned_training_attempts": 7 * len(panel), "recovery_reserve": 14, "maximum_training_attempts": 7 * len(panel) + 14,
            "confirmation": "worst_of_two_candidate_below_best_of_two_incumbent_by_margin",
            "final_test_used": False, "inference_input": "audio_only", "known_training_midi_enters_proposer": False}
    output = Path(output)
    if resume:
        data = torch.load(output / "data.pt", map_location="cpu", weights_only=True)
        if data["plan"] != plan:
            raise ValueError("candidate plan changed; cannot resume")
        validate_dataset(data, pool, source, frontend, model)
        history = [json.loads(line) for line in (output / "renders.jsonl").read_text().splitlines()]
        if history[:len(data["history"])] != data["history"]:
            raise ValueError("committed candidate native log changed")
        data["history"] = history
    else:
        output.mkdir(parents=True, exist_ok=False)
        write_json(output / "plan.json", plan)
        (output / "renders.jsonl").touch()
        data = {"format": FORMAT, "plan": plan, "records": [], "cases": [], "history": []}
    started, monitor = perf_counter(), MemoryMonitor()

    def save():
        data["sha256"] = dataset_digest(data)
        atomic_torch_save(output / "data.pt", data)

    def render(target, performance, case, method):
        if len(data["history"]) >= plan["maximum_training_attempts"]:
            raise ValueError("frozen candidate training-render budget exhausted")
        request = RenderRequest(preset_json=pool.codec.decode(target).decode(), performance=performance,
                                sample_rate=pool.codec.contract.sample_rate, channels=2, tail_seconds=pool.codec.contract.tail_seconds,
                                renderer_id=pool.codec.contract.renderer_id)
        audio, metadata = render_isolated(request, renderer_factory=renderer_factory or NativeRendererFactory(), timeout=config.timeout)
        event = {"attempt": len(data["history"]), "case": case, "method": method, "request_id": request.request_id, **metadata}
        data["history"].append(event)
        with (output / "renders.jsonl").open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(event, allow_nan=False) + "\n")
            handle.flush()
        if metadata["status"] != "error":
            identity = {key: value for key, value in metadata["provenance"].items() if key != "request_id"}
            if identity != source["renderer_identity"]:
                raise ValueError("candidate renderer differs from the trained source")
        return audio, metadata

    save()
    monitor.start()
    try:
        ledger = SplitLedger(pool.manifest)
        for index in range(len(data["cases"]), len(panel)):
            entry, performance = panel[index], initial_bank()[index % 4]
            recipe = RenderRecipe.create(pool.codec.decode(entry.target), performance.performance(), schema=pool.codec.schema,
                                         catalog=pool.codec.catalog, contract=pool.codec.contract, ledger=ledger, purpose="train", sequence_id=index)
            audio, metadata = render(entry.target, performance.performance(), index, "source")
            row = {"case": index, "recipe_id": recipe.recipe_id, "source_accepted": audio is not None,
                   "proposals": {}, "scores": {}, "selected": None}
            if audio is not None:
                features = frontend(audio)
                with torch.no_grad():
                    prediction = model(batch_features([{"features": features}]))
                    candidates = propose_candidates(prediction, features, source["replay"], pool.codec, seed=config.seed + index)
                row["proposals"] = {name: target.to_dict() for name, target in candidates.items()}
                for name, target in candidates.items():
                    candidate, _ = render(target, performance.performance(), index, name)
                    row["scores"][name] = [audio_distance(audio, candidate)["mrstft"] if candidate is not None else None]
                valid = [(values[0], name) for name, values in row["scores"].items() if values[0] is not None]
                winner = min(valid, key=lambda item: item[0])[1] if valid else "model"
                for name in dict.fromkeys(("model", winner)):
                    repeated, _ = render(candidates[name], performance.performance(), index, name + "_confirmation")
                    row["scores"][name].append(audio_distance(audio, repeated)["mrstft"] if repeated is not None else None)
                selected = confirmed_improvement(row["scores"], relative_margin=config.confirmation_margin)
                row["selected"] = selected
                data["records"].append({"case": index, "state_id": entry.state_id, "performance_id": performance.performance_id,
                                        "target": entry.target.to_dict(), "improved_target": candidates[selected].to_dict() if selected else None,
                                        "audio_sha256": metadata["qa"]["audio_float32_sha256"], "features": features})
            data["cases"].append(row)
            save()
            if progress:
                progress({"stage": "candidate_collection", "completed": index + 1, "planned": len(panel),
                          "confirmed": sum(item["improved_target"] is not None for item in data["records"])})
        validate_dataset(data, pool, source, frontend, model)
        families = Counter(pool.codec.family(PilotTarget.from_dict(item["target"])) if hasattr(pool.codec, "family") else "pilot"
                           for item in data["records"])
        if len(data["records"]) < .8 * len(panel) or (hasattr(pool.codec, "family") and set(families) != set(pool.codec.to_dict()["families"])):
            raise ValueError("native QA left insufficient candidate training coverage")
        summary = {"attempted": len(data["history"]), "accepted": sum(event["status"] == "accepted" for event in data["history"]),
                   "planned_cases": len(panel), "accepted_sources": len(data["records"]),
                   "confirmed_improvements": sum(item["improved_target"] is not None for item in data["records"]),
                   "families": dict(families), "feature_bytes_including_source": replay_bytes([*source["replay"], *data["records"]]),
                   "elapsed_seconds_this_segment": perf_counter() - started, "peak_process_tree_rss_bytes": monitor.peak_bytes,
                   "data_sha256": data["sha256"], "audio_files_written": 0, "final_test_used": False}
        write_json(output / "summary.json", summary)
        return data, summary
    finally:
        monitor.close()
