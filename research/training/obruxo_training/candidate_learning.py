"""Matched inverse learning with source supervision and confirmed render-selected targets."""

from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import torch

from obruxo_data.hashing import canonical_sha256

from .audit import write_json
from .benchmark import MemoryMonitor
from .candidate_data import CandidateConfig, validate_dataset
from .codec import PilotTarget
from .features import tensor_digest
from .model import PilotInverse, batch_features, loss_terms
from .trainer import atomic_torch_save, export_model


FORMAT = "obruxo_render_selected_training_v1"


def learning_contract(source, data, config, auxiliary_weight):
    keys = ("pool_id", "contract", "codec", "model", "frontend", "precision")
    return {**{key: source["run_contract"][key] for key in keys}, "trainer": FORMAT, "training": asdict(config),
            "source_run_id": canonical_sha256(source["run_contract"]), "source_model_sha256": tensor_digest(source["model"]),
            "candidate_data_sha256": data["sha256"], "auxiliary_weight": auxiliary_weight,
            "optimizer": "AdamW_weight_decay_0_fresh_state", "objective": "source_parameter_anchor_plus_confirmed_candidate_v1",
            "minibatches": "half_original_replay_half_new_training_audio", "inference_input": "audio_only"}


def restore_candidate_checkpoint(saved, pool):
    from .evaluate import restore_evaluation_checkpoint
    if saved.get("format") != FORMAT or saved["source"].get("format") != "obruxo_pilot_training_v1":
        raise ValueError("candidate checkpoint needs an original supervised source lineage")
    source, original, frontend, sampler = restore_evaluation_checkpoint(saved["source"], pool)
    config = CandidateConfig(**saved["run_contract"]["training"])
    if asdict(config) != saved["candidate_data"]["plan"]["config"]:
        raise ValueError("candidate learning configuration differs from its frozen data plan")
    weight = saved["run_contract"]["auxiliary_weight"]
    if weight not in {0., config.candidate_weight}:
        raise ValueError("candidate checkpoint has an undeclared auxiliary objective")
    validate_dataset(saved["candidate_data"], pool, source, frontend, original)
    if len(saved["candidate_data"]["cases"]) != config.training_states:
        raise ValueError("candidate learning requires the complete planned training prefix")
    if saved["run_contract"] != learning_contract(source, saved["candidate_data"], config, weight):
        raise ValueError("candidate training contracts changed")
    if (not 0 <= saved["step"] <= config.steps or tensor_digest(saved["model"]) != saved["model_sha256"]
            or saved["renderer_identity"] != source["renderer_identity"] or saved["cursor"] != source["cursor"]
            or tensor_digest(saved["basic_pitch"]) != frontend.frozen_digest):
        raise ValueError("candidate model, renderer or frozen frontend changed")
    # Retrieval remains the original training-only bank; extra selected targets are never retrieval labels.
    from .evaluate import retrieval_digest
    if retrieval_digest(saved["replay"]) != retrieval_digest(source["replay"]):
        raise ValueError("candidate checkpoint changed the fixed retrieval bank")
    model = PilotInverse(pool.codec, original.conditioning).eval()
    model.load_state_dict(saved["model"], strict=True)
    return saved, model, frontend, sampler


def train_candidates(pool, source, data, output, *, config=None, auxiliary_weight=0., resume=False, progress=None, after_step=None):
    from .evaluate import restore_evaluation_checkpoint
    config = config or CandidateConfig()
    if asdict(config) != data["plan"]["config"]:
        raise ValueError("candidate learning configuration differs from its frozen data plan")
    if auxiliary_weight not in {0., config.candidate_weight}:
        raise ValueError("candidate arm weight differs from the frozen configuration")
    _, initial, frontend, _ = restore_evaluation_checkpoint(source, pool)
    validate_dataset(data, pool, source, frontend, initial)
    if len(data["cases"]) != config.training_states or not data["records"]:
        raise ValueError("candidate learning requires complete collected training cases")
    torch.set_num_threads(config.cpu_threads)
    torch.manual_seed(config.seed)
    contract = learning_contract(source, data, config, auxiliary_weight)
    model = PilotInverse(pool.codec, initial.conditioning)
    model.load_state_dict(source["model"], strict=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0)
    generator = torch.Generator().manual_seed(config.seed)
    output = Path(output)
    step, trace = 0, []
    uses = {"original": [0] * len(source["replay"]), "fresh": [0] * len(data["records"])}
    if resume:
        saved = torch.load(output / "checkpoint.pt", map_location="cpu", weights_only=True)
        restore_candidate_checkpoint(saved, pool)
        if saved["run_contract"] != contract:
            raise ValueError("candidate learning plan changed; cannot resume")
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        generator.set_state(saved["batch_rng"])
        step, trace, uses = saved["step"], saved["trace"], saved["uses"]
    else:
        output.mkdir(parents=True, exist_ok=False)
        write_json(output / "plan.json", contract)
    started, monitor = perf_counter(), MemoryMonitor()

    def save():
        saved = {"format": FORMAT, "run_contract": contract, "source": source, "candidate_data": data,
                 "model": model.state_dict(), "model_sha256": tensor_digest(model.state_dict()),
                 "basic_pitch": source["basic_pitch"], "replay": source["replay"], "renderer_identity": source["renderer_identity"],
                 "cursor": source["cursor"], "optimizer": optimizer.state_dict(), "batch_rng": generator.get_state(),
                 "step": step, "trace": trace, "uses": uses}
        atomic_torch_save(output / "checkpoint.pt", saved)

    save()
    monitor.start()
    try:
        while step < config.steps:
            indices = {name: torch.randint(len(items), (config.batch_size // 2,), generator=generator).tolist()
                       for name, items in (("original", source["replay"]), ("fresh", data["records"]))}
            old = [source["replay"][index] for index in indices["original"]]
            fresh = [data["records"][index] for index in indices["fresh"]]
            items = [*old, *fresh]
            optimizer.zero_grad(set_to_none=True)
            model.train()
            prediction = model(batch_features(items))
            anchor = loss_terms(prediction, [PilotTarget.from_dict(item["target"]) for item in items], pool.codec)["loss"]
            selected = [len(old) + index for index, item in enumerate(fresh) if item["improved_target"] is not None]
            auxiliary = anchor.new_zeros(())
            if selected and auxiliary_weight:
                subset = {"continuous": prediction["continuous"][selected], "categorical": [head[selected] for head in prediction["categorical"]]}
                targets = [PilotTarget.from_dict(items[index]["improved_target"]) for index in selected]
                auxiliary = loss_terms(subset, targets, pool.codec)["loss"] * len(selected) / len(fresh)
            loss = anchor + auxiliary_weight * auxiliary
            if not torch.isfinite(loss):
                raise ValueError("nonfinite candidate training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
            optimizer.step()
            if any(not torch.isfinite(parameter).all() for parameter in model.parameters()):
                raise ValueError("nonfinite candidate model weights")
            for name, values in indices.items():
                for index in values:
                    uses[name][index] += 1
            step += 1
            if step % config.checkpoint_every == 0 or step == config.steps:
                trace.append({"step": step, "source_parameter_loss": float(anchor.detach()), "candidate_loss": float(auxiliary.detach()),
                              "loss": float(loss.detach())})
                save()
                if progress:
                    progress({"stage": "candidate_learning", "auxiliary_weight": auxiliary_weight, **trace[-1]})
            if after_step:
                after_step(step)
        frontend.require_frozen()
        model.eval()
        export_model(output / "model.pt", model, frontend, pool.codec, pool.pool_id)
        summary = {"format": FORMAT, "complete": step == config.steps, "step": step, "auxiliary_weight": auxiliary_weight,
                   "model_sha256": tensor_digest(model.state_dict()), "source_model_sha256": tensor_digest(source["model"]),
                   "uses": uses, "batch_rng_sha256": tensor_digest({"rng": generator.get_state()}), "trace": trace,
                   "elapsed_seconds_this_segment": perf_counter() - started, "peak_process_tree_rss_bytes": monitor.peak_bytes,
                   "new_renders": 0, "frozen_frontend_unchanged": True, "final_test_used": False}
        write_json(output / "summary.json", summary)
        return summary
    finally:
        monitor.close()
