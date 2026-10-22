"""Checkpointed paired updates; execution policy is supplied by a separately validated fit plan."""

import json

import torch

from obruxo_data.hashing import canonical_sha256
from obruxo_data.midi import Performance

from .audit import write_json
from .checkpoints import atomic_torch_save
from .codec import PilotTarget
from .native_fit import append_durable, record_features
from .paired_native_step import matched_pair_step
from .spectral import SpectralTargets, spectral_loss
from .trainer import capture_rng, restore_rng

CHECKPOINT_FORMAT = "paired_native_inverse_checkpoint_v1"


def require_pair(update):
    examples = update.get("examples", [])
    if len(examples) != 2:
        raise ValueError("paired schedule requires two examples per update")
    original, wider = examples
    if (original["state"] != wider["state"] or original["input_view"] != 0 or original["view"] not in (1, 2)
            or wider["input_view"] != original["view"] or wider["view"] != 0):
        raise ValueError("paired schedule requires reciprocal original/wider views of one state")
    return examples


def fit_pairs(models, records, session, output, plan, saved=None, *, after_step=None):
    policy = plan["production_training"]
    if policy["forward"] != "unperturbed_argmax" or policy["parameter_logits"] != "raw":
        raise ValueError("paired training requires production choices and raw control logits")
    for update in plan["updates"]:
        require_pair(update)
    optimizers = {name: torch.optim.Adam(model.parameters(), lr=plan["optimizer"]["learning_rate"]) for name, model in models.items()}
    for model in models.values():
        model.train()
    trace = []
    if saved:
        if (saved["format"] != CHECKPOINT_FORMAT or saved["plan_id"] != canonical_sha256(plan)
                or saved["step"] != len(saved["trace"]) or saved["production_training"] != policy):
            raise ValueError("paired checkpoint contract changed")
        for name in models:
            models[name].load_state_dict(saved["models"][name])
            optimizers[name].load_state_dict(saved["optimizers"][name])
        trace = list(saved["trace"])
        restore_rng(saved["rng"], "cpu")
    else:
        torch.manual_seed(plan["optimizer"]["seed"])
    charges_path = output / "updates.jsonl"
    charges = [json.loads(line) for line in charges_path.read_text().splitlines()] if charges_path.exists() else []
    for index, charge in enumerate(charges):
        step = charge["step"]
        if (charge["attempt"] != index or not 1 <= step <= len(plan["updates"])
                or (index == 0 and step != 1) or (index > 0 and step not in (charges[index - 1]["step"], charges[index - 1]["step"] + 1))
                or charge["examples"] != plan["updates"][step - 1]["examples"]):
            raise ValueError("paired optimizer-attempt ledger changed")
    for step, row in enumerate(trace):
        charge = {key: row[key] for key in ("attempt", "step", "examples")}
        if row["step"] != step + 1 or row["attempt"] >= len(charges) or charges[row["attempt"]] != charge:
            raise ValueError("paired trace differs from charged optimizer attempts")
    measure = SpectralTargets()
    for step in range(len(trace), len(plan["updates"])):
        records.check_limits()
        if len(charges) >= plan["limits"]["optimizer_attempts"]:
            raise RuntimeError("paired optimizer-attempt budget exhausted")
        examples = require_pair(plan["updates"][step])
        selected = [records[row["state"]] for row in examples]
        charge = {"attempt": len(charges), "step": step + 1, "examples": examples}
        append_durable(charges_path, charge)
        charges.append(charge)
        query = 0

        def evaluate(index, candidate):
            nonlocal query
            if index not in (0, 1):
                raise ValueError("paired native callback index changed")
            row = examples[index]
            recipe = plan["training"][row["state"]]
            performance = Performance.from_dict(recipe["performances"][row["view"]])
            audio, receipt = session.render(candidate, performance, "training", f"gradient:{step}:{index}:{query}")
            current = query
            query += 1
            if audio is None:
                raise RuntimeError("paired native sound render was not accepted")
            loss = float(spectral_loss(measure(torch.from_numpy(audio.copy()).T[None]), selected[index]["targets"][row["view"]])["loss"])
            append_durable(output / "measurements.jsonl", {"optimizer_attempt": charge["attempt"], "step": step + 1,
                "query": current, "example": index, "state": row["state"], "input_view": row["input_view"], "view": row["view"],
                "candidate": candidate.to_dict(), "loss": loss, "request_id": receipt["request_id"],
                "audio_sha256": receipt["qa"]["audio_float32_sha256"]})
            return loss

        features = [record_features(record, row["input_view"]) for row, record in zip(examples, selected, strict=True)]
        targets = [PilotTarget.from_dict(plan["training"][row["state"]]["target"]) for row in examples]
        result = matched_pair_step(models, optimizers, features, targets, session.codec, evaluate, step, len(plan["updates"]), scales=policy["scales"])
        trace.append({**charge, **result})
        state = {"format": CHECKPOINT_FORMAT, "plan_id": canonical_sha256(plan), "step": step + 1,
                 "models": {name: model.state_dict() for name, model in models.items()},
                 "optimizers": {name: optimizer.state_dict() for name, optimizer in optimizers.items()},
                 "rng": capture_rng("cpu"), "trace": trace, "production_training": policy}
        atomic_torch_save(output / "checkpoint.pt", state)
        write_json(output / "progress.json", {"committed_updates": step + 1, "planned_updates": len(plan["updates"]), "losses": result["losses"]})
        if after_step:
            after_step(state)
        if (step + 1) % 8 == 0:
            print(json.dumps({"stage": "paired_training", "step": step + 1, "planned": len(plan["updates"]), "losses": result["losses"]}), flush=True)
    return trace
