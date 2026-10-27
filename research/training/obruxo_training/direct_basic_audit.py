"""Audit every direct request, replay Adam exactly and rescore retained endpoints."""

from time import perf_counter

from obruxo_data.hashing import canonical_sha256

from .audit import write_json
from .benchmark import MemoryMonitor
from .codec import load_codec
from .conditioned_bridge import read
from .direct_basic_fit import checkpoint, reconstruct
from .direct_basic_store import DirectStore
from .native_fit import digest
from .paired_native_evidence import rescore_waveform
from .unison_sound_fit import lines


def audit_ledger(events, rows, plan):
    attempts, outcomes = {}, {}
    fields = ("phase", "label", "request_id", "source_sha256", "controlled_sha256")
    for event in events:
        key = event["phase"], event["attempt"]
        if event["event"] not in ("attempt", "outcome") or key[0] not in ("training", "qualification"):
            raise ValueError("unknown direct native event")
        destination = attempts if event["event"] == "attempt" else outcomes
        if key in destination:
            raise ValueError("duplicate direct native event")
        destination[key] = event
    if attempts.keys() != outcomes.keys() or len(attempts) > plan["maximum_native_requests"]:
        raise ValueError("unobserved direct request or total budget exceeded")
    for phase in ("training", "qualification"):
        sequence = sorted(key[1] for key in attempts if key[0] == phase)
        if sequence != list(range(len(sequence))) or len(sequence) > plan["limits"][phase + "_attempts"]:
            raise ValueError("direct native phase counter or allocation changed")
    cited, recovered = set(), []
    for row in rows.values():
        receipt = row["receipt"]
        key = receipt["phase"], receipt["attempt"]
        raw = {name: value for name, value in receipt.items() if name != "retained_audio"}
        if key in cited or raw != outcomes.get(key) or receipt["label"] != "direct:" + canonical_sha256(row["query"]):
            raise ValueError("direct measurement cites a duplicate, altered or foreign receipt")
        cited.add(key)
    for key, outcome in outcomes.items():
        if any(outcome.get(field) != attempts[key].get(field) for field in fields):
            raise ValueError("direct native outcome differs from its charged request")
        if outcome["status"] in ("accepted", "rejected"):
            if key not in cited:
                raise ValueError("direct native audio outcome lacks a measurement record")
            continue
        if outcome["status"] != "error" or outcome.get("error_type") != "RenderTimeout" or "qa" in outcome:
            raise ValueError("unsupported direct native failure")
        later = [row for next_key, row in outcomes.items() if next_key[0] == key[0] and next_key[1] > key[1]
                 and row["status"] == "accepted" and all(row.get(field) == outcome.get(field) for field in fields)]
        if not later:
            raise ValueError("direct timeout lacks an accepted identical recovery")
        recovered.append({"phase": key[0], "failed_attempt": key[1], "accepted_attempt": later[0]["attempt"]})
    if len(recovered) > 1:
        raise ValueError("direct timeout recovery allocation exceeded")
    return {"attempts": len(attempts), "accepted": sum(row["status"] == "accepted" for row in outcomes.values()),
            "invalid_trials": sum(row["status"] == "rejected" for row in outcomes.values()), "recovered_timeouts": recovered}


def verify(plan, fit, output):
    summary = read(output / "summary.json")
    if not summary["complete"] or not summary["within_resource_limits"] or summary["plan_id"] != canonical_sha256(plan):
        raise ValueError("direct verification requires a complete run within its unchanged limits")
    if (output / "verification.json").exists():
        raise FileExistsError("preserve the existing direct verification")
    started, monitor = perf_counter(), MemoryMonitor()
    monitor.start()
    try:
        codec = load_codec(plan["codec"])
        store = DirectStore(output, plan, codec, fit)
        ledger = audit_ledger(lines(output / "renders.jsonl"), store.rows, plan)
        rescored = 0
        for row in store.rows.values():
            store.require_receipt(row)
            if "retained_audio" in row["receipt"]:
                reference, _targets = store.source(row["query"]["case"], row["query"]["view"])
                if rescore_waveform(output, row["receipt"], reference, codec.contract.tail_seconds) != row["scores"]:
                    raise ValueError("direct retained endpoint score changed")
                rescored += 1
        replay = reconstruct(plan, store, checkpoint(output, plan))
        recorded = read(output / "result.json")
        if recorded != {name: value for name, value in replay.items() if name not in ("states", "trace")} or store.used != store.rows.keys():
            raise ValueError("direct result changed or native measurements were silently excluded")
        attempts = lines(output / "attempts.jsonl")
        completed = {(row["case"], row["result"]["step"]) for row in replay["trace"]}
        if ({(row["case"], row["step"]) for row in attempts} != completed
                or not len(completed) <= len(attempts) <= len(completed) + 1
                or any(row["ordinal"] != i for i, row in enumerate(attempts))
                or any(sum(row["case"] == case for row in attempts) > plan["limits"]["optimizer_attempts_per_case"] for case in range(3))):
            raise ValueError("direct charged updates differ from the replayed work or allocation")
        executions = lines(output / "executions.jsonl")
        if (not 1 <= len(executions) <= plan["limits"]["native_executions"] or rescored != 24
                or summary["native_attempts"] != ledger["attempts"] or summary["committed_updates"] != len(replay["trace"])):
            raise ValueError("direct execution, endpoint or summary accounting changed")
        evidence = {path.relative_to(output).as_posix(): digest(path) for path in output.rglob("*") if path.is_file()
                    and path.suffix in (".pt", ".json", ".jsonl") and path.name != "verification.json"}
    finally:
        monitor.close()
    elapsed = perf_counter() - started
    if elapsed > 600 or not monitor.peak_bytes or monitor.peak_bytes > 4 * 1024**3 or monitor.errors:
        raise ValueError("direct verification exceeded ten minutes or 4 GiB")
    report = {"plan_id": canonical_sha256(plan), "verified": True, "complete": True, "offline_trajectory_and_Adam_replay_exact": True,
              "comparison_waveforms_rescored": rescored, "scalar_measurements_replayed": len(store.rows), "native_ledger": ledger,
              "charged_direct_update_attempts": len(attempts), "committed_direct_updates": len(replay["trace"]),
              "result": recorded, "direct_quality_passed": recorded["pass"], "new_native_requests": 0,
              "inverse_training_updates": 0, "qualified_for_release": False, "test_split_used": False, "evidence_sha256": evidence,
              "resources": {"elapsed_seconds": elapsed, "peak_process_tree_rss_bytes": monitor.peak_bytes, "rss_sampling_errors": monitor.errors}}
    write_json(output / "verification.json", report)
    return report
