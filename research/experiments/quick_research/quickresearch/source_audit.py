from __future__ import annotations

from pathlib import Path, PurePosixPath
import re
import subprocess
from urllib.parse import quote, urlsplit

from .common import BudgetExpired, Context, digest, file_hash, finish_report, read_json, write_json

PINNED_REVISION = "636ca0ef517a4db087a6a08a6a8a5e704e21f836"
QUESTIONS = {
    "loading": "How do connection arrays and all modulation_<slot>_* controls map to runtime slots?",
    "graph": "How are route traversal, dependency scheduling and mono/poly accumulation ordered?",
    "state": "What smoothing, ramp and initialization/reset state belongs to each route, and in which versions?",
    "dependencies": "What happens when a route targets another route's amount, including cycles and relabeling?",
    "numerics": "Which guarantees are semantic, numerical-with-tolerance, or bit-identical, and what exceptions apply?",
}
PATTERNS = {
    "loading": r"ModulationConnection|modulation_connections|modulation_.*amount",
    "graph": r"topolog|reorder|processModulation|process_modulation|modulation.*(mono|poly)",
    "state": r"ramp_up|ramp_down|smooth|reset.*modulation",
    "dependencies": r"modulation.*destination|destination.*modulation|modulation.*amount|cycle",
    "numerics": r"accumul|sum.*modulation|modulation.*sum|ModulationConnection",
}


def git(context: Context, source: Path, *arguments: str, missing_ok: bool = False) -> str:
    context.check(3)
    result = subprocess.run(["git", "-C", str(source), *arguments], capture_output=True, text=True, encoding="utf-8", errors="replace",
                            timeout=max(.1, min(30, context.remaining - 2)),
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if result.returncode and not (missing_ok and result.returncode == 1):
        raise ValueError(f"git source inspection failed ({result.returncode}); verify checkout and revision")
    return result.stdout.strip()


def repository_url(remote: str) -> str | None:
    if remote.startswith("git@github.com:"):
        remote = "https://github.com/" + remote.split(":", 1)[1]
    parsed = urlsplit(remote)
    if parsed.hostname != "github.com":
        return None
    path = parsed.path.removesuffix(".git").strip("/")
    if len(path.split("/")) != 2:
        return None
    return "https://github.com/" + path


def validate_range(path: str, start: int, end: int, line_count: int) -> None:
    parsed = PurePosixPath(path)
    if parsed.is_absolute() or ".." in parsed.parts or ":" in path or "\\" in path:
        raise ValueError("citation requires a repository-relative POSIX path")
    if type(start) is not int or type(end) is not int or not 1 <= start <= end <= line_count or end - start > 100:
        raise ValueError("citation range must exist and contain at most 101 lines")


def run(context: Context, args) -> dict:
    source = Path(args.source).resolve(strict=True)
    if context.output.is_relative_to(source):
        raise ValueError("source checkout is read-only; output must be outside it")
    if not re.fullmatch(r"[0-9a-fA-F]{40}", args.revision):
        raise ValueError("source revision must be an immutable full 40-character commit SHA")
    revision = git(context, source, "rev-parse", "--verify", f"{args.revision}^{{commit}}")
    if not re.fullmatch(r"[0-9a-f]{40}", revision) or revision != args.revision.lower():
        raise ValueError("source revision did not resolve to the exact requested commit")
    remote = git(context, source, "config", "--get", "remote.origin.url", missing_ok=True)
    url = repository_url(remote)
    blobs, evidence = {}, []
    complete = True
    try:
        for question, pattern in PATTERNS.items():
            hits = git(context, source, "grep", "-n", "-I", "-i", "-E", pattern, revision, "--", "src", missing_ok=True).splitlines()
            selected = hits[:args.max_hits]
            for hit in selected:
                relative, number, _ = hit.removeprefix(revision + ":").split(":", 2)
                if relative not in blobs:
                    blobs[relative] = git(context, source, "show", f"{revision}:{relative}").splitlines()
                lines, number = blobs[relative], int(number)
                start, end = max(1, number - 6), min(len(lines), number + 6)
                evidence.append({"question": question, "path": relative, "start": start, "end": end,
                                 "lines": [f"{i}: {lines[i - 1]}" for i in range(start, end + 1)],
                                 "url": f"{url}/blob/{revision}/{quote(relative, safe='/')}#L{start}-L{end}" if url else None})
            write_json(context.output / "evidence.json", evidence)
            context.progress(question=question, captured_contexts=len(evidence))
    except BudgetExpired:
        complete = False
    packet = {"schema": "modulation_evidence_v1", "source_checkout": str(source), "source_commit": revision, "repository_url": url,
              "collection_complete": complete, "search_is_exhaustive": False, "max_hits_per_question": args.max_hits,
              "contexts": len(evidence), "evidence_hash": digest(evidence), "build_match": "unverified",
              "installed_renderer_reference": "Vital 1.6.4; source/build equivalence not established by a commit lookup"}
    write_json(context.output / "source_packet.json", packet)
    answers = {"schema": "modulation_review_v1", "source_commit": revision,
               "version_notes": "", "build_match": "unverified", "build_match_evidence": "",
               "cases": [{"question": question, "prompt": prompt, "status": "unresolved", "answer": "", "citations": []}
                         for question, prompt in QUESTIONS.items()]}
    if not (context.output / "review_answers.json").exists():
        write_json(context.output / "review_answers.json", answers)
    report = (f"# Modulation source evidence packet\n\nCollected {len(evidence)} bounded search contexts from `{revision}`. "
              "**Semantic audit remains unperformed.** Search hits are navigation aids, not conclusions.\n\n"
              "Read evidence.json, follow full call/data flow in the pinned checkout, and complete review_answers.json. "
              "Use source-review to validate citation ranges and produce the final source-backed report. "
              "Neither packet collection nor report formatting verifies the truth of a human-written interpretation.\n\n"
              "Source/build equivalence to installed Vital 1.6.4 remains unverified. Do not apply older source conclusions to newer ramp behavior without evidence. "
              "No renderer or model is imported. Source files are never changed.\n")
    return finish_report(context, "SOURCE_AUDIT_WORKSHEET.md", report,
                         "evidence_collected_review_required" if complete else "budget_exhausted", contexts=len(evidence))


def review(context: Context, args) -> dict:
    packet_path, answers_path = Path(args.packet).resolve(), Path(args.answers).resolve()
    packet, answers = read_json(packet_path), read_json(answers_path)
    if packet["source_commit"] != answers.get("source_commit") or answers.get("schema") != "modulation_review_v1":
        raise ValueError("answers do not match this source evidence packet")
    if answers.get("build_match") not in ("unverified", "verified", "different"):
        raise ValueError("invalid build-match status")
    if answers.get("build_match") != "unverified" and not answers.get("build_match_evidence", "").strip():
        raise ValueError("a build-match assertion requires explicit evidence")
    source, revision = Path(packet["source_checkout"]), packet["source_commit"]
    cases = answers.get("cases", [])
    if len(cases) != len(QUESTIONS) or {case["question"] for case in cases} != set(QUESTIONS):
        raise ValueError("review must cover exactly the five audit questions")
    report = (f"# Modulation slot semantics audit\n\nSource: `{revision}`. Build relationship: {answers['build_match']}.\n\n"
              f"Version boundary: {answers.get('version_notes') or 'Unresolved; no newer-build guarantee is established.'}\n\n"
              f"Build evidence: {answers.get('build_match_evidence') or 'None supplied.'}\n\n")
    resolved = 0
    for case in cases:
        state, answer, citations = case.get("status"), case.get("answer", ""), case.get("citations", [])
        if state not in ("unresolved", "supported", "exception", "version_limited"):
            raise ValueError("unknown review case status")
        if state != "unresolved" and (not answer.strip() or not citations):
            raise ValueError("a resolved case requires an explanation and verified source citations")
        resolved += state != "unresolved"
        report += f"## {case['question']}: {state}\n\n{QUESTIONS[case['question']]}\n\n{answer or 'Unresolved.'}\n\n"
        for citation in citations:
            path, start, end = citation["path"], citation["start"], citation["end"]
            validate_range(path, start, end, 2**31 - 1)
            lines = git(context, source, "show", f"{revision}:{path}").splitlines()
            validate_range(path, start, end, len(lines))
            url = packet["repository_url"]
            target = f"{url}/blob/{revision}/{quote(path, safe='/')}#L{start}-L{end}" if url else None
            report += f"- [{path}:{start}–{end}]({target})\n" if target else f"- `{path}:{start}–{end}` at `{revision}` (no verified remote URL).\n"
        report += "\n"
    report += ("Citation existence/ranges were checked mechanically. Interpretations are reviewer-authored; this tool does not prove them. "
               "Semantic interchangeability, numerical tolerance, and bit identity must remain distinct. No permutation renders were run.\n")
    write_json(context.output / "review_provenance.json", {"packet_sha256": file_hash(packet_path), "answers_sha256": file_hash(answers_path),
               "source_commit": revision, "resolved_questions": resolved, "build_match": answers["build_match"]})
    return finish_report(context, "MODULATION_SLOT_SEMANTICS_AUDIT.md", report,
                         "review_recorded" if resolved == len(QUESTIONS) else "review_partial", resolved_questions=resolved)
