from __future__ import annotations

import argparse
import sqlite3
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from phase2 import PHASE2_ARMS, load_phase2_schedule
from config import (
    ARCHIVE,
    DATA,
    LEDGER_PATH,
    OPENROUTER_LIMIT_WITH_KIMI,
    OPENROUTER_LIMIT_WITHOUT_KIMI,
    RAW_DIR,
    RUNS_DIR,
    SCHEDULE_PATH,
    SEARCH_CACHE_PATH,
    SEARCH_LIMIT,
    SYSTEMS,
    atomic_write,
    canonical_json,
    load_claims,
    load_json,
    pretty_json,
    read_jsonl,
)
from judge import TERMINAL_FAILURE_STATUS, expected_agent_params_hash
from openrouter import DB_PATH
from run_experiments import (
    is_blocked_replay_failure,
    prospective_minimum_searches,
    validate_strict_v3_search_result,
    validate_v5_query_exhausted_result,
)
from schedule import ACTIVE_PILOT_ARMS, DIAGNOSTIC_PILOT_ARMS, pilot_judge_ids
from search import V3_CACHE_TABLE, normalize_query

AUDIT_DIR = DATA / "audits"
PHASES = ("preflight", "pilot", "auto", "full")
TERMINAL_STATUSES = {
    "agent_complete",
    "judged",
    TERMINAL_FAILURE_STATUS,
    "terminal_invalid",
}
LEDGER_STATUSES = {"complete", "failed"}
MANIFEST_FIELDS = (
    "run_id",
    "claim_id",
    "arm",
    "framing",
    "system",
    "model",
    "repeat",
    "search_backend",
    "params_hash",
    "queries",
    "searches",
    "result_urls",
    "report",
    "citations",
    "calls",
    "search_count",
    "started_at",
    "completed_at",
    "status",
)


class Audit:
    def __init__(self, phase: str) -> None:
        self.phase = phase
        self.errors: list[dict] = []
        self.warnings: list[dict] = []
        self.counts: Counter = Counter()

    def error(self, code: str, message: str, run_id: str | None = None) -> None:
        self.errors.append({"code": code, "message": message, "run_id": run_id})

    def warn(self, code: str, message: str, run_id: str | None = None) -> None:
        self.warnings.append({"code": code, "message": message, "run_id": run_id})


def load_schedule(audit: Audit) -> list[dict]:
    if not SCHEDULE_PATH.exists():
        audit.error("schedule_missing", "data/schedule.json does not exist")
        return []
    rows = load_json(SCHEDULE_PATH)
    if not isinstance(rows, list) or not rows:
        audit.error("schedule_invalid", "schedule is not a nonempty list")
        return []
    # Second-round arms live in their own additive file so the registered
    # schedule stays hash-authenticated, but their runs are audited alike.
    rows = list(rows) + load_phase2_schedule()
    run_ids = [str(row.get("run_id")) for row in rows]
    if len(run_ids) != len(set(run_ids)):
        audit.error("schedule_duplicate_ids", "schedule contains duplicate run IDs")
    for row in rows:
        for field in ("run_id", "claim_id", "arm", "framing", "system", "model", "repeat"):
            if row.get(field) in (None, ""):
                audit.error(
                    "schedule_field",
                    f"schedule row is missing {field}",
                    run_id=str(row.get("run_id")),
                )
                break
    active = Counter(
        row["arm"] for row in rows if row.get("arm") in ACTIVE_PILOT_ARMS
    )
    expected_active = {arm: (6 if arm.startswith("kimi") else 40) for arm in ACTIVE_PILOT_ARMS}
    if dict(active) != expected_active:
        audit.error(
            "schedule_active_pilot",
            f"active pilot rows are {dict(active)}, expected {expected_active}",
        )
    return rows


def load_manifest(path: Path, audit: Audit) -> dict | None:
    try:
        manifest = load_json(path)
    except ValueError:
        audit.error("manifest_parse", f"{path.name} is not valid JSON")
        return None
    if not isinstance(manifest, dict):
        audit.error("manifest_type", f"{path.name} is not a JSON object")
        return None
    return manifest


def validate_manifest(manifest: dict, row: dict, audit: Audit) -> None:
    run_id = str(manifest.get("run_id"))
    for field in MANIFEST_FIELDS:
        if field not in manifest:
            audit.error("manifest_field", f"manifest lacks {field}", run_id=run_id)
            return
    for field in ("claim_id", "arm", "framing", "system", "model", "repeat"):
        if manifest.get(field) != row.get(field):
            audit.error(
                "manifest_schedule_mismatch",
                f"{field} differs from the frozen schedule row",
                run_id=run_id,
            )
    status = manifest.get("status")
    if status not in TERMINAL_STATUSES:
        audit.error("manifest_status", f"unknown status {status!r}", run_id=run_id)
        return
    try:
        expected_hash = expected_agent_params_hash(row)
    except Exception as error:
        audit.error("params_contract", f"cannot derive params hash: {error}", run_id=run_id)
        return
    if manifest.get("params_hash") != expected_hash:
        audit.error("params_hash", "params hash differs from the frozen contract", run_id=run_id)

    if status == TERMINAL_FAILURE_STATUS:
        if not manifest.get("failure"):
            audit.error("failure_record", "terminal failure lacks a failure record", run_id=run_id)
        return

    queries = manifest.get("queries")
    searches = manifest.get("searches")
    if not isinstance(queries, list) or not isinstance(searches, list):
        audit.error("search_lists", "queries or searches is not a list", run_id=run_id)
        return
    search_count = manifest.get("search_count")
    is_replay = row.get("search_backend") is None
    if is_replay:
        if queries or manifest.get("search_backend") != "fixed_replay":
            audit.error("replay_shape", "replay run carries live search state", run_id=run_id)
    else:
        minimum = prospective_minimum_searches(row)
        if manifest.get("minimum_searches") != minimum:
            audit.error("minimum_searches", "minimum searches differs from protocol", run_id=run_id)
        if (
            not isinstance(search_count, int)
            or isinstance(search_count, bool)
            or not minimum <= search_count <= SEARCH_LIMIT
        ):
            audit.error("search_count", f"search count {search_count!r} out of range", run_id=run_id)
            return
        if len(queries) != search_count or len(searches) != search_count:
            audit.error("search_lengths", "query and search lists do not match the count", run_id=run_id)
            return
        backend = str(row.get("search_backend"))
        for query, search in zip(queries, searches):
            try:
                validate_strict_v3_search_result(search, query, backend)
            except (RuntimeError, TypeError) as error:
                audit.error("search_record", str(error), run_id=run_id)
                break
        for item in manifest.get("query_exhaustions", []):
            try:
                validate_v5_query_exhausted_result(item, item.get("query"), backend)
            except (RuntimeError, TypeError) as error:
                audit.error("exhaustion_record", str(error), run_id=run_id)
                break
        tool_calls = manifest.get("search_tool_call_count", search_count)
        if (
            not isinstance(tool_calls, int)
            or isinstance(tool_calls, bool)
            or not search_count <= tool_calls <= SEARCH_LIMIT
        ):
            audit.error("tool_call_count", f"tool call count {tool_calls!r} out of range", run_id=run_id)
        exhaustions = manifest.get("query_exhaustions", [])
        if manifest.get("query_exhaustion_count", len(exhaustions)) != len(exhaustions):
            audit.error("exhaustion_count", "exhaustion count differs from its list", run_id=run_id)
        if len(manifest.get("search_tool_queries", queries)) != tool_calls:
            audit.error("issued_queries", "issued query list does not match tool calls", run_id=run_id)

    report = manifest.get("report")
    if not isinstance(report, str) or not report.strip():
        audit.error("report_empty", "completed run has an empty report", run_id=run_id)
    result_urls = manifest.get("result_urls")
    citations = manifest.get("citations")
    if not isinstance(citations, list) or not isinstance(result_urls, list):
        audit.error("citation_lists", "citations or result URLs is not a list", run_id=run_id)
    else:
        retrieved = set(result_urls)
        membership = all(url in retrieved for url in citations)
        recorded = (manifest.get("judge_labels") or {}).get("citation_presence")
        if isinstance(recorded, dict) and recorded.get("passes") is True and not membership:
            audit.error(
                "citation_source",
                "recorded citation record claims membership the URLs do not support",
                run_id=run_id,
            )

    calls = manifest.get("calls")
    if not isinstance(calls, list) or not calls:
        audit.error("call_rows", "manifest has no call rows", run_id=run_id)
        return
    for call in calls:
        if not isinstance(call, dict) or not call.get("generation_id"):
            audit.error("call_generation", "a call row lacks generation metadata", run_id=run_id)
            break

    if status == "judged":
        labels = manifest.get("judge_labels")
        if not isinstance(labels, dict) or "conclusion" not in labels:
            audit.error("judge_labels", "judged run lacks a conclusion label", run_id=run_id)
        query_labels = manifest.get("query_labels")
        if not isinstance(query_labels, list):
            audit.error("query_labels", "judged run lacks query labels", run_id=run_id)


def validate_raw(manifest: dict, audit: Audit) -> None:
    run_id = str(manifest.get("run_id"))
    path = RAW_DIR / str(manifest.get("arm")) / f"{run_id}.jsonl"
    if not path.exists():
        audit.error("raw_missing", "raw archive file is missing", run_id=run_id)
        return
    try:
        events = read_jsonl(path)
    except (ValueError, TypeError) as error:
        audit.error("raw_parse", str(error), run_id=run_id)
        return
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if line != canonical_json(events[number - 1]) + "\n":
                audit.error(
                    "raw_roundtrip",
                    f"line {number} does not round-trip canonically",
                    run_id=run_id,
                )
                return
    responses = [event for event in events if event.get("event") == "model_response"]
    reconciled = any(
        event.get("event") == "transport_reconciliation" for event in events
    )
    blocked_replay = any(is_blocked_replay_failure(event) for event in events)
    # A provider that refuses a request produces no response, so the recorded
    # trajectory failure is the whole story for that run.
    refused = any(event.get("event") == "trajectory_failure" for event in events)
    if not responses and not (
        (reconciled or blocked_replay or refused)
        and manifest.get("status") == "terminal_failed"
    ):
        audit.error("raw_no_response", "raw archive has no model response", run_id=run_id)
        return
    for event in responses:
        if not event.get("request_body") or not event.get("response_body"):
            audit.error("raw_wire", "a model response lacks request or response body", run_id=run_id)
            return
    manifest_calls = manifest.get("calls")
    if isinstance(manifest_calls, list):
        raw_indexes = {event.get("call_index") for event in responses}
        wanted = {call.get("call_index") for call in manifest_calls if isinstance(call, dict)}
        if not wanted <= raw_indexes:
            audit.error("raw_call_coverage", "manifest calls lack raw responses", run_id=run_id)
    for event in events:
        if event.get("event") != "tool_results":
            continue
        for item in event.get("tool_results", []):
            if not isinstance(item, dict) or not item.get("tool_call_id") or "result" not in item:
                audit.error("raw_tool_result", "a tool result lacks linkage or payload", run_id=run_id)
                return


def validate_cache_exclusion(manifests: list[dict], audit: Audit) -> None:
    exhausted: list[tuple[str, str]] = []
    for manifest in manifests:
        for item in manifest.get("query_exhaustions", []):
            query = item.get("query")
            if isinstance(query, str):
                exhausted.append((str(manifest.get("run_id")), query))
    if not exhausted or not SEARCH_CACHE_PATH.exists():
        return
    connection = sqlite3.connect(SEARCH_CACHE_PATH)
    try:
        for run_id, query in exhausted:
            found = connection.execute(
                f"SELECT 1 FROM {V3_CACHE_TABLE} WHERE logical_backend='web' AND query_key=?",
                (normalize_query(query),),
            ).fetchone()
            if found:
                audit.error(
                    "exhausted_query_cached",
                    f"exhausted query was inserted into the cache: {query!r}",
                    run_id=run_id,
                )
    finally:
        connection.close()


def validate_ledger(audit: Audit, active_ids: set[str]) -> dict:
    if not DB_PATH.exists():
        audit.error("ledger_missing", "call ledger database is missing")
        return {}
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    try:
        rows = [dict(row) for row in connection.execute("SELECT * FROM calls")]
        attempts = {
            (row["run_id"], row["call_index"])
            for row in connection.execute(
                "SELECT DISTINCT run_id, call_index FROM wire_attempts"
            )
        }
    finally:
        connection.close()
    spend = 0.0
    upstream = 0.0
    for row in rows:
        key = f"{row['run_id']}#{row['call_index']}"
        if row["status"] not in LEDGER_STATUSES:
            audit.error("ledger_status", f"{key} has active status {row['status']!r}")
        if row["budget_breach"]:
            audit.error("ledger_breach", f"{key} recorded a budget breach")
        spend += float(row["openrouter_cost"] or 0)
        upstream += float(row["upstream_cost"] or 0)
        if row["status"] != "complete":
            continue
        if not row["generation_id"]:
            audit.error("ledger_generation", f"{key} is complete without a generation ID")
        if row["run_id"] in active_ids and (row["run_id"], row["call_index"]) not in attempts:
            audit.error("ledger_wire", f"{key} has no wire attempt record")
        if (
            row["run_id"] in active_ids
            and row["model"] == SYSTEMS["gemini"]
            and not row["is_byok"]
        ):
            audit.error("ledger_byok", f"{key} ran Gemini outside BYOK routing")
    if spend > OPENROUTER_LIMIT_WITH_KIMI:
        audit.error("budget_cap", f"OpenRouter spend ${spend:.4f} exceeds the hard cap")
    elif spend > OPENROUTER_LIMIT_WITHOUT_KIMI:
        audit.warn("budget_kimi_band", f"spend ${spend:.4f} requires the Kimi cap decision")
    if LEDGER_PATH.exists():
        csv_lines = LEDGER_PATH.read_text(encoding="utf-8").strip().splitlines()
        if len(csv_lines) - 1 != len(rows):
            audit.error(
                "ledger_csv_rows",
                f"ledger.csv has {len(csv_lines) - 1} rows, sqlite has {len(rows)}",
            )
    else:
        audit.error("ledger_csv_missing", "data/ledger.csv is missing")
    by_status = Counter(row["status"] for row in rows)
    return {
        "calls": len(rows),
        "by_status": dict(by_status),
        "openrouter_spend": round(spend, 10),
        "upstream_spend": round(upstream, 10),
    }


def validate_preflight(audit: Audit) -> None:
    preflight_path = DATA / "preflight_v5.json"
    byok_path = ARCHIVE / "byok_verification_v5.json"
    if not preflight_path.exists():
        audit.error("preflight_missing", "data/preflight_v5.json does not exist")
        return
    record = load_json(preflight_path)
    if not isinstance(record, dict) or record.get("status") != "passed":
        audit.error("preflight_status", "the v5 preflight record did not pass")
        return
    calls = record.get("calls")
    if not isinstance(calls, list) or len(calls) != 6:
        audit.error("preflight_calls", "the v5 preflight record must hold six calls")
        return
    for call in calls:
        if call.get("reasoning_present") is not True:
            audit.error("preflight_reasoning", f"{call.get('run_id')} lacks reasoning evidence")
        if call.get("tool_contract_valid") is not True:
            audit.error("preflight_tool", f"{call.get('run_id')} failed the tool contract")
    if not byok_path.exists():
        audit.error("byok_missing", "archive/byok_verification_v5.json does not exist")
        return
    byok = load_json(byok_path)
    gemini_calls = byok.get("calls") if isinstance(byok, dict) else None
    if not isinstance(gemini_calls, list) or len(gemini_calls) != 3:
        audit.error("byok_calls", "BYOK verification must hold the three Gemini calls")
        return
    for call in gemini_calls:
        summary = call.get("call") or {}
        if not summary.get("is_byok") or summary.get("provider") != "Google AI Studio":
            audit.error("byok_route", f"{call.get('run_id')} did not prove BYOK routing")


def run_audit(phase: str) -> dict:
    audit = Audit(phase)
    claims_ok = True
    try:
        load_claims()
    except (OSError, TypeError, ValueError) as error:
        claims_ok = False
        audit.error("claims", str(error))
    rows = load_schedule(audit)
    rows_by_id = {str(row.get("run_id")): row for row in rows}
    sealed_arms = set(DIAGNOSTIC_PILOT_ARMS)
    active_ids = {
        run_id for run_id, row in rows_by_id.items() if row.get("arm") not in sealed_arms
    }

    if phase in ("preflight", "pilot", "full"):
        validate_preflight(audit)

    manifests = []
    for path in sorted(RUNS_DIR.glob("*.json")) if RUNS_DIR.exists() else []:
        manifest = load_manifest(path, audit)
        if manifest is None:
            continue
        run_id = str(manifest.get("run_id"))
        row = rows_by_id.get(run_id)
        if row is None:
            audit.error("manifest_unscheduled", "manifest has no schedule row", run_id=run_id)
            continue
        if row.get("arm") in sealed_arms:
            audit.counts["sealed_manifests"] += 1
            continue
        manifests.append((row, manifest))

    for row, manifest in manifests:
        validate_manifest(manifest, row, audit)
        validate_raw(manifest, audit)
        audit.counts[f"status_{manifest.get('status')}"] += 1
    validate_cache_exclusion([manifest for _, manifest in manifests], audit)
    budget = validate_ledger(audit, active_ids)

    if phase == "pilot" and claims_ok:
        have = {str(manifest.get("run_id")): manifest for _, manifest in manifests}
        pilot_rows = [row for row in rows if row.get("arm") in ACTIVE_PILOT_ARMS]
        # A passed gate record means the pilots ran and were accepted, after which
        # their manifests were retired. Without one, an absent pilot run is a run
        # that never happened and the phase must still fail closed.
        gate_path = DATA / "pilot_gate.json"
        gate = load_json(gate_path) if gate_path.exists() else None
        accepted = isinstance(gate, dict) and gate.get("all_passed") is True
        retired = accepted and not any(str(row.get("run_id")) in have for row in pilot_rows)
        if retired:
            audit.counts["pilot_cohort_retired"] += 1
            pilot_rows = []
        for row in pilot_rows:
            manifest = have.get(str(row.get("run_id")))
            if manifest is None:
                audit.error("pilot_incomplete", "pilot run has no manifest", run_id=str(row.get("run_id")))
            elif manifest.get("status") not in TERMINAL_STATUSES:
                audit.error("pilot_nonterminal", "pilot run is not terminal", run_id=str(row.get("run_id")))
        for run_id in [] if retired else pilot_judge_ids(rows):
            manifest = have.get(run_id)
            resolved = manifest is not None and (
                manifest.get("status") in {"judged", "terminal_failed"}
                or manifest.get("judge_status") == "terminal_invalid"
            )
            if not resolved:
                audit.error("pilot_judging", "pilot sample run has no judging outcome", run_id=run_id)

    if phase == "full" and claims_ok:
        have = {str(manifest.get("run_id")): manifest for _, manifest in manifests}
        for run_id, row in rows_by_id.items():
            if row.get("arm") in sealed_arms or row.get("arm") in ACTIVE_PILOT_ARMS:
                continue
            # Second-round arms are opportunistic: they are registered so their
            # rows are auditable, but their coverage is reported rather than
            # required, so a planned row without a manifest is not an error.
            if row.get("arm") in PHASE2_ARMS:
                continue
            manifest = have.get(run_id)
            if manifest is None:
                audit.error("full_incomplete", "scheduled run has no manifest", run_id=run_id)
            elif manifest.get("status") not in TERMINAL_STATUSES:
                audit.error("full_nonterminal", "scheduled run is not terminal", run_id=run_id)

    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return {
        "phase": phase,
        "generated_at": generated_at,
        "status": "green" if not audit.errors else "red",
        "errors": audit.errors,
        "warnings": audit.warnings,
        "counts": dict(audit.counts),
        "manifests_checked": len(manifests),
        "budget": budget,
    }


def write_outputs(result: dict) -> Path:
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = result["generated_at"].replace(":", "").replace("-", "").replace("+0000", "Z")
    path = AUDIT_DIR / f"audit_{stamp}.json"
    atomic_write(path, pretty_json(result))
    atomic_write(AUDIT_DIR / "latest.json", pretty_json(result))
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=PHASES, default="auto")
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()
    result = run_audit(args.phase)
    if not args.no_write:
        write_outputs(result)
    print(
        f"audit {result['phase']}: {result['status']}, "
        f"{result['manifests_checked']} manifests, "
        f"{len(result['errors'])} errors, {len(result['warnings'])} warnings"
    )
    for issue in result["errors"][:20]:
        run_part = f" [{issue['run_id']}]" if issue.get("run_id") else ""
        print(f"  {issue['code']}{run_part}: {issue['message']}")
    if len(result["errors"]) > 20:
        print(f"  ... and {len(result['errors']) - 20} more")
    if result["status"] != "green":
        sys.exit(1)


if __name__ == "__main__":
    main()
