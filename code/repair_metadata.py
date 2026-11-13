from __future__ import annotations

import argparse
import asyncio
import base64
import binascii
import fcntl
import json
import math
import os
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from config import (
    ARCHIVE,
    DATA,
    JUDGES,
    LEDGER_PATH,
    MODEL_CONCURRENCY,
    OPENROUTER_LIMIT_WITHOUT_KIMI,
    OPENROUTER_LIMIT_WITH_KIMI,
    RAW_DIR,
    SCHEDULE_PATH,
    SEARCH_LIMIT,
    append_jsonl,
    canonical_json,
    load_json,
    now_iso,
    pretty_json,
    read_jsonl,
    sha256_bytes,
)
from openrouter import (
    RESERVATION_POLICY,
    V3_RESERVATION_POLICY,
    Ledger,
    OpenRouterClient,
    generation_cost,
    reservation_cost,
    reservation_provenance,
    usage_fields,
)
from schedule import ACTIVE_PILOT_ARMS, DIAGNOSTIC_PILOT_ARMS, PILOT_V3_ARMS


PENDING_STATUSES = ("response_received", "metadata_pending")
ACTIVE_STATUSES = (
    "inflight",
    "response_received",
    "metadata_pending",
    "unknown_inflight",
)
RECOVERY_RECORD_SCHEMA = 1
JUDGE_CALLS = {
    "judge_conclusion_gemini": (101, JUDGES["gemini"]),
    "judge_queries_gemini": (102, JUDGES["gemini"]),
    "judge_conclusion_luna": (201, JUDGES["luna"]),
    "judge_queries_luna": (203, JUDGES["luna"]),
}
# The stance instrument issues one call per search and repeats a sweep in its
# own index band, so its calls are identified by band rather than a fixed index.
STANCE_CALL_KIND = "judge_evidence_stance_gemini"
STANCE_INDEX_FLOOR = 301
# Post-hoc measurement calls stand alone: one labels a single search, the other
# relabels a query list. A refusal on either does not close the trajectory the
# way an agent or primary-judge failure does.
INDEPENDENT_MEASUREMENT_KINDS = frozenset({STANCE_CALL_KIND, "judge_queries_luna"})


def judge_call_identity_valid(call_kind: str, call_index: int, model: str) -> bool:
    if call_kind == STANCE_CALL_KIND:
        return call_index >= STANCE_INDEX_FLOOR and model == JUDGES["gemini"]
    return JUDGE_CALLS.get(call_kind) == (call_index, model)


def require_repairable_selection(
    schedule: dict[str, dict], selected_run_ids: tuple[str, ...]
) -> None:
    missing = sorted(set(selected_run_ids) - set(schedule))
    if missing:
        raise RuntimeError(
            f"selected recovery IDs are absent from the frozen schedule: {missing[:8]}"
        )
    arms = {str(schedule[run_id].get("arm") or "") for run_id in selected_run_ids}
    sealed = sorted(arms & set(DIAGNOSTIC_PILOT_ARMS))
    if sealed:
        raise RuntimeError(
            "sealed diagnostic cohorts cannot be repaired: " + ", ".join(sealed)
        )


def recovery_preflight_anchor(ledger_path: Path) -> dict:
    preflight_path = DATA / "preflight_v5.json"
    if not preflight_path.exists():
        raise RuntimeError("metadata repair requires the passed v5 preflight record")
    record = load_json(preflight_path)
    if not isinstance(record, dict) or record.get("status") != "passed":
        raise RuntimeError("metadata repair requires the passed v5 preflight record")
    return {
        "kind": "pilot_v7_recovery_anchor",
        "pilot_cohort": "v7",
        "preflight_path": "data/preflight_v5.json",
        "preflight_completed_at": record.get("completed_at"),
        "ledger_path": str(ledger_path),
    }


@dataclass(frozen=True)
class PendingCall:
    run_id: str
    call_index: int
    generation_id: str


@dataclass(frozen=True)
class RawResponse:
    path: Path
    payload: dict
    prior_repair: dict | None
    event: dict | None = None


@dataclass(frozen=True)
class RepairSummary:
    pending: int
    repaired: int
    recovered_from_log: int
    failed: tuple[str, ...]
    remaining: tuple[str, ...]

    @property
    def exit_code(self) -> int:
        return 0 if not self.failed and not self.remaining else 1


@dataclass(frozen=True)
class RecoveryPlan:
    calls: tuple[PendingCall, ...]
    selected_run_ids: tuple[str, ...]
    active_rows: tuple[dict, ...]
    raw_before_prefixes: dict[str, dict]
    evidence: dict[tuple[str, int], RawResponse]
    preflight_anchor: dict


async def pending_calls(ledger: Ledger) -> list[PendingCall]:
    placeholders = ",".join("?" for _ in PENDING_STATUSES)
    async with ledger.lock:
        with ledger.connect() as connection:
            rows = connection.execute(
                f"""SELECT run_id,call_index,generation_id
                    FROM calls WHERE status IN ({placeholders})
                    ORDER BY seq""",
                PENDING_STATUSES,
            ).fetchall()
    return [
        PendingCall(
            run_id=str(row["run_id"]),
            call_index=int(row["call_index"]),
            generation_id=str(row["generation_id"] or ""),
        )
        for row in rows
    ]


def call_name(call: PendingCall) -> str:
    return f"{call.run_id}/{call.call_index}"


def _event_matches(event: dict, call: PendingCall) -> bool:
    return event.get("run_id") == call.run_id and event.get("call_index") == call.call_index


def locate_raw_responses(
    calls: list[PendingCall], raw_dir: Path
) -> tuple[dict[tuple[str, int], RawResponse], dict[tuple[str, int], str]]:
    files_by_run: dict[str, list[Path]] = defaultdict(list)
    wanted_runs = {call.run_id for call in calls}
    if raw_dir.exists():
        for path in raw_dir.rglob("*.jsonl"):
            if path.stem in wanted_runs:
                files_by_run[path.stem].append(path)

    located: dict[tuple[str, int], RawResponse] = {}
    errors: dict[tuple[str, int], str] = {}
    events_by_path: dict[Path, list[dict]] = {}
    path_errors: dict[Path, str] = {}
    for paths in files_by_run.values():
        for path in paths:
            try:
                events_by_path[path] = read_jsonl(path)
            except (OSError, TypeError, ValueError) as error:
                path_errors[path] = f"cannot read {path}: {error}"

    for call in calls:
        key = (call.run_id, call.call_index)
        paths = sorted(files_by_run.get(call.run_id, []))
        if not call.generation_id:
            errors[key] = "ledger row has no generation ID"
            continue
        unreadable = [path_errors[path] for path in paths if path in path_errors]
        if unreadable:
            errors[key] = "; ".join(unreadable)
            continue
        matches: list[tuple[Path, dict]] = []
        repairs: list[dict] = []
        for path in paths:
            for event in events_by_path[path]:
                if not _event_matches(event, call):
                    continue
                if event.get("event") == "model_response":
                    matches.append((path, event))
                elif (
                    event.get("event") == "generation_metadata_repair"
                    and event.get("generation_id") == call.generation_id
                    and isinstance(event.get("generation"), dict)
                    and event["generation"]
                    and event.get("error") is None
                ):
                    repairs.append(event)
        if len(matches) != 1:
            errors[key] = f"expected one raw model_response, found {len(matches)}"
            continue
        path, response = matches[0]
        payload = response.get("response_body")
        if not isinstance(payload, dict):
            errors[key] = "raw model_response has no response_body object"
            continue
        raw_generation_id = str(payload.get("id") or "")
        if raw_generation_id != call.generation_id:
            errors[key] = (
                f"generation ID mismatch: ledger={call.generation_id!r}, "
                f"raw={raw_generation_id!r}"
            )
            continue
        prior_repair = repairs[-1] if repairs else None
        located[key] = RawResponse(path=path, payload=payload, prior_repair=prior_repair)
    return located, errors


def _float_equal(left: Any, right: Any) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return False
    try:
        # The generation endpoint truncates costs to about nine decimal
        # places, so exact equality against response usage is impossible.
        return math.isclose(
            float(left), float(right), rel_tol=1e-5, abs_tol=1e-9
        )
    except (TypeError, ValueError):
        return False


def _wire_bytes(value: Any, label: str) -> bytes:
    if not isinstance(value, dict) or set(value) != {
        "base64",
        "sha256",
        "size_bytes",
    }:
        raise RuntimeError(f"{label} wire record is invalid")
    encoded = value.get("base64")
    digest = value.get("sha256")
    size = value.get("size_bytes")
    if (
        not isinstance(encoded, str)
        or not isinstance(digest, str)
        or isinstance(size, bool)
        or not isinstance(size, int)
        or size < 0
    ):
        raise RuntimeError(f"{label} wire record has invalid fields")
    try:
        decoded = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as error:
        raise RuntimeError(f"{label} wire base64 is invalid") from error
    if len(decoded) != size or sha256_bytes(decoded) != digest:
        raise RuntimeError(f"{label} wire size/hash mismatch")
    return decoded


def _generation_id(payload: dict) -> str:
    for field in ("id", "generation_id", "generationId"):
        value = str(payload.get(field) or "").strip()
        if value:
            return value
    return ""


def validate_generation_metadata(
    generation: Any,
    call: PendingCall,
    payload: dict,
    model: str,
) -> None:
    if not isinstance(generation, dict) or not generation:
        raise RuntimeError("generation endpoint returned no metadata")
    if str(generation.get("id") or "") != call.generation_id:
        raise RuntimeError("generation metadata ID does not match the received response")
    generation_model = str(generation.get("model") or "")
    if not (
        generation_model == model
        or generation_model.startswith(f"{model}-")
        or (
            model.endswith(":exacto")
            and (
                generation_model == model.removesuffix(":exacto")
                or generation_model.startswith(f"{model.removesuffix(':exacto')}-")
            )
        )
    ):
        raise RuntimeError("generation metadata model differs from the frozen model")
    provider = str(
        generation.get("provider_name") or generation.get("provider") or ""
    ).strip()
    if not provider:
        raise RuntimeError("generation metadata lacks provider identity")
    usage = usage_fields(payload)
    generation_byok = generation.get("is_byok")
    if not isinstance(generation_byok, bool) or generation_byok is not usage["is_byok"]:
        raise RuntimeError("generation metadata BYOK state differs from response usage")
    total_cost = generation.get("total_cost")
    upstream_cost = generation.get("upstream_inference_cost")
    if model == "google/gemini-3.6-flash":
        if provider != "Google AI Studio" or generation_byok is not True:
            raise RuntimeError("Gemini recovery is not Google AI Studio BYOK")
        if not _float_equal(total_cost, 0) or not (
            _float_equal(upstream_cost, usage["upstream_cost"])
            and float(upstream_cost) > 0
        ):
            raise RuntimeError("Gemini generation cost metadata is inconsistent")
    else:
        if generation_byok is not False:
            raise RuntimeError("non-Gemini recovery unexpectedly reports BYOK")
        if not _float_equal(total_cost, usage["openrouter_cost"]) or float(
            total_cost or 0
        ) <= 0:
            raise RuntimeError("generation total cost differs from response usage")


def _project_active_row(row: Any) -> dict:
    return {
        "run_id": str(row["run_id"]),
        "call_index": int(row["call_index"]),
        "call_kind": str(row["call_kind"]),
        "model": str(row["model"]),
        "status": str(row["status"]),
        "reserved_cost": float(row["reserved_cost"]),
        "hard_cap": float(row["hard_cap"]),
        "reservation_policy": str(row["reservation_policy"]),
        "generation_id": str(row["generation_id"] or ""),
        "budget_breach": int(row["budget_breach"]),
        "input_tokens": int(row["input_tokens"]),
        "output_tokens": int(row["output_tokens"]),
        "reasoning_tokens": int(row["reasoning_tokens"]),
        "native_cost": float(row["native_cost"]),
        "openrouter_cost": float(row["openrouter_cost"]),
        "upstream_cost": float(row["upstream_cost"]),
        "is_byok": int(row["is_byok"]),
    }


async def active_rows(ledger: Ledger) -> list[dict]:
    placeholders = ",".join("?" for _ in ACTIVE_STATUSES)
    async with ledger.lock:
        with ledger.connect() as connection:
            rows = connection.execute(
                f"""SELECT run_id,call_index,call_kind,model,status,reserved_cost,
                           hard_cap,reservation_policy,generation_id,budget_breach,
                           input_tokens,output_tokens,reasoning_tokens,native_cost,
                           openrouter_cost,upstream_cost,is_byok
                    FROM calls WHERE status IN ({placeholders})
                    ORDER BY run_id,call_index""",
                ACTIVE_STATUSES,
            ).fetchall()
    return [_project_active_row(row) for row in rows]


def _load_schedule(path: Path) -> dict[str, dict]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot read frozen schedule: {error}") from error
    if not isinstance(value, list):
        raise RuntimeError("frozen schedule is not a list")
    rows: dict[str, dict] = {}
    for row in value:
        run_id = str(row.get("run_id") or "") if isinstance(row, dict) else ""
        if not run_id or run_id in rows:
            raise RuntimeError("frozen schedule has a missing or duplicate run ID")
        rows[run_id] = row
    return rows


def _relative_raw_path(path: Path, raw_dir: Path) -> str:
    try:
        relative = path.resolve().relative_to(raw_dir.resolve())
    except ValueError as error:
        raise RuntimeError("metadata recovery raw path is outside archive/raw") from error
    if relative.suffix != ".jsonl" or not relative.parts:
        raise RuntimeError("metadata recovery raw path is not canonical JSONL")
    return relative.as_posix()


def _raw_prefix(path: Path, run_id: str) -> dict:
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise RuntimeError(f"{run_id}: raw log is unreadable: {error}") from error
    if not payload or not payload.endswith(b"\n"):
        raise RuntimeError(f"{run_id}: raw log is not a complete JSONL prefix")
    return {
        "run_id": run_id,
        "size_bytes": len(payload),
        "sha256": sha256_bytes(payload),
        "line_count": payload.count(b"\n"),
    }


def _collect_raw_prefixes(
    raw_dir: Path,
    schedule: dict[str, dict],
    run_ids: list[str] | tuple[str, ...],
) -> dict[str, dict]:
    prefixes: dict[str, dict] = {}
    for run_id in sorted(run_ids):
        scheduled = schedule.get(run_id)
        if not isinstance(scheduled, dict) or not isinstance(scheduled.get("arm"), str):
            raise RuntimeError(f"{run_id}: recovery raw path is outside the schedule")
        expected_path = raw_dir / scheduled["arm"] / f"{run_id}.jsonl"
        paths = sorted(raw_dir.rglob(f"{run_id}.jsonl")) if raw_dir.exists() else []
        if paths != [expected_path]:
            raise RuntimeError(f"{run_id}: recovery raw path differs from frozen schedule")
        prefixes[_relative_raw_path(expected_path, raw_dir)] = _raw_prefix(
            expected_path, run_id
        )
    return dict(sorted(prefixes.items()))


def _all_call_rows(ledger: Ledger, run_ids: tuple[str, ...]) -> dict[str, dict[int, dict]]:
    placeholders = ",".join("?" for _ in run_ids)
    with ledger.connect() as connection:
        rows = connection.execute(
            f"""SELECT run_id,call_index,call_kind,model,status,reserved_cost,
                       hard_cap,reservation_policy,generation_id,budget_breach,
                       input_tokens,output_tokens,reasoning_tokens,native_cost,
                       openrouter_cost,upstream_cost,is_byok
                FROM calls WHERE run_id IN ({placeholders})
                ORDER BY run_id,call_index""",
            run_ids,
        ).fetchall()
    grouped: dict[str, dict[int, dict]] = defaultdict(dict)
    for row in rows:
        grouped[str(row["run_id"])][int(row["call_index"])] = dict(row)
    return grouped


def _wire_attempts(ledger: Ledger, run_id: str, call_index: int) -> list[dict]:
    with ledger.connect() as connection:
        rows = connection.execute(
            """SELECT attempt,request_sha256,response_sha256,response_status,
                      generation_id,error
               FROM wire_attempts WHERE run_id=? AND call_index=? ORDER BY attempt""",
            (run_id, call_index),
        ).fetchall()
    return [dict(row) for row in rows]


def _validate_pending_settlement(active: dict, payload: dict) -> None:
    run_call = f"{active['run_id']}/{active['call_index']}"
    if active.get("budget_breach") != 0:
        raise RuntimeError(f"{run_call}: pending row latched the budget hard stop")
    settlement_fields = {
        "input_tokens": 0,
        "output_tokens": 0,
        "reasoning_tokens": 0,
        "native_cost": 0.0,
        "openrouter_cost": 0.0,
        "upstream_cost": 0.0,
        "is_byok": 0,
    }
    if active.get("status") == "metadata_pending":
        try:
            usage = usage_fields(payload)
        except (TypeError, ValueError, OverflowError) as error:
            raise RuntimeError(
                f"{run_call}: raw usage cannot bind pending settlement: {error}"
            ) from error
        if any(
            usage[field] < 0
            for field in ("input_tokens", "output_tokens", "reasoning_tokens")
        ) or any(
            usage[field] < 0
            for field in ("openrouter_cost", "upstream_cost")
        ):
            raise RuntimeError(f"{run_call}: raw usage contains negative values")
        settlement_fields = {
            "input_tokens": usage["input_tokens"],
            "output_tokens": usage["output_tokens"],
            "reasoning_tokens": usage["reasoning_tokens"],
            "native_cost": generation_cost({}, usage),
            "openrouter_cost": usage["openrouter_cost"],
            "upstream_cost": usage["upstream_cost"],
            "is_byok": int(usage["is_byok"]),
        }
    for field, expected in settlement_fields.items():
        observed = active.get(field)
        matches = (
            _float_equal(observed, expected)
            if isinstance(expected, float)
            else observed == expected
        )
        if not matches:
            raise RuntimeError(
                f"{run_call}: pending settlement metadata differs: {field}"
            )


def _validate_raw_call(
    ledger: Ledger,
    active: dict,
    schedule_row: dict,
    events: list[dict],
    path: Path,
    ledger_calls: dict[int, dict],
) -> RawResponse:
    run_id = active["run_id"]
    call_index = active["call_index"]
    for event in events:
        if event.get("run_id") != run_id:
            raise RuntimeError(f"{run_id}: raw event run ID mismatch")
        event_name = event.get("event")
        event_index = event.get("call_index")
        if event_name in {
            "model_call",
            "model_response",
            "generation_metadata",
            "generation_metadata_repair",
        } and (
            isinstance(event_index, bool) or not isinstance(event_index, int)
        ):
            raise RuntimeError(f"{run_id}: raw provider event has invalid call index")
        if isinstance(event_index, int) and event_index not in ledger_calls:
            raise RuntimeError(
                f"{run_id}/{event_index}: raw call evidence has no ledger row"
            )
    responses = [
        event
        for event in events
        if event.get("event") == "model_response"
        and event.get("call_index") == call_index
    ]
    if len(responses) != 1:
        raise RuntimeError(
            f"{run_id}/{call_index}: expected one raw model_response, found {len(responses)}"
        )
    response = responses[0]
    if (
        response.get("call_kind") != active.get("call_kind")
        or response.get("response_status") != 200
        or response.get("terminal") is not True
    ):
        raise RuntimeError(f"{run_id}/{call_index}: raw response envelope is invalid")
    request_body = response.get("request_body")
    payload = response.get("response_body")
    if not isinstance(request_body, dict) or not isinstance(payload, dict):
        raise RuntimeError(f"{run_id}/{call_index}: raw request/response body is missing")
    request_wire = _wire_bytes(response.get("request_wire"), "request")
    if request_wire != canonical_json(request_body).encode("utf-8"):
        raise RuntimeError(f"{run_id}/{call_index}: request body/wire mismatch")
    response_wire = _wire_bytes(response.get("response_wire"), "response")
    try:
        decoded_response = json.loads(response_wire.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"{run_id}/{call_index}: response wire is not JSON") from error
    if decoded_response != payload:
        raise RuntimeError(f"{run_id}/{call_index}: response body/wire mismatch")
    _validate_pending_settlement(active, payload)
    if request_body.get("model") != active.get("model"):
        raise RuntimeError(f"{run_id}/{call_index}: raw request model mismatch")
    if _generation_id(payload) != active["generation_id"]:
        raise RuntimeError(f"{run_id}/{call_index}: generation ID mismatch")
    attempt = response.get("attempt")
    if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt < 1:
        raise RuntimeError(f"{run_id}/{call_index}: raw response attempt is invalid")
    attempts = _wire_attempts(ledger, run_id, call_index)
    matching = [row for row in attempts if int(row["attempt"]) == attempt]
    if len(matching) != 1 or int(attempts[-1]["attempt"] if attempts else 0) != attempt:
        raise RuntimeError(f"{run_id}/{call_index}: wire-attempt identity mismatch")
    wire_attempt = matching[0]
    if (
        wire_attempt.get("request_sha256") != response["request_wire"]["sha256"]
        or wire_attempt.get("response_sha256") != response["response_wire"]["sha256"]
        or wire_attempt.get("response_status") != 200
        or str(wire_attempt.get("generation_id") or "") != active["generation_id"]
        or wire_attempt.get("error") is not None
    ):
        raise RuntimeError(f"{run_id}/{call_index}: ledger/raw wire attempt mismatch")
    reservation = response.get("budget_reservation")
    if not isinstance(reservation, dict):
        raise RuntimeError(f"{run_id}/{call_index}: raw reservation evidence is missing")
    try:
        expected_reservation = reservation_provenance(
            request_body,
            active["hard_cap"] == OPENROUTER_LIMIT_WITH_KIMI,
            hard_cap=active["hard_cap"],
            policy=active["reservation_policy"],
        )
    except ValueError as error:
        raise RuntimeError(
            f"{run_id}/{call_index}: reservation identity mismatch"
        ) from error
    if (
        reservation != expected_reservation
        or reservation.get("policy") != active["reservation_policy"]
        or reservation.get("request_sha256") != response["request_wire"]["sha256"]
        or not _float_equal(reservation.get("reserved_cost"), active["reserved_cost"])
        or not _float_equal(reservation_cost(request_body), active["reserved_cost"])
        or not _float_equal(reservation.get("hard_cap"), active["hard_cap"])
    ):
        raise RuntimeError(f"{run_id}/{call_index}: reservation identity mismatch")
    repairs = [
        event
        for event in events
        if event.get("event") in {"generation_metadata", "generation_metadata_repair"}
        and event.get("call_index") == call_index
        and event.get("generation_id") == active["generation_id"]
        and isinstance(event.get("generation"), dict)
        and event.get("generation")
        and event.get("error") is None
    ]
    if len(repairs) > 1:
        raise RuntimeError(f"{run_id}/{call_index}: duplicate successful repair events")
    call = PendingCall(run_id, call_index, active["generation_id"])
    if repairs:
        validate_generation_metadata(
            repairs[0]["generation"], call, payload, str(active["model"])
        )
    return RawResponse(path, payload, repairs[0] if repairs else None, response)


async def build_recovery_plan(
    ledger: Ledger,
    raw_dir: Path,
    schedule_path: Path,
    selected_run_ids: list[str],
) -> RecoveryPlan:
    if not selected_run_ids:
        raise RuntimeError("metadata recovery requires explicit selected run IDs")
    if len(selected_run_ids) != len(set(selected_run_ids)):
        raise RuntimeError("metadata recovery selectors contain duplicate run IDs")
    selected = tuple(sorted(selected_run_ids))
    current = await active_rows(ledger)
    if not current:
        raise RuntimeError("metadata recovery found no active reservations")
    active_ids = [row["run_id"] for row in current]
    # The agent loop keeps one call in flight per trajectory, but the judging
    # instruments issue several per run, so identity is the call, not the run.
    active_calls = [(row["run_id"], row["call_index"]) for row in current]
    duplicates = sorted(
        call for call in set(active_calls) if active_calls.count(call) != 1
    )
    if duplicates:
        raise RuntimeError(
            f"metadata recovery found duplicate active call rows for {duplicates[:8]}"
        )
    if set(selected) != set(active_ids):
        missing = sorted(set(active_ids) - set(selected))
        extra = sorted(set(selected) - set(active_ids))
        raise RuntimeError(
            f"selected recovery IDs do not equal the global active set; "
            f"missing={missing[:8]} extra={extra[:8]}"
        )
    unsafe = [row for row in current if row["status"] not in PENDING_STATUSES]
    if unsafe:
        raise RuntimeError(
            "metadata recovery refuses active statuses other than "
            f"{PENDING_STATUSES}: {unsafe[0]['run_id']}/{unsafe[0]['call_index']} "
            f"is {unsafe[0]['status']}"
        )
    schedule = _load_schedule(schedule_path)
    calls_by_run = _all_call_rows(ledger, selected)
    active_caps = {float(row["hard_cap"]) for row in current}
    full_schedule_has_kimi = any(
        row.get("arm") == "kimi" for row in schedule.values()
    )
    expected_caps = set()
    for row in current:
        scheduled = schedule.get(row["run_id"])
        if not isinstance(scheduled, dict):
            continue
        policy = row.get("reservation_policy")
        if policy == V3_RESERVATION_POLICY:
            expected_caps.add(OPENROUTER_LIMIT_WITHOUT_KIMI)
        elif policy == RESERVATION_POLICY:
            if scheduled.get("arm") in ACTIVE_PILOT_ARMS | PILOT_V3_ARMS:
                raise RuntimeError(
                    f"{row['run_id']}: strict-v3 pilot call lacks the fixed-$185 "
                    "reservation policy"
                )
            expected_caps.add(
                OPENROUTER_LIMIT_WITH_KIMI
                if full_schedule_has_kimi
                else OPENROUTER_LIMIT_WITHOUT_KIMI
            )
        else:
            raise RuntimeError(
                f"{row['run_id']}: reservation policy/cap/cost is invalid"
            )
    if len(active_caps) != 1 or active_caps != expected_caps:
        raise RuntimeError(
            "metadata recovery hard cap differs from the frozen phase/schedule decision"
        )
    evidence: dict[tuple[str, int], RawResponse] = {}
    raw_prefixes: dict[str, dict] = {}
    for active in current:
        run_id = active["run_id"]
        scheduled = schedule.get(run_id)
        if not isinstance(scheduled, dict):
            raise RuntimeError(f"{run_id}: active call is not in the frozen schedule")
        if active["call_kind"] == "agent":
            identity_valid = (
                1 <= active["call_index"] <= SEARCH_LIMIT + 3
                and active["model"] == scheduled.get("model")
            )
        else:
            identity_valid = judge_call_identity_valid(
                active["call_kind"], active["call_index"], active["model"]
            )
        if (
            not identity_valid
            or active["generation_id"] == ""
        ):
            raise RuntimeError(f"{run_id}: ledger/schedule call identity mismatch")
        if (
            active["reservation_policy"]
            not in {RESERVATION_POLICY, V3_RESERVATION_POLICY}
            or active["reserved_cost"] <= 0
        ):
            raise RuntimeError(f"{run_id}: reservation policy/cap/cost is invalid")
        prior_calls = calls_by_run.get(run_id, {})
        if any(
            row.get("status") in {"failed", "inflight", "unknown_inflight"}
            and row.get("call_kind") not in INDEPENDENT_MEASUREMENT_KINDS
            for row in prior_calls.values()
        ):
            raise RuntimeError(f"{run_id}: call history contains an unsafe status")
        expected_path = raw_dir / str(scheduled["arm"]) / f"{run_id}.jsonl"
        duplicate_paths = sorted(raw_dir.rglob(f"{run_id}.jsonl")) if raw_dir.exists() else []
        if duplicate_paths != [expected_path]:
            raise RuntimeError(
                f"{run_id}: expected one raw file at the frozen arm path, "
                f"found {len(duplicate_paths)}"
            )
        try:
            events = read_jsonl(expected_path)
        except (OSError, TypeError, ValueError) as error:
            raise RuntimeError(f"{run_id}: raw log is unreadable: {error}") from error
        key = (run_id, active["call_index"])
        evidence[key] = _validate_raw_call(
            ledger, active, scheduled, events, expected_path, prior_calls
        )
        relative_raw_path = _relative_raw_path(expected_path, raw_dir)
        raw_prefixes[relative_raw_path] = _raw_prefix(expected_path, run_id)
    try:
        require_repairable_selection(schedule, selected)
        anchor = recovery_preflight_anchor(ledger.db_path)
    except (OSError, RuntimeError, ValueError) as error:
        raise RuntimeError(f"recovery preflight anchor failed: {error}") from error
    return RecoveryPlan(
        calls=tuple(
            PendingCall(row["run_id"], row["call_index"], row["generation_id"])
            for row in current
        ),
        selected_run_ids=selected,
        active_rows=tuple(current),
        raw_before_prefixes=dict(sorted(raw_prefixes.items())),
        evidence=evidence,
        preflight_anchor=anchor,
    )


async def repair_pending(
    ledger: Ledger,
    raw_dir: Path,
    client: Any,
    *,
    concurrency: int = MODEL_CONCURRENCY,
    plan: RecoveryPlan | None = None,
) -> RepairSummary:
    if concurrency < 1 or concurrency > 32:
        raise ValueError("concurrency must be between 1 and 32")
    if plan is None:
        calls = await pending_calls(ledger)
        located, locate_errors = locate_raw_responses(calls, raw_dir)
        models: dict[tuple[str, int], str] = {}
    else:
        current = await active_rows(ledger)
        if current != list(plan.active_rows):
            raise RuntimeError("global active reservation set changed after validation")
        calls = list(plan.calls)
        located = plan.evidence
        locate_errors = {}
        models = {
            (row["run_id"], row["call_index"]): row["model"]
            for row in plan.active_rows
        }
    semaphore = asyncio.Semaphore(concurrency)

    async def repair_one(call: PendingCall) -> tuple[str, bool, str | None]:
        recovered = False
        try:
            key = (call.run_id, call.call_index)
            if key in locate_errors:
                return "failed", False, locate_errors[key]
            raw = located[key]
            current_row = await ledger.get_call(call.run_id, call.call_index)
            if (
                not isinstance(current_row, dict)
                or current_row.get("status") not in PENDING_STATUSES
                or str(current_row.get("generation_id") or "")
                != call.generation_id
            ):
                raise RuntimeError("ledger call identity changed after validation")
            recovered = raw.prior_repair is not None
            if recovered:
                generation = raw.prior_repair["generation"]
            else:
                async with semaphore:
                    generation: Any = {}
                    try:
                        generation = await client.generation(call.generation_id)
                        if not isinstance(generation, dict) or not generation:
                            raise RuntimeError("generation endpoint returned no metadata")
                        if plan is not None:
                            validate_generation_metadata(
                                generation,
                                call,
                                raw.payload,
                                models[key],
                            )
                    except Exception as error:
                        append_jsonl(
                            raw.path,
                            {
                                "event": "generation_metadata_repair",
                                "run_id": call.run_id,
                                "call_index": call.call_index,
                                "generation_id": call.generation_id,
                                "timestamp": now_iso(),
                                "generation": (
                                    generation
                                    if isinstance(generation, dict)
                                    else {}
                                ),
                                "error": f"{type(error).__name__}: {error}",
                            },
                        )
                        return (
                            "failed",
                            False,
                            f"metadata fetch/validation failed: {error}",
                        )
                    append_jsonl(
                        raw.path,
                        {
                            "event": "generation_metadata_repair",
                            "run_id": call.run_id,
                            "call_index": call.call_index,
                            "generation_id": call.generation_id,
                            "timestamp": now_iso(),
                            "generation": generation,
                            "error": None,
                        },
                    )
            if plan is not None:
                validate_generation_metadata(
                    generation,
                    call,
                    raw.payload,
                    models[key],
                )
            usage = usage_fields(raw.payload)
            await ledger.repair_metadata(
                call.run_id,
                call.call_index,
                {
                    **usage,
                    "generation_id": call.generation_id,
                    "native_cost": generation_cost(generation, usage),
                },
            )
        except Exception as error:
            return "failed", recovered, str(error)
        return "repaired", recovered, None

    results = await asyncio.gather(*(repair_one(call) for call in calls))
    failures = sorted(
        f"{call_name(call)}: {error}"
        for call, (status, _, error) in zip(calls, results, strict=True)
        if status == "failed" and error is not None
    )
    remaining = tuple(call_name(call) for call in await pending_calls(ledger))
    return RepairSummary(
        pending=len(calls),
        repaired=sum(status == "repaired" for status, _, _ in results),
        recovered_from_log=sum(
            status == "repaired" and recovered for status, recovered, _ in results
        ),
        failed=tuple(failures),
        remaining=remaining,
    )


def print_summary(summary: RepairSummary) -> None:
    print(
        "metadata_repair "
        f"pending={summary.pending} repaired={summary.repaired} "
        f"recovered_from_log={summary.recovered_from_log} "
        f"failed={len(summary.failed)} remaining={len(summary.remaining)}"
    )
    for failure in summary.failed:
        print(f"FAILED {failure}")
    if summary.remaining:
        print("REMAINING " + ", ".join(summary.remaining))


@contextmanager
def recovery_lock(db_path: Path):
    lock_path = db_path.with_name(f"{db_path.name}.metadata_repair.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("another metadata recovery process holds the lock") from error
        yield lock_path
    finally:
        os.close(descriptor)


def _hash_value(value: Any) -> str:
    return sha256_bytes(canonical_json(value).encode("utf-8"))


def _authorization(authority: str, evidence: str) -> dict:
    value = {
        "authority": " ".join(authority.split()),
        "evidence": " ".join(evidence.split()),
    }
    if not value["authority"] or not value["evidence"]:
        raise RuntimeError("recovery authorization fields must be nonempty")
    value["sha256"] = _hash_value(value)
    return value


def _event_digest(event: dict) -> str:
    return _hash_value({key: value for key, value in event.items() if key != "event_sha256"})


def _relative_data_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(DATA.resolve())
    except ValueError as error:
        raise RuntimeError("metadata recovery record must be written under data/") from error
    if resolved.suffix != ".jsonl":
        raise RuntimeError("metadata recovery record must be JSONL")
    return f"data/{relative.as_posix()}"


def recovery_preview(plan: RecoveryPlan) -> dict:
    active = list(plan.active_rows)
    return {
        "schema_version": RECOVERY_RECORD_SCHEMA,
        "mode": "scheduled_generation_metadata_only",
        "selected_run_ids": list(plan.selected_run_ids),
        "selected_run_ids_sha256": _hash_value(list(plan.selected_run_ids)),
        "before": {
            "active_rows": active,
            "active_rows_sha256": _hash_value(active),
            "raw_prefixes": plan.raw_before_prefixes,
            "raw_prefixes_sha256": _hash_value(plan.raw_before_prefixes),
        },
        "preflight_anchor": plan.preflight_anchor,
        "permitted_provider_operation": "GET /generation",
        "chat_completions_permitted": False,
        "trajectory_continuation_permitted": False,
    }


async def finalize_zero_active_interruption(
    ledger: Ledger,
    args: argparse.Namespace,
    record_path: Path,
    events: list[dict],
    authorization: dict,
    reason: str,
) -> RepairSummary | None:
    if not events or events[-1].get("event") not in {
        "recovery_attempt_started",
        "recovery_attempt_resumed",
    }:
        return None
    if await active_rows(ledger):
        return None
    first = events[0]
    if (
        first.get("authorization") != authorization
        or first.get("reason") != reason
    ):
        raise RuntimeError("zero-active recovery authorization differs")
    start = events[-1]
    selected = start.get("selected_run_ids")
    if not isinstance(selected, list) or sorted(args.run_ids) != sorted(selected):
        raise RuntimeError("zero-active finalization selectors differ from interrupted attempt")
    schedule = _load_schedule(args.schedule)
    require_repairable_selection(schedule, tuple(sorted(selected)))
    anchor = recovery_preflight_anchor(ledger.db_path)
    if anchor != start.get("preflight_anchor"):
        raise RuntimeError("zero-active finalization preflight anchor differs")
    calls_by_run = _all_call_rows(ledger, tuple(sorted(selected)))
    before_rows = start.get("before", {}).get("active_rows")
    if not isinstance(before_rows, list) or len(before_rows) != len(selected):
        raise RuntimeError("interrupted attempt lacks its exact before rows")
    for before in before_rows:
        run_id = str(before.get("run_id") or "")
        call_index = before.get("call_index")
        current = calls_by_run.get(run_id, {}).get(call_index)
        if current is None or current.get("status") != "complete":
            raise RuntimeError(f"{run_id}/{call_index}: interrupted repair is not complete")
        for field in (
            "run_id", "call_index", "call_kind", "model", "reserved_cost",
            "hard_cap", "reservation_policy", "generation_id",
        ):
            if current.get(field) != before.get(field):
                raise RuntimeError(f"{run_id}/{call_index}: repaired identity changed")
        scheduled = schedule.get(run_id)
        if not isinstance(scheduled, dict):
            raise RuntimeError(f"{run_id}: interrupted repair is outside the frozen schedule")
        path = args.raw_dir / str(scheduled["arm"]) / f"{run_id}.jsonl"
        raw = _validate_raw_call(
            ledger,
            before,
            scheduled,
            read_jsonl(path),
            path,
            calls_by_run[run_id],
        )
        if raw.prior_repair is None:
            raise RuntimeError(f"{run_id}/{call_index}: successful repair event is missing")
    original_ids = events[0].get("selected_run_ids")
    if not isinstance(original_ids, list):
        raise RuntimeError("metadata recovery record lacks original selectors")
    raw_after = _collect_raw_prefixes(
        args.raw_dir, schedule, tuple(sorted(original_ids))
    )
    summary = RepairSummary(len(selected), len(selected), len(selected), (), ())
    append_recovery_record(
        record_path,
        {
            "schema_version": RECOVERY_RECORD_SCHEMA,
            "mode": "scheduled_generation_metadata_only",
            "record_path": _relative_data_path(record_path),
            "event": "recovery_attempt_finished",
            "timestamp": now_iso(),
            "status": "completed",
            "summary": {
                "pending": summary.pending,
                "repaired": summary.repaired,
                "recovered_from_log": summary.recovered_from_log,
                "failed": [],
                "remaining": [],
            },
            "after": {
                "active_rows": [],
                "active_rows_sha256": _hash_value([]),
                "raw_prefixes": raw_after,
                "raw_prefixes_sha256": _hash_value(raw_after),
            },
            "previous_event_sha256": start["event_sha256"],
            "finalized_without_provider_call": True,
        },
    )
    return summary


def append_recovery_record(path: Path, event: dict, *, create: bool = False) -> dict:
    _relative_data_path(path)
    if create:
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError as error:
            raise RuntimeError(
                f"metadata recovery record already exists: {path.name}"
            ) from error
        else:
            os.close(descriptor)
    value = dict(event)
    value["event_sha256"] = _event_digest(value)
    append_jsonl(path, value)
    return value


def read_recovery_record(path: Path) -> list[dict]:
    try:
        events = read_jsonl(path)
    except (OSError, TypeError, ValueError) as error:
        raise RuntimeError(f"metadata recovery record is unreadable: {error}") from error
    if not events:
        raise RuntimeError("metadata recovery record is empty")
    previous = None
    for index, event in enumerate(events):
        if event.get("event_sha256") != _event_digest(event):
            raise RuntimeError(f"metadata recovery event {index + 1} hash differs")
        if event.get("previous_event_sha256") != previous:
            raise RuntimeError(f"metadata recovery event {index + 1} chain differs")
        previous = event["event_sha256"]
    return events


async def execute(args: argparse.Namespace) -> int:
    if not args.dry_run:
        canonical_paths = {
            "db": ARCHIVE / "call_ledger.sqlite3",
            "csv": LEDGER_PATH,
            "raw_dir": RAW_DIR,
            "schedule": SCHEDULE_PATH,
        }
        for field, expected in canonical_paths.items():
            if Path(getattr(args, field)).resolve() != expected.resolve():
                raise RuntimeError(
                    f"production metadata recovery forbids a noncanonical --{field.replace('_', '-')}"
                )
    ledger = Ledger(args.db, args.csv)
    with recovery_lock(args.db):
        record_path = args.record.resolve() if args.record is not None else None
        existing_events = (
            read_recovery_record(record_path)
            if record_path is not None and record_path.exists()
            else []
        )
        normalized_reason = " ".join(args.reason.split())
        authorization = (
            None
            if args.dry_run
            else _authorization(args.authorized_by, args.authorization_evidence)
        )
        if existing_events and not args.dry_run:
            first = existing_events[0]
            if (
                first.get("authorization") != authorization
                or first.get("reason") != normalized_reason
            ):
                raise RuntimeError(
                    "recovery resume reason or authorization differs before write"
                )
        if not args.dry_run and record_path is not None and existing_events:
            finalized = await finalize_zero_active_interruption(
                ledger,
                args,
                record_path,
                existing_events,
                authorization,
                normalized_reason,
            )
            if finalized is not None:
                print_summary(finalized)
                return 0
        plan = await build_recovery_plan(
            ledger,
            args.raw_dir,
            args.schedule,
            list(args.run_ids),
        )
        preview = recovery_preview(plan)
        if args.dry_run:
            print(pretty_json({**preview, "status": "validated"}), end="")
            return 0
        if args.record is None:
            raise RuntimeError("executing metadata recovery requires --record")
        record_path = args.record.resolve()
        existing_events = read_recovery_record(record_path) if record_path.exists() else []
        if existing_events and existing_events[-1].get("event") in {
            "recovery_attempt_started",
            "recovery_attempt_resumed",
        }:
            interrupted_start = existing_events[-1]
            if existing_events[0].get("preflight_anchor") != plan.preflight_anchor:
                raise RuntimeError(
                    "interrupted recovery preflight anchor differs before write"
                )
            original_ids = existing_events[0].get("selected_run_ids")
            if not isinstance(original_ids, list):
                raise RuntimeError("interrupted recovery lacks original selectors")
            schedule = _load_schedule(args.schedule)
            interrupted_raw = _collect_raw_prefixes(
                args.raw_dir, schedule, tuple(sorted(original_ids))
            )
            interrupted_selected = interrupted_start.get("selected_run_ids")
            if not isinstance(interrupted_selected, list):
                raise RuntimeError("interrupted recovery selectors are invalid")
            if not set(plan.selected_run_ids) <= set(interrupted_selected):
                raise RuntimeError("interrupted recovery remaining selectors differ")
            remaining_names = [call_name(call) for call in plan.calls]
            remaining_ids = {call.run_id for call in plan.calls}
            repaired_count = sum(
                run_id not in remaining_ids for run_id in interrupted_selected
            )
            failures = [
                f"{name}: interrupted attempt has no terminal repair record"
                for name in remaining_names
            ]
            append_recovery_record(
                record_path,
                {
                    "schema_version": RECOVERY_RECORD_SCHEMA,
                    "mode": "scheduled_generation_metadata_only",
                    "record_path": _relative_data_path(record_path),
                    "event": "recovery_attempt_finished",
                    "timestamp": now_iso(),
                    "status": "incomplete",
                    "summary": {
                        "pending": len(interrupted_selected),
                        "repaired": repaired_count,
                        "recovered_from_log": repaired_count,
                        "failed": failures,
                        "remaining": remaining_names,
                    },
                    "after": {
                        "active_rows": list(plan.active_rows),
                        "active_rows_sha256": _hash_value(list(plan.active_rows)),
                        "raw_prefixes": interrupted_raw,
                        "raw_prefixes_sha256": _hash_value(interrupted_raw),
                    },
                    "previous_event_sha256": interrupted_start["event_sha256"],
                    "closed_after_interruption": True,
                },
            )
            existing_events = read_recovery_record(record_path)
        prior_event = existing_events[-1] if existing_events else None
        if prior_event is not None:
            prior_after = prior_event.get("after")
            if prior_event.get("event") == "recovery_attempt_finished":
                if prior_event.get("status") != "incomplete":
                    raise RuntimeError("completed recovery record cannot be resumed")
                if not isinstance(prior_after, dict):
                    raise RuntimeError("incomplete recovery record lacks after state")
                if prior_after.get("active_rows") != list(plan.active_rows):
                    raise RuntimeError("recovery resume active rows differ from prior after state")
                prior_raw = prior_after.get("raw_prefixes")
                if not isinstance(prior_raw, dict) or any(
                    prior_raw.get(path) != descriptor
                    for path, descriptor in plan.raw_before_prefixes.items()
                ):
                    raise RuntimeError(
                        "recovery resume raw prefixes differ from prior after state"
                    )
            elif prior_event.get("event") not in {
                "recovery_attempt_started",
                "recovery_attempt_resumed",
            }:
                raise RuntimeError("metadata recovery record has an invalid terminal event")
            first = existing_events[0]
            if (
                first.get("authorization") != authorization
                or first.get("reason") != normalized_reason
                or first.get("preflight_anchor") != plan.preflight_anchor
            ):
                raise RuntimeError("recovery resume authorization/anchor differs")
        start_event = {
            **preview,
            "record_path": _relative_data_path(record_path),
            "event": (
                "recovery_attempt_resumed"
                if existing_events
                else "recovery_attempt_started"
            ),
            "timestamp": now_iso(),
            "reason": normalized_reason,
            "authorization": authorization,
            "previous_event_sha256": (
                prior_event.get("event_sha256") if prior_event else None
            ),
            "interrupted_previous_attempt": bool(
                prior_event
                and prior_event.get("event")
                in {"recovery_attempt_started", "recovery_attempt_resumed"}
            ),
        }
        if not start_event["reason"]:
            raise RuntimeError("metadata recovery reason must be nonempty")
        start_event = append_recovery_record(
            record_path, start_event, create=not existing_events
        )
        client = OpenRouterClient(ledger, timeout=args.timeout)
        try:
            summary = await repair_pending(
                ledger,
                args.raw_dir,
                client,
                concurrency=args.concurrency,
                plan=plan,
            )
        finally:
            await client.close()
        after_active = await active_rows(ledger)
        original_run_ids = (
            existing_events[0]["selected_run_ids"]
            if existing_events
            else list(plan.selected_run_ids)
        )
        schedule = _load_schedule(args.schedule)
        raw_after = _collect_raw_prefixes(
            args.raw_dir, schedule, tuple(sorted(original_run_ids))
        )
        finished = {
            "schema_version": RECOVERY_RECORD_SCHEMA,
            "mode": "scheduled_generation_metadata_only",
            "record_path": _relative_data_path(record_path),
            "event": "recovery_attempt_finished",
            "timestamp": now_iso(),
            "status": (
                "completed"
                if summary.exit_code == 0 and not after_active
                else "incomplete"
            ),
            "summary": {
                "pending": summary.pending,
                "repaired": summary.repaired,
                "recovered_from_log": summary.recovered_from_log,
                "failed": list(summary.failed),
                "remaining": list(summary.remaining),
            },
            "after": {
                "active_rows": after_active,
                "active_rows_sha256": _hash_value(after_active),
                "raw_prefixes": raw_after,
                "raw_prefixes_sha256": _hash_value(raw_after),
            },
            "previous_event_sha256": start_event["event_sha256"],
        }
        finished = append_recovery_record(record_path, finished)
    print_summary(summary)
    return 0 if finished["status"] == "completed" else 1


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Repair generation metadata for received responses without resubmitting "
            "any model request."
        )
    )
    parser.add_argument("--db", type=Path, default=ARCHIVE / "call_ledger.sqlite3")
    parser.add_argument("--csv", type=Path, default=LEDGER_PATH)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--schedule", type=Path, default=SCHEDULE_PATH)
    parser.add_argument("--run-ids", nargs="+", required=True)
    parser.add_argument("--concurrency", type=int, default=MODEL_CONCURRENCY)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--record", type=Path)
    parser.add_argument("--reason", default="")
    parser.add_argument("--authorized-by", default="")
    parser.add_argument("--authorization-evidence", default="")
    args = parser.parse_args()
    if args.dry_run and (
        args.record
        or args.reason
        or args.authorized_by
        or args.authorization_evidence
    ):
        parser.error("dry-run does not accept write or authorization options")
    if not args.dry_run and (
        args.record is None
        or not args.reason.strip()
        or not args.authorized_by.strip()
        or not args.authorization_evidence.strip()
    ):
        parser.error(
            "execution requires --record, --reason, --authorized-by, and "
            "--authorization-evidence"
        )
    raise SystemExit(asyncio.run(execute(args)))


if __name__ == "__main__":
    main()
