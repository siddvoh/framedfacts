from __future__ import annotations

import asyncio
import base64
import copy
import csv
import json
import os
import random
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, ROUND_CEILING
from pathlib import Path
from typing import Any

import httpx

from config import (
    ARCHIVE,
    KIMI_MODEL,
    LEDGER_PATH,
    OPENROUTER_LIMIT_WITH_KIMI,
    OPENROUTER_LIMIT_WITHOUT_KIMI,
    append_jsonl,
    canonical_json,
    load_environment,
    now_iso,
    read_jsonl,
    sha256_bytes,
    sha256_file,
)


DB_PATH = ARCHIVE / "call_ledger.sqlite3"
TRANSPORT_RECONCILIATION_EVENT = "transport_reconciliation"
RECONCILIATION_EVENT_SCHEMA = 1
RECONCILIATION_EVIDENCE_TYPE = (
    "provider_confirmed_no_execution_and_no_billing_v1"
)
RECONCILIATION_DECISION = "confirmed_unbilled_no_execution_terminal_failure"
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
REQUEST_RESERVATION_POLICY = "request_byte_token_ceiling_v1"
V3_FIXED_185_RESERVATION_POLICY = "request_byte_token_ceiling_fixed_185_v3"
FIELDS = [
    "timestamp",
    "run_id",
    "call_index",
    "call_kind",
    "model",
    "generation_id",
    "input_tokens",
    "output_tokens",
    "reasoning_tokens",
    "native_cost",
    "openrouter_cost",
    "upstream_cost",
    "is_byok",
    "cumulative_openrouter_spend",
    "cumulative_gemini_upstream",
    "status",
]


@dataclass
class Totals:
    openrouter: float
    gemini_upstream: float
    calls: int
    reservations: float


class Ledger:
    def __init__(self, db_path: Path = DB_PATH, csv_path: Path = LEDGER_PATH) -> None:
        self.db_path = db_path
        self.csv_path = csv_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = asyncio.Lock()
        self._initialize()

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=DELETE")
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def _initialize(self) -> None:
        with self.connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS calls(
                    seq INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    call_index INTEGER NOT NULL,
                    call_kind TEXT NOT NULL,
                    model TEXT NOT NULL,
                    status TEXT NOT NULL,
                    reserved_cost REAL NOT NULL,
                    hard_cap REAL NOT NULL,
                    reservation_policy TEXT NOT NULL,
                    budget_breach INTEGER NOT NULL DEFAULT 0,
                    generation_id TEXT,
                    input_tokens INTEGER NOT NULL DEFAULT 0,
                    output_tokens INTEGER NOT NULL DEFAULT 0,
                    reasoning_tokens INTEGER NOT NULL DEFAULT 0,
                    native_cost REAL NOT NULL DEFAULT 0,
                    openrouter_cost REAL NOT NULL DEFAULT 0,
                    upstream_cost REAL NOT NULL DEFAULT 0,
                    is_byok INTEGER NOT NULL DEFAULT 0,
                    timestamp TEXT NOT NULL,
                    UNIQUE(run_id, call_index)
                );
                CREATE TABLE IF NOT EXISTS wire_attempts(
                    run_id TEXT NOT NULL,
                    call_index INTEGER NOT NULL,
                    attempt INTEGER NOT NULL,
                    started_at TEXT NOT NULL,
                    finished_at TEXT NOT NULL,
                    request_sha256 TEXT NOT NULL,
                    response_sha256 TEXT NOT NULL,
                    response_status INTEGER,
                    generation_id TEXT,
                    error TEXT,
                    PRIMARY KEY(run_id, call_index, attempt),
                    FOREIGN KEY(run_id, call_index) REFERENCES calls(run_id, call_index)
                );
                CREATE TABLE IF NOT EXISTS reconciliation_events(
                    run_id TEXT NOT NULL,
                    call_index INTEGER NOT NULL,
                    prior_status TEXT NOT NULL,
                    evidence TEXT NOT NULL CHECK(length(trim(evidence)) > 0),
                    reconciled_at TEXT NOT NULL,
                    attempt INTEGER,
                    evidence_type TEXT,
                    evidence_artifact_sha256 TEXT,
                    decision TEXT,
                    raw_event_sha256 TEXT,
                    PRIMARY KEY(run_id, call_index),
                    FOREIGN KEY(run_id, call_index) REFERENCES calls(run_id, call_index)
                );
                """
            )
            columns = {
                str(row[1])
                for row in connection.execute("PRAGMA table_info(calls)")
            }
            if "hard_cap" not in columns:
                connection.execute(
                    "ALTER TABLE calls ADD COLUMN hard_cap REAL NOT NULL DEFAULT 185.0"
                )
            if "reservation_policy" not in columns:
                connection.execute(
                    "ALTER TABLE calls ADD COLUMN reservation_policy TEXT NOT NULL "
                    "DEFAULT 'legacy_fixed_v0'"
                )
            if "budget_breach" not in columns:
                connection.execute(
                    "ALTER TABLE calls ADD COLUMN budget_breach INTEGER NOT NULL DEFAULT 0"
                )
            reconciliation_columns = {
                str(row[1])
                for row in connection.execute(
                    "PRAGMA table_info(reconciliation_events)"
                )
            }
            for name, sql_type in (
                ("attempt", "INTEGER"),
                ("evidence_type", "TEXT"),
                ("evidence_artifact_sha256", "TEXT"),
                ("decision", "TEXT"),
                ("raw_event_sha256", "TEXT"),
            ):
                if name not in reconciliation_columns:
                    connection.execute(
                        f"ALTER TABLE reconciliation_events ADD COLUMN {name} {sql_type}"
                    )
            if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                raise RuntimeError("call ledger failed integrity_check")

    def _totals(self, connection: sqlite3.Connection) -> Totals:
        row = connection.execute(
            """SELECT
                coalesce(sum(openrouter_cost), 0),
                coalesce(sum(CASE WHEN is_byok=1 AND model='google/gemini-3.6-flash'
                    THEN upstream_cost ELSE 0 END), 0),
                count(*),
                coalesce(sum(CASE WHEN status IN (
                    'inflight','response_received','metadata_pending','unknown_inflight'
                )
                    THEN reserved_cost ELSE 0 END), 0)
                FROM calls"""
        ).fetchone()
        return Totals(float(row[0]), float(row[1]), int(row[2]), float(row[3]))

    async def totals(self) -> Totals:
        async with self.lock:
            with self.connect() as connection:
                return self._totals(connection)

    async def reserve(
        self,
        *,
        run_id: str,
        call_index: int,
        call_kind: str,
        model: str,
        reserved_cost: float,
        include_kimi: bool,
        reservation_policy: str = "caller_supplied_v0",
        hard_cap: float | None = None,
    ) -> None:
        if reserved_cost <= 0:
            raise ValueError("reserved cost must be positive")
        if not isinstance(reservation_policy, str) or not reservation_policy.strip():
            raise ValueError("reservation policy must be nonempty")
        policy = reservation_policy.strip()
        default_limit = (
            OPENROUTER_LIMIT_WITH_KIMI
            if include_kimi
            else OPENROUTER_LIMIT_WITHOUT_KIMI
        )
        if hard_cap is None:
            if policy == V3_FIXED_185_RESERVATION_POLICY:
                raise ValueError("strict-v3 reservations require an explicit hard cap")
            limit = default_limit
        else:
            if isinstance(hard_cap, bool) or not isinstance(hard_cap, (int, float)):
                raise ValueError("reservation hard cap must be numeric")
            limit = float(hard_cap)
            if limit not in {
                OPENROUTER_LIMIT_WITHOUT_KIMI,
                OPENROUTER_LIMIT_WITH_KIMI,
            }:
                raise ValueError("reservation hard cap is not a frozen limit")
        if (
            policy == V3_FIXED_185_RESERVATION_POLICY
            and limit != OPENROUTER_LIMIT_WITHOUT_KIMI
        ):
            raise ValueError("strict-v3 reservations must use the fixed $185 hard cap")
        async with self.lock:
            with self.connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                breached = connection.execute(
                    "SELECT run_id,call_index FROM calls WHERE budget_breach=1 LIMIT 1"
                ).fetchone()
                if breached is not None:
                    raise RuntimeError(
                        "budget hard stop is latched after reservation underflow at "
                        f"{breached['run_id']}/{breached['call_index']}"
                    )
                existing = connection.execute(
                    "SELECT status FROM calls WHERE run_id=? AND call_index=?",
                    (run_id, call_index),
                ).fetchone()
                if existing is not None:
                    raise RuntimeError(
                        f"call {run_id}/{call_index} already exists with status {existing['status']}"
                    )
                totals = self._totals(connection)
                if totals.openrouter + totals.reservations + reserved_cost > limit:
                    raise RuntimeError(
                        "OpenRouter hard stop reservation failed: "
                        f"${totals.openrouter:.4f} settled + ${totals.reservations:.4f} reserved "
                        f"+ ${reserved_cost:.4f} > ${limit:.2f}"
                    )
                connection.execute(
                    """INSERT INTO calls(
                        run_id,call_index,call_kind,model,status,reserved_cost,hard_cap,
                        reservation_policy,timestamp
                    ) VALUES(?,?,?,?,?,?,?,?,?)""",
                    (
                        run_id,
                        call_index,
                        call_kind,
                        model,
                        "inflight",
                        reserved_cost,
                        limit,
                        policy,
                        now_iso(),
                    ),
                )

    async def attempt(self, row: dict) -> None:
        async with self.lock:
            with self.connect() as connection:
                connection.execute(
                    """INSERT INTO wire_attempts(
                        run_id,call_index,attempt,started_at,finished_at,request_sha256,
                        response_sha256,response_status,generation_id,error
                    ) VALUES(?,?,?,?,?,?,?,?,?,?)""",
                    (
                        row["run_id"],
                        row["call_index"],
                        row["attempt"],
                        row["started_at"],
                        row["finished_at"],
                        row["request_sha256"],
                        row["response_sha256"],
                        row.get("response_status"),
                        row.get("generation_id"),
                        row.get("error"),
                    ),
                )

    async def response_received(self, run_id: str, call_index: int, generation_id: str) -> None:
        async with self.lock:
            with self.connect() as connection:
                connection.execute(
                    """UPDATE calls SET status='response_received',generation_id=?,timestamp=?
                       WHERE run_id=? AND call_index=? AND status='inflight'""",
                    (generation_id, now_iso(), run_id, call_index),
                )
                if connection.total_changes != 1:
                    raise RuntimeError("illegal response_received transition")

    async def settle(self, run_id: str, call_index: int, values: dict, status: str) -> None:
        if status not in {"complete", "metadata_pending", "failed", "unknown_inflight"}:
            raise ValueError(f"invalid terminal status: {status}")
        async with self.lock:
            budget_error: str | None = None
            with self.connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                current = connection.execute(
                    "SELECT * FROM calls WHERE run_id=? AND call_index=?",
                    (run_id, call_index),
                ).fetchone()
                if current is None or current["status"] not in {"inflight", "response_received"}:
                    raise RuntimeError(f"illegal call transition to {status}")
                openrouter_cost = float(values.get("openrouter_cost") or 0)
                reserved_cost = float(current["reserved_cost"])
                hard_cap = float(current["hard_cap"])
                other = connection.execute(
                    """SELECT
                        coalesce(sum(openrouter_cost), 0),
                        coalesce(sum(CASE WHEN status IN (
                            'inflight','response_received','metadata_pending','unknown_inflight'
                        ) THEN reserved_cost ELSE 0 END), 0)
                       FROM calls WHERE NOT (run_id=? AND call_index=?)""",
                    (run_id, call_index),
                ).fetchone()
                committed_after = float(other[0]) + float(other[1]) + openrouter_cost
                breach = (
                    openrouter_cost > reserved_cost + 1e-12
                    or committed_after > hard_cap + 1e-9
                )
                if breach:
                    budget_error = (
                        "budget reservation underflow: "
                        f"actual ${openrouter_cost:.10f}, reserved ${reserved_cost:.10f}, "
                        f"committed ${committed_after:.10f}, cap ${hard_cap:.2f}"
                    )
                connection.execute(
                    """UPDATE calls SET
                        status=?,generation_id=?,input_tokens=?,output_tokens=?,reasoning_tokens=?,
                        native_cost=?,openrouter_cost=?,upstream_cost=?,is_byok=?,budget_breach=?,
                        timestamp=?
                       WHERE run_id=? AND call_index=?""",
                    (
                        status,
                        values.get("generation_id") or None,
                        int(values.get("input_tokens") or 0),
                        int(values.get("output_tokens") or 0),
                        int(values.get("reasoning_tokens") or 0),
                        float(values.get("native_cost") or 0),
                        openrouter_cost,
                        float(values.get("upstream_cost") or 0),
                        int(values.get("is_byok") is True),
                        int(bool(current["budget_breach"]) or breach),
                        now_iso(),
                        run_id,
                        call_index,
                    ),
                )
            self._export_locked()
            if budget_error is not None:
                raise RuntimeError(budget_error)

    async def get_call(self, run_id: str, call_index: int) -> dict | None:
        async with self.lock:
            with self.connect() as connection:
                row = connection.execute(
                    "SELECT * FROM calls WHERE run_id=? AND call_index=?",
                    (run_id, call_index),
                ).fetchone()
        return dict(row) if row is not None else None

    async def repair_metadata(self, run_id: str, call_index: int, values: dict) -> None:
        async with self.lock:
            budget_error: str | None = None
            with self.connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                current = connection.execute(
                    "SELECT * FROM calls WHERE run_id=? AND call_index=?",
                    (run_id, call_index),
                ).fetchone()
                if current is None or current["status"] not in {"response_received", "metadata_pending"}:
                    raise RuntimeError("metadata repair requires a received response")
                openrouter_cost = float(values.get("openrouter_cost") or 0)
                reserved_cost = float(current["reserved_cost"])
                hard_cap = float(current["hard_cap"])
                other = connection.execute(
                    """SELECT
                        coalesce(sum(openrouter_cost), 0),
                        coalesce(sum(CASE WHEN status IN (
                            'inflight','response_received','metadata_pending','unknown_inflight'
                        ) THEN reserved_cost ELSE 0 END), 0)
                       FROM calls WHERE NOT (run_id=? AND call_index=?)""",
                    (run_id, call_index),
                ).fetchone()
                committed_after = float(other[0]) + float(other[1]) + openrouter_cost
                breach = (
                    openrouter_cost > reserved_cost + 1e-12
                    or committed_after > hard_cap + 1e-9
                )
                if breach:
                    budget_error = (
                        "budget reservation underflow during metadata repair: "
                        f"actual ${openrouter_cost:.10f}, reserved ${reserved_cost:.10f}, "
                        f"committed ${committed_after:.10f}, cap ${hard_cap:.2f}"
                    )
                connection.execute(
                    """UPDATE calls SET
                        status='complete',generation_id=?,input_tokens=?,output_tokens=?,
                        reasoning_tokens=?,native_cost=?,openrouter_cost=?,upstream_cost=?,
                        is_byok=?,budget_breach=?,timestamp=? WHERE run_id=? AND call_index=?""",
                    (
                        values["generation_id"],
                        int(values.get("input_tokens") or 0),
                        int(values.get("output_tokens") or 0),
                        int(values.get("reasoning_tokens") or 0),
                        float(values.get("native_cost") or 0),
                        openrouter_cost,
                        float(values.get("upstream_cost") or 0),
                        int(values.get("is_byok") is True),
                        int(bool(current["budget_breach"]) or breach),
                        now_iso(),
                        run_id,
                        call_index,
                    ),
                )
            self._export_locked()
            if budget_error is not None:
                raise RuntimeError(budget_error)

    async def release_unattempted_reservation(self, run_id: str, call_index: int) -> None:
        """Close a reservation whose request never reached the transport.

        A killed process can leave a reserved call with no wire attempt and no
        cost. Nothing was sent, so the row is recorded as a terminal failure
        rather than left active and blocking the audit.
        """

        async with self.lock:
            with self.connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                current = connection.execute(
                    "SELECT * FROM calls WHERE run_id=? AND call_index=?",
                    (run_id, call_index),
                ).fetchone()
                if current is None or current["status"] != "inflight":
                    raise RuntimeError(
                        f"{run_id}/{call_index} is not an inflight reservation"
                    )
                if current["generation_id"]:
                    raise RuntimeError(
                        f"{run_id}/{call_index} carries a generation ID and was attempted"
                    )
                attempts = connection.execute(
                    "SELECT count(*) FROM wire_attempts WHERE run_id=? AND call_index=?",
                    (run_id, call_index),
                ).fetchone()[0]
                if attempts:
                    raise RuntimeError(
                        f"{run_id}/{call_index} has {attempts} wire attempts and cannot "
                        "be released as unattempted"
                    )
                nonzero = [
                    field
                    for field in (
                        "input_tokens",
                        "output_tokens",
                        "reasoning_tokens",
                        "native_cost",
                        "openrouter_cost",
                        "upstream_cost",
                    )
                    if float(current[field]) != 0.0
                ]
                if nonzero:
                    raise RuntimeError(
                        f"{run_id}/{call_index} records usage or cost: " + ", ".join(nonzero)
                    )
                connection.execute(
                    "UPDATE calls SET status='failed' WHERE run_id=? AND call_index=?",
                    (run_id, call_index),
                )
                connection.execute(
                    """INSERT OR REPLACE INTO reconciliation_events(
                        run_id, call_index, prior_status, evidence, reconciled_at,
                        decision
                    ) VALUES(?,?,?,?,?,?)""",
                    (
                        run_id,
                        call_index,
                        "inflight",
                        "reservation had no wire attempt, no generation ID, and no cost",
                        now_iso(),
                        "released_unattempted_reservation",
                    ),
                )
            self._export_locked()

    async def reconcile_unbilled_failure(
        self,
        run_id: str,
        call_index: int,
        *,
        attempt: int,
        evidence_artifact_path: Path,
        raw_path: Path,
    ) -> None:
        """Terminally close an ambiguous call after provider absence proof."""
        if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt < 1:
            raise ValueError("reconciliation requires a positive attempt number")
        evidence_root = (self.db_path.parent / "reconciliation_evidence").resolve()
        resolved_evidence_path = evidence_artifact_path.resolve()
        try:
            resolved_evidence_path.relative_to(evidence_root)
        except ValueError as error:
            raise ValueError(
                "reconciliation evidence artifact must be under archive/reconciliation_evidence"
            ) from error
        if resolved_evidence_path.suffix != ".json":
            raise ValueError("reconciliation evidence artifact must be JSON")
        try:
            artifact_bytes = resolved_evidence_path.read_bytes()
            evidence_artifact = json.loads(artifact_bytes.decode("utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(
                f"reconciliation evidence artifact is unreadable: {error}"
            ) from error
        required_artifact_fields = {
            "provider",
            "request_sha256",
            "execution_found",
            "billing_found",
            "provider_record",
        }
        if (
            not isinstance(evidence_artifact, dict)
            or set(evidence_artifact) != required_artifact_fields
        ):
            raise ValueError(
                "reconciliation evidence artifact must have the exact frozen schema"
            )
        if (
            not isinstance(evidence_artifact.get("provider"), str)
            or not evidence_artifact["provider"].strip()
            or not isinstance(evidence_artifact.get("request_sha256"), str)
            or SHA256_PATTERN.fullmatch(evidence_artifact["request_sha256"]) is None
            or evidence_artifact.get("execution_found") is not False
            or evidence_artifact.get("billing_found") is not False
            or not isinstance(evidence_artifact.get("provider_record"), dict)
            or not evidence_artifact["provider_record"]
        ):
            raise ValueError(
                "reconciliation evidence must prove no provider execution and no billing"
            )
        if raw_path.suffix != ".jsonl" or raw_path.stem != run_id:
            raise ValueError("reconciliation raw path does not match the run ID")
        try:
            raw_path.resolve().relative_to((self.db_path.parent / "raw").resolve())
        except ValueError as error:
            raise ValueError("reconciliation raw path must be under archive/raw") from error
        artifact = json.loads(canonical_json(evidence_artifact))
        artifact_sha = sha256_file(resolved_evidence_path)
        project_root = self.db_path.parent.parent.resolve()
        try:
            artifact_relative = resolved_evidence_path.relative_to(project_root).as_posix()
        except ValueError as error:
            raise ValueError("reconciliation evidence is outside the project") from error
        async with self.lock:
            with self.connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                current = connection.execute(
                    "SELECT * FROM calls WHERE run_id=? AND call_index=?",
                    (run_id, call_index),
                ).fetchone()
                if current is None or current["status"] not in {
                    "unknown_inflight",
                    "failed",
                }:
                    raise RuntimeError(
                        "unbilled failure reconciliation requires an unknown_inflight "
                        "call or an exact committed reconciliation replay"
                    )
                ledger_generation_id = current["generation_id"]
                if ledger_generation_id not in {None, ""} and not (
                    evidence_artifact["provider_record"].get("generation_id")
                    == ledger_generation_id
                    and evidence_artifact["provider_record"].get("http_status") == 404
                ):
                    raise RuntimeError("cannot reconcile a call with a generation ID")
                usage_and_cost_fields = (
                    "input_tokens",
                    "output_tokens",
                    "reasoning_tokens",
                    "native_cost",
                    "openrouter_cost",
                    "upstream_cost",
                )
                nonzero_fields = [
                    field_name
                    for field_name in usage_and_cost_fields
                    if float(current[field_name]) != 0.0
                ]
                if nonzero_fields:
                    raise RuntimeError(
                        "cannot reconcile a call with nonzero usage or cost: "
                        + ", ".join(nonzero_fields)
                    )
                wire_attempt = connection.execute(
                    """SELECT * FROM wire_attempts
                       WHERE run_id=? AND call_index=? AND attempt=?""",
                    (run_id, call_index, attempt),
                ).fetchone()
                if wire_attempt is None:
                    raise RuntimeError(
                        "reconciliation attempt is absent from the wire-attempt ledger"
                    )
                latest_attempt = connection.execute(
                    """SELECT max(attempt) FROM wire_attempts
                       WHERE run_id=? AND call_index=?""",
                    (run_id, call_index),
                ).fetchone()[0]
                if attempt != latest_attempt:
                    raise RuntimeError("reconciliation must link the latest wire attempt")
                if evidence_artifact["request_sha256"] != wire_attempt["request_sha256"]:
                    raise RuntimeError(
                        "reconciliation evidence request hash differs from the wire attempt"
                    )
                try:
                    raw_events = read_jsonl(raw_path)
                except (OSError, TypeError, ValueError) as error:
                    raise RuntimeError(
                        f"reconciliation raw log is unavailable: {error}"
                    ) from error
                model_events = [
                    event
                    for event in raw_events
                    if event.get("event") in {"model_call", "model_response"}
                    and event.get("call_index") == call_index
                    and event.get("attempt") == attempt
                ]
                if len(model_events) != 1:
                    raise RuntimeError(
                        "reconciliation requires exactly one matching raw model attempt"
                    )
                model_event = model_events[0]
                request_wire = model_event.get("request_wire")
                raw_request_sha = (
                    request_wire.get("sha256")
                    if isinstance(request_wire, dict)
                    else None
                )
                if raw_request_sha != wire_attempt["request_sha256"]:
                    raise RuntimeError(
                        "reconciliation raw request hash differs from the wire attempt"
                    )
                billing_evidence = model_event.get("billing_evidence")
                response_body = model_event.get("response_body")
                response_body = response_body if isinstance(response_body, dict) else {}
                choices = response_body.get("choices")
                generation_id = next(
                    (
                        response_body.get(key)
                        for key in ("id", "generation_id", "generationId")
                        if str(response_body.get(key) or "").strip()
                    ),
                    None,
                )
                provider_record = evidence_artifact["provider_record"]
                proved_absent_ids = {
                    value
                    for value in (generation_id, current["generation_id"])
                    if value
                    and provider_record.get("generation_id") == value
                    and provider_record.get("http_status") == 404
                }
                ambiguity_resolved = bool(proved_absent_ids)
                billing_evidence_ok = billing_evidence in (None, [], ()) or (
                    ambiguity_resolved and list(billing_evidence) == ["generation_id"]
                )
                if (
                    model_event.get("terminal") is not True
                    or (
                        model_event.get("billing_ambiguous") is True
                        and not ambiguity_resolved
                    )
                    or not billing_evidence_ok
                    or (generation_id is not None and generation_id not in proved_absent_ids)
                    or (isinstance(choices, list) and bool(choices))
                ):
                    raise RuntimeError(
                        "raw attempt contains execution or billing evidence"
                    )
                existing_events = [
                    event
                    for event in raw_events
                    if event.get("event") == TRANSPORT_RECONCILIATION_EVENT
                    and event.get("call_index") == call_index
                ]
                if len(existing_events) > 1:
                    raise RuntimeError("duplicate raw transport reconciliation events")
                expected_subset = {
                    "schema_version": RECONCILIATION_EVENT_SCHEMA,
                    "event": TRANSPORT_RECONCILIATION_EVENT,
                    "run_id": run_id,
                    "call_index": call_index,
                    "attempt": attempt,
                    "evidence_type": RECONCILIATION_EVIDENCE_TYPE,
                    "evidence_artifact_path": artifact_relative,
                    "evidence_artifact_sha256": artifact_sha,
                    "provider_execution_evidence": False,
                    "provider_billing_evidence": False,
                    "decision": RECONCILIATION_DECISION,
                    "prior_status": "unknown_inflight",
                }
                if existing_events:
                    reconciliation = existing_events[0]
                    expected_keys = set(expected_subset) | {"checked_at", "timestamp"}
                    checked_at = reconciliation.get("checked_at")
                    try:
                        parsed_checked_at = datetime.fromisoformat(
                            str(checked_at).replace("Z", "+00:00")
                        )
                        if parsed_checked_at.tzinfo is None:
                            raise ValueError("missing time zone")
                    except ValueError:
                        parsed_checked_at = None
                    if (
                        set(reconciliation) != expected_keys
                        or any(
                            reconciliation.get(key) != value
                            for key, value in expected_subset.items()
                        )
                        or parsed_checked_at is None
                        or reconciliation.get("timestamp")
                        != checked_at
                    ):
                        raise RuntimeError(
                            "existing raw transport reconciliation event conflicts"
                        )
                    reconciled_at = reconciliation["checked_at"]
                else:
                    reconciled_at = now_iso()
                    reconciliation = {
                        **expected_subset,
                        "checked_at": reconciled_at,
                        "timestamp": reconciled_at,
                    }
                    # The append is deliberately durable before releasing the
                    # reservation in SQLite. A crash can safely replay this event.
                    append_jsonl(raw_path, reconciliation)
                raw_event_sha = sha256_bytes(
                    canonical_json(reconciliation).encode("utf-8")
                )
                existing_db_event = connection.execute(
                    """SELECT * FROM reconciliation_events
                       WHERE run_id=? AND call_index=?""",
                    (run_id, call_index),
                ).fetchone()
                expected_db_event = {
                    "run_id": run_id,
                    "call_index": call_index,
                    "prior_status": "unknown_inflight",
                    "evidence": artifact_relative,
                    "reconciled_at": reconciled_at,
                    "attempt": attempt,
                    "evidence_type": RECONCILIATION_EVIDENCE_TYPE,
                    "evidence_artifact_sha256": artifact_sha,
                    "decision": RECONCILIATION_DECISION,
                    "raw_event_sha256": raw_event_sha,
                }
                if current["status"] == "failed":
                    if (
                        existing_db_event is None
                        or any(
                            existing_db_event[key] != value
                            for key, value in expected_db_event.items()
                        )
                        or current["timestamp"] != reconciled_at
                    ):
                        raise RuntimeError(
                            "failed call is not an exact committed reconciliation replay"
                        )
                else:
                    if existing_db_event is not None:
                        raise RuntimeError(
                            "unknown_inflight call already has a reconciliation row"
                        )
                    proved_absent_id = (
                        provider_record.get("generation_id")
                        if provider_record.get("http_status") == 404
                        else ""
                    ) or ""
                    cursor = connection.execute(
                        """UPDATE calls SET status='failed',timestamp=?
                           WHERE run_id=? AND call_index=?
                             AND status='unknown_inflight'
                             AND (coalesce(generation_id,'')='' OR generation_id=?)
                             AND input_tokens=0 AND output_tokens=0 AND reasoning_tokens=0
                             AND native_cost=0 AND openrouter_cost=0 AND upstream_cost=0""",
                        (reconciled_at, run_id, call_index, proved_absent_id),
                    )
                    if cursor.rowcount != 1:
                        raise RuntimeError("reconciliation preconditions changed")
                    connection.execute(
                        """INSERT INTO reconciliation_events(
                            run_id,call_index,prior_status,evidence,reconciled_at,
                            attempt,evidence_type,evidence_artifact_sha256,decision,
                            raw_event_sha256
                        ) VALUES(?,?,?,?,?,?,?,?,?,?)""",
                        tuple(expected_db_event.values()),
                    )
            self._export_locked()

    def _export_locked(self) -> None:
        with self.connect() as connection:
            rows = connection.execute("SELECT * FROM calls ORDER BY seq").fetchall()
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        # A fixed temp name lets a second process delete this one's file mid-write.
        temporary = self.csv_path.with_name(f".{self.csv_path.name}.{os.getpid()}.tmp")
        cumulative_openrouter = 0.0
        cumulative_gemini = 0.0
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=FIELDS,
                lineterminator="\n",
            )
            writer.writeheader()
            for row in rows:
                cumulative_openrouter += float(row["openrouter_cost"])
                if row["is_byok"] and row["model"] == "google/gemini-3.6-flash":
                    cumulative_gemini += float(row["upstream_cost"])
                writer.writerow(
                    {
                        "timestamp": row["timestamp"],
                        "run_id": row["run_id"],
                        "call_index": row["call_index"],
                        "call_kind": row["call_kind"],
                        "model": row["model"],
                        "generation_id": row["generation_id"] or "",
                        "input_tokens": row["input_tokens"],
                        "output_tokens": row["output_tokens"],
                        "reasoning_tokens": row["reasoning_tokens"],
                        "native_cost": f"{row['native_cost']:.10f}",
                        "openrouter_cost": f"{row['openrouter_cost']:.10f}",
                        "upstream_cost": f"{row['upstream_cost']:.10f}",
                        "is_byok": str(bool(row["is_byok"])).lower(),
                        "cumulative_openrouter_spend": f"{cumulative_openrouter:.10f}",
                        "cumulative_gemini_upstream": f"{cumulative_gemini:.10f}",
                        "status": row["status"],
                    }
                )
        temporary.replace(self.csv_path)


API_ROOT = "https://openrouter.ai/api/v1"
TRANSIENT_STATUS = {408, 409, 429, 500, 502, 503, 504}
GENERATION_METADATA_MAX_ATTEMPTS = 200
GENERATION_METADATA_MAX_WAIT_SECONDS = 900.0
GENERATION_METADATA_PROVISIONAL_WAIT_SECONDS = 20.0
GENERATION_METADATA_GET_TIMEOUT_SECONDS = 10.0
GENERATION_METADATA_RETRY_STATUS = {404, 408, 409, 425, 429}
RESERVATION_POLICY = REQUEST_RESERVATION_POLICY
V3_RESERVATION_POLICY = V3_FIXED_185_RESERVATION_POLICY
RESERVATION_TOKEN_OVERHEAD = 4096
RESERVATION_PRICE_MULTIPLIER = Decimal("1.25")
RESERVATION_FIXED_BUFFER = Decimal("0.10")
MODEL_PRICE_CEILINGS = {
    # Per-token ceilings frozen from archive/model_snapshot_2026-08-01.json.
    # Terra/Luna use the more expensive long-context override.
    "anthropic/claude-sonnet-5": (Decimal("0.000002"), Decimal("0.00001")),
    "openai/gpt-5.6-terra": (Decimal("0.000002"), Decimal("0.000009")),
    "google/gemini-3.6-flash": (Decimal("0.0000015"), Decimal("0.0000075")),
    "z-ai/glm-5.2": (Decimal("0.0000007168"), Decimal("0.0000022528")),
    "openai/gpt-5.6-luna": (Decimal("0.0000002"), Decimal("0.0000009")),
    "moonshotai/kimi-k3:exacto": (Decimal("0.000003"), Decimal("0.000015")),
}


@dataclass
class ModelCall:
    payload: dict
    generation: dict
    call_index: int
    input_tokens: int
    output_tokens: int
    reasoning_tokens: int
    native_cost: float
    openrouter_cost: float
    upstream_cost: float
    is_byok: bool
    metadata_status: str = "complete"


def body_record(value: bytes) -> dict:
    return {
        "base64": base64.b64encode(value).decode("ascii"),
        "sha256": sha256_bytes(value),
        "size_bytes": len(value),
    }


def safe_headers(headers: httpx.Headers) -> dict[str, str]:
    excluded = ("authorization", "api-key", "cookie", "secret", "token")
    return {
        key: value
        for key, value in headers.items()
        if not any(part in key.lower() for part in excluded)
    }


def usage_fields(payload: dict) -> dict:
    usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else {}
    details = usage.get("cost_details") if isinstance(usage.get("cost_details"), dict) else {}
    completion = (
        usage.get("completion_tokens_details")
        if isinstance(usage.get("completion_tokens_details"), dict)
        else {}
    )
    return {
        "input_tokens": int(usage.get("prompt_tokens") or 0),
        "output_tokens": int(usage.get("completion_tokens") or 0),
        "reasoning_tokens": int(completion.get("reasoning_tokens") or 0),
        "openrouter_cost": float(usage.get("cost") or 0.0),
        "upstream_cost": float(details.get("upstream_inference_cost") or 0.0),
        "is_byok": usage.get("is_byok") is True,
    }


def guarded_usage_fields(payload: dict) -> tuple[dict, str | None]:
    try:
        return usage_fields(payload), None
    except (TypeError, ValueError, OverflowError) as error:
        return {
            "input_tokens": 0,
            "output_tokens": 0,
            "reasoning_tokens": 0,
            "openrouter_cost": 0.0,
            "upstream_cost": 0.0,
            "is_byok": False,
        }, f"invalid response usage metadata: {type(error).__name__}: {error}"


def generation_cost(generation: dict, fallback: dict) -> float:
    for key in ("total_cost", "native_cost", "cost"):
        value = generation.get(key)
        if value is not None and float(value) > 0:
            return float(value)
    if fallback.get("is_byok") is True and float(fallback.get("upstream_cost") or 0) > 0:
        return float(fallback["upstream_cost"])
    return float(fallback.get("openrouter_cost") or 0.0)


def response_generation_id(payload: Any, headers: httpx.Headers) -> str:
    """Return generation evidence without confusing ordinary request IDs for it."""
    payload = payload if isinstance(payload, dict) else {}
    candidates = [
        payload.get("id"),
        payload.get("generation_id"),
        payload.get("generationId"),
        headers.get("x-generation-id"),
        headers.get("x-openrouter-generation-id"),
    ]
    return next(
        (str(value).strip() for value in candidates if str(value or "").strip()),
        "",
    )


def response_tool_calls(payload: Any) -> list:
    if not isinstance(payload, dict):
        return []
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        return []
    message = choices[0].get("message")
    if not isinstance(message, dict):
        return []
    calls = message.get("tool_calls")
    return calls if isinstance(calls, list) else []


def response_structure_error(payload: Any, status_code: int) -> str | None:
    if not isinstance(payload, dict):
        return "response JSON is not an object"
    choices = payload.get("choices")
    if status_code == 200 and not isinstance(choices, list):
        return "HTTP 200 response has no choices list"
    if choices is None:
        return None
    if not isinstance(choices, list):
        return "response choices is not a list"
    if not choices:
        return "response choices is empty" if status_code == 200 else None
    if not isinstance(choices[0], dict):
        return "response choice is not an object"
    if not isinstance(choices[0].get("message"), dict):
        return "response message is not an object"
    return None


def billing_evidence(payload: Any, headers: httpx.Headers) -> list[str]:
    """Conservatively identify evidence that a chat request may have executed."""
    payload = payload if isinstance(payload, dict) else {}
    evidence: list[str] = []
    if response_generation_id(payload, headers):
        evidence.append("generation_id")
    choices = payload.get("choices")
    if isinstance(choices, list) and choices:
        evidence.append("choices")
    for field_name in ("output", "output_text"):
        value = payload.get(field_name)
        if value is not None and value is not False and value not in ("", [], {}):
            evidence.append(field_name)

    def positive_metering(value: Any, parent_key: str = "") -> bool:
        if isinstance(value, dict):
            return any(positive_metering(item, str(key)) for key, item in value.items())
        if isinstance(value, list):
            return any(positive_metering(item, parent_key) for item in value)
        key = parent_key.lower()
        if not any(fragment in key for fragment in ("usage", "token", "cost")):
            return False
        if isinstance(value, bool):
            return False
        try:
            return float(value) > 0
        except (TypeError, ValueError):
            return bool(str(value or "").strip())

    if positive_metering(payload):
        evidence.append("positive_usage_or_cost")
    return evidence


def reservation_cost(body: dict) -> float:
    """Conservative request-specific upper bound for the OpenRouter hard cap.

    UTF-8 bytes plus fixed provider overhead upper-bound normal text token counts.
    The frozen maximum token prices are then padded by 25% and $0.10 before
    rounding upward to the next cent. BYOK calls reserve their full native-cost
    ceiling even though normal OpenRouter burn is only the BYOK fee.
    """
    model = str(body.get("model") or "")
    prices = MODEL_PRICE_CEILINGS.get(model)
    if prices is None:
        raise ValueError(f"no frozen reservation ceiling for model {model!r}")
    max_tokens = body.get("max_tokens")
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 1:
        raise ValueError("request requires a positive integer max_tokens for reservation")
    prompt_ceiling = len(canonical_json(body).encode("utf-8")) + RESERVATION_TOKEN_OVERHEAD
    prompt_price, completion_price = prices
    native_ceiling = (
        Decimal(prompt_ceiling) * prompt_price
        + Decimal(max_tokens) * completion_price
    )
    padded = (
        native_ceiling * RESERVATION_PRICE_MULTIPLIER
        + RESERVATION_FIXED_BUFFER
    )
    return float(padded.quantize(Decimal("0.01"), rounding=ROUND_CEILING))


def reservation_provenance(
    body: dict,
    include_kimi: bool,
    *,
    hard_cap: float | None = None,
    policy: str = RESERVATION_POLICY,
) -> dict:
    if not isinstance(policy, str) or not policy.strip():
        raise ValueError("reservation policy must be nonempty")
    selected_cap = (
        OPENROUTER_LIMIT_WITH_KIMI
        if include_kimi
        else OPENROUTER_LIMIT_WITHOUT_KIMI
    )
    if hard_cap is not None:
        if isinstance(hard_cap, bool) or not isinstance(hard_cap, (int, float)):
            raise ValueError("reservation hard cap must be numeric")
        selected_cap = float(hard_cap)
        if selected_cap not in {
            OPENROUTER_LIMIT_WITHOUT_KIMI,
            OPENROUTER_LIMIT_WITH_KIMI,
        }:
            raise ValueError("reservation hard cap is not a frozen limit")
    if policy.strip() == V3_RESERVATION_POLICY:
        if hard_cap is None:
            raise ValueError("strict-v3 reservation provenance requires an explicit cap")
        if selected_cap != OPENROUTER_LIMIT_WITHOUT_KIMI:
            raise ValueError("strict-v3 reservation provenance must use the $185 cap")
    return {
        "policy": policy.strip(),
        "reserved_cost": reservation_cost(body),
        "hard_cap": selected_cap,
        "request_sha256": sha256_bytes(canonical_json(body).encode("utf-8")),
    }


class OpenRouterClient:
    def __init__(self, ledger: Ledger, timeout: float = 180.0) -> None:
        load_environment()
        self.ledger = ledger
        self.client = httpx.AsyncClient(
            base_url=API_ROOT,
            headers={
                "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
                "Content-Type": "application/json",
            },
            timeout=httpx.Timeout(timeout),
            limits=httpx.Limits(max_connections=40, max_keepalive_connections=24),
        )

    async def close(self) -> None:
        await self.client.aclose()

    async def generation(
        self,
        generation_id: str,
        *,
        max_wait_seconds: float = GENERATION_METADATA_MAX_WAIT_SECONDS,
    ) -> dict:
        last_error = "generation metadata unavailable"
        loop = asyncio.get_running_loop()
        deadline = loop.time() + max_wait_seconds
        for attempt in range(1, GENERATION_METADATA_MAX_ATTEMPTS + 1):
            remaining = deadline - loop.time()
            if remaining <= 0:
                break
            try:
                response = await self.client.get(
                    "/generation",
                    params={"id": generation_id},
                    timeout=min(GENERATION_METADATA_GET_TIMEOUT_SECONDS, remaining),
                )
                if response.status_code == 200:
                    payload = response.json()
                    data = payload.get("data") if isinstance(payload, dict) else None
                    if isinstance(data, dict):
                        return data
                    last_error = "generation response has no data object"
                else:
                    last_error = f"generation HTTP {response.status_code}: {response.text[:500]}"
                    retryable = (
                        response.status_code in GENERATION_METADATA_RETRY_STATUS
                        or 500 <= response.status_code <= 599
                    )
                    if not retryable:
                        raise RuntimeError(last_error)
            except (httpx.TimeoutException, httpx.TransportError) as error:
                last_error = f"{type(error).__name__}: {error}"
            if attempt == GENERATION_METADATA_MAX_ATTEMPTS:
                break
            delay = min(0.5 * attempt, 5.0)
            remaining = deadline - loop.time()
            if remaining <= 0:
                break
            await asyncio.sleep(min(delay, remaining))
        raise RuntimeError(last_error)

    async def call(
        self,
        *,
        body: dict,
        run_id: str,
        call_index: int,
        call_kind: str,
        raw_path: Path,
        include_kimi: bool = False,
        reservation_hard_cap: float | None = None,
        reservation_policy: str = RESERVATION_POLICY,
        max_attempts: int = 4,
        allow_provisional_metadata: bool = False,
    ) -> ModelCall:
        request_body = canonical_json(body).encode("utf-8")
        last_error = "request was not attempted"
        reservation = reservation_provenance(
            body,
            include_kimi,
            hard_cap=reservation_hard_cap,
            policy=reservation_policy,
        )
        await self.ledger.reserve(
            run_id=run_id,
            call_index=call_index,
            call_kind=call_kind,
            model=body["model"],
            reserved_cost=reservation["reserved_cost"],
            include_kimi=include_kimi,
            reservation_policy=reservation["policy"],
            # Preserve the legacy ledger-owned phase-cap decision unless the
            # caller deliberately selected an override. Strict-v3 call sites
            # always provide the fixed-$185 override explicitly.
            hard_cap=(
                reservation["hard_cap"]
                if reservation_hard_cap is not None
                else None
            ),
        )

        for attempt in range(1, max_attempts + 1):
            started_at = now_iso()
            try:
                response = await self.client.post("/chat/completions", content=request_body)
                response_body = response.content
                try:
                    payload: Any = response.json()
                except (ValueError, UnicodeDecodeError):
                    payload = {}
                record = {
                    "event": "model_call",
                    "run_id": run_id,
                    "call_index": call_index,
                    "call_kind": call_kind,
                    "attempt": attempt,
                    "started_at": started_at,
                    "finished_at": now_iso(),
                    "request_body": copy.deepcopy(body),
                    "request_wire": body_record(request_body),
                    "response_status": response.status_code,
                    "response_headers": safe_headers(response.headers),
                    "response_body": payload,
                    "response_wire": body_record(response_body),
                    "tool_calls": response_tool_calls(payload),
                    "tool_results": [],
                    "budget_reservation": copy.deepcopy(reservation),
                }

                generation_id = response_generation_id(payload, response.headers)
                execution_evidence = billing_evidence(payload, response.headers)
                structure_error = response_structure_error(payload, response.status_code)
                payload_object = payload if isinstance(payload, dict) else {}
                reported_usage, usage_error = guarded_usage_fields(payload_object)
                if usage_error is not None:
                    record["usage_error"] = usage_error
                error_value = payload_object.get("error")
                error_obj = error_value if isinstance(error_value, dict) else {}
                http_error = None
                if response.status_code != 200:
                    http_error = str(
                        error_obj.get("message")
                        or response.text[:1000]
                        or response.status_code
                    )
                elif not generation_id:
                    http_error = "missing generation ID"
                await self.ledger.attempt(
                    {
                        "run_id": run_id,
                        "call_index": call_index,
                        "attempt": attempt,
                        "started_at": record["started_at"],
                        "finished_at": record["finished_at"],
                        "request_sha256": record["request_wire"]["sha256"],
                        "response_sha256": record["response_wire"]["sha256"],
                        "response_status": response.status_code,
                        "generation_id": generation_id or None,
                        "error": structure_error or http_error,
                    }
                )
                if structure_error is not None:
                    last_error = structure_error
                    record["error"] = last_error
                    record["terminal"] = True
                    record["response_malformed"] = True
                    if execution_evidence:
                        record["billing_ambiguous"] = True
                        record["billing_evidence"] = execution_evidence
                    append_jsonl(raw_path, record)
                    await self.ledger.settle(
                        run_id,
                        call_index,
                        {
                            **reported_usage,
                            "generation_id": generation_id or None,
                            "native_cost": generation_cost({}, reported_usage),
                        },
                        "unknown_inflight",
                    )
                    raise RuntimeError(
                        "malformed response was not resubmitted; reconciliation "
                        f"is required: {last_error}"
                    )
                if response.status_code == 200 and generation_id:
                    await self.ledger.response_received(run_id, call_index, generation_id)
                    record["event"] = "model_response"
                    record["generation"] = None
                    record["generation_error"] = None
                    record["terminal"] = True
                    append_jsonl(raw_path, record)
                    generation_error = None
                    try:
                        generation = await self.generation(
                            generation_id,
                            max_wait_seconds=(
                                GENERATION_METADATA_PROVISIONAL_WAIT_SECONDS
                                if allow_provisional_metadata
                                else GENERATION_METADATA_MAX_WAIT_SECONDS
                            ),
                        )
                    except RuntimeError as error:
                        generation = {}
                        generation_error = str(error)
                    append_jsonl(
                        raw_path,
                        {
                            "event": "generation_metadata",
                            "run_id": run_id,
                            "call_index": call_index,
                            "generation_id": generation_id,
                            "timestamp": now_iso(),
                            "generation": generation,
                            "error": generation_error,
                        },
                    )
                    native_cost = generation_cost(generation, reported_usage)
                    settlement_error = generation_error or usage_error
                    await self.ledger.settle(
                        run_id,
                        call_index,
                        {
                            "generation_id": generation_id,
                            "input_tokens": reported_usage["input_tokens"],
                            "output_tokens": reported_usage["output_tokens"],
                            "reasoning_tokens": reported_usage["reasoning_tokens"],
                            "native_cost": f"{native_cost:.10f}",
                            "openrouter_cost": f"{reported_usage['openrouter_cost']:.10f}",
                            "upstream_cost": f"{reported_usage['upstream_cost']:.10f}",
                            "is_byok": reported_usage["is_byok"],
                        },
                        "complete" if settlement_error is None else "metadata_pending",
                    )
                    if settlement_error is not None:
                        if allow_provisional_metadata and generation_error is not None:
                            return ModelCall(
                                payload=payload,
                                generation={},
                                call_index=call_index,
                                input_tokens=reported_usage["input_tokens"],
                                output_tokens=reported_usage["output_tokens"],
                                reasoning_tokens=reported_usage["reasoning_tokens"],
                                native_cost=native_cost,
                                openrouter_cost=reported_usage["openrouter_cost"],
                                upstream_cost=reported_usage["upstream_cost"],
                                is_byok=reported_usage["is_byok"],
                                metadata_status="metadata_pending",
                            )
                        raise RuntimeError(
                            "response metadata is pending; the received response "
                            f"must be repaired before resume: {settlement_error}"
                        )
                    return ModelCall(
                        payload=payload,
                        generation=generation,
                        call_index=call_index,
                        input_tokens=reported_usage["input_tokens"],
                        output_tokens=reported_usage["output_tokens"],
                        reasoning_tokens=reported_usage["reasoning_tokens"],
                        native_cost=native_cost,
                        openrouter_cost=reported_usage["openrouter_cost"],
                        upstream_cost=reported_usage["upstream_cost"],
                        is_byok=reported_usage["is_byok"],
                    )

                if execution_evidence:
                    # A response carrying output or metering evidence may already
                    # have incurred a charge even when its HTTP envelope is an
                    # error or omits the generation ID. Preserve it and require
                    # reconciliation. Never create a duplicate paid generation.
                    last_error = str(http_error or "ambiguous billed response")
                    record["error"] = last_error
                    record["terminal"] = True
                    record["billing_ambiguous"] = True
                    record["billing_evidence"] = execution_evidence
                    append_jsonl(raw_path, record)
                    await self.ledger.settle(
                        run_id,
                        call_index,
                        {
                            **reported_usage,
                            "generation_id": generation_id or None,
                            "native_cost": generation_cost({}, reported_usage),
                        },
                        "unknown_inflight",
                    )
                    raise RuntimeError(
                        "potentially billed response was not resubmitted; "
                        f"reconciliation is required: {last_error}"
                    )

                last_error = str(http_error or "request failed")
                record["error"] = last_error
                record["terminal"] = response.status_code not in TRANSIENT_STATUS or attempt == max_attempts
                append_jsonl(raw_path, record)
                if record["terminal"]:
                    break
            except (httpx.TimeoutException, httpx.TransportError) as error:
                last_error = f"{type(error).__name__}: {error}"
                error_record = {
                        "event": "model_call",
                        "run_id": run_id,
                        "call_index": call_index,
                        "call_kind": call_kind,
                        "attempt": attempt,
                        "started_at": started_at,
                        "finished_at": now_iso(),
                        "request_body": copy.deepcopy(body),
                        "request_wire": body_record(request_body),
                        "response_status": None,
                        "response_headers": {},
                        "response_body": {},
                        "response_wire": body_record(b""),
                        "error": last_error,
                        "terminal": True,
                        "tool_calls": [],
                        "tool_results": [],
                        "budget_reservation": copy.deepcopy(reservation),
                    }
                append_jsonl(raw_path, error_record)
                await self.ledger.attempt(
                    {
                        "run_id": run_id,
                        "call_index": call_index,
                        "attempt": attempt,
                        "started_at": error_record["started_at"],
                        "finished_at": error_record["finished_at"],
                        "request_sha256": error_record["request_wire"]["sha256"],
                        "response_sha256": error_record["response_wire"]["sha256"],
                        "response_status": None,
                        "generation_id": None,
                        "error": last_error,
                    }
                )
                await self.ledger.settle(
                    run_id,
                    call_index,
                    {"generation_id": None},
                    "unknown_inflight",
                )
                raise RuntimeError(
                    f"ambiguous transport failure; call was not resubmitted: {last_error}"
                ) from error
            delay = min(2 ** (attempt - 1), 20) + random.uniform(0.0, 0.75)
            await asyncio.sleep(delay)

        await self.ledger.settle(
            run_id,
            call_index,
            {
                "generation_id": "",
                "input_tokens": 0,
                "output_tokens": 0,
                "reasoning_tokens": 0,
                "native_cost": "0.0000000000",
                "openrouter_cost": "0.0000000000",
                "upstream_cost": "0.0000000000",
                "is_byok": False,
            },
            "failed",
        )
        raise RuntimeError(last_error)


def provider_body(model: str) -> dict:
    if model == "google/gemini-3.6-flash":
        return {
            "order": ["google-ai-studio"],
            "only": ["google-ai-studio"],
            "allow_fallbacks": False,
            "require_parameters": False,
        }
    if model == KIMI_MODEL:
        return {"allow_fallbacks": True, "require_parameters": True}
    return {"allow_fallbacks": True, "require_parameters": True}
