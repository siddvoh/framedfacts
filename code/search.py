from __future__ import annotations

import asyncio
import hashlib
import importlib.metadata
import json
import os
import re
import sqlite3
import ssl
import time
import uuid
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlsplit, urlunsplit

import certifi
import httpx

from config import (
    ARCHIVE,
    SEARCH_CACHE_PATH,
    SEARCH_RESULT_COUNT,
    SNIPPET_LIMIT,
    append_jsonl,
    canonical_json,
    now_iso,
    read_jsonl,
)


CACHE_PROTOCOL_V2 = "strict_v2"
CACHE_PROTOCOL_V3 = "strict_v3"
LEGACY_CACHE_TABLE = "search_cache"
V2_CACHE_TABLE = "search_cache_v2"
V3_CACHE_TABLE = "search_cache_v3"
WEB_CANDIDATE_COUNT = SEARCH_RESULT_COUNT * 2
SEARCH_BACKEND_ATTEMPT_EVENT = "search_backend_attempt"
SEARCH_BACKEND_SWITCH_EVENT = "backend_switch"
SEARCH_BACKEND_EVENT_SCHEMA = 1
SEARCH_FAILURE_WINDOW = 200
SEARCH_FAILURE_THRESHOLD_V2 = 0.05
SEARCH_FAILURE_THRESHOLD = 0.25
SEARCH_EXECUTION_PROTOCOL = "strict_v2_execution_v1"

# v3 execution is isolated from the sealed v2 cache and backend history. The v2
# constants and validators above and below remain the authoritative readers for
# the historical evidence.
SEARCH_EXECUTION_ID_PROTOCOL_V3 = "strict_v3_search_execution_id_v1"
SEARCH_EXECUTION_PROTOCOL_V3 = SEARCH_EXECUTION_ID_PROTOCOL_V3
SEARCH_BACKEND_EVENT_SCHEMA_V3 = 2
SEARCH_BACKEND_POLICY_TERMINAL_EVENT = "backend_policy_terminal"
SEARCH_TRANSPORT_ID_V3 = "strict_v3_ddgs_bing_v1"
SEARCH_BACKEND_HISTORY_ID_V3 = "search_backend_events_v3"
DDGS_ENGINE_V3 = "bing"
DDGS_REGION_V3 = "us-en"
DDGS_SAFESEARCH_V3 = "moderate"
DDGS_TIMELIMIT_V3 = None
DDGS_PAGE_V3 = 1
BRAVE_REQUEST_BUDGET = 900
DDGS_VERSION_V3 = "9.14.4"
CERTIFI_VERSION_V3 = "2026.2.25"
HTTPX_VERSION_V3 = "0.28.1"
MAX_SEARCH_ATTEMPTS_V3 = 4
WEB_MIN_INTERVAL_SECONDS_V3 = 1.5
WEB_PACING_JITTER_MAX_SECONDS_V3 = 0.25
WEB_RETRY_JITTER_MAX_SECONDS_V3 = 0.5
BRAVE_SOURCE_ENGINE_V3 = "brave_web_search_api"
SCHOLARLY_SOURCE_ENGINE_V3 = "ncbi_pubmed_wikipedia"
# Wikimedia rejects the default client agent, which failed the first
# scholarly window with HTTP 403 on every call.
SCHOLARLY_USER_AGENT = (
    "SycophanticSearchStudy/1.0 (academic research; contact in released README)"
)
# The first repaired window tripped 429s at three requests per second across
# two hosts, so the pace is halved and throttling is retried rather than fatal.
SCHOLARLY_MIN_INTERVAL_SECONDS = 1.0
SCHOLARLY_MAX_ATTEMPTS = 4
SCHOLARLY_BACKOFF_SECONDS = 2.0
V3_ATTEMPT_EVENT_FIELDS = frozenset(
    {
        "schema_version",
        "event",
        "history_id",
        "transport_id",
        "transport_sha256",
        "configured_web_backend",
        "search_execution_id",
        "attempt",
        "logical_backend",
        "source_backend",
        "source_engine",
        "transport_metadata",
        "pacing",
        "query",
        "started_at",
        "finished_at",
        "success",
        "candidate_results",
        "candidate_results_sha256",
        "error_type",
        "error",
    }
)


class SearchResultValidationError(RuntimeError):
    pass


class SearchBackendHistoryError(RuntimeError):
    pass


def url_identity(value: str) -> str | None:
    if not value or any(character.isspace() for character in value):
        return None
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError:
        return None
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
        return None
    host = parsed.hostname.lower()
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    default_port = (parsed.scheme.lower(), port) in {("http", 80), ("https", 443)}
    netloc = host if port is None or default_port else f"{host}:{port}"
    path = parsed.path or "/"
    return urlunsplit((parsed.scheme.lower(), netloc, path, parsed.query, ""))


def prepare_result_set(rows: Any) -> list[dict[str, str]]:
    """Filter an upstream response in rank order without padding or query changes."""
    if not isinstance(rows, list):
        raise SearchResultValidationError("search backend did not return a result list")
    prepared: list[dict[str, str]] = []
    seen_urls: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        title = row.get("title")
        url = row.get("url")
        snippet = row.get("snippet")
        if not all(isinstance(value, str) for value in (title, url, snippet)):
            continue
        title = title.strip()
        url = url.strip()
        # Slicing can expose whitespace at the clamp boundary. Strip again so
        # applying the frozen normalizer twice cannot change the result.
        snippet = snippet.strip()[:SNIPPET_LIMIT].rstrip()
        identity = url_identity(url)
        if not title or identity is None or identity in seen_urls:
            continue
        seen_urls.add(identity)
        prepared.append({"title": title, "url": url, "snippet": snippet})
        if len(prepared) == SEARCH_RESULT_COUNT:
            break
    if len(prepared) != SEARCH_RESULT_COUNT:
        raise SearchResultValidationError(
            f"search backend returned {len(prepared)} valid results; "
            f"expected {SEARCH_RESULT_COUNT}"
        )
    return prepared


def validate_result_set(rows: Any) -> list[dict[str, str]]:
    """Validate a stored/result payload without repairing or reordering it."""
    if not isinstance(rows, list):
        raise SearchResultValidationError("cached search result is not a list")
    if len(rows) != SEARCH_RESULT_COUNT:
        raise SearchResultValidationError(
            f"cached search result has {len(rows)} rows; expected {SEARCH_RESULT_COUNT}"
        )
    seen_urls: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != {"title", "url", "snippet"}:
            raise SearchResultValidationError(f"result {index} has an invalid schema")
        if not all(isinstance(row.get(field), str) for field in ("title", "url", "snippet")):
            raise SearchResultValidationError(f"result {index} has a non-string field")
        if not row["title"].strip():
            raise SearchResultValidationError(f"result {index} has an empty title")
        identity = url_identity(row["url"])
        if identity is None:
            raise SearchResultValidationError(f"result {index} has an invalid URL")
        if identity in seen_urls:
            raise SearchResultValidationError(f"result {index} repeats a URL")
        seen_urls.add(identity)
        if len(row["snippet"]) > SNIPPET_LIMIT:
            raise SearchResultValidationError(
                f"result {index} snippet exceeds {SNIPPET_LIMIT} characters"
            )
    return rows


def logical_backend(source_backend: str) -> str:
    if source_backend in {"ddgs", "brave"}:
        return "web"
    if source_backend == "scholarly":
        return "scholarly"
    raise ValueError(f"unknown source backend: {source_backend}")


def _event_timestamp(value: Any) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise SearchBackendHistoryError("backend event has an invalid timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise SearchBackendHistoryError(
            "backend event has an invalid timestamp"
        ) from error
    if parsed.tzinfo is None:
        raise SearchBackendHistoryError("backend event timestamp lacks a timezone")
    return parsed


@dataclass(frozen=True)
class SearchBackendHistory:
    attempts: int
    web_attempts: int
    switches: int
    current_web_backend: str
    web_failure_window: tuple[bool, ...]
    pending_web_switch: tuple[dict[str, Any], int, float] | None = None


def backend_switch_event(
    *,
    configured_backend: str,
    trigger: dict[str, Any],
    failed_attempts: int,
    failure_rate: float,
    timestamp: str,
) -> dict[str, Any]:
    """Build the sole allowed DDGS-to-Brave switch record."""
    return {
        "schema_version": SEARCH_BACKEND_EVENT_SCHEMA,
        "timestamp": timestamp,
        "event": SEARCH_BACKEND_SWITCH_EVENT,
        "configured_web_backend": configured_backend,
        "from": "ddgs",
        "to": "brave",
        "window_size": SEARCH_FAILURE_WINDOW,
        "observed_attempts": SEARCH_FAILURE_WINDOW,
        "failed_attempts": failed_attempts,
        "failure_rate": failure_rate,
        "trigger_search_execution_id": trigger["search_execution_id"],
        "trigger_attempt": trigger["attempt"],
        "last_error": trigger["error"],
    }


def validate_search_backend_history(
    rows: Any, *, configured_backend: str, allow_pending_switch: bool = False
) -> SearchBackendHistory:
    """Validate the append-only attempt log and reconstruct the fallback window."""
    if configured_backend not in {"ddgs", "brave"}:
        raise SearchBackendHistoryError(
            f"SEARCH_BACKEND must be ddgs or brave, found {configured_backend!r}"
        )
    if not isinstance(rows, list):
        raise SearchBackendHistoryError("backend history is not a list")

    current_backend = configured_backend
    failure_window: deque[bool] = deque(maxlen=SEARCH_FAILURE_WINDOW)
    seen_attempts: set[tuple[str, int, str]] = set()
    execution_state: dict[str, tuple[str, str, int, bool]] = {}
    attempt_count = 0
    web_attempt_count = 0
    switch_count = 0
    pending_switch: tuple[dict[str, Any], int, float] | None = None

    for index, row in enumerate(rows, 1):
        if not isinstance(row, dict):
            raise SearchBackendHistoryError(f"backend history row {index} is not an object")
        event_name = row.get("event")
        if pending_switch is not None and event_name != SEARCH_BACKEND_SWITCH_EVENT:
            raise SearchBackendHistoryError(
                f"backend history row {index} omits the required DDGS-to-Brave switch"
            )
        if row.get("schema_version") != SEARCH_BACKEND_EVENT_SCHEMA:
            raise SearchBackendHistoryError(
                f"backend history row {index} has an invalid schema version"
            )
        if row.get("configured_web_backend") != configured_backend:
            raise SearchBackendHistoryError(
                f"backend history row {index} changes the configured web backend"
            )

        if event_name == SEARCH_BACKEND_ATTEMPT_EVENT:
            execution_id = row.get("search_execution_id")
            attempt = row.get("attempt")
            logical = row.get("logical_backend")
            source = row.get("source_backend")
            if (
                not isinstance(execution_id, str)
                or len(execution_id) != 32
                or any(character not in "0123456789abcdef" for character in execution_id)
                or not isinstance(attempt, int)
                or not 1 <= attempt <= 4
            ):
                raise SearchBackendHistoryError(
                    f"backend history row {index} has invalid attempt identity"
                )
            key = (execution_id, attempt, str(logical))
            if key in seen_attempts:
                raise SearchBackendHistoryError(
                    f"backend history row {index} duplicates attempt {key!r}"
                )
            seen_attempts.add(key)
            if not (
                (logical == "web" and source in {"ddgs", "brave"})
                or (logical == "scholarly" and source == "scholarly")
            ):
                raise SearchBackendHistoryError(
                    f"backend history row {index} has an invalid backend pair"
                )
            if logical == "web" and source != current_backend:
                raise SearchBackendHistoryError(
                    f"backend history row {index} does not use active backend {current_backend}"
                )
            if not isinstance(row.get("query"), str) or not row["query"].strip():
                raise SearchBackendHistoryError(
                    f"backend history row {index} has an invalid query"
                )
            prior_execution = execution_state.get(execution_id)
            if prior_execution is None:
                if attempt != 1:
                    raise SearchBackendHistoryError(
                        f"backend history row {index} does not start at attempt 1"
                    )
            elif (
                prior_execution[0] != logical
                or prior_execution[1] != row["query"]
                or attempt != prior_execution[2] + 1
                or prior_execution[3]
            ):
                raise SearchBackendHistoryError(
                    f"backend history row {index} changes or resumes a search execution"
                )
            started = _event_timestamp(row.get("started_at"))
            finished = _event_timestamp(row.get("finished_at"))
            if finished < started:
                raise SearchBackendHistoryError(
                    f"backend history row {index} has out-of-order timestamps"
                )
            success = row.get("success")
            if not isinstance(success, bool):
                raise SearchBackendHistoryError(
                    f"backend history row {index} has an invalid outcome"
                )
            candidates = row.get("candidate_results")
            candidate_hash = row.get("candidate_results_sha256")
            if candidates is None:
                if candidate_hash is not None or success:
                    raise SearchBackendHistoryError(
                        f"backend history row {index} has inconsistent candidates"
                    )
            elif not isinstance(candidates, list) or candidate_hash != hashlib.sha256(
                canonical_json(candidates).encode()
            ).hexdigest():
                raise SearchBackendHistoryError(
                    f"backend history row {index} has a candidate hash mismatch"
                )
            if success:
                try:
                    prepare_result_set(candidates)
                except SearchResultValidationError as error:
                    raise SearchBackendHistoryError(
                        f"backend history row {index} marks an invalid result set successful"
                    ) from error
                if row.get("error_type") is not None or row.get("error") is not None:
                    raise SearchBackendHistoryError(
                        f"backend history row {index} gives a successful attempt an error"
                    )
            else:
                if isinstance(candidates, list):
                    try:
                        prepare_result_set(candidates)
                    except SearchResultValidationError:
                        pass
                    else:
                        raise SearchBackendHistoryError(
                            f"backend history row {index} rejects a valid result set"
                        )
                if any(
                    not isinstance(row.get(field_name), str)
                    or not row[field_name].strip()
                    for field_name in ("error_type", "error")
                ):
                    raise SearchBackendHistoryError(
                        f"backend history row {index} lacks failure provenance"
                    )
            execution_state[execution_id] = (
                str(logical), str(row["query"]), attempt, success
            )
            attempt_count += 1
            if logical == "web":
                web_attempt_count += 1
                failure_window.append(not success)
                failures = sum(failure_window)
                rate = failures / len(failure_window)
                if (
                    current_backend == "ddgs"
                    and len(failure_window) == SEARCH_FAILURE_WINDOW
                    and rate > SEARCH_FAILURE_THRESHOLD_V2
                ):
                    pending_switch = (row, failures, rate)
            continue

        if event_name != SEARCH_BACKEND_SWITCH_EVENT:
            raise SearchBackendHistoryError(
                f"backend history row {index} has unknown event {event_name!r}"
            )
        if pending_switch is None:
            raise SearchBackendHistoryError(
                f"backend history row {index} records an unjustified switch"
            )
        trigger, failed_attempts, failure_rate = pending_switch
        switched_at = _event_timestamp(row.get("timestamp"))
        trigger_finished = _event_timestamp(trigger.get("finished_at"))
        if switched_at < trigger_finished:
            raise SearchBackendHistoryError(
                f"backend history row {index} switch timestamp is out of order"
            )
        expected = {
            "from": "ddgs",
            "to": "brave",
            "window_size": SEARCH_FAILURE_WINDOW,
            "observed_attempts": SEARCH_FAILURE_WINDOW,
            "failed_attempts": failed_attempts,
            "failure_rate": failure_rate,
            "trigger_search_execution_id": trigger["search_execution_id"],
            "trigger_attempt": trigger["attempt"],
            "last_error": trigger["error"],
        }
        if any(row.get(field_name) != value for field_name, value in expected.items()):
            raise SearchBackendHistoryError(
                f"backend history row {index} switch disagrees with its 200-attempt window"
            )
        current_backend = "brave"
        switch_count += 1
        pending_switch = None

    if pending_switch is not None and not allow_pending_switch:
        raise SearchBackendHistoryError(
            "backend history ends before the required DDGS-to-Brave switch"
        )
    return SearchBackendHistory(
        attempts=attempt_count,
        web_attempts=web_attempt_count,
        switches=switch_count,
        current_web_backend=current_backend,
        web_failure_window=tuple(failure_window),
        pending_web_switch=pending_switch,
    )


def _package_version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError as error:
        raise RuntimeError(f"required package {distribution!r} is not installed") from error


def certifi_bundle_sha256_v3() -> str:
    import certifi

    return hashlib.sha256(Path(certifi.where()).read_bytes()).hexdigest()


def validate_search_transport_runtime_v3() -> dict[str, str]:
    expected = {
        "ddgs": DDGS_VERSION_V3,
        "certifi": CERTIFI_VERSION_V3,
        "httpx": HTTPX_VERSION_V3,
    }
    installed = {name: _package_version(name) for name in expected}
    mismatches = [
        f"{name}=={installed[name]} (expected {version})"
        for name, version in expected.items()
        if installed[name] != version
    ]
    if mismatches:
        raise RuntimeError("search transport dependency drift: " + ", ".join(mismatches))
    return {**installed, "certifi_ca_bundle_sha256": certifi_bundle_sha256_v3()}


def search_transport_projection_v3() -> dict[str, Any]:
    runtime = validate_search_transport_runtime_v3()
    return {
        "schema_version": 1,
        "kind": "search_transport_v3",
        "transport_id": SEARCH_TRANSPORT_ID_V3,
        "history_id": SEARCH_BACKEND_HISTORY_ID_V3,
        "cache_protocol": CACHE_PROTOCOL_V3,
        "cache_table": V3_CACHE_TABLE,
        "event_schema_version": SEARCH_BACKEND_EVENT_SCHEMA_V3,
        "web": {
            "primary": {
                "source_backend": "ddgs",
                "source_engine": DDGS_ENGINE_V3,
                "package": "ddgs",
                "package_version": runtime["ddgs"],
                "verify": "certifi.where()",
                "certifi_package_version": runtime["certifi"],
                "region": DDGS_REGION_V3,
                "safesearch": DDGS_SAFESEARCH_V3,
                "timelimit": DDGS_TIMELIMIT_V3,
                "page": DDGS_PAGE_V3,
                "max_results": WEB_CANDIDATE_COUNT,
            },
            "fallback": {
                "source_backend": "brave",
                "source_engine": BRAVE_SOURCE_ENGINE_V3,
                "endpoint": "https://api.search.brave.com/res/v1/web/search",
                "client_package": "httpx",
                "client_package_version": runtime["httpx"],
                "verify": "certifi.where()",
                "certifi_package_version": runtime["certifi"],
                "activation_window_attempts": SEARCH_FAILURE_WINDOW,
                "activation_failure_rate_strictly_greater_than": (
                    SEARCH_FAILURE_THRESHOLD
                ),
                "credential_environment_variable": "BRAVE_API_KEY",
            },
            "same_query_max_attempts": MAX_SEARCH_ATTEMPTS_V3,
            "minimum_interval_seconds": WEB_MIN_INTERVAL_SECONDS_V3,
            "pacing_jitter_max_seconds": WEB_PACING_JITTER_MAX_SECONDS_V3,
            "retry_jitter_max_seconds": WEB_RETRY_JITTER_MAX_SECONDS_V3,
        },
        "result_count": SEARCH_RESULT_COUNT,
        "snippet_limit": SNIPPET_LIMIT,
    }


def search_transport_sha256_v3() -> str:
    return hashlib.sha256(
        canonical_json(search_transport_projection_v3()).encode()
    ).hexdigest()


def source_transport_metadata_v3(source_backend: str) -> dict[str, Any]:
    projection = search_transport_projection_v3()
    if source_backend == "ddgs":
        return dict(projection["web"]["primary"])
    if source_backend == "brave":
        return dict(projection["web"]["fallback"])
    if source_backend == "scholarly":
        runtime = validate_search_transport_runtime_v3()
        return {
            "source_backend": "scholarly",
            "source_engine": SCHOLARLY_SOURCE_ENGINE_V3,
            "client_package": "httpx",
            "client_package_version": runtime["httpx"],
            "verify": "certifi.where()",
            "certifi_package_version": runtime["certifi"],
            "ca_bundle_sha256": runtime["certifi_ca_bundle_sha256"],
            "pubmed_endpoint": (
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
            ),
            "wikipedia_endpoint": "https://en.wikipedia.org/w/api.php",
        }
    raise ValueError(f"unknown source backend: {source_backend}")


def _stable_unit_interval_v3(identity: str) -> tuple[str, float]:
    digest = hashlib.sha256(identity.encode()).hexdigest()
    return digest, int(digest[:16], 16) / float(0xFFFFFFFFFFFFFFFF)


def expected_web_pacing_v3(execution_id: str, attempt: int) -> dict[str, Any]:
    pacing_identity = (
        f"{SEARCH_TRANSPORT_ID_V3}|{execution_id}|{attempt}|web-pacing"
    )
    pacing_hash, pacing_fraction = _stable_unit_interval_v3(pacing_identity)
    retry_identity = (
        f"{SEARCH_TRANSPORT_ID_V3}|{execution_id}|{attempt}|retry-backoff"
    )
    retry_hash, retry_fraction = _stable_unit_interval_v3(retry_identity)
    retry_base = 0.0 if attempt == 1 else float(min(2 ** (attempt - 2), 8))
    return {
        "minimum_interval_seconds": WEB_MIN_INTERVAL_SECONDS_V3,
        "pacing_jitter_seconds": round(
            pacing_fraction * WEB_PACING_JITTER_MAX_SECONDS_V3, 9
        ),
        "pacing_identity_sha256": pacing_hash,
        "retry_backoff_seconds": round(
            retry_base + retry_fraction * WEB_RETRY_JITTER_MAX_SECONDS_V3, 9
        )
        if attempt > 1
        else 0.0,
        "retry_identity_sha256": retry_hash,
    }


@dataclass(frozen=True)
class SearchBackendHistoryV3:
    attempts: int
    web_attempts: int
    switches: int
    current_web_backend: str
    web_failure_window: tuple[bool, ...]
    terminal_latched: bool
    terminal_reason: str | None
    pending_web_policy: tuple[dict[str, Any], int, float] | None = None


def _credential_check_v3(nonempty: bool) -> dict[str, Any]:
    return {
        "environment_variable": "BRAVE_API_KEY",
        "nonempty": nonempty,
    }


def backend_policy_event_v3(
    *,
    trigger: dict[str, Any],
    failed_attempts: int,
    failure_rate: float,
    timestamp: str,
    brave_key_nonempty: bool,
) -> dict[str, Any]:
    switched = brave_key_nonempty and trigger["source_backend"] == "ddgs"
    terminal_reason = (
        None
        if switched
        else (
            "brave_api_key_missing"
            if trigger["source_backend"] == "ddgs"
            else "brave_failure_rate_exceeded"
        )
    )
    event = {
        "schema_version": SEARCH_BACKEND_EVENT_SCHEMA_V3,
        "event": (
            SEARCH_BACKEND_SWITCH_EVENT
            if switched
            else SEARCH_BACKEND_POLICY_TERMINAL_EVENT
        ),
        "history_id": SEARCH_BACKEND_HISTORY_ID_V3,
        "transport_id": SEARCH_TRANSPORT_ID_V3,
        "transport_sha256": search_transport_sha256_v3(),
        "timestamp": timestamp,
        "configured_web_backend": "ddgs",
        "from": trigger["source_backend"],
        "to": "brave" if switched else None,
        "window_size": SEARCH_FAILURE_WINDOW,
        "observed_attempts": SEARCH_FAILURE_WINDOW,
        "failed_attempts": failed_attempts,
        "failure_rate": failure_rate,
        "trigger_search_execution_id": trigger["search_execution_id"],
        "trigger_attempt": trigger["attempt"],
        "last_error": trigger["error"],
        "credential_check": _credential_check_v3(brave_key_nonempty),
    }
    if terminal_reason is not None:
        event["reason"] = terminal_reason
    return event


def _validate_pacing_v3(row: dict[str, Any], index: int) -> None:
    pacing = row.get("pacing")
    if row.get("logical_backend") != "web":
        if pacing is not None:
            raise SearchBackendHistoryError(
                f"backend history row {index} gives scholarly search web pacing"
            )
        return
    if not isinstance(pacing, dict):
        raise SearchBackendHistoryError(
            f"backend history row {index} lacks pacing provenance"
        )
    expected = expected_web_pacing_v3(
        str(row.get("search_execution_id")), int(row.get("attempt", 0))
    )
    for key, value in expected.items():
        if pacing.get(key) != value:
            raise SearchBackendHistoryError(
                f"backend history row {index} has invalid deterministic pacing"
            )
    scheduled = pacing.get("scheduled_wait_seconds")
    if not isinstance(scheduled, (int, float)) or scheduled < 0:
        raise SearchBackendHistoryError(
            f"backend history row {index} has invalid scheduled wait"
        )


def validate_search_backend_history_v3(
    rows: Any, *, allow_pending_policy: bool = False
) -> SearchBackendHistoryV3:
    """Validate prospective strict-v3 history without consulting sealed v2 rows."""
    if not isinstance(rows, list):
        raise SearchBackendHistoryError("backend history is not a list")
    current_backend = "ddgs"
    failure_window: deque[bool] = deque(maxlen=SEARCH_FAILURE_WINDOW)
    attempts = 0
    web_attempts = 0
    switches = 0
    terminal_latched = False
    terminal_reason: str | None = None
    pending: tuple[dict[str, Any], int, float] | None = None
    seen_attempts: set[tuple[str, int, str]] = set()
    execution_state: dict[str, tuple[str, str, int, bool]] = {}
    transport_hash = search_transport_sha256_v3()

    for index, row in enumerate(rows, 1):
        if not isinstance(row, dict):
            raise SearchBackendHistoryError(f"backend history row {index} is not an object")
        event_name = row.get("event")
        if pending is not None and event_name not in {
            SEARCH_BACKEND_SWITCH_EVENT,
            SEARCH_BACKEND_POLICY_TERMINAL_EVENT,
        }:
            raise SearchBackendHistoryError(
                f"backend history row {index} omits the required backend policy event"
            )
        common = {
            "schema_version": SEARCH_BACKEND_EVENT_SCHEMA_V3,
            "history_id": SEARCH_BACKEND_HISTORY_ID_V3,
            "transport_id": SEARCH_TRANSPORT_ID_V3,
            "transport_sha256": transport_hash,
            "configured_web_backend": "ddgs",
        }
        if any(row.get(key) != value for key, value in common.items()):
            raise SearchBackendHistoryError(
                f"backend history row {index} changes strict-v3 transport identity"
            )
        if terminal_latched and event_name != SEARCH_BACKEND_ATTEMPT_EVENT:
            raise SearchBackendHistoryError(
                f"backend history row {index} occurs after terminal policy latch"
            )
        if terminal_latched:
            # Attempts already in flight when the latch fired still record
            # honestly. They change nothing about the latched policy.
            attempts += 1
            continue

        if event_name == SEARCH_BACKEND_ATTEMPT_EVENT:
            if set(row) != V3_ATTEMPT_EVENT_FIELDS:
                raise SearchBackendHistoryError(
                    f"backend history row {index} has an invalid strict-v3 event shape"
                )
            execution_id = row.get("search_execution_id")
            attempt = row.get("attempt")
            logical = row.get("logical_backend")
            source = row.get("source_backend")
            engine = row.get("source_engine")
            if (
                not isinstance(execution_id, str)
                or len(execution_id) != 32
                or any(character not in "0123456789abcdef" for character in execution_id)
                or not isinstance(attempt, int)
                or not 1 <= attempt <= MAX_SEARCH_ATTEMPTS_V3
            ):
                raise SearchBackendHistoryError(
                    f"backend history row {index} has invalid attempt identity"
                )
            pair = (logical, source, engine)
            if pair not in {
                ("web", "ddgs", DDGS_ENGINE_V3),
                ("web", "brave", BRAVE_SOURCE_ENGINE_V3),
                ("scholarly", "scholarly", SCHOLARLY_SOURCE_ENGINE_V3),
            }:
                raise SearchBackendHistoryError(
                    f"backend history row {index} has an invalid backend/engine pair"
                )
            if logical == "web" and source != current_backend:
                raise SearchBackendHistoryError(
                    f"backend history row {index} does not use active backend {current_backend}"
                )
            if row.get("transport_metadata") != source_transport_metadata_v3(str(source)):
                raise SearchBackendHistoryError(
                    f"backend history row {index} has transport metadata drift"
                )
            _validate_pacing_v3(row, index)
            query = row.get("query")
            if not isinstance(query, str) or not query.strip():
                raise SearchBackendHistoryError(
                    f"backend history row {index} has an invalid query"
                )
            key = (execution_id, attempt, str(logical))
            if key in seen_attempts:
                raise SearchBackendHistoryError(
                    f"backend history row {index} duplicates attempt {key!r}"
                )
            seen_attempts.add(key)
            previous = execution_state.get(execution_id)
            if previous is None and attempt != 1:
                raise SearchBackendHistoryError(
                    f"backend history row {index} does not start at attempt 1"
                )
            if previous is not None and (
                previous[0] != logical
                or previous[1] != query
                or attempt != previous[2] + 1
                or previous[3]
            ):
                raise SearchBackendHistoryError(
                    f"backend history row {index} changes or resumes a search execution"
                )
            started = _event_timestamp(row.get("started_at"))
            finished = _event_timestamp(row.get("finished_at"))
            if finished < started:
                raise SearchBackendHistoryError(
                    f"backend history row {index} has out-of-order timestamps"
                )
            success = row.get("success")
            if not isinstance(success, bool):
                raise SearchBackendHistoryError(
                    f"backend history row {index} has an invalid outcome"
                )
            candidates = row.get("candidate_results")
            candidate_hash = row.get("candidate_results_sha256")
            if candidates is None:
                if candidate_hash is not None or success:
                    raise SearchBackendHistoryError(
                        f"backend history row {index} has inconsistent candidates"
                    )
            elif not isinstance(candidates, list) or candidate_hash != hashlib.sha256(
                canonical_json(candidates).encode()
            ).hexdigest():
                raise SearchBackendHistoryError(
                    f"backend history row {index} has a candidate hash mismatch"
                )
            if success:
                try:
                    prepare_result_set(candidates)
                except SearchResultValidationError as error:
                    raise SearchBackendHistoryError(
                        f"backend history row {index} marks invalid results successful"
                    ) from error
                if row.get("error_type") is not None or row.get("error") is not None:
                    raise SearchBackendHistoryError(
                        f"backend history row {index} gives success an error"
                    )
            else:
                if isinstance(candidates, list):
                    try:
                        prepare_result_set(candidates)
                    except SearchResultValidationError:
                        pass
                    else:
                        raise SearchBackendHistoryError(
                            f"backend history row {index} rejects valid results"
                        )
                if any(
                    not isinstance(row.get(field), str) or not row[field].strip()
                    for field in ("error_type", "error")
                ):
                    raise SearchBackendHistoryError(
                        f"backend history row {index} lacks failure provenance"
                    )
            execution_state[execution_id] = (str(logical), query, attempt, success)
            attempts += 1
            if logical == "web":
                web_attempts += 1
                failure_window.append(not success)
                failures = sum(failure_window)
                failure_rate = failures / len(failure_window)
                if (
                    len(failure_window) == SEARCH_FAILURE_WINDOW
                    and failure_rate > SEARCH_FAILURE_THRESHOLD
                ):
                    pending = (row, failures, failure_rate)
            continue

        if event_name not in {
            SEARCH_BACKEND_SWITCH_EVENT,
            SEARCH_BACKEND_POLICY_TERMINAL_EVENT,
        }:
            raise SearchBackendHistoryError(
                f"backend history row {index} has unknown event {event_name!r}"
            )
        if pending is None:
            raise SearchBackendHistoryError(
                f"backend history row {index} records unjustified policy"
            )
        trigger, failures, failure_rate = pending
        key_nonempty = bool(row.get("credential_check", {}).get("nonempty"))
        expected = backend_policy_event_v3(
            trigger=trigger,
            failed_attempts=failures,
            failure_rate=failure_rate,
            timestamp=str(row.get("timestamp")),
            brave_key_nonempty=key_nonempty,
        )
        if row != expected:
            raise SearchBackendHistoryError(
                f"backend history row {index} disagrees with its 200-attempt window"
            )
        if _event_timestamp(row["timestamp"]) < _event_timestamp(trigger["finished_at"]):
            raise SearchBackendHistoryError(
                f"backend history row {index} policy timestamp is out of order"
            )
        if event_name == SEARCH_BACKEND_SWITCH_EVENT:
            current_backend = "brave"
            switches += 1
            failure_window.clear()
        else:
            terminal_latched = True
            terminal_reason = str(row["reason"])
        pending = None

    if pending is not None and not allow_pending_policy:
        raise SearchBackendHistoryError(
            "backend history ends before the required backend policy event"
        )
    return SearchBackendHistoryV3(
        attempts=attempts,
        web_attempts=web_attempts,
        switches=switches,
        current_web_backend=current_backend,
        web_failure_window=tuple(failure_window),
        terminal_latched=terminal_latched,
        terminal_reason=terminal_reason,
        pending_web_policy=pending,
    )


def initialize_v3_cache_schema(connection: sqlite3.Connection) -> None:
    connection.execute(
        f"""CREATE TABLE IF NOT EXISTS {V3_CACHE_TABLE}(
            logical_backend TEXT NOT NULL CHECK(logical_backend IN ('web','scholarly')),
            query_key TEXT NOT NULL,
            query TEXT NOT NULL,
            source_backend TEXT NOT NULL CHECK(source_backend IN ('ddgs','brave','scholarly')),
            source_engine TEXT NOT NULL,
            results_json TEXT NOT NULL,
            fetched_at TEXT NOT NULL,
            transport_id TEXT NOT NULL CHECK(transport_id='{SEARCH_TRANSPORT_ID_V3}'),
            backend_attempt_sha256 TEXT NOT NULL,
            CHECK(
                (source_backend='ddgs' AND source_engine='{DDGS_ENGINE_V3}') OR
                (source_backend='brave' AND source_engine='{BRAVE_SOURCE_ENGINE_V3}') OR
                (source_backend='scholarly' AND source_engine='{SCHOLARLY_SOURCE_ENGINE_V3}')
            ),
            PRIMARY KEY(logical_backend, query_key)
        )"""
    )
    connection.execute(
        f"""CREATE TRIGGER IF NOT EXISTS {V3_CACHE_TABLE}_no_update
        BEFORE UPDATE ON {V3_CACHE_TABLE}
        BEGIN SELECT RAISE(ABORT, 'v3 search cache is append-only'); END"""
    )
    connection.execute(
        f"""CREATE TRIGGER IF NOT EXISTS {V3_CACHE_TABLE}_no_delete
        BEFORE DELETE ON {V3_CACHE_TABLE}
        BEGIN SELECT RAISE(ABORT, 'v3 search cache is append-only'); END"""
    )


def canonical_v3_cache_table_projection() -> dict[str, Any]:
    connection = sqlite3.connect(":memory:")
    try:
        initialize_v3_cache_schema(connection)
        return project_cache_table_connection(connection, V3_CACHE_TABLE)
    finally:
        connection.close()


def validate_v3_cache_schema(
    connection: sqlite3.Connection, *, require_empty: bool = False
) -> dict[str, Any]:
    actual = project_cache_table_connection(connection, V3_CACHE_TABLE)
    canonical = canonical_v3_cache_table_projection()
    if not actual["exists"] or any(
        actual[field] != canonical[field]
        for field in ("schema_sha256", "triggers_sha256")
    ):
        raise RuntimeError("strict-v3 cache schema or append-only triggers are noncanonical")
    if require_empty and actual["row_count"] != 0:
        raise RuntimeError("strict-v3 cache table must contain zero rows")
    return actual


def _read_only_connection(cache_path: Path) -> sqlite3.Connection:
    if not cache_path.exists():
        raise FileNotFoundError(cache_path)
    return sqlite3.connect(f"file:{cache_path.resolve()}?mode=ro", uri=True)


def project_cache_table_connection(
    connection: sqlite3.Connection, table: str
) -> dict[str, Any]:
    if table not in {LEGACY_CACHE_TABLE, V2_CACHE_TABLE, V3_CACHE_TABLE}:
        raise ValueError(f"unsupported cache table projection: {table}")
    schema = connection.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (table,)
    ).fetchone()
    if schema is None:
        projection = {
            "table": table,
            "exists": False,
            "schema_sha256": None,
            "triggers_sha256": hashlib.sha256(canonical_json([]).encode()).hexdigest(),
            "row_count": 0,
            "rows_sha256": hashlib.sha256(canonical_json([]).encode()).hexdigest(),
        }
    else:
        columns = [
            str(row[1]) for row in connection.execute(f"PRAGMA table_info({table})")
        ]
        order = ",".join(f'"{column}"' for column in columns)
        rows = [
            list(row)
            for row in connection.execute(
                f"SELECT {order} FROM {table} ORDER BY {order}"
            ).fetchall()
        ]
        triggers = connection.execute(
            "SELECT name,sql FROM sqlite_master WHERE type='trigger' "
            "AND tbl_name=? ORDER BY name",
            (table,),
        ).fetchall()
        projection = {
            "table": table,
            "exists": True,
            "schema_sha256": hashlib.sha256(str(schema[0]).encode()).hexdigest(),
            "triggers_sha256": hashlib.sha256(
                canonical_json([list(row) for row in triggers]).encode()
            ).hexdigest(),
            "row_count": len(rows),
            "rows_sha256": hashlib.sha256(canonical_json(rows).encode()).hexdigest(),
        }
    return {
        **projection,
        "projection_sha256": hashlib.sha256(canonical_json(projection).encode()).hexdigest(),
    }


def project_cache_table(cache_path: Path, table: str) -> dict[str, Any]:
    with _read_only_connection(cache_path) as connection:
        return project_cache_table_connection(connection, table)


def initialize_cache_schema(connection: sqlite3.Connection) -> None:
    connection.execute(
        f"""CREATE TABLE IF NOT EXISTS {LEGACY_CACHE_TABLE}(
            backend TEXT NOT NULL,
            query_key TEXT NOT NULL,
            query TEXT NOT NULL,
            results_json TEXT NOT NULL,
            fetched_at TEXT NOT NULL,
            PRIMARY KEY(backend, query_key)
        )"""
    )
    connection.execute(
        f"""CREATE TABLE IF NOT EXISTS {V2_CACHE_TABLE}(
            logical_backend TEXT NOT NULL CHECK(logical_backend IN ('web','scholarly')),
            query_key TEXT NOT NULL,
            query TEXT NOT NULL,
            source_backend TEXT NOT NULL CHECK(source_backend IN ('ddgs','brave','scholarly')),
            results_json TEXT NOT NULL,
            fetched_at TEXT NOT NULL,
            origin TEXT NOT NULL CHECK(origin IN ('legacy_v1','live_v2')),
            legacy_backend TEXT,
            legacy_rowid INTEGER,
            PRIMARY KEY(logical_backend, query_key),
            UNIQUE(legacy_backend, legacy_rowid)
        )"""
    )
    connection.execute(
        f"""CREATE TRIGGER IF NOT EXISTS {V2_CACHE_TABLE}_no_update
        BEFORE UPDATE ON {V2_CACHE_TABLE}
        BEGIN SELECT RAISE(ABORT, 'v2 search cache is append-only'); END"""
    )
    connection.execute(
        f"""CREATE TRIGGER IF NOT EXISTS {V2_CACHE_TABLE}_no_delete
        BEFORE DELETE ON {V2_CACHE_TABLE}
        BEGIN SELECT RAISE(ABORT, 'v2 search cache is append-only'); END"""
    )


def legacy_rows_sha256(rows: list[tuple[Any, ...]]) -> str:
    return hashlib.sha256(canonical_json([list(row) for row in rows]).encode()).hexdigest()


@dataclass(frozen=True)
class MigrationSummary:
    legacy_rows: int
    eligible_rows: int
    invalid_rows: int
    collision_rows: int
    existing_v2_rows: int
    inserted_rows: int
    legacy_rows_sha256: str
    migrated_rows: int
    migrated_rows_sha256: str
    invalid_keys: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "protocol": CACHE_PROTOCOL_V2,
            "legacy_rows": self.legacy_rows,
            "eligible_rows": self.eligible_rows,
            "invalid_rows": self.invalid_rows,
            "collision_rows": self.collision_rows,
            "existing_v2_rows": self.existing_v2_rows,
            "inserted_rows": self.inserted_rows,
            "legacy_rows_sha256": self.legacy_rows_sha256,
            "migrated_rows": self.migrated_rows,
            "migrated_rows_sha256": self.migrated_rows_sha256,
            "invalid_keys": list(self.invalid_keys),
        }


def migrate_valid_legacy_rows(cache_path: Path, *, apply: bool) -> MigrationSummary:
    if not cache_path.exists():
        if not apply:
            raise FileNotFoundError(cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(cache_path)
    try:
        if apply:
            connection.execute("BEGIN IMMEDIATE")
            initialize_cache_schema(connection)
        elif connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
            (LEGACY_CACHE_TABLE,),
        ).fetchone() is None:
            raise RuntimeError("legacy search cache table is absent")
        legacy_rows = connection.execute(
            f"SELECT rowid,backend,query_key,query,results_json,fetched_at "
            f"FROM {LEGACY_CACHE_TABLE} ORDER BY rowid"
        ).fetchall()
        before_hash = legacy_rows_sha256(legacy_rows)
        table_exists = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
            (V2_CACHE_TABLE,),
        ).fetchone() is not None
        existing_v2 = (
            connection.execute(f"SELECT COUNT(*) FROM {V2_CACHE_TABLE}").fetchone()[0]
            if table_exists
            else 0
        )
        occupied = set()
        if table_exists:
            for backend, query_key, results_json in connection.execute(
                f"SELECT logical_backend,query_key,results_json FROM {V2_CACHE_TABLE}"
            ):
                validate_result_set(json.loads(results_json))
                occupied.add((str(backend), str(query_key)))
        eligible = 0
        invalid: list[str] = []
        collisions = 0
        inserted = 0
        for rowid, backend, query_key, query, results_json, fetched_at in legacy_rows:
            try:
                namespace = logical_backend(str(backend))
                results = json.loads(results_json)
                validate_result_set(results)
                if not isinstance(query, str) or not query.strip():
                    raise SearchResultValidationError("empty source query")
                datetime.fromisoformat(str(fetched_at).replace("Z", "+00:00"))
            except (ValueError, TypeError, json.JSONDecodeError, SearchResultValidationError):
                invalid.append(f"{backend}:{query_key}")
                continue
            key = (namespace, str(query_key))
            eligible += 1
            if key in occupied:
                collisions += 1
                continue
            occupied.add(key)
            if apply:
                cursor = connection.execute(
                    f"INSERT OR IGNORE INTO {V2_CACHE_TABLE} VALUES(?,?,?,?,?,?,?,?,?)",
                    (
                        namespace,
                        query_key,
                        query,
                        backend,
                        results_json,
                        fetched_at,
                        "legacy_v1",
                        backend,
                        rowid,
                    ),
                )
                inserted += cursor.rowcount
        after_rows = connection.execute(
            f"SELECT rowid,backend,query_key,query,results_json,fetched_at "
            f"FROM {LEGACY_CACHE_TABLE} ORDER BY rowid"
        ).fetchall()
        if legacy_rows_sha256(after_rows) != before_hash:
            raise RuntimeError("legacy cache rows changed during v2 initialization")
        if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise RuntimeError("search cache failed integrity_check")
        migrated_rows = (
            connection.execute(
                f"SELECT logical_backend,query_key,query,source_backend,results_json,"
                f"fetched_at,origin,legacy_backend,legacy_rowid FROM {V2_CACHE_TABLE} "
                "WHERE origin='legacy_v1' ORDER BY logical_backend,query_key"
            ).fetchall()
            if table_exists or apply
            else []
        )
        if apply:
            connection.commit()
        return MigrationSummary(
            legacy_rows=len(legacy_rows),
            eligible_rows=eligible,
            invalid_rows=len(invalid),
            collision_rows=collisions,
            existing_v2_rows=int(existing_v2),
            inserted_rows=inserted,
            legacy_rows_sha256=before_hash,
            migrated_rows=len(migrated_rows),
            migrated_rows_sha256=legacy_rows_sha256(migrated_rows),
            invalid_keys=tuple(invalid),
        )
    except Exception:
        if apply:
            connection.rollback()
        raise
    finally:
        connection.close()


TRAILING_PUNCTUATION = re.compile(r"[^\w\s]+$")
SEARCH_ATTEMPT_REJECTED_EVENT = "search_attempt_rejected"
SEARCH_QUERY_EXHAUSTED_STATUS = "query_exhausted"
SEARCH_QUERY_EXHAUSTION_PROTOCOL_V5 = "v5_visible_search_query_exhaustion_v1"
SearchAttemptRecorder = Callable[[dict[str, Any]], None]
SearchProvenanceRecorder = Callable[[dict[str, Any]], None]


def normalize_query(query: str) -> str:
    collapsed = " ".join(query.lower().split())
    return TRAILING_PUNCTUATION.sub("", collapsed).strip()


def clamp_result(result: dict) -> dict:
    return {
        "title": str(result.get("title") or "").strip(),
        "url": str(result.get("url") or "").strip(),
        "snippet": str(result.get("snippet") or "").strip()[:SNIPPET_LIMIT].rstrip(),
    }


def _error_text(error: Exception) -> str:
    return (" ".join(str(error).split()) or type(error).__name__)[:2000]


class SearchQueryExhausted(RuntimeError):
    """Four exact DDGS attempts found no results for one query."""

    def __init__(self, tool_result: dict[str, Any]) -> None:
        self.tool_result = tool_result
        super().__init__(str(tool_result["error"]))


def is_ddgs_no_results_event(event: Any) -> bool:
    """Recognize only the provider's query-local no-results failure."""
    if not isinstance(event, dict):
        return False
    error = " ".join(str(event.get("error") or "").lower().split()).rstrip(".")
    return (
        event.get("event") == SEARCH_BACKEND_ATTEMPT_EVENT
        and event.get("logical_backend") == "web"
        and event.get("source_backend") == "ddgs"
        and event.get("source_engine") == DDGS_ENGINE_V3
        and event.get("success") is False
        and event.get("candidate_results") is None
        and event.get("candidate_results_sha256") is None
        and event.get("error_type") == "DDGSException"
        and error == "no results found"
    )


def is_insufficient_results_event(event: Any) -> bool:
    """Recognize a live web attempt that returned fewer valid results than the contract."""
    if not isinstance(event, dict):
        return False
    error = " ".join(str(event.get("error") or "").split())
    return (
        event.get("event") == SEARCH_BACKEND_ATTEMPT_EVENT
        and event.get("logical_backend") == "web"
        and event.get("success") is False
        and event.get("error_type") == "SearchResultValidationError"
        and error.startswith("search backend returned ")
        and "valid results; expected" in error
    )


class SearchTool:
    """Strict prospective search transport.

    It never reads or writes the sealed v2 cache/history. The v3 table lives in
    the same SQLite file only so the migration can prove the old projections did
    not change.
    """

    def __init__(
        self,
        cache_path: Path = SEARCH_CACHE_PATH,
        *,
        events_path: Path | None = None,
        initialize_if_missing: bool = False,
    ) -> None:
        configured = os.getenv("SEARCH_BACKEND")
        if configured is not None and configured.strip().lower() not in {"", "ddgs"}:
            raise ValueError(
                "strict-v3 fixes the primary web backend to ddgs; "
                "SEARCH_BACKEND may be absent or 'ddgs'"
            )
        self.runtime_metadata = validate_search_transport_runtime_v3()
        self.transport_sha256 = search_transport_sha256_v3()
        self.cache_path = Path(cache_path)
        self.initialize_if_missing = initialize_if_missing
        self.events_path = Path(
            events_path
            or ARCHIVE / "search_backend_events_ddgs_bing_v3.jsonl"
        )
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_lock = asyncio.Lock()
        self.query_locks_guard = asyncio.Lock()
        self.query_locks: dict[tuple[str, str], asyncio.Lock] = {}
        self.web_lock = asyncio.Lock()
        self.scholarly_lock = asyncio.Lock()
        self.backend_event_lock = asyncio.Lock()
        self.last_web_request = 0.0
        self.last_web_request_epoch = 0.0
        self.last_scholarly_request = 0.0
        self.configured_web_backend = "ddgs"
        self.web_backend = "ddgs"
        self.terminal_latched = False
        self.terminal_reason: str | None = None
        self._initialize()
        context = ssl.create_default_context(cafile=certifi.where())
        self.http = httpx.AsyncClient(timeout=httpx.Timeout(30.0), verify=context)
        history = self._load_backend_history()
        self.web_backend = history.current_web_backend
        self.terminal_latched = history.terminal_latched
        self.terminal_reason = history.terminal_reason
        self.failures: deque[bool] = deque(
            history.web_failure_window, maxlen=SEARCH_FAILURE_WINDOW
        )
        web_starts = [
            event.get("started_at")
            for event in self.backend_events
            if event.get("event") == SEARCH_BACKEND_ATTEMPT_EVENT
            and event.get("logical_backend") == "web"
        ]
        if web_starts:
            self.last_web_request_epoch = max(
                datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()
                for value in web_starts
            )

    def _initialize(self) -> None:
        if not self.cache_path.exists() and not self.initialize_if_missing:
            raise RuntimeError(
                "strict-v3 cache database is absent; run the v3 cache migration first"
            )
        with sqlite3.connect(self.cache_path) as connection:
            exists = connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                (V3_CACHE_TABLE,),
            ).fetchone()
            if exists is None:
                if not self.initialize_if_missing:
                    raise RuntimeError(
                        "strict-v3 cache table is absent; run the v3 cache migration first"
                    )
                initialize_v3_cache_schema(connection)
            validate_v3_cache_schema(connection)
            if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                raise RuntimeError("search cache failed integrity_check")

    def _load_backend_history(self):
        try:
            rows = read_jsonl(self.events_path) if self.events_path.exists() else []
            history = validate_search_backend_history_v3(
                rows, allow_pending_policy=True
            )
        except (SearchBackendHistoryError, ValueError, TypeError) as error:
            raise RuntimeError(f"invalid persisted v3 search history: {error}") from error
        if history.pending_web_policy is not None:
            trigger, failed_attempts, failure_rate = history.pending_web_policy
            key_nonempty = bool(os.getenv("BRAVE_API_KEY", "").strip())
            policy = backend_policy_event_v3(
                trigger=trigger,
                failed_attempts=failed_attempts,
                failure_rate=failure_rate,
                timestamp=now_iso(),
                brave_key_nonempty=key_nonempty,
            )
            append_jsonl(self.events_path, policy)
            rows.append(policy)
            try:
                history = validate_search_backend_history_v3(rows)
            except SearchBackendHistoryError as error:
                raise RuntimeError(
                    f"invalid recovered v3 search history: {error}"
                ) from error
        self.backend_events = rows
        return history

    async def close(self) -> None:
        await self.http.aclose()

    async def _query_lock(self, logical_backend: str, query_key: str) -> asyncio.Lock:
        key = (logical_backend, query_key)
        async with self.query_locks_guard:
            lock = self.query_locks.get(key)
            if lock is None:
                lock = asyncio.Lock()
                self.query_locks[key] = lock
            return lock

    @staticmethod
    def _result(
        query: str,
        query_key: str,
        logical_backend: str,
        source_backend: str,
        source_engine: str,
        cache_hit: bool,
        results: list[dict],
    ) -> dict:
        return {
            "query": query,
            "normalized_query": query_key,
            "backend": source_backend,
            "source_backend": source_backend,
            "source_engine": source_engine,
            "transport_id": SEARCH_TRANSPORT_ID_V3,
            "history_id": SEARCH_BACKEND_HISTORY_ID_V3,
            "cache_protocol": CACHE_PROTOCOL_V3,
            "cache_namespace": logical_backend,
            "cache_hit": cache_hit,
            "results": results,
        }

    @staticmethod
    def _query_exhausted_result(query: str, query_key: str) -> dict[str, Any]:
        return {
            "status": SEARCH_QUERY_EXHAUSTED_STATUS,
            "query_exhaustion_protocol": SEARCH_QUERY_EXHAUSTION_PROTOCOL_V5,
            "query": query,
            "normalized_query": query_key,
            "backend": "ddgs",
            "source_backend": "ddgs",
            "source_engine": DDGS_ENGINE_V3,
            "transport_id": SEARCH_TRANSPORT_ID_V3,
            "history_id": SEARCH_BACKEND_HISTORY_ID_V3,
            "cache_protocol": CACHE_PROTOCOL_V3,
            "cache_namespace": "web",
            "cache_hit": False,
            "attempt_count": MAX_SEARCH_ATTEMPTS_V3,
            "error_type": "SearchQueryExhausted",
            "error": (
                "No results found after four exact DDGS attempts. "
                "Broaden the query and search again."
            ),
        }

    @classmethod
    def _query_exhausted_error(
        cls,
        *,
        execution_id: str,
        query: str,
        query_key: str,
        events: list[dict[str, Any]],
    ) -> SearchQueryExhausted | None:
        ordered = sorted(events, key=lambda event: int(event.get("attempt") or 0))
        if (
            len(ordered) != MAX_SEARCH_ATTEMPTS_V3
            or [event.get("attempt") for event in ordered]
            != list(range(1, MAX_SEARCH_ATTEMPTS_V3 + 1))
            or any(
                event.get("search_execution_id") != execution_id
                or event.get("query") != query
                or not (
                    is_ddgs_no_results_event(event)
                    or is_insufficient_results_event(event)
                )
                for event in ordered
            )
        ):
            return None
        return SearchQueryExhausted(cls._query_exhausted_result(query, query_key))

    async def _cached(
        self, logical_backend: str, query_key: str
    ) -> tuple[str, str, list[dict]] | None:
        async with self.cache_lock:
            with sqlite3.connect(self.cache_path) as connection:
                row = connection.execute(
                    f"SELECT source_backend,source_engine,results_json "
                    f"FROM {V3_CACHE_TABLE} WHERE logical_backend=? AND query_key=?",
                    (logical_backend, query_key),
                ).fetchone()
        if row is None:
            return None
        value = json.loads(row[2])
        validate_result_set(value)
        return str(row[0]), str(row[1]), value

    async def _store(
        self,
        logical_backend: str,
        source_backend: str,
        source_engine: str,
        query_key: str,
        query: str,
        results: list[dict],
        backend_attempt_sha256: str,
    ) -> tuple[str, str, list[dict], bool]:
        validate_result_set(results)
        if not re.fullmatch(r"[0-9a-f]{64}", backend_attempt_sha256):
            raise RuntimeError("v3 cache insertion lacks backend-attempt provenance")
        encoded = json.dumps(results, ensure_ascii=False, separators=(",", ":"))
        async with self.cache_lock:
            with sqlite3.connect(self.cache_path) as connection:
                cursor = connection.execute(
                    f"INSERT OR IGNORE INTO {V3_CACHE_TABLE}("
                    "logical_backend,query_key,query,source_backend,source_engine,"
                    "results_json,fetched_at,transport_id,backend_attempt_sha256) "
                    "VALUES(?,?,?,?,?,?,?,?,?)",
                    (
                        logical_backend,
                        query_key,
                        query,
                        source_backend,
                        source_engine,
                        encoded,
                        now_iso(),
                        SEARCH_TRANSPORT_ID_V3,
                        backend_attempt_sha256,
                    ),
                )
                row = connection.execute(
                    f"SELECT source_backend,source_engine,results_json "
                    f"FROM {V3_CACHE_TABLE} WHERE logical_backend=? AND query_key=?",
                    (logical_backend, query_key),
                ).fetchone()
        if row is None:
            raise RuntimeError("stored search result could not be read back")
        persisted = json.loads(row[2])
        validate_result_set(persisted)
        return str(row[0]), str(row[1]), persisted, cursor.rowcount == 1

    @staticmethod
    def _backend_event_sha256(event: dict[str, Any]) -> str:
        return hashlib.sha256(canonical_json(event).encode()).hexdigest()

    def _execution_events(
        self, execution_id: str, logical_backend: str, query: str
    ) -> list[dict[str, Any]]:
        matching = [
            event
            for event in self.backend_events
            if event.get("event") == SEARCH_BACKEND_ATTEMPT_EVENT
            and event.get("search_execution_id") == execution_id
        ]
        if any(
            event.get("logical_backend") != logical_backend
            or event.get("query") != query
            for event in matching
        ):
            raise RuntimeError(
                "persisted search execution ID is linked to a different query/backend"
            )
        return matching

    @classmethod
    def _rejected_from_backend_event(cls, event: dict[str, Any]) -> dict[str, Any]:
        return {
            "event": SEARCH_ATTEMPT_REJECTED_EVENT,
            "history_id": event["history_id"],
            "transport_id": event["transport_id"],
            "transport_sha256": event["transport_sha256"],
            "search_execution_id": event["search_execution_id"],
            "attempt": event["attempt"],
            "logical_backend": event["logical_backend"],
            "source_backend": event["source_backend"],
            "source_engine": event["source_engine"],
            "query": event["query"],
            "started_at": event["started_at"],
            "finished_at": event["finished_at"],
            "candidate_results": event["candidate_results"],
            "candidate_results_sha256": event["candidate_results_sha256"],
            "backend_attempt_sha256": cls._backend_event_sha256(event),
            "error_type": event["error_type"],
            "error": event["error"],
        }

    @classmethod
    def _policy_rejected_from_backend_event(
        cls, event: dict[str, Any], error: Exception
    ) -> dict[str, Any]:
        return {
            **cls._rejected_from_backend_event(event),
            "error_type": type(error).__name__,
            "error": _error_text(error),
        }

    @classmethod
    def _completion_provenance(
        cls,
        *,
        execution_id: str,
        query: str,
        query_key: str,
        logical_backend: str,
        result: dict[str, Any],
        prior_rejected_attempts: int,
        accepted_event: dict[str, Any] | None,
        recovered_from_backend_history: bool,
    ) -> dict[str, Any]:
        accepted_results = (
            prepare_result_set(accepted_event["candidate_results"])
            if accepted_event is not None
            else None
        )
        return {
            "protocol": SEARCH_EXECUTION_PROTOCOL_V3,
            "history_id": SEARCH_BACKEND_HISTORY_ID_V3,
            "transport_id": SEARCH_TRANSPORT_ID_V3,
            "transport_sha256": search_transport_sha256_v3(),
            "search_execution_id": execution_id,
            "query": query,
            "normalized_query": query_key,
            "logical_backend": logical_backend,
            "returned_source_backend": result["backend"],
            "returned_source_engine": result["source_engine"],
            "cache_protocol": CACHE_PROTOCOL_V3,
            "cache_hit": result["cache_hit"],
            "upstream_attempted": bool(prior_rejected_attempts or accepted_event),
            "prior_rejected_attempts": prior_rejected_attempts,
            "accepted_attempt": (
                accepted_event["attempt"] if accepted_event is not None else None
            ),
            "accepted_source_backend": (
                accepted_event["source_backend"] if accepted_event is not None else None
            ),
            "accepted_source_engine": (
                accepted_event["source_engine"] if accepted_event is not None else None
            ),
            "accepted_backend_attempt_sha256": (
                cls._backend_event_sha256(accepted_event)
                if accepted_event is not None
                else None
            ),
            "accepted_results_sha256": (
                hashlib.sha256(canonical_json(accepted_results).encode()).hexdigest()
                if accepted_results is not None
                else None
            ),
            "result_sha256": hashlib.sha256(canonical_json(result).encode()).hexdigest(),
            "recovered_from_backend_history": recovered_from_backend_history,
        }

    @staticmethod
    def _record_completion(
        recorder: SearchProvenanceRecorder | None, provenance: dict[str, Any]
    ) -> None:
        if recorder is not None:
            recorder(provenance)

    async def search(
        self,
        query: str,
        backend: str = "web",
        *,
        attempt_recorder: SearchAttemptRecorder | None = None,
        provenance_recorder: SearchProvenanceRecorder | None = None,
        search_execution_id: str | None = None,
    ) -> dict:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("search query must be a non-empty string")
        if backend not in {"web", "scholarly"}:
            raise ValueError(f"unknown search backend: {backend}")
        query_key = normalize_query(query)
        execution_id = search_execution_id or uuid.uuid4().hex
        if not re.fullmatch(r"[0-9a-f]{32}", execution_id):
            raise ValueError("search_execution_id must be 32 lowercase hex characters")
        replayed_attempts: set[int] = set()

        async def resume_or_none() -> tuple[dict[str, Any] | None, int, int]:
            prior = self._execution_events(execution_id, backend, query)
            rejected = [event for event in prior if event.get("success") is False]
            for event in rejected:
                attempt = int(event["attempt"])
                if attempt_recorder is not None and attempt not in replayed_attempts:
                    attempt_recorder(self._rejected_from_backend_event(event))
                    replayed_attempts.add(attempt)
            accepted = [event for event in prior if event.get("success") is True]
            if len(accepted) > 1:
                raise RuntimeError("search execution has multiple successful attempts")
            if accepted:
                accepted_event = accepted[0]
                terminal_trigger = next(
                    (
                        event
                        for event in self.backend_events
                        if event.get("event") == SEARCH_BACKEND_POLICY_TERMINAL_EVENT
                        and event.get("trigger_search_execution_id") == execution_id
                        and event.get("trigger_attempt") == accepted_event["attempt"]
                    ),
                    None,
                )
                if terminal_trigger is not None:
                    policy_error = RuntimeError(
                        "strict-v3 web search policy rejected the triggering result: "
                        + str(terminal_trigger["reason"])
                    )
                    attempt = int(accepted_event["attempt"])
                    if attempt_recorder is not None and attempt not in replayed_attempts:
                        attempt_recorder(
                            self._policy_rejected_from_backend_event(
                                accepted_event, policy_error
                            )
                        )
                        replayed_attempts.add(attempt)
                    raise policy_error
                accepted_results = prepare_result_set(
                    accepted_event["candidate_results"]
                )
                stored_backend, _stored_engine, stored_results, inserted = await self._store(
                    backend,
                    str(accepted_event["source_backend"]),
                    str(accepted_event["source_engine"]),
                    query_key,
                    query,
                    accepted_results,
                    self._backend_event_sha256(accepted_event),
                )
                result = self._result(
                    query,
                    query_key,
                    backend,
                    stored_backend,
                    _stored_engine,
                    False,
                    stored_results,
                )
                self._record_completion(
                    provenance_recorder,
                    self._completion_provenance(
                        execution_id=execution_id,
                        query=query,
                        query_key=query_key,
                        logical_backend=backend,
                        result=result,
                        prior_rejected_attempts=len(rejected),
                        accepted_event=accepted_event,
                        recovered_from_backend_history=True,
                    ),
                )
                return result, len(prior) + 1, len(rejected)
            if len(rejected) >= MAX_SEARCH_ATTEMPTS_V3:
                exhausted = self._query_exhausted_error(
                    execution_id=execution_id,
                    query=query,
                    query_key=query_key,
                    events=rejected,
                )
                if exhausted is not None:
                    self._record_completion(
                        provenance_recorder,
                        self._completion_provenance(
                            execution_id=execution_id,
                            query=query,
                            query_key=query_key,
                            logical_backend=backend,
                            result=exhausted.tool_result,
                            prior_rejected_attempts=MAX_SEARCH_ATTEMPTS_V3,
                            accepted_event=None,
                            recovered_from_backend_history=True,
                        ),
                    )
                    raise exhausted
                last = rejected[-1]
                raise RuntimeError(f"{last['error_type']}: {last['error']}")
            return None, len(prior) + 1, len(rejected)

        resumed, next_attempt, rejected_count = await resume_or_none()
        if resumed is not None:
            return resumed
        if backend == "web" and self.terminal_latched:
            raise RuntimeError(
                "web search is durably latched by strict-v3 policy: "
                f"{self.terminal_reason}"
            )
        cached = await self._cached(backend, query_key)
        if cached is not None:
            cached_backend, _cached_engine, cached_results = cached
            result = self._result(
                query,
                query_key,
                backend,
                cached_backend,
                _cached_engine,
                True,
                cached_results,
            )
            self._record_completion(
                provenance_recorder,
                self._completion_provenance(
                    execution_id=execution_id,
                    query=query,
                    query_key=query_key,
                    logical_backend=backend,
                    result=result,
                    prior_rejected_attempts=rejected_count,
                    accepted_event=None,
                    recovered_from_backend_history=bool(rejected_count),
                ),
            )
            return result

        query_lock = await self._query_lock(backend, query_key)
        async with query_lock:
            resumed, next_attempt, rejected_count = await resume_or_none()
            if resumed is not None:
                return resumed
            if backend == "web" and self.terminal_latched:
                raise RuntimeError(
                    "web search is durably latched by strict-v3 policy: "
                    f"{self.terminal_reason}"
                )
            cached = await self._cached(backend, query_key)
            if cached is not None:
                cached_backend, _cached_engine, cached_results = cached
                result = self._result(
                    query,
                    query_key,
                    backend,
                    cached_backend,
                    _cached_engine,
                    True,
                    cached_results,
                )
                self._record_completion(
                    provenance_recorder,
                    self._completion_provenance(
                        execution_id=execution_id,
                        query=query,
                        query_key=query_key,
                        logical_backend=backend,
                        result=result,
                        prior_rejected_attempts=rejected_count,
                        accepted_event=None,
                        recovered_from_backend_history=bool(rejected_count),
                    ),
                )
                return result
            if backend == "web":
                try:
                    source_backend, results, accepted_event = await self._web_search(
                        query,
                        execution_id=execution_id,
                        attempt_recorder=attempt_recorder,
                        start_attempt=next_attempt,
                    )
                except SearchQueryExhausted as error:
                    self._record_completion(
                        provenance_recorder,
                        self._completion_provenance(
                            execution_id=execution_id,
                            query=query,
                            query_key=query_key,
                            logical_backend=backend,
                            result=error.tool_result,
                            prior_rejected_attempts=MAX_SEARCH_ATTEMPTS_V3,
                            accepted_event=None,
                            recovered_from_backend_history=next_attempt > 1,
                        ),
                    )
                    raise
            else:
                source_backend, results, accepted_event = (
                    await self._strict_scholarly_search(
                        query,
                        execution_id=execution_id,
                        attempt_recorder=attempt_recorder,
                        start_attempt=next_attempt,
                    )
                )
            results = prepare_result_set(results)
            source_engine = str(accepted_event["source_engine"])
            stored_backend, _stored_engine, stored_results, inserted = await self._store(
                backend,
                source_backend,
                source_engine,
                query_key,
                query,
                results,
                self._backend_event_sha256(accepted_event),
            )
            result = self._result(
                query,
                query_key,
                backend,
                stored_backend,
                _stored_engine,
                not inserted,
                stored_results,
            )
            self._record_completion(
                provenance_recorder,
                self._completion_provenance(
                    execution_id=execution_id,
                    query=query,
                    query_key=query_key,
                    logical_backend=backend,
                    result=result,
                    prior_rejected_attempts=int(accepted_event["attempt"]) - 1,
                    accepted_event=accepted_event,
                    recovered_from_backend_history=False,
                ),
            )
            return result

    @staticmethod
    def _rejected_attempt(
        *,
        execution_id: str,
        attempt: int,
        logical_backend: str,
        source_backend: str,
        query: str,
        started_at: str,
        finished_at: str,
        candidates: list[dict] | None,
        error: Exception,
        backend_attempt_sha256: str | None = None,
        source_engine: str | None = None,
        transport_sha256: str | None = None,
    ) -> dict[str, Any]:
        candidate_hash = (
            hashlib.sha256(canonical_json(candidates).encode()).hexdigest()
            if candidates is not None
            else None
        )
        row = {
            "event": SEARCH_ATTEMPT_REJECTED_EVENT,
            "search_execution_id": execution_id,
            "attempt": attempt,
            "logical_backend": logical_backend,
            "source_backend": source_backend,
            "query": query,
            "started_at": started_at,
            "finished_at": finished_at,
            "candidate_results": candidates,
            "candidate_results_sha256": candidate_hash,
            "backend_attempt_sha256": backend_attempt_sha256,
            "error_type": type(error).__name__,
            "error": _error_text(error),
        }
        # Historical v2 recovery calls this pure formatter without v3 runtime
        # dependencies. Prospective callers pass the v3 linkage explicitly.
        if source_engine is not None or transport_sha256 is not None:
            row.update(
                {
                    "history_id": SEARCH_BACKEND_HISTORY_ID_V3,
                    "transport_id": SEARCH_TRANSPORT_ID_V3,
                    "transport_sha256": transport_sha256,
                    "source_engine": source_engine,
                }
            )
        return row

    async def _web_search(
        self,
        query: str,
        *,
        execution_id: str,
        attempt_recorder: SearchAttemptRecorder | None,
        start_attempt: int = 1,
    ) -> tuple[str, list[dict], dict[str, Any]]:
        last_error = "web search failed"
        for attempt in range(start_attempt, MAX_SEARCH_ATTEMPTS_V3 + 1):
            raw_results: list[dict] | None = None
            policy_error: Exception | None = None
            async with self.web_lock:
                if self.terminal_latched:
                    raise RuntimeError(
                        "web search is durably latched by strict-v3 policy: "
                        f"{self.terminal_reason}"
                    )
                pacing = await self._wait_for_web_slot(execution_id, attempt)
                started_at = now_iso()
                source_backend = self.web_backend
                source_engine = (
                    DDGS_ENGINE_V3
                    if source_backend == "ddgs"
                    else BRAVE_SOURCE_ENGINE_V3
                )
                try:
                    raw_results = (
                        await self._ddgs(query)
                        if source_backend == "ddgs"
                        else await self._brave(query)
                    )
                    results = prepare_result_set(raw_results)
                except Exception as error:
                    finished_at = now_iso()
                    backend_event, policy_error = await self._record_backend_attempt(
                        execution_id=execution_id,
                        attempt=attempt,
                        logical_backend="web",
                        source_backend=source_backend,
                        source_engine=source_engine,
                        query=query,
                        started_at=started_at,
                        finished_at=finished_at,
                        candidates=raw_results,
                        error=error,
                        pacing=pacing,
                    )
                    if attempt_recorder is not None:
                        attempt_recorder(self._rejected_from_backend_event(backend_event))
                    last_error = f"{type(error).__name__}: {error}"
                else:
                    backend_event, policy_error = await self._record_backend_attempt(
                        execution_id=execution_id,
                        attempt=attempt,
                        logical_backend="web",
                        source_backend=source_backend,
                        source_engine=source_engine,
                        query=query,
                        started_at=started_at,
                        finished_at=now_iso(),
                        candidates=raw_results,
                        error=None,
                        pacing=pacing,
                    )
                    if policy_error is not None:
                        if attempt_recorder is not None:
                            attempt_recorder(
                                self._policy_rejected_from_backend_event(
                                    backend_event, policy_error
                                )
                            )
                        raise policy_error
                    return source_backend, results, backend_event
            if policy_error is not None:
                raise policy_error
        exhausted = self._query_exhausted_error(
            execution_id=execution_id,
            query=query,
            query_key=normalize_query(query),
            events=self._execution_events(execution_id, "web", query),
        )
        if exhausted is not None:
            raise exhausted
        raise RuntimeError(last_error)

    async def _strict_scholarly_search(
        self,
        query: str,
        *,
        execution_id: str,
        attempt_recorder: SearchAttemptRecorder | None,
        start_attempt: int = 1,
    ) -> tuple[str, list[dict], dict[str, Any]]:
        last_error = "scholarly search failed"
        for attempt in range(start_attempt, MAX_SEARCH_ATTEMPTS_V3 + 1):
            if attempt > start_attempt:
                pacing = expected_web_pacing_v3(execution_id, attempt)
                await asyncio.sleep(float(pacing["retry_backoff_seconds"]))
            started_at = now_iso()
            raw_results: list[dict] | None = None
            try:
                raw_results = await self._scholarly_search(query)
                results = prepare_result_set(raw_results)
            except Exception as error:
                finished_at = now_iso()
                last_error = f"{type(error).__name__}: {error}"
                backend_event, _policy_error = await self._record_backend_attempt(
                    execution_id=execution_id,
                    attempt=attempt,
                    logical_backend="scholarly",
                    source_backend="scholarly",
                    source_engine=SCHOLARLY_SOURCE_ENGINE_V3,
                    query=query,
                    started_at=started_at,
                    finished_at=finished_at,
                    candidates=raw_results,
                    error=error,
                    pacing=None,
                )
                if attempt_recorder is not None:
                    attempt_recorder(self._rejected_from_backend_event(backend_event))
                continue
            backend_event, _policy_error = await self._record_backend_attempt(
                execution_id=execution_id,
                attempt=attempt,
                logical_backend="scholarly",
                source_backend="scholarly",
                source_engine=SCHOLARLY_SOURCE_ENGINE_V3,
                query=query,
                started_at=started_at,
                finished_at=now_iso(),
                candidates=raw_results,
                error=None,
                pacing=None,
            )
            return "scholarly", results, backend_event
        raise RuntimeError(last_error)

    async def _record_backend_attempt(
        self,
        *,
        execution_id: str,
        attempt: int,
        logical_backend: str,
        source_backend: str,
        source_engine: str,
        query: str,
        started_at: str,
        finished_at: str,
        candidates: list[dict] | None,
        error: Exception | None,
        pacing: dict[str, Any] | None,
    ) -> tuple[dict[str, Any], Exception | None]:
        event = {
            "schema_version": SEARCH_BACKEND_EVENT_SCHEMA_V3,
            "event": SEARCH_BACKEND_ATTEMPT_EVENT,
            "history_id": SEARCH_BACKEND_HISTORY_ID_V3,
            "transport_id": SEARCH_TRANSPORT_ID_V3,
            "transport_sha256": self.transport_sha256,
            "configured_web_backend": self.configured_web_backend,
            "search_execution_id": execution_id,
            "attempt": attempt,
            "logical_backend": logical_backend,
            "source_backend": source_backend,
            "source_engine": source_engine,
            "transport_metadata": source_transport_metadata_v3(source_backend),
            "pacing": pacing,
            "query": query,
            "started_at": started_at,
            "finished_at": finished_at,
            "success": error is None,
            "candidate_results": candidates,
            "candidate_results_sha256": (
                hashlib.sha256(canonical_json(candidates).encode()).hexdigest()
                if candidates is not None
                else None
            ),
            "error_type": type(error).__name__ if error is not None else None,
            "error": _error_text(error) if error is not None else None,
        }
        policy_error: Exception | None = None
        async with self.backend_event_lock:
            append_jsonl(self.events_path, event)
            self.backend_events.append(event)
            if logical_backend == "web":
                self.failures.append(error is not None)
                policy_error = self._check_failure_rate_locked(
                    event, attempt_failed=error is not None
                )
        return event, policy_error

    def _check_failure_rate_locked(
        self, trigger: dict[str, Any], *, attempt_failed: bool
    ) -> Exception | None:
        if (
            len(self.failures) < SEARCH_FAILURE_WINDOW
            or sum(self.failures) / len(self.failures) <= SEARCH_FAILURE_THRESHOLD
        ):
            return None
        failures = sum(self.failures)
        failure_rate = failures / len(self.failures)
        key_nonempty = bool(os.getenv("BRAVE_API_KEY", "").strip())
        policy = backend_policy_event_v3(
            trigger=trigger,
            failed_attempts=failures,
            failure_rate=failure_rate,
            timestamp=now_iso(),
            brave_key_nonempty=key_nonempty,
        )
        append_jsonl(self.events_path, policy)
        self.backend_events.append(policy)
        if policy["event"] == "backend_switch":
            self.web_backend = "brave"
            self.failures.clear()
            return None
        self.terminal_latched = True
        self.terminal_reason = str(policy["reason"])
        return RuntimeError(
            "strict-v3 web search policy terminated: " + self.terminal_reason
        )

    async def _wait_for_web_slot(
        self, execution_id: str, attempt: int
    ) -> dict[str, Any]:
        pacing = expected_web_pacing_v3(execution_id, attempt)
        minimum = float(pacing["minimum_interval_seconds"])
        monotonic_wait = (
            max(0.0, minimum - (time.monotonic() - self.last_web_request))
            if self.last_web_request
            else 0.0
        )
        wall_wait = (
            max(0.0, minimum - (time.time() - self.last_web_request_epoch))
            if self.last_web_request_epoch
            else 0.0
        )
        interval_wait = max(monotonic_wait, wall_wait)
        wait = (
            interval_wait
            + float(pacing["pacing_jitter_seconds"])
            + float(pacing["retry_backoff_seconds"])
        )
        pacing = {**pacing, "scheduled_wait_seconds": round(wait, 9)}
        if wait:
            await asyncio.sleep(wait)
        self.last_web_request = time.monotonic()
        self.last_web_request_epoch = time.time()
        return pacing

    async def _ddgs(self, query: str) -> list[dict]:
        validate_search_transport_runtime_v3()

        def call() -> list[dict]:
            from ddgs import DDGS

            rows = list(
                DDGS(verify=certifi.where()).text(
                    query,
                    region=DDGS_REGION_V3,
                    safesearch=DDGS_SAFESEARCH_V3,
                    timelimit=DDGS_TIMELIMIT_V3,
                    backend=DDGS_ENGINE_V3,
                    page=DDGS_PAGE_V3,
                    max_results=WEB_CANDIDATE_COUNT,
                )
            )
            return [
                {
                    "title": row.get("title"),
                    "url": row.get("href") or row.get("url"),
                    "snippet": row.get("body") or row.get("snippet"),
                }
                for row in rows
            ]

        return await asyncio.to_thread(call)

    async def _brave(self, query: str) -> list[dict]:
        key = os.getenv("BRAVE_API_KEY", "").strip()
        if not key:
            raise RuntimeError("BRAVE_API_KEY is not set")
        used = sum(
            1
            for event in self.backend_events
            if event.get("event") == SEARCH_BACKEND_ATTEMPT_EVENT
            and event.get("source_backend") == "brave"
        )
        if used >= BRAVE_REQUEST_BUDGET:
            raise RuntimeError(
                f"brave request budget of {BRAVE_REQUEST_BUDGET} requests is exhausted"
            )
        response = await self.http.get(
            "https://api.search.brave.com/res/v1/web/search",
            params={"q": query, "count": WEB_CANDIDATE_COUNT, "country": "us"},
            headers={"X-Subscription-Token": key, "Accept": "application/json"},
        )
        response.raise_for_status()
        rows = response.json().get("web", {}).get("results", [])
        return [
            {
                "title": row.get("title"),
                "url": row.get("url"),
                "snippet": row.get("description"),
            }
            for row in rows
        ]

    async def _scholarly_get(self, url: str, **kwargs) -> httpx.Response:
        headers = {**kwargs.pop("headers", {}), "User-Agent": SCHOLARLY_USER_AGENT}
        for attempt in range(SCHOLARLY_MAX_ATTEMPTS):
            async with self.scholarly_lock:
                wait = max(
                    0.0,
                    SCHOLARLY_MIN_INTERVAL_SECONDS
                    - (time.monotonic() - self.last_scholarly_request),
                )
                if wait:
                    await asyncio.sleep(wait)
                self.last_scholarly_request = time.monotonic()
                response = await self.http.get(url, headers=headers, **kwargs)
            if response.status_code != 429 or attempt == SCHOLARLY_MAX_ATTEMPTS - 1:
                break
            await asyncio.sleep(SCHOLARLY_BACKOFF_SECONDS * (attempt + 1))
        response.raise_for_status()
        return response

    async def _scholarly_search(self, query: str) -> list[dict]:
        search_params: dict[str, Any] = {
            "db": "pubmed",
            "term": query,
            "retmode": "json",
            "retmax": SEARCH_RESULT_COUNT,
            "tool": "sycophantic_search",
        }
        if os.getenv("NCBI_API_KEY"):
            search_params["api_key"] = os.environ["NCBI_API_KEY"]
        search_response = await self._scholarly_get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
            params=search_params,
        )
        ids = search_response.json().get("esearchresult", {}).get("idlist", [])
        pubmed: list[dict] = []
        if ids:
            summary_params: dict[str, Any] = {
                "db": "pubmed",
                "id": ",".join(ids),
                "retmode": "json",
                "tool": "sycophantic_search",
            }
            if os.getenv("NCBI_API_KEY"):
                summary_params["api_key"] = os.environ["NCBI_API_KEY"]
            summary_response = await self._scholarly_get(
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi",
                params=summary_params,
            )
            records = summary_response.json().get("result", {})
            for pubmed_id in ids:
                record = records.get(pubmed_id, {})
                pubmed.append(
                    {
                        "title": record.get("title"),
                        "url": f"https://pubmed.ncbi.nlm.nih.gov/{pubmed_id}/",
                        "snippet": " ".join(
                            part
                            for part in (record.get("source"), record.get("pubdate"))
                            if part
                        ),
                    }
                )
        wiki_response = await self._scholarly_get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "opensearch",
                "search": query,
                "limit": SEARCH_RESULT_COUNT,
                "namespace": 0,
                "format": "json",
            },
        )
        wiki = wiki_response.json()
        wikipedia = [
            {"title": title, "url": url, "snippet": snippet}
            for title, snippet, url in zip(wiki[1], wiki[2], wiki[3])
        ]
        return pubmed[:5] + wikipedia[:3] + pubmed[5:] + wikipedia[3:]
