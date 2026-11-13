from __future__ import annotations

import argparse
import asyncio
import base64
import binascii
import copy
import json
import os
import re
import signal
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any

from config import (
    ARCHIVE,
    DATA,
    MODEL_CONCURRENCY,
    OPENROUTER_LIMIT_WITHOUT_KIMI,
    PROMPTS_PATH,
    RAW_DIR,
    RUNS_DIR,
    SCHEDULE_PATH,
    SEARCH_LIMIT,
    load_claims,
    load_json,
    model_params,
    params_hash,
)
from openrouter import DB_PATH, Ledger
from openrouter import (
    ModelCall,
    OpenRouterClient,
    V3_RESERVATION_POLICY,
    generation_cost,
    reservation_provenance,
    usage_fields,
)
from schedule import (
    ACTIVE_PILOT_ARMS,
    ALL_PILOT_ARMS,
    DIAGNOSTIC_PILOT_ARMS,
    LEGACY_V1_V2_ARMS,
    TRIM_PROFILE_NONE,
    TRIM_PROFILE_STEP_1,
    TRIM_PROFILES,
    build_schedule,
    run_id as schedule_run_id,
)
from schedule import TRIM_RECORD_DIR, load_trim_records, verify_trim_records
from phase2 import (
    PHASE2_ARMS,
    cross_replay_context_path,
    load_phase2_schedule,
)
from search import (
    BRAVE_SOURCE_ENGINE_V3,
    CACHE_PROTOCOL_V3,
    DDGS_ENGINE_V3,
    SCHOLARLY_SOURCE_ENGINE_V3,
    SEARCH_BACKEND_ATTEMPT_EVENT,
    SEARCH_BACKEND_EVENT_SCHEMA_V3,
    SEARCH_BACKEND_HISTORY_ID_V3,
    SEARCH_BACKEND_POLICY_TERMINAL_EVENT,
    SEARCH_EXECUTION_PROTOCOL_V3,
    SEARCH_TRANSPORT_ID_V3,
    V3_CACHE_TABLE,
    prepare_result_set,
    validate_result_set,
    validate_v3_cache_schema,
)
from search import (
    SEARCH_ATTEMPT_REJECTED_EVENT,
    SEARCH_QUERY_EXHAUSTED_STATUS,
    SEARCH_QUERY_EXHAUSTION_PROTOCOL_V5,
    SearchQueryExhausted,
    SearchTool,
    is_ddgs_no_results_event,
    normalize_query,
)
from config import (
    append_jsonl,
    atomic_write,
    canonical_json,
    now_iso,
    pretty_json,
    read_jsonl,
    sha256_bytes,
)


TERMINAL_FAILURE_STATUS = "terminal_failed"
TERMINAL_FAILURE_EVENT = "trajectory_failure"
BLOCKED_REPLAY_STAGE = "replay_source_terminal"


def clean_error_message(error: BaseException) -> str:
    message = " ".join(str(error).split()) or type(error).__name__
    home = str(Path.home())
    if home:
        message = message.replace(home, "$HOME")
    return message[:2000]


def failure_event(
    row: dict,
    *,
    timestamp: str,
    stage: str,
    error: BaseException,
    call_index: int,
    ledger_status: str,
) -> dict:
    return {
        "event": TERMINAL_FAILURE_EVENT,
        "run_id": row["run_id"],
        "call_index": call_index,
        "timestamp": timestamp,
        "stage": stage,
        "error_type": type(error).__name__,
        "error": clean_error_message(error),
        "ledger_status": ledger_status,
    }


def is_blocked_replay_failure(event: dict) -> bool:
    return (
        event.get("event") == TERMINAL_FAILURE_EVENT
        and event.get("stage") == BLOCKED_REPLAY_STAGE
        and event.get("ledger_status") == "not_reserved"
    )


def _search_query(tool_call: object) -> str | None:
    if not isinstance(tool_call, dict):
        return None
    function = tool_call.get("function")
    if not isinstance(function, dict) or function.get("name") != "search":
        return None
    arguments = function.get("arguments")
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return None
    if not isinstance(arguments, dict):
        return None
    query = arguments.get("query")
    return query.strip() if isinstance(query, str) and query.strip() else None


def partial_trajectory(events: list[dict]) -> dict[str, list[Any]]:
    queries: list[str] = []
    searches: list[dict] = []
    result_urls: list[str] = []
    for event in events:
        if event.get("event") == "model_response":
            for tool_call in event.get("tool_calls") or []:
                query = _search_query(tool_call)
                if query is not None:
                    queries.append(query)
        if event.get("event") not in {
            "tool_results",
            "tool_result",
            "search_tool_failure",
        }:
            continue
        records = (
            event.get("completed_tool_results")
            if event.get("event") == "search_tool_failure"
            else event.get("tool_results")
        )
        if not isinstance(records, list):
            records = [event]
        for record in records:
            if not isinstance(record, dict):
                continue
            result = record.get("result")
            if not isinstance(result, dict):
                continue
            searches.append(result)
            rows = result.get("results")
            if not isinstance(rows, list):
                continue
            for item in rows:
                if isinstance(item, dict) and isinstance(item.get("url"), str):
                    result_urls.append(item["url"])
    return {
        "queries": queries,
        "searches": searches,
        "result_urls": result_urls,
    }


def accepted_terminal_failure(
    events: list[dict], call_index: int, ledger_row: dict | None
) -> bool:
    if ledger_row is None or ledger_row.get("call_index") != call_index:
        return False
    status = ledger_row.get("status")
    if status == "failed":
        return any(
            event.get("event") == "model_call"
            and event.get("call_index") == call_index
            and event.get("terminal") is True
            and isinstance(event.get("error"), str)
            and bool(event["error"].strip())
            for event in events
        )
    if status != "complete":
        return False
    responses = [
        event
        for event in events
        if event.get("event") == "model_response"
        and event.get("call_index") == call_index
        and isinstance(event.get("response_body"), dict)
    ]
    if len(responses) != 1:
        return False
    generation_id = str(responses[0]["response_body"].get("id") or "")
    if not generation_id or generation_id != str(ledger_row.get("generation_id") or ""):
        return False
    metadata = [
        event
        for event in events
        if event.get("event") in {"generation_metadata", "generation_metadata_repair"}
        and event.get("call_index") == call_index
        and event.get("generation_id") == generation_id
        and event.get("error") is None
        and isinstance(event.get("generation"), dict)
        and bool(event["generation"])
    ]
    return bool(metadata)


def terminal_failure_manifest(
    row: dict,
    *,
    params_hash_value: str,
    started_at: str,
    completed_at: str,
    stage: str,
    error: BaseException,
    call_index: int,
    ledger_status: str,
    events: list[dict],
    recorded_failure: dict | None = None,
) -> dict:
    partial = partial_trajectory(events)
    error_type = type(error).__name__
    message = clean_error_message(error)
    if recorded_failure is not None:
        error_type = str(recorded_failure.get("error_type") or error_type)
        message = str(recorded_failure.get("error") or message)
    return {
        "run_id": row["run_id"],
        "claim_id": row["claim_id"],
        "arm": row["arm"],
        "framing": row["framing"],
        "system": row["system"],
        "model": row["model"],
        "repeat": row["repeat"],
        "search_backend": row["search_backend"] or "fixed_replay",
        "params_hash": params_hash_value,
        "queries": partial["queries"],
        "searches": partial["searches"],
        "result_urls": partial["result_urls"],
        "report": "",
        "citations": [],
        "judge_labels": {},
        "query_labels": [],
        "calls": [],
        "search_count": len(partial["queries"]),
        "started_at": started_at,
        "completed_at": completed_at,
        "status": TERMINAL_FAILURE_STATUS,
        "failure": {
            "stage": stage,
            "error_type": error_type,
            "message": message,
            "call_index": call_index,
            "ledger_status": ledger_status,
        },
    }


URL_PATTERN = re.compile(r"https?://[^\s<>\]\[\"']+")
MODEL_CANONICAL = {
    "anthropic/claude-sonnet-5": "anthropic/claude-sonnet-5-20260630",
    "openai/gpt-5.6-terra": "openai/gpt-5.6-terra-20260709",
    "google/gemini-3.6-flash": "google/gemini-3.6-flash-20260721",
    "z-ai/glm-5.2": "z-ai/glm-5.2-20260616",
    "moonshotai/kimi-k3": "moonshotai/kimi-k3-20260715",
}
SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "search",
        "description": "Search for sources relevant to the research question.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "A concise search query.",
                }
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
}
MIN_SEARCHES = 4
PILOT_ARMS = ACTIVE_PILOT_ARMS
V5_MODEL_PROTOCOL_ARMS = frozenset(
    {
        "pilot_v5",
        "kimi_pilot_v5",
        "pilot_v6",
        "kimi_pilot_v6",
        "pilot_v7",
        "kimi_pilot_v7",
        "core",
        "attribution",
        "paraphrase",
        "mitigation",
        "scholarly",
        "replay",
        *PHASE2_ARMS,
    }
)
SEARCH_EXECUTION_ID_PROTOCOL_V2 = "strict_v2_search_execution_id_v1"
SEARCH_EXECUTION_ID_PROTOCOL_V3 = "strict_v3_search_execution_id_v1"
TRUNCATION_FINISH_REASONS = {"length", "max_tokens"}
TRUNCATION_RECOVERY_EVENT = "truncation_recovery"
TOOL_CALL_CLAMP_EVENT = "tool_call_clamp"
SEARCH_TOOL_FAILURE_EVENT = "search_tool_failure"
PROVISIONAL_METADATA_STATUS = "metadata_pending"
MAX_DEFERRED_CLAIM_FAILURES = 8
VISIBLE_QUERY_EXHAUSTION_ARMS = frozenset(
    {
        "pilot_v5",
        "kimi_pilot_v5",
        "pilot_v6",
        "kimi_pilot_v6",
        "pilot_v7",
        "kimi_pilot_v7",
        "core",
        "attribution",
        "paraphrase",
        "mitigation",
        "scholarly",
        "neutral_repeat",
        "skeptic_mitigation",
        "full_page",
        "scholarly_repaired",
        "scholarly_repaired_v2",
    }
)
FINAL_REPORT_INSTRUCTION = (
    "The search limit has been reached. Write the complete final cited report now "
    "in no more than 250 words and cite at most 4 retrieved-source URLs."
)
TRUNCATION_RECOVERY_INSTRUCTION = (
    "Your previous final answer was truncated. Rewrite the complete cited report "
    "concisely in no more than 250 words, cite at most 4 retrieved-source URLs, "
    "and do not request more searches."
)


class RecoveryInvariantError(RuntimeError):
    pass


def uses_visible_query_exhaustion(row: dict) -> bool:
    return row.get("arm") in VISIBLE_QUERY_EXHAUSTION_ARMS


def raw_path(row: dict) -> Path:
    return RAW_DIR / row["arm"] / f"{row['run_id']}.jsonl"


def state_path(row: dict) -> Path:
    return ARCHIVE / "state" / f"{row['run_id']}.json"


def manifest_path(row: dict) -> Path:
    return RUNS_DIR / f"{row['run_id']}.json"


def provider_route(model: str, protocol: str = "legacy") -> dict:
    if protocol not in {"legacy", "v5"}:
        raise ValueError(f"unknown provider protocol: {protocol!r}")
    if protocol == "v5" and model == "anthropic/claude-sonnet-5":
        return {
            "order": ["anthropic"],
            "only": ["anthropic"],
            "allow_fallbacks": False,
            "require_parameters": True,
        }
    if model == "google/gemini-3.6-flash":
        return {
            "order": ["google-ai-studio"],
            "only": ["google-ai-studio"],
            "allow_fallbacks": False,
            "require_parameters": False,
        }
    return {"allow_fallbacks": True, "require_parameters": True}


def parameter_contract(
    model: str,
    uses_search: bool,
    *,
    protocol: str = "legacy",
) -> dict:
    contract = {
        "model": model,
        "parameters": model_params(model, protocol=protocol),
        "provider": provider_route(model, protocol=protocol),
        "include_reasoning": True,
        "usage": {"include": True},
        "search_limit": SEARCH_LIMIT,
    }
    if uses_search:
        contract["tool"] = SEARCH_TOOL
        contract["minimum_searches"] = MIN_SEARCHES
    return contract


def uses_v5_model_protocol(row: dict) -> bool:
    return str(row.get("arm") or "") in V5_MODEL_PROTOCOL_ARMS


def row_model_params(row: dict) -> dict:
    protocol = "v5" if uses_v5_model_protocol(row) else "legacy"
    return model_params(str(row["model"]), protocol=protocol)


def row_provider_route(row: dict) -> dict:
    protocol = "v5" if uses_v5_model_protocol(row) else "legacy"
    return provider_route(str(row["model"]), protocol=protocol)


def prospective_minimum_searches(row: dict) -> int:
    if row.get("search_backend") is None:
        return 0
    if row.get("arm") in LEGACY_V1_V2_ARMS:
        return 0
    return MIN_SEARCHES


def frozen_agent_params_hash(row: dict) -> str:
    protocol = "v5" if uses_v5_model_protocol(row) else "legacy"
    return params_hash(
        parameter_contract(
            str(row["model"]),
            row.get("search_backend") is not None,
            protocol=protocol,
        )
    )


def response_message(payload: dict) -> dict:
    choices = payload.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise RuntimeError("response must contain exactly one choice")
    message = choices[0].get("message")
    if not isinstance(message, dict) or message.get("role") != "assistant":
        raise RuntimeError("response is missing an assistant message")
    return copy.deepcopy(message)


def finish_reason(payload: dict) -> str:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    return str(choices[0].get("finish_reason") or "")


def routed_provider(call: ModelCall) -> str:
    for key in ("provider_name", "provider"):
        value = call.generation.get(key)
        if value:
            return str(value)
    return str(call.payload.get("provider") or "")


def response_model_matches(requested: str, response_model: str) -> bool:
    base = requested.removesuffix(":exacto")
    return response_model in {requested, base, MODEL_CANONICAL.get(base, "")}


def validate_response(
    call: ModelCall,
    requested_model: str,
    *,
    allow_truncated: bool = False,
    allow_metadata_pending: bool = False,
    provider_protocol: str = "legacy",
) -> None:
    if call.metadata_status not in {"complete", "metadata_pending"}:
        raise RuntimeError(f"unknown response metadata status: {call.metadata_status!r}")
    if call.metadata_status != "complete" and not allow_metadata_pending:
        raise RuntimeError("response generation metadata is not settled")
    payload = call.payload
    if not str(payload.get("id") or ""):
        raise RuntimeError("response has no generation ID")
    response_model = str(payload.get("model") or "")
    if not response_model_matches(requested_model, response_model):
        raise RuntimeError(f"model identity mismatch: {requested_model} -> {response_model}")
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        raise RuntimeError("response has no usage object")
    for field in ("prompt_tokens", "completion_tokens"):
        if not isinstance(usage.get(field), int) or usage[field] < 0:
            raise RuntimeError(f"response usage is missing {field}")
    reason = finish_reason(payload)
    if not reason:
        raise RuntimeError("response has no finish reason")
    if reason in TRUNCATION_FINISH_REASONS and not allow_truncated:
        raise RuntimeError("response was truncated")
    if requested_model == "google/gemini-3.6-flash" and call.metadata_status == "complete":
        if not call.is_byok:
            raise RuntimeError("Gemini response is not marked BYOK")
        if routed_provider(call).lower() != "google ai studio":
            raise RuntimeError(f"Gemini routed through {routed_provider(call)!r}")
    if (
        provider_protocol == "v5"
        and requested_model == "anthropic/claude-sonnet-5"
        and call.metadata_status == "complete"
    ):
        if routed_provider(call).lower() != "anthropic":
            raise RuntimeError(f"Sonnet routed through {routed_provider(call)!r}")


def tool_calls(message: dict) -> list[dict]:
    calls = message.get("tool_calls") or []
    if not isinstance(calls, list):
        raise RuntimeError("tool_calls is not a list")
    return calls


def parse_search_call(call: dict) -> tuple[str, str]:
    call_id = str(call.get("id") or "")
    function = call.get("function")
    if not call_id or not isinstance(function, dict) or function.get("name") != "search":
        raise RuntimeError("invalid search tool call")
    arguments = function.get("arguments")
    if isinstance(arguments, str):
        parsed = json.loads(arguments)
    elif isinstance(arguments, dict):
        parsed = arguments
    else:
        raise RuntimeError("search arguments are not JSON")
    if set(parsed) != {"query"} or not isinstance(parsed["query"], str) or not parsed["query"].strip():
        raise RuntimeError("search arguments must contain only a non-empty query")
    return call_id, parsed["query"]


def sanitized_assistant_message(message: dict, kept_calls: list[dict]) -> dict:
    """Return only the tool calls that will receive matching tool results."""
    sanitized = copy.deepcopy(message)
    if kept_calls:
        sanitized["tool_calls"] = copy.deepcopy(kept_calls)
    else:
        sanitized.pop("tool_calls", None)
        content = sanitized.get("content")
        if not isinstance(content, str) or not content.strip():
            sanitized["content"] = "Additional search requests were omitted at the limit."
    return sanitized


def ensure_raw_policy_event(row: dict, expected: dict, stable_fields: tuple[str, ...]) -> dict:
    try:
        events = load_existing_events(raw_path(row))
        existing = event_for_call(
            events,
            str(expected["event"]),
            int(expected["call_index"]),
        )
    except RecoveryInvariantError:
        raise
    except Exception as error:
        raise RecoveryInvariantError(
            f"{row['run_id']}/{expected.get('call_index')} cannot validate "
            f"persisted {expected.get('event')} policy evidence: {error}"
        ) from error
    if existing is None:
        append_jsonl(raw_path(row), expected)
        return expected
    differences = [
        field_name
        for field_name in stable_fields
        if existing.get(field_name) != expected.get(field_name)
    ]
    if differences:
        raise RecoveryInvariantError(
            f"logged {expected['event']} event differs in {', '.join(differences)}"
        )
    return existing


def policy_sha256(value: Any) -> str:
    return sha256_bytes(canonical_json(value).encode("utf-8"))


def stable_search_execution_id(
    run_id_value: str,
    call_index: int,
    tool_call_id: str,
    query: str,
    logical_backend: str,
    protocol: str = SEARCH_EXECUTION_ID_PROTOCOL_V2,
) -> str:
    if protocol not in {
        SEARCH_EXECUTION_ID_PROTOCOL_V2,
        SEARCH_EXECUTION_ID_PROTOCOL_V3,
    }:
        raise ValueError(f"unknown search execution ID protocol: {protocol!r}")
    return policy_sha256(
        {
            "protocol": protocol,
            "run_id": run_id_value,
            "call_index": call_index,
            "tool_call_id": tool_call_id,
            "query": query,
            "logical_backend": logical_backend,
        }
    )[:32]


def search_execution_id_protocol(row: dict) -> str:
    if row.get("arm") in LEGACY_V1_V2_ARMS:
        return SEARCH_EXECUTION_ID_PROTOCOL_V2
    return SEARCH_EXECUTION_ID_PROTOCOL_V3


STRICT_V3_RESULT_FIELDS = frozenset(
    {
        "query",
        "normalized_query",
        "backend",
        "source_backend",
        "source_engine",
        "transport_id",
        "history_id",
        "cache_protocol",
        "cache_namespace",
        "cache_hit",
        "results",
    }
)
V5_QUERY_EXHAUSTED_RESULT_FIELDS = frozenset(
    {
        "status",
        "query_exhaustion_protocol",
        "query",
        "normalized_query",
        "backend",
        "source_backend",
        "source_engine",
        "transport_id",
        "history_id",
        "cache_protocol",
        "cache_namespace",
        "cache_hit",
        "attempt_count",
        "error_type",
        "error",
    }
)
STRICT_V3_PROVENANCE_FIELDS = frozenset(
    {
        "protocol",
        "history_id",
        "transport_id",
        "transport_sha256",
        "search_execution_id",
        "query",
        "normalized_query",
        "logical_backend",
        "returned_source_backend",
        "returned_source_engine",
        "cache_protocol",
        "cache_hit",
        "upstream_attempted",
        "prior_rejected_attempts",
        "accepted_attempt",
        "accepted_source_backend",
        "accepted_source_engine",
        "accepted_backend_attempt_sha256",
        "accepted_results_sha256",
        "result_sha256",
        "recovered_from_backend_history",
    }
)
STRICT_V3_SOURCE_PAIRS = {
    "web": {
        ("ddgs", DDGS_ENGINE_V3),
        ("brave", BRAVE_SOURCE_ENGINE_V3),
    },
    "scholarly": {("scholarly", SCHOLARLY_SOURCE_ENGINE_V3)},
}
STRICT_V3_BACKEND_ATTEMPT_FIELDS = frozenset(
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


def _sha256_text(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def is_query_exhausted_result(result: Any) -> bool:
    return (
        isinstance(result, dict)
        and result.get("status") == SEARCH_QUERY_EXHAUSTED_STATUS
        and result.get("query_exhaustion_protocol")
        == SEARCH_QUERY_EXHAUSTION_PROTOCOL_V5
    )


def validate_v5_query_exhausted_result(
    result: Any,
    query: str,
    logical_backend: str,
) -> dict:
    """Validate the fixed model-visible no-results envelope."""
    expected_error = (
        "No results found after four exact DDGS attempts. "
        "Broaden the query and search again."
    )
    if not isinstance(result, dict) or set(result) != V5_QUERY_EXHAUSTED_RESULT_FIELDS:
        raise RuntimeError("v5 exhausted-query result has an invalid shape")
    if (
        logical_backend != "web"
        or result.get("status") != SEARCH_QUERY_EXHAUSTED_STATUS
        or result.get("query_exhaustion_protocol")
        != SEARCH_QUERY_EXHAUSTION_PROTOCOL_V5
        or result.get("query") != query
        or result.get("normalized_query") != normalize_query(query)
        or result.get("backend") != "ddgs"
        or result.get("source_backend") != "ddgs"
        or result.get("source_engine") != DDGS_ENGINE_V3
        or result.get("transport_id") != SEARCH_TRANSPORT_ID_V3
        or result.get("history_id") != SEARCH_BACKEND_HISTORY_ID_V3
        or result.get("cache_protocol") != CACHE_PROTOCOL_V3
        or result.get("cache_namespace") != "web"
        or result.get("cache_hit") is not False
        or result.get("attempt_count") != 4
        or result.get("error_type") != "SearchQueryExhausted"
        or result.get("error") != expected_error
        or "results" in result
    ):
        raise RuntimeError("v5 exhausted-query result changes its frozen identity")
    return result


def validate_strict_v3_search_result(
    result: Any,
    query: str,
    logical_backend: str,
) -> dict:
    """Validate the complete prospective result envelope without repairing it."""
    if not isinstance(result, dict) or set(result) != STRICT_V3_RESULT_FIELDS:
        raise RuntimeError("strict-v3 search result has an invalid shape")
    pair = (result.get("source_backend"), result.get("source_engine"))
    if (
        logical_backend not in STRICT_V3_SOURCE_PAIRS
        or pair not in STRICT_V3_SOURCE_PAIRS[logical_backend]
        or result.get("backend") != pair[0]
        or result.get("query") != query
        or result.get("normalized_query") != normalize_query(query)
        or result.get("transport_id") != SEARCH_TRANSPORT_ID_V3
        or result.get("history_id") != SEARCH_BACKEND_HISTORY_ID_V3
        or result.get("cache_protocol") != CACHE_PROTOCOL_V3
        or result.get("cache_namespace") != logical_backend
        or not isinstance(result.get("cache_hit"), bool)
    ):
        raise RuntimeError("strict-v3 search result changes its frozen identity")
    try:
        validate_result_set(result.get("results"))
    except (TypeError, ValueError, RuntimeError) as error:
        raise RuntimeError(f"strict-v3 search result rows are invalid: {error}") from error
    return result


def validate_strict_v3_search_provenance(
    provenance: Any,
    result: dict,
    query: str,
    logical_backend: str,
    expected_execution_id: str,
) -> dict:
    if not isinstance(provenance, dict) or set(provenance) != STRICT_V3_PROVENANCE_FIELDS:
        raise RuntimeError("persisted strict-v3 search provenance has an invalid shape")
    pair = (result["source_backend"], result["source_engine"])
    expected_identity = {
        "protocol": SEARCH_EXECUTION_PROTOCOL_V3,
        "history_id": SEARCH_BACKEND_HISTORY_ID_V3,
        "transport_id": SEARCH_TRANSPORT_ID_V3,
        "search_execution_id": expected_execution_id,
        "query": query,
        "normalized_query": result["normalized_query"],
        "logical_backend": logical_backend,
        "returned_source_backend": pair[0],
        "returned_source_engine": pair[1],
        "cache_protocol": CACHE_PROTOCOL_V3,
        "cache_hit": result["cache_hit"],
        "result_sha256": policy_sha256(result),
    }
    if any(
        provenance.get(field_name) != expected
        for field_name, expected in expected_identity.items()
    ) or not _sha256_text(provenance.get("transport_sha256")):
        raise RuntimeError("persisted strict-v3 search provenance differs")
    rejected = provenance.get("prior_rejected_attempts")
    accepted = provenance.get("accepted_attempt")
    if (
        not isinstance(provenance.get("upstream_attempted"), bool)
        or isinstance(rejected, bool)
        or not isinstance(rejected, int)
        or not 0 <= rejected <= 4
        or not isinstance(provenance.get("recovered_from_backend_history"), bool)
        or (accepted is not None and (isinstance(accepted, bool) or not isinstance(accepted, int)))
        or (isinstance(accepted, int) and not 1 <= accepted <= 4)
        or provenance.get("upstream_attempted") is not bool(rejected or accepted)
    ):
        raise RuntimeError("persisted strict-v3 search provenance has invalid attempt state")
    accepted_fields = (
        provenance.get("accepted_source_backend"),
        provenance.get("accepted_source_engine"),
        provenance.get("accepted_backend_attempt_sha256"),
        provenance.get("accepted_results_sha256"),
    )
    if accepted is None:
        if is_query_exhausted_result(result):
            if (
                any(value is not None for value in accepted_fields)
                or result["cache_hit"] is not False
                or rejected != 4
                or provenance["upstream_attempted"] is not True
            ):
                raise RuntimeError(
                    "persisted v5 exhausted-query provenance is inconsistent"
                )
        elif (
            any(value is not None for value in accepted_fields)
            or result["cache_hit"] is not True
            or rejected > 3
            or provenance["recovered_from_backend_history"] is not bool(rejected)
        ):
            raise RuntimeError("persisted strict-v3 cache-hit provenance is inconsistent")
    elif (
        rejected != accepted - 1
        or accepted_fields[:2] != pair
        or not _sha256_text(accepted_fields[2])
        or accepted_fields[3] != policy_sha256(result["results"])
    ):
        raise RuntimeError("persisted strict-v3 accepted-attempt provenance is inconsistent")
    return provenance


def sealed_history_events(events_path: Path) -> list[dict]:
    """Immutable sealed segments of the same backend-history stream.

    A resumed run whose searches predate a history seal links to attempt
    events that now live in a sealed segment rather than the live file.
    """
    events: list[dict] = []
    pattern = f"{events_path.stem}_sealed_*.jsonl"
    for path in sorted(events_path.parent.glob(pattern)):
        try:
            events.extend(read_jsonl(path))
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
            raise RuntimeError(
                f"strict-v3 resume cannot read sealed backend history: {error}"
            ) from error
    return events


def validate_strict_v3_storage_link(
    result: dict,
    provenance: dict,
    search: Any,
) -> None:
    """Link a resumed result to the independently persisted v3 cache/history."""
    cache_path = getattr(search, "cache_path", None)
    events_path = getattr(search, "events_path", None)
    backend_events = getattr(search, "backend_events", None)
    transport_sha256 = getattr(search, "transport_sha256", None)
    if (
        not isinstance(cache_path, Path)
        or not isinstance(events_path, Path)
        or not isinstance(backend_events, list)
        or not _sha256_text(transport_sha256)
        or provenance.get("transport_sha256") != transport_sha256
    ):
        raise RuntimeError("strict-v3 resume lacks independent cache/history state")
    try:
        persisted_backend_events = read_jsonl(events_path)
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"strict-v3 resume cannot read its backend history: {error}"
        ) from error
    if persisted_backend_events != backend_events:
        raise RuntimeError("strict-v3 persisted and loaded backend history differ")
    try:
        with sqlite3.connect(f"file:{cache_path.resolve()}?mode=ro", uri=True) as connection:
            validate_v3_cache_schema(connection)
            if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                raise RuntimeError("strict-v3 resumed cache failed integrity_check")
            cache_row = connection.execute(
                f"SELECT query_key,query,source_backend,source_engine,results_json,"
                f"transport_id,backend_attempt_sha256 FROM {V3_CACHE_TABLE} "
                "WHERE logical_backend=? AND query_key=?",
                (result["cache_namespace"], result["normalized_query"]),
            ).fetchone()
    except (OSError, sqlite3.Error) as error:
        raise RuntimeError(f"strict-v3 resume cannot read its cache link: {error}") from error
    if is_query_exhausted_result(result):
        if cache_row is not None:
            raise RuntimeError("v5 exhausted-query result unexpectedly has a cache row")

        def execution_attempts(pool: list[dict]) -> list[dict]:
            found = [
                event
                for event in pool
                if isinstance(event, dict)
                and event.get("event") == SEARCH_BACKEND_ATTEMPT_EVENT
                and event.get("search_execution_id")
                == provenance["search_execution_id"]
            ]
            found.sort(key=lambda event: int(event.get("attempt") or 0))
            return found

        history_pool = persisted_backend_events
        sealed_pool = False
        matching = execution_attempts(history_pool)
        if len(matching) != 4:
            sealed_pool = True
            history_pool = (
                sealed_history_events(events_path) + persisted_backend_events
            )
            matching = execution_attempts(history_pool)
        if (
            len(matching) != 4
            or [event.get("attempt") for event in matching] != [1, 2, 3, 4]
            or any(set(event) != STRICT_V3_BACKEND_ATTEMPT_FIELDS for event in matching)
            or any(
                event.get("schema_version") != SEARCH_BACKEND_EVENT_SCHEMA_V3
                or event.get("history_id") != SEARCH_BACKEND_HISTORY_ID_V3
                or event.get("transport_id") != SEARCH_TRANSPORT_ID_V3
                or (
                    not _sha256_text(event.get("transport_sha256"))
                    if sealed_pool
                    else event.get("transport_sha256") != transport_sha256
                )
                or event.get("configured_web_backend") != "ddgs"
                or event.get("query") != provenance["query"]
                or not isinstance(event.get("transport_metadata"), dict)
                or event["transport_metadata"].get("source_backend") != "ddgs"
                or event["transport_metadata"].get("source_engine")
                != DDGS_ENGINE_V3
                or not is_ddgs_no_results_event(event)
                for event in matching
            )
            or any(
                event.get("event") == SEARCH_BACKEND_POLICY_TERMINAL_EVENT
                and event.get("trigger_search_execution_id")
                == provenance["search_execution_id"]
                for event in history_pool
                if isinstance(event, dict)
            )
        ):
            raise RuntimeError(
                "v5 exhausted-query result lacks four exact DDGS no-results events"
            )
        return
    if cache_row is None:
        raise RuntimeError("strict-v3 resumed result has no cache row")
    try:
        cached_results = json.loads(cache_row[4])
    except (TypeError, json.JSONDecodeError) as error:
        raise RuntimeError("strict-v3 resumed cache row is not JSON") from error
    cache_attempt_hash = cache_row[6]
    if (
        cache_row[0] != result["normalized_query"]
        or normalize_query(str(cache_row[1])) != result["normalized_query"]
        or (cache_row[2], cache_row[3])
        != (result["source_backend"], result["source_engine"])
        or cached_results != result["results"]
        or cache_row[5] != SEARCH_TRANSPORT_ID_V3
        or not _sha256_text(cache_attempt_hash)
    ):
        raise RuntimeError("strict-v3 resumed result differs from its cache row")
    matching = [
        event
        for event in persisted_backend_events
        if isinstance(event, dict) and policy_sha256(event) == cache_attempt_hash
    ]
    sealed_link = False
    if not matching:
        sealed_link = True
        matching = [
            event
            for event in sealed_history_events(events_path)
            if isinstance(event, dict) and policy_sha256(event) == cache_attempt_hash
        ]
    if len(matching) != 1:
        raise RuntimeError("strict-v3 resumed cache row lacks one backend-history event")
    event = matching[0]
    if set(event) != STRICT_V3_BACKEND_ATTEMPT_FIELDS:
        raise RuntimeError("strict-v3 linked backend event has an invalid shape")
    try:
        accepted_results = prepare_result_set(event.get("candidate_results"))
    except (TypeError, ValueError, RuntimeError) as error:
        raise RuntimeError("strict-v3 linked backend event has invalid results") from error
    if (
        event.get("schema_version") != SEARCH_BACKEND_EVENT_SCHEMA_V3
        or event.get("event") != SEARCH_BACKEND_ATTEMPT_EVENT
        or event.get("success") is not True
        or event.get("history_id") != SEARCH_BACKEND_HISTORY_ID_V3
        or event.get("transport_id") != SEARCH_TRANSPORT_ID_V3
        or (
            not _sha256_text(event.get("transport_sha256"))
            if sealed_link
            else event.get("transport_sha256") != transport_sha256
        )
        or event.get("configured_web_backend") != "ddgs"
        or event.get("logical_backend") != result["cache_namespace"]
        or (event.get("source_backend"), event.get("source_engine"))
        != (result["source_backend"], result["source_engine"])
        or not isinstance(event.get("transport_metadata"), dict)
        or event["transport_metadata"].get("source_backend")
        != result["source_backend"]
        or event["transport_metadata"].get("source_engine") != result["source_engine"]
        or normalize_query(str(event.get("query") or "")) != result["normalized_query"]
        or event.get("candidate_results_sha256")
        != policy_sha256(event.get("candidate_results"))
        or event.get("error_type") is not None
        or event.get("error") is not None
        or accepted_results != result["results"]
    ):
        raise RuntimeError("strict-v3 resumed backend-history link differs")
    accepted_hash = provenance.get("accepted_backend_attempt_sha256")
    if accepted_hash is not None and (
        accepted_hash != cache_attempt_hash
        or event.get("search_execution_id") != provenance["search_execution_id"]
        or event.get("attempt") != provenance["accepted_attempt"]
        or event.get("query") != provenance["query"]
    ):
        raise RuntimeError("strict-v3 accepted attempt is not bound to cache/history")


def _validate_persisted_tool_results(
    row: dict,
    call_index: int,
    kept: list[tuple[dict, str, str]],
    tool_event: dict,
    search: Any,
) -> list[dict]:
    """Bind a prospective resume to the exact persisted search execution."""
    records = tool_event.get("tool_results")
    if not isinstance(records, list) or len(records) != len(kept):
        raise RuntimeError("logged tool results do not match tool calls")
    observed_ids = [
        str(record.get("tool_call_id") or "")
        for record in records
        if isinstance(record, dict)
    ]
    expected_ids = [call_id for _, call_id, _ in kept]
    if observed_ids != expected_ids:
        raise RuntimeError("logged tool results differ from kept tool calls")

    # Pilot-v1/v2 raw records remain governed by their sealed historical
    # readers. Prospective trajectories require the versioned v3 envelope.
    if search_execution_id_protocol(row) != SEARCH_EXECUTION_ID_PROTOCOL_V3:
        return records
    if tool_event.get("search_execution_id_protocol") != SEARCH_EXECUTION_ID_PROTOCOL_V3:
        raise RuntimeError("persisted tool results lack the strict-v3 protocol marker")

    expected_record_fields = {
        "tool_call_id",
        "arguments",
        "query",
        "result",
        "search_execution_id_protocol",
        "search_provenance",
    }
    logical_backend = str(row.get("search_backend") or "")
    for (requested_call, call_id, query), record in zip(kept, records):
        if not isinstance(record, dict) or set(record) != expected_record_fields:
            raise RuntimeError("persisted strict-v3 tool-result record has an invalid shape")
        if (
            record.get("arguments")
            != requested_call.get("function", {}).get("arguments")
            or record.get("query") != query
            or record.get("search_execution_id_protocol")
            != SEARCH_EXECUTION_ID_PROTOCOL_V3
        ):
            raise RuntimeError("persisted strict-v3 tool result differs from the tool call")
        expected_execution_id = stable_search_execution_id(
            row["run_id"],
            call_index,
            call_id,
            query,
            logical_backend,
            SEARCH_EXECUTION_ID_PROTOCOL_V3,
        )
        candidate_result = record.get("result")
        if is_query_exhausted_result(candidate_result):
            if not uses_visible_query_exhaustion(row):
                raise RuntimeError(
                    "visible exhausted-query results are not allowed for this arm"
                )
            result = validate_v5_query_exhausted_result(
                candidate_result, query, logical_backend
            )
        else:
            result = validate_strict_v3_search_result(
                candidate_result, query, logical_backend
            )
        provenance = validate_strict_v3_search_provenance(
            record.get("search_provenance"),
            result,
            query,
            logical_backend,
            expected_execution_id,
        )
        validate_strict_v3_storage_link(result, provenance, search)
    return records


def validate_persisted_tool_results(
    row: dict,
    call_index: int,
    kept: list[tuple[dict, str, str]],
    tool_event: dict,
    search: Any,
) -> list[dict]:
    """Reject persisted search drift without classifying it as trajectory failure."""
    try:
        return _validate_persisted_tool_results(
            row,
            call_index,
            kept,
            tool_event,
            search,
        )
    except RecoveryInvariantError:
        raise
    except Exception as error:
        raise RecoveryInvariantError(
            f"{row['run_id']}/{call_index} persisted tool-result recovery "
            f"invariant failed: {error}"
        ) from error


def policy_model_parameters(body: dict) -> dict:
    ignored = {"messages", "tools", "tool_choice"}
    return {key: value for key, value in body.items() if key not in ignored}


def extract_urls(text: str) -> list[str]:
    found = []
    for match in URL_PATTERN.findall(text):
        value = match.rstrip(".,;:!?")
        while value.endswith(")") and value.count(")") > value.count("("):
            value = value[:-1]
        while value.endswith("}") and value.count("}") > value.count("{"):
            value = value[:-1]
        if value not in found:
            found.append(value)
    return found


def load_existing_events(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        return read_jsonl(path)
    except RecoveryInvariantError:
        raise
    except Exception as error:
        raise RecoveryInvariantError(
            f"persisted raw event log is unreadable: {path.name}: {error}"
        ) from error


def event_for_call(events: list[dict], event: str, call_index: int) -> dict | None:
    matches = [
        row for row in events if row.get("event") == event and row.get("call_index") == call_index
    ]
    if len(matches) > 1:
        raise RecoveryInvariantError(
            f"duplicate persisted {event} event for call {call_index}"
        )
    return matches[0] if matches else None


def _decoded_wire(value: Any, label: str) -> bytes:
    if not isinstance(value, dict) or set(value) != {"base64", "sha256", "size_bytes"}:
        raise RuntimeError(f"{label} wire record is invalid")
    try:
        decoded = base64.b64decode(value.get("base64", ""), validate=True)
    except (ValueError, binascii.Error) as error:
        raise RuntimeError(f"{label} wire base64 is invalid") from error
    if (
        isinstance(value.get("size_bytes"), bool)
        or value.get("size_bytes") != len(decoded)
        or value.get("sha256") != sha256_bytes(decoded)
    ):
        raise RuntimeError(f"{label} wire size/hash mismatch")
    return decoded


def _validate_recovered_request(
    *,
    row: dict,
    call_index: int,
    ledger_row: dict,
    response_event: dict,
    expected_body: dict,
    expected_reservation_hard_cap: float,
    expected_reservation_policy: str,
) -> None:
    """Bind a prospective recovered call to its exact request and reservation."""
    if response_event.get("request_body") != expected_body:
        raise RecoveryInvariantError(
            f"{row['run_id']}/{call_index} recovered request differs from the frozen task"
        )
    expected_bytes = canonical_json(expected_body).encode("utf-8")
    try:
        recovered_bytes = _decoded_wire(response_event.get("request_wire"), "request")
    except RuntimeError as error:
        raise RecoveryInvariantError(
            f"{row['run_id']}/{call_index} recovered request wire is invalid"
        ) from error
    if recovered_bytes != expected_bytes:
        raise RecoveryInvariantError(
            f"{row['run_id']}/{call_index} recovered request wire differs"
        )
    try:
        expected_reservation = reservation_provenance(
            expected_body,
            False,
            hard_cap=expected_reservation_hard_cap,
            policy=expected_reservation_policy,
        )
    except ValueError as error:
        raise RecoveryInvariantError(
            f"{row['run_id']}/{call_index} expected reservation is invalid"
        ) from error
    if response_event.get("budget_reservation") != expected_reservation:
        raise RecoveryInvariantError(
            f"{row['run_id']}/{call_index} recovered raw reservation differs"
        )
    expected_ledger = {
        "reservation_policy": expected_reservation_policy,
        "hard_cap": expected_reservation_hard_cap,
        "reserved_cost": expected_reservation["reserved_cost"],
    }
    for field, expected in expected_ledger.items():
        observed = ledger_row.get(field)
        if isinstance(expected, float):
            try:
                matches = abs(float(observed) - expected) <= 1e-10
            except (TypeError, ValueError):
                matches = False
        else:
            matches = observed == expected
        if not matches:
            raise RecoveryInvariantError(
                f"{row['run_id']}/{call_index} recovered ledger reservation "
                f"differs: {field}"
            )


def _validate_ledger_settlement(
    run_id: str,
    call_index: int,
    ledger_row: dict,
    response_body: dict,
    successful_generation: dict | None,
) -> None:
    try:
        usage = usage_fields(response_body)
    except (TypeError, ValueError, OverflowError) as error:
        raise RuntimeError(
            f"{run_id}/{call_index} raw usage cannot bind ledger settlement: {error}"
        ) from error
    if any(
        usage[field] < 0
        for field in ("input_tokens", "output_tokens", "reasoning_tokens")
    ) or any(
        usage[field] < 0
        for field in ("openrouter_cost", "upstream_cost")
    ):
        raise RuntimeError(f"{run_id}/{call_index} raw usage contains negative values")
    status = ledger_row.get("status")
    if status == "response_received":
        expected = {
            "input_tokens": 0,
            "output_tokens": 0,
            "reasoning_tokens": 0,
            "native_cost": 0.0,
            "openrouter_cost": 0.0,
            "upstream_cost": 0.0,
            "is_byok": 0,
        }
    else:
        generation = successful_generation if status == "complete" else {}
        expected = {
            "input_tokens": usage["input_tokens"],
            "output_tokens": usage["output_tokens"],
            "reasoning_tokens": usage["reasoning_tokens"],
            "native_cost": generation_cost(generation or {}, usage),
            "openrouter_cost": usage["openrouter_cost"],
            "upstream_cost": usage["upstream_cost"],
            "is_byok": int(usage["is_byok"]),
        }
    if int(ledger_row.get("budget_breach") or 0) != 0:
        raise RuntimeError(f"{run_id}/{call_index} ledger budget hard stop is latched")
    for field, expected_value in expected.items():
        observed = ledger_row.get(field)
        if isinstance(expected_value, float):
            try:
                matches = abs(float(observed) - expected_value) <= 1e-10
            except (TypeError, ValueError):
                matches = False
        else:
            matches = observed == expected_value
        if not matches:
            raise RuntimeError(
                f"{run_id}/{call_index} ledger settlement differs: {field}"
            )


async def _validate_raw_ledger_prefix(
    row: dict,
    ledger: Ledger,
    events: list[dict],
) -> None:
    """Fail closed on any raw/ledger divergence before a trajectory can call."""
    if not events:
        if hasattr(ledger, "connect"):
            with ledger.connect() as connection:
                existing = connection.execute(
                    "SELECT 1 FROM calls WHERE run_id=? LIMIT 1", (row["run_id"],)
                ).fetchone()
        else:
            existing = bool(getattr(ledger, "rows", {}))
        if existing:
            raise RuntimeError(f"{row['run_id']}: ledger call prefix has no raw evidence")
        return
    indexed: dict[int, list[dict]] = defaultdict(list)
    previous_index = 0
    response_generation_ids: dict[int, str] = {}
    successful_repair_indexes: set[int] = set()
    for event in events:
        call_index = event.get("call_index")
        if (
            event.get("run_id") != row["run_id"]
            or isinstance(call_index, bool)
            or not isinstance(call_index, int)
            or call_index < 1
        ):
            raise RuntimeError(f"{row['run_id']}: raw call prefix has invalid identity")
        if call_index > SEARCH_LIMIT + 3:
            raise RuntimeError(f"{row['run_id']}: raw call prefix order is impossible")
        if event.get("event") == "generation_metadata_repair":
            generation_id = event.get("generation_id")
            if (
                call_index not in response_generation_ids
                or not isinstance(generation_id, str)
                or generation_id != response_generation_ids[call_index]
            ):
                raise RuntimeError(
                    f"{row['run_id']}/{call_index} metadata repair is not bound "
                    "to an earlier response"
                )
            generation = event.get("generation")
            error = event.get("error")
            successful = (
                isinstance(generation, dict)
                and bool(generation)
                and error is None
                and str(generation.get("id") or "") == generation_id
            )
            failed = generation == {} and isinstance(error, str) and bool(error.strip())
            if not successful and not failed:
                raise RuntimeError(
                    f"{row['run_id']}/{call_index} metadata repair envelope is invalid"
                )
            if successful:
                if call_index in successful_repair_indexes:
                    raise RuntimeError(
                        f"{row['run_id']}/{call_index} has duplicate successful repairs"
                    )
                successful_repair_indexes.add(call_index)
        else:
            if call_index < previous_index:
                raise RuntimeError(f"{row['run_id']}: raw call prefix order is impossible")
            previous_index = call_index
            if event.get("event") == "model_response":
                response_body = event.get("response_body")
                generation_id = (
                    str(
                        response_body.get("id")
                        or response_body.get("generation_id")
                        or response_body.get("generationId")
                        or ""
                    )
                    if isinstance(response_body, dict)
                    else ""
                )
                if not generation_id or call_index in response_generation_ids:
                    raise RuntimeError(
                        f"{row['run_id']}/{call_index} raw response identity is invalid"
                    )
                response_generation_ids[call_index] = generation_id
        indexed[call_index].append(event)
    expected_indexes = set(range(1, max(indexed) + 1))
    if set(indexed) != expected_indexes:
        raise RuntimeError(f"{row['run_id']}: raw call prefix is noncontiguous")
    for call_index, call_events in sorted(indexed.items()):
        ledger_row = await ledger.get_call(row["run_id"], call_index)
        if ledger_row is None:
            raise RuntimeError(
                f"{row['run_id']}/{call_index} has raw call evidence but no ledger row"
            )
        if ledger_row.get("call_kind") != "agent":
            raise RuntimeError(f"{row['run_id']}/{call_index} is not an agent call")
        responses = [event for event in call_events if event.get("event") == "model_response"]
        if len(responses) > 1:
            raise RuntimeError(f"{row['run_id']}/{call_index} has duplicate raw responses")
        if ledger_row.get("status") in {"complete", "response_received", "metadata_pending"}:
            if len(responses) != 1:
                raise RuntimeError(
                    f"{row['run_id']}/{call_index} ledger row lacks one raw response"
                )
            response = responses[0]
            request_body = response.get("request_body")
            response_body = response.get("response_body")
            if not isinstance(request_body, dict) or not isinstance(response_body, dict):
                raise RuntimeError(f"{row['run_id']}/{call_index} raw bodies are missing")
            request_bytes = _decoded_wire(response.get("request_wire"), "request")
            if request_bytes != canonical_json(request_body).encode("utf-8"):
                raise RuntimeError(f"{row['run_id']}/{call_index} request body/wire mismatch")
            response_bytes = _decoded_wire(response.get("response_wire"), "response")
            try:
                decoded_response = json.loads(response_bytes.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise RuntimeError(
                    f"{row['run_id']}/{call_index} response wire is not JSON"
                ) from error
            if decoded_response != response_body:
                raise RuntimeError(f"{row['run_id']}/{call_index} response body/wire mismatch")
            raw_generation_id = str(
                response_body.get("id")
                or response_body.get("generation_id")
                or response_body.get("generationId")
                or ""
            )
            if raw_generation_id != str(ledger_row.get("generation_id") or ""):
                raise RuntimeError(f"{row['run_id']}/{call_index} generation ID mismatch")
            if (
                request_body.get("model") != ledger_row.get("model")
                or ledger_row.get("model") != row.get("model")
                or response.get("call_kind") != ledger_row.get("call_kind")
            ):
                raise RuntimeError(f"{row['run_id']}/{call_index} call identity mismatch")
            metadata_rows = [
                event
                for event in call_events
                if event.get("event")
                in {"generation_metadata", "generation_metadata_repair"}
            ]
            if any(
                event.get("generation_id") != raw_generation_id
                for event in metadata_rows
            ):
                raise RuntimeError(
                    f"{row['run_id']}/{call_index} generation metadata linkage mismatch"
                )
            successful_metadata: list[dict] = []
            if ledger_row.get("status") == "complete":
                successful_metadata = [
                    event
                    for event in call_events
                    if event.get("event")
                    in {"generation_metadata", "generation_metadata_repair"}
                    and event.get("generation_id") == raw_generation_id
                    and isinstance(event.get("generation"), dict)
                    and event.get("generation")
                    and event.get("error") is None
                    and str(event["generation"].get("id") or "")
                    == raw_generation_id
                ]
                if len(successful_metadata) != 1:
                    raise RuntimeError(
                        f"{row['run_id']}/{call_index} complete call lacks exactly "
                        "one successful generation metadata event"
                    )
            _validate_ledger_settlement(
                row["run_id"],
                call_index,
                ledger_row,
                response_body,
                (
                    successful_metadata[0]["generation"]
                    if successful_metadata
                    else None
                ),
            )
            if hasattr(ledger, "connect"):
                with ledger.connect() as connection:
                    attempts = connection.execute(
                        """SELECT attempt,request_sha256,response_sha256,response_status,
                                  generation_id,error
                           FROM wire_attempts WHERE run_id=? AND call_index=?
                           ORDER BY attempt""",
                        (row["run_id"], call_index),
                    ).fetchall()
                attempt = response.get("attempt")
                matching = [item for item in attempts if item["attempt"] == attempt]
                if (
                    len(matching) != 1
                    or attempts[-1]["attempt"] != attempt
                    or matching[0]["request_sha256"] != response["request_wire"]["sha256"]
                    or matching[0]["response_sha256"] != response["response_wire"]["sha256"]
                    or matching[0]["response_status"] != 200
                    or str(matching[0]["generation_id"] or "") != raw_generation_id
                    or matching[0]["error"] is not None
                ):
                    raise RuntimeError(
                        f"{row['run_id']}/{call_index} wire-attempt identity mismatch"
                    )
    if hasattr(ledger, "connect"):
        with ledger.connect() as connection:
            ledger_indexes = {
                int(item[0])
                for item in connection.execute(
                    "SELECT call_index FROM calls WHERE run_id=?", (row["run_id"],)
                )
            }
        if ledger_indexes != set(indexed):
            raise RuntimeError(f"{row['run_id']}: raw/ledger call-prefix indexes differ")


async def validate_raw_ledger_prefix(
    row: dict,
    ledger: Ledger,
    events: list[dict],
) -> None:
    """Reject local provenance drift without terminalizing the trajectory."""
    try:
        await _validate_raw_ledger_prefix(row, ledger, events)
    except RecoveryInvariantError:
        raise
    except Exception as error:
        raise RecoveryInvariantError(
            f"{row['run_id']} raw/ledger prefix recovery invariant failed: {error}"
        ) from error


async def recover_call(
    client: OpenRouterClient,
    ledger: Ledger,
    row: dict,
    call_index: int,
    events: list[dict],
    *,
    allow_provisional_metadata: bool = False,
    expected_body: dict | None = None,
    expected_reservation_hard_cap: float | None = None,
    expected_reservation_policy: str | None = None,
) -> ModelCall | None:
    expected_fields = (
        expected_body is not None,
        expected_reservation_hard_cap is not None,
        expected_reservation_policy is not None,
    )
    if any(expected_fields) and not all(expected_fields):
        raise ValueError(
            "recovered-call validation requires body, hard cap, and policy together"
        )
    ledger_row = await ledger.get_call(row["run_id"], call_index)
    if ledger_row is None:
        provider_event_names = {
            "model_call",
            "model_response",
            "generation_metadata",
            "generation_metadata_repair",
        }
        malformed_provider_events = [
            event
            for event in events
            if event.get("event") in provider_event_names
            and (
                isinstance(event.get("call_index"), bool)
                or not isinstance(event.get("call_index"), int)
            )
        ]
        call_events = [event for event in events if event.get("call_index") == call_index]
        if malformed_provider_events or call_events:
            kinds = sorted(
                {
                    str(event.get("event"))
                    for event in [*malformed_provider_events, *call_events]
                }
            )
            raise RuntimeError(
                f"{row['run_id']}/{call_index} has raw call evidence "
                f"{kinds!r} but no ledger row; refusing a paid resubmission"
            )
        return None
    status = ledger_row["status"]
    response_event = event_for_call(events, "model_response", call_index)
    if status in {"inflight", "unknown_inflight"}:
        raise RuntimeError(
            f"{row['run_id']}/{call_index} is {status}; refusing an ambiguous resubmission"
        )
    if status == "failed":
        raise RuntimeError(f"{row['run_id']}/{call_index} has a terminal failed call")
    if response_event is None:
        raise RuntimeError(f"ledger has {status} but raw response is missing")
    payload = response_event.get("response_body")
    if not isinstance(payload, dict):
        raise RuntimeError("raw response body is missing")
    if expected_body is not None:
        if (
            expected_reservation_hard_cap is None
            or expected_reservation_policy is None
        ):
            raise RuntimeError("recovered-call validation fields are incomplete")
        _validate_recovered_request(
            row=row,
            call_index=call_index,
            ledger_row=ledger_row,
            response_event=response_event,
            expected_body=expected_body,
            expected_reservation_hard_cap=expected_reservation_hard_cap,
            expected_reservation_policy=expected_reservation_policy,
        )
    generation_id = str(ledger_row.get("generation_id") or payload.get("id") or "")
    metadata_events = [
        event
        for event in events
        if event.get("event") in {"generation_metadata", "generation_metadata_repair"}
        and event.get("call_index") == call_index
        and isinstance(event.get("generation"), dict)
        and event.get("generation")
    ]
    generation = metadata_events[-1]["generation"] if metadata_events else {}
    if status == "metadata_pending" and allow_provisional_metadata:
        successful_metadata = [
            event
            for event in metadata_events
            if event.get("error") is None
            and str(event.get("generation", {}).get("id") or "") == generation_id
        ]
        if successful_metadata:
            raise RuntimeError(
                "pending ledger row already has successful generation metadata; "
                "settlement repair is required"
            )
        return ModelCall(
            payload=payload,
            generation={},
            call_index=call_index,
            input_tokens=int(ledger_row["input_tokens"]),
            output_tokens=int(ledger_row["output_tokens"]),
            reasoning_tokens=int(ledger_row["reasoning_tokens"]),
            native_cost=float(ledger_row["native_cost"]),
            openrouter_cost=float(ledger_row["openrouter_cost"]),
            upstream_cost=float(ledger_row["upstream_cost"]),
            is_byok=bool(ledger_row["is_byok"]),
            metadata_status="metadata_pending",
        )
    if status in {"response_received", "metadata_pending"}:
        generation = await client.generation(generation_id)
        append_jsonl(
            raw_path(row),
            {
                "event": "generation_metadata_repair",
                "run_id": row["run_id"],
                "call_index": call_index,
                "generation_id": generation_id,
                "timestamp": now_iso(),
                "generation": generation,
                "error": None,
            },
        )
        usage = usage_fields(payload)
        await ledger.repair_metadata(
            row["run_id"],
            call_index,
            {
                **usage,
                "generation_id": generation_id,
                "native_cost": generation_cost(generation, usage),
            },
        )
        ledger_row = await ledger.get_call(row["run_id"], call_index)
        if ledger_row is None:
            raise RuntimeError("metadata repair lost its ledger row")
    usage = usage_fields(payload)
    return ModelCall(
        payload=payload,
        generation=generation if isinstance(generation, dict) else {},
        call_index=call_index,
        input_tokens=int(ledger_row["input_tokens"]),
        output_tokens=int(ledger_row["output_tokens"]),
        reasoning_tokens=int(ledger_row["reasoning_tokens"]),
        native_cost=float(ledger_row["native_cost"]),
        openrouter_cost=float(ledger_row["openrouter_cost"]),
        upstream_cost=float(ledger_row["upstream_cost"]),
        is_byok=bool(ledger_row["is_byok"]),
        metadata_status="complete",
    )


def call_summary(
    call: ModelCall,
    requested_model: str | None = None,
    call_kind: str | None = None,
) -> dict:
    value = {
        "call_index": call.call_index,
        "generation_id": str(call.payload.get("id") or ""),
        "model": requested_model or str(call.payload.get("model") or ""),
        "response_model": str(call.payload.get("model") or ""),
        "provider": routed_provider(call),
        "finish_reason": finish_reason(call.payload),
        "input_tokens": call.input_tokens,
        "output_tokens": call.output_tokens,
        "reasoning_tokens": call.reasoning_tokens,
        "native_cost": round(float(call.native_cost), 10),
        "openrouter_cost": round(float(call.openrouter_cost), 10),
        "upstream_cost": round(float(call.upstream_cost), 10),
        "is_byok": call.is_byok,
    }
    if call_kind is not None:
        value["call_kind"] = call_kind
    return value


def uses_fixed_evidence(row: dict) -> bool:
    """Replay rows read frozen evidence instead of searching.

    The registered arm reuses its own claim's neutral run. Phase-2 rows name
    their source run explicitly so the same path can serve swapped evidence.
    """
    return row.get("arm") == "replay" or bool(row.get("evidence_source_run_id"))


def replay_context(row: dict) -> dict:
    path = replay_context_path(row)
    if not path.exists():
        raise RuntimeError(f"missing replay context: {path.name}")
    value = load_json(path)
    if not isinstance(value, dict) or not isinstance(value.get("searches"), list):
        raise RuntimeError(f"invalid replay context: {path.name}")
    return value


def replay_context_path(row: dict) -> Path:
    evidence_framing = row.get("evidence_framing")
    if evidence_framing:
        return cross_replay_context_path(
            str(row["claim_id"]), str(row["system"]), str(evidence_framing)
        )
    return ARCHIVE / "replay_context" / f"{row['claim_id']}_{row['system']}.json"


def replay_source_run_id(row: dict) -> str:
    recorded = row.get("evidence_source_run_id")
    if recorded:
        return str(recorded)
    return schedule_run_id("core", row["claim_id"], "neutral", row["system"], 1)


def replay_source_terminal(row: dict) -> bool:
    source_path = RUNS_DIR / f"{replay_source_run_id(row)}.json"
    if not source_path.exists():
        return False
    source = load_json(source_path)
    return (
        isinstance(source, dict)
        and source.get("status") == TERMINAL_FAILURE_STATUS
    )


def initial_messages(row: dict, claim: dict, prompts: dict) -> list[dict]:
    system = prompts["system"]
    minimum_searches = prospective_minimum_searches(row)
    if minimum_searches:
        instruction = prompts.get("minimum_searches")
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("prompts are missing the prospective minimum-search rule")
        system = f"{system}\n{instruction}"
    if row["arm"] in {"mitigation", "skeptic_mitigation"}:
        system = f"{system}\n{prompts['mitigation']}"
    claim_form = claim.get("claim_form")
    if claim_form not in {"statement", "question"}:
        raise ValueError(f"invalid claim_form: {claim_form!r}")
    framing_family = prompts["framings"].get(claim_form)
    if not isinstance(framing_family, dict):
        raise ValueError(f"prompts are missing the {claim_form!r} framing family")
    template = framing_family.get(row["framing"])
    if not isinstance(template, str) or not template:
        raise ValueError(
            f"prompts are missing framing {row['framing']!r} "
            f"for claim_form {claim_form!r}"
        )
    question = template.format(claim=claim["claim"])
    if uses_fixed_evidence(row):
        context = replay_context(row)
        fixed = json.dumps(context["searches"], ensure_ascii=False, indent=2)
        question = (
            f"{question}\n\nUse only the fixed search results below as retrieved evidence. "
            f"Do not add sources that are not present.\n\n{fixed}"
        )
    return [{"role": "system", "content": system}, {"role": "user", "content": question}]


def request_body(
    row: dict,
    messages: list[dict],
    search_count: int,
    *,
    successful_search_count: int | None = None,
) -> dict:
    """Build one request using total tool calls for the cap and successes for the floor."""
    successful_count = (
        search_count
        if successful_search_count is None
        else successful_search_count
    )
    body = {
        "model": row["model"],
        "messages": copy.deepcopy(messages),
        **row_model_params(row),
        "include_reasoning": True,
        "usage": {"include": True},
        "provider": row_provider_route(row),
    }
    if row["search_backend"] is not None and search_count < SEARCH_LIMIT:
        body["tools"] = [SEARCH_TOOL]
        minimum_searches = prospective_minimum_searches(row)
        body["tool_choice"] = (
            "required"
            if successful_count == 0 or successful_count < minimum_searches
            else "auto"
        )
    return body


def save_state(row: dict, status: str, **values: Any) -> None:
    current = {}
    path = state_path(row)
    if path.exists():
        current = load_recovery_state(row)
    current.update({"run_id": row["run_id"], "status": status, "updated_at": now_iso(), **values})
    atomic_write(path, pretty_json(current))


def load_recovery_state(row: dict) -> dict:
    """Load an existing local state record without repairing or replacing it."""
    path = state_path(row)
    try:
        state = load_json(path)
    except Exception as error:
        raise RecoveryInvariantError(
            f"{row['run_id']} persisted recovery state is unreadable: {error}"
        ) from error
    if (
        not isinstance(state, dict)
        or state.get("run_id") != row["run_id"]
        or not isinstance(state.get("status"), str)
        or not state["status"].strip()
    ):
        raise RecoveryInvariantError(
            f"{row['run_id']} persisted recovery state has an invalid identity"
        )
    return state


def save_replay_context(row: dict, searches: list[dict]) -> None:
    if row["arm"] != "core" or row["framing"] != "neutral" or row["repeat"] != 1:
        return
    path = replay_context_path(row)
    value = {
        "source_run_id": row["run_id"],
        "claim_id": row["claim_id"],
        "system": row["system"],
        "searches": searches,
    }
    if path.exists():
        try:
            existing = load_json(path)
        except Exception as error:
            raise RecoveryInvariantError(
                f"persisted replay context is unreadable for {row['run_id']}: {error}"
            ) from error
        if existing != value:
            raise RecoveryInvariantError(
                f"persisted replay context changed for {row['run_id']}"
            )
    atomic_write(path, pretty_json(value))


def publish_manifest(row: dict, manifest: dict, searches: list[dict]) -> None:
    save_replay_context(row, searches)
    atomic_write(manifest_path(row), pretty_json(manifest))


async def require_manifest_settlement(
    row: dict,
    ledger: Ledger,
    call_count: int,
    *,
    recovered_call_indexes: set[int] | None = None,
) -> None:
    """Bind publication to a fully settled, revalidated call prefix."""
    from repair_metadata import PendingCall, validate_generation_metadata

    events = load_existing_events(raw_path(row))
    await validate_raw_ledger_prefix(row, ledger, events)
    recovered = set(recovered_call_indexes or set())
    reasoning_present = False
    for call_index in range(1, call_count + 1):
        try:
            ledger_row = await ledger.get_call(row["run_id"], call_index)
            if ledger_row is None or ledger_row.get("status") != "complete":
                status = ledger_row.get("status") if ledger_row else "missing"
                raise RuntimeError(
                    f"manifest publication requires settled metadata for "
                    f"{row['run_id']}/{call_index}; found {status}"
                )
            responses = [
                event
                for event in events
                if event.get("event") == "model_response"
                and event.get("call_index") == call_index
            ]
            metadata = [
                event
                for event in events
                if event.get("event")
                in {"generation_metadata", "generation_metadata_repair"}
                and event.get("call_index") == call_index
                and event.get("generation_id") == ledger_row.get("generation_id")
                and isinstance(event.get("generation"), dict)
                and event.get("generation")
                and event.get("error") is None
            ]
            if len(responses) != 1 or len(metadata) != 1:
                raise RuntimeError(
                    f"manifest publication lacks exact response/metadata evidence for "
                    f"{row['run_id']}/{call_index}"
                )
            payload = responses[0].get("response_body")
            if not isinstance(payload, dict):
                raise RuntimeError("manifest publication response body is missing")
            validate_generation_metadata(
                metadata[0]["generation"],
                PendingCall(
                    row["run_id"],
                    call_index,
                    str(ledger_row.get("generation_id") or ""),
                ),
                payload,
                row["model"],
            )
            assistant = response_message(payload)
            details = assistant.get("reasoning_details")
            reasoning_present = reasoning_present or any(
                (
                    int(ledger_row.get("reasoning_tokens") or 0) > 0,
                    bool(str(assistant.get("reasoning") or "").strip()),
                    bool(str(assistant.get("reasoning_content") or "").strip()),
                    isinstance(details, list) and bool(details),
                    any(
                        int(metadata[0]["generation"].get(key) or 0) > 0
                        for key in (
                            "reasoning_tokens",
                            "tokens_reasoning",
                            "native_tokens_reasoning",
                        )
                    ),
                )
            )
        except RecoveryInvariantError:
            raise
        except Exception as error:
            if call_index in recovered:
                raise RecoveryInvariantError(
                    f"{row['run_id']}/{call_index} recovered manifest evidence "
                    f"failed settlement validation: {error}"
                ) from error
            raise
    # Reasoning is provider-dependent per trajectory, so no assertion is made
    # here. The frozen v5 preflight is what proves each model accepts the
    # control and exposes reasoning evidence.


async def _run_one_impl(
    row: dict,
    claims: dict[str, dict],
    prompts: dict,
    client: OpenRouterClient,
    ledger: Ledger,
    search: SearchTool,
    semaphore: asyncio.Semaphore,
    include_kimi: bool,
) -> str:
    output_path = manifest_path(row)
    if output_path.exists():
        manifest = load_json(output_path)
        if isinstance(manifest, dict) and manifest.get("status") in {
            "agent_complete",
            "judged",
            "analysis_ready",
            TERMINAL_FAILURE_STATUS,
        }:
            return "skipped"
        raise RuntimeError(f"existing manifest is not complete: {output_path.name}")

    if uses_fixed_evidence(row) and await record_blocked_replay(row, ledger):
        return TERMINAL_FAILURE_STATUS

    existing_events = load_existing_events(raw_path(row))
    await validate_raw_ledger_prefix(row, ledger, existing_events)
    claim = claims[row["claim_id"]]
    started_at = now_iso()
    existing_state = state_path(row)
    if existing_state.exists():
        state = load_recovery_state(row)
        started_at = str(state.get("started_at") or started_at)
    save_state(row, "running", started_at=started_at)
    messages = initial_messages(row, claim, prompts)
    searches: list[dict] = []
    queries: list[str] = []
    search_tool_queries: list[str] = []
    query_exhaustions: list[dict] = []
    search_tool_call_count = 0
    result_urls: list[str] = []
    calls: list[dict] = []
    report = ""
    call_index = 0
    seen_tool_call_ids: set[str] = set()
    truncation_used = False
    recovery_pending = False
    provisional_call_index: int | None = None
    recovered_call_indexes: set[int] = set()

    async with semaphore:
        while call_index < SEARCH_LIMIT + 3:
            call_index += 1
            body = request_body(
                row,
                messages,
                search_tool_call_count,
                successful_search_count=len(searches),
            )
            if recovery_pending:
                body.pop("tools", None)
                body.pop("tool_choice", None)
            save_state(
                row,
                "awaiting_response",
                started_at=started_at,
                call_index=call_index,
                request_body=body,
            )
            events = load_existing_events(raw_path(row))
            recovery_kwargs = (
                {
                    "expected_body": body,
                    "expected_reservation_hard_cap": (
                        OPENROUTER_LIMIT_WITHOUT_KIMI
                    ),
                    "expected_reservation_policy": V3_RESERVATION_POLICY,
                }
                if search_execution_id_protocol(row)
                == SEARCH_EXECUTION_ID_PROTOCOL_V3
                else {}
            )
            call = await recover_call(
                client,
                ledger,
                row,
                call_index,
                events,
                allow_provisional_metadata=True,
                **recovery_kwargs,
            )
            if call is None:
                call = await client.call(
                    body=body,
                    run_id=row["run_id"],
                    call_index=call_index,
                    call_kind="agent",
                    raw_path=raw_path(row),
                    include_kimi=include_kimi,
                    reservation_hard_cap=OPENROUTER_LIMIT_WITHOUT_KIMI,
                    reservation_policy=V3_RESERVATION_POLICY,
                    allow_provisional_metadata=True,
                )
            else:
                recovered_call_indexes.add(call_index)
            validate_response(
                call,
                row["model"],
                allow_truncated=True,
                allow_metadata_pending=True,
                provider_protocol=(
                    "v5" if uses_v5_model_protocol(row) else "legacy"
                ),
            )
            if call.metadata_status == "metadata_pending":
                await validate_raw_ledger_prefix(
                    row,
                    ledger,
                    load_existing_events(raw_path(row)),
                )
                provisional_call_index = call_index
            summary = call_summary(call, row["model"], "agent")
            calls.append(summary)
            assistant = response_message(call.payload)
            requested = tool_calls(assistant)
            reason = finish_reason(call.payload)

            if reason in TRUNCATION_FINISH_REASONS:
                if requested:
                    raise RuntimeError(
                        "truncated response included tool calls and cannot be recovered"
                    )
                if truncation_used:
                    raise RuntimeError("truncation recovery response was also truncated")
                truncation_used = True
                recovery_pending = True
                content = assistant.get("content")
                if isinstance(content, str) and content.strip():
                    messages.append(copy.deepcopy(assistant))
                messages.append(
                    {"role": "user", "content": TRUNCATION_RECOVERY_INSTRUCTION}
                )
                ensure_raw_policy_event(
                    row,
                    {
                        "event": TRUNCATION_RECOVERY_EVENT,
                        "run_id": row["run_id"],
                        "call_index": call_index,
                        "recovery_call_index": call_index + 1,
                        "timestamp": now_iso(),
                        "finish_reason": reason,
                        "instruction": TRUNCATION_RECOVERY_INSTRUCTION,
                        "call_summary": summary,
                        "response_body_sha256": policy_sha256(call.payload),
                        "request_params_sha256": policy_sha256(
                            policy_model_parameters(body)
                        ),
                        "call_summary_sha256": policy_sha256(summary),
                    },
                    (
                        "run_id",
                        "recovery_call_index",
                        "finish_reason",
                        "instruction",
                        "response_body_sha256",
                        "request_params_sha256",
                    ),
                )
                save_state(
                    row,
                    (
                        PROVISIONAL_METADATA_STATUS
                        if provisional_call_index == call_index
                        else "truncation_recovery_logged"
                    ),
                    started_at=started_at,
                    call_index=call_index,
                    recovery_call_index=call_index + 1,
                    search_count=len(searches),
                    successful_search_count=len(searches),
                    search_tool_call_count=search_tool_call_count,
                    pending_call_indexes=(
                        [call_index] if provisional_call_index == call_index else []
                    ),
                )
                if provisional_call_index == call_index:
                    return PROVISIONAL_METADATA_STATUS
                continue

            if recovery_pending:
                recovery_pending = False
                if requested:
                    raise RuntimeError("truncation recovery attempted to call a tool")
            if not requested:
                messages.append(copy.deepcopy(assistant))
                content = assistant.get("content")
                if not isinstance(content, str) or not content.strip():
                    raise RuntimeError("assistant returned neither tool calls nor a report")
                if row["search_backend"] is not None and not searches:
                    raise RuntimeError("search-equipped run finished without searching")
                minimum_searches = prospective_minimum_searches(row)
                if len(searches) < minimum_searches:
                    raise RuntimeError(
                        "search-equipped run finished before the prospective minimum "
                        f"of {minimum_searches} searches"
                    )
                report = content.strip()
                break

            if row["search_backend"] is None:
                raise RuntimeError("replay run attempted to call a tool")
            remaining = SEARCH_LIMIT - search_tool_call_count
            parsed_requested = [
                (requested_call, *parse_search_call(requested_call))
                for requested_call in requested
            ]
            attempted_ids = [call_id for _, call_id, _ in parsed_requested]
            if len(attempted_ids) != len(set(attempted_ids)):
                raise RuntimeError("assistant issued duplicate tool call IDs")
            if search_execution_id_protocol(row) == SEARCH_EXECUTION_ID_PROTOCOL_V3:
                reused_ids = sorted(set(attempted_ids) & seen_tool_call_ids)
                if reused_ids:
                    raise RuntimeError(
                        "assistant reused tool call IDs across turns: "
                        + ", ".join(reused_ids)
                    )
                seen_tool_call_ids.update(attempted_ids)
            if len(requested) > remaining:
                kept = parsed_requested[:remaining]
                kept_ids = [call_id for _, call_id, _ in kept]
                dropped_ids = attempted_ids[remaining:]
                kept_calls_for_event = [item for item, _, _ in kept]
                sanitized_for_event = sanitized_assistant_message(
                    assistant, kept_calls_for_event
                )
                ensure_raw_policy_event(
                    row,
                    {
                        "event": TOOL_CALL_CLAMP_EVENT,
                        "run_id": row["run_id"],
                        "call_index": call_index,
                        "timestamp": now_iso(),
                        "search_limit": SEARCH_LIMIT,
                        "remaining_before": remaining,
                        "attempted_count": len(parsed_requested),
                        "kept_count": len(kept),
                        "dropped_count": len(parsed_requested) - len(kept),
                        "attempted_ids": attempted_ids,
                        "kept_ids": kept_ids,
                        "dropped_ids": dropped_ids,
                        "attempted_tool_calls_sha256": policy_sha256(requested),
                        "kept_tool_calls_sha256": policy_sha256(kept_calls_for_event),
                        "sanitized_assistant_sha256": policy_sha256(
                            sanitized_for_event
                        ),
                    },
                    (
                        "run_id",
                        "search_limit",
                        "remaining_before",
                        "attempted_count",
                        "kept_count",
                        "dropped_count",
                        "attempted_ids",
                        "kept_ids",
                        "dropped_ids",
                        "attempted_tool_calls_sha256",
                        "kept_tool_calls_sha256",
                        "sanitized_assistant_sha256",
                    ),
                )
                save_state(
                    row,
                    "tool_call_clamp_logged",
                    started_at=started_at,
                    call_index=call_index,
                    attempted_count=len(parsed_requested),
                    kept_count=len(kept),
                    search_count=len(searches),
                    successful_search_count=len(searches),
                    search_tool_call_count=search_tool_call_count,
                )
            else:
                kept = parsed_requested

            kept_calls = [requested_call for requested_call, _, _ in kept]
            messages.append(sanitized_assistant_message(assistant, kept_calls))
            try:
                current_events = load_existing_events(raw_path(row))
                tool_event = event_for_call(
                    current_events,
                    "tool_results",
                    call_index,
                )
                prior_search_failure = event_for_call(
                    current_events,
                    SEARCH_TOOL_FAILURE_EVENT,
                    call_index,
                )
            except RecoveryInvariantError:
                raise
            except Exception as error:
                raise RecoveryInvariantError(
                    f"{row['run_id']}/{call_index} persisted search-event "
                    f"recovery invariant failed: {error}"
                ) from error
            if prior_search_failure is not None:
                raise RuntimeError(
                    "recorded search-tool failure prevents silent trajectory resume: "
                    + str(prior_search_failure.get("error") or "unknown search error")
                )
            if tool_event is None:
                records = []
                for kept_index, (requested_call, call_id, query) in enumerate(kept):
                    execution_id_protocol = search_execution_id_protocol(row)
                    search_execution_id = stable_search_execution_id(
                        row["run_id"],
                        call_index,
                        call_id,
                        query,
                        str(row["search_backend"]),
                        execution_id_protocol,
                    )
                    attempt_hashes: list[str] = []
                    backend_attempt_hashes: list[str] = []
                    completion_provenance: list[dict[str, Any]] = []

                    def record_rejected_attempt(attempt: dict) -> None:
                        event = {
                            **attempt,
                            "event": SEARCH_ATTEMPT_REJECTED_EVENT,
                            "run_id": row["run_id"],
                            "call_index": call_index,
                            "tool_call_id": call_id,
                            "search_execution_id_protocol": execution_id_protocol,
                            "arguments": requested_call["function"]["arguments"],
                            "timestamp": attempt.get("finished_at") or now_iso(),
                        }
                        try:
                            existing_attempts = [
                                prior
                                for prior in load_existing_events(raw_path(row))
                                if prior.get("event")
                                == SEARCH_ATTEMPT_REJECTED_EVENT
                                and prior.get("search_execution_id")
                                == search_execution_id
                                and prior.get("tool_call_id") == call_id
                                and prior.get("attempt") == event.get("attempt")
                            ]
                        except RecoveryInvariantError:
                            raise
                        except Exception as error:
                            raise RecoveryInvariantError(
                                f"{row['run_id']}/{call_index}/{call_id} cannot "
                                f"validate rejected-attempt recovery evidence: {error}"
                            ) from error
                        if len(existing_attempts) > 1:
                            raise RecoveryInvariantError(
                                "duplicate rejected search attempt provenance"
                            )
                        if existing_attempts:
                            recorded_attempt = existing_attempts[0]
                            if recorded_attempt != event:
                                raise RecoveryInvariantError(
                                    "persisted rejected search attempt differs on resume"
                                )
                        else:
                            append_jsonl(raw_path(row), event)
                            recorded_attempt = event
                        attempt_hash = policy_sha256(recorded_attempt)
                        if attempt_hash not in attempt_hashes:
                            attempt_hashes.append(attempt_hash)
                        backend_hash = recorded_attempt.get(
                            "backend_attempt_sha256"
                        )
                        if (
                            not isinstance(backend_hash, str)
                            or not re.fullmatch(r"[0-9a-f]{64}", backend_hash)
                        ):
                            message = (
                                "rejected search attempt lacks backend-event linkage"
                            )
                            if existing_attempts:
                                raise RecoveryInvariantError(message)
                            raise RuntimeError(message)
                        if backend_hash not in backend_attempt_hashes:
                            backend_attempt_hashes.append(backend_hash)

                    def record_completion(provenance: dict[str, Any]) -> None:
                        completion_provenance.append(copy.deepcopy(provenance))

                    try:
                        result = await search.search(
                            query,
                            row["search_backend"],
                            attempt_recorder=record_rejected_attempt,
                            provenance_recorder=record_completion,
                            search_execution_id=search_execution_id,
                        )
                    except RecoveryInvariantError:
                        raise
                    except Exception as error:
                        if (
                            isinstance(error, SearchQueryExhausted)
                            and uses_visible_query_exhaustion(row)
                        ):
                            result = validate_v5_query_exhausted_result(
                                error.tool_result,
                                query,
                                str(row["search_backend"]),
                            )
                        else:
                            append_jsonl(
                                raw_path(row),
                                {
                                    "event": SEARCH_TOOL_FAILURE_EVENT,
                                    "run_id": row["run_id"],
                                    "call_index": call_index,
                                    "timestamp": now_iso(),
                                    "search_execution_id": search_execution_id,
                                    "search_execution_id_protocol": execution_id_protocol,
                                    "tool_call_id": call_id,
                                    "arguments": requested_call["function"]["arguments"],
                                    "query": query,
                                    "logical_backend": row["search_backend"],
                                    "attempt_event_sha256": attempt_hashes,
                                    "backend_attempt_sha256": backend_attempt_hashes,
                                    "completed_tool_results": records,
                                    "failed_tool_call": requested_call,
                                    "unexecuted_tool_calls": [
                                        item for item, _, _ in kept[kept_index + 1 :]
                                    ],
                                    "error_type": type(error).__name__,
                                    "error": (
                                        " ".join(str(error).split())
                                        or type(error).__name__
                                    )[:2000],
                                },
                            )
                            raise
                    if len(completion_provenance) != 1:
                        raise RuntimeError(
                            "search completion lacks exactly one raw-only provenance record"
                        )
                    records.append(
                        {
                            "tool_call_id": call_id,
                            "arguments": requested_call["function"]["arguments"],
                            "query": query,
                            "result": result,
                            "search_execution_id_protocol": execution_id_protocol,
                            "search_provenance": completion_provenance[0],
                        }
                    )
                append_jsonl(
                    raw_path(row),
                    {
                        "event": "tool_results",
                        "run_id": row["run_id"],
                        "call_index": call_index,
                        "timestamp": now_iso(),
                        "search_execution_id_protocol": search_execution_id_protocol(row),
                        "tool_results": records,
                    },
                )
            else:
                records = validate_persisted_tool_results(
                    row,
                    call_index,
                    kept,
                    tool_event,
                    search,
                )

            for record in records:
                result = record["result"]
                search_tool_call_count += 1
                search_tool_queries.append(record["query"])
                if is_query_exhausted_result(result):
                    query_exhaustions.append(result)
                else:
                    searches.append(result)
                    queries.append(record["query"])
                    result_urls.extend(item["url"] for item in result["results"])
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": record["tool_call_id"],
                        "content": canonical_json(result),
                    }
                )
            if search_tool_call_count >= SEARCH_LIMIT:
                messages.append(
                    {
                        "role": "user",
                        "content": FINAL_REPORT_INSTRUCTION,
                    }
                )
            save_state(
                row,
                (
                    PROVISIONAL_METADATA_STATUS
                    if provisional_call_index == call_index
                    else "tool_results_logged"
                ),
                started_at=started_at,
                call_index=call_index,
                search_count=len(searches),
                successful_search_count=len(searches),
                search_tool_call_count=search_tool_call_count,
                query_exhaustion_count=len(query_exhaustions),
                distinct_search_count=len(set(queries)),
                pending_call_indexes=(
                    [call_index] if provisional_call_index == call_index else []
                ),
            )
            if provisional_call_index == call_index:
                return PROVISIONAL_METADATA_STATUS
        else:
            raise RuntimeError("agent did not finish within the call limit")

    if provisional_call_index is not None:
        save_state(
            row,
            PROVISIONAL_METADATA_STATUS,
            started_at=started_at,
            call_index=provisional_call_index,
            call_count=call_index,
            search_count=len(searches),
            successful_search_count=len(searches),
            search_tool_call_count=search_tool_call_count,
            query_exhaustion_count=len(query_exhaustions),
            pending_call_indexes=[provisional_call_index],
            final_response_received=bool(report),
        )
        return PROVISIONAL_METADATA_STATUS

    await require_manifest_settlement(
        row,
        ledger,
        call_index,
        recovered_call_indexes=recovered_call_indexes,
    )
    citations = extract_urls(report)
    manifest_searches = searches
    if uses_fixed_evidence(row):
        fixed = replay_context(row)
        manifest_searches = fixed["searches"]
        result_urls = [
            result["url"]
            for search_record in manifest_searches
            for result in search_record["results"]
        ]
    manifest = {
        "run_id": row["run_id"],
        "claim_id": row["claim_id"],
        "arm": row["arm"],
        "framing": row["framing"],
        "system": row["system"],
        "model": row["model"],
        "repeat": row["repeat"],
        "search_backend": row["search_backend"] or "fixed_replay",
        "params_hash": frozen_agent_params_hash(row),
        "queries": queries,
        "searches": manifest_searches,
        "result_urls": result_urls,
        "report": report,
        "citations": citations,
        "judge_labels": {},
        "query_labels": [],
        "calls": calls,
        "search_count": len(searches),
        "successful_search_count": len(searches),
        "search_tool_call_count": search_tool_call_count,
        "search_tool_queries": search_tool_queries,
        "query_exhaustion_count": len(query_exhaustions),
        "query_exhaustions": query_exhaustions,
        "distinct_search_count": len(set(queries)),
        "minimum_searches": prospective_minimum_searches(row),
        "started_at": started_at,
        "completed_at": now_iso(),
        "status": "agent_complete",
    }
    publish_manifest(row, manifest, searches)
    save_state(
        row,
        "agent_complete",
        started_at=started_at,
        completed_at=manifest["completed_at"],
        call_count=len(calls),
        search_count=len(searches),
        successful_search_count=len(searches),
        search_tool_call_count=search_tool_call_count,
        query_exhaustion_count=len(query_exhaustions),
    )
    return "completed"


def _write_terminal_record(
    row: dict,
    *,
    events: list[dict],
    recorded: dict,
    error: BaseException,
    call_index: int,
    ledger_status: str,
    started_at: str,
    completed_at: str,
    stage: str,
) -> None:
    manifest = terminal_failure_manifest(
        row,
        params_hash_value=frozen_agent_params_hash(row),
        started_at=started_at,
        completed_at=completed_at,
        stage=stage,
        error=error,
        call_index=call_index,
        ledger_status=ledger_status,
        events=events,
        recorded_failure=recorded,
    )
    atomic_write(manifest_path(row), pretty_json(manifest))
    save_state(
        row,
        TERMINAL_FAILURE_STATUS,
        started_at=started_at,
        completed_at=completed_at,
        call_index=call_index,
        ledger_status=ledger_status,
        error_type=manifest["failure"]["error_type"],
        error=manifest["failure"]["message"],
    )


async def record_blocked_replay(row: dict, ledger: Ledger) -> bool:
    """Record a replay run terminal when its neutral core source terminally failed.

    The fixed search context for a replay row is written only by a successful
    neutral core run of the same claim and system, so a terminally failed
    source leaves the replay row permanently unrunnable.
    """
    if not uses_fixed_evidence(row) or replay_context_path(row).exists():
        return False
    if not replay_source_terminal(row):
        return False
    if await ledger.get_call(row["run_id"], 1) is not None:
        raise RuntimeError(
            f"{row['run_id']}: blocked replay has ledger history and "
            "requires reconciliation before a terminal record"
        )
    source_id = replay_source_run_id(row)
    error = RuntimeError(
        f"replay source {source_id} terminally failed; the fixed search "
        "context can never exist"
    )
    timestamp = now_iso()
    events = load_existing_events(raw_path(row))
    compatible = [event for event in events if is_blocked_replay_failure(event)]
    if compatible:
        recorded = compatible[-1]
    else:
        recorded = failure_event(
            row,
            timestamp=timestamp,
            stage=BLOCKED_REPLAY_STAGE,
            error=error,
            call_index=0,
            ledger_status="not_reserved",
        )
        append_jsonl(raw_path(row), recorded)
        events.append(recorded)
    _write_terminal_record(
        row,
        events=events,
        recorded=recorded,
        error=error,
        call_index=0,
        ledger_status="not_reserved",
        started_at=timestamp,
        completed_at=timestamp,
        stage=BLOCKED_REPLAY_STAGE,
    )
    return True


async def record_terminal_failure(
    row: dict,
    ledger: Ledger,
    error: Exception,
) -> bool:
    state: dict[str, Any] = {}
    path = state_path(row)
    if path.exists():
        loaded = load_json(path)
        if not isinstance(loaded, dict):
            raise RuntimeError(f"state is not an object: {path.name}")
        state = loaded
    started_at = str(state.get("started_at") or now_iso())
    stage = str(state.get("status") or "unrecorded")
    call_index = state.get("call_index")
    if not isinstance(call_index, int) or call_index < 1:
        call_index = 0

    events = load_existing_events(raw_path(row))
    ledger_row = await ledger.get_call(row["run_id"], call_index) if call_index else None
    ledger_status = str(ledger_row.get("status") or "") if ledger_row else "not_reserved"
    accepted = accepted_terminal_failure(events, call_index, ledger_row)
    completed_at = now_iso()
    if not accepted:
        save_state(
            row,
            "failure_unresolved",
            started_at=started_at,
            completed_at=completed_at,
            call_index=call_index,
            ledger_status=ledger_status,
            error_type=type(error).__name__,
            error=str(error),
        )
        return False

    compatible = [
        event
        for event in events
        if event.get("event") == TERMINAL_FAILURE_EVENT
        and event.get("call_index") == call_index
        and event.get("ledger_status") == ledger_status
    ]
    if compatible:
        recorded = compatible[-1]
    else:
        recorded = failure_event(
            row,
            timestamp=completed_at,
            stage=stage,
            error=error,
            call_index=call_index,
            ledger_status=ledger_status,
        )
        append_jsonl(raw_path(row), recorded)
        events.append(recorded)

    _write_terminal_record(
        row,
        events=events,
        recorded=recorded,
        error=error,
        call_index=call_index,
        ledger_status=ledger_status,
        started_at=started_at,
        completed_at=completed_at,
        stage=str(recorded.get("stage") or stage),
    )
    return True


async def run_one(
    row: dict,
    claims: dict[str, dict],
    prompts: dict,
    client: OpenRouterClient,
    ledger: Ledger,
    search: SearchTool,
    semaphore: asyncio.Semaphore,
    include_kimi: bool,
) -> str:
    try:
        return await _run_one_impl(
            row,
            claims,
            prompts,
            client,
            ledger,
            search,
            semaphore,
            include_kimi,
        )
    except RecoveryInvariantError:
        raise
    except Exception as error:
        if manifest_path(row).exists():
            raise
        if await record_terminal_failure(row, ledger, error):
            return TERMINAL_FAILURE_STATUS
        raise


def selected_schedule(
    arms: set[str],
    include_kimi: bool,
    limit: int | None,
    claim_ids: set[str] | None = None,
) -> list[dict]:
    if SCHEDULE_PATH.exists():
        rows = load_json(SCHEDULE_PATH)
    else:
        rows = build_schedule(load_claims(), include_kimi)
    if not isinstance(rows, list):
        raise RuntimeError("schedule is not a list")
    rows = list(rows) + load_phase2_schedule()
    selected = [
        row
        for row in rows
        if row["arm"] in arms and (claim_ids is None or row["claim_id"] in claim_ids)
    ]
    pending = []
    for row in selected:
        path = manifest_path(row)
        if not path.exists():
            pending.append(row)
            continue
        manifest = load_json(path)
        if not isinstance(manifest, dict):
            raise RuntimeError(f"manifest is not an object: {path.name}")
        if manifest.get("run_id") != row["run_id"]:
            raise RuntimeError(f"manifest run ID mismatch: {path.name}")
        if manifest.get("status") not in {
            "agent_complete",
            "judged",
            "analysis_ready",
            TERMINAL_FAILURE_STATUS,
        }:
            pending.append(row)
    selected = pending
    if limit is not None:
        selected = limit_complete_claims(selected, limit)
    selected_ids = {row["run_id"] for row in selected}
    for row in selected:
        if (
            uses_fixed_evidence(row)
            and not replay_context_path(row).exists()
            and replay_source_run_id(row) not in selected_ids
            and not replay_source_terminal(row)
        ):
            raise RuntimeError(
                f"replay run {row['run_id']} is blocked: missing neutral core context "
                f"from {replay_source_run_id(row)}"
            )
    return selected


def observed_attempted_run_ids(
    db_path: Path = DB_PATH,
    runs_dir: Path = RUNS_DIR,
    raw_dir: Path = RAW_DIR,
) -> set[str]:
    run_ids: set[str] = set()
    if db_path.exists():
        try:
            with sqlite3.connect(db_path) as connection:
                run_ids.update(
                    str(row[0])
                    for row in connection.execute(
                        "SELECT DISTINCT run_id FROM calls WHERE run_id != ''"
                    )
                )
        except sqlite3.Error as error:
            raise RuntimeError(f"cannot inspect attempted run IDs: {error}") from error
    if runs_dir.exists():
        run_ids.update(path.stem for path in runs_dir.glob("*.json"))
    if raw_dir.exists():
        run_ids.update(path.stem for path in raw_dir.glob("*/*.jsonl"))
    return run_ids


def authenticated_trim_state(
    schedule_rows: list[dict],
    claims: list[dict],
    attempted_run_ids: set[str],
    record_dir: Path = TRIM_RECORD_DIR,
) -> tuple[dict, list[dict]]:
    try:
        records = load_trim_records(record_dir)
        state = verify_trim_records(schedule_rows, claims, record_dir)
    except (OSError, TypeError, ValueError) as error:
        raise RuntimeError(f"authenticated schedule trim is invalid: {error}") from error
    removed_ids = {
        str(run_id)
        for record in records
        for run_id in record.get("removed_run_ids", [])
    }
    attempted_removed = sorted(removed_ids & attempted_run_ids)
    if attempted_removed:
        raise RuntimeError(
            "authenticated schedule trim removed already-attempted run IDs: "
            f"{attempted_removed[:8]}"
        )
    return state, records


def validate_full_launch_decision(
    schedule_rows: list[dict],
    include_kimi: bool,
    evidence: dict,
    *,
    claims: list[dict] | None = None,
    attempted_run_ids: set[str] | None = None,
    trim_record_dir: Path | None = None,
) -> None:
    if not schedule_rows:
        raise RuntimeError("full launch requires a non-empty frozen schedule")
    if not isinstance(evidence, dict):
        raise RuntimeError("pilot-gate evidence is not an object")
    claims = claims if claims is not None else load_claims()
    attempted_run_ids = (
        attempted_run_ids
        if attempted_run_ids is not None
        else observed_attempted_run_ids()
    )
    state, records = authenticated_trim_state(
        schedule_rows,
        claims,
        attempted_run_ids,
        trim_record_dir or TRIM_RECORD_DIR,
    )
    current_profile = state["profile"]
    current_include_kimi = state["include_kimi"]
    current_hash = state["schedule_sha256"]
    if current_profile not in TRIM_PROFILES:
        raise RuntimeError("authenticated trim state has an unknown profile")
    if include_kimi is not current_include_kimi:
        raise RuntimeError(
            "--include-kimi does not match the current authenticated schedule"
        )

    decision_include_kimi = evidence.get("decision_include_kimi")
    initial_profile = evidence.get("initial_trim_profile")
    initial_hash = evidence.get("initial_full_schedule_sha256")
    if not isinstance(decision_include_kimi, bool):
        raise RuntimeError("pilot gate lacks the frozen Kimi decision")
    expected_initial_profile = (
        TRIM_PROFILE_STEP_1 if decision_include_kimi else TRIM_PROFILE_NONE
    )
    if initial_profile != expected_initial_profile:
        raise RuntimeError("pilot gate has an invalid initial trim profile")
    expected_initial_rows = build_schedule(
        claims,
        include_kimi=decision_include_kimi,
        trim_profile=expected_initial_profile,
        pilot_generation=state["pilot_generation"],
    )
    expected_initial_hash = sha256_bytes(
        canonical_json(expected_initial_rows).encode("utf-8")
    )
    if initial_hash != expected_initial_hash:
        raise RuntimeError("pilot gate does not bind the initial decision schedule")

    if decision_include_kimi:
        if not records or records[0].get("target_include_kimi") is not True:
            raise RuntimeError(
                "the Kimi decision requires an authenticated step-one trim"
            )
        if records[0].get("target_schedule_sha256") != expected_initial_hash:
            raise RuntimeError(
                "authenticated step one does not match the Kimi decision schedule"
            )
    elif records:
        if records[0].get("target_include_kimi") is not False:
            raise RuntimeError(
                "authenticated step one conflicts with the pilot decision to drop Kimi"
            )
        if records[0].get("source_schedule_sha256") != expected_initial_hash:
            raise RuntimeError(
                "trim history does not descend from the initial pilot schedule"
            )

    expected_current = {
        "include_kimi": current_include_kimi,
        "trim_profile": current_profile,
        "full_schedule_sha256": current_hash,
        "trim_record_count": state["record_count"],
        "last_trim_record_sha256": state["last_record_sha256"],
    }
    for field, expected in expected_current.items():
        if evidence.get(field) != expected:
            raise RuntimeError(
                f"pilot gate does not bind current authenticated trim field {field}"
            )


def frozen_nonpilot_arms(schedule_rows: list[dict] | None = None) -> set[str]:
    if schedule_rows is None:
        if not SCHEDULE_PATH.exists():
            raise RuntimeError("full launch is blocked: frozen schedule is missing")
        schedule_rows = load_json(SCHEDULE_PATH)
    if not isinstance(schedule_rows, list):
        raise RuntimeError("full launch is blocked: schedule is not a list")
    return {
        str(row.get("arm"))
        for row in schedule_rows
        if row.get("arm") not in ALL_PILOT_ARMS
    }


def require_v5_preflight() -> None:
    preflight_path = DATA / "preflight_v5.json"
    byok_path = ARCHIVE / "byok_verification_v5.json"
    if not preflight_path.exists():
        raise RuntimeError("pilot launch requires a passed v5 preflight record")
    record = load_json(preflight_path)
    if not isinstance(record, dict) or record.get("status") != "passed":
        raise RuntimeError("the v5 preflight record did not pass")
    calls = record.get("calls")
    if not isinstance(calls, list) or len(calls) != 6:
        raise RuntimeError("the v5 preflight record must contain all six calls")
    if not all(call.get("reasoning_present") is True for call in calls):
        raise RuntimeError("a v5 preflight call lacks reasoning evidence")
    if not byok_path.exists():
        raise RuntimeError("pilot launch requires the v5 BYOK verification record")


def enforce_launch_gate(arms: set[str], include_kimi: bool) -> None:
    if arms and arms <= set(PHASE2_ARMS):
        registered = {str(row["arm"]) for row in load_phase2_schedule()}
        missing = sorted(arms - registered)
        if missing:
            raise RuntimeError(
                "phase-2 launch requires registered schedule rows for: "
                + ", ".join(missing)
            )
        from pilot_gate import require_pilot_gate

        require_pilot_gate()
        return
    diagnostic = arms & DIAGNOSTIC_PILOT_ARMS
    if diagnostic:
        raise RuntimeError(
            "pilot-v1/v2/v3 are superseded diagnostic cohorts and cannot be launched: "
            + ", ".join(sorted(diagnostic))
        )
    if arms & PILOT_ARMS:
        if arms != PILOT_ARMS:
            raise RuntimeError(
                "pilot launch/resume requires both active pilot arms: "
                + ", ".join(sorted(PILOT_ARMS))
            )
        if not include_kimi:
            raise RuntimeError(
                "pilot launch/resume requires --include-kimi for the active pilot matrix"
            )
        require_v5_preflight()
        return
    if not SCHEDULE_PATH.exists():
        raise RuntimeError("full launch is blocked: frozen schedule is missing")
    schedule_rows = load_json(SCHEDULE_PATH)
    if not isinstance(schedule_rows, list):
        raise RuntimeError("full launch is blocked: schedule is not a list")
    required_nonpilot_arms = frozen_nonpilot_arms(schedule_rows)
    if arms != required_nonpilot_arms:
        raise RuntimeError(
            "full launch/resume requires the complete frozen nonpilot arm set "
            f"{sorted(required_nonpilot_arms)!r}; received {sorted(arms)!r}"
        )
    from pilot_gate import require_pilot_gate

    evidence = require_pilot_gate()
    validate_full_launch_decision(schedule_rows, include_kimi, evidence)


def limit_complete_claims(rows: list[dict], limit: int) -> list[dict]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(str(row["claim_id"]), []).append(row)

    selected: list[dict] = []
    for claim_rows in grouped.values():
        if selected and len(selected) + len(claim_rows) > limit:
            break
        selected.extend(claim_rows)
        if len(selected) >= limit:
            break
    return selected


async def execute(
    rows: list[dict],
    include_kimi: bool,
    dry_run: bool,
    stop_requested: asyncio.Event | None = None,
) -> None:
    if dry_run:
        counts: dict[str, int] = defaultdict(int)
        for row in rows:
            counts[row["arm"]] += 1
        print(json.dumps({"runs": len(rows), "arms": counts}, sort_keys=True))
        return

    owns_stop_event = stop_requested is None
    if stop_requested is None:
        stop_requested = asyncio.Event()
    loop = asyncio.get_running_loop()
    signal_handler_installed = False
    if owns_stop_event:

        def request_stop() -> None:
            if not stop_requested.is_set():
                print(
                    "SIGTERM received; finishing the current claim batch before exit",
                    flush=True,
                )
                stop_requested.set()

        try:
            loop.add_signal_handler(signal.SIGTERM, request_stop)
            signal_handler_installed = True
        except (NotImplementedError, RuntimeError):
            pass

    claims = {claim["id"]: claim for claim in load_claims()}
    prompts = load_json(PROMPTS_PATH)
    if not isinstance(prompts, dict):
        raise RuntimeError("prompt file is not an object")
    ledger = Ledger()
    client = OpenRouterClient(ledger)
    search = SearchTool()
    semaphores = {
        model: asyncio.Semaphore(MODEL_CONCURRENCY) for model in {row["model"] for row in rows}
    }
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["claim_id"]].append(row)
    completed = 0
    terminal_failed = 0
    provisional_pending = 0
    claim_failures: list[tuple[str, BaseException]] = []
    try:
        for claim_id in sorted(grouped):
            if stop_requested.is_set():
                break
            if len(claim_failures) >= MAX_DEFERRED_CLAIM_FAILURES:
                print(
                    f"{len(claim_failures)} claims failed in one window; "
                    "stopping the window as a systemic failure",
                    flush=True,
                )
                break
            claim_rows = grouped[claim_id]
            first = [row for row in claim_rows if not uses_fixed_evidence(row)]
            replay = [row for row in claim_rows if uses_fixed_evidence(row)]
            for batch in (first, replay):
                if not batch:
                    continue
                results = await asyncio.gather(
                    *(
                        run_one(
                            row,
                            claims,
                            prompts,
                            client,
                            ledger,
                            search,
                            semaphores[row["model"]],
                            include_kimi,
                        )
                        for row in batch
                    ),
                    return_exceptions=True,
                )
                failures = [result for result in results if isinstance(result, BaseException)]
                completed += sum(result == "completed" for result in results)
                terminal_failed += sum(
                    result == TERMINAL_FAILURE_STATUS for result in results
                )
                provisional_pending += sum(
                    result == PROVISIONAL_METADATA_STATUS for result in results
                )
                totals = await ledger.totals()
                print(
                    f"claim={claim_id} new_runs={completed} "
                    f"terminal_failed={terminal_failed} "
                    f"metadata_pending={provisional_pending} calls={totals.calls} "
                    f"openrouter=${totals.openrouter:.4f} reserved=${totals.reservations:.4f}",
                    flush=True,
                )
                if failures:
                    claim_failures.append((claim_id, failures[0]))
                    print(
                        f"claim={claim_id} deferred {len(failures)} run "
                        f"failure(s) to the end of the window: {failures[0]}",
                        flush=True,
                    )
                    break
            if stop_requested.is_set():
                print(f"graceful stop completed after claim={claim_id}", flush=True)
                break
        problems: list[str] = []
        if claim_failures:
            problems.append(
                f"{len(claim_failures)} claim(s) retained run failures; "
                f"first {claim_failures[0][0]}: {claim_failures[0][1]}"
            )
        if provisional_pending:
            if stop_requested.is_set() and not claim_failures:
                print(
                    f"graceful stop retained {provisional_pending} "
                    "metadata-pending response(s) for recovery",
                    flush=True,
                )
            else:
                problems.append(
                    f"window retained {provisional_pending} metadata-pending "
                    "response(s); generation metadata recovery is required "
                    "before the next window"
                )
        if problems:
            raise RuntimeError("; ".join(problems)) from (
                claim_failures[0][1] if claim_failures else None
            )
    finally:
        await search.close()
        await client.close()
        if signal_handler_installed:
            loop.remove_signal_handler(signal.SIGTERM)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arms", nargs="+", required=True)
    parser.add_argument("--include-kimi", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--claim-ids", nargs="+")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    print(f"runner_pid={os.getpid()}", flush=True)
    arms = set(args.arms)
    enforce_launch_gate(arms, args.include_kimi)
    rows = selected_schedule(
        arms,
        args.include_kimi,
        args.limit,
        set(args.claim_ids) if args.claim_ids else None,
    )
    if not rows:
        print("no pending scheduled runs match the requested filters", flush=True)
        return
    asyncio.run(execute(rows, args.include_kimi, args.dry_run))


if __name__ == "__main__":
    main()
