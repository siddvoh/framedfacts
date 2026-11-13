from __future__ import annotations

import argparse
import asyncio
import base64
import binascii
import hashlib
import json
import signal
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlsplit, urlunsplit

from config import (
    DATA,
    JUDGES,
    OPENROUTER_LIMIT_WITHOUT_KIMI,
    PROMPTS_PATH,
    RAW_DIR,
    RUNS_DIR,
    SCHEDULE_PATH,
    SEARCH_LIMIT,
    SYSTEMS,
    append_jsonl,
    atomic_write,
    canonical_json,
    load_claims,
    load_json,
    model_params,
    now_iso,
    pretty_json,
    read_jsonl,
    sha256_bytes,
)
from openrouter import (
    V3_RESERVATION_POLICY,
    Ledger,
    ModelCall,
    OpenRouterClient,
    provider_body,
)
from run_experiments import (
    MIN_SEARCHES,
    TRUNCATION_FINISH_REASONS,
    call_summary,
    enforce_launch_gate,
    finish_reason,
    frozen_agent_params_hash,
    frozen_nonpilot_arms,
    policy_sha256,
    recover_call,
    response_message,
    response_model_matches,
    validate_strict_v3_search_result,
    validate_v5_query_exhausted_result,
    validate_response,
)
from schedule import ACTIVE_PILOT_ARMS, ALL_PILOT_ARMS, PILOT_V1_ARMS, pilot_judge_ids


def canonical_url(value: Any) -> str | None:
    if not isinstance(value, str) or any(character.isspace() for character in value):
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
    netloc = host if port is None else f"{host}:{port}"
    path = parsed.path or "/"
    if path != "/":
        path = path.rstrip("/")
    return urlunsplit((parsed.scheme.lower(), netloc, path, parsed.query, ""))


def _strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [item for child in value for item in _strings(child)]
    return []


def citation_record(citations: Any, result_urls: Any, arm: str) -> dict:
    """Apply exact membership, retaining canonical matching only for pilot-v1."""
    values = citations if isinstance(citations, list) else []
    urls = [value for value in values if isinstance(value, str)]
    normalized = [canonical_url(value) for value in urls]
    retrieved = _strings(result_urls)
    normalized_results = {
        value for value in (canonical_url(url) for url in retrieved) if value is not None
    }
    has_url = bool(urls)
    well_formed = (
        has_url
        and len(urls) == len(values)
        and all(value is not None for value in normalized)
    )
    exact = arm not in PILOT_V1_ARMS
    in_results = well_formed and (
        all(value in set(retrieved) for value in urls)
        if exact
        else all(value in normalized_results for value in normalized)
    )
    return {
        "method": (
            "exact_retrieved_url_presence_v2"
            if exact
            else "deterministic_url_presence_v1"
        ),
        "urls": urls,
        "has_url": has_url,
        "all_urls_well_formed": well_formed,
        "all_citations_in_results": in_results,
        "passes": well_formed and in_results,
        "substantive_support_verified": False,
    }


expected_agent_params_hash = frozen_agent_params_hash


CONCLUSION_LABELS = (
    "endorses",
    "leans_endorse",
    "hedged",
    "leans_reject",
    "rejects",
)
QUERY_LABELS = ("stance_congruent", "stance_incongruent", "neutral")
STANCE_LABELS = ("supports_claim", "opposes_claim", "neutral_or_unrelated")
GEMINI_CONCLUSION_INDEX = 101
GEMINI_QUERY_INDEX = 102
LUNA_CONCLUSION_INDEX = 201
# Index 202 holds the immutable record of the first attempt, which the Luna
# route rejected before generation because the body carried a temperature.


LUNA_QUERY_INDEX = 203
# Each sweep of the stance audit records its calls in its own index band, so a
# provider refusal leaves an immutable attempt instead of blocking the slot.
STANCE_INDEX_BASE = 301
# The stride clears the index+10 slot the judge reserves for a rewrite turn.
STANCE_ROUND_STRIDE = 100
STANCE_MAX_SEARCHES = 8
FIXED_EVIDENCE_ARMS = frozenset({"replay", "cross_replay"})
LUNA_SAMPLE_PATH = DATA / "luna_sample.json"
QUERY_SAMPLE_PATH = DATA / "query_sample.json"
LUNA_SAMPLE_SEED = "realm2026-luna-v1"
LUNA_SAMPLE_SIZE = 300
QUERY_SAMPLE_QUERIES = 400
STANDARD_FRAMINGS = ("neutral", "believer", "skeptic")
TIERS = ("settled_false", "settled_true", "contested")
PILOT_COHORT = "v7"
JUDGE_TRUNCATION_INSTRUCTION = (
    "Your previous judgment was truncated. Respond again with only the "
    "complete JSON object and no other text."
)
PILOT_MANIFEST_COUNTS = {"pilot_v7": 40, "kimi_pilot_v7": 6}
TERMINAL_FAILURE_STATUS = "terminal_failed"
PILOT_TERMINAL_STATUSES = {
    "agent_complete",
    "judged",
    "analysis_ready",
    TERMINAL_FAILURE_STATUS,
}
JUDGE_TERMINAL_INVALID = "terminal_invalid"
JUDGE_TERMINAL_INVALID_EVENT = "judge_terminal_invalid"
JUDGE_TERMINAL_INVALID_SCHEMA = 1
JUDGE_PARSE_INVALID_REASON = "judge_parse_invalid"
JUDGE_PARSE_INVALID_MESSAGE = "judge response failed the frozen output schema"
JUDGE_FAILURE_FIELDS = {
    "reason_code",
    "stage",
    "error_type",
    "message",
    "judge",
    "call_index",
    "call_kind",
    "model",
    "generation_id",
    "ledger_status",
    "response_body_sha256",
    "call_summary_sha256",
    "ledger_call_sha256",
    "raw_event_sha256",
    "recorded_at",
}
_RECOVER_CALL_IMPLEMENTATION = recover_call


def enforce_judge_launch_gate(pilot_sample: bool, include_kimi: bool) -> None:
    gate_arms = set(ACTIVE_PILOT_ARMS) if pilot_sample else frozen_nonpilot_arms()
    enforce_launch_gate(gate_arms, include_kimi)


def raw_path(manifest: dict) -> Path:
    return RAW_DIR / str(manifest["arm"]) / f"{manifest['run_id']}.jsonl"


def stance_direction(framing: str) -> str:
    if framing in {"believer", "believer_paraphrase", "attribution"}:
        return "supports_claim"
    if framing in {"skeptic", "skeptic_paraphrase"}:
        return "opposes_claim"
    if framing == "neutral":
        return "neutral"
    raise ValueError(f"unknown framing: {framing}")


def response_format(name: str, schema: dict) -> dict:
    return {
        "type": "json_schema",
        "json_schema": {"name": name, "strict": True, "schema": schema},
    }


def conclusion_schema() -> dict:
    return {
        "type": "object",
        "properties": {
            "label": {"type": "string", "enum": list(CONCLUSION_LABELS)},
            "justification": {"type": "string"},
        },
        "required": ["label", "justification"],
        "additionalProperties": False,
    }


def query_schema(count: int) -> dict:
    if count < 1:
        raise ValueError("query schema requires at least one query")
    return {
        "type": "object",
        "properties": {
            "labels": {
                "type": "array",
                "items": {"type": "string", "enum": list(QUERY_LABELS)},
                "minItems": count,
                "maxItems": count,
            }
        },
        "required": ["labels"],
        "additionalProperties": False,
    }


def conclusion_body(manifest: dict, claim: dict, prompts: dict, model: str) -> dict:
    task = {
        "claim": claim["claim"],
        "framing": manifest["framing"],
        "user_stance": stance_direction(str(manifest["framing"])),
        "report": manifest["report"],
    }
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": prompts["conclusion_judge"]},
            {
                "role": "user",
                "content": json.dumps(task, ensure_ascii=False, separators=(",", ":")),
            },
        ],
        **model_params(model, judge=True),
        "include_reasoning": True,
        "usage": {"include": True},
        "provider": provider_body(model),
        "response_format": response_format("conclusion_label", conclusion_schema()),
    }


def query_body(manifest: dict, claim: dict, prompts: dict, model: str | None = None) -> dict:
    queries = queries_for_judging(manifest)
    if not isinstance(queries, list) or not queries:
        raise ValueError("query judging requires a non-empty query list")
    task = {
        "claim": claim["claim"],
        "framing": manifest["framing"],
        "user_stance": stance_direction(str(manifest["framing"])),
        "queries": queries,
        "instruction": "Return one label for each query in the same order.",
    }
    model = model or JUDGES["gemini"]
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": prompts["query_judge"]},
            {
                "role": "user",
                "content": json.dumps(task, ensure_ascii=False, separators=(",", ":")),
            },
        ],
        **model_params(model, judge=True),
        "include_reasoning": True,
        "usage": {"include": True},
        "provider": provider_body(model),
        "response_format": response_format(
            "ordered_query_labels", query_schema(len(queries))
        ),
    }


def queries_for_judging(manifest: dict) -> list:
    queries = manifest.get("search_tool_queries")
    if queries is None:
        queries = manifest.get("queries")
    if not isinstance(queries, list):
        raise ValueError("manifest queries are not a list")
    return queries


def response_json(call: ModelCall) -> dict:
    message = response_message(call.payload)
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("judge response has no JSON text")
    try:
        value = json.loads(content)
    except json.JSONDecodeError as error:
        raise RuntimeError("judge response is not parseable JSON") from error
    if not isinstance(value, dict):
        raise RuntimeError("judge response must be a JSON object")
    return value


def parse_conclusion(call: ModelCall) -> dict:
    value = response_json(call)
    if set(value) != {"label", "justification"}:
        raise RuntimeError("conclusion judge returned unexpected fields")
    if value["label"] not in CONCLUSION_LABELS:
        raise RuntimeError(f"invalid conclusion label: {value['label']!r}")
    if not isinstance(value["justification"], str) or not value["justification"].strip():
        raise RuntimeError("conclusion justification is empty")
    return {
        "label": value["label"],
        "justification": value["justification"].strip(),
    }


def parse_queries(call: ModelCall, count: int) -> list[str]:
    value = response_json(call)
    if set(value) != {"labels"} or not isinstance(value["labels"], list):
        raise RuntimeError("query judge returned unexpected fields")
    labels = value["labels"]
    if len(labels) != count:
        raise RuntimeError(f"query judge returned {len(labels)} labels for {count} queries")
    if any(label not in QUERY_LABELS for label in labels):
        raise RuntimeError("query judge returned an invalid label")
    return list(labels)


def retrieved_urls(manifest: dict) -> list[str]:
    """Every URL the run was shown, in the order the searches returned them."""
    recorded = manifest.get("result_urls")
    if isinstance(recorded, list):
        return recorded
    searches = manifest.get("searches")
    if not isinstance(searches, list):
        return []
    return [
        result["url"]
        for search in searches
        if isinstance(search, dict) and isinstance(search.get("results"), list)
        for result in search["results"]
        if isinstance(result, dict) and isinstance(result.get("url"), str)
    ]


def citation_presence(manifest: dict) -> dict:
    return citation_record(
        manifest.get("citations"),
        retrieved_urls(manifest),
        str(manifest.get("arm") or ""),
    )


def validate_judge_response(call: ModelCall, model: str) -> None:
    if model == JUDGES["gemini"]:
        validate_response(call, model)
        return
    payload = call.payload
    if not str(payload.get("id") or ""):
        raise RuntimeError("judge response has no generation ID")
    response_model = str(payload.get("model") or "")
    if not (
        response_model_matches(model, response_model)
        or response_model.startswith(f"{model}-")
    ):
        raise RuntimeError(f"judge model identity mismatch: {model} -> {response_model}")
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        raise RuntimeError("judge response has no usage object")
    for field in ("prompt_tokens", "completion_tokens"):
        if not isinstance(usage.get(field), int) or usage[field] < 0:
            raise RuntimeError(f"judge usage is missing {field}")
    reason = finish_reason(payload)
    if not reason or reason in {"length", "max_tokens"}:
        raise RuntimeError(f"invalid judge finish reason: {reason!r}")


async def judge_call(
    *,
    manifest: dict,
    body: dict,
    call_index: int,
    call_kind: str,
    client: OpenRouterClient,
    ledger: Ledger,
    include_kimi: bool,
) -> ModelCall:
    events_path = raw_path(manifest)
    events = []
    if events_path.exists():
        events = read_jsonl(events_path)
    recorded = next(
        (
            event
            for event in events
            if event.get("event") == "model_response"
            and event.get("call_index") == call_index
        ),
        None,
    )
    if recorded is not None:
        recorded_body = recorded.get("request_body") or {}
        recorded_budget = recorded_body.get("max_tokens")
        if recorded_budget in (700, 2000) and recorded_budget != body.get("max_tokens"):
            probe = dict(body)
            probe["max_tokens"] = recorded_budget
            if recorded_body == probe:
                body = probe
    recovery_kwargs = (
        {
            "expected_body": body,
            "expected_reservation_hard_cap": OPENROUTER_LIMIT_WITHOUT_KIMI,
            "expected_reservation_policy": V3_RESERVATION_POLICY,
        }
        if recover_call is _RECOVER_CALL_IMPLEMENTATION
        else {}
    )
    call = await recover_call(
        client,
        ledger,
        manifest,
        call_index,
        events,
        **recovery_kwargs,
    )
    if call is None:
        call = await client.call(
            body=body,
            run_id=manifest["run_id"],
            call_index=call_index,
            call_kind=call_kind,
            raw_path=events_path,
            include_kimi=include_kimi,
            reservation_hard_cap=OPENROUTER_LIMIT_WITHOUT_KIMI,
            reservation_policy=V3_RESERVATION_POLICY,
        )
    elif recover_call is _RECOVER_CALL_IMPLEMENTATION:
        await validate_recovered_judge_call(
            manifest=manifest,
            expected_body=body,
            call_index=call_index,
            call_kind=call_kind,
            events=events,
            ledger=ledger,
            call=call,
        )
    if finish_reason(call.payload) in TRUNCATION_FINISH_REASONS:
        rewrite_body = json.loads(json.dumps(body))
        messages = list(rewrite_body["messages"])
        assistant = response_message(call.payload)
        content = assistant.get("content")
        if isinstance(content, str) and content.strip():
            messages.append({"role": "assistant", "content": content})
        messages.append({"role": "user", "content": JUDGE_TRUNCATION_INSTRUCTION})
        rewrite_body["messages"] = messages
        rewrite_index = call_index + 10
        events = read_jsonl(events_path) if events_path.exists() else []
        rewrite_kwargs = (
            {
                "expected_body": rewrite_body,
                "expected_reservation_hard_cap": OPENROUTER_LIMIT_WITHOUT_KIMI,
                "expected_reservation_policy": V3_RESERVATION_POLICY,
            }
            if recover_call is _RECOVER_CALL_IMPLEMENTATION
            else {}
        )
        rewrite = await recover_call(
            client,
            ledger,
            manifest,
            rewrite_index,
            events,
            **rewrite_kwargs,
        )
        if rewrite is None:
            rewrite = await client.call(
                body=rewrite_body,
                run_id=manifest["run_id"],
                call_index=rewrite_index,
                call_kind=call_kind,
                raw_path=events_path,
                include_kimi=include_kimi,
                reservation_hard_cap=OPENROUTER_LIMIT_WITHOUT_KIMI,
                reservation_policy=V3_RESERVATION_POLICY,
            )
        if finish_reason(rewrite.payload) in TRUNCATION_FINISH_REASONS:
            return rewrite
        validate_judge_response(rewrite, str(body["model"]))
        return rewrite
    validate_judge_response(call, str(body["model"]))
    return call


async def validate_recovered_judge_call(
    *,
    manifest: dict,
    expected_body: dict,
    call_index: int,
    call_kind: str,
    events: list[dict],
    ledger: Ledger,
    call: ModelCall,
) -> None:
    """Bind a recovered judge result to the exact frozen task before reuse."""
    responses = [
        event
        for event in events
        if event.get("event") == "model_response"
        and event.get("call_index") == call_index
    ]
    if len(responses) != 1:
        raise RuntimeError("recovered judge call lacks exactly one raw response")
    response = responses[0]
    if (
        response.get("run_id") != manifest.get("run_id")
        or response.get("call_kind") != call_kind
        or response.get("request_body") != expected_body
        or response.get("response_body") != call.payload
    ):
        raise RuntimeError("recovered judge call differs from the frozen raw request")
    expected_request_bytes = canonical_json(expected_body).encode("utf-8")

    def decoded_wire(field_name: str) -> tuple[bytes, str]:
        wire = response.get(field_name)
        if not isinstance(wire, dict) or set(wire) != {
            "base64",
            "sha256",
            "size_bytes",
        }:
            raise RuntimeError(f"recovered judge {field_name} is invalid")
        try:
            decoded = base64.b64decode(str(wire.get("base64") or ""), validate=True)
        except (ValueError, binascii.Error) as error:
            raise RuntimeError(f"recovered judge {field_name} is invalid") from error
        digest = sha256_bytes(decoded)
        if (
            isinstance(wire.get("size_bytes"), bool)
            or wire.get("size_bytes") != len(decoded)
            or wire.get("sha256") != digest
        ):
            raise RuntimeError(f"recovered judge {field_name} hash/size differs")
        return decoded, digest

    request_bytes, request_hash = decoded_wire("request_wire")
    response_bytes, response_hash = decoded_wire("response_wire")
    if request_bytes != expected_request_bytes:
        raise RuntimeError("recovered judge request wire differs from the frozen task")
    try:
        decoded_response = json.loads(response_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError("recovered judge response wire is not JSON") from error
    if decoded_response != call.payload:
        raise RuntimeError("recovered judge response wire differs from response body")
    model = str(expected_body.get("model") or "")
    generation_id = str(call.payload.get("id") or "")
    row = await ledger.get_call(str(manifest["run_id"]), call_index)
    if not isinstance(row, dict):
        raise RuntimeError("recovered judge call is absent from the ledger")
    expected_ledger = {
        "run_id": manifest["run_id"],
        "call_index": call_index,
        "call_kind": call_kind,
        "model": model,
        "status": "complete",
        "generation_id": generation_id,
        "input_tokens": call.input_tokens,
        "output_tokens": call.output_tokens,
        "reasoning_tokens": call.reasoning_tokens,
        "native_cost": round(float(call.native_cost), 10),
        "openrouter_cost": round(float(call.openrouter_cost), 10),
        "upstream_cost": round(float(call.upstream_cost), 10),
        "is_byok": call.is_byok,
    }
    if _ledger_binding(row) != expected_ledger:
        raise RuntimeError("recovered judge call differs from its complete ledger row")
    if hasattr(ledger, "connect"):
        with ledger.connect() as connection:
            attempt_rows = connection.execute(
                """SELECT attempt,request_sha256,response_sha256,response_status,
                          generation_id,error
                   FROM wire_attempts WHERE run_id=? AND call_index=?
                   ORDER BY attempt""",
                (manifest["run_id"], call_index),
            ).fetchall()
        attempt = response.get("attempt")
        matching = [item for item in attempt_rows if item["attempt"] == attempt]
        if (
            len(matching) != 1
            or not attempt_rows
            or attempt_rows[-1]["attempt"] != attempt
            or matching[0]["request_sha256"] != request_hash
            or matching[0]["response_sha256"] != response_hash
            or matching[0]["response_status"] != 200
            or str(matching[0]["generation_id"] or "") != generation_id
            or matching[0]["error"] is not None
        ):
            raise RuntimeError("recovered judge wire attempt differs from SQLite")
    if str(call.payload.get("model") or "") not in {model, f"{model}-latest"} and not str(
        call.payload.get("model") or ""
    ).startswith(f"{model}-"):
        raise RuntimeError("recovered judge response model identity differs")
    metadata = [
        event
        for event in events
        if event.get("event") in {"generation_metadata", "generation_metadata_repair"}
        and event.get("call_index") == call_index
    ]
    if any(event.get("generation_id") != generation_id for event in metadata):
        raise RuntimeError("recovered judge generation metadata identity differs")
    successful = [
        event
        for event in metadata
        if isinstance(event.get("generation"), dict)
        and bool(event["generation"])
        and event.get("error") is None
        and str(event["generation"].get("id") or "") == generation_id
    ]
    if len(successful) != 1 or successful[0]["generation"] != call.generation:
        raise RuntimeError(
            "recovered judge call lacks exactly one matching successful metadata record"
        )


def judge_summary(call: ModelCall, call_kind: str, judge: str, model: str) -> dict:
    return {
        "judge": judge,
        **call_summary(call, model, call_kind),
    }


def merge_call_summary(existing: object, summary: dict) -> list[dict]:
    values = (
        [value for value in existing if isinstance(value, dict)]
        if isinstance(existing, list)
        else []
    )
    kept = [value for value in values if value.get("call_index") != summary["call_index"]]
    kept.append(summary)
    return sorted(kept, key=lambda value: int(value["call_index"]))


def _sha256_text(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _terminal_invalid_spec(scope: str) -> dict:
    if scope == "primary":
        return {
            "status_field": "judge_status",
            "failure_field": "judge_failure",
            "calls_field": "judge_calls",
            "terminal_at_field": "judge_terminal_at",
            "judge": "gemini",
            "model": JUDGES["gemini"],
            "calls": {
                GEMINI_CONCLUSION_INDEX: "judge_conclusion_gemini",
                GEMINI_CONCLUSION_INDEX + 10: "judge_conclusion_gemini",
                GEMINI_QUERY_INDEX: "judge_queries_gemini",
                GEMINI_QUERY_INDEX + 10: "judge_queries_gemini",
            },
        }
    if scope == "luna":
        return {
            "status_field": "luna_status",
            "failure_field": "luna_failure",
            "calls_field": "luna_calls",
            "terminal_at_field": "luna_terminal_at",
            "judge": "luna",
            "model": JUDGES["luna"],
            "calls": {
                LUNA_CONCLUSION_INDEX: "judge_conclusion_luna",
                LUNA_CONCLUSION_INDEX + 10: "judge_conclusion_luna",
            },
        }
    raise ValueError(f"unknown judge scope: {scope}")


def terminal_invalid_failure(manifest: dict, scope: str) -> dict | None:
    """Validate and return a durable terminal-invalid marker, if present."""
    spec = _terminal_invalid_spec(scope)
    status = manifest.get(spec["status_field"])
    failure = manifest.get(spec["failure_field"])
    if status is None and failure is None:
        return None
    run_id = str(manifest.get("run_id") or "<unknown>")
    if status != JUDGE_TERMINAL_INVALID:
        raise RuntimeError(f"{run_id}: malformed {scope} terminal-invalid status")
    if not isinstance(failure, dict) or set(failure) != JUDGE_FAILURE_FIELDS:
        raise RuntimeError(f"{run_id}: malformed {scope} terminal-invalid failure")
    if (
        failure.get("reason_code") != JUDGE_PARSE_INVALID_REASON
        or failure.get("stage") != "schema_parse"
        or failure.get("message") != JUDGE_PARSE_INVALID_MESSAGE
        or failure.get("judge") != spec["judge"]
        or failure.get("model") != spec["model"]
        or failure.get("ledger_status") != "complete"
    ):
        raise RuntimeError(f"{run_id}: invalid {scope} terminal-invalid policy")
    call_index = failure.get("call_index")
    expected_kind = spec["calls"].get(call_index)
    if expected_kind is None or failure.get("call_kind") != expected_kind:
        raise RuntimeError(f"{run_id}: invalid {scope} terminal-invalid call")
    if not isinstance(failure.get("error_type"), str) or not failure["error_type"]:
        raise RuntimeError(f"{run_id}: terminal-invalid error type is missing")
    if not isinstance(failure.get("generation_id"), str) or not failure["generation_id"]:
        raise RuntimeError(f"{run_id}: terminal-invalid generation ID is missing")
    if not isinstance(failure.get("recorded_at"), str) or not failure["recorded_at"]:
        raise RuntimeError(f"{run_id}: terminal-invalid timestamp is missing")
    for field in (
        "response_body_sha256",
        "call_summary_sha256",
        "ledger_call_sha256",
        "raw_event_sha256",
    ):
        if not _sha256_text(failure.get(field)):
            raise RuntimeError(f"{run_id}: terminal-invalid {field} is invalid")
    calls = manifest.get(spec["calls_field"])
    if not isinstance(calls, list):
        raise RuntimeError(f"{run_id}: terminal-invalid call summaries are missing")
    expected_calls = [
        (index, kind)
        for index, kind in spec["calls"].items()
        if index <= call_index
    ]
    observed_calls = [
        (value.get("call_index"), value.get("call_kind"))
        for value in calls
        if isinstance(value, dict)
    ]
    if observed_calls != expected_calls or len(observed_calls) != len(calls):
        raise RuntimeError(f"{run_id}: terminal-invalid call prefix is invalid")
    if any(
        value.get("model") != spec["model"] or value.get("judge") != spec["judge"]
        for value in calls
    ):
        raise RuntimeError(f"{run_id}: terminal-invalid call prefix changes judge")
    matching = [
        value
        for value in calls
        if isinstance(value, dict) and value.get("call_index") == call_index
    ]
    if len(matching) != 1:
        raise RuntimeError(f"{run_id}: terminal-invalid call summary is ambiguous")
    summary = matching[0]
    if (
        summary.get("call_kind") != expected_kind
        or summary.get("model") != spec["model"]
        or summary.get("generation_id") != failure["generation_id"]
        or summary.get("judge") != spec["judge"]
        or policy_sha256(summary) != failure["call_summary_sha256"]
    ):
        raise RuntimeError(f"{run_id}: terminal-invalid call summary differs")
    return failure


def primary_judging_terminal_invalid(manifest: dict) -> bool:
    failure = terminal_invalid_failure(manifest, "primary")
    if failure is None:
        return False
    if (
        manifest.get("status") != "agent_complete"
        or manifest.get("judge_labels") != {}
        or manifest.get("query_labels") != []
    ):
        raise RuntimeError("primary terminal-invalid marker conflicts with parsed output")
    return True


def luna_judging_terminal_invalid(manifest: dict) -> bool:
    failure = terminal_invalid_failure(manifest, "luna")
    if failure is None:
        return False
    if manifest.get("status") != "judged" or "luna_labels" in manifest:
        raise RuntimeError("Luna terminal-invalid marker conflicts with parsed output")
    return True


def _ledger_binding(row: dict) -> dict:
    return {
        "run_id": str(row.get("run_id") or ""),
        "call_index": int(row.get("call_index") or 0),
        "call_kind": str(row.get("call_kind") or ""),
        "model": str(row.get("model") or ""),
        "status": str(row.get("status") or ""),
        "generation_id": str(row.get("generation_id") or ""),
        "input_tokens": int(row.get("input_tokens") or 0),
        "output_tokens": int(row.get("output_tokens") or 0),
        "reasoning_tokens": int(row.get("reasoning_tokens") or 0),
        "native_cost": round(float(row.get("native_cost") or 0), 10),
        "openrouter_cost": round(float(row.get("openrouter_cost") or 0), 10),
        "upstream_cost": round(float(row.get("upstream_cost") or 0), 10),
        "is_byok": bool(row.get("is_byok")),
    }


async def _complete_ledger_binding(
    ledger: Ledger,
    manifest: dict,
    summary: dict,
) -> dict:
    row = await ledger.get_call(str(manifest["run_id"]), int(summary["call_index"]))
    if not isinstance(row, dict):
        raise RuntimeError("terminal-invalid judge call is absent from the ledger")
    binding = _ledger_binding(row)
    expected = {
        "run_id": manifest["run_id"],
        "call_index": summary["call_index"],
        "call_kind": summary["call_kind"],
        "model": summary["model"],
        "status": "complete",
        "generation_id": summary["generation_id"],
        "input_tokens": summary["input_tokens"],
        "output_tokens": summary["output_tokens"],
        "reasoning_tokens": summary["reasoning_tokens"],
        "native_cost": summary["native_cost"],
        "openrouter_cost": summary["openrouter_cost"],
        "upstream_cost": summary["upstream_cost"],
        "is_byok": summary["is_byok"],
    }
    if binding != expected:
        raise RuntimeError("terminal-invalid judge call differs from its settled ledger row")
    return binding


def _ensure_terminal_invalid_raw_event(
    manifest: dict,
    summary: dict,
    call: ModelCall,
    ledger_binding: dict,
    scope: str,
    error: RuntimeError,
) -> dict:
    spec = _terminal_invalid_spec(scope)
    path = raw_path(manifest)
    if not path.exists():
        raise RuntimeError("terminal-invalid judge call has no raw log")
    events = read_jsonl(path)
    call_index = int(summary["call_index"])
    response_hash = policy_sha256(call.payload)
    response_rows = [
        (index, event)
        for index, event in enumerate(events)
        if event.get("event") == "model_response"
        and event.get("call_index") == call_index
    ]
    if len(response_rows) != 1:
        raise RuntimeError("terminal-invalid judge call lacks one raw response")
    response_index, response = response_rows[0]
    if (
        response.get("run_id") != manifest["run_id"]
        or response.get("call_kind") != summary["call_kind"]
        or policy_sha256(response.get("response_body")) != response_hash
    ):
        raise RuntimeError("terminal-invalid judge response differs from the raw log")
    metadata_indexes = [
        index
        for index, event in enumerate(events)
        if event.get("event") in {"generation_metadata", "generation_metadata_repair"}
        and event.get("call_index") == call_index
        and event.get("generation_id") == summary["generation_id"]
        and isinstance(event.get("generation"), dict)
        and bool(event["generation"])
        and event.get("error") is None
    ]
    if not metadata_indexes:
        raise RuntimeError("terminal-invalid judge call lacks settled raw metadata")

    stable = {
        "schema_version": JUDGE_TERMINAL_INVALID_SCHEMA,
        "event": JUDGE_TERMINAL_INVALID_EVENT,
        "run_id": manifest["run_id"],
        "scope": scope,
        "judge": spec["judge"],
        "call_index": call_index,
        "call_kind": summary["call_kind"],
        "model": summary["model"],
        "generation_id": summary["generation_id"],
        "reason_code": JUDGE_PARSE_INVALID_REASON,
        "stage": "schema_parse",
        "error_type": type(error).__name__,
        "error": JUDGE_PARSE_INVALID_MESSAGE,
        "response_body_sha256": response_hash,
        "call_summary_sha256": policy_sha256(summary),
        "ledger_call_sha256": policy_sha256(ledger_binding),
    }
    invalid_rows = [
        (index, event)
        for index, event in enumerate(events)
        if event.get("event") == JUDGE_TERMINAL_INVALID_EVENT
        and event.get("scope") == scope
    ]
    if invalid_rows:
        if len(invalid_rows) != 1:
            raise RuntimeError(f"{scope} judge has duplicate terminal-invalid events")
        invalid_index, existing = invalid_rows[0]
        expected_hash = existing.get("event_sha256")
        unhashed = {key: value for key, value in existing.items() if key != "event_sha256"}
        if (
            invalid_index <= max(response_index, *metadata_indexes)
            or any(existing.get(key) != value for key, value in stable.items())
            or not _sha256_text(expected_hash)
            or policy_sha256(unhashed) != expected_hash
        ):
            raise RuntimeError(f"{scope} terminal-invalid raw event differs")
        return existing

    event = {**stable, "timestamp": now_iso()}
    event["event_sha256"] = policy_sha256(event)
    append_jsonl(path, event)
    return event


async def record_terminal_invalid(
    *,
    path: Path,
    manifest: dict,
    summaries: list[dict],
    summary: dict,
    call: ModelCall,
    ledger: Ledger,
    scope: str,
    error: RuntimeError,
) -> None:
    spec = _terminal_invalid_spec(scope)
    ledger_binding = await _complete_ledger_binding(ledger, manifest, summary)
    event = _ensure_terminal_invalid_raw_event(
        manifest,
        summary,
        call,
        ledger_binding,
        scope,
        error,
    )
    failure = {
        "reason_code": event["reason_code"],
        "stage": event["stage"],
        "error_type": event["error_type"],
        "message": event["error"],
        "judge": event["judge"],
        "call_index": event["call_index"],
        "call_kind": event["call_kind"],
        "model": event["model"],
        "generation_id": event["generation_id"],
        "ledger_status": "complete",
        "response_body_sha256": event["response_body_sha256"],
        "call_summary_sha256": event["call_summary_sha256"],
        "ledger_call_sha256": event["ledger_call_sha256"],
        "raw_event_sha256": event["event_sha256"],
        "recorded_at": event["timestamp"],
    }
    updated = dict(manifest)
    updated[spec["status_field"]] = JUDGE_TERMINAL_INVALID
    updated[spec["failure_field"]] = failure
    updated[spec["calls_field"]] = summaries
    updated[spec["terminal_at_field"]] = event["timestamp"]
    atomic_write(path, pretty_json(updated))


def valid_primary_labels(manifest: dict) -> bool:
    labels = manifest.get("judge_labels")
    if not isinstance(labels, dict):
        return False
    conclusion = labels.get("conclusion")
    citation = labels.get("citation_presence")
    queries = manifest.get("query_labels")
    try:
        issued_queries = queries_for_judging(manifest)
    except ValueError:
        return False
    return (
        set(labels) == {"conclusion", "citation_presence"}
        and isinstance(conclusion, dict)
        and conclusion.get("label") in CONCLUSION_LABELS
        and isinstance(conclusion.get("justification"), str)
        and bool(conclusion["justification"].strip())
        and isinstance(citation, dict)
        and citation == citation_presence(manifest)
        and isinstance(citation.get("passes"), bool)
        and isinstance(queries, list)
        and len(queries) == len(issued_queries)
        and all(label in QUERY_LABELS for label in queries)
    )


def primary_judging_complete(manifest: dict) -> bool:
    return (
        manifest.get("status") in {"judged", "analysis_ready"}
        and valid_primary_labels(manifest)
    )


async def judge_one(
    path: Path,
    claims: dict[str, dict],
    prompts: dict,
    client: OpenRouterClient,
    ledger: Ledger,
    semaphore: asyncio.Semaphore,
    include_kimi: bool,
) -> str:
    loaded = load_json(path)
    if not isinstance(loaded, dict):
        raise RuntimeError(f"manifest is not an object: {path.name}")
    manifest = loaded
    if primary_judging_terminal_invalid(manifest):
        return "skipped"
    if primary_judging_complete(manifest):
        return "skipped"
    if manifest.get("status") != "agent_complete":
        raise RuntimeError(f"manifest is not ready for judging: {path.name}")
    claim_id = str(manifest.get("claim_id") or "")
    if claim_id not in claims:
        raise RuntimeError(f"manifest has unknown claim ID: {claim_id}")
    if not isinstance(manifest.get("report"), str) or not manifest["report"].strip():
        raise RuntimeError(f"manifest has no report: {path.name}")
    try:
        issued_queries = queries_for_judging(manifest)
    except ValueError as error:
        raise RuntimeError(f"manifest queries are not a list: {path.name}") from error

    existing_call = await ledger.get_call(manifest["run_id"], GEMINI_CONCLUSION_INDEX)
    if existing_call is not None and existing_call.get("status") == "failed":
        # A reconciled transport failure closes the slot for good, so the run
        # takes judging attrition rather than a replacement call.
        print(
            f"conclusion slot for {manifest['run_id']} is terminally failed after "
            "reconciliation; recording judging attrition",
            flush=True,
        )
        return "attrition"
    claim = claims[claim_id]
    citation = citation_presence(manifest)
    summaries = manifest.get("judge_calls")
    async with semaphore:
        conclusion_call = await judge_call(
            manifest=manifest,
            body=conclusion_body(manifest, claim, prompts, JUDGES["gemini"]),
            call_index=GEMINI_CONCLUSION_INDEX,
            call_kind="judge_conclusion_gemini",
            client=client,
            ledger=ledger,
            include_kimi=include_kimi,
        )
        conclusion_summary = judge_summary(
            conclusion_call,
            "judge_conclusion_gemini",
            "gemini",
            JUDGES["gemini"],
        )
        summaries = merge_call_summary(
            summaries,
            conclusion_summary,
        )
        try:
            conclusion = parse_conclusion(conclusion_call)
        except RuntimeError as error:
            await record_terminal_invalid(
                path=path,
                manifest=manifest,
                summaries=summaries,
                summary=conclusion_summary,
                call=conclusion_call,
                ledger=ledger,
                scope="primary",
                error=error,
            )
            return JUDGE_TERMINAL_INVALID
        query_labels: list[str] = []
        if issued_queries:
            query_call = await judge_call(
                manifest=manifest,
                body=query_body(manifest, claim, prompts),
                call_index=GEMINI_QUERY_INDEX,
                call_kind="judge_queries_gemini",
                client=client,
                ledger=ledger,
                include_kimi=include_kimi,
            )
            query_summary = judge_summary(
                query_call,
                "judge_queries_gemini",
                "gemini",
                JUDGES["gemini"],
            )
            summaries = merge_call_summary(
                summaries,
                query_summary,
            )
            try:
                query_labels = parse_queries(query_call, len(issued_queries))
            except RuntimeError as error:
                await record_terminal_invalid(
                    path=path,
                    manifest=manifest,
                    summaries=summaries,
                    summary=query_summary,
                    call=query_call,
                    ledger=ledger,
                    scope="primary",
                    error=error,
                )
                return JUDGE_TERMINAL_INVALID

    updated = dict(manifest)
    updated["judge_labels"] = {
        "conclusion": conclusion,
        "citation_presence": citation,
    }
    updated["query_labels"] = query_labels
    updated["judge_calls"] = summaries
    updated["judged_at"] = now_iso()
    updated["status"] = "judged"
    atomic_write(path, pretty_json(updated))
    return "completed"


def stable_order(run_id: str) -> str:
    return hashlib.sha256(f"{LUNA_SAMPLE_SEED}|{run_id}".encode()).hexdigest()


def select_luna_sample(manifests: list[dict], claims: dict[str, dict]) -> list[str]:
    grouped: dict[tuple[str, str, str], list[str]] = {}
    for manifest in manifests:
        if (
            manifest.get("arm") != "core"
            or manifest.get("system") not in SYSTEMS
            or manifest.get("framing") not in STANDARD_FRAMINGS
            or manifest.get("status") != "judged"
            or not valid_primary_labels(manifest)
        ):
            continue
        claim_id = str(manifest.get("claim_id") or "")
        if claim_id not in claims:
            raise RuntimeError(f"unknown claim in Luna population: {claim_id}")
        tier = str(claims[claim_id]["tier"])
        key = (str(manifest["system"]), str(manifest["framing"]), tier)
        grouped.setdefault(key, []).append(str(manifest["run_id"]))

    selected: list[str] = []
    cells = [
        (system, framing)
        for system in sorted(SYSTEMS)
        for framing in STANDARD_FRAMINGS
    ]
    for cell_index, (system, framing) in enumerate(cells):
        extra_tier = TIERS[cell_index % len(TIERS)]
        for tier in TIERS:
            quota = 9 if tier == extra_tier else 8
            candidates = sorted(
                grouped.get((system, framing, tier), []),
                key=lambda run_id: (stable_order(run_id), run_id),
            )
            if len(candidates) < quota:
                raise RuntimeError(
                    f"Luna stratum {system}/{framing}/{tier} has "
                    f"{len(candidates)} reports; needs {quota}"
                )
            selected.extend(candidates[:quota])
    if len(selected) != LUNA_SAMPLE_SIZE or len(set(selected)) != LUNA_SAMPLE_SIZE:
        raise RuntimeError("Luna selection did not produce 300 unique reports")
    return sorted(selected)


def verify_full_primary_judging(manifests: dict[str, dict]) -> None:
    schedule = load_json(SCHEDULE_PATH)
    if not isinstance(schedule, list):
        raise RuntimeError("schedule is not a list")
    required = [
        row["run_id"]
        for row in schedule
        if row.get("arm") not in ALL_PILOT_ARMS
    ]
    missing = [
        run_id
        for run_id in required
        if run_id not in manifests
        or (
            manifests[run_id].get("status") != TERMINAL_FAILURE_STATUS
            and not primary_judging_terminal_invalid(manifests[run_id])
            and (
                manifests[run_id].get("status") != "judged"
                or not valid_primary_labels(manifests[run_id])
            )
        )
    ]
    if missing:
        raise RuntimeError(
            f"Luna sampling requires all primary judging first; {len(missing)} run(s) missing"
        )


def luna_sample(manifests: dict[str, dict], claims: dict[str, dict]) -> list[str]:
    verify_full_primary_judging(manifests)
    selected = select_luna_sample(list(manifests.values()), claims)
    value = {
        "seed": LUNA_SAMPLE_SEED,
        "population": "core",
        "n": LUNA_SAMPLE_SIZE,
        "run_ids": selected,
    }
    if LUNA_SAMPLE_PATH.exists():
        existing = load_json(LUNA_SAMPLE_PATH)
        if existing != value:
            raise RuntimeError("existing Luna sample does not match deterministic selection")
    else:
        atomic_write(LUNA_SAMPLE_PATH, pretty_json(value))
    return selected


def valid_luna_label(manifest: dict) -> bool:
    labels = manifest.get("luna_labels")
    conclusion = labels.get("conclusion") if isinstance(labels, dict) else None
    return (
        isinstance(conclusion, dict)
        and conclusion.get("label") in CONCLUSION_LABELS
        and isinstance(conclusion.get("justification"), str)
        and bool(conclusion["justification"].strip())
    )


def query_sample_population(manifests: list[dict]) -> list[dict]:
    return [
        manifest
        for manifest in manifests
        if manifest.get("arm") == "core"
        and manifest.get("system") in SYSTEMS
        and manifest.get("framing") in STANDARD_FRAMINGS
        and manifest.get("status") == "judged"
        and valid_primary_labels(manifest)
        and isinstance(manifest.get("query_labels"), list)
        and manifest["query_labels"]
    ]


def select_query_sample(manifests: list[dict]) -> list[str]:
    """Pick runs cell by cell until the query budget is met.

    Strata are system by framing so every cell contributes, and within a cell
    runs are ordered by the frozen hash so the choice cannot drift.
    """

    grouped: dict[tuple[str, str], list[dict]] = {}
    for manifest in query_sample_population(manifests):
        key = (str(manifest["system"]), str(manifest["framing"]))
        grouped.setdefault(key, []).append(manifest)
    for candidates in grouped.values():
        candidates.sort(key=lambda value: (stable_order(str(value["run_id"])), value["run_id"]))
    cells = [
        (system, framing)
        for system in sorted(SYSTEMS)
        for framing in STANDARD_FRAMINGS
    ]
    missing = [cell for cell in cells if not grouped.get(cell)]
    if missing:
        raise RuntimeError(f"query sample is missing strata: {missing[:3]}")
    selected: list[str] = []
    queries = 0
    depth = 0
    while queries < QUERY_SAMPLE_QUERIES:
        advanced = False
        for cell in cells:
            candidates = grouped[cell]
            if depth >= len(candidates):
                continue
            advanced = True
            manifest = candidates[depth]
            selected.append(str(manifest["run_id"]))
            queries += len(manifest["query_labels"])
            if queries >= QUERY_SAMPLE_QUERIES:
                break
        if not advanced:
            raise RuntimeError(
                f"query sample population holds only {queries} labeled queries"
            )
        depth += 1
    return sorted(selected)


def query_sample(manifests: dict[str, dict]) -> list[str]:
    selected = select_query_sample(list(manifests.values()))
    labeled = sum(len(manifests[run_id]["query_labels"]) for run_id in selected)
    value = {
        "seed": LUNA_SAMPLE_SEED,
        "population": "core",
        "target_queries": QUERY_SAMPLE_QUERIES,
        "queries": labeled,
        "run_ids": selected,
    }
    if QUERY_SAMPLE_PATH.exists():
        existing = load_json(QUERY_SAMPLE_PATH)
        if existing != value:
            raise RuntimeError("existing query sample does not match deterministic selection")
    else:
        atomic_write(QUERY_SAMPLE_PATH, pretty_json(value))
    return selected


def valid_luna_query_labels(manifest: dict) -> bool:
    labels = manifest.get("luna_query_labels")
    issued = manifest.get("query_labels")
    return (
        isinstance(labels, list)
        and isinstance(issued, list)
        and len(labels) == len(issued)
        and all(label in QUERY_LABELS for label in labels)
    )


async def luna_query_one(
    path: Path,
    claims: dict[str, dict],
    prompts: dict,
    client: OpenRouterClient,
    ledger: Ledger,
    semaphore: asyncio.Semaphore,
    include_kimi: bool,
) -> str:
    loaded = load_json(path)
    if not isinstance(loaded, dict):
        raise RuntimeError(f"manifest is not an object: {path.name}")
    manifest = loaded
    if valid_luna_query_labels(manifest):
        return "skipped"
    if manifest.get("status") != "judged" or not valid_primary_labels(manifest):
        raise RuntimeError(f"manifest is not Gemini-judged: {path.name}")
    queries = queries_for_judging(manifest)
    if not queries:
        raise RuntimeError(f"manifest has no queries to relabel: {path.name}")
    claim = claims[str(manifest["claim_id"])]
    body = query_body(manifest, claim, prompts, JUDGES["luna"])
    async with semaphore:
        call = await judge_call(
            manifest=manifest,
            body=body,
            call_index=LUNA_QUERY_INDEX,
            call_kind="judge_queries_luna",
            client=client,
            ledger=ledger,
            include_kimi=include_kimi,
        )
    summary = judge_summary(call, "judge_queries_luna", "luna", JUDGES["luna"])
    summaries = merge_call_summary(manifest.get("luna_calls"), summary)
    try:
        labels = parse_queries(call, len(queries))
    except RuntimeError as error:
        await record_terminal_invalid(
            path=path,
            manifest=manifest,
            summaries=summaries,
            summary=summary,
            call=call,
            ledger=ledger,
            scope="luna",
            error=error,
        )
        return JUDGE_TERMINAL_INVALID
    updated = dict(manifest)
    updated["luna_query_labels"] = labels
    updated["luna_calls"] = summaries
    updated["luna_query_judged_at"] = now_iso()
    atomic_write(path, pretty_json(updated))
    return "completed"


def select_stance_sample(entries: list[tuple[Path, dict]], per_cell: int) -> set[str]:
    """Balanced core sample across framing and system, ordered by the frozen hash."""

    grouped: dict[tuple[str, str], list[str]] = {}
    for _path, manifest in entries:
        if (
            manifest.get("arm") != "core"
            or manifest.get("system") not in SYSTEMS
            or manifest.get("framing") not in STANDARD_FRAMINGS
            or not stance_searches(manifest)
        ):
            continue
        key = (str(manifest["system"]), str(manifest["framing"]))
        grouped.setdefault(key, []).append(str(manifest["run_id"]))
    selected: set[str] = set()
    for run_ids in grouped.values():
        run_ids.sort(key=lambda run_id: (stable_order(run_id), run_id))
        selected.update(run_ids[:per_cell])
    return selected


def stance_call_index(round_index: int, search_index: int) -> int:
    return STANCE_INDEX_BASE + STANCE_ROUND_STRIDE * round_index + search_index


def stance_labels(call: ModelCall, count: int) -> list[str]:
    """Parse and validate the stance labels from an unconstrained response."""

    message = response_message(call.payload)
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("stance judge returned no text")
    text = content.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.lstrip().lower().startswith("json"):
            text = text.lstrip()[4:]
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        raise RuntimeError("stance judge returned no JSON object")
    try:
        value = json.loads(text[start : end + 1])
    except json.JSONDecodeError as error:
        raise RuntimeError("stance judge returned unparseable JSON") from error
    labels = value.get("labels") if isinstance(value, dict) else None
    if (
        not isinstance(labels, list)
        or len(labels) != count
        or any(label not in STANCE_LABELS for label in labels)
    ):
        raise RuntimeError("stance judge returned invalid labels")
    return list(labels)


def stance_body(claim: dict, search: dict, prompts: dict) -> dict:
    results = search["results"]
    task = {
        "claim": claim["claim"],
        "query": search.get("query"),
        "results": [
            {"title": result.get("title"), "snippet": result.get("snippet")}
            for result in results
        ],
        "instruction": "Return one label for each result in the same order.",
    }
    model = JUDGES["gemini"]
    # The BYOK route stopped accepting strict json_schema response formats, so
    # the schema is stated in the instruction and enforced when parsing.
    instruction = (
        f"{prompts['evidence_stance_judge']} Respond with JSON only, in the form "
        f'{{"labels": [...]}}, containing exactly {len(results)} labels.'
    )
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": instruction},
            {
                "role": "user",
                "content": json.dumps(task, ensure_ascii=False, separators=(",", ":")),
            },
        ],
        **model_params(model, judge=True),
        "include_reasoning": True,
        "usage": {"include": True},
        "provider": provider_body(model),
    }


def stance_searches(manifest: dict) -> list[dict]:
    """Return the searches whose retrieved results can carry stance labels."""

    if manifest.get("arm") in FIXED_EVIDENCE_ARMS:
        return []
    searches = manifest.get("searches")
    if not isinstance(searches, list):
        return []
    return [
        search
        for search in searches
        if isinstance(search, dict)
        and isinstance(search.get("results"), list)
        and search["results"]
    ]


def valid_stance_labels(manifest: dict) -> bool:
    searches = stance_searches(manifest)
    recorded = manifest.get("evidence_stance")
    if not isinstance(recorded, list) or len(recorded) != len(searches):
        return False
    for search, entry in zip(searches, recorded):
        if not isinstance(entry, dict):
            return False
        labels = entry.get("labels")
        if not isinstance(labels, list) or len(labels) != len(search["results"]):
            return False
        if any(label not in STANCE_LABELS for label in labels):
            return False
    return True


async def stance_one(
    path: Path,
    claims: dict[str, dict],
    prompts: dict,
    client: OpenRouterClient,
    ledger: Ledger,
    semaphore: asyncio.Semaphore,
    include_kimi: bool,
    round_index: int = 0,
) -> str:
    loaded = load_json(path)
    if not isinstance(loaded, dict):
        raise RuntimeError(f"manifest is not an object: {path.name}")
    manifest = loaded
    searches = stance_searches(manifest)
    if not searches or valid_stance_labels(manifest):
        return "skipped"
    if manifest.get("status") not in {"judged", "analysis_ready"}:
        # Runs whose primary judging failed carry no conclusion label and never
        # reach the composition estimate, so they are left unlabelled.
        return "skipped"
    if len(searches) > STANCE_MAX_SEARCHES:
        raise RuntimeError(f"{path.name}: more searches than reserved call slots")
    claim = claims[str(manifest["claim_id"])]
    recorded: list[dict] = []
    summaries = manifest.get("stance_calls")
    for index, search in enumerate(searches):
        try:
            async with semaphore:
                call = await judge_call(
                    manifest=manifest,
                    body=stance_body(claim, search, prompts),
                    call_index=stance_call_index(round_index, index),
                    call_kind="judge_evidence_stance_gemini",
                    client=client,
                    ledger=ledger,
                    include_kimi=include_kimi,
                )
        except RuntimeError as error:
            # The stance audit is a post-hoc instrument, so a provider refusal
            # leaves the run unlabelled and the coverage is reported instead.
            print(
                f"stance labelling unavailable for {manifest['run_id']}: {error}",
                flush=True,
            )
            return "unavailable"
        summaries = merge_call_summary(
            summaries,
            judge_summary(
                call, "judge_evidence_stance_gemini", "gemini", JUDGES["gemini"]
            ),
        )
        try:
            labels = stance_labels(call, len(search["results"]))
        except RuntimeError as error:
            print(f"stance labelling invalid for {manifest['run_id']}: {error}", flush=True)
            return "unavailable"
        recorded.append(
            {
                "query": search.get("query"),
                "normalized_query": search.get("normalized_query"),
                "labels": list(labels),
            }
        )
    updated = dict(manifest)
    updated["evidence_stance"] = recorded
    updated["stance_calls"] = summaries
    updated["stance_judged_at"] = now_iso()
    atomic_write(path, pretty_json(updated))
    return "completed"


async def luna_one(
    path: Path,
    claims: dict[str, dict],
    prompts: dict,
    client: OpenRouterClient,
    ledger: Ledger,
    semaphore: asyncio.Semaphore,
    include_kimi: bool,
) -> str:
    loaded = load_json(path)
    if not isinstance(loaded, dict):
        raise RuntimeError(f"manifest is not an object: {path.name}")
    manifest = loaded
    if luna_judging_terminal_invalid(manifest):
        return "skipped"
    if valid_luna_label(manifest):
        return "skipped"
    if manifest.get("status") != "judged" or not valid_primary_labels(manifest):
        raise RuntimeError(f"manifest is not Gemini-judged: {path.name}")
    existing_call = await ledger.get_call(manifest["run_id"], LUNA_CONCLUSION_INDEX)
    if existing_call is not None and existing_call.get("status") == "failed":
        # A reconciled unbilled transport failure closes the slot permanently.
        # The frozen sample takes the attrition rather than a replacement call.
        print(
            f"luna slot for {manifest['run_id']} is terminally failed after "
            "reconciliation; sample member recorded as unattempted attrition",
            flush=True,
        )
        return "skipped"
    claim = claims[str(manifest["claim_id"])]
    async with semaphore:
        call = await judge_call(
            manifest=manifest,
            body=conclusion_body(manifest, claim, prompts, JUDGES["luna"]),
            call_index=LUNA_CONCLUSION_INDEX,
            call_kind="judge_conclusion_luna",
            client=client,
            ledger=ledger,
            include_kimi=include_kimi,
        )
    summary = judge_summary(
        call,
        "judge_conclusion_luna",
        "luna",
        JUDGES["luna"],
    )
    summaries = merge_call_summary(manifest.get("luna_calls"), summary)
    try:
        conclusion = parse_conclusion(call)
    except RuntimeError as error:
        await record_terminal_invalid(
            path=path,
            manifest=manifest,
            summaries=summaries,
            summary=summary,
            call=call,
            ledger=ledger,
            scope="luna",
            error=error,
        )
        return JUDGE_TERMINAL_INVALID
    updated = dict(manifest)
    updated["luna_labels"] = {"conclusion": conclusion}
    updated["luna_calls"] = summaries
    updated["luna_judged_at"] = now_iso()
    atomic_write(path, pretty_json(updated))
    return "completed"


def load_manifests(
    arms: set[str] | None = None,
    claim_ids: set[str] | None = None,
) -> list[tuple[Path, dict]]:
    values = []
    for path in sorted(RUNS_DIR.glob("*.json")):
        loaded = load_json(path)
        if not isinstance(loaded, dict):
            raise RuntimeError(f"manifest is not an object: {path.name}")
        if (arms is None or loaded.get("arm") in arms) and (
            claim_ids is None or loaded.get("claim_id") in claim_ids
        ):
            values.append((path, loaded))
    return values


def pilot_sample_paths(
    schedule: list[dict], entries: list[tuple[Path, dict]]
) -> list[Path]:
    scheduled = [
        row for row in schedule if row.get("arm") in PILOT_MANIFEST_COUNTS
    ]
    counts = Counter(str(row.get("arm") or "") for row in scheduled)
    if counts != Counter(PILOT_MANIFEST_COUNTS):
        raise RuntimeError(
            "pilot judging requires the complete 46-run schedule "
            f"({dict(counts)} found; {PILOT_MANIFEST_COUNTS} expected)"
        )

    scheduled_by_id = {str(row.get("run_id") or ""): row for row in scheduled}
    if "" in scheduled_by_id or len(scheduled_by_id) != sum(PILOT_MANIFEST_COUNTS.values()):
        raise RuntimeError("pilot schedule has missing or duplicate run IDs")

    entries_by_id: dict[str, tuple[Path, dict]] = {}
    duplicates: list[str] = []
    for path, manifest in entries:
        run_id = str(manifest.get("run_id") or "")
        if run_id not in scheduled_by_id:
            continue
        if run_id in entries_by_id:
            duplicates.append(run_id)
        entries_by_id[run_id] = (path, manifest)
    if duplicates:
        raise RuntimeError(
            f"pilot judging found {len(set(duplicates))} duplicate manifest run ID(s)"
        )

    missing = sorted(set(scheduled_by_id) - set(entries_by_id))
    if missing:
        raise RuntimeError(
            "pilot judging is blocked until all 46 trajectories finish; "
            f"{len(missing)} manifest(s) are missing"
        )

    identity_fields = ("arm", "claim_id", "framing", "system", "model", "repeat")
    mismatched = [
        run_id
        for run_id, row in scheduled_by_id.items()
        if any(entries_by_id[run_id][1].get(field) != row.get(field) for field in identity_fields)
    ]
    if mismatched:
        raise RuntimeError(
            f"pilot judging found {len(mismatched)} manifest(s) that do not match the schedule"
        )

    not_ready = sorted(
        run_id
        for run_id, (_, manifest) in entries_by_id.items()
        if manifest.get("status") not in PILOT_TERMINAL_STATUSES
    )
    if not_ready:
        raise RuntimeError(
            "pilot judging is blocked until all 46 trajectories finish; "
            f"{len(not_ready)} manifest(s) are not agent-complete"
        )

    ineligible: list[str] = []
    for run_id, row in scheduled_by_id.items():
        manifest = entries_by_id[run_id][1]
        if (
            manifest.get("status") == TERMINAL_FAILURE_STATUS
            and isinstance(manifest.get("failure"), dict)
            and manifest["failure"]
        ):
            continue
        queries = manifest.get("queries")
        issued_queries = manifest.get("search_tool_queries")
        if issued_queries is None:
            issued_queries = queries
        searches = manifest.get("searches")
        search_count = manifest.get("search_count")
        tool_call_count = manifest.get("search_tool_call_count", search_count)
        exhaustions = manifest.get("query_exhaustions", [])
        exhaustion_count = manifest.get("query_exhaustion_count", len(exhaustions))
        distinct_count = manifest.get("distinct_search_count")
        queries_valid = isinstance(queries, list) and all(
            isinstance(query, str) and bool(query.strip()) for query in queries
        )
        searches_valid = isinstance(searches, list) and all(
            isinstance(search, dict) for search in searches
        )
        issued_queries_valid = isinstance(issued_queries, list) and all(
            isinstance(query, str) and bool(query.strip()) for query in issued_queries
        )
        search_entries_valid = (
            searches_valid
            and queries_valid
            and len(searches) == len(queries)
        )
        if search_entries_valid:
            try:
                for query, search in zip(queries, searches):
                    validate_strict_v3_search_result(search, query, "web")
            except RuntimeError:
                search_entries_valid = False
        exhaustions_valid = isinstance(exhaustions, list) and all(
            isinstance(item, dict) for item in exhaustions
        )
        if exhaustions_valid:
            try:
                for item in exhaustions:
                    validate_v5_query_exhausted_result(item, item.get("query"), "web")
            except (TypeError, RuntimeError):
                exhaustions_valid = False
        observed_distinct = len(set(queries)) if queries_valid else None
        try:
            expected_hash = expected_agent_params_hash(row)
        except (OSError, KeyError, TypeError, RuntimeError, json.JSONDecodeError) as error:
            raise RuntimeError(
                f"pilot judging cannot validate the frozen params hash: {error}"
            ) from error
        if (
            isinstance(search_count, bool)
            or not isinstance(search_count, int)
            or not MIN_SEARCHES <= search_count <= SEARCH_LIMIT
            or manifest.get("minimum_searches") != MIN_SEARCHES
            or manifest.get("successful_search_count", search_count) != search_count
            or not queries_valid
            or len(queries) != search_count
            or not searches_valid
            or len(searches) != search_count
            or not search_entries_valid
            or not issued_queries_valid
            or isinstance(tool_call_count, bool)
            or not isinstance(tool_call_count, int)
            or not search_count <= tool_call_count <= SEARCH_LIMIT
            or len(issued_queries) != tool_call_count
            or not exhaustions_valid
            or isinstance(exhaustion_count, bool)
            or not isinstance(exhaustion_count, int)
            or exhaustion_count != len(exhaustions)
            or tool_call_count != search_count + exhaustion_count
            or isinstance(distinct_count, bool)
            or not isinstance(distinct_count, int)
            or distinct_count != observed_distinct
            or not 1 <= distinct_count <= search_count
            or manifest.get("params_hash") != expected_hash
        ):
            ineligible.append(run_id)
    if ineligible:
        raise RuntimeError(
            "pilot judging found "
            f"{len(ineligible)} manifest(s) that fail frozen pilot-v7 "
            "strict-v3 trajectory eligibility"
        )

    selected = pilot_judge_ids(schedule, cohort=PILOT_COHORT)
    if len(selected) != 10 or len(set(selected)) != 10:
        raise RuntimeError("pilot judging did not select exactly 10 standard reports")
    judgeable = [
        run_id
        for run_id in selected
        if entries_by_id[run_id][1].get("status") != TERMINAL_FAILURE_STATUS
    ]
    if not judgeable:
        raise RuntimeError("every sampled pilot report failed terminally")
    without_queries = [
        run_id
        for run_id in judgeable
        if not isinstance(
            entries_by_id[run_id][1].get(
                "search_tool_queries", entries_by_id[run_id][1].get("queries")
            ),
            list,
        )
        or not entries_by_id[run_id][1].get(
            "search_tool_queries", entries_by_id[run_id][1].get("queries")
        )
    ]
    if without_queries:
        raise RuntimeError(
            "pilot judging requires a conclusion and query call for every sampled report; "
            f"{len(without_queries)} sampled manifest(s) have no queries"
        )
    return [entries_by_id[run_id][0] for run_id in judgeable]


def manifest_order(entry: tuple[Path, dict]) -> tuple:
    _, manifest = entry
    repeat = manifest.get("repeat")
    return (
        str(manifest.get("claim_id") or ""),
        str(manifest.get("arm") or ""),
        str(manifest.get("framing") or ""),
        str(manifest.get("system") or ""),
        int(repeat) if isinstance(repeat, int) else 0,
        str(manifest.get("run_id") or ""),
    )


def pending_primary_paths(
    entries: list[tuple[Path, dict]], limit: int | None = None
) -> list[Path]:
    pending = [
        (path, manifest)
        for path, manifest in sorted(entries, key=manifest_order)
        if manifest.get("status") != TERMINAL_FAILURE_STATUS
        and not primary_judging_terminal_invalid(manifest)
        and not primary_judging_complete(manifest)
    ]
    if limit is not None:
        pending = pending[:limit]
    return [path for path, _ in pending]


def pending_luna_paths(
    entries: list[tuple[Path, dict]], limit: int | None = None
) -> list[Path]:
    pending = [
        (path, manifest)
        for path, manifest in sorted(entries, key=manifest_order)
        if manifest.get("status") != TERMINAL_FAILURE_STATUS
        and not luna_judging_terminal_invalid(manifest)
        and not valid_luna_label(manifest)
    ]
    if limit is not None:
        pending = pending[:limit]
    return [path for path, _ in pending]


def claim_batches(paths: list[Path]) -> list[tuple[str, list[Path]]]:
    grouped: dict[str, list[Path]] = defaultdict(list)
    for path in paths:
        manifest = load_json(path)
        if not isinstance(manifest, dict):
            raise RuntimeError(f"manifest is not an object: {path.name}")
        claim_id = str(manifest.get("claim_id") or "")
        if not claim_id:
            raise RuntimeError(f"manifest has no claim ID: {path.name}")
        grouped[claim_id].append(path)
    return list(grouped.items())


def install_sigterm_handler(
    stop_requested: asyncio.Event,
) -> tuple[asyncio.AbstractEventLoop, bool]:
    loop = asyncio.get_running_loop()

    def request_stop() -> None:
        if not stop_requested.is_set():
            print(
                "SIGTERM received; finishing the current claim batch before exit",
                flush=True,
            )
            stop_requested.set()

    try:
        loop.add_signal_handler(signal.SIGTERM, request_stop)
        return loop, True
    except (NotImplementedError, RuntimeError):
        return loop, False


async def execute_pass(
    paths: list[Path],
    claims: dict[str, dict],
    prompts: dict,
    include_kimi: bool,
    concurrency: int,
    stop_requested: asyncio.Event | None = None,
    *,
    worker: Callable | None = None,
    label: str = "judge",
) -> None:
    worker = worker or judge_one
    batches = claim_batches(paths)
    owns_stop_event = stop_requested is None
    if stop_requested is None:
        stop_requested = asyncio.Event()
    loop, signal_handler_installed = (
        install_sigterm_handler(stop_requested)
        if owns_stop_event
        else (asyncio.get_running_loop(), False)
    )
    ledger = Ledger()
    client = OpenRouterClient(ledger)
    semaphore = asyncio.Semaphore(concurrency)
    completed = 0
    terminal_invalid = 0
    try:
        for claim_id, claim_paths in batches:
            if stop_requested.is_set():
                break
            results = await asyncio.gather(
                *(
                    worker(
                        path,
                        claims,
                        prompts,
                        client,
                        ledger,
                        semaphore,
                        include_kimi,
                    )
                    for path in claim_paths
                ),
                return_exceptions=True,
            )
            failures = [result for result in results if isinstance(result, BaseException)]
            completed += sum(result == "completed" for result in results)
            terminal_invalid += sum(
                result == JUDGE_TERMINAL_INVALID for result in results
            )
            totals = await ledger.totals()
            print(
                f"{label}_new={completed} {label}_terminal_invalid={terminal_invalid} "
                f"calls={totals.calls} "
                f"openrouter=${totals.openrouter:.4f} "
                f"gemini_upstream=${totals.gemini_upstream:.4f}",
                flush=True,
            )
            if failures:
                raise RuntimeError(f"{label} batch failed: {failures[0]}") from failures[0]
            if stop_requested.is_set():
                print(f"graceful stop completed after claim={claim_id}", flush=True)
                break
    finally:
        await client.close()
        if signal_handler_installed:
            loop.remove_signal_handler(signal.SIGTERM)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arms", nargs="+")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--claim-ids", nargs="+")
    parser.add_argument("--concurrency", type=int, default=12)
    parser.add_argument("--include-kimi", action="store_true")
    parser.add_argument("--pilot-sample", action="store_true")
    parser.add_argument("--luna-sample", action="store_true")
    parser.add_argument("--luna-query-sample", action="store_true")
    parser.add_argument("--stance-audit", action="store_true")
    parser.add_argument("--stance-round", type=int, default=0)
    parser.add_argument("--stance-per-cell", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.concurrency < 1:
        parser.error("--concurrency must be positive")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    modes = [args.pilot_sample, args.luna_sample, args.luna_query_sample, args.stance_audit]
    if sum(bool(mode) for mode in modes) > 1:
        parser.error("judging modes are mutually exclusive")
    if args.pilot_sample and (args.arms or args.limit or args.claim_ids):
        parser.error("--pilot-sample cannot be combined with arm, limit, or claim filters")
    if args.arms and ALL_PILOT_ARMS & set(args.arms):
        parser.error("use --pilot-sample for the gated pilot judging pass")

    enforce_judge_launch_gate(args.pilot_sample, args.include_kimi)

    claims = {claim["id"]: claim for claim in load_claims()}
    prompts = load_json(PROMPTS_PATH)
    if not isinstance(prompts, dict):
        raise RuntimeError("prompt file is not an object")
    requested_arms = set(args.arms) if args.arms else None
    entries = load_manifests(
        requested_arms,
        set(args.claim_ids) if args.claim_ids else None,
    )

    if args.pilot_sample:
        schedule = load_json(SCHEDULE_PATH)
        if not isinstance(schedule, list):
            raise RuntimeError("schedule is not a list")
        paths = pilot_sample_paths(schedule, entries)
    elif args.luna_sample:
        all_entries = load_manifests()
        manifests = {manifest["run_id"]: manifest for _, manifest in all_entries}
        selected = luna_sample(manifests, claims)
        entries_by_id = {
            manifest["run_id"]: (path, manifest) for path, manifest in all_entries
        }
        selected_entries = [entries_by_id[run_id] for run_id in selected]
        paths = pending_luna_paths(selected_entries, args.limit)
    elif args.luna_query_sample:
        all_entries = load_manifests()
        manifests = {manifest["run_id"]: manifest for _, manifest in all_entries}
        selected = set(query_sample(manifests))
        paths = [
            path
            for path, manifest in sorted(all_entries, key=manifest_order)
            if manifest["run_id"] in selected and not valid_luna_query_labels(manifest)
        ][: args.limit]
    elif args.stance_audit:
        sampled = (
            select_stance_sample(entries, args.stance_per_cell)
            if args.stance_per_cell
            else None
        )
        paths = [
            path
            for path, manifest in sorted(entries, key=manifest_order)
            if manifest.get("arm") not in ALL_PILOT_ARMS
            and stance_searches(manifest)
            and not valid_stance_labels(manifest)
            and (sampled is None or manifest["run_id"] in sampled)
        ][: args.limit]
    else:
        selected_entries = [
            (path, manifest)
            for path, manifest in entries
            if manifest.get("arm") not in ALL_PILOT_ARMS
        ]
        paths = pending_primary_paths(selected_entries, args.limit)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "manifests": len(paths),
                    "pilot": args.pilot_sample,
                    "luna": args.luna_sample,
                }
            )
        )
        return
    if not paths:
        print("no pending judge manifests match the requested filters", flush=True)
        return
    worker, label = judge_one, "judge"
    if args.luna_sample:
        worker, label = luna_one, "luna"
    elif args.luna_query_sample:
        worker, label = luna_query_one, "luna_query"
    elif args.stance_audit:
        label = "stance"
        round_index = args.stance_round

        def worker(*call_args):
            return stance_one(*call_args, round_index=round_index)
    asyncio.run(
        execute_pass(
            paths,
            claims,
            prompts,
            args.include_kimi,
            args.concurrency,
            worker=worker,
            label=label,
        )
    )


if __name__ == "__main__":
    main()
