from __future__ import annotations

import argparse
import asyncio
import json
import re
from pathlib import Path

from config import (
    ARCHIVE,
    DATA,
    SYSTEMS,
    append_jsonl,
    atomic_write,
    model_params,
    now_iso,
    params_hash,
    pretty_json,
    read_jsonl,
)
from openrouter import Ledger, ModelCall, OpenRouterClient
from run_experiments import (
    SEARCH_ATTEMPT_REJECTED_EVENT,
    SEARCH_TOOL,
    call_summary,
    provider_route,
    raw_path,
    recover_call,
    response_message,
    stable_search_execution_id,
    tool_calls,
    validate_response,
)
from search import SearchTool


PREFLIGHT_PROMPT = (
    "Check that the research tool is available. Internally compare at least two useful "
    "query formulations for finding randomized trials of whether exercise lowers blood "
    "pressure. Then call search exactly once with the best query. Do not answer the "
    "research question."
)
REASONING_PROBE_PROMPT = (
    "Compare 47 x 83 with 52 x 74. Think through both products, then give one short "
    "sentence stating which product is larger."
)
PREFLIGHT_CALLS = (
    ("byok_gemini_1", SYSTEMS["gemini"]),
    ("byok_gemini_2", SYSTEMS["gemini"]),
    ("byok_gemini_3", SYSTEMS["gemini"]),
    ("parity_sonnet5_1", SYSTEMS["sonnet5"]),
    ("parity_terra_1", SYSTEMS["terra"]),
    ("parity_glm_1", SYSTEMS["glm"]),
)
BATCH_LABEL_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}")
COHORT_PATTERN = re.compile(r"v[1-9][0-9]*")


def cohort_artifact_paths(cohort: str | None) -> tuple[Path, Path]:
    if cohort is None:
        return DATA / "preflight.json", ARCHIVE / "byok_verification.json"
    if COHORT_PATTERN.fullmatch(cohort) is None:
        raise ValueError("preflight cohort must look like 'v5'")
    return (
        DATA / f"preflight_{cohort}.json",
        ARCHIVE / f"byok_verification_{cohort}.json",
    )


def resolve_batch_label(batch_label: str | None, cohort: str | None) -> str | None:
    if cohort is None:
        return batch_label
    cohort_artifact_paths(cohort)
    resolved = batch_label or f"{cohort}_preflight"
    if not resolved.startswith(f"{cohort}_"):
        raise ValueError(
            f"cohort {cohort} preflight run IDs require a {cohort}_ batch-label prefix"
        )
    return resolved


def preflight_model_params(model: str, protocol: str) -> dict:
    if protocol == "legacy":
        return model_params(model)
    return model_params(model, protocol=protocol)


def row_for(run_id: str, model: str) -> dict:
    return {
        "run_id": run_id,
        "arm": "preflight",
        "model": model,
    }


def preflight_calls(batch_label: str | None = None) -> list[tuple[str, str]]:
    if batch_label is None:
        return list(PREFLIGHT_CALLS)
    if BATCH_LABEL_PATTERN.fullmatch(batch_label) is None:
        raise ValueError(
            "batch label must be 1-64 ASCII letters, digits, underscores, or hyphens "
            "and must start with a letter or digit"
        )
    return [(f"{batch_label}_{run_id}", model) for run_id, model in PREFLIGHT_CALLS]


def reasoning_present(call) -> bool:
    message = response_message(call.payload)
    if str(message.get("reasoning") or "").strip():
        return True
    if str(message.get("reasoning_content") or "").strip():
        return True
    details = message.get("reasoning_details")
    if isinstance(details, list) and details:
        return True
    if call.reasoning_tokens > 0:
        return True
    for key in ("reasoning_tokens", "tokens_reasoning", "native_tokens_reasoning"):
        if int(call.generation.get(key) or 0) > 0:
            return True
    return False


def final_response_present(call) -> bool:
    content = response_message(call.payload).get("content")
    if isinstance(content, str):
        return bool(content.strip())
    if isinstance(content, list):
        for item in content:
            if isinstance(item, str) and item.strip():
                return True
            if isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if isinstance(text, str) and text.strip():
                    return True
    return False


def reasoning_probe_body(model: str, protocol: str = "legacy") -> dict:
    parameters = preflight_model_params(model, protocol)
    return {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are testing whether reasoning metadata is returned.",
            },
            {"role": "user", "content": REASONING_PROBE_PROMPT},
        ],
        **parameters,
        "include_reasoning": True,
        "usage": {"include": True},
        "provider": provider_route(model, protocol=protocol),
    }


async def run_reasoning_probe(
    client: OpenRouterClient,
    ledger: Ledger,
    row: dict,
    path: Path,
    model: str,
    protocol: str = "legacy",
) -> tuple[ModelCall, dict]:
    body = reasoning_probe_body(model, protocol)
    events = read_jsonl(path) if path.exists() else []
    probe = await recover_call(client, ledger, row, 2, events)
    if probe is None:
        probe = await client.call(
            body=body,
            run_id=row["run_id"],
            call_index=2,
            call_kind="preflight",
            raw_path=path,
        )
    validate_response(probe, model, provider_protocol=protocol)
    if not probe.generation:
        raise RuntimeError(f"{model} reasoning probe returned incomplete generation metadata")
    if not final_response_present(probe):
        raise RuntimeError(f"{model} reasoning probe returned no nonempty final response")
    if not reasoning_present(probe):
        raise RuntimeError(
            f"{model} returned no reasoning evidence in either the primary preflight "
            "call or its single reasoning probe"
        )
    return probe, body


async def one_call(
    client: OpenRouterClient,
    ledger: Ledger,
    run_id: str,
    model: str,
    protocol: str = "legacy",
) -> dict:
    row = row_for(run_id, model)
    path = raw_path(row)
    events = read_jsonl(path) if path.exists() else []
    call = await recover_call(client, ledger, row, 1, events)
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are testing a research-agent tool contract."},
            {"role": "user", "content": PREFLIGHT_PROMPT},
        ],
        **preflight_model_params(model, protocol),
        "include_reasoning": True,
        "usage": {"include": True},
        "provider": provider_route(model, protocol=protocol),
        "tools": [SEARCH_TOOL],
        "tool_choice": "required",
    }
    if call is None:
        call = await client.call(
            body=body,
            run_id=run_id,
            call_index=1,
            call_kind="preflight",
            raw_path=path,
        )
    validate_response(call, model, provider_protocol=protocol)
    if not call.generation:
        raise RuntimeError(f"{model} returned incomplete generation metadata")
    calls = tool_calls(response_message(call.payload))
    if len(calls) != 1 or calls[0].get("function", {}).get("name") != "search":
        raise RuntimeError(f"{model} did not return exactly one search tool call")
    arguments_raw = calls[0]["function"].get("arguments")
    arguments = arguments_raw
    if isinstance(arguments, str):
        arguments = json.loads(arguments)
    if (
        not isinstance(arguments, dict)
        or set(arguments) != {"query"}
        or not isinstance(arguments["query"], str)
        or not arguments["query"].strip()
    ):
        raise RuntimeError(f"{model} returned invalid search arguments: {arguments!r}")
    primary_reasoning_present = reasoning_present(call)
    probe = None
    probe_body = None
    if not primary_reasoning_present:
        probe, probe_body = await run_reasoning_probe(
            client, ledger, row, path, model, protocol
        )
    return {
        "run_id": run_id,
        "requested_model": model,
        "params_hash": params_hash(body),
        "tool_contract_valid": True,
        "search_arguments": arguments,
        "tool_call_id": calls[0]["id"],
        "tool_arguments_raw": arguments_raw,
        "reasoning_present": True,
        "primary_reasoning_present": primary_reasoning_present,
        "reasoning_evidence_call_index": 1 if primary_reasoning_present else 2,
        "reasoning_probe": (
            {
                **call_summary(probe, model, "preflight_reasoning_probe"),
                "params_hash": params_hash(probe_body),
            }
            if probe is not None and probe_body is not None
            else None
        ),
        "call": call_summary(call, model, "preflight"),
        "generation": call.generation,
    }


async def collect_model_results(
    client: OpenRouterClient,
    ledger: Ledger,
    calls: list[tuple[str, str]],
    protocol: str = "legacy",
) -> list[dict]:
    outcomes = await asyncio.gather(
        *(
            one_call(client, ledger, run_id, model, protocol)
            for run_id, model in calls
        ),
        return_exceptions=True,
    )
    failures = []
    results = []
    for (run_id, model), outcome in zip(calls, outcomes, strict=True):
        if isinstance(outcome, BaseException):
            failures.append((run_id, model, outcome))
        else:
            results.append(outcome)
    if failures:
        details = "\n".join(
            f"- {run_id} [{model}]: {type(error).__name__}: {error}"
            for run_id, model, error in failures
        )
        raise RuntimeError(
            f"{len(failures)} preflight model call(s) failed after all started calls "
            f"settled:\n{details}"
        )
    return results


async def execute(
    batch_label: str | None = None,
    *,
    cohort: str | None = None,
) -> None:
    batch_label = resolve_batch_label(batch_label, cohort)
    preflight_path, byok_path = cohort_artifact_paths(cohort)
    calls = preflight_calls(batch_label)
    protocol = "v5" if cohort == "v5" else "legacy"
    ledger = Ledger()
    client = OpenRouterClient(ledger)
    try:
        results = await collect_model_results(client, ledger, calls, protocol)
    finally:
        await client.close()
    search = SearchTool()
    try:
        for result in results:
            row = row_for(result["run_id"], result["requested_model"])
            path = raw_path(row)
            existing = read_jsonl(path)
            if not any(event.get("event") == "tool_results" for event in existing):
                query = result["search_arguments"]["query"]
                execution_id = stable_search_execution_id(
                    result["run_id"],
                    1,
                    result["tool_call_id"],
                    query,
                    "web",
                )
                completion_provenance = []

                def record_rejection(attempt: dict) -> None:
                    event = {
                        **attempt,
                        "event": SEARCH_ATTEMPT_REJECTED_EVENT,
                        "run_id": result["run_id"],
                        "call_index": 1,
                        "tool_call_id": result["tool_call_id"],
                        "arguments": result["tool_arguments_raw"],
                        "timestamp": attempt["finished_at"],
                    }
                    matches = [
                        prior
                        for prior in read_jsonl(path)
                        if prior.get("event") == SEARCH_ATTEMPT_REJECTED_EVENT
                        and prior.get("search_execution_id") == execution_id
                        and prior.get("attempt") == event.get("attempt")
                    ]
                    if len(matches) > 1 or (matches and matches[0] != event):
                        raise RuntimeError(
                            "preflight rejected-search provenance differs on resume"
                        )
                    if not matches:
                        append_jsonl(path, event)

                search_result = await search.search(
                    query,
                    "web",
                    attempt_recorder=record_rejection,
                    provenance_recorder=completion_provenance.append,
                    search_execution_id=execution_id,
                )
                if len(completion_provenance) != 1:
                    raise RuntimeError("preflight search completion lacks provenance")
                append_jsonl(
                    path,
                    {
                        "event": "tool_results",
                        "run_id": result["run_id"],
                        "call_index": 1,
                        "timestamp": now_iso(),
                        "tool_results": [
                            {
                                "tool_call_id": result["tool_call_id"],
                                "arguments": result["tool_arguments_raw"],
                                "query": result["search_arguments"]["query"],
                                "result": search_result,
                                "search_provenance": completion_provenance[0],
                            }
                        ],
                    },
                )
            result.pop("tool_arguments_raw")
    finally:
        await search.close()
    byok = [result for result in results if result["requested_model"] == SYSTEMS["gemini"]]
    if len(byok) != 3 or not all(result["call"]["is_byok"] for result in byok):
        raise RuntimeError("the three Gemini preflight calls did not prove BYOK routing")
    if not all(result["call"]["provider"] == "Google AI Studio" for result in byok):
        raise RuntimeError("a Gemini preflight call did not route through Google AI Studio")
    record = {
        "completed_at": now_iso(),
        "status": "passed",
        "calls": results,
    }
    atomic_write(byok_path, pretty_json({"calls": byok}))
    atomic_write(preflight_path, pretty_json(record))
    totals = await ledger.totals()
    print(
        f"preflight passed: {len(results)} calls, OpenRouter ${totals.openrouter:.4f}, "
        f"Gemini upstream ${totals.gemini_upstream:.4f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch-label",
        help=(
            "prefix every preflight run ID with a fresh retry label; omit to preserve "
            "the original run IDs"
        ),
    )
    parser.add_argument(
        "--cohort",
        help="write cohort-specific evidence such as preflight_v5.json without replacing history",
    )
    args = parser.parse_args()
    asyncio.run(execute(args.batch_label, cohort=args.cohort))


if __name__ == "__main__":
    main()
