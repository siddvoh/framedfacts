from __future__ import annotations

import argparse
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

from audit import run_audit
from config import (
    DATA,
    OPENROUTER_LIMIT_WITH_KIMI,
    OPENROUTER_LIMIT_WITHOUT_KIMI,
    RUNS_DIR,
    SCHEDULE_PATH,
    SYSTEMS,
    WORKING,
    atomic_write,
    canonical_json,
    load_claims,
    load_json,
    now_iso,
    pretty_json,
    sha256_bytes,
)
from openrouter import DB_PATH
from run_experiments import authenticated_trim_state, observed_attempted_run_ids
from schedule import (
    ACTIVE_PILOT_ARMS,
    DIAGNOSTIC_PILOT_ARMS,
    TRIM_PROFILE_NONE,
    TRIM_PROFILE_STEP_1,
    build_schedule,
    pilot_judge_ids,
    schedule_pilot_generation,
)


GATE_RECORD_PATH = DATA / "pilot_gate.json"
MANUAL_REVIEW_PATH = WORKING / "pilot_v7_manual_review.json"
RESUME_EVIDENCE_PATHS = (
    WORKING / "pilot_v7_resume_before.json",
    WORKING / "pilot_v7_resume_interrupted.json",
    WORKING / "pilot_v7_resume_after.json",
)
AGENT_SUCCESS_STATUSES = {"agent_complete", "judged"}

ESTIMATED_RUN_COST = {
    "sonnet5": 69.0 / 552,
    "terra": 71.0 / 552,
    "glm": 30.0 / 552,
    "gemini": 0.0,
    "kimi": 26.0 / 144,
}
COST_TOLERANCE = 0.30


def pilot_manifests(rows: list[dict]) -> dict[str, dict]:
    manifests = {}
    for row in rows:
        if row.get("arm") not in ACTIVE_PILOT_ARMS:
            continue
        path = RUNS_DIR / f"{row['run_id']}.json"
        if path.exists():
            manifests[str(row["run_id"])] = load_json(path)
    return manifests


def run_costs() -> dict[str, float]:
    connection = sqlite3.connect(DB_PATH)
    try:
        rows = connection.execute(
            "SELECT run_id, sum(openrouter_cost) FROM calls GROUP BY run_id"
        ).fetchall()
    finally:
        connection.close()
    return {run_id: float(cost or 0) for run_id, cost in rows}


def total_spend() -> float:
    connection = sqlite3.connect(DB_PATH)
    try:
        value = connection.execute("SELECT sum(openrouter_cost) FROM calls").fetchone()[0]
    finally:
        connection.close()
    return float(value or 0)


def query_label_value(item: object) -> str:
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        return str(item.get("label") or "")
    return ""


def cost_projection(
    rows: list[dict], manifests: dict[str, dict], costs: dict[str, float]
) -> dict:
    per_system: dict[str, list[float]] = defaultdict(list)
    for run_id, manifest in manifests.items():
        if manifest.get("status") in AGENT_SUCCESS_STATUSES:
            per_system[str(manifest.get("system"))].append(costs.get(run_id, 0.0))
    means = {
        system: (sum(values) / len(values) if values else None)
        for system, values in per_system.items()
    }
    remaining: dict[str, int] = defaultdict(int)
    sealed = set(DIAGNOSTIC_PILOT_ARMS) | set(ACTIVE_PILOT_ARMS)
    for row in rows:
        if row.get("arm") not in sealed:
            remaining[str(row.get("system"))] += 1
    spent = total_spend()
    projected = spent
    missing = []
    for system, count in sorted(remaining.items()):
        mean = means.get(system)
        if mean is None:
            missing.append(system)
            continue
        projected += mean * count
    within_band = {}
    for system, mean in means.items():
        estimate = ESTIMATED_RUN_COST.get(system)
        if mean is None or estimate is None:
            continue
        if estimate == 0.0:
            within_band[system] = mean == 0.0
        else:
            within_band[system] = mean <= estimate * (1 + COST_TOLERANCE)
    return {
        "mean_cost_per_run": means,
        "remaining_runs": dict(remaining),
        "spent": round(spent, 10),
        "projected_total": round(projected, 10),
        "systems_without_cost_data": missing,
        "within_estimate_band": within_band,
    }


def evaluate_gates() -> dict:
    rows = load_json(SCHEDULE_PATH)
    claims = load_claims()
    manifests = pilot_manifests(rows)
    costs = run_costs()
    gates: dict[str, dict] = {}

    def gate(name: str, passed: bool, detail: str) -> None:
        gates[name] = {"passed": bool(passed), "detail": detail}

    preflight_path = DATA / "preflight_v5.json"
    preflight = load_json(preflight_path) if preflight_path.exists() else None
    byok_ok = isinstance(preflight, dict) and preflight.get("status") == "passed"
    gate("byok_verified", byok_ok, "v5 preflight record exists and passed")

    parity_ok = False
    if isinstance(preflight, dict):
        calls = preflight.get("calls") or []
        models = {call.get("requested_model") for call in calls}
        parity_ok = set(SYSTEMS.values()) <= models and all(
            call.get("reasoning_present") is True for call in calls
        )
    gate(
        "parity_and_reasoning",
        parity_ok,
        "all four systems accepted the unified config with reasoning evidence",
    )

    audit_result = run_audit("pilot")
    manual = load_json(MANUAL_REVIEW_PATH) if MANUAL_REVIEW_PATH.exists() else None
    manual_ok = (
        isinstance(manual, dict)
        and manual.get("confirmed") is True
        and isinstance(manual.get("run_ids"), list)
        and len(manual["run_ids"]) >= 3
        and all(str(run_id) in manifests for run_id in manual["run_ids"])
    )
    gate(
        "audit_green_and_manual_review",
        audit_result["status"] == "green" and manual_ok,
        f"pilot audit is {audit_result['status']} with "
        f"{len(audit_result['errors'])} errors; manual review of three raw files "
        f"{'recorded' if manual_ok else 'missing'}",
    )

    live = [
        manifest
        for manifest in manifests.values()
        if manifest.get("status") in AGENT_SUCCESS_STATUSES
    ]
    search_counts = [int(manifest.get("search_count") or 0) for manifest in live]
    cited = [manifest for manifest in live if manifest.get("citations")]
    mean_searches = sum(search_counts) / len(search_counts) if search_counts else 0.0
    citation_share = len(cited) / len(live) if live else 0.0
    gate(
        "agents_search_and_cite",
        bool(live) and mean_searches >= 4.0 and citation_share >= 0.9,
        f"{len(live)} successful runs, mean {mean_searches:.2f} searches, "
        f"{len(cited)} with extractable citations ({citation_share:.0%})",
    )

    sample_ids = pilot_judge_ids(rows)
    sampled_failed = [
        run_id
        for run_id in sample_ids
        if (manifests.get(run_id) or {}).get("status") == "terminal_failed"
    ]
    recorded_invalid = [
        run_id
        for run_id in sample_ids
        if run_id not in sampled_failed
        and (manifests.get(run_id) or {}).get("judge_status") == "terminal_invalid"
    ]
    judged = [
        manifests.get(run_id)
        for run_id in sample_ids
        if run_id not in sampled_failed and run_id not in recorded_invalid
    ]
    judged_ok = bool(judged) and all(
        manifest is not None and manifest.get("status") == "judged" for manifest in judged
    )
    gate(
        "judge_outputs_parse",
        judged_ok,
        f"{sum(1 for m in judged if m and m.get('status') == 'judged')}/"
        f"{len(judged)} judgeable sampled runs have parse-valid judge output; "
        f"{len(recorded_invalid)} recorded terminal-invalid, "
        f"{len(sampled_failed)} failed terminally",
    )

    projection = cost_projection(rows, manifests, costs)
    include_kimi = (
        not projection["systems_without_cost_data"]
        and projection["projected_total"] <= OPENROUTER_LIMIT_WITH_KIMI
    )
    cap = OPENROUTER_LIMIT_WITH_KIMI if include_kimi else OPENROUTER_LIMIT_WITHOUT_KIMI
    band_ok = all(projection["within_estimate_band"].values()) and bool(
        projection["within_estimate_band"]
    )
    gate(
        "cost_within_budget",
        band_ok and projection["projected_total"] <= cap,
        f"projected total ${projection['projected_total']:.2f} against cap ${cap:.0f}",
    )

    resume_ok = all(path.exists() for path in RESUME_EVIDENCE_PATHS)
    gate(
        "resume_safety",
        resume_ok,
        "kill and restart evidence files are "
        + ("present" if resume_ok else "missing"),
    )

    stanced = 0
    labeled = 0
    for manifest in live:
        if manifest.get("framing") not in ("believer", "skeptic"):
            continue
        for item in manifest.get("query_labels") or []:
            labeled += 1
            if query_label_value(item) == "stance_congruent":
                stanced += 1
    gate(
        "directional_sanity",
        labeled > 0 and stanced > 0,
        f"{stanced}/{labeled} believer and skeptic query labels are stance-congruent"
        + ("" if stanced else "; zero mirroring needs a human decision"),
    )

    initial_profile = TRIM_PROFILE_STEP_1 if include_kimi else TRIM_PROFILE_NONE
    generation = schedule_pilot_generation(rows)
    initial_rows = build_schedule(
        claims,
        include_kimi=include_kimi,
        trim_profile=initial_profile,
        pilot_generation=generation,
    )
    initial_hash = sha256_bytes(canonical_json(initial_rows).encode("utf-8"))
    state, _records = authenticated_trim_state(
        rows, claims, observed_attempted_run_ids()
    )

    return {
        "evaluated_at": now_iso(),
        "gates": gates,
        "all_passed": all(entry["passed"] for entry in gates.values()),
        "cost": projection,
        "decision_include_kimi": include_kimi,
        "initial_trim_profile": initial_profile,
        "initial_full_schedule_sha256": initial_hash,
        "include_kimi": state["include_kimi"],
        "trim_profile": state["profile"],
        "full_schedule_sha256": state["schedule_sha256"],
        "trim_record_count": state["record_count"],
        "last_trim_record_sha256": state["last_record_sha256"],
    }


def require_pilot_gate(path: Path = GATE_RECORD_PATH) -> dict:
    if not path.exists():
        raise RuntimeError("full launch requires a passed pilot gate record")
    record = load_json(path)
    if not isinstance(record, dict) or record.get("all_passed") is not True:
        raise RuntimeError("the pilot gate record did not pass every gate")
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()
    record = evaluate_gates()
    for name, entry in record["gates"].items():
        marker = "pass" if entry["passed"] else "FAIL"
        print(f"{marker}  {name}: {entry['detail']}")
    print(
        f"pilot gate: {'passed' if record['all_passed'] else 'failed'}, "
        f"Kimi decision: {'include' if record['decision_include_kimi'] else 'exclude'}"
    )
    if record["all_passed"] and not args.no_write:
        atomic_write(GATE_RECORD_PATH, pretty_json(record))
        print(f"wrote {GATE_RECORD_PATH}")
    if not record["all_passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
