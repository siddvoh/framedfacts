"""Second-round arms added after the first review round.

The registered schedule in data/schedule.json is hash-authenticated against the
trim records and cannot grow. These arms are recorded in their own additive
schedule file so the sealed matrix stays byte-identical.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from config import (
    ARCHIVE,
    DATA,
    RUNS_DIR,
    SYSTEMS,
    atomic_write,
    canonical_json,
    load_claims,
    load_json,
    pretty_json,
)


PHASE2_SCHEDULE_PATH = DATA / "schedule_phase2.json"
CROSS_REPLAY_CONTEXT_DIR = ARCHIVE / "cross_replay_context"

NEUTRAL_REPEAT_ARM = "neutral_repeat"
SKEPTIC_MITIGATION_ARM = "skeptic_mitigation"
CROSS_REPLAY_ARM = "cross_replay"
SCHOLARLY_REPAIRED_ARM = "scholarly_repaired"
SCHOLARLY_REPAIRED_V2_ARM = "scholarly_repaired_v2"
FULL_PAGE_ARM = "full_page"

PHASE2_ARMS = (
    NEUTRAL_REPEAT_ARM,
    SKEPTIC_MITIGATION_ARM,
    CROSS_REPLAY_ARM,
    SCHOLARLY_REPAIRED_ARM,
    SCHOLARLY_REPAIRED_V2_ARM,
    FULL_PAGE_ARM,
)

# A1 restores the neutral noise floor for the one system the trim ladder cut.
NEUTRAL_REPEAT_SYSTEM = "glm"
NEUTRAL_REPEAT_INDEX = 2

CROSS_REPLAY_DIRECTIONS = (("believer", "skeptic"), ("skeptic", "believer"))

FRAMINGS = ("neutral", "believer", "skeptic")


def phase2_run_id(
    arm: str,
    claim_id: str,
    framing: str,
    system: str,
    repeat: int,
    evidence_framing: str | None = None,
) -> str:
    if evidence_framing is None:
        return f"{arm}_{claim_id}_{framing}_{system}_{repeat}"
    return f"{arm}_{claim_id}_{framing}_on_{evidence_framing}_{system}_{repeat}"


def source_run_id(claim_id: str, framing: str, system: str) -> str:
    return f"core_{claim_id}_{framing}_{system}_1"


def cross_replay_context_path(claim_id: str, system: str, framing: str) -> Path:
    return CROSS_REPLAY_CONTEXT_DIR / f"{claim_id}_{system}_{framing}.json"


def usable_source(run_id: str) -> dict | None:
    path = RUNS_DIR / f"{run_id}.json"
    if not path.exists():
        return None
    manifest = load_json(path)
    if not isinstance(manifest, dict):
        return None
    if manifest.get("status") not in {"agent_complete", "judged", "analysis_ready"}:
        return None
    searches = manifest.get("searches")
    if not isinstance(searches, list) or not searches:
        return None
    if not all(
        isinstance(record, dict) and isinstance(record.get("results"), list)
        for record in searches
    ):
        return None
    return manifest


def build_cross_replay_contexts(claims: list[dict]) -> list[dict]:
    """Freeze each core believer and skeptic run's evidence for the swapped arm."""

    written = []
    for claim in claims:
        claim_id = str(claim["id"])
        for system in SYSTEMS:
            for framing in ("believer", "skeptic"):
                run_id = source_run_id(claim_id, framing, system)
                manifest = usable_source(run_id)
                if manifest is None:
                    continue
                value = {
                    "source_run_id": run_id,
                    "claim_id": claim_id,
                    "system": system,
                    "source_framing": framing,
                    "searches": manifest["searches"],
                }
                path = cross_replay_context_path(claim_id, system, framing)
                if path.exists():
                    existing = load_json(path)
                    if existing != value:
                        raise RuntimeError(
                            f"frozen cross-replay context changed for {run_id}"
                        )
                    continue
                atomic_write(path, pretty_json(value))
                written.append({"path": str(path), "source_run_id": run_id})
    return written


def add_row(
    rows: list[dict],
    arm: str,
    claim_id: str,
    framing: str,
    system: str,
    repeat: int,
    search_backend: str | None,
    evidence_framing: str | None = None,
) -> None:
    row = {
        "run_id": phase2_run_id(arm, claim_id, framing, system, repeat, evidence_framing),
        "arm": arm,
        "claim_id": claim_id,
        "framing": framing,
        "system": system,
        "model": SYSTEMS[system],
        "repeat": repeat,
        "search_backend": search_backend,
    }
    if evidence_framing is not None:
        row["evidence_source_run_id"] = source_run_id(claim_id, evidence_framing, system)
        row["evidence_framing"] = evidence_framing
    rows.append(row)


def build_phase2_schedule(
    claims: list[dict],
    high_movement_ids: list[str],
    scholarly_claim_ids: list[str],
    arms: tuple[str, ...] = PHASE2_ARMS,
) -> list[dict]:
    claim_ids = [str(claim["id"]) for claim in claims]
    known = set(claim_ids)
    for claim_id in list(high_movement_ids) + list(scholarly_claim_ids):
        if claim_id not in known:
            raise ValueError(f"unknown claim id: {claim_id}")
    rows: list[dict] = []

    if NEUTRAL_REPEAT_ARM in arms:
        for claim_id in claim_ids:
            add_row(
                rows,
                NEUTRAL_REPEAT_ARM,
                claim_id,
                "neutral",
                NEUTRAL_REPEAT_SYSTEM,
                NEUTRAL_REPEAT_INDEX,
                "web",
            )

    if SKEPTIC_MITIGATION_ARM in arms:
        for claim_id in claim_ids:
            for system in SYSTEMS:
                add_row(
                    rows, SKEPTIC_MITIGATION_ARM, claim_id, "skeptic", system, 1, "web"
                )

    if CROSS_REPLAY_ARM in arms:
        for claim_id in claim_ids:
            for system in SYSTEMS:
                for framing, evidence_framing in CROSS_REPLAY_DIRECTIONS:
                    if usable_source(source_run_id(claim_id, evidence_framing, system)) is None:
                        continue
                    if usable_source(source_run_id(claim_id, framing, system)) is None:
                        continue
                    add_row(
                        rows,
                        CROSS_REPLAY_ARM,
                        claim_id,
                        framing,
                        system,
                        1,
                        None,
                        evidence_framing,
                    )

    if FULL_PAGE_ARM in arms:
        for claim_id in high_movement_ids:
            for framing in FRAMINGS:
                for system in SYSTEMS:
                    add_row(rows, FULL_PAGE_ARM, claim_id, framing, system, 1, "web_pages")

    for arm in (SCHOLARLY_REPAIRED_ARM, SCHOLARLY_REPAIRED_V2_ARM):
        if arm not in arms:
            continue
        for claim_id in scholarly_claim_ids:
            for framing in FRAMINGS:
                for system in SYSTEMS:
                    add_row(rows, arm, claim_id, framing, system, 1, "scholarly")

    rows.sort(key=lambda row: (row["claim_id"], row["arm"], row["run_id"]))
    validate_phase2_schedule(rows)
    return rows


def validate_phase2_schedule(rows: list[dict]) -> None:
    seen = set()
    for row in rows:
        if row["arm"] not in PHASE2_ARMS:
            raise ValueError(f"unknown phase-2 arm: {row['arm']}")
        if row["run_id"] in seen:
            raise ValueError(f"duplicate phase-2 run ID: {row['run_id']}")
        seen.add(row["run_id"])
        if row["arm"] == CROSS_REPLAY_ARM:
            if row.get("search_backend") is not None:
                raise ValueError("cross-replay rows must not use a search backend")
            if not row.get("evidence_source_run_id"):
                raise ValueError("cross-replay rows require an evidence source")
        elif row.get("search_backend") is None:
            raise ValueError(f"{row['arm']} rows require a search backend")


def schedule_sha256(rows: list[dict]) -> str:
    return hashlib.sha256(canonical_json(rows).encode("utf-8")).hexdigest()


def load_phase2_schedule(path: Path = PHASE2_SCHEDULE_PATH) -> list[dict]:
    if not path.exists():
        return []
    rows = load_json(path)
    if not isinstance(rows, list):
        raise RuntimeError("phase-2 schedule is not a list")
    validate_phase2_schedule(rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--high-movement", nargs="+", required=True)
    parser.add_argument("--scholarly-claims", nargs="+", required=True)
    parser.add_argument("--arms", nargs="+", choices=PHASE2_ARMS, default=list(PHASE2_ARMS))
    args = parser.parse_args()
    claims = load_claims()
    contexts = build_cross_replay_contexts(claims)
    rows = build_phase2_schedule(
        claims, args.high_movement, args.scholarly_claims, tuple(args.arms)
    )
    if PHASE2_SCHEDULE_PATH.exists():
        existing = load_phase2_schedule()
        by_id = {row["run_id"]: row for row in rows}
        changed = [row["run_id"] for row in existing if by_id.get(row["run_id"]) != row]
        if changed:
            raise RuntimeError(
                f"phase-2 rebuild would change {len(changed)} existing rows; "
                f"first={changed[:3]}"
            )
    atomic_write(PHASE2_SCHEDULE_PATH, pretty_json(rows))
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["arm"]] = counts.get(row["arm"], 0) + 1
    print(
        json.dumps(
            {
                "runs": len(rows),
                "arms": counts,
                "contexts_written": len(contexts),
                "schedule_sha256": schedule_sha256(rows),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
