from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from config import (
    DATA,
    KIMI_MODEL,
    SCHEDULE_PATH,
    SYSTEMS,
    atomic_write,
    canonical_json,
    load_claims,
    now_iso,
    pretty_json,
)


FRAMINGS = ("neutral", "believer", "skeptic")
PILOT_JUDGE_SAMPLE_SIZE = 10
PILOT_GENERATION_V1 = 1
PILOT_GENERATION_V2 = 2
PILOT_GENERATION_V3 = 3
PILOT_GENERATION_V4 = 4
PILOT_GENERATION_V5 = 5
PILOT_GENERATION_V6 = 6
PILOT_GENERATION_V7 = 7
PILOT_GENERATIONS = (
    PILOT_GENERATION_V1,
    PILOT_GENERATION_V2,
    PILOT_GENERATION_V3,
    PILOT_GENERATION_V4,
    PILOT_GENERATION_V5,
    PILOT_GENERATION_V6,
    PILOT_GENERATION_V7,
)
PILOT_V1_ARMS = frozenset({"pilot", "kimi_pilot"})
PILOT_V2_ARMS = frozenset({"pilot_v2", "kimi_pilot_v2"})
PILOT_V3_ARMS = frozenset({"pilot_v3", "kimi_pilot_v3"})
PILOT_V4_ARMS = frozenset({"pilot_v4", "kimi_pilot_v4"})
PILOT_V5_ARMS = frozenset({"pilot_v5", "kimi_pilot_v5"})
PILOT_V6_ARMS = frozenset({"pilot_v6", "kimi_pilot_v6"})
PILOT_V7_ARMS = frozenset({"pilot_v7", "kimi_pilot_v7"})
ACTIVE_PILOT_ARMS = PILOT_V7_ARMS
LEGACY_V1_V2_ARMS = PILOT_V1_ARMS | PILOT_V2_ARMS
DIAGNOSTIC_PILOT_ARMS = (
    LEGACY_V1_V2_ARMS | PILOT_V3_ARMS | PILOT_V4_ARMS | PILOT_V5_ARMS | PILOT_V6_ARMS
)
ALL_PILOT_ARMS = DIAGNOSTIC_PILOT_ARMS | PILOT_V7_ARMS
TRIM_PROFILE_NONE = "none"
TRIM_PROFILE_STEP_1 = "kimi_step1_glm_core_repeat2"
TRIM_PROFILE_STEP_2 = "step2_drop_kimi"
TRIM_PROFILE_STEP_3 = "step3_replay_24_claims"
TRIM_PROFILE_STEP_4 = "step4_paraphrase_4_claims"
TRIM_PROFILES = (
    TRIM_PROFILE_NONE,
    TRIM_PROFILE_STEP_1,
    TRIM_PROFILE_STEP_2,
    TRIM_PROFILE_STEP_3,
    TRIM_PROFILE_STEP_4,
)
TRIM_PROFILE_STEPS = {
    profile: step for step, profile in enumerate(TRIM_PROFILES)
}


def trim_profile_step(profile: str) -> int:
    try:
        return TRIM_PROFILE_STEPS[profile]
    except KeyError as error:
        raise ValueError(f"unknown trim profile: {profile}") from error


def structural_trim_profile(profile: str) -> str:
    """Return the profile identifiable from schedule rows alone.

    Step two removes Kimi after step one. A step-one schedule that already
    excludes Kimi has the same rows, so the authenticated migration record is
    what distinguishes those two states.
    """

    if profile == TRIM_PROFILE_STEP_2:
        return TRIM_PROFILE_STEP_1
    trim_profile_step(profile)
    return profile


def run_id(arm: str, claim_id: str, framing: str, system: str, repeat: int) -> str:
    return f"{arm}_{claim_id}_{framing}_{system}_{repeat}"


def add_run(
    rows: list[dict],
    arm: str,
    claim_id: str,
    framing: str,
    system: str,
    model: str,
    repeat: int,
    search_backend: str | None,
) -> None:
    rows.append(
        {
            "run_id": run_id(arm, claim_id, framing, system, repeat),
            "arm": arm,
            "claim_id": claim_id,
            "framing": framing,
            "system": system,
            "model": model,
            "repeat": repeat,
            "search_backend": search_backend,
        }
    )


def selected_ids(claims: list[dict], counts: tuple[int, int, int]) -> list[str]:
    by_tier = {
        tier: [claim["id"] for claim in claims if claim["tier"] == tier]
        for tier in ("settled_false", "settled_true", "contested")
    }
    return (
        by_tier["settled_false"][: counts[0]]
        + by_tier["settled_true"][: counts[1]]
        + by_tier["contested"][: counts[2]]
    )


def pilot_arms(cohort: str) -> tuple[str, str]:
    if cohort == "v1":
        return "pilot", "kimi_pilot"
    if cohort == "v2":
        return "pilot_v2", "kimi_pilot_v2"
    if cohort == "v3":
        return "pilot_v3", "kimi_pilot_v3"
    if cohort == "v4":
        return "pilot_v4", "kimi_pilot_v4"
    if cohort == "v5":
        return "pilot_v5", "kimi_pilot_v5"
    if cohort == "v6":
        return "pilot_v6", "kimi_pilot_v6"
    if cohort == "v7":
        return "pilot_v7", "kimi_pilot_v7"
    raise ValueError(f"unknown pilot cohort: {cohort!r}")


def pilot_judge_ids(rows: list[dict], cohort: str = "v7") -> list[str]:
    standard_arm, _ = pilot_arms(cohort)
    grouped: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        if row.get("arm") != standard_arm:
            continue
        key = (str(row.get("claim_id") or ""), str(row.get("framing") or ""))
        system = str(row.get("system") or "")
        grouped.setdefault(key, {})[system] = str(row.get("run_id") or "")
    if len(grouped) != PILOT_JUDGE_SAMPLE_SIZE:
        raise ValueError(
            f"{cohort} pilot schedule has {len(grouped)} claim/framing cells, "
            f"expected {PILOT_JUDGE_SAMPLE_SIZE}"
        )
    system_order = tuple(SYSTEMS)
    selected = []
    for index, key in enumerate(sorted(grouped)):
        candidates = grouped[key]
        if set(candidates) != set(system_order) or not all(candidates.values()):
            raise ValueError(f"pilot cell {key!r} does not contain all four systems")
        selected.append(candidates[system_order[index % len(system_order)]])
    return selected


def schedule_pilot_generation(rows: list[dict]) -> int:
    counts = {
        arm: sum(row.get("arm") == arm for row in rows)
        for arm in ALL_PILOT_ARMS
    }
    v1 = (counts["pilot"], counts["kimi_pilot"])
    v2 = (counts["pilot_v2"], counts["kimi_pilot_v2"])
    v3 = (counts["pilot_v3"], counts["kimi_pilot_v3"])
    v4 = (counts["pilot_v4"], counts["kimi_pilot_v4"])
    v5 = (counts["pilot_v5"], counts["kimi_pilot_v5"])
    v6 = (counts["pilot_v6"], counts["kimi_pilot_v6"])
    v7 = (counts["pilot_v7"], counts["kimi_pilot_v7"])
    if (
        v1 == (40, 6)
        and v2 == (0, 0)
        and v3 == (0, 0)
        and v4 == (0, 0)
        and v5 == (0, 0)
    ):
        return PILOT_GENERATION_V1
    if (
        v1 == (40, 6)
        and v2 == (40, 6)
        and v3 == (0, 0)
        and v4 == (0, 0)
        and v5 == (0, 0)
    ):
        return PILOT_GENERATION_V2
    if (
        v1 == (40, 6)
        and v2 == (40, 6)
        and v3 == (40, 6)
        and v4 == (0, 0)
        and v5 == (0, 0)
    ):
        return PILOT_GENERATION_V3
    if (
        v1 == (40, 6)
        and v2 == (40, 6)
        and v3 == (40, 6)
        and v4 == (40, 6)
        and v5 == (0, 0)
    ):
        return PILOT_GENERATION_V4
    if all(value == (40, 6) for value in (v1, v2, v3, v4, v5)) and v6 == (0, 0) and v7 == (0, 0):
        return PILOT_GENERATION_V5
    if all(value == (40, 6) for value in (v1, v2, v3, v4, v5, v6)) and v7 == (0, 0):
        return PILOT_GENERATION_V6
    if all(value == (40, 6) for value in (v1, v2, v3, v4, v5, v6, v7)):
        return PILOT_GENERATION_V7
    raise ValueError(
        "invalid pilot cohort counts: "
        f"v1={v1}, v2={v2}, v3={v3}, v4={v4}, v5={v5}, v6={v6}, v7={v7}"
    )


def add_pilot_cohort(
    rows: list[dict],
    pilot_claims: list[str],
    pilot_cells: list[tuple[str, str]],
    *,
    cohort: str,
) -> None:
    standard_arm, kimi_arm = pilot_arms(cohort)
    for claim_id, framing in pilot_cells:
        for system, model in SYSTEMS.items():
            add_run(rows, standard_arm, claim_id, framing, system, model, 1, "web")
    kimi_framings = ("neutral", "believer", "skeptic", "neutral", "believer", "skeptic")
    for claim_id, framing in zip(pilot_claims, kimi_framings):
        add_run(rows, kimi_arm, claim_id, framing, "kimi", KIMI_MODEL, 1, "web")


def build_schedule(
    claims: list[dict],
    include_kimi: bool = False,
    trim_profile: str = TRIM_PROFILE_NONE,
    pilot_generation: int = PILOT_GENERATION_V7,
) -> list[dict]:
    if trim_profile not in TRIM_PROFILES:
        raise ValueError(f"unknown trim profile: {trim_profile}")
    if include_kimi and trim_profile != TRIM_PROFILE_STEP_1:
        raise ValueError(
            "full Kimi inclusion requires trim profile "
            f"{TRIM_PROFILE_STEP_1!r}"
        )
    if pilot_generation not in PILOT_GENERATIONS:
        raise ValueError(f"unknown pilot generation: {pilot_generation}")
    rows: list[dict] = []
    claim_ids = [claim["id"] for claim in claims]

    pilot_claims = selected_ids(claims, (2, 2, 2))
    pilot_cells = [
        (pilot_claims[0], "neutral"),
        (pilot_claims[0], "believer"),
        (pilot_claims[1], "skeptic"),
        (pilot_claims[2], "neutral"),
        (pilot_claims[2], "skeptic"),
        (pilot_claims[3], "believer"),
        (pilot_claims[4], "neutral"),
        (pilot_claims[4], "believer"),
        (pilot_claims[5], "neutral"),
        (pilot_claims[5], "skeptic"),
    ]
    add_pilot_cohort(rows, pilot_claims, pilot_cells, cohort="v1")
    if pilot_generation >= PILOT_GENERATION_V2:
        add_pilot_cohort(rows, pilot_claims, pilot_cells, cohort="v2")
    if pilot_generation >= PILOT_GENERATION_V3:
        add_pilot_cohort(rows, pilot_claims, pilot_cells, cohort="v3")
    if pilot_generation >= PILOT_GENERATION_V4:
        add_pilot_cohort(rows, pilot_claims, pilot_cells, cohort="v4")
    if pilot_generation >= PILOT_GENERATION_V5:
        add_pilot_cohort(rows, pilot_claims, pilot_cells, cohort="v5")
    if pilot_generation >= PILOT_GENERATION_V6:
        add_pilot_cohort(rows, pilot_claims, pilot_cells, cohort="v6")
    if pilot_generation >= PILOT_GENERATION_V7:
        add_pilot_cohort(rows, pilot_claims, pilot_cells, cohort="v7")

    for claim_id in claim_ids:
        for framing in FRAMINGS:
            for system, model in SYSTEMS.items():
                for repeat in (1, 2):
                    if (
                        trim_profile_step(trim_profile) >= 1
                        and system == "glm"
                        and repeat == 2
                    ):
                        continue
                    add_run(rows, "core", claim_id, framing, system, model, repeat, "web")

    for claim_id in selected_ids(claims, (8, 4, 4)):
        for system, model in SYSTEMS.items():
            for repeat in (1, 2):
                add_run(rows, "attribution", claim_id, "attribution", system, model, repeat, "web")

    paraphrase_counts = (
        (2, 1, 1) if trim_profile_step(trim_profile) >= 4 else (3, 2, 3)
    )
    for claim_id in selected_ids(claims, paraphrase_counts):
        for framing in ("believer_paraphrase", "skeptic_paraphrase"):
            for system, model in SYSTEMS.items():
                add_run(rows, "paraphrase", claim_id, framing, system, model, 1, "web")

    for claim_id in claim_ids:
        for system, model in SYSTEMS.items():
            add_run(rows, "mitigation", claim_id, "believer", system, model, 1, "web")

    for claim_id in selected_ids(claims, (3, 2, 3)):
        for framing in FRAMINGS:
            for system, model in SYSTEMS.items():
                add_run(rows, "scholarly", claim_id, framing, system, model, 1, "scholarly")

    replay_claim_ids = (
        selected_ids(claims, (8, 8, 8))
        if trim_profile_step(trim_profile) >= 3
        else claim_ids
    )
    for claim_id in replay_claim_ids:
        for framing in FRAMINGS:
            for system, model in SYSTEMS.items():
                add_run(rows, "replay", claim_id, framing, system, model, 1, None)

    if include_kimi:
        for claim_id in claim_ids:
            for framing in FRAMINGS:
                add_run(rows, "kimi", claim_id, framing, "kimi", KIMI_MODEL, 1, "web")

    rows.sort(key=lambda row: (row["claim_id"], row["arm"], row["framing"], row["system"], row["repeat"]))
    validate_schedule(rows, include_kimi, trim_profile, pilot_generation)
    return rows


def schedule_trim_profile(rows: list[dict]) -> str:
    glm_repeat_2 = [
        row
        for row in rows
        if row.get("arm") == "core"
        and row.get("system") == "glm"
        and row.get("repeat") == 2
    ]
    if len(glm_repeat_2) not in {0, 144}:
        raise ValueError(
            "schedule partially applies trim step 1: "
            f"found {len(glm_repeat_2)}/144 GLM core repeat-2 rows"
        )
    replay = [row for row in rows if row.get("arm") == "replay"]
    paraphrase = [row for row in rows if row.get("arm") == "paraphrase"]
    shape = (len(glm_repeat_2), len(replay), len(paraphrase))
    profiles = {
        (144, 576, 64): TRIM_PROFILE_NONE,
        (0, 576, 64): TRIM_PROFILE_STEP_1,
        (0, 288, 64): TRIM_PROFILE_STEP_3,
        (0, 288, 32): TRIM_PROFILE_STEP_4,
    }
    if shape not in profiles:
        raise ValueError(
            "schedule does not match a complete trim-ladder shape: "
            f"glm_core_repeat2={shape[0]}, replay={shape[1]}, "
            f"paraphrase={shape[2]}"
        )
    return profiles[shape]


def migrate_schedule_to_pilot_v2(rows: list[dict], claims: list[dict]) -> list[dict]:
    if schedule_pilot_generation(rows) != PILOT_GENERATION_V1:
        raise ValueError("pilot-v2 migration requires a v1-only source schedule")
    trim_profile = schedule_trim_profile(rows)
    include_kimi = any(row.get("arm") == "kimi" for row in rows)
    validate_schedule(
        rows,
        include_kimi,
        trim_profile,
        PILOT_GENERATION_V1,
    )
    migrated = build_schedule(
        claims,
        include_kimi,
        trim_profile,
        PILOT_GENERATION_V2,
    )
    migrated_by_id = {row["run_id"]: row for row in migrated}
    changed = [row["run_id"] for row in rows if migrated_by_id.get(row["run_id"]) != row]
    if changed:
        raise ValueError(
            f"pilot-v2 migration would change {len(changed)} existing rows; first={changed[:3]}"
        )
    if len(migrated) != len(rows) + 46:
        raise ValueError("pilot-v2 migration did not add exactly 46 rows")
    return migrated


def migrate_schedule_to_pilot_v3(rows: list[dict], claims: list[dict]) -> list[dict]:
    if schedule_pilot_generation(rows) != PILOT_GENERATION_V2:
        raise ValueError("pilot-v3 migration requires a complete v2 source schedule")
    trim_profile = schedule_trim_profile(rows)
    include_kimi = any(row.get("arm") == "kimi" for row in rows)
    validate_schedule(
        rows,
        include_kimi,
        trim_profile,
        PILOT_GENERATION_V2,
    )
    migrated = build_schedule(
        claims,
        include_kimi,
        trim_profile,
        PILOT_GENERATION_V3,
    )
    migrated_by_id = {row["run_id"]: row for row in migrated}
    changed = [row["run_id"] for row in rows if migrated_by_id.get(row["run_id"]) != row]
    if changed:
        raise ValueError(
            f"pilot-v3 migration would change {len(changed)} existing rows; first={changed[:3]}"
        )
    if len(migrated) != len(rows) + 46:
        raise ValueError("pilot-v3 migration did not add exactly 46 rows")
    return migrated


def migrate_schedule_to_pilot_v4(rows: list[dict], claims: list[dict]) -> list[dict]:
    if schedule_pilot_generation(rows) != PILOT_GENERATION_V3:
        raise ValueError("pilot-v4 migration requires a complete v3 source schedule")
    trim_profile = schedule_trim_profile(rows)
    include_kimi = any(row.get("arm") == "kimi" for row in rows)
    validate_schedule(
        rows,
        include_kimi,
        trim_profile,
        PILOT_GENERATION_V3,
    )
    migrated = build_schedule(
        claims,
        include_kimi,
        trim_profile,
        PILOT_GENERATION_V4,
    )
    migrated_by_id = {row["run_id"]: row for row in migrated}
    changed = [row["run_id"] for row in rows if migrated_by_id.get(row["run_id"]) != row]
    if changed:
        raise ValueError(
            f"pilot-v4 migration would change {len(changed)} existing rows; first={changed[:3]}"
        )
    if len(migrated) != len(rows) + 46:
        raise ValueError("pilot-v4 migration did not add exactly 46 rows")
    return migrated


def migrate_schedule_to_pilot_v5(rows: list[dict], claims: list[dict]) -> list[dict]:
    if schedule_pilot_generation(rows) != PILOT_GENERATION_V4:
        raise ValueError("pilot-v5 migration requires a complete v4 source schedule")
    trim_profile = schedule_trim_profile(rows)
    include_kimi = any(row.get("arm") == "kimi" for row in rows)
    validate_schedule(
        rows,
        include_kimi,
        trim_profile,
        PILOT_GENERATION_V4,
    )
    migrated = build_schedule(
        claims,
        include_kimi,
        trim_profile,
        PILOT_GENERATION_V5,
    )
    migrated_by_id = {row["run_id"]: row for row in migrated}
    changed = [row["run_id"] for row in rows if migrated_by_id.get(row["run_id"]) != row]
    if changed:
        raise ValueError(
            f"pilot-v5 migration would change {len(changed)} existing rows; first={changed[:3]}"
        )
    if len(migrated) != len(rows) + 46:
        raise ValueError("pilot-v5 migration did not add exactly 46 rows")
    return migrated


def migrate_schedule_to_pilot_v6(rows: list[dict], claims: list[dict]) -> list[dict]:
    if schedule_pilot_generation(rows) != PILOT_GENERATION_V5:
        raise ValueError("pilot-v6 migration requires a complete v5 source schedule")
    trim_profile = schedule_trim_profile(rows)
    include_kimi = any(row.get("arm") == "kimi" for row in rows)
    validate_schedule(
        rows,
        include_kimi,
        trim_profile,
        PILOT_GENERATION_V5,
    )
    migrated = build_schedule(
        claims,
        include_kimi,
        trim_profile,
        PILOT_GENERATION_V6,
    )
    migrated_by_id = {row["run_id"]: row for row in migrated}
    changed = [row["run_id"] for row in rows if migrated_by_id.get(row["run_id"]) != row]
    if changed:
        raise ValueError(
            f"pilot-v6 migration would change {len(changed)} existing rows; first={changed[:3]}"
        )
    if len(migrated) != len(rows) + 46:
        raise ValueError("pilot-v6 migration did not add exactly 46 rows")
    return migrated


def migrate_schedule_to_pilot_v7(rows: list[dict], claims: list[dict]) -> list[dict]:
    if schedule_pilot_generation(rows) != PILOT_GENERATION_V6:
        raise ValueError("pilot-v7 migration requires a complete v6 source schedule")
    trim_profile = schedule_trim_profile(rows)
    include_kimi = any(row.get("arm") == "kimi" for row in rows)
    validate_schedule(
        rows,
        include_kimi,
        trim_profile,
        PILOT_GENERATION_V6,
    )
    migrated = build_schedule(
        claims,
        include_kimi,
        trim_profile,
        PILOT_GENERATION_V7,
    )
    migrated_by_id = {row["run_id"]: row for row in migrated}
    changed = [row["run_id"] for row in rows if migrated_by_id.get(row["run_id"]) != row]
    if changed:
        raise ValueError(
            f"pilot-v7 migration would change {len(changed)} existing rows; first={changed[:3]}"
        )
    if len(migrated) != len(rows) + 46:
        raise ValueError("pilot-v7 migration did not add exactly 46 rows")
    return migrated


def validate_schedule(
    rows: list[dict],
    include_kimi: bool,
    trim_profile: str = TRIM_PROFILE_NONE,
    pilot_generation: int = PILOT_GENERATION_V7,
) -> None:
    if trim_profile not in TRIM_PROFILES:
        raise ValueError(f"unknown trim profile: {trim_profile}")
    if pilot_generation not in PILOT_GENERATIONS:
        raise ValueError(f"unknown pilot generation: {pilot_generation}")
    counts: dict[str, int] = {}
    ids = set()
    for row in rows:
        counts[row["arm"]] = counts.get(row["arm"], 0) + 1
        if row["run_id"] in ids:
            raise ValueError(f"duplicate run ID: {row['run_id']}")
        ids.add(row["run_id"])
    step = trim_profile_step(trim_profile)
    expected = {
        "pilot": 40,
        "kimi_pilot": 6,
        "core": 1008 if step >= 1 else 1152,
        "attribution": 128,
        "paraphrase": 32 if step >= 4 else 64,
        "mitigation": 192,
        "scholarly": 96,
        "replay": 288 if step >= 3 else 576,
    }
    if pilot_generation >= PILOT_GENERATION_V2:
        expected.update({"pilot_v2": 40, "kimi_pilot_v2": 6})
    if pilot_generation >= PILOT_GENERATION_V3:
        expected.update({"pilot_v3": 40, "kimi_pilot_v3": 6})
    if pilot_generation >= PILOT_GENERATION_V4:
        expected.update({"pilot_v4": 40, "kimi_pilot_v4": 6})
    if pilot_generation >= PILOT_GENERATION_V5:
        expected.update({"pilot_v5": 40, "kimi_pilot_v5": 6})
    if pilot_generation >= PILOT_GENERATION_V6:
        expected.update({"pilot_v6": 40, "kimi_pilot_v6": 6})
    if pilot_generation >= PILOT_GENERATION_V7:
        expected.update({"pilot_v7": 40, "kimi_pilot_v7": 6})
    if include_kimi:
        if trim_profile != TRIM_PROFILE_STEP_1:
            raise ValueError(
                "full Kimi inclusion is valid only at trim-ladder step one"
            )
        expected["kimi"] = 144
    if counts != expected:
        raise ValueError(f"schedule count mismatch: {counts} != {expected}")
    if schedule_pilot_generation(rows) != pilot_generation:
        raise ValueError("schedule pilot generation differs from requested generation")
    inferred_profile = schedule_trim_profile(rows)
    expected_structural = structural_trim_profile(trim_profile)
    if inferred_profile != expected_structural:
        raise ValueError(
            "schedule trim profile mismatch: "
            f"{inferred_profile} != {expected_structural} "
            f"(declared {trim_profile})"
        )
    core = [row for row in rows if row["arm"] == "core"]
    core_counts = {
        (system, repeat): sum(
            row["system"] == system and row["repeat"] == repeat for row in core
        )
        for system in SYSTEMS
        for repeat in (1, 2)
    }
    expected_core_counts = {
        (system, repeat): (
            0
            if step >= 1
            and system == "glm"
            and repeat == 2
            else 144
        )
        for system in SYSTEMS
        for repeat in (1, 2)
    }
    if core_counts != expected_core_counts:
        raise ValueError(
            f"core schedule balance mismatch: {core_counts} != {expected_core_counts}"
        )
    pilot_judge_ids(rows, "v1")
    if pilot_generation >= PILOT_GENERATION_V2:
        pilot_judge_ids(rows, "v2")
    if pilot_generation >= PILOT_GENERATION_V3:
        pilot_judge_ids(rows, "v3")
    if pilot_generation >= PILOT_GENERATION_V4:
        pilot_judge_ids(rows, "v4")
    if pilot_generation >= PILOT_GENERATION_V5:
        pilot_judge_ids(rows, "v5")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--include-kimi", action="store_true")
    parser.add_argument(
        "--trim-profile",
        choices=TRIM_PROFILES,
        default=TRIM_PROFILE_NONE,
    )
    parser.add_argument(
        "--pilot-generation",
        type=int,
        choices=PILOT_GENERATIONS,
        default=PILOT_GENERATION_V7,
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite an existing schedule",
    )
    args = parser.parse_args()
    rows = build_schedule(
        load_claims(),
        args.include_kimi,
        args.trim_profile,
        args.pilot_generation,
    )
    if SCHEDULE_PATH.exists() and not args.force:
        raise SystemExit(
            f"{SCHEDULE_PATH} already exists; the released schedule is frozen. "
            "Pass --force to overwrite it."
        )
    atomic_write(SCHEDULE_PATH, pretty_json(rows))
    print(
        f"wrote {len(rows)} scheduled runs to {SCHEDULE_PATH} "
        f"(include_kimi={args.include_kimi}, trim_profile={args.trim_profile}, "
        f"pilot_generation={args.pilot_generation})"
    )


TRIM_RECORD_DIR = DATA / "schedule_trims"
TRIM_RECORD_KIND = "authenticated_schedule_trim"
TRIM_RECORD_SCHEMA_VERSION = 1


def schedule_sha256(rows: list[dict]) -> str:
    return hashlib.sha256(canonical_json(rows).encode("utf-8")).hexdigest()


def string_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def rows_for_profile(
    claims: list[dict],
    profile: str,
    include_kimi: bool,
    pilot_generation: int,
) -> list[dict]:
    return build_schedule(
        claims,
        include_kimi=include_kimi,
        trim_profile=profile,
        pilot_generation=pilot_generation,
    )


def trim_profile_spec(
    claims: list[dict],
    profile: str,
    include_kimi: bool,
    pilot_generation: int,
) -> dict[str, Any]:
    step = trim_profile_step(profile)
    replay_claim_ids = (
        selected_ids(claims, (8, 8, 8))
        if step >= 3
        else [str(claim["id"]) for claim in claims]
    )
    paraphrase_claim_ids = selected_ids(
        claims, (2, 1, 1) if step >= 4 else (3, 2, 3)
    )
    rows = rows_for_profile(claims, profile, include_kimi, pilot_generation)
    return {
        "profile": profile,
        "step": step,
        "pilot_generation": pilot_generation,
        "include_kimi": include_kimi,
        "operations_applied": [
            operation
            for operation, minimum_step in (
                ("drop_glm_core_repeat_2", 1),
                ("drop_kimi_probe", 2),
                ("limit_replay_to_24_claims", 3),
                ("limit_paraphrase_to_4_claims", 4),
            )
            if step >= minimum_step
        ],
        "core_repeat_2_systems": (
            ["sonnet5", "terra", "gemini", "glm"]
            if step == 0
            else ["sonnet5", "terra", "gemini"]
        ),
        "replay_claim_ids": replay_claim_ids,
        "paraphrase_claim_ids": paraphrase_claim_ids,
        "arm_counts": dict(sorted(Counter(row["arm"] for row in rows).items())),
        "run_count": len(rows),
        "schedule_sha256": schedule_sha256(rows),
    }


def transition_rows(
    source_rows: list[dict],
    claims: list[dict],
    source_profile: str,
    target_profile: str,
    *,
    target_include_kimi: bool = False,
) -> tuple[list[dict], dict[str, Any]]:
    source_step = trim_profile_step(source_profile)
    target_step = trim_profile_step(target_profile)
    if target_step != source_step + 1:
        raise ValueError(
            "trim migration must advance exactly one ladder step: "
            f"{source_profile!r} -> {target_profile!r}"
        )
    source_kimi_count = sum(row.get("arm") == "kimi" for row in source_rows)
    if source_kimi_count not in {0, 144}:
        raise ValueError(
            f"source schedule has a partial Kimi probe ({source_kimi_count}/144 rows)"
        )
    source_include_kimi = source_kimi_count == 144
    if target_step >= 2 and target_include_kimi:
        raise ValueError("Kimi must be absent at trim-ladder step two and later")
    if target_step == 1 and target_profile != TRIM_PROFILE_STEP_1:
        raise ValueError("trim-ladder step one has an invalid target profile")
    pilot_generation = schedule_pilot_generation(source_rows)
    if pilot_generation not in PILOT_GENERATIONS:
        raise ValueError("source schedule has an invalid pilot generation")
    expected_source = rows_for_profile(
        claims,
        source_profile,
        source_include_kimi,
        pilot_generation,
    )
    if source_rows != expected_source:
        raise ValueError(
            "source schedule is not the exact deterministic source profile"
        )
    target_rows = rows_for_profile(
        claims,
        target_profile,
        target_include_kimi,
        pilot_generation,
    )
    source_by_id = {str(row["run_id"]): row for row in source_rows}
    target_by_id = {str(row["run_id"]): row for row in target_rows}
    preserved_ids = sorted(set(source_by_id) & set(target_by_id))
    changed = [
        run_id
        for run_id in preserved_ids
        if source_by_id[run_id] != target_by_id[run_id]
    ]
    if changed:
        raise ValueError(
            "trim migration would mutate existing schedule rows; "
            f"first={changed[:3]}"
        )
    removed_ids = sorted(set(source_by_id) - set(target_by_id))
    added_ids = sorted(set(target_by_id) - set(source_by_id))
    if target_step != 1 and added_ids:
        raise ValueError(
            f"trim step {target_step} unexpectedly adds schedule rows"
        )
    expected_removed = {1: 144, 2: 144 if source_include_kimi else 0, 3: 288, 4: 32}
    if len(removed_ids) != expected_removed[target_step]:
        raise ValueError(
            f"trim step {target_step} removes {len(removed_ids)} rows, "
            f"expected {expected_removed[target_step]}"
        )
    if target_step == 1 and len(added_ids) not in {0, 144}:
        raise ValueError(
            f"trim step one adds {len(added_ids)} rows, expected zero or 144 Kimi rows"
        )
    if target_step == 1 and any(
        source_by_id[run_id].get("arm") != "core"
        or source_by_id[run_id].get("system") != "glm"
        or source_by_id[run_id].get("repeat") != 2
        for run_id in removed_ids
    ):
        raise ValueError("trim step one removes a row outside GLM core repeat two")
    if target_step == 2 and any(
        source_by_id[run_id].get("arm") != "kimi" for run_id in removed_ids
    ):
        raise ValueError("trim step two removes a non-Kimi row")
    if target_step == 3 and any(
        source_by_id[run_id].get("arm") != "replay" for run_id in removed_ids
    ):
        raise ValueError("trim step three removes a non-replay row")
    if target_step == 4 and any(
        source_by_id[run_id].get("arm") != "paraphrase" for run_id in removed_ids
    ):
        raise ValueError("trim step four removes a non-paraphrase row")
    if added_ids and any(
        target_by_id[run_id].get("arm") != "kimi" for run_id in added_ids
    ):
        raise ValueError("trim step one adds a non-Kimi row")
    preserved_rows = [target_by_id[run_id] for run_id in preserved_ids]
    transition = {
        "pilot_generation": pilot_generation,
        "source_include_kimi": source_include_kimi,
        "target_include_kimi": target_include_kimi,
        "source_schedule_sha256": schedule_sha256(source_rows),
        "target_schedule_sha256": schedule_sha256(target_rows),
        "source_run_count": len(source_rows),
        "target_run_count": len(target_rows),
        "preserved_run_count": len(preserved_ids),
        "preserved_rows_sha256": schedule_sha256(preserved_rows),
        "removed_run_ids": removed_ids,
        "removed_run_ids_sha256": hashlib.sha256(
            canonical_json(removed_ids).encode("utf-8")
        ).hexdigest(),
        "added_run_ids": added_ids,
        "added_run_ids_sha256": hashlib.sha256(
            canonical_json(added_ids).encode("utf-8")
        ).hexdigest(),
    }
    return target_rows, transition


def _parse_timestamp(value: object) -> bool:
    if not isinstance(value, str) or not value.strip():
        return False
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return parsed.tzinfo is not None


def authorization_record(authority: str, evidence: str) -> dict[str, Any]:
    authority = authority.strip()
    evidence = evidence.strip()
    if not authority:
        raise ValueError("authorization authority must be non-empty")
    if not evidence:
        raise ValueError("authorization evidence must be non-empty")
    return {
        "authorized": True,
        "authority": authority,
        "evidence": evidence,
        "evidence_sha256": string_sha256(evidence),
    }


def build_trim_record(
    source_rows: list[dict],
    claims: list[dict],
    source_profile: str,
    target_profile: str,
    *,
    target_include_kimi: bool,
    recorded_at: str,
    reason: str,
    authority: str,
    authorization_evidence: str,
    previous_record_sha256: str | None,
) -> tuple[list[dict], dict[str, Any]]:
    if not _parse_timestamp(recorded_at):
        raise ValueError("recorded_at must be an ISO-8601 timestamp with a timezone")
    reason = reason.strip()
    if not reason:
        raise ValueError("trim reason must be non-empty")
    target_rows, transition = transition_rows(
        source_rows,
        claims,
        source_profile,
        target_profile,
        target_include_kimi=target_include_kimi,
    )
    record: dict[str, Any] = {
        "schema_version": TRIM_RECORD_SCHEMA_VERSION,
        "kind": TRIM_RECORD_KIND,
        "step": trim_profile_step(target_profile),
        "source_profile": source_profile,
        "target_profile": target_profile,
        "recorded_at": recorded_at,
        "reason": reason,
        "authorization": authorization_record(authority, authorization_evidence),
        "previous_record_sha256": previous_record_sha256,
        "target_profile_spec": trim_profile_spec(
            claims,
            target_profile,
            target_include_kimi,
            transition["pilot_generation"],
        ),
        **transition,
    }
    record["record_sha256"] = hashlib.sha256(
        canonical_json(record).encode("utf-8")
    ).hexdigest()
    return target_rows, record


def load_trim_records(record_dir: Path = TRIM_RECORD_DIR) -> list[dict[str, Any]]:
    if not record_dir.exists():
        return []
    if not record_dir.is_dir():
        raise ValueError(f"{record_dir}: trim-record path is not a directory")
    files = sorted(record_dir.glob("step_*.json"))
    unexpected = sorted(
        file.name for file in record_dir.glob("*.json") if file not in files
    )
    if unexpected:
        raise ValueError(
            f"{record_dir}: unexpected trim-record files: {unexpected[:3]}"
        )
    expected_names = [f"step_{index}.json" for index in range(1, len(files) + 1)]
    if [file.name for file in files] != expected_names:
        raise ValueError(f"{record_dir}: trim records are missing, extra, or non-contiguous")
    if len(files) > len(TRIM_PROFILES) - 1:
        raise ValueError(f"{record_dir}: too many trim records")
    records = []
    for file in files:
        try:
            value = json.loads(file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"{file}: unreadable trim record: {error}") from error
        if not isinstance(value, dict):
            raise ValueError(f"{file}: trim record must be an object")
        records.append(value)
    return records


def validate_trim_record_sequence(
    records: list[dict[str, Any]],
    schedule_rows: list[dict],
    claims: list[dict],
) -> dict[str, Any]:
    pilot_generation = schedule_pilot_generation(schedule_rows)
    current_profile = TRIM_PROFILE_NONE
    current_include_kimi = False
    current_rows = rows_for_profile(
        claims,
        current_profile,
        current_include_kimi,
        pilot_generation,
    )
    previous_hash = None
    previous_time: datetime | None = None
    for expected_step, observed in enumerate(records, 1):
        if expected_step >= len(TRIM_PROFILES):
            raise ValueError("trim record advances beyond the defined ladder")
        target_profile = TRIM_PROFILES[expected_step]
        target_include_kimi = observed.get("target_include_kimi")
        if not isinstance(target_include_kimi, bool):
            raise ValueError(f"trim step {expected_step}: target_include_kimi is invalid")
        try:
            _target_rows, expected = build_trim_record(
                current_rows,
                claims,
                current_profile,
                target_profile,
                target_include_kimi=target_include_kimi,
                recorded_at=observed.get("recorded_at"),
                reason=observed.get("reason"),
                authority=(observed.get("authorization") or {}).get("authority"),
                authorization_evidence=(observed.get("authorization") or {}).get(
                    "evidence"
                ),
                previous_record_sha256=previous_hash,
            )
        except (AttributeError, TypeError, ValueError) as error:
            raise ValueError(f"trim step {expected_step}: {error}") from error
        if observed != expected:
            raise ValueError(
                f"trim step {expected_step}: record differs from the deterministic transition"
            )
        observed_time = datetime.fromisoformat(
            str(observed["recorded_at"]).replace("Z", "+00:00")
        )
        if previous_time is not None and observed_time < previous_time:
            raise ValueError(
                f"trim step {expected_step}: timestamp predates the prior trim step"
            )
        previous_time = observed_time
        current_rows = _target_rows
        current_profile = target_profile
        current_include_kimi = target_include_kimi
        previous_hash = expected["record_sha256"]
    if schedule_rows != current_rows:
        raise ValueError(
            "current schedule differs from the final authenticated trim profile"
        )
    return {
        "profile": current_profile,
        "step": trim_profile_step(current_profile),
        "include_kimi": current_include_kimi,
        "pilot_generation": pilot_generation,
        "schedule_sha256": schedule_sha256(schedule_rows),
        "record_count": len(records),
        "last_record_sha256": previous_hash,
        "profile_spec": trim_profile_spec(
            claims,
            current_profile,
            current_include_kimi,
            pilot_generation,
        ),
    }


def verify_trim_records(
    schedule_rows: list[dict],
    claims: list[dict],
    record_dir: Path = TRIM_RECORD_DIR,
) -> dict[str, Any]:
    return validate_trim_record_sequence(
        load_trim_records(record_dir), schedule_rows, claims
    )


def apply_trim_step(
    schedule_path: Path,
    claims: list[dict],
    record_dir: Path,
    target_profile: str,
    *,
    target_include_kimi: bool,
    reason: str,
    authority: str,
    authorization_evidence: str,
    recorded_at: str | None = None,
) -> dict[str, Any]:
    try:
        source_rows = json.loads(schedule_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{schedule_path}: unreadable schedule: {error}") from error
    if not isinstance(source_rows, list) or not all(
        isinstance(row, dict) for row in source_rows
    ):
        raise ValueError(f"{schedule_path}: schedule must be a list of objects")
    records = load_trim_records(record_dir)
    if records and records[-1].get("target_profile") == target_profile:
        if records[-1].get("target_include_kimi") != target_include_kimi:
            raise ValueError(
                "existing trim record has a different Kimi target decision"
            )
        state = validate_trim_record_sequence(records, source_rows, claims)
        return {**state, "status": "already_applied"}
    state = validate_trim_record_sequence(records, source_rows, claims)
    source_profile = str(state["profile"])
    expected_target = TRIM_PROFILES[trim_profile_step(source_profile) + 1]
    if target_profile != expected_target:
        raise ValueError(
            f"next authorized trim profile must be {expected_target!r}, "
            f"not {target_profile!r}"
        )
    previous_hash = state["last_record_sha256"]
    target_rows, record = build_trim_record(
        source_rows,
        claims,
        source_profile,
        target_profile,
        target_include_kimi=target_include_kimi,
        recorded_at=recorded_at or now_iso(),
        reason=reason,
        authority=authority,
        authorization_evidence=authorization_evidence,
        previous_record_sha256=previous_hash,
    )
    record_dir.mkdir(parents=True, exist_ok=True)
    record_path = record_dir / f"step_{record['step']}.json"
    if record_path.exists():
        raise ValueError(f"refusing to overwrite immutable trim record {record_path}")
    # Record first: interruption leaves a fail-closed schedule/record mismatch.
    atomic_write(record_path, pretty_json(record))
    atomic_write(schedule_path, pretty_json(target_rows))
    final_state = verify_trim_records(target_rows, claims, record_dir)
    return {
        **final_state,
        "status": "applied",
        "record_path": str(record_path),
        "removed_runs": len(record["removed_run_ids"]),
        "added_runs": len(record["added_run_ids"]),
    }


if __name__ == "__main__":
    main()
