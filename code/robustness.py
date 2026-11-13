"""Robustness checks a reviewer can ask for without new model calls.

Four questions, all answered from logs already on disk:

* did runs fail in a way that depends on the user's framing, which would break
  exchangeability between the analyzed conditions,
* do the headline contrasts survive dropping every run that cited a URL outside
  its own retrieved results,
* do the paraphrase and third-party attribution arms carry the same contrast as
  the core arm,
* does the net query-direction result survive label noise at the rate and shape
  the cross-family judge actually exhibited.

Each returns plain dictionaries. ``analysis`` folds them into ``values.json``
and writes the row-level tables beside the other released artefacts.
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict

FRAMINGS = ("believer", "neutral", "skeptic")
STANCE_LABELS = ("stance_congruent", "stance_incongruent", "neutral")
NOISE_REPLICATES = 2000
NOISE_SEED = 20260804


def _rate(events: int, total: int) -> float | None:
    return events / total if total else None


def _breakdown(rows: list[tuple[str, bool]]) -> dict[str, dict]:
    totals: Counter[str] = Counter()
    failures: Counter[str] = Counter()
    for key, failed in rows:
        totals[key] += 1
        if failed:
            failures[key] += 1
    return {
        key: {
            "planned": totals[key],
            "failed": failures[key],
            "rate": _rate(failures[key], totals[key]),
        }
        for key in sorted(totals)
    }


def summarize_failures(
    missingness: list[dict],
    phase2_records: list[dict],
    ignored: list[dict],
    schedule_rows: list[dict],
    claims_by_id: dict[str, dict],
    chi_square,
) -> dict:
    """Terminal-failure breakdown, and the framing test that closes exchangeability.

    The framing comparison is run on the arms that actually cross all three
    framings. Restricting it is not cherry-picking: ``attribution`` and the two
    paraphrase framings each live in exactly one arm, so a marginal test over
    every framing label is really an arm test wearing a framing name.
    """

    registered = [row for row in missingness if row.get("planned")]
    failed_flag = [
        (row, row.get("status") == "terminal_failed") for row in registered
    ]
    balanced_arms = {"core", "replay"}

    by_framing = _breakdown(
        [(str(row.get("framing")), failed) for row, failed in failed_flag]
    )
    by_system = _breakdown(
        [(str(row.get("system")), failed) for row, failed in failed_flag]
    )
    by_tier = _breakdown([(str(row.get("tier")), failed) for row, failed in failed_flag])
    by_arm = _breakdown([(str(row.get("arm")), failed) for row, failed in failed_flag])

    balanced = [
        (row, failed)
        for row, failed in failed_flag
        if row.get("arm") in balanced_arms and row.get("framing") in FRAMINGS
    ]
    balanced_by_framing = _breakdown(
        [(str(row.get("framing")), failed) for row, failed in balanced]
    )
    test = _framing_test(balanced_by_framing, chi_square)

    reasons = Counter(
        str(row.get("reason_code"))
        for row, failed in failed_flag
        if failed and row.get("reason_code")
    )

    phase2 = _phase2_failures(phase2_records, ignored, schedule_rows, claims_by_id)
    return {
        "registered": {
            "planned": len(registered),
            "failed": sum(1 for _row, failed in failed_flag if failed),
            "by_framing": by_framing,
            "by_system": by_system,
            "by_tier": by_tier,
            "by_arm": by_arm,
            "balanced_arms": sorted(balanced_arms),
            "balanced_by_framing": balanced_by_framing,
            "framing_test": test,
            "reason_codes": dict(reasons.most_common()),
        },
        "second_round": phase2,
    }


def _framing_test(by_framing: dict[str, dict], chi_square) -> dict:
    """Largest pairwise framing gap, with the two-by-two test behind it.

    ``pearson_chi_square`` compares two proportions, so the reported test is the
    widest pair rather than an omnibus over three. That is the comparison a
    reviewer worried about exchangeability would make anyway, and it is the
    conservative one: any omnibus statistic is bounded by its widest pair.
    """

    usable = {
        framing: entry
        for framing, entry in by_framing.items()
        if entry["planned"] and entry["rate"] is not None
    }
    if len(usable) < 2:
        return {"estimable": False, "diagnostic": "fewer than two framings planned"}
    ordered = sorted(usable.items(), key=lambda item: item[1]["rate"])
    low_name, low = ordered[0]
    high_name, high = ordered[-1]
    result = chi_square(high["failed"], high["planned"], low["failed"], low["planned"])
    result["widest_pair"] = [high_name, low_name]
    result["gap_points"] = (high["rate"] - low["rate"]) * 100.0
    return result


def _phase2_failures(
    phase2_records: list[dict],
    ignored: list[dict],
    schedule_rows: list[dict],
    claims_by_id: dict[str, dict],
) -> dict:
    """Second-round failures over what was attempted, never over what was planned.

    The budget stop left 236 second-round rows never attempted, so a planned
    denominator would report abandonment as failure.
    """

    scheduled = {row["run_id"]: row for row in schedule_rows}
    attempted: list[tuple[dict, bool]] = []
    for row in phase2_records:
        attempted.append((row, False))
    for row in ignored:
        if row.get("run_id") in scheduled:
            attempted.append((row, row.get("ignored_reason") == "terminal_failed"))

    def field(row: dict, name: str) -> str:
        value = row.get(name) or scheduled.get(row.get("run_id"), {}).get(name)
        if name == "tier" and not value:
            claim = claims_by_id.get(str(row.get("claim_id")))
            value = claim.get("tier") if claim else None
        return str(value)

    return {
        "planned": len(schedule_rows),
        "attempted": len(attempted),
        "failed": sum(1 for _row, failed in attempted if failed),
        "by_framing": _breakdown(
            [(field(row, "framing"), failed) for row, failed in attempted]
        ),
        "by_system": _breakdown(
            [(field(row, "system"), failed) for row, failed in attempted]
        ),
        "by_arm": _breakdown([(field(row, "arm"), failed) for row, failed in attempted]),
        "by_tier": _breakdown([(field(row, "tier"), failed) for row, failed in attempted]),
    }


def failure_rows(summary: dict) -> list[dict]:
    """One released row per breakdown cell, registered and second round alike."""

    rows = []
    for scope, block in (
        ("registered", summary["registered"]),
        ("second_round", summary["second_round"]),
    ):
        denominator = "planned" if scope == "registered" else "attempted"
        for dimension in ("by_framing", "by_system", "by_tier", "by_arm"):
            for key, entry in block[dimension].items():
                rows.append(
                    {
                        "scope": scope,
                        "dimension": dimension.removeprefix("by_"),
                        "group": key,
                        "denominator_kind": denominator,
                        "denominator": entry["planned"],
                        "terminal_failed": entry["failed"],
                        "rate": "" if entry["rate"] is None else f"{entry['rate']:.4f}",
                    }
                )
    return rows


def summarize_citation_sensitivity(
    records: list[dict],
    paired_deltas,
    delta_summary,
    draws: int,
) -> dict:
    flagged = [row for row in records if not row.get("all_citations_in_results", True)]
    splits = {}
    for dimension in ("framing", "system", "tier"):
        totals: Counter[str] = Counter()
        events: Counter[str] = Counter()
        for row in records:
            key = str(row.get(dimension))
            totals[key] += 1
            if not row.get("all_citations_in_results", True):
                events[key] += 1
        splits[dimension] = {
            key: {
                "runs": totals[key],
                "outside": events[key],
                "share": _rate(events[key], totals[key]),
            }
            for key in sorted(totals)
        }

    kept = [row for row in records if row.get("all_citations_in_results", True)]
    contrasts = {}
    for key, first, second in (
        ("believer_vs_skeptic", "believer", "skeptic"),
        ("skeptic_vs_neutral", "skeptic", "neutral"),
        ("believer_vs_neutral", "believer", "neutral"),
    ):
        seed_key = f"decomposition:score:{key}"
        full = delta_summary(paired_deltas(records, "core", first, second), seed_key, draws)
        excluded = delta_summary(paired_deltas(kept, "core", first, second), seed_key, draws)
        contrasts[key] = {
            "full": full,
            "excluding_out_of_set": excluded,
            "pairs_lost": full["n"] - excluded["n"],
            "still_excludes_zero": _excludes_zero(excluded),
            "sign_preserved": _same_sign(full, excluded),
        }
    return {
        "runs": len(records),
        "out_of_set_runs": len(flagged),
        "share": _rate(len(flagged), len(records)),
        "splits": splits,
        "contrasts": contrasts,
        "conclusions_change": any(
            not entry["sign_preserved"]
            or entry["still_excludes_zero"] != _excludes_zero(entry["full"])
            for entry in contrasts.values()
        ),
    }


def _excludes_zero(summary: dict) -> bool:
    low, high = summary.get("ci_low"), summary.get("ci_high")
    if low is None or high is None:
        return False
    return low > 0 or high < 0


def _same_sign(first: dict, second: dict) -> bool:
    """Zero counts as agreeing with either side: a move to the null is not a flip."""

    a, b = first.get("mean_delta"), second.get("mean_delta")
    if a is None or b is None:
        return False
    return a == 0 or b == 0 or (a > 0) == (b > 0)


def citation_rows(summary: dict) -> list[dict]:
    rows = []
    for dimension, groups in summary["splits"].items():
        for group, entry in groups.items():
            rows.append(
                {
                    "kind": "split",
                    "dimension": dimension,
                    "group": group,
                    "runs": entry["runs"],
                    "out_of_set_runs": entry["outside"],
                    "share": "" if entry["share"] is None else f"{entry['share']:.4f}",
                    "mean_delta": "",
                    "ci_low": "",
                    "ci_high": "",
                    "n": "",
                }
            )
    for contrast, entry in summary["contrasts"].items():
        for sample, key in (("full", "full"), ("excluding_out_of_set", "excluding_out_of_set")):
            estimate = entry[key]
            rows.append(
                {
                    "kind": "contrast",
                    "dimension": contrast,
                    "group": sample,
                    "runs": "",
                    "out_of_set_runs": "",
                    "share": "",
                    "mean_delta": _round(estimate.get("mean_delta")),
                    "ci_low": _round(estimate.get("ci_low")),
                    "ci_high": _round(estimate.get("ci_high")),
                    "n": estimate.get("n", ""),
                }
            )
    return rows


def _round(value: object) -> str:
    return "" if not isinstance(value, (int, float)) else f"{float(value):.4f}"


def summarize_control_arms(
    records: list[dict],
    score_index,
    paired_deltas,
    delta_summary,
    draws: int,
) -> dict:
    """Control arms on the ordinal scale, not just on the flip rule.

    The paraphrase arm carries both stances, so it pairs within itself. The
    attribution arm carries one framing by design, third-party belief, so it
    pairs against the matched core neutral run instead.
    """

    paraphrase = delta_summary(
        paired_deltas(records, "paraphrase", "believer_paraphrase", "skeptic_paraphrase"),
        "control:paraphrase_ordinal",
        draws,
    )
    attribution_index = score_index(records, "attribution")
    core_index = score_index(records, "core")
    pairs = []
    for (claim_id, system, _framing, repeat), row in sorted(attribution_index.items()):
        neutral = core_index.get((claim_id, system, "neutral", repeat)) or core_index.get(
            (claim_id, system, "neutral", 1)
        )
        if neutral is None:
            continue
        pairs.append(
            {
                "claim_id": claim_id,
                "tier": row["tier"],
                "system": system,
                "repeat": repeat,
                "first": row["score"],
                "second": neutral["score"],
                "delta": row["score"] - neutral["score"],
            }
        )
    attribution = delta_summary(pairs, "control:attribution_vs_neutral", draws)
    return {"paraphrase": paraphrase, "attribution_vs_neutral": attribution}


def summarize_query_label_noise(
    records: list[dict],
    query_judge: dict,
    query_direction: dict,
    replicates: int = NOISE_REPLICATES,
    seed: int = NOISE_SEED,
) -> dict:
    """Resample every primary query label through the measured judge confusion.

    This asks what the net direction statistic would have been had the primary
    classifier erred the way the cross-family judge says it errs. It is not a
    bound on arbitrary error: the answer depends on the shape of the confusion,
    which is reported alongside so a reader can see what is being assumed.
    """

    transition = _transition_matrix(query_judge.get("confusion") or [])
    if not transition:
        return {"estimable": False, "diagnostic": "no cross-family confusion counts"}

    observed_labels: dict[str, list[str]] = defaultdict(list)
    for row in records:
        if row.get("arm") != "core":
            continue
        framing = str(row.get("framing"))
        if framing not in FRAMINGS:
            continue
        for label in row.get("query_labels") or []:
            observed_labels[framing].append(str(label))

    observed = {
        framing: _net_points(labels, framing, query_direction)
        for framing, labels in observed_labels.items()
    }
    if not observed:
        return {"estimable": False, "diagnostic": "no primary query labels"}

    rng = random.Random(seed)
    draws: dict[str, list[float]] = {framing: [] for framing in observed}
    gaps: list[float] = []
    for _ in range(replicates):
        nets = {}
        for framing, labels in observed_labels.items():
            resampled = [_resample(label, transition, rng) for label in labels]
            nets[framing] = _net_points(resampled, framing, query_direction)
            draws[framing].append(nets[framing])
        if "skeptic" in nets and "believer" in nets:
            gaps.append(nets["skeptic"] - nets["believer"])

    per_framing = {}
    for framing, values in draws.items():
        values = sorted(values)
        point = observed[framing]
        per_framing[framing] = {
            "n_queries": len(observed_labels[framing]),
            "observed_points": point,
            "mean_points": sum(values) / len(values),
            "pct2_5_points": _percentile(values, 0.025),
            "pct97_5_points": _percentile(values, 0.975),
            "sign_survival": _rate(
                sum(1 for value in values if value * point > 0), len(values)
            ),
        }
    return {
        "estimable": True,
        "replicates": replicates,
        "seed": seed,
        "agreement": (query_judge.get("overall") or {}).get("agreement"),
        "labeled_queries": (query_judge.get("overall") or {}).get("n"),
        "transition": transition,
        "off_diagonal": _off_diagonal(query_judge.get("confusion") or []),
        "by_framing": per_framing,
        "skeptic_below_believer": _rate(
            sum(1 for gap in gaps if gap < 0), len(gaps)
        )
        if gaps
        else None,
    }


def _transition_matrix(confusion: list[dict]) -> dict[str, dict[str, float]]:
    """P(cross-family label | primary label), rows normalised over observed counts."""

    totals: Counter[str] = Counter()
    cells: dict[tuple[str, str], int] = {}
    for row in confusion:
        primary = str(row.get("gemini"))
        secondary = str(row.get("luna"))
        count = int(row.get("count") or 0)
        totals[primary] += count
        cells[(primary, secondary)] = cells.get((primary, secondary), 0) + count
    matrix = {}
    for primary in STANCE_LABELS:
        total = totals.get(primary, 0)
        if not total:
            # An unobserved primary label cannot be resampled, so it is held
            # fixed rather than given an invented error rate.
            matrix[primary] = {primary: 1.0}
            continue
        matrix[primary] = {
            secondary: cells.get((primary, secondary), 0) / total
            for secondary in STANCE_LABELS
        }
    return matrix


def _off_diagonal(confusion: list[dict]) -> list[dict]:
    return [
        {
            "primary": str(row.get("gemini")),
            "cross_family": str(row.get("luna")),
            "count": int(row.get("count") or 0),
        }
        for row in confusion
        if str(row.get("gemini")) != str(row.get("luna")) and int(row.get("count") or 0)
    ]


def _resample(label: str, transition: dict[str, dict[str, float]], rng: random.Random) -> str:
    row = transition.get(label) or {label: 1.0}
    draw = rng.random()
    cumulative = 0.0
    for candidate, probability in row.items():
        cumulative += probability
        if draw < cumulative:
            return candidate
    return label


def _net_points(labels: list[str], framing: str, query_direction: dict) -> float:
    """Net claim-affirming share, in percentage points, over the framing's queries."""

    if not labels:
        return 0.0
    net = sum(query_direction.get((framing, label), 0) for label in labels)
    return 100.0 * net / len(labels)


def _percentile(sorted_values: list[float], fraction: float) -> float:
    if not sorted_values:
        raise ValueError("percentile of an empty sample")
    index = min(len(sorted_values) - 1, max(0, round(fraction * (len(sorted_values) - 1))))
    return sorted_values[index]
