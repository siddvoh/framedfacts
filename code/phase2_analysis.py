"""Estimands for the second-round arms and the review's follow-up questions.

Nothing here enters a registered denominator. Each summary is reported as a
post-hoc addition beside the confirmatory results.
"""

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np


CROSS_REPLAY_DIRECTIONS = (("believer", "skeptic"), ("skeptic", "believer"))
STANCE_LABELS = ("supports_claim", "opposes_claim", "neutral_or_unrelated")
QUERY_LABELS = ("stance_congruent", "stance_incongruent", "neutral")


def score_of(record: dict) -> float | None:
    score = record.get("score")
    return None if score is None else float(score)


def usable(record: dict) -> bool:
    return not record.get("ignored") and record.get("score") is not None


def index_runs(records: list[dict], arm: str) -> dict[tuple, dict]:
    return {
        (row["claim_id"], row["system"], row["framing"], row["repeat"]): row
        for row in records
        if row.get("arm") == arm and usable(row)
    }


def summarize_neutral_noise_by_system(
    records: list[dict],
    phase2_records: list[dict],
    delta_summary,
    draws: int,
) -> dict:
    """Neutral repeat-to-repeat movement per system.

    The trim ladder cut GLM's second core neutral run, so its noise floor comes
    from the second-round repeat instead. Both sources are neutral framing on
    the same claim and system.
    """

    core = index_runs(records, "core")
    extra = index_runs(phase2_records, "neutral_repeat")
    per_system: dict[str, list[dict]] = defaultdict(list)
    for (claim_id, system, framing, repeat), row in sorted(core.items()):
        if framing != "neutral" or repeat != 1:
            continue
        second = core.get((claim_id, system, "neutral", 2))
        source = "core_repeat_2"
        if second is None:
            second = extra.get((claim_id, system, "neutral", 2))
            source = "phase2_repeat"
        if second is None:
            continue
        per_system[system].append(
            {
                "claim_id": claim_id,
                "tier": row["tier"],
                "system": system,
                "repeat": 1,
                "source": source,
                "first": score_of(row),
                "second": score_of(second),
                "delta": score_of(row) - score_of(second),
            }
        )
    summary = {}
    for system, pairs in sorted(per_system.items()):
        entry = delta_summary(pairs, f"noise:{system}", draws)
        entry["sources"] = dict(Counter(pair["source"] for pair in pairs))
        summary[system] = entry
    return summary


def summarize_skeptic_mitigation(
    records: list[dict],
    phase2_records: list[dict],
    delta_summary,
    draws: int,
) -> dict:
    """Mitigated skeptic runs against their matched core skeptic runs.

    The registered mitigation arm attached the both-sides line to the believer
    condition, where the study found no effect to remove.
    """

    core = index_runs(records, "core")
    mitigated = index_runs(phase2_records, "skeptic_mitigation")
    pairs = []
    for (claim_id, system, framing, _repeat), row in sorted(mitigated.items()):
        base = core.get((claim_id, system, "skeptic", 1))
        if base is None:
            continue
        pairs.append(
            {
                "claim_id": claim_id,
                "tier": row["tier"],
                "system": system,
                "repeat": 1,
                "first": score_of(row),
                "second": score_of(base),
                "delta": score_of(row) - score_of(base),
            }
        )
    summary = {"overall": delta_summary(pairs, "skeptic_mitigation", draws)}
    for tier in ("settled_false", "settled_true", "contested"):
        summary[tier] = delta_summary(
            [pair for pair in pairs if pair["tier"] == tier],
            f"skeptic_mitigation:{tier}",
            draws,
        )
    summary["by_system"] = {
        system: delta_summary(
            [pair for pair in pairs if pair["system"] == system],
            f"skeptic_mitigation:{system}",
            draws,
        )
        for system in sorted({pair["system"] for pair in pairs})
    }
    summary["claims"] = len({pair["claim_id"] for pair in pairs})
    summary["deviation"] = mitigation_deviation(records, phase2_records)
    return summary


def mitigation_deviation(records: list[dict], phase2_records: list[dict]) -> dict:
    def counts(rows: list[dict]) -> dict:
        settled = [
            row
            for row in rows
            if usable(row) and row["tier"] in {"settled_true", "settled_false"}
        ]
        deviating = sum(1 for row in settled if row.get("deviation"))
        return {"deviating": deviating, "total": len(settled)}

    mitigated_rows = [
        row for row in phase2_records if row.get("arm") == "skeptic_mitigation"
    ]
    claim_systems = {
        (row["claim_id"], row["system"]) for row in mitigated_rows if usable(row)
    }
    core_rows = [
        row
        for row in records
        if row.get("arm") == "core"
        and row.get("framing") == "skeptic"
        and (row.get("claim_id"), row.get("system")) in claim_systems
    ]
    return {"mitigated": counts(mitigated_rows), "core": counts(core_rows)}


def summarize_cross_replay(
    records: list[dict],
    phase2_records: list[dict],
    delta_summary,
    draws: int,
) -> dict:
    """Split the believer-minus-skeptic gap into a synthesis and an evidence channel.

    The two crossed runs are the off-diagonal cells of a framing-by-evidence
    square whose diagonal is the matching pair of core runs. Averaging the two
    framing contrasts down that square gives the synthesis channel and averaging
    the two evidence contrasts across it gives the evidence channel, so the two
    add back to the end-to-end gap in every cell. Each average also pairs one
    live run against one replayed run on each side, which cancels any constant
    difference between issuing a search and reading a frozen transcript.
    """

    cross = {
        (row["claim_id"], row["system"], row["framing"]): row
        for row in phase2_records
        if row.get("arm") == "cross_replay" and usable(row)
    }
    core = index_runs(records, "core")

    synthesis_pairs = []
    evidence_pairs = []
    end_to_end = []
    for (claim_id, system, framing), believer_on_skeptic in sorted(cross.items()):
        if framing != "believer":
            continue
        skeptic_on_believer = cross.get((claim_id, system, "skeptic"))
        believer_own = core.get((claim_id, system, "believer", 1))
        skeptic_own = core.get((claim_id, system, "skeptic", 1))
        if skeptic_on_believer is None or believer_own is None or skeptic_own is None:
            continue
        own_b = score_of(believer_own)
        own_s = score_of(skeptic_own)
        crossed_b = score_of(believer_on_skeptic)
        crossed_s = score_of(skeptic_on_believer)
        cell = {
            "claim_id": claim_id,
            "tier": believer_own["tier"],
            "system": system,
            "repeat": 1,
        }
        synthesis_pairs.append(
            # Believer minus skeptic on the skeptic's evidence, then on the believer's.
            {**cell, "delta": ((own_b - crossed_s) + (crossed_b - own_s)) / 2}
        )
        evidence_pairs.append(
            # Believer's evidence minus the skeptic's, under each framing in turn.
            {**cell, "delta": ((own_b - crossed_b) + (crossed_s - own_s)) / 2}
        )
        end_to_end.append({**cell, "delta": own_b - own_s})

    return {
        "synthesis_channel": delta_summary(synthesis_pairs, "cross:synthesis", draws),
        "evidence_channel": delta_summary(evidence_pairs, "cross:evidence", draws),
        "end_to_end_matched": delta_summary(end_to_end, "cross:end_to_end", draws),
    }


def summarize_evidence_stance(
    records: list[dict], bootstrap_cluster_stat, keyed_seed, draws: int
) -> dict:
    retrieved: dict[str, list[dict]] = defaultdict(list)
    cited: dict[str, list[dict]] = defaultdict(list)
    for row in records:
        if row.get("ignored") or row.get("arm") != "core":
            continue
        manifest = row.get("manifest") or {}
        stance = manifest.get("evidence_stance")
        searches = [
            search
            for search in manifest.get("searches", [])
            if isinstance(search, dict) and search.get("results")
        ]
        if not isinstance(stance, list) or len(stance) != len(searches):
            continue
        citations = {str(url) for url in manifest.get("citations", [])}
        framing = str(row["framing"])
        for search, entry in zip(searches, stance):
            labels = entry.get("labels") or []
            for result, label in zip(search["results"], labels):
                item = {
                    "claim_id": row["claim_id"],
                    "label": label,
                    "cited": str(result.get("url")) in citations,
                }
                retrieved[framing].append(item)
                if item["cited"]:
                    cited[framing].append(item)

    def composition(items: list[dict], key: str) -> dict:
        total = len(items)
        entry = {"n": total, "claims": len({item["claim_id"] for item in items})}
        for label in STANCE_LABELS:
            count = sum(1 for item in items if item["label"] == label)
            entry[label] = {
                "count": count,
                "share": count / total if total else None,
            }
            if total and len({item["claim_id"] for item in items}) >= 2:
                clusters: dict[str, list[float]] = defaultdict(list)
                for item in items:
                    clusters[item["claim_id"]].append(
                        1.0 if item["label"] == label else 0.0
                    )
                low, high = bootstrap_cluster_stat(
                    dict(clusters),
                    lambda sample: float(sample.mean()),
                    draws=draws,
                    seed=keyed_seed(f"stance:{key}:{label}"),
                )
                entry[label]["ci_low"] = low
                entry[label]["ci_high"] = high
        if total:
            opposing = sum(1 for item in items if item["label"] == "opposes_claim")
            supporting = sum(1 for item in items if item["label"] == "supports_claim")
            entry["net_supporting"] = (supporting - opposing) / total
        return entry

    summary = {
        "retrieved": {
            framing: composition(items, f"retrieved:{framing}")
            for framing, items in sorted(retrieved.items())
        },
        "cited": {
            framing: composition(items, f"cited:{framing}")
            for framing, items in sorted(cited.items())
        },
    }
    summary["retrieved_contrasts"] = stance_contrasts(retrieved, bootstrap_cluster_stat, keyed_seed, draws)
    # The sweep walked runs in claim order and stopped when the budget did, so
    # coverage is a prefix of the claim set rather than a per-cell draw. Record
    # what it actually reached so the paper can say so.
    audited = sorted({item["claim_id"] for items in retrieved.values() for item in items})
    summary["coverage"] = {
        "claims_reached": len(audited),
        "claims_total": len({row["claim_id"] for row in records}),
        "claim_ids": audited,
    }
    return summary


def stance_contrasts(
    grouped: dict[str, list[dict]], bootstrap_cluster_stat, keyed_seed, draws: int
) -> dict:
    """Claim-clustered differences in opposing-evidence share between framings."""

    contrasts = {}
    for first, second in (("skeptic", "neutral"), ("believer", "neutral"), ("believer", "skeptic")):
        left, right = grouped.get(first, []), grouped.get(second, [])
        if not left or not right:
            continue
        by_claim: dict[str, list[float]] = defaultdict(list)
        left_share = share_by_claim(left)
        right_share = share_by_claim(right)
        shared = sorted(set(left_share) & set(right_share))
        for claim_id in shared:
            by_claim[claim_id].append(left_share[claim_id] - right_share[claim_id])
        if len(by_claim) < 2:
            continue
        deltas = [value for values in by_claim.values() for value in values]
        low, high = bootstrap_cluster_stat(
            dict(by_claim),
            lambda sample: float(sample.mean()),
            draws=draws,
            seed=keyed_seed(f"stance_contrast:{first}_{second}"),
        )
        contrasts[f"{first}_vs_{second}"] = {
            "claims": len(by_claim),
            "mean_difference": float(np.mean(deltas)),
            "ci_low": low,
            "ci_high": high,
        }
    return contrasts


def share_by_claim(items: list[dict]) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for item in items:
        grouped[item["claim_id"]].append(1.0 if item["label"] == "opposes_claim" else 0.0)
    return {claim_id: float(np.mean(values)) for claim_id, values in grouped.items()}


def summarize_query_judge_agreement(records: list[dict]) -> dict:
    pairs = []
    for row in records:
        if row.get("ignored"):
            continue
        manifest = row.get("manifest") or {}
        primary = manifest.get("query_labels")
        secondary = manifest.get("luna_query_labels")
        if not isinstance(primary, list) or not isinstance(secondary, list):
            continue
        if len(primary) != len(secondary) or not primary:
            continue
        for first, second in zip(primary, secondary):
            pairs.append(
                {
                    "framing": row["framing"],
                    "system": row["system"],
                    "gemini": first,
                    "luna": second,
                }
            )
    summary = {
        "queries": len(pairs),
        "runs": len(
            {
                row["run_id"]
                for row in records
                if isinstance((row.get("manifest") or {}).get("luna_query_labels"), list)
            }
        ),
        "overall": agreement_stats(pairs),
        "by_framing": {
            framing: agreement_stats([pair for pair in pairs if pair["framing"] == framing])
            for framing in sorted({pair["framing"] for pair in pairs})
        },
        "by_system": {
            system: agreement_stats([pair for pair in pairs if pair["system"] == system])
            for system in sorted({pair["system"] for pair in pairs})
        },
        "confusion": [
            {
                "gemini": first,
                "luna": second,
                "count": sum(
                    1
                    for pair in pairs
                    if pair["gemini"] == first and pair["luna"] == second
                ),
            }
            for first in QUERY_LABELS
            for second in QUERY_LABELS
        ],
    }
    return summary


def agreement_stats(pairs: list[dict]) -> dict:
    total = len(pairs)
    if not total:
        return {"n": 0, "agreement": None, "kappa": None}
    agree = sum(1 for pair in pairs if pair["gemini"] == pair["luna"])
    observed = agree / total
    expected = 0.0
    for label in QUERY_LABELS:
        first = sum(1 for pair in pairs if pair["gemini"] == label) / total
        second = sum(1 for pair in pairs if pair["luna"] == label) / total
        expected += first * second
    kappa = (observed - expected) / (1 - expected) if expected < 1 else None
    return {"n": total, "agreement": observed, "kappa": kappa}


def summarize_report_agreement_by_system(records: list[dict]) -> dict:
    """Whether the Gemini judge treats Gemini's own reports differently."""

    grouped: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for row in records:
        if row.get("ignored"):
            continue
        manifest = row.get("manifest") or {}
        primary = (manifest.get("judge_labels") or {}).get("conclusion")
        secondary = (manifest.get("luna_labels") or {}).get("conclusion")
        if not isinstance(primary, dict) or not isinstance(secondary, dict):
            continue
        grouped[str(row["system"])].append((primary["label"], secondary["label"]))
    summary = {}
    for system, items in sorted(grouped.items()):
        total = len(items)
        agree = sum(1 for first, second in items if first == second)
        summary[system] = {"n": total, "agreement": agree / total if total else None}
    return summary


def characterize_high_movement(
    records: list[dict], claims: dict[str, dict], top_n: int
) -> dict:
    means: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in records:
        if row.get("arm") != "core" or not usable(row):
            continue
        means[row["claim_id"]][row["framing"]].append(score_of(row))
    movement = {}
    neutral_means = {}
    for claim_id, framings in means.items():
        if not all(key in framings for key in ("neutral", "believer", "skeptic")):
            continue
        neutral = float(np.mean(framings["neutral"]))
        believer = float(np.mean(framings["believer"]))
        skeptic = float(np.mean(framings["skeptic"]))
        movement[claim_id] = abs(believer - neutral) + abs(skeptic - neutral)
        neutral_means[claim_id] = neutral
    ranked = sorted(movement.items(), key=lambda item: (-item[1], item[0]))
    total = sum(value for _claim, value in ranked)
    top = [claim_id for claim_id, _value in ranked[:top_n]]
    rows = []
    for claim_id in top:
        claim = claims[claim_id]
        rows.append(
            {
                "claim_id": claim_id,
                "tier": claim.get("tier"),
                "claim_form": claim.get("claim_form"),
                "movement": movement[claim_id],
                "neutral_mean": neutral_means[claim_id],
                "endpoint_distance": 2.0 - abs(neutral_means[claim_id]),
            }
        )
    others = [claim_id for claim_id, _value in ranked[top_n:]]
    return {
        "top_n": top_n,
        "claims": rows,
        "share_of_movement": (
            sum(movement[claim_id] for claim_id in top) / total if total else None
        ),
        "tier_mix": dict(Counter(row["tier"] for row in rows)),
        "form_mix": dict(Counter(row["claim_form"] for row in rows)),
        "mean_endpoint_distance": (
            float(np.mean([row["endpoint_distance"] for row in rows])) if rows else None
        ),
        "other_mean_endpoint_distance": (
            float(np.mean([2.0 - abs(neutral_means[claim_id]) for claim_id in others]))
            if others
            else None
        ),
    }


def summarize_paraphrase_ordinal(records: list[dict], delta_summary, draws: int) -> dict:
    """Ordinal contrasts for the paraphrased framings, not only flips."""

    index = index_runs(records, "paraphrase")
    pairs = []
    for (claim_id, system, framing, repeat), row in sorted(index.items()):
        if framing != "believer_paraphrase":
            continue
        other = index.get((claim_id, system, "skeptic_paraphrase", repeat))
        if other is None:
            continue
        pairs.append(
            {
                "claim_id": claim_id,
                "tier": row["tier"],
                "system": system,
                "repeat": repeat,
                "first": score_of(row),
                "second": score_of(other),
                "delta": score_of(row) - score_of(other),
            }
        )
    return {"believer_vs_skeptic": delta_summary(pairs, "paraphrase:bs", draws)}


def summarize_jaccard_sensitivity(
    source_rows: list[dict], records: list[dict], bootstrap_cluster_stat, keyed_seed, draws: int
) -> dict:
    outside = {
        row["run_id"]
        for row in records
        if not row.get("ignored") and row.get("all_citations_in_results") is False
    }
    applicable = [
        row
        for row in source_rows
        if row.get("analysis_applicable") and row.get("subset_runs")
    ]
    if not applicable:
        applicable = [row for row in source_rows if row.get("analysis_applicable")]

    def median_by(rows: list[dict]) -> dict:
        believer = [
            row["domain_jaccard"] for row in rows if row["comparison"] == "believer_skeptic"
        ]
        neutral = [
            row["domain_jaccard"] for row in rows if row["comparison"] == "neutral_repeat"
        ]
        if not believer or not neutral:
            return {"estimable": False}
        return {
            "estimable": True,
            "believer_skeptic_median": float(np.median(believer)),
            "neutral_repeat_median": float(np.median(neutral)),
            "gap": float(np.median(neutral)) - float(np.median(believer)),
            "believer_skeptic_pairs": len(believer),
            "neutral_repeat_pairs": len(neutral),
        }

    full = median_by(applicable)
    clean = median_by(
        [
            row
            for row in applicable
            if not (set(row.get("subset_runs") or []) & outside)
        ]
    )
    permutation = None
    if full.get("estimable"):
        permutation = permutation_gap(applicable, keyed_seed("jaccard_gap"), draws)
    return {
        "all_pairs": full,
        "excluding_unmatched_citations": clean,
        "excluded_runs": len(outside),
        "permutation": permutation,
    }


def permutation_gap(rows: list[dict], seed: int, draws: int) -> dict:
    """Cluster permutation of the comparison label within each claim."""

    rng = np.random.default_rng(seed)
    by_claim: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_claim[row["claim_id"]].append(row)

    def gap(assignment: list[tuple[str, float]]) -> float | None:
        believer = [value for label, value in assignment if label == "believer_skeptic"]
        neutral = [value for label, value in assignment if label == "neutral_repeat"]
        if not believer or not neutral:
            return None
        return float(np.median(neutral)) - float(np.median(believer))

    observed_assignment = [
        (row["comparison"], row["domain_jaccard"]) for row in rows
    ]
    observed = gap(observed_assignment)
    if observed is None:
        return {"estimable": False}
    extreme = 0
    valid = 0
    for _ in range(draws):
        shuffled = []
        for claim_rows in by_claim.values():
            labels = [row["comparison"] for row in claim_rows]
            rng.shuffle(labels)
            shuffled.extend(
                (label, row["domain_jaccard"]) for label, row in zip(labels, claim_rows)
            )
        value = gap(shuffled)
        if value is None:
            continue
        valid += 1
        if value >= observed:
            extreme += 1
    return {
        "estimable": True,
        "observed_gap": observed,
        "draws": valid,
        "p_one_sided": (extreme + 1) / (valid + 1) if valid else None,
    }
