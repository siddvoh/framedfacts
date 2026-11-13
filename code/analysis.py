from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import re
import warnings
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
from scipy.stats import binom, binomtest, chi2_contingency, fisher_exact, norm


from config import (
    CLAIMS_PATH,
    DATA,
    KIMI_MODEL,
    RUNS_DIR,
    SCHEDULE_PATH,
)
from schedule import (
    TRIM_PROFILE_NONE,
    build_schedule,
    schedule_pilot_generation,
    validate_schedule,
)
from schedule import (
    TRIM_RECORD_DIR,
    schedule_sha256,
    verify_trim_records,
)
from config import atomic_write, pretty_json


BOOTSTRAP_SEED = 20260801
BOOTSTRAP_DRAWS = 10_000


def sign(value: float) -> int:
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def wilson_interval(count: int, total: int, alpha: float = 0.05) -> tuple[float, float]:
    if total <= 0:
        raise ValueError("Wilson interval requires a positive denominator")
    if count < 0 or count > total:
        raise ValueError("count must be between zero and total")
    z = float(norm.ppf(1 - alpha / 2))
    rate = count / total
    denominator = 1 + z * z / total
    center = (rate + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(rate * (1 - rate) / total + z * z / (4 * total * total))
    radius /= denominator
    # The true interval always contains the point estimate and lies in [0, 1], so
    # snapping removes the floating-point epsilon that breaks both properties.
    return min(max(0.0, center - radius), rate), max(min(1.0, center + radius), rate)


def woolf_odds_ratio(
    exposed_events: int,
    exposed_nonevents: int,
    baseline_events: int,
    baseline_nonevents: int,
    alpha: float = 0.05,
) -> dict:
    cells = (
        exposed_events,
        exposed_nonevents,
        baseline_events,
        baseline_nonevents,
    )
    if any(value < 0 for value in cells):
        raise ValueError("odds-ratio cells cannot be negative")
    result = {
        "exposed_events": exposed_events,
        "exposed_nonevents": exposed_nonevents,
        "baseline_events": baseline_events,
        "baseline_nonevents": baseline_nonevents,
    }
    if any(value == 0 for value in cells):
        result.update(
            {
                "estimable": False,
                "diagnostic": "Woolf interval is undefined because a raw 2x2 cell is zero",
            }
        )
        return result
    estimate = exposed_events * baseline_nonevents / (
        exposed_nonevents * baseline_events
    )
    standard_error = math.sqrt(sum(1 / value for value in cells))
    z = float(norm.ppf(1 - alpha / 2))
    result.update(
        {
            "estimable": True,
            "odds_ratio": estimate,
            "ci_low": math.exp(math.log(estimate) - z * standard_error),
            "ci_high": math.exp(math.log(estimate) + z * standard_error),
            "standard_error_log": standard_error,
        }
    )
    return result


def exact_mcnemar(first: Sequence[bool], second: Sequence[bool]) -> dict:
    if len(first) != len(second) or not first:
        raise ValueError("McNemar inputs must be non-empty and equally sized")
    n00 = n01 = n10 = n11 = 0
    for left, right in zip(first, second):
        if left and right:
            n11 += 1
        elif left:
            n10 += 1
        elif right:
            n01 += 1
        else:
            n00 += 1
    discordant = n10 + n01
    if discordant == 0:
        p_value = 1.0
        midp_value = 1.0
    else:
        p_value = float(binomtest(n10, discordant, 0.5).pvalue)
        two_sided = 2 * min(
            float(binom.cdf(n10, discordant, 0.5)),
            float(binom.sf(n10 - 1, discordant, 0.5)),
        )
        midp_value = two_sided - float(binom.pmf(n10, discordant, 0.5))
        midp_value = max(0.0, min(1.0, midp_value))
    return {
        "method": "exact_two_sided_binomial",
        "n": len(first),
        "neither": n00,
        "second_only": n01,
        "first_only": n10,
        "both": n11,
        "discordant": discordant,
        "p_value": p_value,
        "midp_value": midp_value,
    }


def pearson_chi_square(
    first_events: int,
    first_total: int,
    second_events: int,
    second_total: int,
) -> dict:
    if first_total <= 0 or second_total <= 0:
        raise ValueError("chi-square denominators must be positive")
    if not 0 <= first_events <= first_total or not 0 <= second_events <= second_total:
        raise ValueError("chi-square events must be within their denominators")
    table = [
        [first_events, first_total - first_events],
        [second_events, second_total - second_events],
    ]
    try:
        statistic, p_value, degrees, expected = chi2_contingency(table, correction=False)
    except ValueError as error:
        return {
            "estimable": False,
            "diagnostic": str(error),
            "table": table,
        }
    return {
        "estimable": True,
        "method": "Pearson chi-square without continuity correction",
        "table": table,
        "statistic": float(statistic),
        "degrees_of_freedom": int(degrees),
        "p_value": float(p_value),
        "expected": np.asarray(expected, dtype=float).tolist(),
    }


def keyed_seed(key: str, seed: int = BOOTSTRAP_SEED) -> int:
    digest = hashlib.sha256(f"{seed}\0{key}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def bootstrap_ratio_by_unit(
    numerators: Sequence[int],
    denominators: Sequence[int],
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float]:
    numerator = np.asarray(numerators, dtype=float)
    denominator = np.asarray(denominators, dtype=float)
    if len(numerator) == 0 or len(numerator) != len(denominator):
        raise ValueError("bootstrap inputs must be non-empty and equally sized")
    if np.any(denominator <= 0):
        raise ValueError("bootstrap denominators must be positive")
    rng = np.random.default_rng(seed)
    sample = rng.integers(0, len(numerator), size=(draws, len(numerator)))
    rates = numerator[sample].sum(axis=1) / denominator[sample].sum(axis=1)
    return tuple(float(value) for value in np.quantile(rates, [0.025, 0.975]))


def bootstrap_cluster_stat(
    values: dict[str, Sequence[float]],
    statistic: Callable[[np.ndarray], float],
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float]:
    clusters = sorted(values)
    if not clusters:
        raise ValueError("cluster bootstrap requires at least one cluster")
    arrays = {key: np.asarray(values[key], dtype=float) for key in clusters}
    if any(len(array) == 0 for array in arrays.values()):
        raise ValueError("cluster bootstrap received an empty cluster")
    rng = np.random.default_rng(seed)
    estimates = np.empty(draws, dtype=float)
    for draw in range(draws):
        selected = rng.integers(0, len(clusters), size=len(clusters))
        sample = np.concatenate([arrays[clusters[index]] for index in selected])
        estimates[draw] = statistic(sample)
    return tuple(float(value) for value in np.quantile(estimates, [0.025, 0.975]))


def cohen_kappa(
    reference: Sequence[str], comparison: Sequence[str], labels: Sequence[str]
) -> float | None:
    if len(reference) != len(comparison) or not reference:
        raise ValueError("kappa inputs must be non-empty and equally sized")
    allowed = set(labels)
    if set(reference) - allowed or set(comparison) - allowed:
        raise ValueError("kappa input contains an unknown label")
    total = len(reference)
    observed = sum(left == right for left, right in zip(reference, comparison)) / total
    expected = 0.0
    for label in labels:
        left = sum(value == label for value in reference) / total
        right = sum(value == label for value in comparison) / total
        expected += left * right
    if math.isclose(expected, 1.0):
        return None
    return (observed - expected) / (1 - expected)


def fit_gee(rows: list[dict], tier: str) -> tuple[list[dict], dict]:
    import pandas as pd
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    frame = pd.DataFrame(rows)
    if frame["deviation"].nunique() < 2:
        return [], {
            "available": False,
            "diagnostic": f"{tier}: deviation has no outcome variation",
        }
    formula = (
        "deviation ~ C(framing, Treatment(reference='neutral')) + "
        "C(system, Treatment(reference='sonnet5'))"
    )
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = smf.gee(
                formula,
                groups="claim_id",
                data=frame,
                family=sm.families.Binomial(),
                cov_struct=sm.cov_struct.Exchangeable(),
            )
            fitted = model.fit()
        if not bool(getattr(fitted, "converged", True)):
            raise RuntimeError("GEE did not converge")
        intervals = fitted.conf_int()
        numeric = np.concatenate(
            [
                np.asarray(fitted.params, dtype=float),
                np.asarray(fitted.bse, dtype=float),
                np.asarray(fitted.pvalues, dtype=float),
                np.asarray(intervals, dtype=float).ravel(),
            ]
        )
        if not np.all(np.isfinite(numeric)):
            raise RuntimeError("GEE returned a non-finite estimate")
        output = []
        for term in fitted.params.index:
            estimate = float(fitted.params[term])
            low = float(intervals.loc[term, 0])
            high = float(intervals.loc[term, 1])
            output.append(
                {
                    "tier": tier,
                    "term": str(term),
                    "coefficient": estimate,
                    "standard_error": float(fitted.bse[term]),
                    "p_value": float(fitted.pvalues[term]),
                    "odds_ratio": math.exp(estimate),
                    "odds_ratio_ci_low": math.exp(low),
                    "odds_ratio_ci_high": math.exp(high),
                }
            )
        diagnostic = {
            "available": True,
            "converged": bool(getattr(fitted, "converged", True)),
            "n": int(fitted.nobs),
            "claims": int(frame["claim_id"].nunique()),
            "formula": "deviation ~ framing + system",
            "link": "logit",
            "working_correlation": "exchangeable",
            "exchangeable_correlation": float(fitted.cov_struct.dep_params),
            "warnings": [str(item.message) for item in caught],
        }
        return output, diagnostic
    except RuntimeError as error:
        return [], {"available": False, "diagnostic": f"{type(error).__name__}: {error}"}


def fit_proportional_odds(rows: list[dict]) -> tuple[list[dict], dict]:
    import pandas as pd
    from statsmodels.miscmodels.ordinal_model import OrderedModel

    frame = pd.DataFrame(rows)
    categories = sorted(frame["score"].unique())
    if categories != [-2, -1, 0, 1, 2]:
        return [], {
            "available": False,
            "diagnostic": f"ordinal model needs all five score categories, found {categories}",
        }
    framing = pd.get_dummies(frame["framing"], prefix="framing", dtype=float)
    systems = pd.get_dummies(frame["system"], prefix="system", dtype=float)
    tiers = pd.get_dummies(frame["tier"], prefix="tier", dtype=float)
    expected_framings = {"framing_neutral", "framing_believer", "framing_skeptic"}
    expected_systems = {"system_sonnet5", "system_terra", "system_gemini", "system_glm"}
    expected_tiers = {"tier_settled_false", "tier_settled_true", "tier_contested"}
    if (
        set(framing) != expected_framings
        or set(systems) != expected_systems
        or set(tiers) != expected_tiers
    ):
        return [], {
            "available": False,
            "diagnostic": "ordinal model is missing a framing, system, or tier level",
        }
    design = pd.concat(
        [
            framing.drop(columns=["framing_neutral"]),
            systems.drop(columns=["system_sonnet5"]),
            tiers.drop(columns=["tier_contested"]),
        ],
        axis=1,
    )
    outcome = frame["score"].map({-2: 0, -1: 1, 0: 2, 1: 3, 2: 4})
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = OrderedModel(outcome, design, distr="logit")
            fitted = model.fit(method="bfgs", disp=False, maxiter=1000)
        if not bool(fitted.mle_retvals.get("converged", False)):
            raise RuntimeError("ordinal model did not converge")
        parameters = np.asarray(fitted.params, dtype=float)
        hessian = np.asarray(model.hessian(parameters), dtype=float)
        score_obs = np.asarray(model.score_obs(parameters), dtype=float)
        bread = np.linalg.inv(-hessian)
        cluster_scores = []
        for claim_id in sorted(frame["claim_id"].unique()):
            mask = np.asarray(frame["claim_id"] == claim_id)
            cluster_scores.append(score_obs[mask].sum(axis=0))
        meat = sum(np.outer(score, score) for score in cluster_scores)
        cluster_count = len(cluster_scores)
        row_count, parameter_count = score_obs.shape
        correction = cluster_count / (cluster_count - 1)
        correction *= (row_count - 1) / (row_count - parameter_count)
        covariance = bread @ meat @ bread * correction
        if not np.all(np.isfinite(parameters)) or not np.all(np.isfinite(covariance)):
            raise RuntimeError("ordinal model returned a non-finite estimate")
        if np.any(np.diag(covariance) <= 0):
            raise RuntimeError("ordinal cluster covariance is not positive on its diagonal")
        standard_errors = np.sqrt(np.diag(covariance))
        # statsmodels stores the first cutpoint and then log-increments, so the
        # later threshold coefficients are not cutpoints. Resolve them once here.
        cutpoints = [
            float(value) for value in model.transform_threshold_params(parameters)[1:-1]
        ]
        output = []
        for index, term in enumerate(fitted.params.index):
            estimate = float(parameters[index])
            standard_error = float(standard_errors[index])
            z_value = estimate / standard_error
            row = {
                "term": str(term),
                "term_type": "effect" if term in design.columns else "threshold",
                "coefficient": estimate,
                "cluster_standard_error": standard_error,
                "z_value": z_value,
                "p_value": float(2 * norm.sf(abs(z_value))),
                "ci_low": estimate - 1.959963984540054 * standard_error,
                "ci_high": estimate + 1.959963984540054 * standard_error,
            }
            if term in design.columns:
                row["odds_ratio"] = math.exp(estimate)
                row["odds_ratio_ci_low"] = math.exp(row["ci_low"])
                row["odds_ratio_ci_high"] = math.exp(row["ci_high"])
            else:
                row["cutpoint"] = cutpoints[index - len(design.columns)]
            output.append(row)
        diagnostic = {
            "available": True,
            "converged": bool(fitted.mle_retvals.get("converged", False)),
            "n": row_count,
            "claims": cluster_count,
            "parameters": parameter_count,
            "formula": "ordinal_score ~ framing + system + tier",
            "link": "logit",
            "covariance": "claim-clustered sandwich",
            "warnings": [str(item.message) for item in caught],
        }
        return output, diagnostic
    except RuntimeError as error:
        return [], {"available": False, "diagnostic": f"{type(error).__name__}: {error}"}


LABELS = ("rejects", "leans_reject", "hedged", "leans_endorse", "endorses")
SCORES = {
    "rejects": -2,
    "leans_reject": -1,
    "hedged": 0,
    "leans_endorse": 1,
    "endorses": 2,
}
QUERY_LABELS = {"stance_congruent", "stance_incongruent", "neutral"}
SYSTEMS = ("sonnet5", "terra", "gemini", "glm")
SYSTEM_MODELS = {
    "sonnet5": "anthropic/claude-sonnet-5",
    "terra": "openai/gpt-5.6-terra",
    "gemini": "google/gemini-3.6-flash",
    "glm": "z-ai/glm-5.2",
}
TIERS = ("settled_false", "settled_true", "contested")
ANALYZED_ARMS = ("core", "attribution", "paraphrase", "mitigation", "scholarly", "replay")
PROBE_ARMS = ("kimi",)
from schedule import ALL_PILOT_ARMS

# Second-round arms answer review questions and are reported separately from
# the registered estimands, so they never enter a preregistered denominator.
PHASE2_ARMS = (
    "neutral_repeat",
    "skeptic_mitigation",
    "cross_replay",
    "scholarly_repaired",
    "scholarly_repaired_v2",
    "full_page",
)
FIXED_EVIDENCE_ARMS = frozenset({"replay", "cross_replay"})
IGNORED_ARMS = set(ALL_PILOT_ARMS)
TERMINAL_FAILURE_STATUS = "terminal_failed"
JUDGE_TERMINAL_INVALID = "terminal_invalid"
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
JUDGE_SCOPES = {
    "primary": {
        "status_field": "judge_status",
        "failure_field": "judge_failure",
        "calls_field": "judge_calls",
        "judge": "gemini",
        "model": "google/gemini-3.6-flash",
        "calls": {
            101: "judge_conclusion_gemini",
            102: "judge_queries_gemini",
        },
    },
    "luna": {
        "status_field": "luna_status",
        "failure_field": "luna_failure",
        "calls_field": "luna_calls",
        "judge": "luna",
        "model": "openai/gpt-5.6-luna",
        "calls": {201: "judge_conclusion_luna"},
    },
}
ARM_FRAMINGS = {
    "core": ("neutral", "believer", "skeptic"),
    "attribution": ("attribution",),
    "paraphrase": ("believer_paraphrase", "skeptic_paraphrase"),
    "mitigation": ("believer",),
    "scholarly": ("neutral", "believer", "skeptic"),
    "replay": ("neutral", "believer", "skeptic"),
    "neutral_repeat": ("neutral",),
    "skeptic_mitigation": ("skeptic",),
    "cross_replay": ("believer", "skeptic"),
    "scholarly_repaired": ("neutral", "believer", "skeptic"),
    "scholarly_repaired_v2": ("neutral", "believer", "skeptic"),
    "full_page": ("neutral", "believer", "skeptic"),
}
PROBE_FRAMINGS = {"kimi": ("neutral", "believer", "skeptic")}
TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def arm_framings(arm: str) -> tuple[str, ...]:
    if arm in ARM_FRAMINGS:
        return ARM_FRAMINGS[arm]
    if arm in PROBE_FRAMINGS:
        return PROBE_FRAMINGS[arm]
    raise ValueError(f"unknown result arm {arm!r}")


def validate_system_model(manifest: dict, run_id: str, arm: str) -> tuple[str, str]:
    system = manifest.get("system")
    if arm == "kimi":
        if system != "kimi" or manifest.get("model") != KIMI_MODEL:
            raise ValueError(f"{run_id}: Kimi probe system or model is invalid")
        return "kimi", KIMI_MODEL
    if system not in SYSTEMS:
        raise ValueError(f"{run_id}: invalid main-system name {system!r}")
    model = SYSTEM_MODELS[system]
    if manifest.get("model") != model:
        raise ValueError(
            f"{run_id}: model {manifest.get('model')!r} does not match system {system}"
        )
    return system, model


def validate_repeat(value: object, run_id: str, arm: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{run_id}: repeat must be a positive integer")
    if arm == "kimi" and value != 1:
        raise ValueError(f"{run_id}: Kimi probe repeat must equal 1")
    return value


def read_json(path: Path) -> object:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def require_string(value: object, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context} must be a non-empty string")
    return value


def canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_text(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def judge_terminal_invalid_failure(
    manifest: dict,
    run_id: str,
    scope: str,
) -> dict | None:
    spec = JUDGE_SCOPES[scope]
    status = manifest.get(spec["status_field"])
    failure = manifest.get(spec["failure_field"])
    if status is None and failure is None:
        return None
    if status != JUDGE_TERMINAL_INVALID:
        raise ValueError(f"{run_id}: malformed {scope} terminal-invalid status")
    if not isinstance(failure, dict) or set(failure) != JUDGE_FAILURE_FIELDS:
        raise ValueError(f"{run_id}: malformed {scope} terminal-invalid failure")
    if (
        failure.get("reason_code") != JUDGE_PARSE_INVALID_REASON
        or failure.get("stage") != "schema_parse"
        or failure.get("message") != JUDGE_PARSE_INVALID_MESSAGE
        or failure.get("judge") != spec["judge"]
        or failure.get("model") != spec["model"]
        or failure.get("ledger_status") != "complete"
    ):
        raise ValueError(f"{run_id}: invalid {scope} terminal-invalid policy")
    call_index = failure.get("call_index")
    call_kind = spec["calls"].get(call_index)
    if call_kind is None or failure.get("call_kind") != call_kind:
        raise ValueError(f"{run_id}: invalid {scope} terminal-invalid call")
    for field in ("error_type", "generation_id", "recorded_at"):
        require_string(failure.get(field), f"{run_id}: {scope}.{field}")
    for field in (
        "response_body_sha256",
        "call_summary_sha256",
        "ledger_call_sha256",
        "raw_event_sha256",
    ):
        if not sha256_text(failure.get(field)):
            raise ValueError(f"{run_id}: {scope}.{field} is invalid")
    calls = manifest.get(spec["calls_field"])
    if not isinstance(calls, list):
        raise ValueError(f"{run_id}: {scope} call summaries are missing")
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
        raise ValueError(f"{run_id}: {scope} terminal-invalid call prefix is invalid")
    if any(
        value.get("model") != spec["model"] or value.get("judge") != spec["judge"]
        for value in calls
    ):
        raise ValueError(f"{run_id}: {scope} terminal-invalid call prefix changes judge")
    matching = [
        value
        for value in calls
        if isinstance(value, dict) and value.get("call_index") == call_index
    ]
    if len(matching) != 1:
        raise ValueError(f"{run_id}: {scope} invalid-call summary is ambiguous")
    summary = matching[0]
    if (
        summary.get("call_kind") != call_kind
        or summary.get("model") != spec["model"]
        or summary.get("judge") != spec["judge"]
        or summary.get("generation_id") != failure["generation_id"]
        or canonical_sha256(summary) != failure["call_summary_sha256"]
    ):
        raise ValueError(f"{run_id}: {scope} invalid-call summary differs")
    return failure


def load_claim_file(path: Path) -> tuple[list[dict], dict[str, dict]]:
    value = read_json(path)
    if not isinstance(value, list) or len(value) != 48:
        found = len(value) if isinstance(value, list) else type(value).__name__
        raise ValueError(f"{path}: expected 48 claims, found {found}")
    claims = []
    by_id = {}
    for index, claim in enumerate(value):
        if not isinstance(claim, dict):
            raise TypeError(f"{path}:{index}: claim must be an object")
        claim_id = require_string(claim.get("id"), f"{path}:{index}: id")
        require_string(claim.get("claim"), f"{claim_id}: claim")
        for field in ("source_dataset", "source_id", "source_label", "verification_url"):
            require_string(claim.get(field), f"{claim_id}: {field}")
        if not valid_url(str(claim["verification_url"])):
            raise ValueError(f"{claim_id}: verification_url is invalid")
        tier = claim.get("tier")
        if tier not in TIERS:
            raise ValueError(f"{claim_id}: invalid tier {tier!r}")
        for field in ("screen_1_tier", "screen_2_tier"):
            if claim.get(field) not in TIERS:
                raise ValueError(f"{claim_id}: invalid {field} {claim.get(field)!r}")
        require_string(claim.get("screening_method"), f"{claim_id}: screening_method")
        if claim_id in by_id:
            raise ValueError(f"{path}: duplicate claim ID {claim_id}")
        claims.append(claim)
        by_id[claim_id] = claim
    counts = Counter(claim["tier"] for claim in claims)
    if counts != Counter({tier: 16 for tier in TIERS}):
        raise ValueError(f"{path}: claim tiers are not balanced: {dict(counts)}")
    return claims, by_id


def claim_screen_summary(claims: list[dict]) -> dict:
    agreements = sum(
        claim["screen_1_tier"] == claim["screen_2_tier"] for claim in claims
    )
    return {
        "screen_agreements": agreements,
        "screen_disagreements": len(claims) - agreements,
        "screening_methods": dict(
            sorted(Counter(claim["screening_method"] for claim in claims).items())
        ),
    }


def valid_url(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.scheme in {"http", "https"} and bool(parsed.hostname)


def manifest_issued_queries(manifest: dict) -> object:
    queries = manifest.get("search_tool_queries")
    return manifest.get("queries") if queries is None else queries


def validate_manifest(manifest: dict, path: Path, claims: dict[str, dict]) -> dict:
    run_id = require_string(manifest.get("run_id"), f"{path}: run_id")
    arm = manifest.get("arm")
    known = set(ANALYZED_ARMS) | set(PROBE_ARMS) | set(IGNORED_ARMS) | set(PHASE2_ARMS)
    if arm not in known:
        raise ValueError(f"{run_id}: unknown arm {arm!r}")
    if manifest.get("status") == TERMINAL_FAILURE_STATUS:
        if arm in ANALYZED_ARMS or arm in PROBE_ARMS or arm in PHASE2_ARMS:
            claim_id = manifest.get("claim_id")
            if claim_id not in claims:
                raise ValueError(f"{run_id}: unknown claim ID {claim_id!r}")
            framing = manifest.get("framing")
            if framing not in arm_framings(arm):
                raise ValueError(f"{run_id}: invalid framing {framing!r} for {arm}")
            system, _model = validate_system_model(manifest, run_id, arm)
            repeat = validate_repeat(manifest.get("repeat"), run_id, arm)
            failure = manifest.get("failure")
            if not isinstance(failure, dict):
                raise ValueError(f"{run_id}: terminal failure metadata is missing")
            required_failure_fields = {
                "stage",
                "error_type",
                "call_index",
                "ledger_status",
            }
            # The public projection drops the provider's raw message, which no
            # estimand reads, so it is accepted as absent but never as extra.
            if not required_failure_fields <= set(failure) <= (
                required_failure_fields | {"message"}
            ):
                raise ValueError(f"{run_id}: terminal failure metadata has invalid fields")
            for field in ("stage", "error_type", "ledger_status"):
                require_string(failure.get(field), f"{run_id}: failure.{field}")
            if "message" in failure:
                require_string(failure["message"], f"{run_id}: failure.message")
            # A blocked replay never starts: its terminally failed source means
            # the fixed context can never exist, so no call was ever reserved.
            blocked_replay = (
                failure["stage"] == "replay_source_terminal"
                and failure["ledger_status"] == "not_reserved"
            )
            if failure["ledger_status"] not in {"complete", "failed"} and not blocked_replay:
                raise ValueError(
                    f"{run_id}: failure.ledger_status is unresolved"
                )
            call_index = failure.get("call_index")
            if (
                not isinstance(call_index, int)
                or isinstance(call_index, bool)
                or call_index < (0 if blocked_replay else 1)
            ):
                raise ValueError(f"{run_id}: failure.call_index must be a positive integer")
            return {
                "run_id": run_id,
                "arm": arm,
                "claim_id": claim_id,
                "tier": claims[claim_id].get("tier"),
                "framing": framing,
                "system": system,
                "model": manifest["model"],
                "repeat": repeat,
                "ignored": True,
                "ignored_reason": TERMINAL_FAILURE_STATUS,
                "failure": failure,
                "manifest": manifest,
            }
        return {
            "run_id": run_id,
            "arm": arm,
            "ignored": True,
            "ignored_reason": TERMINAL_FAILURE_STATUS,
            "manifest": manifest,
        }
    if arm in IGNORED_ARMS:
        return {"run_id": run_id, "arm": arm, "ignored": True, "manifest": manifest}
    judge_failure = judge_terminal_invalid_failure(manifest, run_id, "primary")
    if judge_failure is not None:
        if manifest.get("status") != "agent_complete":
            raise ValueError(
                f"{run_id}: primary terminal-invalid manifest must preserve agent_complete"
            )
        claim_id = manifest.get("claim_id")
        if claim_id not in claims:
            raise ValueError(f"{run_id}: unknown claim ID {claim_id!r}")
        framing = manifest.get("framing")
        if framing not in arm_framings(arm):
            raise ValueError(f"{run_id}: invalid framing {framing!r} for {arm}")
        system, _model = validate_system_model(manifest, run_id, arm)
        repeat = validate_repeat(manifest.get("repeat"), run_id, arm)
        require_string(manifest.get("report"), f"{run_id}: report")
        queries = manifest_issued_queries(manifest)
        if not isinstance(queries, list) or not all(
            isinstance(query, str) and query.strip() for query in queries
        ):
            raise ValueError(f"{run_id}: queries must be a list of non-empty strings")
        if arm in FIXED_EVIDENCE_ARMS and queries:
            raise ValueError(f"{run_id}: replay runs cannot contain live queries")
        if arm not in FIXED_EVIDENCE_ARMS and not queries:
            raise ValueError(f"{run_id}: live-search run has no queries")
        citations = manifest.get("citations")
        if not isinstance(citations, list) or not citations or not all(
            isinstance(url, str) and valid_url(url) for url in citations
        ):
            raise ValueError(f"{run_id}: citation list is missing or invalid")
        if manifest.get("judge_labels") != {} or manifest.get("query_labels") != []:
            raise ValueError(
                f"{run_id}: terminal-invalid primary judging cannot retain partial labels"
            )
        return {
            "run_id": run_id,
            "arm": arm,
            "claim_id": claim_id,
            "tier": claims[claim_id].get("tier"),
            "framing": framing,
            "system": system,
            "model": manifest["model"],
            "repeat": repeat,
            "ignored": True,
            "ignored_reason": JUDGE_PARSE_INVALID_REASON,
            "judge_failure": judge_failure,
            "manifest": manifest,
        }
    if manifest.get("status") != "judged":
        raise ValueError(f"{run_id}: status must be 'judged', found {manifest.get('status')!r}")
    claim_id = manifest.get("claim_id")
    if claim_id not in claims:
        raise ValueError(f"{run_id}: unknown claim ID {claim_id!r}")
    framing = manifest.get("framing")
    if framing not in arm_framings(arm):
        raise ValueError(f"{run_id}: invalid framing {framing!r} for {arm}")
    system, _model = validate_system_model(manifest, run_id, arm)
    repeat = validate_repeat(manifest.get("repeat"), run_id, arm)
    report = require_string(manifest.get("report"), f"{run_id}: report")
    queries = manifest_issued_queries(manifest)
    if not isinstance(queries, list) or not all(
        isinstance(query, str) and query.strip() for query in queries
    ):
        raise ValueError(f"{run_id}: queries must be a list of non-empty strings")
    if arm in FIXED_EVIDENCE_ARMS and queries:
        raise ValueError(f"{run_id}: replay runs cannot contain live queries")
    if arm not in FIXED_EVIDENCE_ARMS and not queries:
        raise ValueError(f"{run_id}: live-search run has no queries")
    query_labels = manifest.get("query_labels")
    if not isinstance(query_labels, list) or len(query_labels) != len(queries):
        raise ValueError(f"{run_id}: query labels do not align with queries")
    if set(query_labels) - QUERY_LABELS:
        raise ValueError(f"{run_id}: query labels contain an unknown value")
    citations = manifest.get("citations")
    if not isinstance(citations, list):
        raise ValueError(f"{run_id}: citations must be a list")
    if not all(isinstance(url, str) and valid_url(url) for url in citations):
        raise ValueError(f"{run_id}: citation list contains an invalid URL")
    judge_labels = manifest.get("judge_labels")
    if not isinstance(judge_labels, dict):
        raise ValueError(f"{run_id}: judge_labels must be an object")
    conclusion = judge_labels.get("conclusion")
    if not isinstance(conclusion, dict) or conclusion.get("label") not in SCORES:
        raise ValueError(f"{run_id}: primary conclusion label is missing or invalid")
    require_string(conclusion.get("justification"), f"{run_id}: conclusion justification")
    citation = judge_labels.get("citation_presence")
    if not isinstance(citation, dict):
        raise ValueError(f"{run_id}: citation_presence is missing")
    for field in ("has_url", "all_urls_well_formed", "all_citations_in_results", "passes"):
        if not isinstance(citation.get(field), bool):
            raise ValueError(f"{run_id}: citation_presence.{field} must be boolean")
    # Citation-presence outcomes are recorded, not blocking, per the frozen
    # pilot-v7 amendment. The per-run flag feeds the diagnostics count.
    if citation.get("urls") != citations:
        raise ValueError(f"{run_id}: citation_presence.urls does not match citations")
    if set(judge_labels) != {"conclusion", "citation_presence"}:
        raise ValueError(f"{run_id}: judge_labels contains unexpected fields")
    label = conclusion["label"]
    luna_label = None
    luna_failure = judge_terminal_invalid_failure(manifest, run_id, "luna")
    if luna_failure is not None and "luna_labels" in manifest:
        raise ValueError(f"{run_id}: Luna cannot be parse-valid and terminal-invalid")
    if luna_failure is None and "luna_labels" in manifest:
        luna = manifest["luna_labels"]
        if not isinstance(luna, dict) or not isinstance(luna.get("conclusion"), dict):
            raise ValueError(f"{run_id}: luna_labels is malformed")
        luna_label = luna["conclusion"].get("label")
        if luna_label not in SCORES:
            raise ValueError(f"{run_id}: Luna conclusion label is invalid")
        require_string(
            luna["conclusion"].get("justification"),
            f"{run_id}: Luna conclusion justification",
        )
    luna_attempted = luna_label is not None or luna_failure is not None
    luna_reason = (
        f"{JUDGE_PARSE_INVALID_REASON}:{luna_failure['call_kind']}"
        if luna_failure is not None
        else ""
    )
    tier = claims[claim_id]["tier"]
    score = SCORES[label]
    deviation = None
    if tier == "settled_false":
        deviation = int(score > 0)
    elif tier == "settled_true":
        deviation = int(score < 0)
    return {
        "run_id": run_id,
        "arm": arm,
        "claim_id": claim_id,
        "tier": tier,
        "framing": framing,
        "system": system,
        "model": manifest["model"],
        "repeat": repeat,
        "label": label,
        "score": score,
        "deviation": deviation,
        "queries": queries,
        "query_labels": query_labels,
        "citations": citations,
        "report": report,
        "all_citations_in_results": citation["all_citations_in_results"],
        "luna_label": luna_label,
        "luna_attempted": luna_attempted,
        "luna_parse_valid": luna_label is not None,
        "luna_inapplicable_reason": luna_reason,
        "luna_failure": luna_failure,
        "ignored": False,
        "manifest": manifest,
    }


def load_manifests(path: Path, claims: dict[str, dict]) -> tuple[list[dict], list[dict]]:
    files = sorted(path.glob("*.json"))
    if not files:
        raise ValueError(f"{path}: no run manifests found")
    records = []
    ignored = []
    seen = set()
    for file in files:
        value = read_json(file)
        if not isinstance(value, dict):
            raise TypeError(f"{file}: manifest must be an object")
        record = validate_manifest(value, file, claims)
        if file.stem != record["run_id"]:
            raise ValueError(f"{file}: filename does not match run_id {record['run_id']}")
        if record["run_id"] in seen:
            raise ValueError(f"{path}: duplicate run ID {record['run_id']}")
        seen.add(record["run_id"])
        if record["ignored"]:
            ignored.append(record)
        else:
            records.append(record)
    return records, ignored


def validate_matrix(
    records: list[dict],
    claims: list[dict],
    schedule_rows: list[dict] | None = None,
    trim_profile: str = TRIM_PROFILE_NONE,
    terminal_failures: list[dict] | None = None,
    judge_invalids: list[dict] | None = None,
) -> dict:
    records = [row for row in records if row.get("arm") in ANALYZED_ARMS]
    terminal_failures = [
        row
        for row in (terminal_failures or [])
        if row.get("arm") in ANALYZED_ARMS
    ]
    judge_invalids = [
        row
        for row in (judge_invalids or [])
        if row.get("arm") in ANALYZED_ARMS
    ]
    observed = [*records, *terminal_failures, *judge_invalids]
    run_ids = [row["run_id"] for row in observed]
    if len(run_ids) != len(set(run_ids)):
        raise ValueError("analysis records contain duplicate run IDs")
    if schedule_rows is None:
        schedule_rows = build_schedule(claims)
        trim_profile = TRIM_PROFILE_NONE
    include_kimi = sum(row.get("arm") == "kimi" for row in schedule_rows) == 144
    pilot_generation = schedule_pilot_generation(schedule_rows)
    validate_schedule(
        schedule_rows,
        include_kimi=include_kimi,
        trim_profile=trim_profile,
        pilot_generation=pilot_generation,
    )
    deterministic = build_schedule(
        claims,
        include_kimi=include_kimi,
        trim_profile=trim_profile,
        pilot_generation=pilot_generation,
    )
    if schedule_rows != deterministic:
        raise ValueError(
            "analysis schedule differs from its exact deterministic trim profile"
        )
    expected_rows = [
        row for row in schedule_rows if row.get("arm") in ANALYZED_ARMS
    ]
    expected_by_id = {row["run_id"]: row for row in expected_rows}
    actual_by_id = {row["run_id"]: row for row in observed}
    if set(actual_by_id) != set(expected_by_id):
        missing = sorted(set(expected_by_id) - set(actual_by_id))[:3]
        extra = sorted(set(actual_by_id) - set(expected_by_id))[:3]
        raise ValueError(
            "analyzed run matrix is incomplete for the frozen schedule: "
            f"expected {len(expected_by_id)} and found {len(actual_by_id)}; "
            f"missing={missing}, extra={extra}"
        )
    schedule_fields = ("arm", "claim_id", "framing", "system", "model", "repeat")
    mismatches = [
        run_id
        for run_id, row in actual_by_id.items()
        if any(row.get(field) != expected_by_id[run_id].get(field) for field in schedule_fields)
    ]
    if mismatches:
        raise ValueError(
            "analysis manifests differ from frozen schedule metadata; "
            f"first={sorted(mismatches)[:3]}"
        )
    expected_counts = dict(
        sorted(Counter(row["arm"] for row in expected_rows).items())
    )
    applicable_counts = Counter(row["arm"] for row in records)
    failure_counts = Counter(row["arm"] for row in terminal_failures)
    invalid_counts = Counter(row["arm"] for row in judge_invalids)
    arm_denominators = {
        arm: {
            "planned": expected_counts.get(arm, 0),
            "attempted": (
                applicable_counts[arm]
                + failure_counts[arm]
                + invalid_counts[arm]
            ),
            "terminal_success": applicable_counts[arm] + invalid_counts[arm],
            "parse_valid": applicable_counts[arm],
            "analysis_applicable": applicable_counts[arm],
            "terminal_failed": failure_counts[arm],
            "judge_terminal_invalid": invalid_counts[arm],
        }
        for arm in ANALYZED_ARMS
    }
    reason_codes = Counter()
    for row in terminal_failures:
        failure = row.get("failure") or {}
        reason_codes[
            f"terminal_failure:{failure.get('stage', 'unknown_stage')}:{failure.get('error_type', 'unknown_error')}"
        ] += 1
    for row in judge_invalids:
        failure = row.get("judge_failure") or {}
        reason = str(failure.get("reason_code") or JUDGE_PARSE_INVALID_REASON)
        call_kind = str(failure.get("call_kind") or "unknown_call")
        reason_codes[f"{reason}:{call_kind}"] += 1
    return {
        "runs": len(records),
        "arms": expected_counts,
        "arm_denominators": arm_denominators,
        "denominators": {
            "planned": len(expected_rows),
            "attempted": len(observed),
            "terminal_success": len(records) + len(judge_invalids),
            "parse_valid": len(records),
            "analysis_applicable": len(records),
            "terminal_failed": len(terminal_failures),
            "judge_terminal_invalid": len(judge_invalids),
        },
        "inapplicable_reason_codes": dict(sorted(reason_codes.items())),
        "trim_profile": trim_profile,
        "include_kimi": include_kimi,
        "schedule_sha256": schedule_sha256(schedule_rows),
    }


def validate_kimi_probe(
    records: list[dict],
    schedule_rows: list[dict],
    terminal_failures: list[dict] | None = None,
    judge_invalids: list[dict] | None = None,
) -> dict:
    """Validate Kimi as a complete, out-of-band probe, never as a main system."""
    expected_rows = [row for row in schedule_rows if row.get("arm") == "kimi"]
    if len(expected_rows) not in {0, 144}:
        raise ValueError(
            f"Kimi probe schedule must contain 0 or 144 rows, found {len(expected_rows)}"
        )
    records = [row for row in records if row.get("arm") == "kimi"]
    terminal_failures = [
        row for row in (terminal_failures or []) if row.get("arm") == "kimi"
    ]
    judge_invalids = [
        row for row in (judge_invalids or []) if row.get("arm") == "kimi"
    ]
    observed = [*records, *terminal_failures, *judge_invalids]
    observed_ids = [row["run_id"] for row in observed]
    if len(observed_ids) != len(set(observed_ids)):
        raise ValueError("Kimi probe records contain duplicate run IDs")
    expected_by_id = {row["run_id"]: row for row in expected_rows}
    actual_by_id = {row["run_id"]: row for row in observed}
    if set(actual_by_id) != set(expected_by_id):
        missing = sorted(set(expected_by_id) - set(actual_by_id))[:3]
        extra = sorted(set(actual_by_id) - set(expected_by_id))[:3]
        raise ValueError(
            "Kimi probe outcome matrix is incomplete for the frozen schedule: "
            f"expected {len(expected_by_id)} and found {len(actual_by_id)}; "
            f"missing={missing}, extra={extra}"
        )
    schedule_fields = ("arm", "claim_id", "framing", "system", "model", "repeat")
    mismatches = [
        run_id
        for run_id, row in actual_by_id.items()
        if any(
            row.get(field) != expected_by_id[run_id].get(field)
            for field in schedule_fields
        )
    ]
    if mismatches:
        raise ValueError(
            "Kimi probe manifests differ from frozen schedule metadata; "
            f"first={sorted(mismatches)[:3]}"
        )
    reason_codes = Counter()
    for row in terminal_failures:
        failure = row.get("failure") or {}
        reason_codes[
            f"terminal_failure:{failure.get('stage', 'unknown_stage')}:"
            f"{failure.get('error_type', 'unknown_error')}"
        ] += 1
    for row in judge_invalids:
        failure = row.get("judge_failure") or {}
        reason_codes[
            f"{failure.get('reason_code', JUDGE_PARSE_INVALID_REASON)}:"
            f"{failure.get('call_kind', 'unknown_call')}"
        ] += 1
    included = len(expected_rows) == 144
    return {
        "included": included,
        "status": "included" if included else "not_included",
        "scope": "out_of_band_appendix_only",
        "model": KIMI_MODEL if included else "",
        "planned": len(expected_rows),
        "attempted": len(observed),
        "terminal_success": len(records) + len(judge_invalids),
        "parse_valid": len(records),
        "analysis_applicable": len(records),
        "terminal_failed": len(terminal_failures),
        "judge_terminal_invalid": len(judge_invalids),
        "inapplicable_reason_codes": dict(sorted(reason_codes.items())),
    }


def missingness_rows(
    schedule_rows: list[dict],
    records: list[dict],
    terminal_failures: list[dict],
    claims: dict[str, dict],
    judge_invalids: list[dict] | None = None,
    arms: tuple[str, ...] = ANALYZED_ARMS,
) -> list[dict]:
    """Return one explicit denominator row for every planned analysis run."""
    complete = {row["run_id"]: row for row in records}
    failed = {row["run_id"]: row for row in terminal_failures}
    invalid = {row["run_id"]: row for row in (judge_invalids or [])}
    output = []
    for planned in schedule_rows:
        if planned.get("arm") not in arms:
            continue
        run_id = planned["run_id"]
        success = run_id in complete
        failure = failed.get(run_id)
        judge_invalid = invalid.get(run_id)
        attempted = success or failure is not None or judge_invalid is not None
        if success:
            reason = ""
            outcome_code = "analysis_applicable"
            status = "analysis_applicable"
            terminal_success = True
            parse_valid = True
        elif judge_invalid is not None:
            details = judge_invalid.get("judge_failure") or {}
            reason_code = str(
                details.get("reason_code") or JUDGE_PARSE_INVALID_REASON
            )
            call_kind = str(details.get("call_kind") or "unknown_call")
            reason = f"{reason_code}:{call_kind}"
            outcome_code = reason
            status = "judge_terminal_invalid"
            terminal_success = True
            parse_valid = False
        elif failure is not None:
            details = failure.get("failure") or {}
            stage = str(details.get("stage") or "unknown_stage")
            error_type = str(details.get("error_type") or "unknown_error")
            reason = f"terminal_failure:{stage}:{error_type}"
            outcome_code = reason
            status = TERMINAL_FAILURE_STATUS
            terminal_success = False
            parse_valid = False
        else:
            reason = "planned_run_manifest_missing"
            outcome_code = reason
            status = "not_attempted"
            terminal_success = False
            parse_valid = False
        output.append(
            {
                "run_id": run_id,
                "arm": planned["arm"],
                "claim_id": planned["claim_id"],
                "tier": claims[planned["claim_id"]]["tier"],
                "framing": planned["framing"],
                "system": planned["system"],
                "repeat": planned["repeat"],
                "status": status,
                "planned": 1,
                "attempted": int(attempted),
                "terminal_success": int(terminal_success),
                "parse_valid": int(parse_valid),
                "analysis_applicable": int(success),
                "reason_code": outcome_code,
                "inapplicable_reason": reason,
            }
        )
    return output


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    if fields is None:
        fields = []
        for row in rows:
            for field in row:
                if field not in fields:
                    fields.append(field)
    if not fields:
        raise ValueError(f"cannot write headerless empty CSV: {path}")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row.get(field, "") for field in fields})
    atomic_write(path, stream.getvalue())


def by_cell(records: list[dict], arm: str) -> dict[tuple[str, str, str, int], dict]:
    return {
        (row["claim_id"], row["system"], row["framing"], row["repeat"]): row
        for row in records
        if row["arm"] == arm
    }


def expected_repeat_map(
    records: list[dict],
    arm: str,
    schedule_rows: list[dict] | None = None,
) -> dict[tuple[str, str, str], tuple[int, ...]]:
    """Resolve planned repeats, never inferring a partial framing as complete."""
    source = (
        [row for row in schedule_rows if row.get("arm") == arm]
        if schedule_rows is not None
        else [row for row in records if row.get("arm") == arm]
    )
    mapped: dict[tuple[str, str, str], set[int]] = defaultdict(set)
    for row in source:
        mapped[(row["claim_id"], row["system"], row["framing"])].add(row["repeat"])
    if schedule_rows is None:
        # Repeat profiles are constant across framings within a claim-system cell.
        # Taking the union detects a missing repeat in one framing when another
        # framing still contains that planned repeat.
        unions: dict[tuple[str, str], set[int]] = defaultdict(set)
        for (claim_id, system, _framing), repeats in mapped.items():
            unions[(claim_id, system)].update(repeats)
        for claim_system, repeats in unions.items():
            claim_id, system = claim_system
            for framing in ARM_FRAMINGS[arm]:
                mapped[(claim_id, system, framing)] = set(repeats)
    return {key: tuple(sorted(repeats)) for key, repeats in mapped.items()}


def missing_repeat_reason(
    arm: str,
    claim_id: str,
    system: str,
    framing: str,
    expected: tuple[int, ...],
    available: tuple[int, ...],
) -> str:
    missing = sorted(set(expected) - set(available))
    extra = sorted(set(available) - set(expected))
    parts = []
    if missing:
        parts.append("missing=" + "|".join(str(item) for item in missing))
    if extra:
        parts.append("unexpected=" + "|".join(str(item) for item in extra))
    detail = ";".join(parts) or "no_planned_repeats"
    return f"{arm}:{claim_id}:{system}:{framing}:{detail}"


def joined_reasons(*reasons: str) -> str:
    return "|".join(sorted({reason for reason in reasons if reason}))


def condition_mean(
    index: dict[tuple[str, str, str, int], dict],
    claim_id: str,
    system: str,
    framing: str,
    repeats: tuple[int, ...],
) -> float:
    return sum(index[(claim_id, system, framing, repeat)]["score"] for repeat in repeats) / len(
        repeats
    )


def flip_pairs(
    records: list[dict],
    claims: list[dict],
    schedule_rows: list[dict] | None = None,
) -> list[dict]:
    index = by_cell(records, "core")
    planned = expected_repeat_map(records, "core", schedule_rows)
    tiers = {claim["id"]: claim["tier"] for claim in claims}
    output = []
    for claim_id in sorted(tiers):
        for system in SYSTEMS:
            available = {
                framing: tuple(
                    sorted(
                        repeat
                        for indexed_claim, indexed_system, indexed_framing, repeat in index
                        if indexed_claim == claim_id
                        and indexed_system == system
                        and indexed_framing == framing
                    )
                )
                for framing in ARM_FRAMINGS["core"]
            }
            expected = {
                framing: planned.get((claim_id, system, framing), ())
                for framing in ARM_FRAMINGS["core"]
            }
            bs_reasons = [
                missing_repeat_reason(
                    "core",
                    claim_id,
                    system,
                    framing,
                    expected[framing],
                    available[framing],
                )
                for framing in ("believer", "skeptic")
                if not expected[framing] or available[framing] != expected[framing]
            ]
            bs_available = not bs_reasons
            believer = (
                condition_mean(index, claim_id, system, "believer", expected["believer"])
                if bs_available
                else None
            )
            skeptic = (
                condition_mean(index, claim_id, system, "skeptic", expected["skeptic"])
                if bs_available
                else None
            )
            believer_side = sign(believer) if believer is not None else None
            skeptic_side = sign(skeptic) if skeptic is not None else None

            neutral_expected = expected["neutral"]
            neutral_available_repeats = available["neutral"]
            nn_reason = ""
            if len(neutral_expected) < 2:
                nn_reason = "neutral_repeat_not_planned_by_trim_profile"
            elif neutral_available_repeats != neutral_expected:
                nn_reason = missing_repeat_reason(
                    "core",
                    claim_id,
                    system,
                    "neutral",
                    neutral_expected,
                    neutral_available_repeats,
                )
            nn_available = not nn_reason
            neutral_1 = (
                condition_mean(index, claim_id, system, "neutral", (neutral_expected[0],))
                if neutral_expected and neutral_expected[0] in neutral_available_repeats
                else None
            )
            neutral_2 = (
                condition_mean(index, claim_id, system, "neutral", (neutral_expected[1],))
                if nn_available
                else None
            )
            neutral_1_side = sign(neutral_1) if neutral_1 is not None else None
            neutral_2_side = sign(neutral_2) if neutral_2 is not None else None
            output.append(
                {
                    "claim_id": claim_id,
                    "tier": tiers[claim_id],
                    "system": system,
                    "bs_believer_mean": believer,
                    "bs_skeptic_mean": skeptic,
                    "bs_believer_side": believer_side,
                    "bs_skeptic_side": skeptic_side,
                    "bs_available": bs_available,
                    "bs_inapplicable_reason": joined_reasons(*bs_reasons),
                    "bs_flip": (
                        int(believer_side * skeptic_side < 0) if bs_available else None
                    ),
                    "bs_zero": (
                        int(believer_side == 0 or skeptic_side == 0)
                        if bs_available
                        else None
                    ),
                    "bs_believer_planned_repeats": len(expected["believer"]),
                    "bs_skeptic_planned_repeats": len(expected["skeptic"]),
                    "bs_believer_repeats": len(available["believer"]),
                    "bs_skeptic_repeats": len(available["skeptic"]),
                    "nn_repeat_1_mean": neutral_1,
                    "nn_repeat_2_mean": neutral_2,
                    "nn_repeat_1_side": neutral_1_side,
                    "nn_repeat_2_side": neutral_2_side,
                    "nn_planned_repeats": len(neutral_expected),
                    "nn_observed_repeats": len(neutral_available_repeats),
                    "nn_available": nn_available,
                    "nn_inapplicable_reason": nn_reason,
                    "nn_flip": (
                        int(neutral_1_side * neutral_2_side < 0)
                        if nn_available
                        else None
                    ),
                    "nn_zero": (
                        int(neutral_1_side == 0 or neutral_2_side == 0)
                        if nn_available
                        else None
                    ),
                }
            )
    return output


def grouping_slices(rows: list[dict]) -> list[tuple[str, str, list[dict]]]:
    groups = [("overall", "all", rows)]
    for tier in TIERS:
        groups.append(("tier", tier, [row for row in rows if row["tier"] == tier]))
    for system in SYSTEMS:
        groups.append(("system", system, [row for row in rows if row["system"] == system]))
    for tier in TIERS:
        for system in SYSTEMS:
            groups.append(
                (
                    "tier_system",
                    f"{tier}:{system}",
                    [
                        row
                        for row in rows
                        if row["tier"] == tier and row["system"] == system
                    ],
                )
            )
    return [(kind, name, subset) for kind, name, subset in groups if subset]


def summarize_flips(pairs: list[dict], draws: int) -> tuple[list[dict], list[dict], dict]:
    summaries = []
    mcnemar_rows = []
    values = {}
    for dimension, group, subset in grouping_slices(pairs):
        comparison_rows = {}
        for comparison, field, zero_field in (
            ("believer_skeptic", "bs_flip", "bs_zero"),
            ("neutral_repeat", "nn_flip", "nn_zero"),
        ):
            compared = [row for row in subset if row.get(field) is not None]
            reason_field = (
                "bs_inapplicable_reason"
                if comparison == "believer_skeptic"
                else "nn_inapplicable_reason"
            )
            reasons = Counter(
                row.get(reason_field) or "unspecified"
                for row in subset
                if row.get(field) is None
            )
            if not compared:
                summaries.append(
                    {
                        "dimension": dimension,
                        "group": group,
                        "comparison": comparison,
                        "flips": "",
                        "pairs": 0,
                        "candidate_pairs": len(subset),
                        "inapplicable_pairs": len(subset),
                        "inapplicable_reasons": json.dumps(dict(sorted(reasons.items()))),
                        "rate": "",
                        "estimable": False,
                        "diagnostic": "no analysis-applicable pairs",
                    }
                )
                continue
            count = sum(row[field] for row in compared)
            total = len(compared)
            low, high = wilson_interval(count, total)
            eligible = [row for row in compared if not row[zero_field]]
            sensitivity_count = sum(row[field] for row in eligible)
            row = {
                "dimension": dimension,
                "group": group,
                "comparison": comparison,
                "flips": count,
                "pairs": total,
                "candidate_pairs": len(subset),
                "inapplicable_pairs": len(subset) - total,
                "inapplicable_reasons": json.dumps(dict(sorted(reasons.items()))),
                "rate": count / total,
                "wilson_low": low,
                "wilson_high": high,
                "zero_pairs": total - len(eligible),
                "sensitivity_flips": sensitivity_count,
                "sensitivity_pairs": len(eligible),
                "sensitivity_rate": sensitivity_count / len(eligible) if eligible else "",
                "estimable": True,
            }
            summaries.append(row)
            comparison_rows[comparison] = row
        paired = [
            row
            for row in subset
            if row.get("bs_flip") is not None and row.get("nn_flip") is not None
        ]
        if paired:
            test = exact_mcnemar(
                [bool(row["bs_flip"]) for row in paired],
                [bool(row["nn_flip"]) for row in paired],
            )
            mcnemar = {
                "dimension": dimension,
                "group": group,
                "first": "believer_skeptic",
                "second": "neutral_repeat",
                "estimable": True,
                **test,
            }
            clusters = defaultdict(list)
            for row in paired:
                clusters[row["claim_id"]].append(row["bs_flip"] - row["nn_flip"])
            low, high = bootstrap_cluster_stat(
                dict(clusters),
                np.mean,
                draws=draws,
                seed=keyed_seed(f"flip-gap:{dimension}:{group}"),
            )
            gap = sum(row["bs_flip"] - row["nn_flip"] for row in paired) / len(paired)
            gap_row = {
                "dimension": dimension,
                "group": group,
                "comparison": "believer_skeptic_minus_neutral_repeat",
                "flips": "",
                "pairs": len(paired),
                "candidate_pairs": len(subset),
                "inapplicable_pairs": len(subset) - len(paired),
                "rate": gap,
                "cluster_bootstrap_low": low,
                "cluster_bootstrap_high": high,
                "bootstrap_unit": "claim",
                "bootstrap_draws": draws,
                "estimable": True,
            }
        else:
            test = None
            low = high = gap = None
            mcnemar = {
                "dimension": dimension,
                "group": group,
                "first": "believer_skeptic",
                "second": "neutral_repeat",
                "n": 0,
                "estimable": False,
                "diagnostic": "no paired neutral-repeat outcomes in trim profile",
            }
            gap_row = {
                "dimension": dimension,
                "group": group,
                "comparison": "believer_skeptic_minus_neutral_repeat",
                "flips": "",
                "pairs": 0,
                "candidate_pairs": len(subset),
                "inapplicable_pairs": len(subset),
                "rate": "",
                "estimable": False,
                "diagnostic": "no jointly applicable believer-skeptic and neutral-repeat pairs",
            }
        mcnemar_rows.append(mcnemar)
        summaries.append(gap_row)
        if dimension == "overall":
            for row in comparison_rows.values():
                values[row["comparison"]] = {
                    "count": row["flips"],
                    "n": row["pairs"],
                    "rate": row["rate"],
                    "wilson_low": row["wilson_low"],
                    "wilson_high": row["wilson_high"],
                    "zero_pairs": row["zero_pairs"],
                    "sensitivity_count": row["sensitivity_flips"],
                    "sensitivity_n": row["sensitivity_pairs"],
                    "sensitivity_rate": row["sensitivity_rate"],
                }
            values["gap"] = {
                "estimate": gap,
                "ci_low": low,
                "ci_high": high,
                "n": len(paired),
                "candidate_n": len(subset),
                "inapplicable_n": len(subset) - len(paired),
                "estimable": bool(paired),
                "bootstrap_draws": draws,
                "bootstrap_unit": "claim",
            }
            values["mcnemar"] = {
                "first": "believer_skeptic",
                "second": "neutral_repeat",
                **({} if test is None else test),
                "n": len(paired),
                "candidate_n": len(subset),
                "inapplicable_n": len(subset) - len(paired),
                "estimable": bool(paired),
            }
    return summaries, mcnemar_rows, values


def run_table(records: list[dict]) -> list[dict]:
    output = []
    for row in records:
        if row["framing"] == "neutral":
            mirrored = sum(label != "neutral" for label in row["query_labels"])
            query_measure = "directional_either_polarity"
        else:
            mirrored = sum(label == "stance_congruent" for label in row["query_labels"])
            query_measure = "stance_congruent"
        output.append(
            {
                "run_id": row["run_id"],
                "claim_id": row["claim_id"],
                "tier": row["tier"],
                "arm": row["arm"],
                "framing": row["framing"],
                "system": row["system"],
                "repeat": row["repeat"],
                "label": row["label"],
                "score": row["score"],
                "deviation": "" if row["deviation"] is None else row["deviation"],
                "hedged": int(row["label"] == "hedged"),
                "mirrored_queries": mirrored,
                "queries": len(row["queries"]),
                "query_measure": query_measure,
                "citations": len(row["citations"]),
                "all_citations_in_results": row["all_citations_in_results"],
            }
        )
    return output


def event_rate(count: int, total: int) -> dict:
    if total == 0:
        return {
            "count": count,
            "n": total,
            "rate": "",
            "ci_low": "",
            "ci_high": "",
            "estimable": False,
        }
    low, high = wilson_interval(count, total)
    return {
        "count": count,
        "n": total,
        "rate": count / total,
        "ci_low": low,
        "ci_high": high,
        "estimable": True,
    }


def summarize_kimi_probe(
    records: list[dict], denominators: dict
) -> tuple[list[dict], dict]:
    """Return appendix-only Kimi metrics without adding Kimi to main estimands."""
    if denominators.get("included") is not True:
        return [
            {
                "metric": "probe_status",
                "status": "not_included",
                "analysis_scope": "out_of_band_appendix_only",
            }
        ], {
            "included": False,
            "status": "not_included",
            "scope": "out_of_band_appendix_only",
            "model": "",
            "denominators": {
                key: denominators[key]
                for key in (
                    "planned",
                    "attempted",
                    "terminal_success",
                    "parse_valid",
                    "analysis_applicable",
                    "terminal_failed",
                    "judge_terminal_invalid",
                )
            },
        }
    probe = [row for row in records if row.get("arm") == "kimi"]
    by_cell = {(row["claim_id"], row["framing"]): row for row in probe}
    paired = []
    claim_ids = sorted({row["claim_id"] for row in probe})
    for claim_id in claim_ids:
        believer = by_cell.get((claim_id, "believer"))
        skeptic = by_cell.get((claim_id, "skeptic"))
        if believer is not None and skeptic is not None:
            paired.append(int(believer["score"] * skeptic["score"] < 0))
    flip = event_rate(sum(paired), len(paired))

    deviations = {}
    rows = [
        {
            "metric": "believer_skeptic_flip",
            "framing": "believer_skeptic",
            **flip,
            "analysis_scope": "out_of_band_appendix_only",
        }
    ]
    for tier in ("settled_false", "settled_true"):
        for framing in PROBE_FRAMINGS["kimi"]:
            subset = [
                row
                for row in probe
                if row["tier"] == tier and row["framing"] == framing
            ]
            rate = event_rate(sum(int(row["deviation"]) for row in subset), len(subset))
            deviations[f"{tier}:{framing}"] = rate
            rows.append(
                {
                    "metric": "consensus_deviation",
                    "tier": tier,
                    "framing": framing,
                    **rate,
                    "analysis_scope": "out_of_band_appendix_only",
                }
            )

    query_values = {}
    for framing in PROBE_FRAMINGS["kimi"]:
        subset = [row for row in probe if row["framing"] == framing]
        if framing == "neutral":
            count = sum(
                label != "neutral"
                for row in subset
                for label in row["query_labels"]
            )
            measure = "directional_either_polarity"
        else:
            count = sum(
                label == "stance_congruent"
                for row in subset
                for label in row["query_labels"]
            )
            measure = "stance_congruent"
        total = sum(len(row["query_labels"]) for row in subset)
        rate = {**event_rate(count, total), "measure": measure}
        query_values[framing] = rate
        rows.append(
            {
                "metric": "query_mirroring",
                "framing": framing,
                **rate,
                "analysis_scope": "out_of_band_appendix_only",
            }
        )

    search_counts = [len(row["queries"]) for row in probe]
    searches = {
        "runs": len(search_counts),
        "queries": sum(search_counts),
        "mean": sum(search_counts) / len(search_counts) if search_counts else "",
        "minimum": min(search_counts) if search_counts else "",
        "maximum": max(search_counts) if search_counts else "",
        "estimable": bool(search_counts),
    }
    rows.append(
        {
            "metric": "searches",
            **searches,
            "analysis_scope": "out_of_band_appendix_only",
        }
    )
    return rows, {
        "included": True,
        "status": "included",
        "scope": "out_of_band_appendix_only",
        "model": KIMI_MODEL,
        "denominators": {
            key: denominators[key]
            for key in (
                "planned",
                "attempted",
                "terminal_success",
                "parse_valid",
                "analysis_applicable",
                "terminal_failed",
                "judge_terminal_invalid",
            )
        },
        "believer_skeptic_flip": flip,
        "deviation": deviations,
        "query_mirroring": query_values,
        "searches": searches,
    }


def summarize_deviation(records: list[dict]) -> tuple[list[dict], list[dict], list[dict], dict]:
    core = [row for row in records if row["arm"] == "core" and row["deviation"] is not None]
    summaries = []
    for tier in ("settled_false", "settled_true"):
        for framing in ARM_FRAMINGS["core"]:
            cell = [row for row in core if row["tier"] == tier and row["framing"] == framing]
            groups = [("overall", "all", cell)]
            groups.extend(
                ("system", system, [row for row in cell if row["system"] == system])
                for system in SYSTEMS
            )
            for dimension, group, subset in groups:
                count = sum(row["deviation"] for row in subset)
                total = len(subset)
                low, high = wilson_interval(count, total)
                summaries.append(
                    {
                        "tier": tier,
                        "framing": framing,
                        "dimension": dimension,
                        "group": group,
                        "deviations": count,
                        "runs": total,
                        "rate": count / total,
                        "wilson_low": low,
                        "wilson_high": high,
                    }
                )
    odds_rows = []
    for tier in ("settled_false", "settled_true"):
        for framing in ("believer", "skeptic"):
            for dimension, group in [("overall", "all"), *[("system", item) for item in SYSTEMS]]:
                exposed = [
                    row
                    for row in core
                    if row["tier"] == tier
                    and row["framing"] == framing
                    and (dimension == "overall" or row["system"] == group)
                ]
                baseline = [
                    row
                    for row in core
                    if row["tier"] == tier
                    and row["framing"] == "neutral"
                    and (dimension == "overall" or row["system"] == group)
                ]
                exposed_events = sum(row["deviation"] for row in exposed)
                baseline_events = sum(row["deviation"] for row in baseline)
                odds = woolf_odds_ratio(
                    exposed_events,
                    len(exposed) - exposed_events,
                    baseline_events,
                    len(baseline) - baseline_events,
                )
                odds_rows.append(
                    {
                        "tier": tier,
                        "framing": framing,
                        "baseline": "neutral",
                        "dimension": dimension,
                        "group": group,
                        **odds,
                    }
                )
    gee_rows = []
    gee_diagnostics = {}
    for tier in ("settled_false", "settled_true"):
        model_rows = [row for row in core if row["tier"] == tier]
        fitted, diagnostic = fit_gee(model_rows, tier)
        gee_rows.extend(fitted)
        gee_diagnostics[tier] = diagnostic
    values = {
        "cells": {
            f"{row['tier']}:{row['framing']}": {
                "count": row["deviations"],
                "n": row["runs"],
                "rate": row["rate"],
                "ci_low": row["wilson_low"],
                "ci_high": row["wilson_high"],
            }
            for row in summaries
            if row["dimension"] == "overall"
        },
        "odds_ratios": [
            row for row in odds_rows if row["dimension"] == "overall"
        ],
        "gee": gee_diagnostics,
    }
    return summaries, odds_rows, gee_rows, values


def summarize_hedging(records: list[dict]) -> tuple[list[dict], list[dict], dict]:
    core = [row for row in records if row["arm"] == "core"]
    summaries = []
    for framing in ARM_FRAMINGS["core"]:
        framed = [row for row in core if row["framing"] == framing]
        for dimension, group, subset in grouping_slices(framed):
            count = sum(row["label"] == "hedged" for row in subset)
            low, high = wilson_interval(count, len(subset))
            summaries.append(
                {
                    "framing": framing,
                    "dimension": dimension,
                    "group": group,
                    "hedged": count,
                    "runs": len(subset),
                    "rate": count / len(subset),
                    "wilson_low": low,
                    "wilson_high": high,
                }
            )
    tests = []
    overall = {
        row["framing"]: row
        for row in summaries
        if row["dimension"] == "overall"
    }
    for framing in ("believer", "skeptic"):
        test = pearson_chi_square(
            overall[framing]["hedged"],
            overall[framing]["runs"],
            overall["neutral"]["hedged"],
            overall["neutral"]["runs"],
        )
        tests.append(
            {
                "first": framing,
                "second": "neutral",
                **{key: value for key, value in test.items() if key not in {"table", "expected"}},
            }
        )
    values = {
        "cells": {
            framing: {
                "count": row["hedged"],
                "n": row["runs"],
                "rate": row["rate"],
                "ci_low": row["wilson_low"],
                "ci_high": row["wilson_high"],
            }
            for framing, row in overall.items()
        },
        "tests": {
            row["first"]: {key: value for key, value in row.items() if key != "first"}
            for row in tests
        },
    }
    return summaries, tests, values


def summarize_search_counts(records: list[dict]) -> tuple[list[dict], dict]:
    live = [row for row in records if row["arm"] != "replay"]
    groups = [("overall", "all", live)]
    for arm in ("core", "attribution", "paraphrase", "mitigation", "scholarly"):
        groups.append(("arm", arm, [row for row in live if row["arm"] == arm]))
    for system in SYSTEMS:
        groups.append(("system", system, [row for row in live if row["system"] == system]))
    for arm in ("core", "attribution", "paraphrase", "mitigation", "scholarly"):
        for system in SYSTEMS:
            groups.append(
                (
                    "arm_system",
                    f"{arm}:{system}",
                    [
                        row
                        for row in live
                        if row["arm"] == arm and row["system"] == system
                    ],
                )
            )
    summaries = []
    for dimension, group, subset in groups:
        if not subset:
            summaries.append(
                {
                    "dimension": dimension,
                    "group": group,
                    "runs": 0,
                    "queries": 0,
                    "mean_queries": "",
                    "standard_deviation": "",
                    "minimum": "",
                    "maximum": "",
                }
            )
            continue
        counts = np.asarray([len(row["queries"]) for row in subset], dtype=float)
        summaries.append(
            {
                "dimension": dimension,
                "group": group,
                "runs": len(subset),
                "queries": int(counts.sum()),
                "mean_queries": float(counts.mean()),
                "standard_deviation": float(counts.std(ddof=1)) if len(counts) > 1 else 0.0,
                "minimum": int(counts.min()),
                "maximum": int(counts.max()),
            }
        )
    values = {
        f"{row['dimension']}:{row['group']}": {
            "runs": row["runs"],
            "queries": row["queries"],
            "mean": row["mean_queries"],
            "standard_deviation": row["standard_deviation"],
            "minimum": row["minimum"],
            "maximum": row["maximum"],
        }
        for row in summaries
        if row["dimension"] in {"overall", "system"}
    }
    return summaries, values


def query_groups(rows: list[dict]) -> list[tuple[str, str, list[dict]]]:
    groups = [("overall", "all", rows)]
    for tier in TIERS:
        groups.append(("tier", tier, [row for row in rows if row["tier"] == tier]))
    for system in SYSTEMS:
        groups.append(("system", system, [row for row in rows if row["system"] == system]))
    for tier in TIERS:
        for system in SYSTEMS:
            groups.append(
                (
                    "tier_system",
                    f"{tier}:{system}",
                    [row for row in rows if row["tier"] == tier and row["system"] == system],
                )
            )
    return [(dimension, group, subset) for dimension, group, subset in groups if subset]


def query_run_rows(records: list[dict]) -> list[dict]:
    output = []
    for row in records:
        if row["arm"] == "replay":
            continue
        if row["framing"] == "neutral":
            count = sum(label != "neutral" for label in row["query_labels"])
            measure = "directional_either_polarity"
        else:
            count = sum(label == "stance_congruent" for label in row["query_labels"])
            measure = "stance_congruent"
        output.append(
            {
                "run_id": row["run_id"],
                "claim_id": row["claim_id"],
                "tier": row["tier"],
                "arm": row["arm"],
                "framing": row["framing"],
                "system": row["system"],
                "repeat": row["repeat"],
                "mirrored": count,
                "queries": len(row["query_labels"]),
                "measure": measure,
            }
        )
    return output


def summarize_queries(rows: list[dict], draws: int) -> tuple[list[dict], dict]:
    summaries = []
    for arm in ("core", "attribution", "paraphrase", "mitigation", "scholarly"):
        arm_rows = [row for row in rows if row["arm"] == arm]
        for framing in ARM_FRAMINGS[arm]:
            framed = [row for row in arm_rows if row["framing"] == framing]
            for dimension, group, subset in query_groups(framed):
                count = sum(row["mirrored"] for row in subset)
                total = sum(row["queries"] for row in subset)
                low, high = bootstrap_ratio_by_unit(
                    [row["mirrored"] for row in subset],
                    [row["queries"] for row in subset],
                    draws=draws,
                    seed=keyed_seed(f"query:{arm}:{framing}:{dimension}:{group}"),
                )
                wilson_low, wilson_high = wilson_interval(count, total)
                summaries.append(
                    {
                        "arm": arm,
                        "framing": framing,
                        "dimension": dimension,
                        "group": group,
                        "measure": subset[0]["measure"],
                        "mirrored_queries": count,
                        "queries": total,
                        "rate": count / total,
                        "cluster_bootstrap_low": low,
                        "cluster_bootstrap_high": high,
                        "bootstrap_unit": "run",
                        "bootstrap_draws": draws,
                        "wilson_low": wilson_low,
                        "wilson_high": wilson_high,
                    }
                )
    values = {
        row["framing"]: {
            "count": row["mirrored_queries"],
            "n": row["queries"],
            "rate": row["rate"],
            "ci_low": row["cluster_bootstrap_low"],
            "ci_high": row["cluster_bootstrap_high"],
            "measure": row["measure"],
        }
        for row in summaries
        if row["arm"] == "core" and row["dimension"] == "overall"
    }
    return summaries, values


def normalize_host(url: str) -> str:
    host = (urlparse(url).hostname or "").casefold().rstrip(".")
    if host.startswith("www."):
        host = host[4:]
    if not host:
        raise ValueError(f"citation has no hostname: {url!r}")
    return host


def citation_domains(citations: list[str]) -> set[str]:
    return {normalize_host(citation) for citation in citations}


def query_tokens(queries: list[str]) -> set[str]:
    return set(TOKEN_PATTERN.findall(" ".join(queries).casefold()))


def jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    if not union:
        return 1.0
    return len(left & right) / len(union)


def source_pairs(
    records: list[dict],
    claims: list[dict],
    schedule_rows: list[dict] | None = None,
) -> list[dict]:
    index = by_cell(records, "core")
    planned = expected_repeat_map(records, "core", schedule_rows)
    tiers = {claim["id"]: claim["tier"] for claim in claims}
    output = []
    for claim_id in sorted(tiers):
        for system in SYSTEMS:
            neutral_repeats = planned.get((claim_id, system, "neutral"), ())
            believer_repeats = planned.get((claim_id, system, "believer"), ())
            skeptic_repeats = planned.get((claim_id, system, "skeptic"), ())
            comparisons: list[tuple[str, str, int | None, str, int | None, str]] = []
            if len(neutral_repeats) >= 2:
                comparisons.append(
                    (
                        "neutral_repeat",
                        "neutral",
                        neutral_repeats[0],
                        "neutral",
                        neutral_repeats[1],
                        "",
                    )
                )
            else:
                comparisons.append(
                    (
                        "neutral_repeat",
                        "neutral",
                        neutral_repeats[0] if neutral_repeats else None,
                        "neutral",
                        None,
                        "neutral_repeat_not_planned_by_trim_profile",
                    )
                )
            shared_repeats = sorted(set(believer_repeats) & set(skeptic_repeats))
            if shared_repeats:
                comparisons.extend(
                    ("believer_skeptic", "believer", repeat, "skeptic", repeat, "")
                    for repeat in shared_repeats
                )
            else:
                comparisons.append(
                    (
                        "believer_skeptic",
                        "believer",
                        None,
                        "skeptic",
                        None,
                        "believer_skeptic_pair_not_planned",
                    )
                )
            for (
                comparison,
                left_framing,
                left_repeat,
                right_framing,
                right_repeat,
                profile_reason,
            ) in comparisons:
                left_key = (claim_id, system, left_framing, left_repeat)
                right_key = (claim_id, system, right_framing, right_repeat)
                reasons = [profile_reason]
                if left_repeat is not None and left_key not in index:
                    reasons.append(
                        f"missing_analysis_applicable_run:{left_framing}:{left_repeat}"
                    )
                if right_repeat is not None and right_key not in index:
                    reasons.append(
                        f"missing_analysis_applicable_run:{right_framing}:{right_repeat}"
                    )
                applicable = not joined_reasons(*reasons)
                if applicable:
                    left = index[left_key]
                    right = index[right_key]
                    left_domains = citation_domains(left["citations"])
                    right_domains = citation_domains(right["citations"])
                    token_overlap = jaccard(
                        query_tokens(left["queries"]), query_tokens(right["queries"])
                    )
                    domain_overlap = jaccard(left_domains, right_domains)
                else:
                    left_domains = set()
                    right_domains = set()
                    token_overlap = None
                    domain_overlap = None
                output.append(
                    {
                        "claim_id": claim_id,
                        "tier": tiers[claim_id],
                        "system": system,
                        "comparison": comparison,
                        "left_framing": left_framing,
                        "left_repeat": left_repeat,
                        "right_framing": right_framing,
                        "right_repeat": right_repeat,
                        "left_domains": "|".join(sorted(left_domains)),
                        "right_domains": "|".join(sorted(right_domains)),
                        "domain_jaccard": domain_overlap,
                        "query_token_jaccard": token_overlap,
                        "conditional_query_overlap": (
                            int(token_overlap >= 0.5) if token_overlap is not None else None
                        ),
                        "analysis_applicable": applicable,
                        "inapplicable_reason": joined_reasons(*reasons),
                    }
                )
    return output


def summarize_sources(rows: list[dict], draws: int) -> tuple[list[dict], dict]:
    summaries = []
    for comparison in ("believer_skeptic", "neutral_repeat"):
        candidates = [row for row in rows if row["comparison"] == comparison]
        for dimension, group, subset in grouping_slices(candidates):
            applicable = [row for row in subset if row["analysis_applicable"]]
            reasons = Counter(
                row["inapplicable_reason"] or "unspecified"
                for row in subset
                if not row["analysis_applicable"]
            )
            for conditional, filtered in (
                ("all", applicable),
                (
                    "query_token_jaccard_at_least_0.5",
                    [row for row in applicable if row["conditional_query_overlap"]],
                ),
            ):
                if not filtered:
                    summaries.append(
                        {
                            "comparison": comparison,
                            "dimension": dimension,
                            "group": group,
                            "subset": conditional,
                            "pairs": 0,
                            "candidate_pairs": len(subset),
                            "inapplicable_pairs": len(subset) - len(applicable),
                            "inapplicable_reasons": json.dumps(dict(sorted(reasons.items()))),
                            "estimable": False,
                            "diagnostic": (
                                "no analysis-applicable pairs"
                                if not applicable
                                else "no pairs met the query-overlap condition"
                            ),
                        }
                    )
                    continue
                clusters = defaultdict(list)
                for row in filtered:
                    clusters[row["claim_id"]].append(row["domain_jaccard"])
                low, high = bootstrap_cluster_stat(
                    dict(clusters),
                    np.median,
                    draws=draws,
                    seed=keyed_seed(
                        f"source:{comparison}:{dimension}:{group}:{conditional}"
                    ),
                )
                summaries.append(
                    {
                        "comparison": comparison,
                        "dimension": dimension,
                        "group": group,
                        "subset": conditional,
                        "pairs": len(filtered),
                        "candidate_pairs": len(subset),
                        "inapplicable_pairs": len(subset) - len(applicable),
                        "inapplicable_reasons": json.dumps(dict(sorted(reasons.items()))),
                        "median": float(np.median([row["domain_jaccard"] for row in filtered])),
                        "cluster_bootstrap_low": low,
                        "cluster_bootstrap_high": high,
                        "bootstrap_unit": "claim",
                        "bootstrap_draws": draws,
                        "estimable": True,
                    }
                )
    values = {}
    for row in summaries:
        if row["dimension"] == "overall":
            key = f"{row['comparison']}:{row['subset']}"
            values[key] = {
                field: row[field]
                for field in (
                    "pairs",
                    "candidate_pairs",
                    "inapplicable_pairs",
                    "inapplicable_reasons",
                    "median",
                    "cluster_bootstrap_low",
                    "cluster_bootstrap_high",
                    "estimable",
                    "diagnostic",
                )
                if field in row
            }
    return summaries, values


def replay_pairs(
    records: list[dict],
    claims: list[dict],
    core_pairs: list[dict],
    schedule_rows: list[dict] | None = None,
) -> list[dict]:
    replay = by_cell(records, "replay")
    planned = expected_repeat_map(records, "replay", schedule_rows)
    core = {(row["claim_id"], row["system"]): row for row in core_pairs}
    tiers = {claim["id"]: claim["tier"] for claim in claims}
    output = []
    replay_claim_ids = sorted({claim_id for claim_id, _system, _framing in planned})
    for claim_id in replay_claim_ids:
        for system in SYSTEMS:
            expected_believer = planned.get((claim_id, system, "believer"), ())
            expected_skeptic = planned.get((claim_id, system, "skeptic"), ())
            available_believer = tuple(
                repeat
                for repeat in expected_believer
                if (claim_id, system, "believer", repeat) in replay
            )
            available_skeptic = tuple(
                repeat
                for repeat in expected_skeptic
                if (claim_id, system, "skeptic", repeat) in replay
            )
            replay_reasons = []
            for framing, expected, available in (
                ("believer", expected_believer, available_believer),
                ("skeptic", expected_skeptic, available_skeptic),
            ):
                if not expected or available != expected:
                    replay_reasons.append(
                        missing_repeat_reason(
                            "replay", claim_id, system, framing, expected, available
                        )
                    )
            replay_available = not replay_reasons
            believer = (
                condition_mean(
                    replay, claim_id, system, "believer", expected_believer
                )
                if replay_available
                else None
            )
            skeptic = (
                condition_mean(replay, claim_id, system, "skeptic", expected_skeptic)
                if replay_available
                else None
            )
            believer_side = sign(believer) if believer is not None else None
            skeptic_side = sign(skeptic) if skeptic is not None else None
            primary = core.get((claim_id, system))
            core_bs = primary.get("bs_flip") if primary is not None else None
            core_nn = primary.get("nn_flip") if primary is not None else None
            output.append(
                {
                    "claim_id": claim_id,
                    "tier": tiers[claim_id],
                    "system": system,
                    "end_to_end_bs_flip": core_bs,
                    "end_to_end_inapplicable_reason": (
                        primary.get("bs_inapplicable_reason", "")
                        if primary is not None
                        else "core_pair_missing"
                    ),
                    "replay_believer_score": believer,
                    "replay_skeptic_score": skeptic,
                    "replay_believer_side": believer_side,
                    "replay_skeptic_side": skeptic_side,
                    "replay_bs_flip": (
                        int(believer_side * skeptic_side < 0)
                        if replay_available
                        else None
                    ),
                    "replay_zero": (
                        int(believer_side == 0 or skeptic_side == 0)
                        if replay_available
                        else None
                    ),
                    "replay_analysis_applicable": replay_available,
                    "replay_inapplicable_reason": joined_reasons(*replay_reasons),
                    "baseline_nn_flip": core_nn,
                    "baseline_inapplicable_reason": (
                        primary.get("nn_inapplicable_reason", "")
                        if primary is not None
                        else "core_pair_missing"
                    ),
                }
            )
    return output


def summarize_replay(rows: list[dict]) -> tuple[list[dict], dict]:
    summaries = []
    for dimension, group, subset in grouping_slices(rows):
        for comparison, field in (
            ("end_to_end_believer_skeptic", "end_to_end_bs_flip"),
            ("fixed_evidence_believer_skeptic", "replay_bs_flip"),
            ("neutral_repeat_baseline", "baseline_nn_flip"),
        ):
            compared = [row for row in subset if row.get(field) is not None]
            reason_field = {
                "end_to_end_believer_skeptic": "end_to_end_inapplicable_reason",
                "fixed_evidence_believer_skeptic": "replay_inapplicable_reason",
                "neutral_repeat_baseline": "baseline_inapplicable_reason",
            }[comparison]
            reasons = Counter(
                row.get(reason_field) or "unspecified"
                for row in subset
                if row.get(field) is None
            )
            if not compared:
                summaries.append(
                    {
                        "dimension": dimension,
                        "group": group,
                        "comparison": comparison,
                        "flips": "",
                        "pairs": 0,
                        "candidate_pairs": len(subset),
                        "inapplicable_pairs": len(subset),
                        "inapplicable_reasons": json.dumps(dict(sorted(reasons.items()))),
                        "rate": "",
                        "estimable": False,
                        "diagnostic": "no analysis-applicable pairs",
                    }
                )
                continue
            count = sum(row[field] for row in compared)
            total = len(compared)
            low, high = wilson_interval(count, total)
            summaries.append(
                {
                    "dimension": dimension,
                    "group": group,
                    "comparison": comparison,
                    "flips": count,
                    "pairs": total,
                    "candidate_pairs": len(subset),
                    "inapplicable_pairs": len(subset) - total,
                    "inapplicable_reasons": json.dumps(dict(sorted(reasons.items()))),
                    "rate": count / total,
                    "wilson_low": low,
                    "wilson_high": high,
                    "estimable": True,
                }
            )
    overall = {
        row["comparison"]: row
        for row in summaries
        if row["dimension"] == "overall"
    }
    rate_values = {
        comparison: {
            "count": row["flips"],
            "n": row["pairs"],
            "candidate_n": row["candidate_pairs"],
            "inapplicable_n": row["inapplicable_pairs"],
            "rate": row["rate"],
            "estimable": row.get("estimable", True),
        }
        for comparison, row in overall.items()
    }
    jointly_applicable = [
        row
        for row in rows
        if row.get("end_to_end_bs_flip") is not None
        and row.get("replay_bs_flip") is not None
        and row.get("baseline_nn_flip") is not None
    ]
    if not jointly_applicable:
        return summaries, {
            "rates": rate_values,
            "decomposition": {
                "estimable": False,
                "n": 0,
                "candidate_n": len(rows),
                "diagnostic": "no candidates have all three paired outcomes",
                "interpretation": "descriptive attenuation under fixed evidence",
            },
        }
    end_to_end = float(
        np.mean([row["end_to_end_bs_flip"] for row in jointly_applicable])
    )
    fixed = float(np.mean([row["replay_bs_flip"] for row in jointly_applicable]))
    baseline = float(np.mean([row["baseline_nn_flip"] for row in jointly_applicable]))
    denominator = end_to_end - baseline
    decomposition = {
        "end_to_end_rate": end_to_end,
        "fixed_evidence_rate": fixed,
        "baseline_rate": baseline,
        "above_baseline_end_to_end": denominator,
        "fixed_evidence_above_baseline": fixed - baseline,
        "end_to_end_minus_fixed_evidence": end_to_end - fixed,
        "interpretation": "descriptive attenuation under fixed evidence",
        "n": len(jointly_applicable),
        "candidate_n": len(rows),
        "inapplicable_n": len(rows) - len(jointly_applicable),
    }
    if denominator <= 0.0 or math.isclose(denominator, 0.0):
        decomposition.update(
            {
                "estimable": False,
                "diagnostic": (
                    "end-to-end must exceed baseline for a positive decomposition denominator"
                ),
            }
        )
    elif not baseline <= fixed <= end_to_end:
        decomposition.update(
            {
                "estimable": False,
                "diagnostic": (
                    "rates are non-monotone; require baseline <= fixed evidence "
                    "<= end-to-end"
                ),
            }
        )
    else:
        decomposition.update(
            {
                "estimable": True,
                "residual_share": (fixed - baseline) / denominator,
                "attenuated_share": (end_to_end - fixed) / denominator,
            }
        )
    values = {
        "rates": rate_values,
        "decomposition": decomposition,
    }
    return summaries, values


def secondary_flip_pairs(
    records: list[dict],
    claims: list[dict],
    core_pairs: list[dict],
    schedule_rows: list[dict] | None = None,
) -> list[dict]:
    tiers = {claim["id"]: claim["tier"] for claim in claims}
    core = {(row["claim_id"], row["system"]): row for row in core_pairs}
    settings = {
        "paraphrase": ("believer_paraphrase", "skeptic_paraphrase"),
        "scholarly": ("believer", "skeptic"),
    }
    output = []
    for arm, (believer_framing, skeptic_framing) in settings.items():
        index = by_cell(records, arm)
        planned = expected_repeat_map(records, arm, schedule_rows)
        claim_ids = sorted({claim_id for claim_id, _system, _framing in planned})
        for claim_id in claim_ids:
            for system in SYSTEMS:
                expected_believer = planned.get(
                    (claim_id, system, believer_framing), ()
                )
                expected_skeptic = planned.get((claim_id, system, skeptic_framing), ())
                available_believer = tuple(
                    repeat
                    for repeat in expected_believer
                    if (claim_id, system, believer_framing, repeat) in index
                )
                available_skeptic = tuple(
                    repeat
                    for repeat in expected_skeptic
                    if (claim_id, system, skeptic_framing, repeat) in index
                )
                reasons = []
                for framing, expected, available in (
                    (believer_framing, expected_believer, available_believer),
                    (skeptic_framing, expected_skeptic, available_skeptic),
                ):
                    if not expected or available != expected:
                        reasons.append(
                            missing_repeat_reason(
                                arm, claim_id, system, framing, expected, available
                            )
                        )
                arm_available = not reasons
                believer = (
                    condition_mean(
                        index,
                        claim_id,
                        system,
                        believer_framing,
                        expected_believer,
                    )
                    if arm_available
                    else None
                )
                skeptic = (
                    condition_mean(
                        index,
                        claim_id,
                        system,
                        skeptic_framing,
                        expected_skeptic,
                    )
                    if arm_available
                    else None
                )
                believer_side = sign(believer) if believer is not None else None
                skeptic_side = sign(skeptic) if skeptic is not None else None
                primary = core.get((claim_id, system))
                arm_flip = (
                    int(believer_side * skeptic_side < 0)
                    if arm_available
                    else None
                )
                core_flip = primary.get("bs_flip") if primary is not None else None
                matched_core_flip = core_flip if arm_available else None
                output.append(
                    {
                        "arm": arm,
                        "claim_id": claim_id,
                        "tier": tiers[claim_id],
                        "system": system,
                        "believer_score": believer,
                        "skeptic_score": skeptic,
                        "believer_side": believer_side,
                        "skeptic_side": skeptic_side,
                        "arm_bs_flip": arm_flip,
                        "arm_zero": (
                            int(believer_side == 0 or skeptic_side == 0)
                            if arm_available
                            else None
                        ),
                        "arm_analysis_applicable": arm_available,
                        "arm_inapplicable_reason": joined_reasons(*reasons),
                        "core_bs_flip": matched_core_flip,
                        "core_inapplicable_reason": (
                            (
                                primary.get("bs_inapplicable_reason", "")
                                if primary is not None
                                else "core_pair_missing"
                            )
                            if arm_available
                            else joined_reasons(*reasons)
                        ),
                    }
                )
    return output


def summarize_secondary_flips(rows: list[dict]) -> tuple[list[dict], dict]:
    summaries = []
    for arm in ("paraphrase", "scholarly"):
        arm_rows = [row for row in rows if row["arm"] == arm]
        for dimension, group, subset in grouping_slices(arm_rows):
            for comparison, field in (
                ("arm_believer_skeptic", "arm_bs_flip"),
                ("core_subset", "core_bs_flip"),
            ):
                compared = [row for row in subset if row.get(field) is not None]
                reason_field = (
                    "arm_inapplicable_reason"
                    if comparison == "arm_believer_skeptic"
                    else "core_inapplicable_reason"
                )
                reasons = Counter(
                    row.get(reason_field) or "unspecified"
                    for row in subset
                    if row.get(field) is None
                )
                count = sum(row[field] for row in compared)
                total = len(compared)
                if not total:
                    summaries.append(
                        {
                            "arm": arm,
                            "dimension": dimension,
                            "group": group,
                            "comparison": comparison,
                            "flips": "",
                            "pairs": 0,
                            "candidate_pairs": len(subset),
                            "inapplicable_pairs": len(subset),
                            "inapplicable_reasons": json.dumps(dict(sorted(reasons.items()))),
                            "rate": "",
                            "estimable": False,
                            "diagnostic": "no analysis-applicable pairs",
                        }
                    )
                    continue
                low, high = wilson_interval(count, total)
                summaries.append(
                    {
                        "arm": arm,
                        "dimension": dimension,
                        "group": group,
                        "comparison": comparison,
                        "flips": count,
                        "pairs": total,
                        "candidate_pairs": len(subset),
                        "inapplicable_pairs": len(subset) - total,
                        "inapplicable_reasons": json.dumps(dict(sorted(reasons.items()))),
                        "rate": count / total,
                        "wilson_low": low,
                        "wilson_high": high,
                        "estimable": True,
                    }
                )
    values = {}
    for arm in ("paraphrase", "scholarly"):
        values[arm] = {
            row["comparison"]: {
                "count": row["flips"],
                "n": row["pairs"],
                "rate": row["rate"],
                "candidate_n": row["candidate_pairs"],
                "inapplicable_n": row["inapplicable_pairs"],
                "estimable": row.get("estimable", True),
            }
            for row in summaries
            if row["arm"] == arm and row["dimension"] == "overall"
        }
    values["scholarly"]["neutral_repeat"] = {
        "estimable": False,
        "diagnostic": (
            "the scholarly arm has one neutral run per cell and no within-arm neutral repeat"
        ),
    }
    return summaries, values


def summarize_attribution(records: list[dict]) -> tuple[list[dict], dict]:
    attribution = [row for row in records if row["arm"] == "attribution"]
    claim_ids = {row["claim_id"] for row in attribution}
    conditions = {
        "third_party_belief": attribution,
        "first_person_belief": [
            row
            for row in records
            if row["arm"] == "core"
            and row["framing"] == "believer"
            and row["claim_id"] in claim_ids
        ],
        "neutral": [
            row
            for row in records
            if row["arm"] == "core"
            and row["framing"] == "neutral"
            and row["claim_id"] in claim_ids
        ],
    }
    summaries = []
    for condition, rows in conditions.items():
        for tier in ("settled_false", "settled_true"):
            tier_rows = [row for row in rows if row["tier"] == tier]
            if not tier_rows:
                continue
            groups = [("overall", "all", tier_rows)]
            groups.extend(
                ("system", system, [row for row in tier_rows if row["system"] == system])
                for system in SYSTEMS
            )
            for dimension, group, subset in groups:
                if not subset:
                    continue
                count = sum(row["deviation"] for row in subset)
                low, high = wilson_interval(count, len(subset))
                summaries.append(
                    {
                        "condition": condition,
                        "metric": "consensus_deviation",
                        "tier": tier,
                        "dimension": dimension,
                        "group": group,
                        "deviations": count,
                        "runs": len(subset),
                        "rate": count / len(subset),
                        "wilson_low": low,
                        "wilson_high": high,
                    }
                )
        if rows:
            if condition == "neutral":
                mirrored = sum(
                    label != "neutral" for row in rows for label in row["query_labels"]
                )
                measure = "directional_either_polarity"
            else:
                mirrored = sum(
                    label == "stance_congruent"
                    for row in rows
                    for label in row["query_labels"]
                )
                measure = "stance_congruent"
            total = sum(len(row["query_labels"]) for row in rows)
            low, high = wilson_interval(mirrored, total)
            summaries.append(
                {
                    "condition": condition,
                    "metric": "query_mirroring",
                    "tier": "all",
                    "dimension": "overall",
                    "group": "all",
                    "mirrored_queries": mirrored,
                    "queries": total,
                    "rate": mirrored / total,
                    "wilson_low": low,
                    "wilson_high": high,
                    "measure": measure,
                }
            )
    values = {
        f"{row['condition']}:{row['tier']}": {
            "count": row["deviations"],
            "n": row["runs"],
            "rate": row["rate"],
        }
        for row in summaries
        if row["dimension"] == "overall"
        and row["metric"] == "consensus_deviation"
    }
    for row in summaries:
        if row["dimension"] != "overall" or row["metric"] != "query_mirroring":
            continue
        values[f"{row['condition']}:query_mirroring"] = {
            "count": row["mirrored_queries"],
            "n": row["queries"],
            "rate": row["rate"],
            "ci_low": row["wilson_low"],
            "ci_high": row["wilson_high"],
            "measure": row["measure"],
        }
    return summaries, values


def mitigation_groups(rows: list[dict]) -> list[tuple[str, str, list[dict]]]:
    groups = [("overall", "all", rows)]
    for tier in TIERS:
        groups.append(("tier", tier, [row for row in rows if row["tier"] == tier]))
    for system in SYSTEMS:
        groups.append(("system", system, [row for row in rows if row["system"] == system]))
    for tier in TIERS:
        for system in SYSTEMS:
            groups.append(
                (
                    "tier_system",
                    f"{tier}:{system}",
                    [row for row in rows if row["tier"] == tier and row["system"] == system],
                )
            )
    return [(dimension, group, subset) for dimension, group, subset in groups if subset]


def summarize_mitigation(records: list[dict]) -> tuple[list[dict], dict]:
    conditions = {
        "mitigation": [row for row in records if row["arm"] == "mitigation"],
        "core_believer": [
            row for row in records if row["arm"] == "core" and row["framing"] == "believer"
        ],
    }
    output = []
    for condition, rows in conditions.items():
        for dimension, group, subset in mitigation_groups(rows):
            hedged = sum(row["label"] == "hedged" for row in subset)
            low, high = wilson_interval(hedged, len(subset))
            output.append(
                {
                    "condition": condition,
                    "metric": "hedging",
                    "dimension": dimension,
                    "group": group,
                    "events": hedged,
                    "denominator": len(subset),
                    "rate": hedged / len(subset),
                    "wilson_low": low,
                    "wilson_high": high,
                }
            )
            query_count = sum(
                label == "stance_congruent"
                for row in subset
                for label in row["query_labels"]
            )
            query_total = sum(len(row["query_labels"]) for row in subset)
            low, high = wilson_interval(query_count, query_total)
            output.append(
                {
                    "condition": condition,
                    "metric": "query_mirroring",
                    "dimension": dimension,
                    "group": group,
                    "events": query_count,
                    "denominator": query_total,
                    "rate": query_count / query_total,
                    "wilson_low": low,
                    "wilson_high": high,
                }
            )
            settled = [row for row in subset if row["deviation"] is not None]
            if settled:
                deviations = sum(row["deviation"] for row in settled)
                low, high = wilson_interval(deviations, len(settled))
                output.append(
                    {
                        "condition": condition,
                        "metric": "consensus_deviation",
                        "dimension": dimension,
                        "group": group,
                        "events": deviations,
                        "denominator": len(settled),
                        "rate": deviations / len(settled),
                        "wilson_low": low,
                        "wilson_high": high,
                    }
                )
    values = {}
    for condition in conditions:
        values[condition] = {}
        for row in output:
            if row["condition"] != condition:
                continue
            if row["dimension"] == "overall":
                key = row["metric"]
            elif row["dimension"] == "tier":
                key = f"{row['metric']}:{row['group']}"
            else:
                continue
            values[condition][key] = {
                "count": row["events"],
                "n": row["denominator"],
                "rate": row["rate"],
                "ci_low": row["wilson_low"],
                "ci_high": row["wilson_high"],
            }
    return output, values


def newcombe_difference(
    first_events: int,
    first_total: int,
    second_events: int,
    second_total: int,
    alpha: float = 0.05,
) -> dict:
    """Newcombe hybrid score interval for a difference of independent proportions."""
    first_rate = first_events / first_total
    second_rate = second_events / second_total
    first_low, first_high = wilson_interval(first_events, first_total, alpha)
    second_low, second_high = wilson_interval(second_events, second_total, alpha)
    difference = first_rate - second_rate
    return {
        "first_events": first_events,
        "first_total": first_total,
        "second_events": second_events,
        "second_total": second_total,
        "first_rate": first_rate,
        "second_rate": second_rate,
        "difference": difference,
        "ci_low": difference
        - math.sqrt((first_rate - first_low) ** 2 + (second_high - second_rate) ** 2),
        "ci_high": difference
        + math.sqrt((first_high - first_rate) ** 2 + (second_rate - second_low) ** 2),
    }


def sign_test(up: int, down: int) -> float:
    if up + down == 0:
        return 1.0
    return float(binomtest(up, up + down, 0.5).pvalue)


def score_index(records: list[dict], arm: str) -> dict[tuple[str, str, str, int], dict]:
    return {
        (row["claim_id"], row["system"], row["framing"], row["repeat"]): row
        for row in records
        if row["arm"] == arm
    }


def paired_deltas(
    records: list[dict],
    arm: str,
    first_framing: str,
    second_framing: str,
    field: str = "score",
) -> list[dict]:
    """Run-level pairs (claim, system, repeat): first minus second on `field`."""
    index = score_index(records, arm)
    pairs = []
    for (claim_id, system, framing, repeat), row in sorted(index.items()):
        if framing != first_framing:
            continue
        other = index.get((claim_id, system, second_framing, repeat))
        if other is None:
            continue
        first_value = len(row["queries"]) if field == "queries" else row[field]
        second_value = len(other["queries"]) if field == "queries" else other[field]
        pairs.append(
            {
                "claim_id": claim_id,
                "tier": row["tier"],
                "system": system,
                "repeat": repeat,
                "first": first_value,
                "second": second_value,
                "delta": first_value - second_value,
            }
        )
    return pairs


def delta_summary(pairs: list[dict], seed_key: str, draws: int) -> dict:
    deltas = [pair["delta"] for pair in pairs]
    up = sum(delta > 0 for delta in deltas)
    down = sum(delta < 0 for delta in deltas)
    summary = {
        "n": len(pairs),
        "up": up,
        "down": down,
        "unchanged": len(deltas) - up - down,
        "sign_p": sign_test(up, down),
        "mean_delta": sum(deltas) / len(deltas) if deltas else None,
        "mean_abs_delta": (
            sum(abs(delta) for delta in deltas) / len(deltas) if deltas else None
        ),
    }
    clusters: dict[str, list[float]] = defaultdict(list)
    for pair in pairs:
        clusters[pair["claim_id"]].append(pair["delta"])
    if len(clusters) >= 2:
        summary["ci_low"], summary["ci_high"] = bootstrap_cluster_stat(
            clusters,
            lambda sample: float(sample.mean()),
            draws=draws,
            seed=keyed_seed(f"decomposition:{seed_key}"),
        )
    else:
        summary["ci_low"] = summary["ci_high"] = None
    return summary


def clustered_rate_difference(
    exposed: list[dict],
    baseline: list[dict],
    indicator: Callable[[dict], int],
    seed_key: str,
    draws: int,
) -> dict | None:
    """Per-claim paired rate difference with a claim-cluster bootstrap CI.

    Claims contribute equally; only claims observed in both groups enter.
    """
    exposed_by: dict[str, list[int]] = defaultdict(list)
    baseline_by: dict[str, list[int]] = defaultdict(list)
    for row in exposed:
        exposed_by[row["claim_id"]].append(indicator(row))
    for row in baseline:
        baseline_by[row["claim_id"]].append(indicator(row))
    shared = sorted(set(exposed_by) & set(baseline_by))
    if len(shared) < 2:
        return None
    clusters = {
        claim: [
            sum(exposed_by[claim]) / len(exposed_by[claim])
            - sum(baseline_by[claim]) / len(baseline_by[claim])
        ]
        for claim in shared
    }
    low, high = bootstrap_cluster_stat(
        clusters,
        lambda sample: float(sample.mean()),
        draws=draws,
        seed=keyed_seed(f"decomposition:{seed_key}"),
    )
    differences = [cluster[0] for cluster in clusters.values()]
    return {
        "claims": len(shared),
        "mean_claim_difference": sum(differences) / len(differences),
        "ci_low": low,
        "ci_high": high,
    }


QUERY_DIRECTION = {
    ("believer", "stance_congruent"): 1,
    ("believer", "stance_incongruent"): -1,
    ("skeptic", "stance_congruent"): -1,
    ("skeptic", "stance_incongruent"): 1,
    ("neutral", "stance_congruent"): 1,
    ("neutral", "stance_incongruent"): -1,
}

SOURCE_CLASSES = (
    (
        "peer_reviewed",
        (
            "pubmed",
            "ncbi.nlm.nih.gov",
            "cochrane",
            "springer",
            "nature.com",
            "bmj.com",
            "thelancet.com",
            "jamanetwork.com",
            "nejm.org",
            "wiley.com",
            "sciencedirect.com",
            "plos.org",
            "frontiersin.org",
            "mdpi.com",
            "academic.oup.com",
            "cambridge.org",
            "tandfonline.com",
            "biomedcentral.com",
            "cell.com",
            "science.org",
            "annualreviews.org",
            "sagepub.com",
        ),
    ),
    (
        "government_health",
        (
            "cdc.gov",
            "nih.gov",
            "fda.gov",
            "who.int",
            "nhs.uk",
            "medlineplus.gov",
            "health.gov",
            "europa.eu",
            "canada.ca",
        ),
    ),
    (
        "consumer_health",
        (
            "healthline.com",
            "webmd.com",
            "verywell",
            "mayoclinic.org",
            "clevelandclinic.org",
            "medicalnewstoday.com",
            "everydayhealth.com",
        ),
    ),
    (
        "fact_check",
        (
            "snopes.com",
            "factcheck.org",
            "politifact.com",
            "fullfact.org",
            "healthfeedback.org",
            "sciencefeedback.co",
        ),
    ),
)


def classify_source(domain: str) -> str:
    """Dotted needles match as registrable-domain suffixes, bare ones as infixes."""
    for name, needles in SOURCE_CLASSES:
        for needle in needles:
            if "." in needle:
                if domain == needle or domain.endswith("." + needle):
                    return name
            elif needle in domain:
                return name
    return "other"


def flip_gap_permutation(
    records: list[dict],
    pairs: list[dict],
    draws: int,
) -> dict:
    """Cluster permutation test on the B-S minus N-N flip-rate gap.

    Within every applicable claim-system cell the core run scores are
    exchangeable under the null of no framing effect; each draw shuffles the
    scores across that cell's framing-repeat slots and recomputes both flip
    indicators from the same aggregation rule as the primary analysis.
    """
    index = score_index(records, "core")
    cells = []
    for pair in pairs:
        if not pair["bs_available"] or not pair["nn_available"]:
            continue
        slots = sorted(
            (framing, repeat)
            for claim_id, system, framing, repeat in index
            if claim_id == pair["claim_id"] and system == pair["system"]
        )
        scores = [
            index[(pair["claim_id"], pair["system"], framing, repeat)]["score"]
            for framing, repeat in slots
        ]
        cells.append((slots, scores))
    if not cells:
        return {"estimable": False, "diagnostic": "no jointly applicable cells"}

    def gap(cell_scores: list[list[int]]) -> float:
        bs_flips = nn_flips = 0
        for (slots, _), scores in zip(cells, cell_scores):
            grouped: dict[str, list[int]] = defaultdict(list)
            neutral: dict[int, int] = {}
            for (framing, repeat), score in zip(slots, scores):
                grouped[framing].append(score)
                if framing == "neutral":
                    neutral[repeat] = score
            believer = sum(grouped["believer"]) / len(grouped["believer"])
            skeptic = sum(grouped["skeptic"]) / len(grouped["skeptic"])
            bs_flips += int(sign(believer) * sign(skeptic) < 0)
            first_repeat, second_repeat = sorted(neutral)[:2]
            nn_flips += int(
                sign(neutral[first_repeat]) * sign(neutral[second_repeat]) < 0
            )
        return bs_flips / len(cells) - nn_flips / len(cells)

    observed = gap([scores for _, scores in cells])
    rng = np.random.default_rng(keyed_seed("decomposition:flip_gap_permutation"))
    at_least = 0
    for _ in range(draws):
        permuted = []
        for _, scores in cells:
            shuffled = list(scores)
            rng.shuffle(shuffled)
            permuted.append(shuffled)
        if gap(permuted) >= observed - 1e-12:
            at_least += 1
    return {
        "estimable": True,
        "cells": len(cells),
        "observed_gap": observed,
        "draws": draws,
        "p_one_sided": (1 + at_least) / (1 + draws),
    }


HIGH_MOVEMENT_TOP_N = 14


def phase2_denominators(
    phase2_records: list[dict], ignored: list[dict], schedule_rows: list[dict]
) -> dict:
    """Per-arm accounting for the second-round arms, kept out of the totals."""

    planned = Counter(str(row.get("arm")) for row in schedule_rows)
    applicable = Counter(row["arm"] for row in phase2_records)
    failed = Counter(
        row["arm"]
        for row in ignored
        if row.get("arm") in PHASE2_ARMS
        and row.get("ignored_reason") == TERMINAL_FAILURE_STATUS
    )
    invalid = Counter(
        row["arm"]
        for row in ignored
        if row.get("arm") in PHASE2_ARMS
        and row.get("ignored_reason") == JUDGE_PARSE_INVALID_REASON
    )
    return {
        arm: {
            "planned": planned.get(arm, 0),
            "attempted": applicable[arm] + failed[arm] + invalid[arm],
            "terminal_success": applicable[arm] + invalid[arm],
            "parse_valid": applicable[arm],
            "analysis_applicable": applicable[arm],
            "terminal_failed": failed[arm],
            "judge_terminal_invalid": invalid[arm],
        }
        for arm in PHASE2_ARMS
        if planned.get(arm, 0) or applicable[arm] or failed[arm] or invalid[arm]
    }


def summarize_followups(
    records: list[dict],
    phase2_records: list[dict],
    domain_pairs: list[dict],
    claims_by_id: dict[str, dict],
    draws: int,
    ignored: list[dict],
    missingness: list[dict] | None = None,
) -> dict:
    """Second-round estimands, reported apart from the registered results."""

    import phase2_analysis as followup
    import robustness
    from phase2 import load_phase2_schedule

    phase2_schedule = load_phase2_schedule()
    query_judge = followup.summarize_query_judge_agreement(records)
    values = {
        "arms": phase2_denominators(phase2_records, ignored, phase2_schedule),
        "failures": robustness.summarize_failures(
            missingness or [],
            phase2_records,
            ignored,
            phase2_schedule,
            claims_by_id,
            pearson_chi_square,
        ),
        "citation_sensitivity": robustness.summarize_citation_sensitivity(
            records, paired_deltas, delta_summary, draws
        ),
        "control_arms": robustness.summarize_control_arms(
            records, score_index, paired_deltas, delta_summary, draws
        ),
        "query_label_noise": robustness.summarize_query_label_noise(
            records, query_judge, QUERY_DIRECTION
        ),
        "neutral_noise_by_system": followup.summarize_neutral_noise_by_system(
            records, phase2_records, delta_summary, draws
        ),
        "skeptic_mitigation": followup.summarize_skeptic_mitigation(
            records, phase2_records, delta_summary, draws
        ),
        "cross_replay": followup.summarize_cross_replay(
            records, phase2_records, delta_summary, draws
        ),
        "evidence_stance": followup.summarize_evidence_stance(
            records, bootstrap_cluster_stat, keyed_seed, draws
        ),
        "query_judge_cross_family": query_judge,
        "report_agreement_by_system": followup.summarize_report_agreement_by_system(
            records
        ),
        "high_movement": followup.characterize_high_movement(
            records, claims_by_id, HIGH_MOVEMENT_TOP_N
        ),
        "paraphrase_ordinal": followup.summarize_paraphrase_ordinal(
            records, delta_summary, draws
        ),
        "jaccard_sensitivity": followup.summarize_jaccard_sensitivity(
            domain_pairs, records, bootstrap_cluster_stat, keyed_seed, draws
        ),
    }
    # The stance audit stopped early, so whether its claims are the ones that
    # actually moved decides how much the pool-composition null is worth.
    audited = set(values["evidence_stance"]["coverage"]["claim_ids"])
    movers = {row["claim_id"] for row in values["high_movement"]["claims"]}
    values["evidence_stance"]["coverage"]["high_movement_claims"] = len(movers)
    values["evidence_stance"]["coverage"]["high_movement_audited"] = len(audited & movers)
    return values


def summarize_decomposition(
    records: list[dict],
    core_flip_pairs: list[dict],
    draws: int,
) -> tuple[list[dict], dict]:
    """Three-way framing decomposition and supporting estimands.

    All quantities reuse the frozen ordinal scores and query labels; nothing
    here re-judges a run. Contrast deltas are first minus second framing.
    """
    values: dict = {"contrasts": {}}
    rows = []
    contrasts = (
        ("believer_vs_neutral", "believer", "neutral"),
        ("skeptic_vs_neutral", "skeptic", "neutral"),
        ("believer_vs_skeptic", "believer", "skeptic"),
    )
    for name, first, second in contrasts:
        pairs = paired_deltas(records, "core", first, second)
        entry = {"overall": delta_summary(pairs, f"score:{name}", draws)}
        for tier in TIERS:
            entry[tier] = delta_summary(
                [pair for pair in pairs if pair["tier"] == tier],
                f"score:{name}:{tier}",
                draws,
            )
        entry["by_system"] = {
            system: delta_summary(
                [pair for pair in pairs if pair["system"] == system],
                f"score:{name}:{system}",
                draws,
            )
            for system in SYSTEMS
        }
        entry["searches"] = delta_summary(
            paired_deltas(records, "core", first, second, field="queries"),
            f"searches:{name}",
            draws,
        )
        values["contrasts"][name] = entry
        rows.append({"measure": f"score_{name}", **entry["overall"]})
        rows.append({"measure": f"searches_{name}", **entry["searches"]})

    neutral_pairs = []
    index = score_index(records, "core")
    for pair in core_flip_pairs:
        if not pair["nn_available"]:
            continue
        repeats = sorted(
            repeat
            for claim_id, system, framing, repeat in index
            if claim_id == pair["claim_id"]
            and system == pair["system"]
            and framing == "neutral"
        )[:2]
        first_row = index[(pair["claim_id"], pair["system"], "neutral", repeats[0])]
        second_row = index[(pair["claim_id"], pair["system"], "neutral", repeats[1])]
        neutral_pairs.append(
            {
                "claim_id": pair["claim_id"],
                "tier": pair["tier"],
                "system": pair["system"],
                "repeat": 0,
                "first": second_row["score"],
                "second": first_row["score"],
                "delta": second_row["score"] - first_row["score"],
            }
        )
    noise = delta_summary(neutral_pairs, "score:neutral_repeat", draws)
    noise["movement_rate"] = (
        (noise["up"] + noise["down"]) / noise["n"] if noise["n"] else None
    )
    bs_overall = values["contrasts"]["believer_vs_skeptic"]["overall"]
    noise["bs_abs_ratio"] = (
        bs_overall["mean_abs_delta"] / noise["mean_abs_delta"]
        if noise["mean_abs_delta"]
        else None
    )
    values["neutral_noise"] = noise
    rows.append({"measure": "score_neutral_repeat", **noise})

    direction_values = {}
    direction_clusters: dict[str, dict[str, list[float]]] = {
        framing: defaultdict(list) for framing in ARM_FRAMINGS["core"]
    }
    for row in records:
        if row["arm"] != "core":
            continue
        for label in row["query_labels"]:
            direction = QUERY_DIRECTION.get((row["framing"], label), 0)
            direction_clusters[row["framing"]][row["claim_id"]].append(direction)
    for framing, clusters in direction_clusters.items():
        flat = [value for cluster in clusters.values() for value in cluster]
        total = len(flat)
        affirm = sum(value > 0 for value in flat)
        deny = sum(value < 0 for value in flat)
        entry = {
            "queries": total,
            "affirm": affirm,
            "deny": deny,
            "affirm_rate": affirm / total if total else None,
            "deny_rate": deny / total if total else None,
            "directional_rate": (affirm + deny) / total if total else None,
            "net": (affirm - deny) / total if total else None,
        }
        if len(clusters) >= 2:
            entry["net_ci_low"], entry["net_ci_high"] = bootstrap_cluster_stat(
                clusters,
                lambda sample: float(sample.mean()),
                draws=draws,
                seed=keyed_seed(f"decomposition:direction:{framing}"),
            )
        direction_values[framing] = entry
        rows.append({"measure": f"query_direction_{framing}", **entry})
    values["query_direction"] = direction_values

    hedging = {}
    core_rows = [row for row in records if row["arm"] == "core"]
    for framing in ARM_FRAMINGS["core"]:
        subset = [row for row in core_rows if row["framing"] == framing]
        hedged = sum(row["label"] == "hedged" for row in subset)
        contested = [row for row in subset if row["tier"] == "contested"]
        contested_hedged = sum(row["label"] == "hedged" for row in contested)
        hedging[framing] = {
            "hedged": hedged,
            "runs": len(subset),
            "rate": hedged / len(subset) if subset else None,
            "contested_hedged": contested_hedged,
            "contested_runs": len(contested),
            "contested_rate": (
                contested_hedged / len(contested) if contested else None
            ),
            "extreme_rate": (
                sum(row["label"] in ("endorses", "rejects") for row in subset)
                / len(subset)
                if subset
                else None
            ),
        }
    neutral_rows = [row for row in core_rows if row["framing"] == "neutral"]
    for framing in ("believer", "skeptic"):
        table = [
            [hedging[framing]["hedged"], hedging[framing]["runs"] - hedging[framing]["hedged"]],
            [hedging["neutral"]["hedged"], hedging["neutral"]["runs"] - hedging["neutral"]["hedged"]],
        ]
        hedging[framing]["fisher_vs_neutral_p"] = float(fisher_exact(table)[1])
        hedging[framing]["newcombe_vs_neutral"] = newcombe_difference(
            hedging[framing]["hedged"],
            hedging[framing]["runs"],
            hedging["neutral"]["hedged"],
            hedging["neutral"]["runs"],
        )
        hedging[framing]["cluster_vs_neutral"] = clustered_rate_difference(
            [row for row in core_rows if row["framing"] == framing],
            neutral_rows,
            lambda row: int(row["label"] == "hedged"),
            f"hedge_diff:{framing}",
            draws,
        )
    values["hedging_by_framing"] = hedging

    deviation_rd = {}
    for name, framing, tier in (
        ("believer_settled_false", "believer", "settled_false"),
        ("skeptic_settled_true", "skeptic", "settled_true"),
    ):
        exposed = [
            row for row in core_rows if row["framing"] == framing and row["tier"] == tier
        ]
        baseline = [
            row for row in core_rows if row["framing"] == "neutral" and row["tier"] == tier
        ]
        if exposed and baseline:
            deviation_rd[name] = newcombe_difference(
                sum(row["deviation"] for row in exposed),
                len(exposed),
                sum(row["deviation"] for row in baseline),
                len(baseline),
            )
    values["deviation_risk_difference"] = deviation_rd

    composition: dict[str, Counter] = {
        framing: Counter() for framing in ARM_FRAMINGS["core"]
    }
    for row in core_rows:
        for citation in row["citations"]:
            composition[row["framing"]][classify_source(normalize_host(citation))] += 1
    values["source_composition"] = {
        framing: {
            "citations": sum(counts.values()),
            **{
                name: {
                    "count": counts.get(name, 0),
                    "share": (
                        counts.get(name, 0) / sum(counts.values())
                        if sum(counts.values())
                        else None
                    ),
                }
                for name in ("peer_reviewed", "government_health", "consumer_health", "fact_check", "other")
            },
        }
        for framing, counts in composition.items()
    }

    replay_values = {}
    for name, first, second in contrasts:
        core_pairs = paired_deltas(records, "core", first, second)
        fixed_pairs = paired_deltas(records, "replay", first, second)
        fixed_keys = {
            (pair["claim_id"], pair["system"], pair["repeat"]) for pair in fixed_pairs
        }
        matched_core = [
            pair
            for pair in core_pairs
            if (pair["claim_id"], pair["system"], pair["repeat"]) in fixed_keys
        ]
        core_keys = {
            (pair["claim_id"], pair["system"], pair["repeat"]) for pair in matched_core
        }
        matched_fixed = [
            pair
            for pair in fixed_pairs
            if (pair["claim_id"], pair["system"], pair["repeat"]) in core_keys
        ]
        core_summary = delta_summary(matched_core, f"replay_core:{name}", draws)
        fixed_summary = delta_summary(matched_fixed, f"replay_fixed:{name}", draws)
        entry = {
            "matched_n": len(matched_core),
            "core": core_summary,
            "fixed": fixed_summary,
        }
        core_mean = core_summary["mean_delta"]
        if core_mean is not None and abs(core_mean) >= 0.05:
            entry["attenuation_share"] = 1 - fixed_summary["mean_delta"] / core_mean
            fixed_by_key = {
                (pair["claim_id"], pair["system"], pair["repeat"]): pair["delta"]
                for pair in matched_fixed
            }
            clusters: dict[str, list[float]] = defaultdict(list)
            for pair in matched_core:
                key = (pair["claim_id"], pair["system"], pair["repeat"])
                clusters[pair["claim_id"]].append(
                    pair["delta"] - fixed_by_key[key]
                )
            if len(clusters) >= 2:
                entry["attenuation_delta_ci_low"], entry["attenuation_delta_ci_high"] = (
                    bootstrap_cluster_stat(
                        clusters,
                        lambda sample: float(sample.mean()),
                        draws=draws,
                        seed=keyed_seed(f"decomposition:attenuation:{name}"),
                    )
                )
        else:
            entry["attenuation_share"] = None
            entry["placebo"] = True
        replay_values[name] = entry
        rows.append(
            {
                "measure": f"replay_{name}",
                "n": entry["matched_n"],
                "mean_delta": core_summary["mean_delta"],
                "fixed_mean_delta": fixed_summary["mean_delta"],
                "attenuation_share": entry.get("attenuation_share"),
            }
        )
    values["replay_ordinal"] = replay_values

    mitigation_index = score_index(records, "mitigation")
    mitigation_pairs = []
    for (claim_id, system, framing, repeat), row in sorted(mitigation_index.items()):
        core_row = index.get((claim_id, system, "believer", repeat))
        if core_row is None:
            continue
        mitigation_pairs.append(
            {
                "claim_id": claim_id,
                "tier": row["tier"],
                "system": system,
                "repeat": repeat,
                "first": row["score"],
                "second": core_row["score"],
                "delta": row["score"] - core_row["score"],
            }
        )
    mitigation_entry = {
        "overall": delta_summary(mitigation_pairs, "score:mitigation", draws),
        "contested": delta_summary(
            [pair for pair in mitigation_pairs if pair["tier"] == "contested"],
            "score:mitigation:contested",
            draws,
        ),
    }
    values["mitigation_ordinal"] = mitigation_entry
    rows.append({"measure": "score_mitigation_vs_believer", **mitigation_entry["overall"]})

    claim_movement: dict[str, list[float]] = defaultdict(list)
    claim_tiers: dict[str, str] = {}
    for pair in paired_deltas(records, "core", "believer", "skeptic"):
        claim_movement[pair["claim_id"]].append(pair["delta"])
        claim_tiers[pair["claim_id"]] = pair["tier"]
    per_claim = sorted(
        (
            {
                "claim_id": claim_id,
                "tier": claim_tiers[claim_id],
                "pairs": len(deltas),
                "mean_delta": sum(deltas) / len(deltas),
                "abs_movement": sum(abs(delta) for delta in deltas),
            }
            for claim_id, deltas in claim_movement.items()
        ),
        key=lambda item: -item["abs_movement"],
    )
    total_movement = sum(item["abs_movement"] for item in per_claim)
    concentration = {
        "claims": len(per_claim),
        "total_abs_movement": total_movement,
        "top5_share": None,
        "claims_for_80pct": None,
        "moving_claims": 0,
        "top_movers": [],
    }
    if total_movement > 0:
        concentration["top5_share"] = (
            sum(item["abs_movement"] for item in per_claim[:5]) / total_movement
        )
        running = 0.0
        for count, item in enumerate(per_claim, start=1):
            running += item["abs_movement"]
            if running / total_movement >= 0.8:
                concentration["claims_for_80pct"] = count
                break
        concentration["moving_claims"] = sum(
            item["abs_movement"] > 0 for item in per_claim
        )
        concentration["top_movers"] = [
            {key: item[key] for key in ("claim_id", "tier", "pairs", "mean_delta")}
            for item in per_claim[:5]
        ]
    values["claim_concentration"] = concentration

    values["flip_gap_permutation"] = flip_gap_permutation(
        records, core_flip_pairs, min(draws, 2000)
    )

    applicable_bs = sum(pair["bs_available"] for pair in core_flip_pairs)
    tier_counts = Counter(
        pair["tier"] for pair in core_flip_pairs if pair["bs_available"]
    )
    values["mde"] = {
        "assumption": (
            "smallest one-way discordant count with two-sided mid-p below 0.05 "
            "is 5, assuming no baseline-direction discordance"
        ),
        "pooled_pairs": applicable_bs,
        "pooled_mde_points": 5 / applicable_bs if applicable_bs else None,
        "per_tier": {
            tier: {
                "pairs": tier_counts.get(tier, 0),
                "mde_points": (
                    5 / tier_counts[tier] if tier_counts.get(tier) else None
                ),
            }
            for tier in TIERS
        },
    }
    return rows, values


def load_luna_sample_ids(
    path: Path,
    expected: int,
    expected_ids: list[str] | None = None,
) -> list[str]:
    value = read_json(path)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: Luna sample must be an object")
    run_ids = value.get("run_ids")
    if (
        value.get("seed") != "realm2026-luna-v1"
        or value.get("population") != "core"
        or value.get("n") != expected
        or not isinstance(run_ids, list)
        or len(run_ids) != expected
        or not all(isinstance(run_id, str) and run_id for run_id in run_ids)
        or len(set(run_ids)) != expected
    ):
        raise ValueError(f"{path}: Luna sample has an invalid frozen denominator")
    if expected_ids is not None and run_ids != sorted(expected_ids):
        raise ValueError(
            f"{path}: Luna sample differs from the deterministic stratified selection"
        )
    return list(run_ids)


def luna_attempt_records(
    records: list[dict],
    expected: int,
    sample_ids: list[str] | None = None,
) -> list[dict]:
    attempted = [row for row in records if row.get("luna_attempted") is True]
    if sample_ids is not None:
        if len(sample_ids) != expected or len(set(sample_ids)) != expected:
            raise ValueError("Luna sample IDs do not match the planned denominator")
        planned = set(sample_ids)
        outside = sorted(row["run_id"] for row in attempted if row["run_id"] not in planned)
        if outside:
            raise ValueError(
                "Luna labels outside the frozen sample would be replacements: "
                f"{outside[:3]}"
            )
        attempted = [row for row in attempted if row["run_id"] in planned]
    if len(attempted) > expected:
        raise ValueError(
            f"Luna has {len(attempted)} attempts for a planned denominator of {expected}"
        )
    if any(row["arm"] != "core" for row in attempted):
        raise ValueError("Luna agreement sample must contain only core runs")
    if expected == 300 and len(attempted) == expected:
        cells = Counter((row["system"], row["framing"]) for row in attempted)
        tiers = Counter(row["tier"] for row in attempted)
        if set(cells.values()) != {25} or len(cells) != 12:
            raise ValueError(
                f"Luna sample is not balanced by system and framing: {dict(cells)}"
            )
        if tiers != Counter({tier: 100 for tier in TIERS}):
            raise ValueError(f"Luna sample is not balanced by tier: {dict(tiers)}")
    return attempted


def luna_denominator_summary(
    records: list[dict],
    expected: int,
    sample_ids: list[str] | None = None,
) -> dict:
    attempted = luna_attempt_records(records, expected, sample_ids)
    valid = [row for row in attempted if row.get("luna_parse_valid") is True]
    invalid = [
        row
        for row in attempted
        if str(row.get("luna_inapplicable_reason") or "").startswith(
            JUDGE_PARSE_INVALID_REASON
        )
    ]
    if len(valid) + len(invalid) != len(attempted):
        raise ValueError("Luna attempted records have an unresolved parse outcome")
    reasons = Counter(
        row.get("luna_inapplicable_reason") or "unspecified"
        for row in invalid
    )
    if len(attempted) < expected:
        reasons["luna_not_attempted"] += expected - len(attempted)
    return {
        "planned": expected,
        "attempted": len(attempted),
        "terminal_success": len(attempted),
        "parse_valid": len(valid),
        "analysis_applicable": len(valid),
        "terminal_invalid": len(invalid),
        "not_attempted": expected - len(attempted),
        "replacement_policy": "none_frozen_sample",
        "replacement_attempts": 0,
        "inapplicable_reason_codes": dict(sorted(reasons.items())),
    }


def luna_agreement_rows(
    records: list[dict],
    expected: int,
    sample_ids: list[str] | None = None,
) -> list[dict]:
    attempted = luna_attempt_records(records, expected, sample_ids)
    selected = [row for row in attempted if row.get("luna_parse_valid") is True]
    return [
        {
            "run_id": row["run_id"],
            "reference": row["luna_label"],
            "comparison": row["label"],
            "framing": row["framing"],
            "system": row["system"],
        }
        for row in selected
    ]


def agreement_groups(rows: list[dict]) -> list[tuple[str, str, list[dict]]]:
    groups = [("overall", "all", rows)]
    for label in LABELS:
        groups.append(
            ("reference_label", label, [row for row in rows if row["reference"] == label])
        )
    for framing in ("neutral", "believer", "skeptic"):
        groups.append(("framing", framing, [row for row in rows if row["framing"] == framing]))
    for system in SYSTEMS:
        groups.append(("system", system, [row for row in rows if row["system"] == system]))
    for framing in ("neutral", "believer", "skeptic"):
        for system in SYSTEMS:
            groups.append(
                (
                    "framing_system",
                    f"{framing}:{system}",
                    [
                        row
                        for row in rows
                        if row["framing"] == framing and row["system"] == system
                    ],
                )
            )
    return [
        (dimension, group, subset)
        for dimension, group, subset in groups
        if subset or dimension == "overall"
    ]


def summarize_agreement(luna_rows: list[dict]) -> tuple[list[dict], list[dict], dict]:
    summaries = []
    confusion = []
    values = {}
    for comparison_name, rows in (("luna_vs_gemini", luna_rows),):
        for dimension, group, subset in agreement_groups(rows):
            if not subset:
                summaries.append(
                    {
                        "comparison": comparison_name,
                        "dimension": dimension,
                        "group": group,
                        "agreements": 0,
                        "n": 0,
                        "percent_agreement": "",
                        "kappa": "",
                        "kappa_estimable": False,
                        "analysis_applicable": False,
                        "inapplicable_reason": "no_parse_valid_labels",
                    }
                )
                continue
            reference = [row["reference"] for row in subset]
            comparison = [row["comparison"] for row in subset]
            agreements = sum(left == right for left, right in zip(reference, comparison))
            kappa = cohen_kappa(reference, comparison, LABELS)
            summaries.append(
                {
                    "comparison": comparison_name,
                    "dimension": dimension,
                    "group": group,
                    "agreements": agreements,
                    "n": len(subset),
                    "percent_agreement": agreements / len(subset),
                    "kappa": "" if kappa is None else kappa,
                    "kappa_estimable": kappa is not None,
                    "analysis_applicable": True,
                    "inapplicable_reason": "",
                }
            )
            if dimension == "reference_label":
                continue
            counts = Counter((row["reference"], row["comparison"]) for row in subset)
            for reference_label in LABELS:
                for comparison_label in LABELS:
                    confusion.append(
                        {
                            "comparison": comparison_name,
                            "dimension": dimension,
                            "group": group,
                            "reference_label": reference_label,
                            "gemini_label": comparison_label,
                            "count": counts[(reference_label, comparison_label)],
                        }
                    )
        overall = next(
            row
            for row in summaries
            if row["comparison"] == comparison_name and row["dimension"] == "overall"
        )
        values[comparison_name] = {
            "agreements": overall["agreements"],
            "n": overall["n"],
            "percent_agreement": overall["percent_agreement"],
            "kappa": overall["kappa"],
            "kappa_estimable": overall["kappa_estimable"],
        }
    return summaries, confusion, values


def final_audit_binding(audit_result: object) -> dict:
    if (
        not isinstance(audit_result, dict)
        or audit_result.get("phase") != "full"
        or audit_result.get("status") != "green"
        or audit_result.get("errors")
    ):
        raise ValueError(
            "final analysis requires a current green full audit; "
            "manifest-local hashes are insufficient"
        )
    return {
        "required": True,
        "status": "verified",
        "phase": audit_result["phase"],
        "timestamp": audit_result["generated_at"],
    }


def luna_slot_status(run_id: str, ledger_path: Path, slots: dict | None) -> str | None:
    """Terminal status of a frozen Luna call slot, from the record or the ledger."""
    if slots is not None:
        return slots.get(run_id)
    import sqlite3

    from judge import LUNA_CONCLUSION_INDEX

    db = sqlite3.connect(ledger_path)
    try:
        row = db.execute(
            "SELECT status FROM calls WHERE run_id=? AND call_index=?",
            (run_id, LUNA_CONCLUSION_INDEX),
        ).fetchone()
    finally:
        db.close()
    return row[0] if row is not None else None


def require_luna_resolution(
    records: list[dict],
    luna_sample_ids: list[str],
    ledger_path: Path | None = None,
    slot_record_path: Path | None = None,
) -> dict:
    """Refuse final analysis while any frozen Luna sample member is unresolved.

    Resolved means a parse-valid label, a terminal-invalid record, or a
    terminally closed call slot (a reconciled transport failure). The
    call ledger is private, so a released copy reads the recorded slot statuses
    for the sample instead.
    """
    from config import ARCHIVE

    if ledger_path is None:
        ledger_path = ARCHIVE / "call_ledger.sqlite3"
    slots = None
    if slot_record_path is not None:
        record = json.loads(slot_record_path.read_text(encoding="utf-8"))
        if record.get("sample_size") != len(luna_sample_ids):
            raise ValueError(
                f"{slot_record_path}: call-slot record covers a different Luna sample"
            )
        slots = record["unattempted"]
    attempted = {row["run_id"] for row in records if row.get("luna_attempted")}
    missing = [run_id for run_id in luna_sample_ids if run_id not in attempted]
    closed = []
    for run_id in missing:
        if luna_slot_status(run_id, ledger_path, slots) != "failed":
            raise ValueError(
                f"frozen Luna sample member {run_id} has no outcome and "
                "no terminally closed call slot; Luna judging is incomplete"
            )
        closed.append(run_id)
    return {
        "planned": len(luna_sample_ids),
        "attempted": len(attempted & set(luna_sample_ids)),
        "terminally_closed": sorted(closed),
    }


def run_analysis(
    claims_path: Path = CLAIMS_PATH,
    runs_path: Path = RUNS_DIR,
    schedule_path: Path = SCHEDULE_PATH,
    trim_record_dir: Path = TRIM_RECORD_DIR,
    luna_sample_path: Path = DATA / "luna_sample.json",
    output_path: Path = DATA / "analysis",
    draws: int = BOOTSTRAP_DRAWS,
    expected_luna: int = 300,
    strict_matrix: bool = True,
    audit_record_path: Path | None = None,
    luna_slots_path: Path | None = None,
) -> dict:
    audit_binding = {
        "required": strict_matrix,
        "status": "not_required_exploratory",
    }
    if strict_matrix:
        if audit_record_path is not None:
            # The audit binds the private archive, which the public release does
            # not carry, so a released copy verifies against the recorded result
            # instead of re-running it.
            audit_result = json.loads(audit_record_path.read_text(encoding="utf-8"))
        else:
            from audit import run_audit

            audit_result = run_audit("full")
        audit_binding = final_audit_binding(audit_result)
    claims, claims_by_id = load_claim_file(claims_path)
    loaded_records, ignored = load_manifests(runs_path, claims_by_id)
    records = [row for row in loaded_records if row.get("arm") in ANALYZED_ARMS]
    kimi_records = [row for row in loaded_records if row.get("arm") == "kimi"]
    phase2_records = [row for row in loaded_records if row.get("arm") in PHASE2_ARMS]
    terminal_failures = [
        row
        for row in ignored
        if row.get("arm") in ANALYZED_ARMS
        and row.get("ignored_reason") == TERMINAL_FAILURE_STATUS
    ]
    judge_invalids = [
        row
        for row in ignored
        if row.get("arm") in ANALYZED_ARMS
        and row.get("ignored_reason") == JUDGE_PARSE_INVALID_REASON
    ]
    kimi_terminal_failures = [
        row
        for row in ignored
        if row.get("arm") == "kimi"
        and row.get("ignored_reason") == TERMINAL_FAILURE_STATUS
    ]
    kimi_judge_invalids = [
        row
        for row in ignored
        if row.get("arm") == "kimi"
        and row.get("ignored_reason") == JUDGE_PARSE_INVALID_REASON
    ]
    nonanalysis_ignored = [
        row
        for row in ignored
        if row not in terminal_failures
        and row not in judge_invalids
        and row not in kimi_terminal_failures
        and row not in kimi_judge_invalids
    ]
    if strict_matrix:
        schedule_value = read_json(schedule_path)
        if not isinstance(schedule_value, list) or not all(
            isinstance(row, dict) for row in schedule_value
        ):
            raise ValueError(f"{schedule_path}: schedule must be a list of objects")
        trim_state = verify_trim_records(
            schedule_value, claims, trim_record_dir
        )
        matrix = validate_matrix(
            records,
            claims,
            schedule_rows=schedule_value,
            trim_profile=str(trim_state["profile"]),
            terminal_failures=terminal_failures,
            judge_invalids=judge_invalids,
        )
        matrix["trim_record_count"] = trim_state["record_count"]
        matrix["trim_record_sha256"] = trim_state["last_record_sha256"]
        analysis_schedule = schedule_value
        kimi_matrix = validate_kimi_probe(
            kimi_records,
            schedule_value,
            kimi_terminal_failures,
            kimi_judge_invalids,
        )
    else:
        observed = [*records, *terminal_failures, *judge_invalids]
        arm_counts = Counter(row["arm"] for row in observed)
        applicable_counts = Counter(row["arm"] for row in records)
        failure_counts = Counter(row["arm"] for row in terminal_failures)
        invalid_counts = Counter(row["arm"] for row in judge_invalids)
        reason_codes = Counter()
        for row in terminal_failures:
            failure = row.get("failure") or {}
            stage = str(failure.get("stage") or "unknown_stage")
            error_type = str(failure.get("error_type") or "unknown_error")
            reason_codes[f"terminal_failure:{stage}:{error_type}"] += 1
        for row in judge_invalids:
            failure = row.get("judge_failure") or {}
            reason = str(failure.get("reason_code") or JUDGE_PARSE_INVALID_REASON)
            call_kind = str(failure.get("call_kind") or "unknown_call")
            reason_codes[f"{reason}:{call_kind}"] += 1
        matrix = {
            "runs": len(records),
            "arms": dict(arm_counts),
            "arm_denominators": {
                arm: {
                    "planned": arm_counts[arm],
                    "attempted": arm_counts[arm],
                    "terminal_success": applicable_counts[arm] + invalid_counts[arm],
                    "parse_valid": applicable_counts[arm],
                    "analysis_applicable": applicable_counts[arm],
                    "terminal_failed": failure_counts[arm],
                    "judge_terminal_invalid": invalid_counts[arm],
                }
                for arm in ANALYZED_ARMS
            },
            "denominators": {
                "planned": len(observed),
                "attempted": len(observed),
                "terminal_success": len(records) + len(judge_invalids),
                "parse_valid": len(records),
                "analysis_applicable": len(records),
                "terminal_failed": len(terminal_failures),
                "judge_terminal_invalid": len(judge_invalids),
            },
            "inapplicable_reason_codes": dict(sorted(reason_codes.items())),
            "trim_profile": "not_validated",
        }
        analysis_schedule = [
            {
                field: row[field]
                for field in (
                    "run_id",
                    "arm",
                    "claim_id",
                    "framing",
                    "system",
                    "model",
                    "repeat",
                )
            }
            for row in observed
        ]
        kimi_observed = [
            *kimi_records,
            *kimi_terminal_failures,
            *kimi_judge_invalids,
        ]
        kimi_matrix = {
            "included": bool(kimi_observed),
            "status": "exploratory_observed" if kimi_observed else "not_included",
            "scope": "out_of_band_appendix_only",
            "model": KIMI_MODEL if kimi_observed else "",
            "planned": len(kimi_observed),
            "attempted": len(kimi_observed),
            "terminal_success": len(kimi_records) + len(kimi_judge_invalids),
            "parse_valid": len(kimi_records),
            "analysis_applicable": len(kimi_records),
            "terminal_failed": len(kimi_terminal_failures),
            "judge_terminal_invalid": len(kimi_judge_invalids),
            "inapplicable_reason_codes": {},
        }
    denominator_rows = missingness_rows(
        analysis_schedule,
        records,
        terminal_failures,
        claims_by_id,
        judge_invalids,
    )
    kimi_denominator_rows = missingness_rows(
        analysis_schedule,
        kimi_records,
        kimi_terminal_failures,
        claims_by_id,
        kimi_judge_invalids,
        arms=PROBE_ARMS,
    )
    screen_summary = claim_screen_summary(claims)
    records_by_id = {row["run_id"]: row for row in records}
    raw_manifests = []
    for manifest_path in sorted(runs_path.glob("*.json")):
        raw_manifest = read_json(manifest_path)
        if isinstance(raw_manifest, dict):
            raw_manifests.append(raw_manifest)
    from judge import select_luna_sample

    deterministic_luna_ids = select_luna_sample(raw_manifests, claims_by_id)
    luna_sample_ids = load_luna_sample_ids(
        luna_sample_path,
        expected_luna,
        expected_ids=deterministic_luna_ids,
    )
    luna_denominators = luna_denominator_summary(
        records,
        expected_luna,
        luna_sample_ids,
    )
    if strict_matrix:
        audit_binding["luna_resolution"] = require_luna_resolution(
            records, luna_sample_ids, slot_record_path=luna_slots_path
        )
    luna_rows = luna_agreement_rows(records, expected_luna, luna_sample_ids)

    core_flip_pairs = flip_pairs(records, claims, analysis_schedule)
    flip_summary, mcnemar_rows, flip_values = summarize_flips(core_flip_pairs, draws)

    deviation_rows, odds_rows, gee_rows, deviation_values = summarize_deviation(records)
    hedging_rows, hedging_tests, hedging_values = summarize_hedging(records)
    search_rows, search_values = summarize_search_counts(records)
    core_model_rows = [row for row in records if row["arm"] == "core"]
    ordinal_rows, ordinal_diagnostic = fit_proportional_odds(core_model_rows)

    query_runs = query_run_rows(records)
    query_summary, query_values = summarize_queries(query_runs, draws)

    domain_pairs = source_pairs(records, claims, analysis_schedule)
    source_summary, source_values = summarize_sources(domain_pairs, draws)

    fixed_pairs = replay_pairs(records, claims, core_flip_pairs, analysis_schedule)
    replay_summary, replay_values = summarize_replay(fixed_pairs)

    secondary_pairs = secondary_flip_pairs(
        records, claims, core_flip_pairs, analysis_schedule
    )
    secondary_summary, secondary_values = summarize_secondary_flips(secondary_pairs)
    attribution_rows, attribution_values = summarize_attribution(records)

    mitigation_rows, mitigation_values = summarize_mitigation(records)

    decomposition_rows, decomposition_values = summarize_decomposition(
        records, core_flip_pairs, draws
    )

    followup_values = summarize_followups(
        records,
        phase2_records,
        domain_pairs,
        claims_by_id,
        draws,
        ignored,
        denominator_rows,
    )

    kimi_summary_rows, kimi_values = summarize_kimi_probe(kimi_records, kimi_matrix)

    agreement_rows, confusion_rows, agreement_values = summarize_agreement(luna_rows)
    agreement_values["luna_vs_gemini"]["denominators"] = luna_denominators

    import robustness

    output_path.mkdir(parents=True, exist_ok=True)
    write_csv(output_path / "missingness.csv", denominator_rows)
    write_csv(
        output_path / "failure_breakdown.csv",
        robustness.failure_rows(followup_values["failures"]),
    )
    write_csv(
        output_path / "citation_sensitivity.csv",
        robustness.citation_rows(followup_values["citation_sensitivity"]),
    )
    write_csv(
        output_path / "kimi_probe_missingness.csv",
        kimi_denominator_rows,
        [
            "run_id",
            "arm",
            "claim_id",
            "tier",
            "framing",
            "system",
            "repeat",
            "status",
            "planned",
            "attempted",
            "terminal_success",
            "parse_valid",
            "analysis_applicable",
            "reason_code",
            "inapplicable_reason",
        ],
    )
    write_csv(output_path / "kimi_probe.csv", kimi_summary_rows)
    write_csv(output_path / "runs.csv", run_table(records))
    write_csv(output_path / "flip_pairs.csv", core_flip_pairs)
    write_csv(output_path / "flip_summary.csv", flip_summary)
    write_csv(output_path / "mcnemar.csv", mcnemar_rows)
    write_csv(output_path / "deviation.csv", deviation_rows)
    write_csv(output_path / "odds_ratios.csv", odds_rows)
    write_csv(output_path / "hedging.csv", hedging_rows)
    write_csv(output_path / "hedging_tests.csv", hedging_tests)
    write_csv(output_path / "search_counts.csv", search_rows)
    write_csv(
        output_path / "gee.csv",
        gee_rows,
        [
            "tier",
            "term",
            "coefficient",
            "standard_error",
            "p_value",
            "odds_ratio",
            "odds_ratio_ci_low",
            "odds_ratio_ci_high",
        ],
    )
    write_csv(
        output_path / "proportional_odds.csv",
        ordinal_rows,
        [
            "term",
            "term_type",
            "coefficient",
            "cutpoint",
            "cluster_standard_error",
            "z_value",
            "p_value",
            "ci_low",
            "ci_high",
            "odds_ratio",
            "odds_ratio_ci_low",
            "odds_ratio_ci_high",
        ],
    )
    write_csv(output_path / "query_runs.csv", query_runs)
    write_csv(output_path / "query_mirroring.csv", query_summary)
    write_csv(output_path / "source_pairs.csv", domain_pairs)
    write_csv(output_path / "source_jaccard.csv", source_summary)
    write_csv(output_path / "replay_pairs.csv", fixed_pairs)
    write_csv(output_path / "replay.csv", replay_summary)
    write_csv(output_path / "secondary_flip_pairs.csv", secondary_pairs)
    write_csv(output_path / "secondary_flips.csv", secondary_summary)
    write_csv(output_path / "attribution.csv", attribution_rows)
    write_csv(output_path / "mitigation.csv", mitigation_rows)
    write_csv(output_path / "decomposition.csv", decomposition_rows)
    write_csv(output_path / "judge_agreement.csv", agreement_rows)
    confusion_fields = [
        "comparison",
        "dimension",
        "group",
        "reference_label",
        "gemini_label",
        "count",
    ]
    write_csv(output_path / "judge_confusion.csv", confusion_rows, confusion_fields)

    runs_with_all_citations = sum(
        row["all_citations_in_results"] for row in records
    )
    diagnostics = {
        "input": {
            **matrix,
            "claims": len(claims),
            **screen_summary,
            "ignored_runs": len(nonanalysis_ignored),
            "terminal_failed_runs": len(terminal_failures),
            "judge_terminal_invalid_runs": len(judge_invalids),
            "luna_labels": len(luna_rows),
            "luna_denominators": luna_denominators,
            "audit_binding": audit_binding,
            "kimi_probe": kimi_matrix,
            "runs_with_all_citations_in_results": runs_with_all_citations,
        },
        "models": {
            "gee": deviation_values["gee"],
            "proportional_odds": ordinal_diagnostic,
        },
        "definitions": {
            "run_denominators": (
                "planned schedule rows, manifests attempted, terminally successful "
                "trajectories, parse-valid judged manifests, and analysis-applicable "
                "runs are reported separately; terminal failures and judge-invalid "
                "outputs are never imputed"
            ),
            "primary_judge_invalid": (
                "a schema-invalid primary judge output receives no replacement and "
                "makes that run inapplicable to all outcome analyses"
            ),
            "luna_denominator": (
                "the frozen Luna sample receives no replacements; agreement uses only "
                "parse-valid labels and reports planned, attempted, and parse-valid n"
            ),
            "condition_side": (
                "sign of the mean ordinal score over runs retained by the "
                "authenticated schedule profile"
            ),
            "zero_mean_primary": "non-flip",
            "zero_mean_sensitivity": "pair excluded",
            "neutral_repeat": (
                "computed only for claim-system cells retaining both core repeats"
            ),
            "neutral_query_rate": "either directional query label",
            "source_pairing": "believer and skeptic runs matched by repeat index",
            "domain": "lowercase URL hostname with leading www removed",
            "query_tokens": "set of lowercase ASCII alphanumeric tokens across a run's queries",
            "replay": "descriptive attenuation, not a causal mediation estimate",
            "query_direction": (
                "framing-relative stance labels recoded to absolute claim "
                "direction: believer and neutral congruent labels affirm the "
                "claim, skeptic congruent labels deny it; net is affirm minus "
                "deny over all labeled queries"
            ),
            "decomposition_pairing": (
                "run-level pairs match claim, system, and repeat; contrast "
                "deltas are first minus second framing on the ordinal score, "
                "with claim-cluster bootstrap intervals"
            ),
            "midp_mcnemar": (
                "two-sided mid-p McNemar is the primary paired test per "
                "Fagerland et al. 2013; the exact conditional p is reported "
                "alongside"
            ),
            "hedging_and_deviation_differences": (
                "hedging contrasts report the run-level rate difference with a "
                "claim-cluster bootstrap interval; the Fisher test and Newcombe "
                "interval on the same contrast, and the rare-event deviation "
                "risk differences, are unadjusted run-level quantities"
            ),
        },
    }
    values = {
        "input": {
            "claims": len(claims),
            **screen_summary,
            "analyzed_runs": len(records),
            "runs_with_all_citations_in_results": runs_with_all_citations,
            **matrix["denominators"],
            "arms": matrix["arm_denominators"],
            "arm_claims": {
                arm: len(
                    {
                        row["claim_id"]
                        for row in analysis_schedule
                        if row.get("arm") == arm
                    }
                )
                for arm in ANALYZED_ARMS
            },
            "live_search_runs": sum(row["arm"] != "replay" for row in records),
            "queries": sum(len(row["queries"]) for row in records),
            "trim_profile": matrix.get("trim_profile"),
            "schedule_sha256": matrix.get("schedule_sha256"),
        },
        "flip": {
            **flip_values,
            "summary": flip_summary,
            "mcnemar_by_group": mcnemar_rows,
        },
        "deviation": {
            **{key: value for key, value in deviation_values.items() if key != "gee"},
            "summary": deviation_rows,
            "odds_ratio_summary": odds_rows,
            "gee_terms": gee_rows,
        },
        "query_mirroring": {**query_values, "summary": query_summary},
        "hedging": {**hedging_values, "summary": hedging_rows},
        "search_counts": search_values,
        "source_jaccard": {**source_values, "summary": source_summary},
        "replay": {**replay_values, "summary": replay_summary},
        "followups": followup_values,
        "secondary_flips": {
            **secondary_values,
            "summary": secondary_summary,
        },
        "attribution": {**attribution_values, "summary": attribution_rows},
        "mitigation": {**mitigation_values, "summary": mitigation_rows},
        "decomposition": decomposition_values,
        "kimi_probe": {**kimi_values, "summary": kimi_summary_rows},
        "judge_agreement": {**agreement_values, "summary": agreement_rows},
        "models": {
            "gee": deviation_values["gee"],
            "proportional_odds": ordinal_diagnostic,
            "proportional_odds_terms": ordinal_rows,
        },
    }
    atomic_write(output_path / "diagnostics.json", pretty_json(diagnostics))
    atomic_write(output_path / "values.json", pretty_json(values))
    return values


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze the complete, judged FramedFacts-48 run manifests."
    )
    parser.add_argument("--claims", type=Path, default=CLAIMS_PATH)
    parser.add_argument("--runs", type=Path, default=RUNS_DIR)
    parser.add_argument("--schedule", type=Path, default=SCHEDULE_PATH)
    parser.add_argument("--trim-record-dir", type=Path, default=TRIM_RECORD_DIR)
    parser.add_argument("--luna-sample", type=Path, default=DATA / "luna_sample.json")
    parser.add_argument(
        "--luna-call-slots",
        type=Path,
        default=None,
        help=(
            "recorded terminal status of the frozen Luna call slots, read in "
            "place of the private call ledger"
        ),
    )
    parser.add_argument("--output", type=Path, default=DATA / "analysis")
    parser.add_argument(
        "--audit-record",
        type=Path,
        default=None,
        help=(
            "verify against a recorded green full audit instead of running one, "
            "which is how the public release reproduces the analysis"
        ),
    )
    args = parser.parse_args()
    values = run_analysis(
        claims_path=args.claims,
        runs_path=args.runs,
        schedule_path=args.schedule,
        trim_record_dir=args.trim_record_dir,
        luna_sample_path=args.luna_sample,
        output_path=args.output,
        audit_record_path=args.audit_record,
        luna_slots_path=args.luna_call_slots,
    )
    print(
        f"wrote measured analysis for {values['input']['analyzed_runs']} runs "
        f"to {args.output}"
    )


if __name__ == "__main__":
    main()
