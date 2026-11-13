from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import time
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

import httpx
from ddgs import DDGS
from dotenv import load_dotenv

from config import (
    ARCHIVE,
    CLAIMS_PATH,
    DATA,
    ROOT,
    WORKING,
    atomic_write,
    canonical_json,
    now_iso,
    pretty_json,
    sha256_file,
)


TIERS = ("settled_false", "settled_true", "contested")
DATASETS = ("pubhealth", "healthfc")
DATASET_QUOTA = 8
DEFAULT_SEED = 20260801
DEFAULT_TOPIC_CAP = 4

DEFAULT_LABEL_MAP = {
    "false": "settled_false",
    "refuted": "settled_false",
    "incorrect": "settled_false",
    "fake": "settled_false",
    "true": "settled_true",
    "supported": "settled_true",
    "correct": "settled_true",
    "accurate": "settled_true",
    "mixture": "contested",
    "mixed": "contested",
    "unproven": "contested",
    "insufficient evidence": "contested",
    "not enough evidence": "contested",
    "uncertain": "contested",
    "inconclusive": "contested",
    "undetermined": "contested",
    "partially true": "contested",
    "partly true": "contested",
    "half true": "contested",
}

SOURCE_FIELDS = {
    "source_id": ("source_id", "claim_id", "id", "public_id"),
    "claim": ("en_claim", "claim", "statement", "text", "question"),
    "source_label": ("source_label", "label", "verdict", "rating", "class"),
    "source_url": (
        "source_url",
        "url",
        "claim_url",
        "fact_check_url",
        "factcheck_url",
    ),
}


def clean_text(value: object) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def normalize_name(value: object) -> str:
    text = clean_text(value).casefold()
    return re.sub(r"[^a-z0-9]+", "", text)


def normalize_label(value: object) -> str:
    text = clean_text(value).casefold().replace("_", " ").replace("-", " ")
    return re.sub(r"\s+", " ", text).strip()


def normalize_dataset(value: object) -> str:
    name = normalize_name(value)
    aliases = {
        "pubhealth": "pubhealth",
        "healthfc": "healthfc",
        "healthfactchecking": "healthfc",
    }
    if name not in aliases:
        raise ValueError(f"unknown source dataset: {clean_text(value)!r}")
    return aliases[name]


def normalize_tier(value: object) -> str:
    label = normalize_label(value)
    aliases = {
        "settled false": "settled_false",
        "false": "settled_false",
        "settled true": "settled_true",
        "true": "settled_true",
        "contested": "contested",
        "mixed": "contested",
        "insufficient evidence": "contested",
    }
    if label not in aliases:
        raise ValueError(f"invalid verified tier: {clean_text(value)!r}")
    return aliases[label]


def parse_bool(value: object, context: str) -> bool:
    label = normalize_label(value)
    if label in {"true", "yes", "1", "eligible", "include"}:
        return True
    if label in {"false", "no", "0", "ineligible", "exclude"}:
        return False
    raise ValueError(f"{context}: eligible must be true or false")


def check_url(value: object, context: str) -> str:
    url = clean_text(value)
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"{context}: invalid URL: {url!r}")
    return url


def read_rows(path: Path) -> list[dict]:
    suffix = path.suffix.casefold()
    if suffix in {".csv", ".tsv"}:
        delimiter = "\t" if suffix == ".tsv" else ","
        with path.open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream, delimiter=delimiter))
    elif suffix in {".jsonl", ".ndjson"}:
        rows = []
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise TypeError(f"{path}:{line_number}: expected a JSON object")
                rows.append(value)
    elif suffix == ".json":
        with path.open(encoding="utf-8") as stream:
            rows = json.load(stream)
        if not isinstance(rows, list):
            raise TypeError(f"{path}: expected a JSON list")
    else:
        raise ValueError(f"unsupported input format: {path}")
    if not all(isinstance(row, dict) for row in rows):
        raise TypeError(f"{path}: every record must be an object")
    return rows


def field_value(row: dict, aliases: tuple[str, ...], context: str, required: bool) -> str:
    normalized = {normalize_name(key): value for key, value in row.items()}
    values = []
    for alias in aliases:
        value = clean_text(normalized.get(normalize_name(alias)))
        if value:
            values.append(value)
    distinct = list(dict.fromkeys(values))
    if len(distinct) > 1:
        raise ValueError(f"{context}: conflicting values for {aliases[0]}")
    if distinct:
        return distinct[0]
    if required:
        raise ValueError(f"{context}: missing {aliases[0]}")
    return ""


def raw_field_value(row: dict, aliases: tuple[str, ...], context: str) -> str:
    """Resolve one source field without changing its licensed text."""
    normalized = {normalize_name(key): value for key, value in row.items()}
    values = []
    for alias in aliases:
        value = normalized.get(normalize_name(alias))
        if value is not None and clean_text(value):
            values.append(str(value))
    distinct = list(dict.fromkeys(values))
    if len({clean_text(value) for value in distinct}) > 1:
        raise ValueError(f"{context}: conflicting values for {aliases[0]}")
    if distinct:
        return distinct[0]
    raise ValueError(f"{context}: missing {aliases[0]}")


def record_hash(row: dict) -> str:
    return hashlib.sha256(canonical_json(row).encode("utf-8")).hexdigest()


def load_source(path: Path, dataset: str, metadata: dict) -> tuple[list[dict], int]:
    rows = read_rows(path)
    records = []
    seen = set()
    dataset_url = check_url(metadata["source_url"], f"{dataset} metadata")
    for row_number, row in enumerate(rows, 1):
        context = f"{path.name}:{row_number}"
        source_id = field_value(row, SOURCE_FIELDS["source_id"], context, True)
        if source_id in seen:
            raise ValueError(f"{context}: duplicate source ID {source_id!r}")
        seen.add(source_id)
        source_url = field_value(row, SOURCE_FIELDS["source_url"], context, False)
        source_claim = (
            raw_field_value(row, SOURCE_FIELDS["claim"], context)
            if dataset == "healthfc"
            else field_value(row, SOURCE_FIELDS["claim"], context, True)
        )
        records.append(
            {
                "source_dataset": dataset,
                "source_id": source_id,
                "source_claim": source_claim,
                "source_label": field_value(row, SOURCE_FIELDS["source_label"], context, True),
                "source_url": check_url(source_url, context) if source_url else dataset_url,
                "source_url_scope": "item" if source_url else "dataset",
                "source_record_sha256": record_hash(row),
            }
        )
    return records, len(rows)


def load_metadata(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        metadata = json.load(stream)
    if not isinstance(metadata, dict):
        raise TypeError(f"{path}: expected a JSON object")
    normalized = {}
    for key, value in metadata.items():
        dataset = normalize_dataset(key)
        if dataset in normalized:
            raise ValueError(f"{path}: duplicate metadata for {dataset}")
        if not isinstance(value, dict):
            raise TypeError(f"{path}: metadata for {dataset} must be an object")
        for field in ("license", "source_url"):
            if not clean_text(value.get(field)):
                raise ValueError(f"{path}: {dataset} metadata is missing {field}")
        check_url(value["source_url"], f"{dataset} metadata")
        if value.get("license_url"):
            check_url(value["license_url"], f"{dataset} metadata")
        normalized[dataset] = value
    missing = set(DATASETS) - set(normalized)
    if missing:
        raise ValueError(f"{path}: missing metadata for {', '.join(sorted(missing))}")
    return normalized


def label_map(metadata: dict) -> dict[str, str]:
    mapping = dict(DEFAULT_LABEL_MAP)
    custom = metadata.get("label_map", {})
    if not isinstance(custom, dict):
        raise TypeError("label_map must be an object")
    for raw_label, tier in custom.items():
        normalized_tier = normalize_tier(tier)
        mapping[normalize_label(raw_label)] = normalized_tier
    return mapping


def adjudication_value(row: dict, name: str, context: str, required: bool = True) -> str:
    return field_value(row, (name,), context, required)


def load_adjudications(path: Path) -> tuple[dict[tuple[str, str], dict], dict]:
    rows = read_rows(path)
    adjudications = {}
    agreements = 0
    disagreements = 0
    eligible_count = 0
    for row_number, row in enumerate(rows, 1):
        context = f"{path.name}:{row_number}"
        normalized_fields = {normalize_name(key) for key in row}
        legacy_fields = {"reviewer1tier", "reviewer2tier"} & normalized_fields
        if legacy_fields:
            raise ValueError(
                f"{context}: legacy reviewer_* fields are not accepted; "
                "use screen_1_tier and screen_2_tier"
            )
        dataset = normalize_dataset(adjudication_value(row, "source_dataset", context))
        source_id = adjudication_value(row, "source_id", context)
        key = (dataset, source_id)
        if key in adjudications:
            raise ValueError(f"{context}: duplicate adjudication for {dataset}/{source_id}")
        eligible = parse_bool(adjudication_value(row, "eligible", context), context)
        item = {"eligible": eligible, "adjudication_record_sha256": record_hash(row)}
        if eligible:
            eligible_count += 1
            screening_method = normalize_label(
                adjudication_value(row, "screening_method", context)
            ).replace(" ", "_")
            if screening_method != "model_assisted_independent":
                raise ValueError(
                    f"{context}: screening_method must be "
                    "'model_assisted_independent'"
                )
            screen_1 = normalize_tier(adjudication_value(row, "screen_1_tier", context))
            screen_2 = normalize_tier(adjudication_value(row, "screen_2_tier", context))
            verified = normalize_tier(adjudication_value(row, "verified_tier", context))
            note = adjudication_value(row, "adjudication_note", context, False)
            if screen_1 == screen_2:
                agreements += 1
                if verified != screen_1:
                    raise ValueError(
                        f"{context}: verified_tier differs from agreeing screen tiers"
                    )
            else:
                disagreements += 1
                if not note:
                    raise ValueError(f"{context}: screen disagreement needs adjudication_note")
            item.update(
                {
                    "claim": adjudication_value(row, "claim", context, False),
                    "topic": adjudication_value(row, "topic", context),
                    "screen_1_tier": screen_1,
                    "screen_2_tier": screen_2,
                    "screening_method": screening_method,
                    "verified_tier": verified,
                    "verification_url": check_url(
                        adjudication_value(row, "verification_url", context), context
                    ),
                }
            )
            for field in ("verification_title", "verification_date", "adjudication_note"):
                value = adjudication_value(row, field, context, False)
                if value:
                    item[field] = value
        adjudications[key] = item
    review = {
        "rows": len(rows),
        "eligible": eligible_count,
        "screen_agreements": agreements,
        "screen_disagreements": disagreements,
    }
    return adjudications, review


def build_candidates(
    sources: list[dict], adjudications: dict, metadata: dict, seed: int
) -> tuple[list[dict], dict]:
    candidates = []
    unknown_labels = Counter()
    reviewed_keys = set()
    for source in sources:
        key = (source["source_dataset"], source["source_id"])
        review = adjudications.get(key)
        mapping = label_map(metadata[source["source_dataset"]])
        source_tier = mapping.get(normalize_label(source["source_label"]))
        if source_tier is None:
            unknown_labels[(source["source_dataset"], source["source_label"])] += 1
        if review is None:
            continue
        reviewed_keys.add(key)
        if not review["eligible"]:
            continue
        if source_tier is None:
            raise ValueError(
                f"eligible source {key[0]}/{key[1]} has unmapped label "
                f"{source['source_label']!r}"
            )
        reviewed_claim = clean_text(review.get("claim"))
        if source["source_dataset"] == "healthfc":
            if reviewed_claim and reviewed_claim != source["source_claim"]:
                raise ValueError(
                    f"eligible source {key[0]}/{key[1]} attempts to rewrite the "
                    "HealthFC en_claim; HealthFC wording must remain verbatim"
                )
            claim = source["source_claim"]
            claim_form = "question"
        else:
            claim = reviewed_claim or source["source_claim"]
            claim_form = "statement"
        if not claim:
            raise ValueError(f"eligible source {key[0]}/{key[1]} has no claim text")
        stable_key = f"{source['source_dataset']}\0{source['source_id']}"
        claim_id = "ff_" + hashlib.sha256(stable_key.encode("utf-8")).hexdigest()[:12]
        rank_text = f"{seed}\0{stable_key}"
        rank = hashlib.sha256(rank_text.encode("utf-8")).hexdigest()
        candidate = dict(source)
        candidate.update(review)
        candidate.update(
            {
                "id": claim_id,
                "claim": claim,
                "claim_form": claim_form,
                "source_tier": source_tier,
                "tier": review["verified_tier"],
                "rank": rank,
                "topic_key": normalize_label(review["topic"]),
            }
        )
        candidates.append(candidate)
    missing_sources = set(adjudications) - reviewed_keys
    if missing_sources:
        dataset, source_id = sorted(missing_sources)[0]
        raise ValueError(f"adjudication has no matching source: {dataset}/{source_id}")
    counts = {
        "source_rows": len(sources),
        "unreviewed_source_rows": len(sources) - len(reviewed_keys),
        "eligible_candidates": len(candidates),
        "unknown_source_labels": [
            {"source_dataset": dataset, "source_label": label, "count": count}
            for (dataset, label), count in sorted(unknown_labels.items())
        ],
    }
    return candidates, counts


def group_key(candidate: dict) -> str:
    return f"{candidate['tier']}|{candidate['source_dataset']}"


def can_fill(candidates: list[dict], quotas: dict[str, int], topic_caps: dict[str, int]) -> bool:
    needed = sum(quotas.values())
    if needed == 0:
        return True
    grouped = Counter(
        (group_key(candidate), candidate["topic_key"])
        for candidate in candidates
        if quotas.get(group_key(candidate), 0) > 0
        and topic_caps.get(candidate["topic_key"], 0) > 0
    )
    source = "__source__"
    sink = "__sink__"
    capacity: dict[str, dict[str, int]] = {}

    def add_edge(left: str, right: str, value: int) -> None:
        capacity.setdefault(left, {})[right] = value
        capacity.setdefault(right, {}).setdefault(left, 0)

    for group, quota in quotas.items():
        if quota > 0:
            add_edge(source, f"g:{group}", quota)
    for (group, topic), count in grouped.items():
        add_edge(f"g:{group}", f"t:{topic}", count)
    for topic, limit in topic_caps.items():
        if limit > 0:
            add_edge(f"t:{topic}", sink, limit)

    flow = 0
    while True:
        parent = {source: ""}
        queue = [source]
        for node in queue:
            for neighbor in sorted(capacity.get(node, {})):
                if neighbor not in parent and capacity[node][neighbor] > 0:
                    parent[neighbor] = node
                    queue.append(neighbor)
                    if neighbor == sink:
                        break
            if sink in parent:
                break
        if sink not in parent:
            break
        amount = needed - flow
        node = sink
        while node != source:
            previous = parent[node]
            amount = min(amount, capacity[previous][node])
            node = previous
        node = sink
        while node != source:
            previous = parent[node]
            capacity[previous][node] -= amount
            capacity[node][previous] += amount
            node = previous
        flow += amount
        if flow == needed:
            return True
    return False


def select_candidates(candidates: list[dict], topic_cap: int) -> list[dict]:
    if topic_cap < 1:
        raise ValueError("topic_cap must be positive")
    quotas = {
        f"{tier}|{dataset}": DATASET_QUOTA for tier in TIERS for dataset in DATASETS
    }
    topics = {candidate["topic_key"] for candidate in candidates}
    topic_caps = {topic: topic_cap for topic in topics}
    ordered = sorted(candidates, key=lambda item: (item["rank"], item["id"]))
    if not can_fill(ordered, quotas, topic_caps):
        available = Counter(group_key(candidate) for candidate in ordered)
        summary = ", ".join(f"{key}={available.get(key, 0)}" for key in sorted(quotas))
        raise ValueError(
            f"eligible candidates cannot satisfy balanced quotas and topic cap {topic_cap}: {summary}"
        )

    selected = []
    for index, candidate in enumerate(ordered):
        group = group_key(candidate)
        topic = candidate["topic_key"]
        if quotas[group] == 0 or topic_caps[topic] == 0:
            continue
        next_quotas = dict(quotas)
        next_topics = dict(topic_caps)
        next_quotas[group] -= 1
        next_topics[topic] -= 1
        if can_fill(ordered[index + 1 :], next_quotas, next_topics):
            selected.append(candidate)
            quotas = next_quotas
            topic_caps = next_topics
        if not sum(quotas.values()):
            break
    if len(selected) != len(TIERS) * len(DATASETS) * DATASET_QUOTA:
        raise RuntimeError("selection ended before all quotas were filled")
    return selected


def public_claim(candidate: dict, metadata: dict) -> dict:
    value = {
        "id": candidate["id"],
        "claim": candidate["claim"],
        "claim_form": candidate["claim_form"],
        "tier": candidate["tier"],
        "topic": candidate["topic"],
        "source_dataset": candidate["source_dataset"],
        "source_id": candidate["source_id"],
        "source_label": candidate["source_label"],
        "source_tier": candidate["source_tier"],
        "source_url": candidate["source_url"],
        "source_url_scope": candidate["source_url_scope"],
        "source_record_sha256": candidate["source_record_sha256"],
        "source_license": clean_text(metadata["license"]),
        "screen_1_tier": candidate["screen_1_tier"],
        "screen_2_tier": candidate["screen_2_tier"],
        "screening_method": candidate["screening_method"],
        "verification_url": candidate["verification_url"],
        "adjudication_record_sha256": candidate["adjudication_record_sha256"],
    }
    if metadata.get("license_url"):
        value["source_license_url"] = metadata["license_url"]
    for field in ("verification_title", "verification_date", "adjudication_note"):
        if candidate.get(field):
            value[field] = candidate[field]
    return value


def build_claim_set(
    pubhealth_path: Path,
    healthfc_path: Path,
    adjudication_path: Path,
    metadata_path: Path,
    output_path: Path = CLAIMS_PATH,
    manifest_path: Path = DATA / "claims" / "selection_manifest.json",
    seed: int = DEFAULT_SEED,
    topic_cap: int = DEFAULT_TOPIC_CAP,
) -> tuple[list[dict], dict]:
    metadata = load_metadata(metadata_path)
    source_paths = {"pubhealth": pubhealth_path, "healthfc": healthfc_path}
    source_rows = []
    input_manifest = {}
    for dataset in DATASETS:
        records, row_count = load_source(source_paths[dataset], dataset, metadata[dataset])
        source_rows.extend(records)
        input_manifest[dataset] = {
            "file": source_paths[dataset].name,
            "sha256": sha256_file(source_paths[dataset]),
            "rows": row_count,
            "metadata": metadata[dataset],
        }
    adjudications, review_counts = load_adjudications(adjudication_path)
    candidates, candidate_counts = build_candidates(source_rows, adjudications, metadata, seed)
    selected = select_candidates(candidates, topic_cap)
    tier_order = {tier: index for index, tier in enumerate(TIERS)}
    selected.sort(key=lambda item: (tier_order[item["tier"]], item["rank"], item["id"]))
    claims = [public_claim(item, metadata[item["source_dataset"]]) for item in selected]
    claim_text = pretty_json(claims)
    claim_hash = hashlib.sha256(claim_text.encode("utf-8")).hexdigest()
    selection_counts = Counter((item["tier"], item["source_dataset"]) for item in selected)
    topic_counts = Counter(item["topic"] for item in selected)
    manifest = {
        "schema_version": 2,
        "seed": seed,
        "topic_cap": topic_cap,
        "dataset_quota_per_tier": DATASET_QUOTA,
        "inputs": {
            **input_manifest,
            "adjudication": {
                "file": adjudication_path.name,
                "sha256": sha256_file(adjudication_path),
                **review_counts,
            },
            "source_metadata": {
                "file": metadata_path.name,
                "sha256": sha256_file(metadata_path),
            },
        },
        "screening": candidate_counts,
        "selection": {
            "total": len(selected),
            "by_tier_and_dataset": {
                f"{tier}|{dataset}": selection_counts[(tier, dataset)]
                for tier in TIERS
                for dataset in DATASETS
            },
            "by_topic": dict(sorted(topic_counts.items())),
            "rows": [
                {
                    "id": item["id"],
                    "source_dataset": item["source_dataset"],
                    "source_id": item["source_id"],
                    "tier": item["tier"],
                    "topic": item["topic"],
                    "rank": item["rank"],
                }
                for item in selected
            ],
        },
        "output": {"file": output_path.name, "sha256": claim_hash, "rows": len(claims)},
    }
    atomic_write(output_path, claim_text)
    atomic_write(manifest_path, pretty_json(manifest))
    return claims, manifest


def build_main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the reviewed, deterministic FramedFacts-48 claim set."
    )
    parser.add_argument("--pubhealth", required=True, type=Path)
    parser.add_argument("--healthfc", required=True, type=Path)
    parser.add_argument("--adjudication", required=True, type=Path)
    parser.add_argument("--source-metadata", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=CLAIMS_PATH)
    parser.add_argument(
        "--manifest", type=Path, default=DATA / "claims" / "selection_manifest.json"
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--topic-cap", type=int, default=DEFAULT_TOPIC_CAP)
    args = parser.parse_args()
    claims, _ = build_claim_set(
        args.pubhealth,
        args.healthfc,
        args.adjudication,
        args.source_metadata,
        args.output,
        args.manifest,
        args.seed,
        args.topic_cap,
    )
    print(f"wrote {len(claims)} reviewed claims to {args.output}")


EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
DATE_FILTER = '("2020/01/01"[Date - Publication] : "2026/12/31"[Date - Publication])'


def clean(value: object) -> str:
    return " ".join(str(value or "").split())


def read_candidates(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    required = {"source_dataset", "source_id", "verification_query"}
    for index, row in enumerate(rows, 2):
        if not required.issubset(row) or not all(clean(row[field]) for field in required):
            raise ValueError(f"{path.name}:{index}: missing required candidate field")
    return rows


def request(client: httpx.Client, endpoint: str, params: dict) -> httpx.Response:
    response = client.get(f"{EUTILS}/{endpoint}", params=params)
    response.raise_for_status()
    return response


def article_year(article: ET.Element) -> str:
    for path in (
        ".//ArticleDate/Year",
        ".//JournalIssue/PubDate/Year",
        ".//PubMedPubDate[@PubStatus='pubmed']/Year",
        ".//PubMedPubDate[@PubStatus='entrez']/Year",
    ):
        value = article.findtext(path)
        if value:
            return clean(value)
    medline_date = article.findtext(".//JournalIssue/PubDate/MedlineDate") or ""
    return clean(medline_date)[:4]


def article_text(node: ET.Element | None) -> str:
    if node is None:
        return ""
    return clean("".join(node.itertext()))


def parse_articles(xml_text: str) -> list[dict]:
    root = ET.fromstring(xml_text)
    articles = []
    for record in root.findall(".//PubmedArticle"):
        pmid = clean(record.findtext(".//PMID"))
        title = article_text(record.find(".//ArticleTitle"))
        abstract = " ".join(
            article_text(item) for item in record.findall(".//Abstract/AbstractText")
        ).strip()
        publication_types = [
            article_text(item) for item in record.findall(".//PublicationType")
        ]
        doi = ""
        for item in record.findall(".//ArticleId"):
            if item.attrib.get("IdType") == "doi":
                doi = article_text(item)
                break
        articles.append(
            {
                "pmid": pmid,
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                "title": title,
                "year": article_year(record),
                "doi": doi,
                "publication_types": publication_types,
                "abstract": abstract,
            }
        )
    return articles


def verify_one(client: httpx.Client, row: dict, raw_dir: Path, api_key: str) -> dict:
    source_dataset = clean(row["source_dataset"]).casefold()
    source_id = clean(row["source_id"])
    raw_path = raw_dir / f"{source_dataset}_{source_id}.json"
    if raw_path.exists():
        with raw_path.open(encoding="utf-8") as stream:
            raw = json.load(stream)
        if raw.get("query") != clean(row["verification_query"]):
            raise RuntimeError(f"cached verification query changed for {source_id}")
        return {
            "source_dataset": row["source_dataset"],
            "source_id": source_id,
            "query": raw["query"],
            "articles": parse_articles(raw["fetch_response"]),
        }

    query = clean(row["verification_query"])
    term = f"({query}) AND {DATE_FILTER}"
    common = {"tool": "sycophantic-search"}
    if api_key:
        common["api_key"] = api_key
    search_params = {
        **common,
        "db": "pubmed",
        "term": term,
        "retmode": "json",
        "retmax": 8,
        "sort": "relevance",
    }
    searched_at = now_iso()
    search_response = request(client, "esearch.fcgi", search_params)
    search_body = search_response.json()
    ids = search_body.get("esearchresult", {}).get("idlist", [])
    if not isinstance(ids, list):
        raise RuntimeError(f"PubMed returned an invalid ID list for {source_id}")
    time.sleep(0.12 if api_key else 0.36)

    fetch_params = {**common, "db": "pubmed", "id": ",".join(ids), "retmode": "xml"}
    fetch_text = "<PubmedArticleSet />"
    if ids:
        fetch_text = request(client, "efetch.fcgi", fetch_params).text
    raw = {
        "schema_version": 1,
        "source_dataset": row["source_dataset"],
        "source_id": source_id,
        "query": query,
        "searched_at": searched_at,
        "search_request": {key: value for key, value in search_params.items() if key != "api_key"},
        "search_response": search_body,
        "fetch_request": {key: value for key, value in fetch_params.items() if key != "api_key"},
        "fetch_response": fetch_text,
    }
    atomic_write(raw_path, pretty_json(raw))
    time.sleep(0.12 if api_key else 0.36)
    return {
        "source_dataset": row["source_dataset"],
        "source_id": source_id,
        "query": query,
        "articles": parse_articles(fetch_text),
    }


def web_queries(row: dict) -> list[str]:
    query = clean(row["verification_query"])
    return [
        f"{query} systematic review meta-analysis",
        f"{query} site:who.int OR site:cdc.gov OR site:fda.gov OR site:nih.gov OR site:cochrane.org",
    ]


def verify_web_one(row: dict, raw_dir: Path) -> dict:
    source_dataset = clean(row["source_dataset"]).casefold()
    source_id = clean(row["source_id"])
    raw_path = raw_dir / f"{source_dataset}_{source_id}.json"
    queries = web_queries(row)
    if raw_path.exists():
        with raw_path.open(encoding="utf-8") as stream:
            raw = json.load(stream)
        if raw.get("queries") != queries:
            raise RuntimeError(f"cached web verification queries changed for {source_id}")
        return raw

    searches = []
    for query in queries:
        last_error = ""
        results = []
        for attempt in range(3):
            try:
                results = list(DDGS().text(query, region="us-en", max_results=8))
                last_error = ""
                break
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                time.sleep(2**attempt)
        searches.append({"query": query, "results": results, "error": last_error})
        time.sleep(1.5)
    raw = {
        "schema_version": 1,
        "source_dataset": row["source_dataset"],
        "source_id": source_id,
        "searched_at": now_iso(),
        "queries": queries,
        "searches": searches,
    }
    atomic_write(raw_path, pretty_json(raw))
    return raw


def verify_main() -> None:
    parser = argparse.ArgumentParser(
        description="Retrieve recent PubMed candidates for claim-verdict verification."
    )
    parser.add_argument(
        "--candidates", type=Path, default=WORKING / "candidate_review_a.csv"
    )
    parser.add_argument(
        "--output", type=Path, default=WORKING / "pubmed_verification_candidates.json"
    )
    parser.add_argument(
        "--raw-dir", type=Path, default=ARCHIVE / "claim_verification" / "pubmed"
    )
    parser.add_argument("--with-web", action="store_true")
    parser.add_argument(
        "--web-output", type=Path, default=WORKING / "web_verification_candidates.json"
    )
    parser.add_argument(
        "--web-raw-dir", type=Path, default=ARCHIVE / "claim_verification" / "web"
    )
    args = parser.parse_args()

    load_dotenv(ROOT / ".env", override=False)
    api_key = os.getenv("NCBI_API_KEY", "").strip()
    rows = read_candidates(args.candidates)
    args.raw_dir.mkdir(parents=True, exist_ok=True)
    results = []
    with httpx.Client(timeout=45, follow_redirects=True) as client:
        for index, row in enumerate(rows, 1):
            results.append(verify_one(client, row, args.raw_dir, api_key))
            print(f"verified search {index}/{len(rows)}: {row['source_dataset']}/{row['source_id']}")
    atomic_write(
        args.output,
        pretty_json(
            {
                "schema_version": 1,
                "retrieved_at": now_iso(),
                "date_filter": DATE_FILTER,
                "rows": results,
            }
        ),
    )
    print(f"wrote {len(results)} candidate evidence records to {args.output}")

    if args.with_web:
        args.web_raw_dir.mkdir(parents=True, exist_ok=True)
        web_results = []
        for index, row in enumerate(rows, 1):
            web_results.append(verify_web_one(row, args.web_raw_dir))
            print(f"verified web search {index}/{len(rows)}: {row['source_dataset']}/{row['source_id']}")
        atomic_write(
            args.web_output,
            pretty_json(
                {
                    "schema_version": 1,
                    "retrieved_at": now_iso(),
                    "rows": web_results,
                }
            ),
        )
        print(f"wrote {len(web_results)} web evidence records to {args.web_output}")


def main() -> None:
    commands = {"build": build_main, "verify": verify_main}
    if len(sys.argv) < 2 or sys.argv[1] not in commands:
        names = "|".join(sorted(commands))
        raise SystemExit(f"usage: claims.py {{{names}}} [options]")
    commands[sys.argv.pop(1)]()


if __name__ == "__main__":
    main()
