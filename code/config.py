from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
ARCHIVE = ROOT / "archive"
WORKING = ROOT / "working"

CLAIMS_PATH = DATA / "claims" / "framedfacts48.json"
PROMPTS_PATH = DATA / "prompts.json"
MODEL_PARAMS_PATH = DATA / "model_params.json"
MODEL_PARAMS_V5_PATH = DATA / "model_params_v5.json"
SCHEDULE_PATH = DATA / "schedule.json"
RUNS_DIR = DATA / "runs"
RAW_DIR = ARCHIVE / "raw"
LEDGER_PATH = DATA / "ledger.csv"
SEARCH_CACHE_PATH = ARCHIVE / "search_cache.sqlite3"

SYSTEMS = {
    "sonnet5": "anthropic/claude-sonnet-5",
    "terra": "openai/gpt-5.6-terra",
    "gemini": "google/gemini-3.6-flash",
    "glm": "z-ai/glm-5.2",
}

JUDGES = {
    "gemini": "google/gemini-3.6-flash",
    "luna": "openai/gpt-5.6-luna",
}

KIMI_MODEL = "moonshotai/kimi-k3:exacto"
MODEL_ALIASES = {slug: alias for alias, slug in {**SYSTEMS, **JUDGES}.items()}
MODEL_ALIASES[KIMI_MODEL] = "kimi"

SEARCH_LIMIT = 8
SEARCH_RESULT_COUNT = 8
SNIPPET_LIMIT = 350
MODEL_CONCURRENCY = 6
OPENROUTER_LIMIT_WITHOUT_KIMI = 185.0
OPENROUTER_LIMIT_WITH_KIMI = 192.0


def load_environment() -> None:
    load_dotenv(ROOT / ".env", override=False)
    if not os.getenv("OPENROUTER_API_KEY"):
        raise RuntimeError("OPENROUTER_API_KEY is not set")


def load_json(path: Path) -> object:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def load_claims() -> list[dict]:
    claims = load_json(CLAIMS_PATH)
    if not isinstance(claims, list):
        raise TypeError("claim file must contain a JSON list")
    if len(claims) != 48:
        raise ValueError(f"expected 48 claims, found {len(claims)}")
    ids = [claim.get("id") for claim in claims]
    if len(ids) != len(set(ids)):
        raise ValueError("claim IDs are not unique")
    tiers = {tier: 0 for tier in ("settled_false", "settled_true", "contested")}
    for claim in claims:
        tier = claim.get("tier")
        if tier not in tiers:
            raise ValueError(f"invalid claim tier: {tier}")
        tiers[tier] += 1
        for field in ("claim", "source_dataset", "source_id", "source_label", "verification_url"):
            if not claim.get(field):
                raise ValueError(f"{claim.get('id')}: missing {field}")
    if set(tiers.values()) != {16}:
        raise ValueError(f"claim tiers are unbalanced: {tiers}")
    return claims


def params_hash(body: dict) -> str:
    encoded = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def model_params(model: str, judge: bool = False, protocol: str = "legacy") -> dict:
    if protocol not in {"legacy", "v5"}:
        raise ValueError(f"unknown model-parameter protocol: {protocol!r}")
    path = MODEL_PARAMS_V5_PATH if protocol == "v5" else MODEL_PARAMS_PATH
    values = load_json(path)
    if not isinstance(values, dict):
        raise TypeError("model parameter file must contain a JSON object")
    section = "judges" if judge else "agents"
    params = values.get(section, {}).get(model)
    if not isinstance(params, dict):
        raise KeyError(f"no {section} parameters for {model}")
    return json.loads(json.dumps(params))


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def pretty_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def append_jsonl(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = canonical_json(value) + "\n"
    with path.open("a", encoding="utf-8", newline="") as stream:
        stream.write(line)
        stream.flush()
        os.fsync(stream.fileno())


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.endswith("\n"):
                raise ValueError(f"{path}:{line_number}: partial JSONL line")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"{path}:{line_number}: expected JSON object")
            rows.append(value)
    return rows
