
"""Write-once Arena challenge discovery logger.

One authenticated active-challenge GET per capture.
No prediction submission, model inference, or retraining.
"""

import argparse
import hashlib
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import requests

from es_arena_daily import BASE, PROJECT_ROOT, authenticate


ARCHIVE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "es_arena_live"
    / "discovery"
)

ENDPOINT = "/api/v1/eval/challenges/active"


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def normalize_challenges(body):
    if isinstance(body, list):
        entries = body
    elif isinstance(body, dict):
        entries = body.get("challenges", body.get("items"))
    else:
        raise ValueError("Unsupported challenge feed type")

    if not isinstance(entries, list):
        raise ValueError("Missing challenge list")

    records = []
    seen = set()

    for item in entries:
        if not isinstance(item, dict):
            raise ValueError("Non-object challenge entry")

        challenge = item.get("challenge", item)

        if not isinstance(challenge, dict):
            raise ValueError("Invalid nested challenge")

        cid = challenge.get("id", challenge.get("challenge_id"))

        if not isinstance(cid, str) or not cid:
            raise ValueError("Challenge ID missing")

        if cid in seen:
            raise ValueError("Duplicate challenge ID")

        seen.add(cid)
        records.append(challenge)

    return records


class ESMetadataError(ValueError):
    """ES opportunity exists, but required metadata is incomplete."""


def validate_es_metadata(challenges):
    """Validate research-critical fields for discovered ES daily challenges."""
    for challenge in challenges:
        cid = challenge.get("id") or challenge.get("challenge_id")

        if not isinstance(cid, str) or not cid.strip():
            raise ESMetadataError("ES challenge has no valid ID")

        contract = challenge.get("contract_symbol")
        if not isinstance(contract, str) or not contract.strip():
            raise ESMetadataError(
                f"ES challenge {cid}: missing contract symbol"
            )

        deadline_raw = (
            challenge.get("deadline_utc")
            or challenge.get("deadline_at")
            or challenge.get("deadline")
        )

        if not isinstance(deadline_raw, str) or not deadline_raw.strip():
            raise ESMetadataError(
                f"ES challenge {cid}: missing deadline"
            )

        try:
            deadline = datetime.fromisoformat(
                deadline_raw.strip().replace("Z", "+00:00")
            )
        except ValueError as exc:
            raise ESMetadataError(
                f"ES challenge {cid}: invalid deadline"
            ) from exc

        if deadline.tzinfo is None or deadline.utcoffset() is None:
            raise ESMetadataError(
                f"ES challenge {cid}: timezone missing"
            )

        created_raw = challenge.get("created_at")
        if created_raw is not None:
            try:
                created = datetime.fromisoformat(
                    created_raw.strip().replace("Z", "+00:00")
                )
            except (AttributeError, ValueError) as exc:
                raise ESMetadataError(
                    f"ES challenge {cid}: invalid creation time"
                ) from exc

            if created.tzinfo is None or created.utcoffset() is None:
                raise ESMetadataError(
                    f"ES challenge {cid}: creation timezone missing"
                )

            if deadline <= created:
                raise ESMetadataError(
                    f"ES challenge {cid}: deadline not after creation"
                )

def capture():
    started = utc_now()

    record = {
        "schema_version": "arena_discovery_v0_1",
        "request_started_at_utc": started,
        "endpoint": ENDPOINT,
        "source": "authenticated_active_challenge_feed",
        "status": "not_started",
        "http_status": None,
        "server_date_header": None,
        "response_sha256": None,
        "response_json": None,
        "es_daily_challenges": [],
        "error_type": None,
    }

    try:
        # Authentication may make its own API request.
        headers = authenticate()

        response = requests.get(
            BASE + ENDPOINT,
            headers=headers,
            timeout=12,
        )

        record["http_status"] = response.status_code
        record["server_date_header"] = response.headers.get("Date")
        record["response_sha256"] = sha256_bytes(response.content)

        response.raise_for_status()

        body = response.json()

        # Preserve the successfully decoded response even if
        # challenge metadata subsequently fails validation.
        record["response_json"] = body

        challenges = normalize_challenges(body)

        es = [
            c for c in challenges
            if str(c.get("asset", c.get("scope_key", ""))).upper() == "ES"
            and c.get("challenge_type") == "daily"
        ]

        validate_es_metadata(es)

        record["status"] = "success"

        record["es_daily_challenges"] = [
            {
                "challenge_id": c.get("id", c.get("challenge_id")),
                "asset": c.get("asset"),
                "contract_symbol": c.get("contract_symbol"),
                "status": c.get("status"),
                "created_at": c.get("created_at"),
                "deadline": c.get("deadline"),
                "session_date": c.get(
                    "session_date", c.get("settlement_date")
                ),
            }
            for c in es
        ]

        record["total_challenges"] = len(challenges)

    except Exception as exc:
        record["status"] = (
            "incomplete"
            if isinstance(exc, ESMetadataError)
            else "failed"
        )
        record["error_type"] = type(exc).__name__

    record["record_completed_at_utc"] = utc_now()

    ARCHIVE.mkdir(parents=True, exist_ok=True)

    filename = (
        "arena_discovery_"
        + started.replace("-", "").replace(":", "").replace("+", "_")
        + "_"
        + uuid.uuid4().hex[:12]
        + ".json"
    )

    output = ARCHIVE / filename

    content = (
        json.dumps(
            record,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")

    # Write once; never overwrite an existing archive record.
    with output.open("xb") as stream:
        stream.write(content)

    print(
        "DISCOVERY RECORD:",
        output.relative_to(PROJECT_ROOT)
        if output.is_relative_to(PROJECT_ROOT)
        else output,
    )
    print("RECORD SHA256:", sha256_bytes(content))
    print("STATUS:", record["status"])
    print("HTTP STATUS:", record["http_status"])
    print("REQUEST STARTED UTC:", started)
    print(
        "RECORD COMPLETED UTC:",
        record["record_completed_at_utc"],
    )
    print("ES DAILY CHALLENGES:", len(record["es_daily_challenges"]))

    for challenge in record["es_daily_challenges"]:
        print(
            "ES:",
            challenge["challenge_id"],
            "DEADLINE:",
            challenge["deadline"],
        )

    if record["status"] != "success":
        print("DISCOVERY FAILURE:", record["error_type"])
        raise SystemExit(1)

    print("PREDICTION SUBMISSIONS: 0")
    print("ORIGINAL FORECAST FILES MODIFIED: 0")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        choices=["capture"],
    )
    args = parser.parse_args()

    if args.mode == "capture":
        capture()


if __name__ == "__main__":
    main()
