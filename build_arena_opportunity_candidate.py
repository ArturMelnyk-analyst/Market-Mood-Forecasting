
from pathlib import Path
import csv
import json
import hashlib

ROOT = Path(__file__).resolve().parent
LIVE = ROOT / "data/processed/es_arena_live"
RESEARCH = ROOT / "data/processed/es_arena_research_v0_1"

OUT = RESEARCH / "arena_opportunity_candidate_20261112.csv"

def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))

rows = []

for forecast_path in sorted((LIVE / "forecasts").glob("forecast_*.json")):
    forecast = load_json(forecast_path)

    challenge_id = forecast["challenge_id"]
    target_date = forecast["target_session_date"]

    attempt_path = (
        LIVE / "receipts" / f"attempt_{challenge_id}.json"
    )
    receipt_path = (
        LIVE / "receipts" / f"receipt_{challenge_id}.json"
    )

    assert attempt_path.is_file()
    assert receipt_path.is_file()

    attempt = load_json(attempt_path)
    receipt = load_json(receipt_path)

    assert attempt["challenge_id"] == challenge_id
    assert receipt["response"]["challenge_id"] == challenge_id

    assert sha256(forecast_path) == attempt["forecast_sha256"]
    assert sha256(forecast_path) == receipt["forecast_sha256"]

    accepted = receipt["response"].get("counts_for_score") is True

    rows.append({
        "settlement_date": target_date,
        "challenge_id": challenge_id,
        "opportunity_status": (
            "confirmed_accepted" if accepted else "review_required"
        ),
        "evidence_type": "native_forecast_attempt_receipt",
        "forecast_path": str(forecast_path.relative_to(ROOT)),
        "attempt_path": str(attempt_path.relative_to(ROOT)),
        "receipt_path": str(receipt_path.relative_to(ROOT)),
        "receipt_sha256": sha256(receipt_path),
        "opportunity_completeness": "not_independently_verified",
        "notes": "Local record; independent UTC server timing not certified",
    })

legacy_dir = RESEARCH / "legacy_submissions"

for path in sorted(legacy_dir.glob("legacy_es_*.json")):
    legacy = load_json(path)

    rows.append({
        "settlement_date": legacy["settlement_date"],
        "challenge_id": legacy["challenge_id"],
        "opportunity_status": (
            "confirmed_accepted"
            if legacy.get("server_response", {}).get("counts_for_score") is True
            else "review_required"
        ),
        "evidence_type": "recovered_legacy_submission",
        "forecast_path": str(path.relative_to(ROOT)),
        "attempt_path": "",
        "receipt_path": "",
        "receipt_sha256": "",
        "opportunity_completeness": "not_independently_verified",
        "notes": "Legacy provenance limitation preserved",
    })

rows.sort(key=lambda x: (x["settlement_date"], x["challenge_id"]))

assert len(rows) == 18, "Unexpected candidate row count"
assert len({r["challenge_id"] for r in rows}) == len(rows)
assert all(r["opportunity_status"] == "confirmed_accepted" for r in rows)

if OUT.exists():
    raise FileExistsError(f"Candidate already exists: {OUT}")

with OUT.open("x", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

print("CANDIDATE CREATED:", OUT)
print("ROWS:", len(rows))
print("FIRST DATE:", rows[0]["settlement_date"])
print("LAST DATE:", rows[-1]["settlement_date"])
print("OUTPUT SHA256:", sha256(OUT))
print("ORIGINAL FORECASTS AND RECEIPTS UNCHANGED")
