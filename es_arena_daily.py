"""Manual, fail-closed ES Headline Arena daily forecaster and one-shot submitter.

Does not download/buy Databento data, modify Notebook 08/09/10/11,
fit/retrain a model, or expose credentials in output.

Run from repository root with active .venv:
    python es_arena_daily.py list
    python es_arena_daily.py explain --forecast <EXISTING_FORECAST_JSON>
    python es_arena_daily.py prepare --challenge-id <ID> --date YYYY-MM-DD \
       --last-session YYYY-MM-DD --contract ESZ6 --sessions <CSV_PATH>
    python es_arena_daily.py submit --forecast <OUTPUT_JSON> \
       --confirm-id <ID> --approve I_APPROVE_ONE_SCORED_SUBMISSION
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from datetime import timedelta
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

BASE = "https://headlinearena.com"
PROJECT_ROOT = Path(__file__).resolve().parent
PRIVATE_DIR = PROJECT_ROOT / "data" / "processed" / "es_arena_live"
MODEL_ROOT = PROJECT_ROOT / "data" / "processed" / "es_arena_nb10_v0_1"
CLASSES = ("bearish", "neutral", "bullish")
FEATURES = (
    "es_return_lag1", "es_return_lag2", "es_return_lag3", "es_return_lag5",
    "es_range_pct_lag1", "es_log_volume_lag1", "es_return_mean_5",
    "es_return_mean_20", "es_vol_5", "es_vol_20",
    "es_abs_return_mean_5", "es_abs_return_mean_20",
)
REQUIRED_COLUMNS = (
    "session_date", "session_start_utc", "session_end_utc", "data_available_utc",
    "open", "high", "low", "close", "volume", "session_return", "raw_symbol",
    "target_eligible", "first_bar_on_expected_boundary", "last_bar_on_expected_boundary",
    "multiple_underlying_symbols", "coverage_reviewed", "source_dataset", "source_schema",
)
ET = ZoneInfo("America/New_York")


def fail(message):
    raise RuntimeError(message)


def require(condition, message):
    if not condition:
        fail(message)


def checksum(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for piece in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(piece)
    return h.hexdigest()


def as_utc(value, label):
    stamp = pd.Timestamp(value)
    require(stamp.tzinfo is not None, f"{label}: missing timezone/offset")
    return stamp.tz_convert("UTC")


def safe_path(path):
    resolved = Path(path).resolve()
    require(resolved.is_relative_to(PROJECT_ROOT.resolve()),
            f"All files must remain inside project: {path}")
    return resolved


def write_private_once(path, obj):
    path = safe_path(path)
    require(path.is_relative_to(PRIVATE_DIR.resolve()), "Output must live in private data/processed")
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    with path.open("xb") as f:
        f.write(data)
    return path


def env_credentials():
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    def first(*keys):
        return next((os.environ[k] for k in keys if os.environ.get(k)), None)
    agent = first("HEADLINE_ARENA_AGENT_ID", "HA_AGENT_ID", "ARENA_AGENT_ID", "AGENT_ID")
    secret = first("HEADLINE_ARENA_CLIENT_SECRET", "HA_CLIENT_SECRET", "ARENA_CLIENT_SECRET", "CLIENT_SECRET")
    require(agent and secret, "Credentials not found. Check agent_id and client_secret key names in private .env (never share values).")
    return agent, secret


def authenticate():
    agent, secret = env_credentials()
    response = requests.post(BASE + "/api/v1/agent/auth/token",
        json={"grant_type": "client_credentials", "agent_id": agent, "client_secret": secret}, timeout=12)
    response.raise_for_status()
    return {"Authorization": "Bearer " + response.json()["access_token"]}


def live_challenges(headers):
    response = requests.get(BASE + "/api/v1/eval/challenges/active", headers=headers, timeout=12)
    response.raise_for_status()
    body = response.json()
    entries = body if isinstance(body, list) else body.get("challenges", body.get("items", []))
    require(isinstance(entries, list), "Unexpected challenge feed format")
    return [entry.get("challenge", entry) for entry in entries]


def fetch_one(headers, cid):
    items = live_challenges(headers)
    matched = [c for c in items if str(c.get("id", c.get("challenge_id", ""))) == cid]
    require(len(matched) == 1, f"Challenge {cid} not uniquely active; no submission")
    item = matched[0]
    actual_scope = item.get("scope_key") or item.get("instrument_root") or item.get("asset")
    require(actual_scope is None or str(actual_scope).upper() == "ES",
            f"Challenge {cid} scope conflicts with ES: {actual_scope}")
    return item


def deadline_for(challenge):
    raw = next((challenge[k] for k in ("deadline_utc", "deadline_at", "deadline")
                if challenge.get(k)), None)
    require(raw is not None, "Active challenge lacks a verifiable deadline; stop and review API/UI")
    return as_utc(raw, "challenge.deadline")


def parse_bool(series, label):
    toks = series.astype(str).str.strip().str.lower()
    require(toks.isin(("true", "false", "1", "0")).all(), f"Invalid {label}")
    return toks.isin(("true", "1"))


def session_boundary(day, hour, prior=False):
    d = pd.Timestamp(day).date() - (timedelta(days=1) if prior else timedelta(0))
    return pd.Timestamp(f"{d.isoformat()} {hour:02d}:00:00", tz=ET).tz_convert("UTC")


def validate_sessions(path, target, require_last, cutoff):
    path = safe_path(path)
    require(path.is_file(), f"Session CSV missing: {path}")
    require("/es_arena_nb11_v0_1/snapshots/" in path.as_posix(),
            "Use the latest immutable Notebook 11 snapshot, not the old canonical CSV")
    mf = path.with_suffix(".manifest.json")
    require(mf.is_file(), f"Missing source manifest: {mf}")
    manifest = json.loads(mf.read_text(encoding="utf-8"))
    source_hash = checksum(path)
    require(source_hash == manifest.get("snapshot_sha256"), "NB11 snapshot hash mismatch")
    expected_rel = path.relative_to(PROJECT_ROOT).as_posix()
    require(manifest.get("snapshot_path") == expected_rel, "NB11 manifest path mismatch")
    df = pd.read_csv(path)
    require(tuple(df.columns) == REQUIRED_COLUMNS, "NB11 CSV schema or column order changed")
    require(len(df) >= 20, "Too few sessions")
    x = df.copy()
    x["session_date"] = pd.to_datetime(x.session_date, errors="raise")
    require(x.session_date.is_unique and x.session_date.is_monotonic_increasing,
            "Session dates duplicate/out of order")
    require((x.session_date < pd.Timestamp(target)).all(), "Same-day/future session leakage")
    require(x.session_date.dt.normalize().eq(x.session_date).all(), "Session dates contain times")
    for key in ("target_eligible", "first_bar_on_expected_boundary", "last_bar_on_expected_boundary",
                "multiple_underlying_symbols", "coverage_reviewed"):
        x[key] = parse_bool(x[key], key)
    require(x.coverage_reviewed.all(), "Session coverage reviews incomplete")
    require((~x.target_eligible | (x.first_bar_on_expected_boundary &
            x.last_bar_on_expected_boundary & ~x.multiple_underlying_symbols)).all(),
            "Eligible session violates source structural requirements")
    require(x.source_dataset.astype(str).eq("GLBX.MDP3").all() and
            x.source_schema.astype(str).eq("ohlcv-1m").all(), "Unexpected source provenance")
    e = x.loc[x.target_eligible].copy()
    require(len(e) >= 20, "Need at least 20 eligible sessions")
    require(e.session_date.max() == pd.Timestamp(require_last),
            f"Expected latest reviewed eligible source {require_last}, got {e.session_date.max()}")
    require(0 < (pd.Timestamp(target) - e.session_date.max()).days <= 7,
            "Latest eligible session is stale or same-day")
    for field in ("open", "high", "low", "close", "volume", "session_return"):
        e[field] = pd.to_numeric(e[field], errors="raise")
    prices = e[["open", "high", "low", "close", "volume", "session_return"]].to_numpy(float)
    require(np.isfinite(prices).all(), "Nonfinite OHLCV/return")
    require((e[["open", "high", "low", "close"]] > 0).all().all(), "Nonpositive OHLC")
    require((e.high >= e[["open", "close", "low"]].max(axis=1)).all() and
            (e.low <= e[["open", "close", "high"]].min(axis=1)).all(), "Inconsistent OHLC")
    require((e.volume >= 0).all(), "Negative volumes")
    require(np.allclose(e.session_return, e.close / e.open - 1,
                        rtol=0, atol=1e-8), "Session return mismatch")
    require(e.raw_symbol.astype(str).str.fullmatch(r"ES[HMUZ][0-9]{1,2}").all(),
            "Contract symbols invalid")
    for row in e.itertuples(index=False):
        d = row.session_date
        start, end, available = (as_utc(row.session_start_utc,"session start"),
                                 as_utc(row.session_end_utc,"session end"),
                                 as_utc(row.data_available_utc,"data available"))
        require(start == session_boundary(d,18,prior=True) and end == session_boundary(d,17),
                f"Session timing mismatch for {d}")
        require(start < end <= available <= cutoff,
                f"Session {d} not completed/available before inference time")
    return e.reset_index(drop=True), source_hash, manifest


def features_from_sessions(e):
    r = e.tail(20).session_return.to_numpy(float)
    recent = e.iloc[-1]
    data = {
        "es_return_lag1": float(r[-1]),
        "es_return_lag2": float(r[-2]),
        "es_return_lag3": float(r[-3]),
        "es_return_lag5": float(r[-5]),
        "es_range_pct_lag1": float((recent.high - recent.low) / recent.open),
        "es_log_volume_lag1": float(np.log1p(recent.volume)),
        "es_return_mean_5": float(np.mean(r[-5:])),
        "es_return_mean_20": float(np.mean(r)),
        "es_vol_5": float(np.std(r[-5:], ddof=1)),
        "es_vol_20": float(np.std(r, ddof=1)),
        "es_abs_return_mean_5": float(np.mean(np.abs(r[-5:]))),
        "es_abs_return_mean_20": float(np.mean(np.abs(r))),
    }
    return pd.DataFrame([[data[k] for k in FEATURES]], columns=FEATURES, dtype=float)


# Public reasoning is presentation-only: it never changes the submitted probabilities.
# Version the wording separately from the frozen ES Daily v0.1 model.
RATIONALE_VERSION = "explanation-v1.1"
FEATURE_DISPLAY_NAMES = {
    "es_return_lag1": "prior session return",
    "es_return_lag2": "2nd prior session return",
    "es_return_lag3": "3rd prior session return",
    "es_return_lag5": "5th prior session return",
    "es_range_pct_lag1": "prior session range",
    "es_log_volume_lag1": "prior session log volume",
    "es_return_mean_5": "5-session mean return",
    "es_return_mean_20": "20-session mean return",
    "es_vol_5": "5-session return volatility",
    "es_vol_20": "20-session return volatility",
    "es_abs_return_mean_5": "5-session mean absolute return",
    "es_abs_return_mean_20": "20-session mean absolute return",
}


def build_forecast_note(probabilities, features, model, model_meta,
                        target_date, last_session, contract):
    """Build an auditable, per-forecast rationale; no fitting or external data access.

    Feature contributions are calculated exactly from standardized multinomial
    logit coefficients for the top class versus its runner-up. They describe
    associations in this fitted model, NOT causal market drivers.
    """
    require(tuple(features.columns) == FEATURES and features.shape == (1, len(FEATURES)),
            "Rationale requires the exact frozen one-row feature contract")
    require(set(probabilities) == set(CLASSES), "Rationale probability classes changed")
    probs = np.array([probabilities[c] for c in CLASSES], dtype=float)
    require(np.isfinite(probs).all() and (probs > 0).all() and
            np.isclose(probs.sum(), 1, atol=1e-10, rtol=0),
            "Rationale requires positive, normalized probabilities")
    require(set(FEATURE_DISPLAY_NAMES) == set(FEATURES), "Rationale feature labels drift")

    scaler = model.named_steps.get("scale")
    lr = model.named_steps.get("logreg")
    require(scaler is not None and lr is not None,
            "Rationale requires frozen scaler and LogisticRegression")
    native_classes = tuple(map(str, lr.classes_))
    require(set(native_classes) == set(CLASSES), "Rationale native class labels drift")
    require(lr.coef_.shape == (len(CLASSES), len(FEATURES)) and
            len(lr.intercept_) == len(CLASSES),
            "Expected three-class multinomial coefficients")
    standardized = np.asarray(scaler.transform(features), dtype=float)
    require(standardized.shape == (1, len(FEATURES)) and np.isfinite(standardized).all(),
            "Invalid standardized feature vector")

    ranking = sorted(CLASSES, key=lambda c: -probabilities[c])
    winner, runner_up = ranking[:2]
    win_idx, runner_idx = (native_classes.index(winner), native_classes.index(runner_up))
    contributions = standardized[0] * (lr.coef_[win_idx] - lr.coef_[runner_idx])
    log_odds_gap = float(lr.intercept_[win_idx] - lr.intercept_[runner_idx] +
                         contributions.sum())
    observed_log_odds_gap = float(np.log(probabilities[winner] / probabilities[runner_up]))
    require(np.isclose(log_odds_gap, observed_log_odds_gap, atol=1e-7, rtol=0),
            "Rationale attribution does not reproduce frozen model log odds")

    def drivers(order, positive, count):
        filtered = [i for i in order if (contributions[i] > 1e-10 if positive
                                          else contributions[i] < -1e-10)]
        return ", ".join(
            f"{FEATURE_DISPLAY_NAMES[FEATURES[i]]} ({contributions[i]:+.2f})"
            for i in filtered[:count]
        )

    supporting = drivers(np.argsort(-contributions), positive=True, count=2)
    opposing = drivers(np.argsort(contributions), positive=False, count=1)
    if not supporting:
        supporting = "none among the feature terms"
    if not opposing:
        opposing = "none among the feature terms"

    fit_rows = model_meta.get("model_training_rows")
    fit_until = model_meta.get("deployment_identity", {}).get("train_through")
    require(isinstance(fit_rows, int) and fit_rows > 0 and
            isinstance(fit_until, str) and fit_until,
            "Frozen deployment lacks documented fitting population")
    require(int(scaler.n_samples_seen_) == fit_rows,
            "Fitting-row count disagrees with saved scaler")

    margin_pp = 100.0 * (probabilities[winner] - probabilities[runner_up])
    last_return_pct = 100.0 * float(features["es_return_lag1"].iloc[0])
    vol_20_pct = 100.0 * float(features["es_vol_20"].iloc[0])
    note = (
        f"Market Mood Forecasting | ES Daily v0.1 | {RATIONALE_VERSION}. "
        f"{contract} challenge {target_date}; +/-0.30% opening-price dead zone. "
        f"Frozen L2 Logistic Regression (C=0.01), trained on {fit_rows:,} "
        f"feature-complete sessions through {fit_until}. "
        f"Last reviewed input session: {last_session}; "
        f"12 lagged, completed-session ES features. "
        f"Model probabilities: bearish {100*probabilities['bearish']:.1f}%, "
        f"neutral {100*probabilities['neutral']:.1f}%, "
        f"bullish {100*probabilities['bullish']:.1f}%. "
        f"Most likely: {winner} ({100*probabilities[winner]:.1f}%); "
        f"runner-up {runner_up} ({100*probabilities[runner_up]:.1f}%), "
        f"margin {margin_pp:.1f} percentage points. "
        f"Observed completed-session inputs: prior ES return {last_return_pct:+.2f}%; "
        f"20-session return volatility {vol_20_pct:.2f}%. "
        f"Top standardized log-odds contributions versus {runner_up}: "
        f"supporting {winner}: {supporting}; opposing: {opposing}. "
        "These are model associations, not causal claims. "
        "No target-session intraday prices, live headlines, future data or retraining. "
        "Market-data reviews and frozen-artifact checks passed. "
        "Probabilities express uncertainty; predictive advantage remains unproven "
        "during the prospective 30-challenge evaluation."
    )
    require(0 < len(note) <= 1800, "Rationale unusually long; review before submission")
    return note


def model_files(model_root):
    base = safe_path(model_root) if model_root else MODEL_ROOT
    if not model_root:
        choices = sorted(base.glob("deployment_*/es_daily_lr_deployment_v0_1.joblib"))
        require(len(choices) == 1,
                "Expected exactly one deployment; specify --model-dir if multiple/none exist")
        base = choices[0].parent
    joblib_path = base / "es_daily_lr_deployment_v0_1.joblib"
    metadata_path = base / "es_daily_lr_deployment_v0_1_metadata.json"
    manifest_path = base / "es_daily_lr_deployment_v0_1_manifest.json"
    require(all(p.is_file() for p in (joblib_path, metadata_path, manifest_path)),
            "Saved Notebook 10 deployment incomplete")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    meta = json.loads(metadata_path.read_text(encoding="utf-8"))
    require(checksum(joblib_path) == manifest.get("model_sha256"), "Frozen joblib checksum mismatch")
    require(checksum(metadata_path) == manifest.get("metadata_sha256"), "Model metadata checksum mismatch")
    ident = meta.get("deployment_identity", {})
    require(ident.get("feature_columns") == list(FEATURES), "Frozen feature order differs")
    require(ident.get("class_order") == list(CLASSES), "Frozen class order differs")
    require(ident.get("train_through") == "2026-09-30", "Unexpected deployment fit cutoff")
    require(ident.get("model_C") == 0.01, "Model C differs from original freeze")
    require(base.name == "deployment_" + str(manifest.get("deployment_id")),
            "Deployment ID does not match directory")
    require(ident.get("notebook10_source_sha256") == manifest.get("source_sha256"),
            "Frozen source identity mismatch")
    return joblib_path, manifest, meta


def check_live_contract(challenge, contract, target):
    live_contract = challenge.get("contract_symbol") or challenge.get("raw_symbol")
    if live_contract is not None:
        require(str(live_contract) == contract, f"Arena contract differs: {live_contract} != {contract}")
    live_day = challenge.get("session_date") or challenge.get("settlement_date")
    if live_day is not None:
        require(str(live_day)[:10] == target, f"Arena settlement date differs: {live_day}")
    live_band = challenge.get("dead_zone_pct")
    if live_band is not None:
        require(abs(float(live_band) - 0.3) < 1e-9, f"Arena neutral band changed: {live_band}")


def run_list():
    headers = authenticate()
    rows = live_challenges(headers)
    print("AUTHENTICATED ACTIVE CHALLENGES (verify ES item in UI):")
    for c in rows:
        print(json.dumps({k:c.get(k) for k in (
            "id","challenge_id","scope_key","asset","contract_symbol","session_date",
            "deadline_utc","dead_zone_pct","title","question") if k in c}, ensure_ascii=False))
    print("Do not infer a challenge ID from yesterday's challenge.")


def run_prepare(args):
    headers = authenticate()
    challenge = fetch_one(headers, args.challenge_id)
    check_live_contract(challenge, args.contract, args.date)
    deadline = deadline_for(challenge)
    now = pd.Timestamp.now(tz="UTC")
    require(now < deadline, "Deadline already passed; refusing to prepare a scored forecast")
    e, source_hash, source_manifest = validate_sessions(args.sessions, args.date, args.last_session, now)
    require(str(e.raw_symbol.iloc[-1]) == args.contract, "Last eligible ES contract differs from challenge")
    joblib_path, model_manifest, model_meta = model_files(args.model_dir)
    # Only load a locally produced, checksum-verified joblib. Never accept downloaded joblib.
    model = joblib.load(joblib_path)
    require(hasattr(model,"named_steps") and "logreg" in model.named_steps,
            "Frozen model must be a fitted LR pipeline")
    native_classes = tuple(str(c) for c in model.named_steps["logreg"].classes_)
    require(set(native_classes) == set(CLASSES), "Frozen class set mismatch")
    features = features_from_sessions(e)
    native_p = np.asarray(model.predict_proba(features), dtype=float)[0]
    p = native_p[[native_classes.index(c) for c in CLASSES]]
    require(np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
            and np.isclose(p.sum(), 1.0, atol=1e-10, rtol=0), "Probability vector invalid")
    probabilities = {cls: float(p[i]) for i, cls in enumerate(CLASSES)}
    note = build_forecast_note(
        probabilities=probabilities, features=features, model=model,
        model_meta=model_meta, target_date=args.date,
        last_session=args.last_session, contract=args.contract,
    )
    forecast_dir = PRIVATE_DIR / "forecasts"
    out = forecast_dir / f"forecast_{args.date.replace('-', '')}_{args.challenge_id}.json"
    payload = {
        "status": "PREVIEW_NOT_SUBMITTED", "challenge_id": args.challenge_id,
        "target_session_date": args.date,
        "deadline_utc": deadline.isoformat(),
        "contract_symbol": args.contract,
        "last_eligible_session": args.last_session,
        "data_available_utc": str(e.data_available_utc.iloc[-1]),
        "source_csv": safe_path(args.sessions).relative_to(PROJECT_ROOT).as_posix(),
        "source_sha256": source_hash,
        "source_manifest_sha256": checksum(safe_path(args.sessions).with_suffix(".manifest.json")),
        "model_dir": joblib_path.parent.relative_to(PROJECT_ROOT).as_posix(),
        "model_sha256": model_manifest["model_sha256"],
        "frozen_model_source_sha256": model_manifest["source_sha256"],
        "feature_order": list(FEATURES),
        "features": {k: float(features[k].iloc[0]) for k in FEATURES},
        "probabilities": probabilities,
        "argmax_direction": max(probabilities, key=probabilities.get),
        "prepared_at_utc": now.isoformat(),
        "note": note,
    }
    write_private_once(out, payload)
    print("\nFROZEN MODEL VERIFIED; NO SUBMISSION")
    print("Challenge:", args.challenge_id, "deadline:", deadline)
    print("Source date:", args.last_session, "contract:", args.contract)
    print("Model probabilities:", probabilities)
    print("Explanation version:", RATIONALE_VERSION)
    print("Explanation to review before submission:\n", note)
    print("Saved immutable forecast:", out)
    print("NEXT: Read and check the official Arena challenge UI, then use submit mode once.")


def run_explain(args):
    """Offline, read-only rationale preview for an existing immutable forecast.

    Does not edit the historical forecast and cannot change its submitted note.
    """
    forecast_path = safe_path(args.forecast)
    require(forecast_path.is_file(), "Forecast JSON missing")
    require(forecast_path.is_relative_to(PRIVATE_DIR.resolve()),
            "Forecast must come from the private Arena output directory")
    forecast = json.loads(forecast_path.read_text(encoding="utf-8"))
    require(forecast.get("status") == "PREVIEW_NOT_SUBMITTED",
            "Unexpected forecast JSON format")
    source = safe_path(PROJECT_ROOT / forecast["source_csv"])
    require(checksum(source) == forecast["source_sha256"],
            "Immutable source snapshot changed")
    require(checksum(source.with_suffix(".manifest.json")) ==
            forecast["source_manifest_sha256"], "Immutable source manifest changed")
    joblib_path, model_manifest, model_meta = model_files(
        PROJECT_ROOT / forecast["model_dir"])
    require(model_manifest["model_sha256"] == forecast["model_sha256"],
            "Frozen model differs from original forecast")
    stored_features = forecast.get("features", {})
    require(set(stored_features) == set(FEATURES), "Stored feature contract changed")
    features = pd.DataFrame(
        [[stored_features[key] for key in FEATURES]], columns=FEATURES, dtype=float)
    require(np.isfinite(features.to_numpy(dtype=float)).all(),
            "Nonfinite stored forecast features")
    # Load only the checksum-verified, locally created frozen model.
    model = joblib.load(joblib_path)
    native_classes = tuple(str(c) for c in model.named_steps["logreg"].classes_)
    raw = np.asarray(model.predict_proba(features), dtype=float)[0]
    probabilities = {c: float(raw[native_classes.index(c)]) for c in CLASSES}
    recorded = forecast.get("probabilities", {})
    require(set(recorded) == set(CLASSES) and
            np.allclose([recorded[c] for c in CLASSES],
                        [probabilities[c] for c in CLASSES], atol=1e-12, rtol=0),
            "Frozen model probabilities do not match saved original forecast")
    note = build_forecast_note(
        probabilities, features, model, model_meta,
        forecast["target_session_date"], forecast["last_eligible_session"],
        forecast["contract_symbol"])
    print("OFFLINE NOTE PREVIEW: ORIGINAL FORECAST VERIFIED; NO FILES CHANGED")
    print("Challenge:", forecast["challenge_id"])
    print("Probabilities:", probabilities)
    print("Rationale length:", len(note), "characters")
    print("\n" + note)
    print("\nNOT SUBMITTED. Existing Arena rationale remains unchanged.")


def run_submit(args):
    forecast_path = safe_path(args.forecast)
    require(forecast_path.is_file(), "Forecast file missing")
    require(forecast_path.is_relative_to(PRIVATE_DIR.resolve()),
            "Forecast must be located in private output directory")
    forecast = json.loads(forecast_path.read_text(encoding="utf-8"))
    cid = str(forecast.get("challenge_id", ""))
    require(cid and cid == args.confirm_id, "Challenge confirmation mismatch")
    require(args.approve == "I_APPROVE_ONE_SCORED_SUBMISSION", "Missing exact submission approval")
    probs = forecast.get("probabilities", {})
    require(set(probs) == set(CLASSES) and all(type(p) in (int,float) and 0 <= p <= 1 for p in probs.values())
            and abs(sum(probs.values()) - 1) <= 1e-8, "Invalid stored probability vector")
    require(forecast.get("status") == "PREVIEW_NOT_SUBMITTED", "Unexpected forecast file")
    source = safe_path(PROJECT_ROOT / forecast["source_csv"])
    require(checksum(source) == forecast["source_sha256"], "Source CSV changed")
    require(checksum(source.with_suffix(".manifest.json")) == forecast["source_manifest_sha256"],
            "Source manifest changed")
    joblib_path, model_manifest, _meta = model_files(PROJECT_ROOT / forecast["model_dir"])
    require(model_manifest["model_sha256"] == forecast["model_sha256"], "Model checksum drift")
    headers = authenticate()
    challenge = fetch_one(headers, cid)
    check_live_contract(challenge, forecast["contract_symbol"], forecast["target_session_date"])
    deadline = deadline_for(challenge)
    require(deadline.isoformat() == forecast["deadline_utc"], "Deadline changed since prediction")
    require(pd.Timestamp.now(tz="UTC") < deadline, "Deadline already passed")
    receipt = PRIVATE_DIR / "receipts" / f"receipt_{cid}.json"
    attempt = PRIVATE_DIR / "receipts" / f"attempt_{cid}.json"
    require(not receipt.exists() and not attempt.exists(),
            "Receipt or submission attempt already exists; do not submit twice")
    attempt_data = {"challenge_id": cid, "forecast_sha256": checksum(forecast_path),
                    "prepared_at_utc": forecast["prepared_at_utc"],
                    "attempt_started_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
                    "warning": "If the POST times out, DO NOT retry until the server state is reconciled"}
    write_private_once(attempt, attempt_data)  # before POST, so a timeout never causes blind double-submission
    try:
        response = requests.post(BASE + f"/api/v1/eval/challenges/{cid}/predict",
            headers=headers, timeout=18,
            json={"probabilities": probs, "reasoning": forecast["note"]})
    except requests.RequestException as exc:
        fail(f"POST result ambiguous ({type(exc).__name__}): do not retry; check Arena before any new POST. Attempt: {attempt}")
    try:
        server_payload = response.json()
    except ValueError:
        server_payload = {"non_json_server_reply": response.text[:1000]}
    server_record = {"http_status": response.status_code, "response": server_payload,
                     "forecast_sha256": checksum(forecast_path),
                     "recorded_at_utc": pd.Timestamp.now(tz="UTC").isoformat()}
    write_private_once(receipt, server_record)
    print("HTTP STATUS:", response.status_code)
    print("RECEIPT:", receipt)
    print("SERVER PREDICTION ID:", server_payload.get("prediction_id"))
    print("COUNTS FOR SCORE:", server_payload.get("counts_for_score"))
    if not (response.status_code == 201 and server_payload.get("counts_for_score") is True
            and str(server_payload.get("challenge_id")) == cid):
        fail("Not verified as first scored forecast. DO NOT automatically retry; examine receipt")
    print("SUCCESS: ONE SCORED FORECAST ACCEPTED; do not submit again.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="List authenticated active challenges, no POST")
    p = commands.add_parser("prepare", help="Verify frozen model + reviewed sessions; write forecast only")
    p.add_argument("--challenge-id", required=True)
    p.add_argument("--date", required=True, help="Challenge settlement date YYYY-MM-DD")
    p.add_argument("--last-session", required=True, help="Latest completed reviewed source date YYYY-MM-DD")
    p.add_argument("--contract", required=True, help="Official ES futures contract, e.g. ESZ6")
    p.add_argument("--sessions", required=True, help="Immutable NB11 snapshot CSV")
    p.add_argument("--model-dir", default=None, help="Optional exact existing frozen deployment directory")
    x = commands.add_parser("explain", help="Offline read-only rationale preview of a saved forecast")
    x.add_argument("--forecast", required=True, help="Exact pre-existing private forecast JSON")
    s = commands.add_parser("submit", help="One authenticated POST, write-once attempt and receipt")
    s.add_argument("--forecast", required=True, help="Exact JSON written by prepare")
    s.add_argument("--confirm-id", required=True, help="Manually verified target challenge ID")
    s.add_argument("--approve", required=True, help="Exact one-shot confirmation phrase")
    args = parser.parse_args()
    if args.command == "list": run_list()
    elif args.command == "prepare": run_prepare(args)
    elif args.command == "explain": run_explain(args)
    elif args.command == "submit": run_submit(args)


if __name__ == "__main__":
    main()
