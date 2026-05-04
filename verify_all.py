"""
Full verification script — runs all 5 checks from the bug spec.
Run: python verify_all.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
errors = []

# ── CHECK 1: predict_factory loads and predicts ────────────────────────────
print("\n=== CHECK 1: predict_factory ===")
try:
    from notebooks.predict_factory import MotorFailurePredictor
    from pathlib import Path
    p = MotorFailurePredictor.from_config(Path("models/production_config.json"))
    r = p.predict(
        {"Air temperature [K]": 302.5, "Process temperature [K]": 318.0,
         "Rotational speed [rpm]": 1380, "Torque [Nm]": 68.0,
         "Tool wear [min]": 215, "Type_enc": 2},
        machine_id="M_C_203"
    )
    print(f"  failure_probability_pct : {r['failure_probability_pct']}")
    print(f"  risk_level              : {r['risk_level']}")
    print(f"  likely_failure_modes    : {r['likely_failure_modes']}")
    print(f"  time_to_failure_estimate: {r.get('time_to_failure_estimate')}")
    assert r["failure_probability_pct"] > 60, f"Expected >60%, got {r['failure_probability_pct']}"
    assert "sensor_snapshot" in r, "Missing sensor_snapshot"
    assert "likely_failure_modes" in r, "Missing likely_failure_modes"
    assert "time_to_failure_estimate" in r, "Missing time_to_failure_estimate"
    print(f"  [{PASS}] predict_factory OK")
except Exception as e:
    print(f"  [{FAIL}] {e}")
    errors.append(f"CHECK 1: {e}")

# ── CHECK 2: generate_report produces valid French text ────────────────────
print("\n=== CHECK 2: generate_report ===")
try:
    from notebooks.generate_report import generate_report
    report = generate_report(r, manufacturer="Siemens")
    text = report["text"]
    print(text[:600])
    assert "RAPPORT" in text, "Missing RAPPORT header"
    assert ("SIEMENS" in text.upper()), "Missing manufacturer"
    # Check risk normalization (BUG 9)
    fake_eng = dict(r)
    fake_eng["risk_level"] = "CRITICAL"  # English key
    rep2 = generate_report(fake_eng, manufacturer="Schneider")
    assert "CRITIQUE" in rep2["text"] or "ARRÊT" in rep2["text"], \
        "BUG 9 not fixed: English risk key not normalized"
    print(f"  [{PASS}] generate_report OK (incl. BUG 9 normalization)")
except Exception as e:
    print(f"  [{FAIL}] {e}")
    errors.append(f"CHECK 2: {e}")

# ── CHECK 3: API imports cleanly ──────────────────────────────────────────
print("\n=== CHECK 3: FastAPI app ===")
try:
    import importlib, sys
    from src.app_api import app
    from fastapi.testclient import TestClient
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200, f"Health check failed: {resp.text}"
    print(f"  /health -> {resp.json()}")
    assert "/report" in [r.path for r in app.routes], "Missing /report endpoint"
    print(f"  [{PASS}] FastAPI OK — /health and /report endpoints present")
except Exception as e:
    print(f"  [{FAIL}] {e}")
    errors.append(f"CHECK 3: {e}")

# ── CHECK 4: DriftDetector ─────────────────────────────────────────────────
print("\n=== CHECK 4: DriftDetector ===")
try:
    import pandas as pd
    from src.integrations.drift_detector import DriftDetector
    det = DriftDetector("data/ai4i2020.csv")
    det.load_reference()
    live = pd.read_csv("data/ai4i2020.csv").head(200)
    result = det.detect_drift(live)
    assert "error" not in result, f"Drift error: {result}"
    print(f"  {result.get('message')}")
    # Check BUG 12: path in error message
    det2 = DriftDetector("data/NONEXISTENT.csv")
    err = det2.detect_drift(live)
    assert "NONEXISTENT.csv" in err.get("error", ""), "BUG 12 not fixed: path missing from error"
    print(f"  [{PASS}] DriftDetector OK (incl. BUG 12 path in error)")
except Exception as e:
    print(f"  [{FAIL}] {e}")
    errors.append(f"CHECK 4: {e}")

# ── CHECK 5: production_config.json threshold ─────────────────────────────
print("\n=== CHECK 5: production_config.json ===")
try:
    import json
    with open("models/production_config.json") as f:
        cfg = json.load(f)
    thr = cfg["threshold"]
    print(f"  threshold       : {thr}")
    print(f"  rpm_mean_value  : {cfg.get('rpm_mean_value')}")
    print(f"  model_name      : {cfg.get('model_name')}")
    assert 0.30 <= thr <= 0.70 and thr != 0.40, \
        f"Threshold must be 0.30-0.70 and not exactly 0.40, got {thr}"
    assert cfg.get("rpm_mean_value"), "Missing rpm_mean_value in config"
    print(f"  [{PASS}] Config OK")
except Exception as e:
    print(f"  [{FAIL}] {e}")
    errors.append(f"CHECK 5: {e}")

# ── CHECK 6: 7-day window (REQ-1) ─────────────────────────────────────────
print("\n=== CHECK 6: predict_7day_window (REQ-1) ===")
try:
    import pandas as pd, numpy as np
    readings = pd.DataFrame([
        {"Air temperature [K]": 300+i*0.3, "Process temperature [K]": 310+i*0.5,
         "Rotational speed [rpm]": 1500-i*5, "Torque [Nm]": 45+i*2,
         "Tool wear [min]": 100+i*8, "Type_enc": 2}
        for i in range(20)
    ])
    window = p.predict_7day_window(readings, machine_id="M_C_TEST")
    print(f"  daily_risk      : {window['daily_risk']}")
    print(f"  trend_slope_pct : {window['trend_slope_pct']}")
    print(f"  days_to_failure : {window['days_to_failure']}")
    assert "predictions" in window
    assert "daily_risk" in window
    print(f"  [{PASS}] 7-day window OK")
except Exception as e:
    print(f"  [{FAIL}] {e}")
    errors.append(f"CHECK 6: {e}")

# -- SUMMARY ---------------------------------------------------------------
print("\n" + "=" * 60)
if not errors:
    print(f"\033[92m ALL CHECKS PASSED \033[0m")
else:
    print(f"\033[91m FAILURES:\033[0m")
    for e in errors:
        print(f"  - {e}")
sys.exit(len(errors))
