"""
=============================================================================
RUN_FACTORY_DEMO.py
Full demo: Train → Predict → Generate French diagnostic reports
Algeria factories — Siemens WinCC / Schneider EcoStruxure
=============================================================================
Run:  python notebooks/run_factory_demo.py
=============================================================================
"""

import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "notebooks"))

from predict_factory import MotorFailurePredictor
from generate_report import generate_report

MODELS_DIR   = ROOT / "models"
REPORTS_DIR  = ROOT / "reports"
REPORTS_DIR.mkdir(exist_ok=True)
CONFIG_PATH  = MODELS_DIR / "production_config.json"

# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Ensure model is trained
# ─────────────────────────────────────────────────────────────────────────────
if not CONFIG_PATH.exists():
    print("[!] Model not found. Running training first...")
    import subprocess, sys
    result = subprocess.run(
        [sys.executable, str(ROOT / "notebooks" / "train_production_model.py")],
        cwd=str(ROOT), capture_output=False
    )
    if result.returncode != 0:
        print("[ERROR] Training failed. Check output above.")
        sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Load predictor
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  LOADING PRODUCTION PREDICTOR")
print("=" * 65)
predictor = MotorFailurePredictor.from_config(CONFIG_PATH)

# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Real Algerian factory scenarios
# ─────────────────────────────────────────────────────────────────────────────

SCENARIOS = [
    # --- WinCC / Siemens tag format ---
    {
        "machine_id"  : "MEE1-M01",
        "manufacturer": "Siemens",
        "description" : "Moteur pompe centrifuge — surchauffe + usure élevée",
        "input": {
            "AmbientTemp": 35.0,   # °C (été Algérie)
            "MotorTemp"  : 58.5,   # °C → risque HDF
            "Speed_RPM"  : 1360,   # sous-vitesse
            "Torque_Nm"  : 65.0,   # couple élevé → OSF
            "RunHours"   : 3.8,    # heures × 60 = 228 min → TWF
        },
    },
    # --- UCI/Direct sensor format ---
    {
        "machine_id"  : "SCHNDR-P3",
        "manufacturer": "Schneider",
        "description" : "Moteur compresseur — puissance hors plage",
        "input": {
            "Air temperature [K]"    : 302.0,
            "Process temperature [K]": 315.0,
            "Rotational speed [rpm]" : 2750,   # sur-vitesse
            "Torque [Nm]"            : 72.0,   # couple très élevé → PWF
            "Tool wear [min]"        : 95,
            "Type_enc"               : 2,      # H = haute qualité
        },
    },
    # --- Normal healthy motor ---
    {
        "machine_id"  : "MEE1-M05",
        "manufacturer": "Siemens",
        "description" : "Moteur convoyeur — état normal",
        "input": {
            "AmbientTemp": 28.0,
            "MotorTemp"  : 38.5,
            "Speed_RPM"  : 1502,
            "Torque_Nm"  : 38.5,
            "RunHours"   : 1.2,    # 72 min — faible usure
        },
    },
    # --- 1-week-ahead warning scenario ---
    {
        "machine_id"  : "MEE1-M03",
        "manufacturer": "Siemens",
        "description" : "Moteur ventilateur — dégradation progressive (2 semaines)",
        "input": {
            "AmbientTemp": 31.0,
            "MotorTemp"  : 44.0,
            "Speed_RPM"  : 1420,
            "Torque_Nm"  : 52.0,
            "RunHours"   : 2.9,    # 174 min — proche seuil TWF
        },
    },
]

all_results = []

for i, scenario in enumerate(SCENARIOS, 1):
    print(f"\n{'─'*65}")
    print(f"  SCÉNARIO {i}: {scenario['description']}")
    print(f"{'─'*65}")

    result = predictor.predict(
        scenario["input"],
        machine_id=scenario["machine_id"]
    )

    # Print prediction summary
    print(f"  Machine        : {result['machine_id']}")
    print(f"  Probabilité    : {result['failure_probability_pct']:.1f} %")
    print(f"  Risque         : {result['risk_icon']} {result['risk_level']}")
    print(f"  Délai panne    : {result['time_to_failure_estimate']}")
    print(f"  Modes détectés : {', '.join(result['likely_failure_modes'])}")
    print(f"  Confiance      : {result['confidence']*100:.1f} %")

    # Generate French report
    report = generate_report(result, manufacturer=scenario["manufacturer"])
    print("\n" + report["text"])

    # Save report to file
    report_path = REPORTS_DIR / f"rapport_{result['machine_id']}_{result['timestamp'][:10]}.txt"
    report["save"](str(report_path))

    all_results.append(result)

# ─────────────────────────────────────────────────────────────────────────────
# Step 4: Summary table
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  RÉSUMÉ — TABLEAU DE BORD USINE")
print("=" * 65)
print(f"  {'Machine':<15} {'Probabilité':>12}  {'Risque':<12}  {'Délai panne':<18}  {'Modes'}")
print(f"  {'─'*13:<15} {'─'*10:>12}  {'─'*10:<12}  {'─'*16:<18}  {'─'*20}")
for r in all_results:
    modes = ", ".join(r["likely_failure_modes"])
    print(f"  {r['machine_id']:<15} {r['failure_probability_pct']:>11.1f}%  "
          f"{r['risk_icon']} {r['risk_level']:<10}  {r['time_to_failure_estimate']:<18}  {modes}")

# Save summary JSON
summary_path = REPORTS_DIR / "summary_dashboard.json"
with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
print(f"\n  Résumé JSON sauvegardé : {summary_path}")
print(f"  Rapports individuels   : {REPORTS_DIR}/")

print("\n" + "=" * 65)
print("  DÉMONSTRATION TERMINÉE")
print("  Le système est prêt pour l'intégration SCADA/WinCC.")
print("=" * 65)
