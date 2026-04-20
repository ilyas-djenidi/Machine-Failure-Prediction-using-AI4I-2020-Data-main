"""
=============================================================================
PRODUCTION PREDICTOR — predict_factory.py  (FULLY FIXED v3)
Motor Failure Prediction — M'Sila Factory, Algeria
=============================================================================
All fixes from code review applied:

FIX 1: Uses saved rpm_mean from JSON (never hardcoded fallback used silently)
FIX 2: Reads correct model name from config
FIX 3: Unit conversion is deterministic — no dict mutation ordering risk
FIX 4: IsolationForest loaded from config path (not guessed)
FIX 5: Input validation with meaningful error messages before prediction
=============================================================================
"""

from __future__ import annotations
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
RISK_LEVELS = [
    (0.80, "CRITIQUE",   "Arrêt immédiat machine — STOP NOW",       "🔴"),
    (0.60, "URGENT",     "Maintenance urgente dans 24h",             "🟠"),
    (0.40, "ATTENTION",  "Planifier maintenance dans 3-5 jours",     "🟡"),
    (0.20, "SURVEILLER", "Surveiller — prochaine inspection planif", "🔵"),
    (0.00, "NORMAL",     "Fonctionnement normal",                    "🟢"),
]

# Real machine types from HMI screen filenames (confirmed from factory project)
MOTOR_TYPE_MAP = {
    "KIRICI":     2,  "PELET_MOTOR": 2,  "KARISTIRICI": 2,   # Heavy / H
    "ELEVATOR":   1,  "VIDA":        1,  "KONDISYONER": 1,   # Medium / M
    "MELASOR":    1,  "GRANUL":      1,  "ELEK":        1,
    "POMPA":      0,  "ASPIRATOR":   0,  "SOGUTUCU":    0,   # Light / L
    "BESLEYICI":  0,  "ZINCIRLI":    0,  "VIBRO":       0,
}

# Sensor bounds for input validation (physical limits)
SENSOR_BOUNDS = {
    "Air temperature [K]"     : (270.0, 380.0),   # -3°C to +107°C
    "Process temperature [K]" : (270.0, 420.0),   # -3°C to +147°C
    "Rotational speed [rpm]"  : (100.0, 3000.0),
    "Torque [Nm]"             : (0.0,   200.0),
    "Tool wear [min]"         : (0.0,   300.0),
}

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING  (must be IDENTICAL to train_final.py)
# ─────────────────────────────────────────────────────────────────────────────
def physics_features(df: pd.DataFrame, rpm_mean: float) -> pd.DataFrame:
    d = df.copy()

    def col(name):
        return d[name] if name in d.columns else pd.Series(0.0, index=d.index)

    at  = col("Air temperature [K]")
    pt  = col("Process temperature [K]")
    rpm = col("Rotational speed [rpm]")
    tor = col("Torque [Nm]")
    tw  = col("Tool wear [min]")

    d["Power_W"]           = tor * rpm * (2 * np.pi / 60)
    d["Temp_Diff_K"]       = pt - at
    d["Wear_Torque"]       = tw * tor
    d["Mechanical_Stress"] = tor / rpm.clip(lower=1)
    d["Thermal_Ratio"]     = pt / at.clip(lower=1)
    d["Speed_Deviation"]   = (rpm - rpm_mean).abs()  # uses saved training mean

    d["HDF_signal"] = ((d["Temp_Diff_K"] < 8.6) & (rpm < 1380)).astype(float)
    d["TWF_signal"] = (tw >= 200).astype(float)
    d["OSF_signal"] = (d["Wear_Torque"] > 13000).astype(float)
    d["PWF_low"]    = (d["Power_W"] < 3500).astype(float)
    d["PWF_high"]   = (d["Power_W"] > 9500).astype(float)

    for cname in ["Rotational speed [rpm]", "Torque [Nm]",
                  "Tool wear [min]", "Power_W", "Wear_Torque"]:
        if cname in d.columns:
            safe = d[cname].clip(lower=0)
            d["log_"  + cname] = np.log1p(safe)
            d["sqrt_" + cname] = np.sqrt(safe)

    d["RPM_x_Temp"]        = rpm * pt
    d["Torque_x_TempDiff"] = tor * d["Temp_Diff_K"]
    d["Wear_x_Power"]      = tw  * d["Power_W"]
    d["Wear_x_Torque_sq"]  = tw  * (tor ** 2)
    d["Torque_sq"]         = tor ** 2
    d["RPM_sq"]            = rpm ** 2
    d["TempDiff_sq"]       = d["Temp_Diff_K"] ** 2
    d["RPM_x_Wear"]        = rpm * tw
    d["Temp_x_Wear"]       = pt  * tw

    return d.replace([np.inf, -np.inf], np.nan).fillna(0)


def sanitize_cols(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {c: c.replace("[","_").replace("]","_")
                    .replace(" ","_").replace("<","lt").strip("_")
               for c in df.columns}
    return df.rename(columns=mapping)


def _detect_likely_modes(sensors: dict) -> list[str]:
    """Rule-based failure mode detection for diagnostic reporting."""
    at  = sensors.get("Air temperature [K]",      300.0)
    pt  = sensors.get("Process temperature [K]",  310.0)
    rpm = sensors.get("Rotational speed [rpm]",  1500.0)
    tor = sensors.get("Torque [Nm]",               40.0)
    tw  = sensors.get("Tool wear [min]",            0.0)

    power = tor * rpm * (2 * np.pi / 60)
    modes = []
    if tw >= 180:                          modes.append("TWF")
    if (pt - at) < 8.6 and rpm < 1380:   modes.append("HDF")
    if power < 3500 or power > 9500:      modes.append("PWF")
    if tw * tor > 12000:                  modes.append("OSF")
    return modes


# ─────────────────────────────────────────────────────────────────────────────
# PREDICTOR CLASS
# ─────────────────────────────────────────────────────────────────────────────
class MotorFailurePredictor:
    """
    Production inference engine for M'Sila factory motors.

    Usage:
        predictor = MotorFailurePredictor.from_config("models/production_config.json")
        result = predictor.predict(
            raw_input={"AmbientTemp": 28.5, "MotorTemp": 39.0,
                       "Speed_RPM": 1480, "Torque_Nm": 68, "RunHours": 3.5},
            machine_id="m_c_206",
            motor_type="VIDA"
        )
    """

    def __init__(self,
                 model,
                 scaler,
                 feature_cols: list,
                 rpm_mean: float,
                 threshold: float = 0.40,
                 anomaly_model=None,
                 model_name: str = "Unknown"):
        self.model        = model
        self.scaler       = scaler
        self.feature_cols = feature_cols
        self.rpm_mean     = rpm_mean
        self.threshold    = threshold
        self.anomaly_model = anomaly_model
        self.model_name   = model_name

    @classmethod
    def from_config(cls, config_path: str | Path) -> "MotorFailurePredictor":
        """
        Load predictor from production_config.json.
        Raises FileNotFoundError with clear message if config missing.
        """
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(
                f"Config not found: {config_path}\n"
                f"Run train_final.py first to generate models.")

        with open(config_path) as f:
            cfg = json.load(f)

        # Load model
        model_path = Path(cfg["model_path"])
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        model = joblib.load(model_path)

        # Load scaler
        scaler = joblib.load(cfg["scaler_path"])

        # Load feature columns
        with open(cfg["feature_cols_path"]) as f:
            fcols = json.load(f)

        # FIX 1: Load rpm_mean — raise if missing, never silently use hardcoded fallback
        rpm_path = Path(cfg.get("rpm_mean_path", ""))
        if not rpm_path.exists():
            raise FileNotFoundError(
                f"rpm_mean file not found: {rpm_path}\n"
                f"This means Speed_Deviation will be wrong. Retrain the model.")
        with open(rpm_path) as f:
            rpm_mean = float(json.load(f)["rpm_mean"])

        # FIX 4: Load anomaly guard from config path (not hardcoded sibling path)
        anomaly_model = None
        anomaly_path  = Path(cfg.get("anomaly_guard_path", ""))
        if anomaly_path.exists():
            anomaly_model = joblib.load(anomaly_path)
            print(f"[PREDICTOR] Anomaly guard loaded: {anomaly_path.name}")
        else:
            print(f"[PREDICTOR] WARNING: Anomaly guard not found at {anomaly_path}")

        # FIX 2: Read correct model name from config
        model_name = cfg.get("model_name", "Unknown")
        threshold  = float(cfg.get("threshold", 0.40))

        print(f"[PREDICTOR] Loaded: {model_name}")
        print(f"[PREDICTOR] Threshold={threshold:.2f}  RPM mean={rpm_mean:.2f}")
        return cls(model, scaler, fcols, rpm_mean, threshold, anomaly_model, model_name)

    def _convert_tags(self, raw: dict) -> dict:
        """
        FIX 3: Convert WinCC/Schneider tag names → UCI feature names.
        All conversions explicit and order-independent.
        """
        out = {}

        # Copy through any UCI-named keys directly
        for k, v in raw.items():
            out[k] = v

        # WinCC Celsius → Kelvin conversions
        if "AmbientTemp" in raw:
            out["Air temperature [K]"] = float(raw["AmbientTemp"]) + 273.15
        if "MotorTemp" in raw:
            out["Process temperature [K]"] = float(raw["MotorTemp"]) + 273.15

        # Direct mappings (no unit conversion)
        if "Speed_RPM" in raw:
            out["Rotational speed [rpm]"] = float(raw["Speed_RPM"])
        if "Torque_Nm" in raw:
            out["Torque [Nm]"] = float(raw["Torque_Nm"])

        # RunHours (hours) → Tool wear (minutes)
        if "RunHours" in raw:
            out["Tool wear [min]"] = float(raw["RunHours"]) * 60.0

        # Remove original WinCC tag names to avoid feature leakage
        for wk in ["AmbientTemp", "MotorTemp", "Speed_RPM", "Torque_Nm", "RunHours"]:
            out.pop(wk, None)

        return out

    def _validate_input(self, sensors: dict) -> list[str]:
        """FIX 5: Validate sensor values are within physical bounds."""
        warnings_list = []
        for feat, (lo, hi) in SENSOR_BOUNDS.items():
            val = sensors.get(feat)
            if val is None:
                warnings_list.append(f"Missing sensor: {feat}")
            elif not (lo <= float(val) <= hi):
                warnings_list.append(
                    f"Out of range: {feat}={val:.1f}  (expected {lo}–{hi})")
        return warnings_list

    def _get_risk(self, prob: float) -> tuple[str, str, str]:
        for min_p, level, action, icon in RISK_LEVELS:
            if prob >= min_p:
                return level, action, icon
        return "NORMAL", "Fonctionnement normal", "🟢"

    def predict(self,
                raw_input: dict,
                machine_id: str = "M_XXX",
                motor_type: str = "VIDA") -> dict:
        """
        Predict failure probability for one motor reading.

        Args:
            raw_input   : dict — sensor values in WinCC tag names or UCI names
            machine_id  : str — motor identifier e.g. "m_c_206"
            motor_type  : str — machine type from MOTOR_TYPE_MAP

        Returns:
            dict with full prediction result
        """
        # 1. Convert tag names and units
        sensors = self._convert_tags(raw_input)

        # 2. Set type encoding from real machine type
        sensors["Type_enc"] = MOTOR_TYPE_MAP.get(motor_type.upper(), 1)

        # 3. FIX 5: Input validation
        val_warnings = self._validate_input(sensors)
        if val_warnings:
            # Return sensor anomaly response for out-of-range inputs
            return {
                "machine_id"      : machine_id,
                "timestamp"       : datetime.now().isoformat(),
                "status"          : "INPUT_VALIDATION_ERROR",
                "skip_prediction" : True,
                "validation_warnings": val_warnings,
                "action"          : "Vérifier les capteurs — valeurs hors limites physiques.",
            }

        # 4. Feature engineering with saved rpm_mean
        defaults = {
            "Air temperature [K]"     : 300.0,
            "Process temperature [K]" : 310.0,
            "Rotational speed [rpm]"  : self.rpm_mean,
            "Torque [Nm]"             : 40.0,
            "Tool wear [min]"         : 0.0,
            "Type_enc"                : 1,
        }
        row = {**defaults, **sensors}

        df_raw = pd.DataFrame([row])
        df_eng = sanitize_cols(physics_features(df_raw, self.rpm_mean))

        # Align to training feature columns
        for c in self.feature_cols:
            if c not in df_eng.columns:
                df_eng[c] = 0.0
        df_eng = df_eng[self.feature_cols]

        # 5. Scale
        X_scaled = pd.DataFrame(
            self.scaler.transform(df_eng), columns=self.feature_cols)

        # 6. Anomaly guard
        if self.anomaly_model is not None:
            anomaly_score = float(self.anomaly_model.decision_function(X_scaled)[0])
            if anomaly_score < -0.3:
                return {
                    "machine_id"      : machine_id,
                    "timestamp"       : datetime.now().isoformat(),
                    "status"          : "SENSOR_ANOMALY",
                    "skip_prediction" : True,
                    "anomaly_score"   : round(anomaly_score, 4),
                    "action"          : "Vérifier l'état des capteurs (valeurs anormales détectées).",
                    "sensor_snapshot" : {k: round(float(v), 3) for k, v in sensors.items()
                                         if k in SENSOR_BOUNDS},
                }

        # 7. Predict
        prob  = float(self.model.predict_proba(X_scaled)[0, 1])
        risk, action, icon = self._get_risk(prob)
        modes = _detect_likely_modes(sensors)

        # Compute derived values for report
        rpm_val  = sensors.get("Rotational speed [rpm]", self.rpm_mean)
        tor_val  = sensors.get("Torque [Nm]", 40.0)
        power_w  = tor_val * rpm_val * (2 * np.pi / 60)
        temp_diff = sensors.get("Process temperature [K]", 310) - sensors.get("Air temperature [K]", 300)

        return {
            "machine_id"              : machine_id,
            "motor_type"              : motor_type,
            "model_name"              : self.model_name,
            "timestamp"               : datetime.now().isoformat(),
            "failure_probability_pct" : round(prob * 100, 2),
            "binary_prediction"       : int(prob >= self.threshold),
            "risk_level"              : risk,
            "risk_icon"               : icon,
            "action"                  : action,
            "likely_failure_modes"    : modes if modes else ["—"],
            "confidence"              : round(abs(prob - 0.5) / 0.5, 4),
            "threshold_used"          : self.threshold,
            "sensor_snapshot"         : {
                "air_temp_K"    : round(sensors.get("Air temperature [K]", 0), 1),
                "motor_temp_K"  : round(sensors.get("Process temperature [K]", 0), 1),
                "temp_diff_K"   : round(temp_diff, 2),
                "rpm"           : round(rpm_val, 1),
                "torque_Nm"     : round(tor_val, 1),
                "wear_min"      : round(sensors.get("Tool wear [min]", 0), 1),
                "power_W"       : round(power_w, 1),
            },
        }


if __name__ == "__main__":
    print("MotorFailurePredictor v3 — ready.")
    print("Usage: predictor = MotorFailurePredictor.from_config('models/production_config.json')")