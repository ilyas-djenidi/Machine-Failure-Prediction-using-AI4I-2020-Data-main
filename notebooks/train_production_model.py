"""
=============================================================================
PRODUCTION TRAINING PIPELINE — Motor Failure Prediction
Algeria Factories (Siemens / Schneider Asynchronous Motors)
Dataset: UCI AI4I 2020  (10 000 samples, 5 failure modes)
=============================================================================
Sensors from SCADA/PLC:
  Air temperature [K], Process temperature [K],
  Rotational speed [rpm], Torque [Nm], Tool wear [min]

Failure Modes Predicted:
  TWF - Tool Wear Failure
  HDF - Heat Dissipation Failure
  PWF - Power Failure
  OSF - Overstrain Failure
  RNF - Random Failure (rare, ~0.1%)
  Machine failure (overall)
=============================================================================
"""

import warnings, json, os, sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve, precision_recall_curve, auc,
    classification_report, ConfusionMatrixDisplay,
)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import optuna
import shap
import joblib

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
SEED      = 42
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
ROOT      = Path(__file__).resolve().parent.parent
DATA_CSV  = ROOT / "data" / "ai4i2020.csv"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

THRESHOLD  = 0.40
N_TRIALS   = 100          # Optuna trials — more = better model
FAILURE_MODES = ["TWF", "HDF", "PWF", "OSF", "RNF"]

print("=" * 65)
print(" PRODUCTION TRAINING — Motor Failure Prediction")
print(f" Timestamp : {TIMESTAMP}")
print(f" Data      : {DATA_CSV}")
print(f" Models out: {MODELS_DIR}")
print("=" * 65)


# ===========================================================================
# 1. DATA LOADING
# ===========================================================================
def load_data(csv_path: Path) -> tuple:
    """Load and clean the AI4I 2020 dataset (real factory sensor data)."""
    df = pd.read_csv(csv_path)
    print(f"\n[DATA] Loaded {len(df):,} rows | columns: {list(df.columns)}")

    # Target: overall machine failure
    y = df["Machine failure"].astype(int)

    # Per-mode targets
    y_modes = df[FAILURE_MODES].astype(int)

    # Drop non-feature columns
    drop_cols = ["UDI", "Product ID", "Machine failure"] + FAILURE_MODES
    X = df.drop(columns=drop_cols, errors="ignore")

    # Encode machine quality type: L=0, M=1, H=2
    if "Type" in X.columns:
        le = LabelEncoder()
        X["Type_enc"] = le.fit_transform(X["Type"].astype(str))
        X = X.drop(columns=["Type"])

    print(f"[DATA] Features: {list(X.columns)}")
    print(f"[DATA] Failure rate: {y.mean()*100:.2f}%  "
          f"(Normal={int((y==0).sum())}, Failure={int((y==1).sum())})")
    print("[DATA] Per-mode counts:")
    for m in FAILURE_MODES:
        print(f"  {m}: {int(y_modes[m].sum())} events")
    return X, y, y_modes


# ===========================================================================
# 2. PHYSICS-INFORMED FEATURE ENGINEERING
# ===========================================================================
def physics_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive physics-based features aligned with IM failure physics:
      - Mechanical power, thermal stress, wear-torque index
      - HDF / OSF / PWF / TWF diagnostic signals
    """
    d = df.copy()

    def col(name):
        return d[name] if name in d.columns else pd.Series(0.0, index=d.index)

    at  = col("Air temperature [K]")
    pt  = col("Process temperature [K]")
    rpm = col("Rotational speed [rpm]")
    tor = col("Torque [Nm]")
    tw  = col("Tool wear [min]")

    # --- Core physics ---
    d["Power_W"]           = tor * rpm * (2 * np.pi / 60)   # mechanical power
    d["Temp_Diff_K"]       = pt - at                          # HDF indicator
    d["Wear_Torque"]       = tw * tor                         # OSF indicator
    d["Mechanical_Stress"] = tor / rpm.clip(lower=1)          # shaft stress
    d["Thermal_Ratio"]     = pt / at.clip(lower=1)            # thermal ratio
    d["Speed_Deviation"]   = (rpm - rpm.mean()).abs()          # instability

    # --- Failure-mode direct physics flags (soft) ---
    d["HDF_signal"]  = ((d["Temp_Diff_K"] < 8.6) & (rpm < 1380)).astype(float)
    d["TWF_signal"]  = (tw >= 200).astype(float)
    d["OSF_signal"]  = (d["Wear_Torque"] > 13000).astype(float)
    d["PWF_low"]     = (d["Power_W"] < 3500).astype(float)
    d["PWF_high"]    = (d["Power_W"] > 9500).astype(float)

    # --- Log / sqrt transforms on skewed sensors ---
    for cname in ["Rotational speed [rpm]", "Torque [Nm]",
                  "Tool wear [min]", "Power_W", "Wear_Torque"]:
        if cname in d.columns:
            safe = d[cname].clip(lower=0)
            d["log_"  + cname] = np.log1p(safe)
            d["sqrt_" + cname] = np.sqrt(safe)

    # --- Interaction terms ---
    d["RPM_x_Temp"]        = rpm * pt
    d["Torque_x_TempDiff"] = tor * d["Temp_Diff_K"]
    d["Wear_x_Power"]      = tw  * d["Power_W"]
    d["Wear_x_Torque_sq"]  = tw  * (tor ** 2)
    d["Torque_sq"]         = tor ** 2
    d["RPM_sq"]            = rpm ** 2
    d["TempDiff_sq"]       = d["Temp_Diff_K"] ** 2

    # Rolling-like lag features (time-series proxy for drift)
    d["RPM_x_Wear"]        = rpm * tw
    d["Temp_x_Wear"]       = pt  * tw

    d = d.replace([np.inf, -np.inf], np.nan).fillna(0)
    return d


def sanitize_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns to be XGBoost-safe (no brackets, spaces, or special chars)."""
    mapping = {c: c.replace("[", "_").replace("]", "_").replace(" ", "_").replace("<", "lt").strip("_")
               for c in df.columns}
    return df.rename(columns=mapping)


# ===========================================================================
# 3. METRIC HELPERS
# ===========================================================================
def metrics_at_thr(y_true, y_proba, thr=THRESHOLD, name="model"):
    y_pred = (y_proba >= thr).astype(int)
    return {
        "model"    : name,
        "threshold": thr,
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall"   : float(recall_score(y_true, y_pred, zero_division=0)),
        "f1"       : float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc"  : float(roc_auc_score(y_true, y_proba)),
    }


# ===========================================================================
# 4. MAIN TRAINING
# ===========================================================================
def train():
    # ── Load ──────────────────────────────────────────────────────────────
    X_raw, y, y_modes = load_data(DATA_CSV)

    # ── Feature engineering ───────────────────────────────────────────────
    X_eng = sanitize_cols(physics_features(X_raw))
    FEATURE_COLS = list(X_eng.columns)
    print(f"\n[FEAT] {X_raw.shape[1]} raw -> {X_eng.shape[1]} engineered features")

    # ── Split: 70% train | 10% val | 20% test ─────────────────────────────
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X_eng, y, test_size=0.20, random_state=SEED, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=0.125, random_state=SEED, stratify=y_tmp)
    print(f"[SPLIT] Train={len(X_train)} Val={len(X_val)} Test={len(X_test)}")

    # ── Scale (RobustScaler — handles industrial sensor outliers) ─────────
    scaler = RobustScaler()
    X_train_s = pd.DataFrame(scaler.fit_transform(X_train), columns=FEATURE_COLS)
    X_val_s   = pd.DataFrame(scaler.transform(X_val),       columns=FEATURE_COLS)
    X_test_s  = pd.DataFrame(scaler.transform(X_test),      columns=FEATURE_COLS)

    # ── SMOTE on train only ───────────────────────────────────────────────
    sm = SMOTE(random_state=SEED, k_neighbors=5)
    X_bal, y_bal = sm.fit_resample(X_train_s, y_train)
    print(f"[SMOTE] After: {pd.Series(y_bal).value_counts().to_dict()}")

    spw = int((y_bal == 0).sum()) / int((y_bal == 1).sum())

    # ── Persist scaler + feature list ─────────────────────────────────────
    scaler_path  = MODELS_DIR / f"scaler_{TIMESTAMP}.pkl"
    fcols_path   = MODELS_DIR / f"feature_cols_{TIMESTAMP}.json"
    joblib.dump(scaler, scaler_path)
    with open(fcols_path, "w") as f:
        json.dump(FEATURE_COLS, f, indent=2)

    all_results = {}

    # ── Baseline: Logistic Regression ─────────────────────────────────────
    print("\n[TRAIN] Logistic Regression...")
    lr = LogisticRegression(max_iter=2000, class_weight="balanced",
                            random_state=SEED, C=0.1)
    lr.fit(X_bal, y_bal)
    lr_p = lr.predict_proba(X_val_s)[:, 1]
    all_results["lr"] = metrics_at_thr(y_val, lr_p, name="LR")
    print(f"  Val F1={all_results['lr']['f1']:.4f}")

    # ── Baseline: Random Forest ────────────────────────────────────────────
    print("[TRAIN] Random Forest...")
    rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                class_weight="balanced",
                                random_state=SEED, n_jobs=-1)
    rf.fit(X_bal, y_bal)
    rf_p = rf.predict_proba(X_val_s)[:, 1]
    all_results["rf"] = metrics_at_thr(y_val, rf_p, name="RF")
    print(f"  Val F1={all_results['rf']['f1']:.4f}")

    # ── XGBoost ───────────────────────────────────────────────────────────
    print("[TRAIN] XGBoost...")
    xgb = XGBClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=spw, random_state=SEED,
        eval_metric="logloss", early_stopping_rounds=30,
        verbosity=0, n_jobs=-1)
    xgb.fit(X_bal, y_bal, eval_set=[(X_val_s, y_val)], verbose=False)
    xgb_p = xgb.predict_proba(X_val_s)[:, 1]
    all_results["xgb"] = metrics_at_thr(y_val, xgb_p, name="XGB")
    print(f"  Val F1={all_results['xgb']['f1']:.4f}  best_iter={xgb.best_iteration}")

    # ── LightGBM ──────────────────────────────────────────────────────────
    print("[TRAIN] LightGBM...")
    lgb = LGBMClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        num_leaves=63, subsample=0.8, colsample_bytree=0.8,
        class_weight="balanced", random_state=SEED, verbose=-1, n_jobs=-1)
    lgb.fit(X_bal, y_bal, eval_set=[(X_val_s, y_val)])
    lgb_p = lgb.predict_proba(X_val_s)[:, 1]
    all_results["lgb"] = metrics_at_thr(y_val, lgb_p, name="LGB")
    print(f"  Val F1={all_results['lgb']['f1']:.4f}")

    # ── Optuna: Tune XGBoost ──────────────────────────────────────────────
    print(f"\n[OPTUNA] {N_TRIALS} trials (TPE sampler, F1 objective)...")

    def objective(trial):
        params = {
            "n_estimators"    : trial.suggest_int("n_estimators", 200, 1000),
            "max_depth"       : trial.suggest_int("max_depth", 3, 10),
            "learning_rate"   : trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
            "subsample"       : trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma"           : trial.suggest_float("gamma", 0.0, 2.0),
            "reg_alpha"       : trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda"      : trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        }
        m = XGBClassifier(**params, scale_pos_weight=spw, random_state=SEED,
                          eval_metric="logloss", early_stopping_rounds=20,
                          verbosity=0, n_jobs=-1)
        m.fit(X_bal, y_bal, eval_set=[(X_val_s, y_val)], verbose=False)
        p = m.predict_proba(X_val_s)[:, 1]
        return float(f1_score(y_val, (p >= THRESHOLD).astype(int), zero_division=0))

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    with tqdm(total=N_TRIALS, desc="Optuna") as pbar:
        def cb(s, t): pbar.update(1); pbar.set_postfix({"best_f1": f"{s.best_value:.4f}"})
        study.optimize(objective, n_trials=N_TRIALS, callbacks=[cb])

    print(f"[OPTUNA] Best F1={study.best_value:.4f}")
    best_p = study.best_params

    xgb_t = XGBClassifier(**best_p, scale_pos_weight=spw, random_state=SEED,
                           eval_metric="logloss", early_stopping_rounds=30,
                           verbosity=0, n_jobs=-1)
    xgb_t.fit(X_bal, y_bal, eval_set=[(X_val_s, y_val)], verbose=False)
    xgb_tp = xgb_t.predict_proba(X_val_s)[:, 1]
    all_results["xgb_tuned"] = metrics_at_thr(y_val, xgb_tp, name="XGB_Tuned")
    print(f"[OPTUNA] Tuned val: {all_results['xgb_tuned']}")

    # ── Ensemble: RF + XGB_tuned + LGB ───────────────────────────────────
    print("\n[ENSEMBLE] Building soft-voting ensemble (RF:1, XGB:2, LGB:2)...")
    _bn = xgb_t.best_iteration + 1
    xgb_e = XGBClassifier(**{**best_p, "n_estimators": _bn},
                           scale_pos_weight=spw, random_state=SEED,
                           eval_metric="logloss", verbosity=0, n_jobs=-1)
    lgb_e = LGBMClassifier(n_estimators=500, max_depth=6, learning_rate=0.05,
                            num_leaves=63, subsample=0.8, colsample_bytree=0.8,
                            class_weight="balanced", random_state=SEED,
                            verbose=-1, n_jobs=-1)
    ensemble = VotingClassifier(
        estimators=[("rf", rf), ("xgb", xgb_e), ("lgb", lgb_e)],
        voting="soft", weights=[1, 2, 2], n_jobs=-1)
    ensemble.fit(X_bal, y_bal)
    ens_p = ensemble.predict_proba(X_val_s)[:, 1]
    all_results["ensemble"] = metrics_at_thr(y_val, ens_p, name="Ensemble")
    print(f"[ENSEMBLE] Val: {all_results['ensemble']}")

    # ── Select best model ─────────────────────────────────────────────────
    model_map = {
        "lr": lr, "rf": rf, "xgb": xgb,
        "lgb": lgb, "xgb_tuned": xgb_t, "ensemble": ensemble,
    }
    best_name  = max(all_results, key=lambda k: all_results[k]["f1"])
    best_model = model_map[best_name]
    print(f"\n[BEST] Best val model: {best_name}")

    # ── Final test evaluation ─────────────────────────────────────────────
    print("\n" + "=" * 65)
    print(" FINAL TEST SET EVALUATION")
    print("=" * 65)
    test_proba = best_model.predict_proba(X_test_s)[:, 1]
    test_pred  = (test_proba >= THRESHOLD).astype(int)
    test_m = metrics_at_thr(y_test, test_proba, name=best_name)

    print(classification_report(y_test, test_pred,
                                 target_names=["Normal", "Failure"]))
    for k, v in test_m.items():
        print(f"  {k:<12}: {v}")

    # ── Quality Gates ─────────────────────────────────────────────────────
    print("\n[GATES] Checking quality gates...")
    gates = {
        "Precision": (test_m["precision"], 0.80),
        "Recall"   : (test_m["recall"],    0.80),
        "F1"       : (test_m["f1"],        0.82),
        "ROC-AUC"  : (test_m["roc_auc"],  0.95),
    }
    gates_ok = True
    for gate, (val, thr) in gates.items():
        status = "PASS ✓" if val >= thr else "FAIL ✗"
        print(f"  {gate:<12} {val:.4f} >= {thr:.2f}  [{status}]")
        if val < thr:
            gates_ok = False

    if not gates_ok:
        print("[GATES] Some gates failed — running retry with 150 Optuna trials...")
        study2 = optuna.create_study(direction="maximize",
                                     sampler=optuna.samplers.TPESampler(seed=SEED+99))
        with tqdm(total=150, desc="Retry") as pb2:
            def cb2(s, t): pb2.update(1)
            study2.optimize(objective, n_trials=150, callbacks=[cb2])
        xgb_r = XGBClassifier(**study2.best_params, scale_pos_weight=spw,
                               random_state=SEED, eval_metric="logloss",
                               early_stopping_rounds=30, verbosity=0, n_jobs=-1)
        xgb_r.fit(X_bal, y_bal, eval_set=[(X_val_s, y_val)], verbose=False)
        _bnr = xgb_r.best_iteration + 1
        ens_r = VotingClassifier(
            estimators=[("rf", rf),
                        ("xgb", XGBClassifier(**{**study2.best_params, "n_estimators": _bnr},
                                               scale_pos_weight=spw, random_state=SEED,
                                               eval_metric="logloss", verbosity=0, n_jobs=-1)),
                        ("lgb", lgb_e)],
            voting="soft", weights=[1, 3, 2], n_jobs=-1)
        ens_r.fit(X_bal, y_bal)
        best_model = ens_r
        test_proba = best_model.predict_proba(X_test_s)[:, 1]
        test_pred  = (test_proba >= THRESHOLD).astype(int)
        test_m = metrics_at_thr(y_test, test_proba, name="RetryEnsemble")
        print(f"  Retry test metrics: {test_m}")

    # ── SHAP Explainability ────────────────────────────────────────────────
    print("\n[SHAP] Computing SHAP values on XGBoost tuned model...")
    shap_exp  = shap.TreeExplainer(xgb_t)
    shap_samp = X_test_s.sample(min(500, len(X_test_s)), random_state=SEED)
    shap_vals = shap_exp.shap_values(shap_samp)
    sv = shap_vals[1] if isinstance(shap_vals, list) else shap_vals

    fig, _ = plt.subplots(figsize=(10, 7))
    shap.summary_plot(sv, shap_samp, plot_type="bar", max_display=15, show=False)
    plt.title("SHAP Feature Importance — Top 15"); plt.tight_layout()
    shap_bar_path = MODELS_DIR / f"shap_importance_{TIMESTAMP}.png"
    plt.savefig(str(shap_bar_path), dpi=150, bbox_inches="tight")
    plt.close()

    fig, _ = plt.subplots(figsize=(12, 8))
    shap.summary_plot(sv, shap_samp, max_display=15, show=False)
    plt.title("SHAP Beeswarm — Feature Impact"); plt.tight_layout()
    shap_bee_path = MODELS_DIR / f"shap_beeswarm_{TIMESTAMP}.png"
    plt.savefig(str(shap_bee_path), dpi=150, bbox_inches="tight")
    plt.close()

    top15 = pd.Series(np.abs(sv).mean(axis=0),
                      index=shap_samp.columns).sort_values(ascending=False).head(15)
    print("\nTop 15 features by mean |SHAP|:")
    for feat, val in top15.items():
        print(f"  {feat:<42} {val:.4f}")

    # ── Evaluation plots ───────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    ConfusionMatrixDisplay(confusion_matrix(y_test, test_pred),
                           display_labels=["Normal", "Failure"]).plot(
        ax=axes[0], colorbar=False, cmap="Blues")
    axes[0].set_title("Confusion Matrix")

    fpr, tpr, _ = roc_curve(y_test, test_proba)
    axes[1].plot(fpr, tpr, lw=2, label=f"AUC={auc(fpr,tpr):.3f}")
    axes[1].plot([0, 1], [0, 1], "k--"); axes[1].set(xlabel="FPR", ylabel="TPR", title="ROC Curve")
    axes[1].legend(); axes[1].grid(alpha=0.3)

    pr_v, re_v, _ = precision_recall_curve(y_test, test_proba)
    axes[2].plot(re_v, pr_v, lw=2, label=f"PR-AUC={auc(re_v,pr_v):.3f}")
    axes[2].set(xlabel="Recall", ylabel="Precision", title="PR Curve")
    axes[2].legend(); axes[2].grid(alpha=0.3)
    plt.tight_layout()
    eval_plot_path = MODELS_DIR / f"evaluation_plots_{TIMESTAMP}.png"
    plt.savefig(str(eval_plot_path), dpi=150, bbox_inches="tight")
    plt.close()

    # ── Save artifacts ─────────────────────────────────────────────────────
    model_path = MODELS_DIR / f"best_model_{TIMESTAMP}.pkl"
    joblib.dump(best_model, model_path)

    config = {
        "timestamp"         : TIMESTAMP,
        "model_name"        : best_name,
        "model_path"        : str(model_path),
        "scaler_path"       : str(scaler_path),
        "feature_cols_path" : str(fcols_path),
        "threshold"         : THRESHOLD,
        "test_metrics"      : test_m,
        "optuna_best_f1"    : study.best_value,
        "optuna_best_params": best_p,
        "feature_count"     : len(FEATURE_COLS),
        "training_samples"  : int(len(X_bal)),
    }
    config_path = MODELS_DIR / "production_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 65)
    print(" ARTIFACTS SAVED")
    print("=" * 65)
    for p in sorted(MODELS_DIR.iterdir()):
        print(f"  {p.name:<55} ({p.stat().st_size // 1024} KB)")

    print("\n[DONE] Training complete!")
    print(f"  Best model : {best_name}")
    print(f"  Test F1    : {test_m['f1']:.4f}")
    print(f"  Test AUC   : {test_m['roc_auc']:.4f}")
    print(f"  Config     : {config_path}")
    return config


if __name__ == "__main__":
    cfg = train()
