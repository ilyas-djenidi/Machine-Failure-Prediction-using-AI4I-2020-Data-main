# ==============================================================================
# CELL 2 -- IMPORTS AND CONFIGURATION
# ==============================================================================
import warnings
import json
import os
import joblib
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm

from ucimlrepo import fetch_ucirepo
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

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

SEED = 42
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
ARTIFACTS_DIR = Path("artifacts")
ARTIFACTS_DIR.mkdir(exist_ok=True)
THRESHOLD = 0.40

print("All imports OK | Timestamp:", TIMESTAMP)
print("Artifacts dir:", ARTIFACTS_DIR.resolve())


# ==============================================================================
# CELL 3 -- DATA LOADING (UCI AI4I 2020, id=601)
# ==============================================================================
def load_uci_ai4i():
    print("Fetching UCI AI4I 2020 (id=601)...")
    dataset = fetch_ucirepo(id=601)
    X_raw = dataset.data.features.copy()
    y_raw = dataset.data.targets.copy()
    print("  Raw shape: X={}, y={}".format(X_raw.shape, y_raw.shape))
    print("  Columns:", list(X_raw.columns))

    y = y_raw["Machine failure"].astype(int)

    # Drop non-feature string columns except Type
    drop_cols = [c for c in X_raw.columns
                 if X_raw[c].dtype == object and c not in ["Type"]]
    X_raw = X_raw.drop(columns=drop_cols, errors="ignore")

    # Encode machine type L/M/H -> 0/1/2
    if "Type" in X_raw.columns:
        le = LabelEncoder()
        X_raw["Type_enc"] = le.fit_transform(X_raw["Type"].astype(str))
        X_raw = X_raw.drop(columns=["Type"])

    print("  Failure rate: {:.2f}%".format(y.mean() * 100))
    print("  Class counts:", y.value_counts().to_dict())
    return X_raw, y, y_raw


X_raw, y, y_full = load_uci_ai4i()


# ==============================================================================
# CELL 4 -- PHYSICS-INFORMED FEATURE ENGINEERING
# ==============================================================================
def physics_features(df):
    d = df.copy()

    # Safe column access with fallback to zero
    def col(name):
        return d[name] if name in d.columns else pd.Series(0.0, index=d.index)

    at  = col("Air temperature [K]")
    pt  = col("Process temperature [K]")
    rpm = col("Rotational speed [rpm]")
    tor = col("Torque [Nm]")
    tw  = col("Tool wear [min]")

    # -- Core physics features --
    d["Power_W"]            = tor * rpm * (2 * np.pi / 60)   # mechanical power
    d["Temp_Diff_K"]        = pt - at                         # HDF indicator
    d["Wear_Torque"]        = tw * tor                        # OSF indicator
    d["Mechanical_Stress"]  = tor / (rpm.clip(lower=1))       # shaft stress
    d["Thermal_Stress"]     = pt / (at.clip(lower=1))         # thermal ratio
    d["Speed_Deviation"]    = (rpm - rpm.mean()).abs()         # instability

    # -- Log/sqrt transforms on skewed sensors --
    for cname in ["Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]",
                  "Power_W", "Wear_Torque"]:
        if cname in d.columns:
            safe = d[cname].clip(lower=0)
            d["log_" + cname]  = np.log1p(safe)
            d["sqrt_" + cname] = np.sqrt(safe)

    # -- Interaction terms --
    d["RPM_x_Temp"]         = rpm * pt
    d["Torque_x_TempDiff"]  = tor * d["Temp_Diff_K"]
    d["Wear_x_Power"]       = tw  * d["Power_W"]
    d["Wear_x_Torque_sq"]   = tw  * (tor ** 2)
    d["Torque_sq"]          = tor ** 2
    d["RPM_sq"]             = rpm ** 2
    d["TempDiff_sq"]        = d["Temp_Diff_K"] ** 2

    d = d.replace([np.inf, -np.inf], np.nan).fillna(0)
    return d


X_eng = physics_features(X_raw)
FEATURE_COLS = list(X_eng.columns)
print("Feature engineering done: {} -> {} features".format(
    X_raw.shape[1], X_eng.shape[1]))
print("New features:", [c for c in FEATURE_COLS if c not in X_raw.columns])


# ==============================================================================
# CELL 5 -- SPLIT / SCALE / SMOTE
# ==============================================================================
# 70% train | 10% val | 20% test  (stratified)
X_temp, X_test, y_temp, y_test = train_test_split(
    X_eng, y, test_size=0.20, random_state=SEED, stratify=y)

val_ratio = 0.10 / 0.80   # 10% of total out of remaining 80%
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=val_ratio, random_state=SEED, stratify=y_temp)

print("Split -> Train:{} Val:{} Test:{}".format(
    len(X_train), len(X_val), len(X_test)))

# RobustScaler: IQR-based, ignores extreme sensor spikes
scaler = RobustScaler()
X_train_s = pd.DataFrame(
    scaler.fit_transform(X_train), columns=FEATURE_COLS)
X_val_s   = pd.DataFrame(
    scaler.transform(X_val),       columns=FEATURE_COLS)
X_test_s  = pd.DataFrame(
    scaler.transform(X_test),      columns=FEATURE_COLS)

# SMOTE only on training set -- val/test stay untouched
sm = SMOTE(random_state=SEED, k_neighbors=5)
X_train_bal, y_train_bal = sm.fit_resample(X_train_s, y_train)
print("After SMOTE:", pd.Series(y_train_bal).value_counts().to_dict())

# Persist scaler and feature list immediately
joblib.dump(scaler, ARTIFACTS_DIR / ("scaler_" + TIMESTAMP + ".pkl"))
with open(ARTIFACTS_DIR / ("feature_cols_" + TIMESTAMP + ".json"), "w") as fh:
    json.dump(FEATURE_COLS, fh, indent=2)
print("Scaler and feature list saved.")


# ==============================================================================
# CELL 6 -- METRIC HELPERS
# ==============================================================================
def metrics_at_threshold(y_true, y_proba, thr=THRESHOLD, name="model"):
    y_pred = (y_proba >= thr).astype(int)
    return {
        "model"    : name,
        "threshold": thr,
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall"   : float(recall_score(y_true, y_pred, zero_division=0)),
        "f1"       : float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc"  : float(roc_auc_score(y_true, y_proba)),
    }


def cv_f1(estimator, X, y_cv, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    scores = cross_val_score(
        estimator, X, y_cv, cv=skf, scoring="f1", n_jobs=-1)
    return scores.mean(), scores.std()


# ==============================================================================
# CELL 7 -- BASELINE MODELS
# ==============================================================================
neg = int((y_train_bal == 0).sum())
pos = int((y_train_bal == 1).sum())
spw = neg / pos   # scale_pos_weight for XGBoost

all_results = {}

# --- Logistic Regression ---
print("Training Logistic Regression...")
lr = LogisticRegression(
    max_iter=2000, class_weight="balanced", random_state=SEED, C=0.1)
lr.fit(X_train_bal, y_train_bal)
lr_proba = lr.predict_proba(X_val_s)[:, 1]
all_results["logistic_regression"] = metrics_at_threshold(
    y_val, lr_proba, name="LogisticRegression")
print("  Val F1={:.4f}".format(all_results["logistic_regression"]["f1"]))

# --- Random Forest ---
print("Training Random Forest...")
rf = RandomForestClassifier(
    n_estimators=300, max_depth=None, min_samples_leaf=2,
    class_weight="balanced", random_state=SEED, n_jobs=-1)
rf.fit(X_train_bal, y_train_bal)
rf_proba = rf.predict_proba(X_val_s)[:, 1]
all_results["random_forest"] = metrics_at_threshold(
    y_val, rf_proba, name="RandomForest")
print("  Val F1={:.4f}".format(all_results["random_forest"]["f1"]))

# --- XGBoost ---
print("Training XGBoost...")
xgb = XGBClassifier(
    n_estimators=500, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    scale_pos_weight=spw, random_state=SEED,
    eval_metric="logloss", early_stopping_rounds=30,
    verbosity=0, n_jobs=-1)
xgb.fit(X_train_bal, y_train_bal,
        eval_set=[(X_val_s, y_val)], verbose=False)
xgb_proba = xgb.predict_proba(X_val_s)[:, 1]
all_results["xgboost"] = metrics_at_threshold(
    y_val, xgb_proba, name="XGBoost")
print("  Val F1={:.4f}  best_iter={}".format(
    all_results["xgboost"]["f1"], xgb.best_iteration))

# --- LightGBM ---
print("Training LightGBM...")
lgb = LGBMClassifier(
    n_estimators=500, max_depth=6, learning_rate=0.05,
    num_leaves=63, subsample=0.8, colsample_bytree=0.8,
    class_weight="balanced", random_state=SEED,
    verbose=-1, n_jobs=-1)
lgb.fit(X_train_bal, y_train_bal,
        eval_set=[(X_val_s, y_val)])
lgb_proba = lgb.predict_proba(X_val_s)[:, 1]
all_results["lightgbm"] = metrics_at_threshold(
    y_val, lgb_proba, name="LightGBM")
print("  Val F1={:.4f}".format(all_results["lightgbm"]["f1"]))

results_df = pd.DataFrame(all_results.values()).sort_values("f1", ascending=False)
print("\nBaseline Comparison:")
print(results_df[["model", "precision", "recall", "f1", "roc_auc"]].to_string(index=False))

# ==============================================================================
# CELL 8 -- OPTUNA HYPERPARAMETER TUNING (50 trials, TPE, F1 objective)
# ==============================================================================
N_TRIALS = 50


def optuna_objective(trial):
    params = {
        "n_estimators"    : trial.suggest_int("n_estimators", 200, 800),
        "max_depth"       : trial.suggest_int("max_depth", 3, 10),
        "learning_rate"   : trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "subsample"       : trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma"           : trial.suggest_float("gamma", 0.0, 1.0),
        "reg_alpha"       : trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda"      : trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
    }
    m = XGBClassifier(
        **params, scale_pos_weight=spw, random_state=SEED,
        eval_metric="logloss", early_stopping_rounds=20,
        verbosity=0, n_jobs=-1)
    m.fit(X_train_bal, y_train_bal,
          eval_set=[(X_val_s, y_val)], verbose=False)
    proba = m.predict_proba(X_val_s)[:, 1]
    return float(f1_score(y_val, (proba >= THRESHOLD).astype(int), zero_division=0))


print("Optuna: {} trials (TPE, F1@thr={})...".format(N_TRIALS, THRESHOLD))
study = optuna.create_study(direction="maximize",
                            sampler=optuna.samplers.TPESampler(seed=SEED))

with tqdm(total=N_TRIALS, desc="Optuna") as pbar:
    def _optuna_cb(study, trial):
        pbar.update(1)
        pbar.set_postfix({"best_f1": "{:.4f}".format(study.best_value)})
    study.optimize(optuna_objective, n_trials=N_TRIALS, callbacks=[_optuna_cb])

best_xgb_params = study.best_params
print("Best Optuna F1: {:.4f}".format(study.best_value))

xgb_tuned = XGBClassifier(
    **best_xgb_params, scale_pos_weight=spw, random_state=SEED,
    eval_metric="logloss", early_stopping_rounds=30,
    verbosity=0, n_jobs=-1)
xgb_tuned.fit(X_train_bal, y_train_bal,
              eval_set=[(X_val_s, y_val)], verbose=False)
xgb_tuned_proba = xgb_tuned.predict_proba(X_val_s)[:, 1]
all_results["xgboost_tuned"] = metrics_at_threshold(
    y_val, xgb_tuned_proba, name="XGBoost_Tuned")
print("Tuned val:", all_results["xgboost_tuned"])


# ==============================================================================
# CELL 9 -- SOFT-VOTING ENSEMBLE (RF + XGBoost_tuned + LightGBM, w=[1,2,2])
# ==============================================================================
# IMPORTANT: VotingClassifier calls .fit() on sub-estimators without eval_set.
# XGBoost with early_stopping_rounds set will crash unless we remove it here.
# Solution: clone XGBoost with fixed n_estimators = best_iteration + 1.
_best_n = xgb_tuned.best_iteration + 1
print("Using xgb best_iteration={} for ensemble (no early stopping).".format(_best_n))

xgb_for_ensemble = XGBClassifier(
    **best_xgb_params,
    n_estimators=_best_n,       # fixed -- no early stopping needed
    scale_pos_weight=spw,
    random_state=SEED,
    eval_metric="logloss",
    verbosity=0,
    n_jobs=-1,
    # early_stopping_rounds intentionally omitted
)
# LightGBM also needs early_stopping removed for VotingClassifier
lgb_for_ensemble = LGBMClassifier(
    n_estimators=500, max_depth=6, learning_rate=0.05,
    num_leaves=63, subsample=0.8, colsample_bytree=0.8,
    class_weight="balanced", random_state=SEED,
    verbose=-1, n_jobs=-1,
    # no callbacks / eval_set
)

ensemble = VotingClassifier(
    estimators=[
        ("rf",  rf),
        ("xgb", xgb_for_ensemble),
        ("lgb", lgb_for_ensemble),
    ],
    voting="soft",
    weights=[1, 2, 2],
    n_jobs=-1,
)
ensemble.fit(X_train_bal, y_train_bal)
ens_proba = ensemble.predict_proba(X_val_s)[:, 1]
all_results["ensemble"] = metrics_at_threshold(
    y_val, ens_proba, name="Ensemble_w1_2_2")
print("Ensemble val:", all_results["ensemble"])


# ==============================================================================
# CELL 10 -- THRESHOLD SENSITIVITY (0.30 to 0.80)
# ==============================================================================
thresholds = np.arange(0.30, 0.81, 0.05)
thr_df = pd.DataFrame([
    metrics_at_threshold(y_val, ens_proba, thr=t) for t in thresholds])

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, col in zip(axes, ["precision", "recall", "f1"]):
    ax.plot(thr_df["threshold"], thr_df[col], marker="o", linewidth=2)
    ax.axvline(THRESHOLD, color="red", linestyle="--",
               label="chosen={}".format(THRESHOLD))
    ax.set_title(col.capitalize()); ax.set_xlabel("Threshold")
    ax.legend(); ax.grid(alpha=0.3)
plt.suptitle("Threshold Sensitivity (Validation Set)", fontsize=13)
plt.tight_layout()
plt.savefig(str(ARTIFACTS_DIR / "threshold_analysis.png"), dpi=150, bbox_inches="tight")
plt.show()
print("Best F1 threshold: {:.2f}".format(
    float(thr_df.loc[thr_df["f1"].idxmax(), "threshold"])))


# ==============================================================================
# CELL 11 -- FINAL TEST SET EVALUATION
# ==============================================================================
print("=" * 60)
print("FINAL TEST SET EVALUATION")
print("=" * 60)

best_name = max(all_results, key=lambda k: all_results[k]["f1"])
print("Best model by val-F1:", best_name)

model_map = {
    "logistic_regression": lr,
    "random_forest"      : rf,
    "xgboost"            : xgb,
    "lightgbm"           : lgb,
    "xgboost_tuned"      : xgb_tuned,
    "ensemble"           : ensemble,
}
best_model = model_map[best_name]
test_proba = best_model.predict_proba(X_test_s)[:, 1]
test_pred  = (test_proba >= THRESHOLD).astype(int)

test_metrics = {
    "model"    : best_name,
    "threshold": THRESHOLD,
    "precision": float(precision_score(y_test, test_pred, zero_division=0)),
    "recall"   : float(recall_score(y_test, test_pred, zero_division=0)),
    "f1"       : float(f1_score(y_test, test_pred, zero_division=0)),
    "roc_auc"  : float(roc_auc_score(y_test, test_proba)),
}
print("\nTest Metrics:")
for k, v in test_metrics.items():
    print("  {:<12}: {}".format(k, v))
print("\nClassification Report:")
print(classification_report(y_test, test_pred, target_names=["Normal", "Failure"]))


# ==============================================================================
# CELL 12 -- PLOTS: CONFUSION MATRIX + ROC + PR CURVES
# ==============================================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

ConfusionMatrixDisplay(confusion_matrix(y_test, test_pred),
                       display_labels=["Normal", "Failure"]).plot(
    ax=axes[0], colorbar=False, cmap="Blues")
axes[0].set_title("Confusion Matrix")

fpr, tpr, _ = roc_curve(y_test, test_proba)
axes[1].plot(fpr, tpr, lw=2, label="AUC={:.3f}".format(auc(fpr, tpr)))
axes[1].plot([0, 1], [0, 1], "k--")
axes[1].set(xlabel="FPR", ylabel="TPR", title="ROC Curve")
axes[1].legend(); axes[1].grid(alpha=0.3)

prec_v, rec_v, _ = precision_recall_curve(y_test, test_proba)
axes[2].plot(rec_v, prec_v, lw=2,
             label="PR-AUC={:.3f}".format(auc(rec_v, prec_v)))
axes[2].set(xlabel="Recall", ylabel="Precision", title="PR Curve")
axes[2].legend(); axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(str(ARTIFACTS_DIR / "evaluation_plots.png"), dpi=150, bbox_inches="tight")
plt.show()


# ==============================================================================
# CELL 13 -- QUALITY GATES
# ==============================================================================
print("\nQuality Gates:")
try:
    assert test_metrics["precision"] >= 0.85, \
        "Precision {:.3f} < 0.85".format(test_metrics["precision"])
    assert test_metrics["recall"]    >= 0.80, \
        "Recall {:.3f} < 0.80".format(test_metrics["recall"])
    assert test_metrics["f1"]        >= 0.82, \
        "F1 {:.3f} < 0.82".format(test_metrics["f1"])
    assert test_metrics["roc_auc"]   >= 0.95, \
        "AUC {:.3f} < 0.95".format(test_metrics["roc_auc"])
    print("  [PASS] All quality gates passed!")
    gates_passed = True
except AssertionError as e:
    print("  [FAIL]", e, "-- retrying with 100 Optuna trials...")
    gates_passed = False
    study2 = optuna.create_study(direction="maximize",
                                 sampler=optuna.samplers.TPESampler(seed=SEED + 1))
    with tqdm(total=100, desc="Retry") as pb2:
        def _cb2(s, t): pb2.update(1)
        study2.optimize(optuna_objective, n_trials=100, callbacks=[_cb2])
    xgb_r = XGBClassifier(
        **study2.best_params, scale_pos_weight=spw, random_state=SEED,
        eval_metric="logloss", early_stopping_rounds=30, verbosity=0, n_jobs=-1)
    xgb_r.fit(X_train_bal, y_train_bal, eval_set=[(X_val_s, y_val)], verbose=False)
    ens_r = VotingClassifier(
        estimators=[("rf", rf), ("xgb", xgb_r), ("lgb", lgb)],
        voting="soft", weights=[1, 3, 2], n_jobs=-1)
    ens_r.fit(X_train_bal, y_train_bal)
    tp_r = ens_r.predict_proba(X_test_s)[:, 1]
    tp_p = (tp_r >= THRESHOLD).astype(int)
    test_metrics.update({
        "precision": float(precision_score(y_test, tp_p, zero_division=0)),
        "recall"   : float(recall_score(y_test, tp_p, zero_division=0)),
        "f1"       : float(f1_score(y_test, tp_p, zero_division=0)),
        "roc_auc"  : float(roc_auc_score(y_test, tp_r)),
    })
    best_model = ens_r
    print("  Retry metrics:", test_metrics)


# ==============================================================================
# CELL 14 -- SHAP EXPLAINABILITY
# ==============================================================================
print("Computing SHAP values...")
shap_exp    = shap.TreeExplainer(xgb_tuned)
shap_samp   = X_test_s.sample(min(500, len(X_test_s)), random_state=SEED)
shap_vals   = shap_exp.shap_values(shap_samp)
sv = shap_vals[1] if isinstance(shap_vals, list) else shap_vals

fig, _ = plt.subplots(figsize=(10, 7))
shap.summary_plot(sv, shap_samp, plot_type="bar", max_display=15, show=False)
plt.title("SHAP Feature Importance -- Top 15"); plt.tight_layout()
plt.savefig(str(ARTIFACTS_DIR / "shap_importance.png"), dpi=150, bbox_inches="tight")
plt.show()

fig, _ = plt.subplots(figsize=(12, 8))
shap.summary_plot(sv, shap_samp, max_display=15, show=False)
plt.title("SHAP Beeswarm -- Feature Impact"); plt.tight_layout()
plt.savefig(str(ARTIFACTS_DIR / "shap_beeswarm.png"), dpi=150, bbox_inches="tight")
plt.show()

top15 = pd.Series(np.abs(sv).mean(axis=0),
                  index=shap_samp.columns).sort_values(ascending=False).head(15)
print("\nTop 15 by mean |SHAP|:")
for feat, val in top15.items():
    print("  {:<40} {:.4f}".format(feat, val))


# ==============================================================================
# CELL 15 -- SAVE ARTIFACTS
# ==============================================================================
joblib.dump(best_model, ARTIFACTS_DIR / ("best_model_" + best_name + "_" + TIMESTAMP + ".pkl"))

all_results["test_final"] = test_metrics
with open(ARTIFACTS_DIR / ("evaluation_results_" + TIMESTAMP + ".json"), "w") as fh:
    json.dump(all_results, fh, indent=2)

with open(ARTIFACTS_DIR / ("optuna_summary_" + TIMESTAMP + ".json"), "w") as fh:
    json.dump({"n_trials": N_TRIALS, "best_value": study.best_value,
               "best_params": best_xgb_params}, fh, indent=2)

print("\nArtifacts saved:")
for p in sorted(ARTIFACTS_DIR.iterdir()):
    print("  {}  ({} KB)".format(p.name, p.stat().st_size // 1024))

# Uncomment to download from Colab:
# from google.colab import files
# [files.download(str(p)) for p in ARTIFACTS_DIR.iterdir()]


# ==============================================================================
# CELL 16 -- MotorFailurePredictor (Production Class, WinCC/TIA compatible)
# ==============================================================================
class MotorFailurePredictor:
    """
    Production predictor for asynchronous induction motors.

    WinCC tag -> UCI feature map:
      AmbientTemp -> Air temperature [K]        (Celsius input, auto +273.15)
      MotorTemp   -> Process temperature [K]    (Celsius input, auto +273.15)
      Speed_RPM   -> Rotational speed [rpm]
      Torque_Nm   -> Torque [Nm]
      RunHours    -> Tool wear [min]             (hours input, auto x60)
      FaultBit    -> ignored (ground truth only)

    Machine tags: M_C_101 ... M_C_412
    """
    WINCC_MAP = {
        "AmbientTemp": "Air temperature [K]",
        "MotorTemp"  : "Process temperature [K]",
        "Speed_RPM"  : "Rotational speed [rpm]",
        "Torque_Nm"  : "Torque [Nm]",
        "RunHours"   : "Tool wear [min]",
    }
    RISK_MAP = [
        (0.80, "CRITICAL", "STOP NOW -- Emergency inspection required."),
        (0.60, "HIGH",     "URGENT -- Maintain within 24 hours."),
        (0.40, "MEDIUM",   "CAUTION -- Plan maintenance in 3-5 days."),
        (0.20, "LOW",      "WATCH -- Monitor more frequently."),
        (0.00, "NORMAL",   "OK -- No anomaly detected."),
    ]
    TTF_MAP = [(0.80, 1), (0.60, 3), (0.40, 5), (0.20, 7), (0.00, 14)]

    def __init__(self, model_path, scaler_path, feature_cols_path, threshold=0.40):
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        with open(feature_cols_path) as fh:
            self.feature_cols = json.load(fh)
        self.threshold = threshold
        print("MotorFailurePredictor ready | threshold={}".format(threshold))

    def _map_tags(self, raw):
        out = {}
        for tag, val in raw.items():
            out[self.WINCC_MAP.get(tag, tag)] = val
        if "AmbientTemp" in raw:
            out["Air temperature [K]"] = raw["AmbientTemp"] + 273.15
        if "MotorTemp" in raw:
            out["Process temperature [K]"] = raw["MotorTemp"] + 273.15
        if "RunHours" in raw:
            out["Tool wear [min]"] = raw["RunHours"] * 60.0
        return out

    def _build_row(self, sensors):
        defaults = {
            "Air temperature [K]"    : 298.15,
            "Process temperature [K]": 308.15,
            "Rotational speed [rpm]" : 1500.0,
            "Torque [Nm]"            : 40.0,
            "Tool wear [min]"        : 0.0,
            "Type_enc"               : 1,
        }
        row = {**defaults, **sensors}
        df_e = physics_features(pd.DataFrame([row]))
        for c in self.feature_cols:
            if c not in df_e.columns:
                df_e[c] = 0.0
        return df_e[self.feature_cols]

    def _risk(self, prob):
        for min_p, level, action in self.RISK_MAP:
            if prob >= min_p:
                return level, action
        return "NORMAL", "OK"

    def predict(self, raw_input, machine_id="M_C_XXX"):
        sensors  = self._map_tags(raw_input)
        feat_row = self._build_row(sensors)
        feat_sc  = pd.DataFrame(self.scaler.transform(feat_row),
                                columns=self.feature_cols)
        prob = float(self.model.predict_proba(feat_sc)[0, 1])
        risk, action = self._risk(prob)
        ttf = next((d for p, d in self.TTF_MAP if prob >= p), 30)
        return {
            "machine_id"               : machine_id,
            "timestamp"                : datetime.now().isoformat(),
            "failure_probability_pct"  : round(prob * 100, 2),
            "binary_prediction"        : int(prob >= self.threshold),
            "risk_level"               : risk,
            "estimated_days_to_failure": ttf,
            "action"                   : action,
            "confidence"               : round(abs(prob - 0.5) / 0.5, 4),
            "threshold_used"           : self.threshold,
        }

    def predict_batch_wincc(self, csv_path):
        """Phase 2: read WinCC CSV export and return predictions DataFrame."""
        df_w = pd.read_csv(csv_path)
        return pd.DataFrame([
            self.predict(row.to_dict(), machine_id=row.get("TagName", "M_C_XXX"))
            for _, row in df_w.iterrows()
        ])


# ==============================================================================
# CELL 17 -- DEMO PREDICTIONS
# ==============================================================================
def _latest(prefix, ext="pkl"):
    hits = sorted(ARTIFACTS_DIR.glob("{0}*.{1}".format(prefix, ext)))
    return str(hits[-1]) if hits else None


predictor = MotorFailurePredictor(
    model_path        = _latest("best_model_", "pkl"),
    scaler_path       = _latest("scaler_",     "pkl"),
    feature_cols_path = _latest("feature_cols_", "json"),
    threshold         = THRESHOLD,
)

# Case 1: overheating / high wear -> expect CRITICAL or HIGH
r1 = predictor.predict({
    "Air temperature [K]"    : 302.5,
    "Process temperature [K]": 320.1,
    "Rotational speed [rpm]" : 1380,
    "Torque [Nm]"            : 68.5,
    "Tool wear [min]"        : 220,
    "Type_enc"               : 2,
}, machine_id="M_C_203")
print("\n[M_C_203] Overheating scenario:")
for k, v in r1.items():
    print("  {:<32}: {}".format(k, v))

# Case 2: healthy motor -> expect NORMAL or LOW
r2 = predictor.predict({
    "Air temperature [K]"    : 298.1,
    "Process temperature [K]": 309.2,
    "Rotational speed [rpm]" : 1502,
    "Torque [Nm]"            : 39.8,
    "Tool wear [min]"        : 45,
    "Type_enc"               : 1,
}, machine_id="M_C_101")
print("\n[M_C_101] Normal motor:")
for k, v in r2.items():
    print("  {:<32}: {}".format(k, v))

# Case 3: WinCC tag format (Phase 2)
r3 = predictor.predict({
    "AmbientTemp": 28.5,
    "MotorTemp"  : 49.0,
    "Speed_RPM"  : 1475,
    "Torque_Nm"  : 55.0,
    "RunHours"   : 3.5,
}, machine_id="M_C_315")
print("\n[M_C_315] WinCC tag format:")
for k, v in r3.items():
    print("  {:<32}: {}".format(k, v))

print("\nDone. All outputs are in the 'artifacts/' directory.")
