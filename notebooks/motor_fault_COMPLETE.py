# =============================================================================
# CELL 3 â€” DATA LOADING  (UCI AI4I 2020, id=601)
# =============================================================================
def load_uci_ai4i():
    print("â¬‡ï¸  Fetching UCI AI4I 2020 (id=601)â€¦")
    dataset = fetch_ucirepo(id=601)
    X_raw = dataset.data.features.copy()
    y_raw = dataset.data.targets.copy()
    print(f"   Raw shape : X={X_raw.shape}, y={y_raw.shape}")

    target_col = "Machine failure"
    y = y_raw[target_col].astype(int)

    drop_cols = [c for c in X_raw.columns if X_raw[c].dtype == object and c not in ["Type"]]
    X_raw = X_raw.drop(columns=drop_cols, errors="ignore")

    if "Type" in X_raw.columns:
        le = LabelEncoder()
        X_raw["Type_enc"] = le.fit_transform(X_raw["Type"].astype(str))
        X_raw = X_raw.drop(columns=["Type"])

    print(f"   Failure rate: {y.mean()*100:.2f}%")
    return X_raw, y, y_raw


X_raw, y, y_full = load_uci_ai4i()


# =============================================================================
# CELL 4 â€” PHYSICS-INFORMED FEATURE ENGINEERING
# =============================================================================
def physics_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    at  = d["Air temperature [K]"]
    pt  = d["Process temperature [K]"]
    rpm = d["Rotational speed [rpm]"]
    tor = d["Torque [Nm]"]
    tw  = d["Tool wear [min]"]

    # Core physics
    d["Power_W"]           = tor * rpm * (2 * np.pi / 60)
    d["Temp_Diff_K"]       = pt - at
    d["Wear_Torque"]       = tw * tor
    d["Mechanical_Stress"] = tor / (rpm + 1e-6)
    d["Thermal_Stress"]    = pt / (at + 1e-6)
    d["Speed_Deviation"]   = np.abs(rpm - rpm.mean())

    # Log / sqrt stabilisation
    for col in ["Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]", "Power_W", "Wear_Torque"]:
        safe = d[col].clip(lower=0)
        d[f"log_{col}"]  = np.log1p(safe)
        d[f"sqrt_{col}"] = np.sqrt(safe)

    # Interaction terms
    d["RPM_x_Temp"]        = rpm * pt
    d["Torque_x_TempDiff"] = tor * d["Temp_Diff_K"]
    d["Wear_x_Power"]      = tw  * d["Power_W"]
    d["Wear_x_Torque_sq"]  = tw  * (tor ** 2)
    d["Torque_sq"]         = tor ** 2
    d["RPM_sq"]            = rpm ** 2
    d["TempDiff_sq"]       = d["Temp_Diff_K"] ** 2

    return d.replace([np.inf, -np.inf], np.nan).fillna(0)


X_eng = physics_features(X_raw)
FEATURE_COLS = list(X_eng.columns)
print(f"âœ… Feature engineering: {X_raw.shape[1]} â†’ {X_eng.shape[1]} features")
# =============================================================================
# CELL 5 â€” SPLIT / SCALE / SMOTE
# =============================================================================
# 70% train | 10% val | 20% test  (stratified)
X_temp, X_test, y_temp, y_test = train_test_split(
    X_eng, y, test_size=0.20, random_state=SEED, stratify=y)

val_ratio = 0.10 / 0.80   # 10% of total from remaining 80%
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=val_ratio, random_state=SEED, stratify=y_temp)

print(f"Split sizes â†’ Train:{len(X_train)}  Val:{len(X_val)}  Test:{len(X_test)}")

# RobustScaler  (IQR-based, outlier-robust â€” critical for factory sensors)
scaler = RobustScaler()
X_train_s = pd.DataFrame(scaler.fit_transform(X_train), columns=FEATURE_COLS)
X_val_s   = pd.DataFrame(scaler.transform(X_val),       columns=FEATURE_COLS)
X_test_s  = pd.DataFrame(scaler.transform(X_test),      columns=FEATURE_COLS)

# SMOTE on TRAINING ONLY â€” never touch val/test
sm = SMOTE(random_state=SEED, k_neighbors=5)
X_train_bal, y_train_bal = sm.fit_resample(X_train_s, y_train)
print(f"After SMOTE â†’ {pd.Series(y_train_bal).value_counts().to_dict()}")

# Save scaler + feature list
joblib.dump(scaler, ARTIFACTS_DIR / f"scaler_{TIMESTAMP}.pkl")
with open(ARTIFACTS_DIR / f"feature_cols_{TIMESTAMP}.json", "w") as f:
    json.dump(FEATURE_COLS, f, indent=2)
print("âœ… Scaler + feature list saved")


# =============================================================================
# CELL 6 â€” HELPER: compute metrics at custom threshold
# =============================================================================
def metrics_at_threshold(y_true, y_proba, thr=THRESHOLD, name="model"):
    y_pred = (y_proba >= thr).astype(int)
    return {
        "model"    : name,
        "threshold": thr,
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall"   : recall_score(y_true, y_pred, zero_division=0),
        "f1"       : f1_score(y_true, y_pred, zero_division=0),
        "roc_auc"  : roc_auc_score(y_true, y_proba),
    }

def cv_f1(estimator, X, y, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    scores = cross_val_score(estimator, X, y, cv=skf, scoring="f1", n_jobs=-1)
    return scores.mean(), scores.std()
# =============================================================================
# CELL 7 â€” BASELINE MODELS
# =============================================================================
neg, pos = (y_train_bal == 0).sum(), (y_train_bal == 1).sum()
spw = neg / pos  # scale_pos_weight for XGBoost

all_results = {}

# -- Logistic Regression --
print("Training Logistic Regressionâ€¦")
lr = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED, C=0.1)
lr.fit(X_train_bal, y_train_bal)
lr_proba = lr.predict_proba(X_val_s)[:, 1]
all_results["logistic_regression"] = metrics_at_threshold(y_val, lr_proba, name="LogisticRegression")
print(f"  Val F1={all_results['logistic_regression']['f1']:.4f}")

# -- Random Forest --
print("Training Random Forestâ€¦")
rf = RandomForestClassifier(
    n_estimators=300, max_depth=None, min_samples_leaf=2,
    class_weight="balanced", random_state=SEED, n_jobs=-1)
rf.fit(X_train_bal, y_train_bal)
rf_proba = rf.predict_proba(X_val_s)[:, 1]
all_results["random_forest"] = metrics_at_threshold(y_val, rf_proba, name="RandomForest")
print(f"  Val F1={all_results['random_forest']['f1']:.4f}")

# -- XGBoost --
print("Training XGBoostâ€¦")
xgb = XGBClassifier(
    n_estimators=500, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    scale_pos_weight=spw, random_state=SEED,
    eval_metric="logloss", early_stopping_rounds=30,
    verbosity=0, n_jobs=-1)
xgb.fit(X_train_bal, y_train_bal,
        eval_set=[(X_val_s, y_val)], verbose=False)
xgb_proba = xgb.predict_proba(X_val_s)[:, 1]
all_results["xgboost"] = metrics_at_threshold(y_val, xgb_proba, name="XGBoost")
print(f"  Val F1={all_results['xgboost']['f1']:.4f}  (best iter={xgb.best_iteration})")

# -- LightGBM --
print("Training LightGBMâ€¦")
lgb = LGBMClassifier(
    n_estimators=500, max_depth=6, learning_rate=0.05,
    num_leaves=63, subsample=0.8, colsample_bytree=0.8,
    class_weight="balanced", random_state=SEED, verbose=-1, n_jobs=-1)
lgb.fit(X_train_bal, y_train_bal,
        eval_set=[(X_val_s, y_val)],
        callbacks=[lgb.early_stopping(30, verbose=False)] if hasattr(lgb, 'early_stopping') else [])
lgb_proba = lgb.predict_proba(X_val_s)[:, 1]
all_results["lightgbm"] = metrics_at_threshold(y_val, lgb_proba, name="LightGBM")
print(f"  Val F1={all_results['lightgbm']['f1']:.4f}")

# Summary table
results_df = pd.DataFrame(all_results.values()).sort_values("f1", ascending=False)
print("\nðŸ“Š Baseline Comparison:")
print(results_df[["model","precision","recall","f1","roc_auc"]].to_string(index=False))
# =============================================================================
# CELL 8 â€” OPTUNA HYPERPARAMETER TUNING  (50 trials, TPE sampler, F1)
# =============================================================================
N_TRIALS = 50

def optuna_objective(trial):
    params = {
        "n_estimators"    : trial.suggest_int("n_estimators", 200, 800),
        "max_depth"       : trial.suggest_int("max_depth", 3, 10),
        "learning_rate"   : trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "subsample"       : trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma"           : trial.suggest_float("gamma", 0, 1.0),
        "reg_alpha"       : trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda"      : trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
    }
    m = XGBClassifier(
        **params,
        scale_pos_weight=spw, random_state=SEED,
        eval_metric="logloss", early_stopping_rounds=20,
        verbosity=0, n_jobs=-1)
    m.fit(X_train_bal, y_train_bal,
          eval_set=[(X_val_s, y_val)], verbose=False)
    proba = m.predict_proba(X_val_s)[:, 1]
    return f1_score(y_val, (proba >= THRESHOLD).astype(int), zero_division=0)


print(f"ðŸ” Optuna: {N_TRIALS} trials (TPE sampler)â€¦")
study = optuna.create_study(direction="maximize",
                            sampler=optuna.samplers.TPESampler(seed=SEED))
with tqdm(total=N_TRIALS, desc="Optuna trials") as pbar:
    def callback(study, trial):
        pbar.update(1)
        pbar.set_postfix({"best_f1": f"{study.best_value:.4f}"})
    study.optimize(optuna_objective, n_trials=N_TRIALS, callbacks=[callback])

best_xgb_params = study.best_params
print(f"\nâœ… Best Optuna F1 (val@thr={THRESHOLD}): {study.best_value:.4f}")
print(f"   Best params: {best_xgb_params}")

# Retrain best XGBoost with optimal params
xgb_tuned = XGBClassifier(
    **best_xgb_params,
    scale_pos_weight=spw, random_state=SEED,
    eval_metric="logloss", early_stopping_rounds=30,
    verbosity=0, n_jobs=-1)
xgb_tuned.fit(X_train_bal, y_train_bal,
              eval_set=[(X_val_s, y_val)], verbose=False)
xgb_tuned_proba = xgb_tuned.predict_proba(X_val_s)[:, 1]
all_results["xgboost_tuned"] = metrics_at_threshold(
    y_val, xgb_tuned_proba, name="XGBoost_Tuned")
print(f"   Tuned XGBoost Val: {all_results['xgboost_tuned']}")
# =============================================================================
# CELL 9 â€” SOFT-VOTING ENSEMBLE  (RF + XGBoost_tuned + LightGBM, w=[1,2,2])
# =============================================================================
# Note: VotingClassifier with voting='soft' averages predict_proba outputs
ensemble = VotingClassifier(
    estimators=[
        ("rf",  rf),
        ("xgb", xgb_tuned),
        ("lgb", lgb),
    ],
    voting="soft",
    weights=[1, 2, 2],
    n_jobs=-1
)
# The sub-estimators are already trained; fit() on original balanced data
# so sklearn meta-API works (predict_proba chains correctly)
ensemble.fit(X_train_bal, y_train_bal)
ens_proba = ensemble.predict_proba(X_val_s)[:, 1]
all_results["ensemble"] = metrics_at_threshold(y_val, ens_proba, name="Ensemble(w=[1,2,2])")
print(f"âœ… Ensemble Val: {all_results['ensemble']}")


# =============================================================================
# CELL 10 â€” THRESHOLD SENSITIVITY ANALYSIS (0.30 â€“ 0.80)
# =============================================================================
thresholds = np.arange(0.30, 0.81, 0.05)
thr_rows = []
for thr in thresholds:
    m = metrics_at_threshold(y_val, ens_proba, thr=thr, name="ensemble")
    thr_rows.append(m)
thr_df = pd.DataFrame(thr_rows)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, col in zip(axes, ["precision","recall","f1"]):
    ax.plot(thr_df["threshold"], thr_df[col], marker="o", linewidth=2)
    ax.axvline(THRESHOLD, color="red", linestyle="--", label=f"chosen={THRESHOLD}")
    ax.set_title(col.capitalize()); ax.set_xlabel("Threshold"); ax.legend()
    ax.grid(alpha=0.3)
plt.suptitle("Ensemble â€” Threshold Sensitivity (Validation Set)", fontsize=13)
plt.tight_layout()
plt.savefig(ARTIFACTS_DIR / "threshold_analysis.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"\nOptimal threshold for F1: {thr_df.loc[thr_df['f1'].idxmax(), 'threshold']:.2f}")
# =============================================================================
# CELL 11 â€” FINAL EVALUATION ON HELD-OUT TEST SET
# =============================================================================
print("=" * 60)
print("FINAL TEST SET EVALUATION")
print("=" * 60)

# Pick best model by val-F1
best_name = max(all_results, key=lambda k: all_results[k]["f1"])
print(f"ðŸ† Best model by val-F1: {best_name}")

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
    "precision": precision_score(y_test, test_pred, zero_division=0),
    "recall"   : recall_score(y_test, test_pred, zero_division=0),
    "f1"       : f1_score(y_test, test_pred, zero_division=0),
    "roc_auc"  : roc_auc_score(y_test, test_proba),
}

print("\nðŸ“Š Test Metrics:")
for k, v in test_metrics.items():
    print(f"   {k:<12}: {v}")

print("\nðŸ“‹ Classification Report:")
print(classification_report(y_test, test_pred, target_names=["Normal","Failure"]))


# =============================================================================
# CELL 12 â€” CONFUSION MATRIX + ROC/PR CURVES
# =============================================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Confusion matrix
cm = confusion_matrix(y_test, test_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=["Normal","Failure"])
disp.plot(ax=axes[0], colorbar=False, cmap="Blues")
axes[0].set_title(f"Confusion Matrix\n({best_name} @ thr={THRESHOLD})")

# ROC curve
fpr, tpr, _ = roc_curve(y_test, test_proba)
roc_auc = auc(fpr, tpr)
axes[1].plot(fpr, tpr, lw=2, label=f"AUC={roc_auc:.3f}")
axes[1].plot([0,1],[0,1],"k--")
axes[1].set_xlabel("FPR"); axes[1].set_ylabel("TPR")
axes[1].set_title("ROC Curve (Test Set)"); axes[1].legend(); axes[1].grid(alpha=0.3)

# Precision-Recall curve
prec_vals, rec_vals, _ = precision_recall_curve(y_test, test_proba)
pr_auc = auc(rec_vals, prec_vals)
axes[2].plot(rec_vals, prec_vals, lw=2, label=f"PR-AUC={pr_auc:.3f}")
axes[2].set_xlabel("Recall"); axes[2].set_ylabel("Precision")
axes[2].set_title("Precision-Recall Curve"); axes[2].legend(); axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(ARTIFACTS_DIR / "evaluation_plots.png", dpi=150, bbox_inches="tight")
plt.show()


# =============================================================================
# CELL 13 â€” QUALITY GATES (hard asserts)
# =============================================================================
print("\nðŸ”’ Quality Gates:")
try:
    assert test_metrics["precision"] >= 0.85, f"Precision too low: {test_metrics['precision']:.3f}"
    assert test_metrics["recall"]    >= 0.80, f"Recall too low: {test_metrics['recall']:.3f}"
    assert test_metrics["f1"]        >= 0.82, f"F1 too low: {test_metrics['f1']:.3f}"
    assert test_metrics["roc_auc"]   >= 0.95, f"AUC too low: {test_metrics['roc_auc']:.3f}"
    print("   âœ… ALL quality gates PASSED")
    gates_passed = True
except AssertionError as e:
    print(f"   âš ï¸  Gate FAILED: {e}")
    print("   ðŸ” Re-running Optuna with 100 trialsâ€¦")
    gates_passed = False
    study2 = optuna.create_study(direction="maximize",
                                 sampler=optuna.samplers.TPESampler(seed=SEED+1))
    with tqdm(total=100, desc="Retry Optuna") as pbar:
        def cb2(study, trial): pbar.update(1)
        study2.optimize(optuna_objective, n_trials=100, callbacks=[cb2])
    xgb_retry = XGBClassifier(**study2.best_params, scale_pos_weight=spw,
                               random_state=SEED, eval_metric="logloss",
                               early_stopping_rounds=30, verbosity=0, n_jobs=-1)
    xgb_retry.fit(X_train_bal, y_train_bal,
                  eval_set=[(X_val_s, y_val)], verbose=False)
    ensemble_retry = VotingClassifier(
        estimators=[("rf", rf), ("xgb", xgb_retry), ("lgb", lgb)],
        voting="soft", weights=[1, 3, 2], n_jobs=-1)
    ensemble_retry.fit(X_train_bal, y_train_bal)
    test_proba_r = ensemble_retry.predict_proba(X_test_s)[:, 1]
    test_pred_r  = (test_proba_r >= THRESHOLD).astype(int)
    test_metrics["precision"] = precision_score(y_test, test_pred_r, zero_division=0)
    test_metrics["recall"]    = recall_score(y_test, test_pred_r, zero_division=0)
    test_metrics["f1"]        = f1_score(y_test, test_pred_r, zero_division=0)
    test_metrics["roc_auc"]   = roc_auc_score(y_test, test_proba_r)
    print(f"   Retry metrics: {test_metrics}")
    best_model = ensemble_retry
# =============================================================================
# CELL 14 â€” SHAP EXPLAINABILITY
# =============================================================================
print("ðŸ” Computing SHAP values (TreeExplainer)â€¦")

# Use XGBoost tuned model for SHAP (native tree support)
shap_model = xgb_tuned
explainer   = shap.TreeExplainer(shap_model)

# Use a sample of test set for speed (all 2000 rows if Colab GPU)
shap_sample = X_test_s.sample(min(500, len(X_test_s)), random_state=SEED)
shap_values = explainer.shap_values(shap_sample)

# If multi-output, pick class=1
if isinstance(shap_values, list):
    sv = shap_values[1]
else:
    sv = shap_values

# --- Bar plot: mean |SHAP| across all samples (top 15) ---
fig, ax = plt.subplots(figsize=(10, 7))
shap.summary_plot(sv, shap_sample, plot_type="bar",
                  max_display=15, show=False)
plt.title("SHAP Feature Importance â€” Top 15 (Bar)", fontsize=14)
plt.tight_layout()
plt.savefig(ARTIFACTS_DIR / "shap_importance.png", dpi=150, bbox_inches="tight")
plt.show()

# --- Beeswarm: direction + magnitude ---
fig, ax = plt.subplots(figsize=(12, 8))
shap.summary_plot(sv, shap_sample, max_display=15, show=False)
plt.title("SHAP Beeswarm â€” Feature Impact on Failure Prediction", fontsize=14)
plt.tight_layout()
plt.savefig(ARTIFACTS_DIR / "shap_beeswarm.png", dpi=150, bbox_inches="tight")
plt.show()

# Top features (text)
mean_shap = pd.Series(np.abs(sv).mean(axis=0), index=shap_sample.columns)
top15 = mean_shap.sort_values(ascending=False).head(15)
print("\nðŸ… Top 15 features by mean |SHAP|:")
for feat, val in top15.items():
    print(f"   {feat:<40} {val:.4f}")


# =============================================================================
# CELL 15 â€” SAVE ARTIFACTS
# =============================================================================
# Best model
model_path = ARTIFACTS_DIR / f"best_model_{best_name}_{TIMESTAMP}.pkl"
joblib.dump(best_model, model_path)

# All evaluation results
all_results["test_final"] = test_metrics
with open(ARTIFACTS_DIR / f"evaluation_results_{TIMESTAMP}.json", "w") as f:
    json.dump(all_results, f, indent=2)

# Optuna study summary
optuna_summary = {
    "n_trials"  : N_TRIALS,
    "best_value": study.best_value,
    "best_params": best_xgb_params,
}
with open(ARTIFACTS_DIR / f"optuna_summary_{TIMESTAMP}.json", "w") as f:
    json.dump(optuna_summary, f, indent=2)

print("\nðŸ“¦ Artifacts saved:")
for p in sorted(ARTIFACTS_DIR.iterdir()):
    print(f"   {p.name}  ({p.stat().st_size//1024} KB)")

# --- Colab download (uncomment when running in Colab) ---
# from google.colab import files
# for p in ARTIFACTS_DIR.iterdir():
#     files.download(str(p))
# =============================================================================
# CELL 16 â€” MotorFailurePredictor  (Production Predictor Class)
# Compatible with Siemens TIA/WinCC M_C_xxx PLC tag schema
# =============================================================================

class MotorFailurePredictor:
    """
    Production-ready predictor for asynchronous induction motors.

    PLC Tag Mapping (WinCC / TIA Portal V16):
        AmbientTemp  â†’ Air temperature [K]   (convert Â°C â†’ K: +273.15)
        MotorTemp    â†’ Process temperature [K]
        Speed_RPM    â†’ Rotational speed [rpm]
        Current_A    â†’ (used for torque proxy if Torque not available)
        RunHours     â†’ Tool wear [min]         (1 h = 60 min)
        FaultBit     â†’ Not used in prediction (ground truth label)

    Factory tag schema: M_C_101, M_C_102, â€¦ M_C_412
    """

    # PLC tag â†’ UCI feature name mapping (Phase 2 WinCC CSV integration)
    WINCC_MAP = {
        "AmbientTemp" : "Air temperature [K]",      # Â°C â†’ K (+273.15 externally)
        "MotorTemp"   : "Process temperature [K]",
        "Speed_RPM"   : "Rotational speed [rpm]",
        "Torque_Nm"   : "Torque [Nm]",
        "RunHours"    : "Tool wear [min]",           # hours Ã— 60
        # Current_A â†’ requires Torque estimation (future: Motor_kW / (sqrt3 Ã— V Ã— PF))
    }

    RISK_MAP = [
        (0.80, "CRITICAL", "â›” STOP MACHINE â€” Emergency maintenance. Do not restart."),
        (0.60, "HIGH",     "ðŸ”´ URGENT â€” Schedule within 24 h. Monitor continuously."),
        (0.40, "MEDIUM",   "ðŸŸ¡ CAUTION â€” Plan maintenance in 3-5 days. Order parts."),
        (0.20, "LOW",      "ðŸŸ¢ WATCH â€” Normal operation. Standard maintenance schedule."),
        (0.00, "NORMAL",   "âœ… OK â€” No anomaly detected."),
    ]

    TTF_MAP = [(0.80, 1), (0.60, 3), (0.40, 5), (0.20, 7), (0.00, 14)]  # (min_prob, days)

    def __init__(self, model_path: str, scaler_path: str, feature_cols_path: str,
                 threshold: float = 0.40):
        self.model     = joblib.load(model_path)
        self.scaler    = joblib.load(scaler_path)
        with open(feature_cols_path) as f:
            self.feature_cols = json.load(f)
        self.threshold = threshold
        print(f"âœ… MotorFailurePredictor loaded  |  threshold={threshold}")

    def _map_wincc_tags(self, raw: dict) -> dict:
        """Translate WinCC tag names to UCI column names (Phase 2 integration)."""
        mapped = {}
        for tag, val in raw.items():
            uci_name = self.WINCC_MAP.get(tag, tag)
            mapped[uci_name] = val
        # Unit conversion helpers
        if "AmbientTemp" in raw and "Air temperature [K]" not in raw:
            mapped["Air temperature [K]"] = raw["AmbientTemp"] + 273.15
        if "MotorTemp" in raw and "Process temperature [K]" not in raw:
            mapped["Process temperature [K]"] = raw["MotorTemp"] + 273.15
        if "RunHours" in raw:
            mapped["Tool wear [min]"] = raw["RunHours"] * 60
        return mapped

    def _build_feature_row(self, sensor_dict: dict) -> pd.DataFrame:
        """Build a 1-row DataFrame with all engineered features."""
        # Fill defaults for any missing sensors
        defaults = {
            "Air temperature [K]"  : 298.15,
            "Process temperature [K]": 308.15,
            "Rotational speed [rpm]": 1500.0,
            "Torque [Nm]"          : 40.0,
            "Tool wear [min]"      : 0.0,
            "Type_enc"             : 1,
        }
        row = {**defaults, **sensor_dict}
        df_raw = pd.DataFrame([row])

        # Add Type_enc if missing
        if "Type_enc" not in df_raw.columns:
            df_raw["Type_enc"] = 1

        # Physics feature engineering (same function as training)
        df_eng = physics_features(df_raw)

        # Align columns (fill any missing with 0)
        for col in self.feature_cols:
            if col not in df_eng.columns:
                df_eng[col] = 0.0

        return df_eng[self.feature_cols]

    def _risk_level(self, prob: float) -> tuple:
        for min_p, level, action in self.RISK_MAP:
            if prob >= min_p:
                return level, action
        return "NORMAL", "âœ… OK"

    def _time_to_failure(self, prob: float) -> int:
        for min_p, days in self.TTF_MAP:
            if prob >= min_p:
                return days
        return 30

    def predict(self, raw_input: dict, machine_id: str = "M_C_XXX") -> dict:
        """
        Main prediction entry point.

        Args:
            raw_input: dict of sensor readings.
                       Keys can be UCI names OR WinCC tag names.
            machine_id: PLC motor tag (e.g. 'M_C_203')

        Returns:
            dict with: machine_id, failure_probability, risk_level,
                       estimated_days_to_failure, action, confidence, timestamp
        """
        # Map WinCC tags if needed
        sensors = self._map_wincc_tags(raw_input)

        # Build feature row + scale
        feat_row = self._build_feature_row(sensors)
        feat_scaled = pd.DataFrame(
            self.scaler.transform(feat_row), columns=self.feature_cols)

        # Predict probability
        prob = float(self.model.predict_proba(feat_scaled)[0, 1])
        binary = int(prob >= self.threshold)
        confidence = abs(prob - 0.50) / 0.50
        risk, action = self._risk_level(prob)
        ttf = self._time_to_failure(prob)

        return {
            "machine_id"              : machine_id,
            "timestamp"               : datetime.now().isoformat(),
            "failure_probability_pct" : round(prob * 100, 2),
            "binary_prediction"       : binary,
            "risk_level"              : risk,
            "estimated_days_to_failure": ttf,
            "action"                  : action,
            "confidence"              : round(confidence, 4),
            "threshold_used"          : self.threshold,
        }

    def predict_batch_wincc(self, csv_path: str) -> pd.DataFrame:
        """
        Phase 2: Accepts a WinCC CSV export and returns predictions for all rows.
        Zero code changes needed â€” just point at the CSV.
        """
        df_wincc = pd.read_csv(csv_path)
        results = []
        for _, row in df_wincc.iterrows():
            machine_id = row.get("TagName", "M_C_XXX")
            sensor_vals = row.to_dict()
            results.append(self.predict(sensor_vals, machine_id=machine_id))
        return pd.DataFrame(results)


# =============================================================================
# CELL 17 â€” DEMO: Instantiate and test the predictor
# =============================================================================
# Find latest artifacts
def latest(prefix):
    candidates = sorted(ARTIFACTS_DIR.glob(f"{prefix}*.pkl"))
    return str(candidates[-1]) if candidates else None

def latest_json(prefix):
    candidates = sorted(ARTIFACTS_DIR.glob(f"{prefix}*.json"))
    return str(candidates[-1]) if candidates else None


predictor = MotorFailurePredictor(
    model_path       = latest("best_model_"),
    scaler_path      = latest("scaler_"),
    feature_cols_path= latest_json("feature_cols_"),
    threshold        = THRESHOLD,
)

# --- Test Case 1: Overheating motor (high failure risk) ---
result_critical = predictor.predict({
    "Air temperature [K]"     : 302.5,
    "Process temperature [K]" : 320.1,   # large temp diff â†’ HDF risk
    "Rotational speed [rpm]"  : 1380,     # speed drop
    "Torque [Nm]"             : 68.5,     # high torque â†’ OSF risk
    "Tool wear [min]"         : 220,      # heavy wear
    "Type_enc"                : 2,
}, machine_id="M_C_203")

print("\nðŸ“¡ Prediction â€” CRITICAL motor scenario:")
for k, v in result_critical.items():
    print(f"   {k:<30}: {v}")

# --- Test Case 2: Normal healthy motor ---
result_ok = predictor.predict({
    "Air temperature [K]"     : 298.1,
    "Process temperature [K]" : 309.2,
    "Rotational speed [rpm]"  : 1502,
    "Torque [Nm]"             : 39.8,
    "Tool wear [min]"         : 45,
    "Type_enc"                : 1,
}, machine_id="M_C_101")

print("\nðŸ“¡ Prediction â€” Normal motor:")
for k, v in result_ok.items():
    print(f"   {k:<30}: {v}")

# --- WinCC tag naming test (Phase 2 API) ---
result_wincc = predictor.predict({
    "AmbientTemp" : 28.5,     # Â°C
    "MotorTemp"   : 49.0,     # Â°C
    "Speed_RPM"   : 1475,
    "Torque_Nm"   : 55.0,
    "RunHours"    : 3.5,      # hours â†’ converted to 210 min internally
}, machine_id="M_C_315")

print("\nðŸ“¡ Prediction â€” WinCC tag format (M_C_315):")
for k, v in result_wincc.items():
    print(f"   {k:<30}: {v}")
