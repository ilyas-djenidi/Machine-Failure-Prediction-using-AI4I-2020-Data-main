import pandas as pd
import numpy as np
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix
)
import joblib

# Constants and Thresholds
DATA_PATH = r"c:\Users\HP\Desktop\Machine-Failure-Prediction-using-AI4I-2020-Data-main\data\ai4i2020.csv"
RESULTS_DIR = r"c:\Users\HP\Desktop\Machine-Failure-Prediction-using-AI4I-2020-Data-main\tests\results"
THRESHOLDS = {
    "accuracy": 0.80,
    "precision": 0.80,
    "recall": 0.75,
    "f1": 0.77,
    "roc_auc": 0.90
}

def load_and_preprocess():
    df = pd.read_csv(DATA_PATH)
    # Drop IDs
    X = df.drop(['UDI', 'Product ID', 'Machine failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF'], axis=1)
    y = df['Machine failure']
    
    # Sanitize column names for XGBoost compatibility (no [ or ])
    X.columns = [col.replace('[', '').replace(']', '').replace(' ', '_') for col in X.columns]
    
    # Encode categorical 'Type'
    X = pd.get_dummies(X, columns=['Type'], drop_first=True)
    
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob)
    }
    
    cm = confusion_matrix(y_test, y_pred)
    return metrics, cm

def run_tests():
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
        
    X_train, X_test, y_train, y_test = load_and_preprocess()
    
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42, scale_pos_weight=y_train.value_counts()[0]/y_train.value_counts()[1]),
        "LightGBM": LGBMClassifier(random_state=42, class_weight='balanced', verbose=-1)
    }
    
    results = {}
    best_model_name = ""
    best_f1 = 0
    
    print("\n" + "="*50)
    print("Machine Failure Prediction - Model Accuracy Tests")
    print("="*50)
    print(f"{'Model':<20} | {'Acc':<6} | {'Prec':<6} | {'Rec':<6} | {'F1':<6} | {'AUC':<6}")
    print("-" * 70)
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        metrics, cm = evaluate_model(name, model, X_test, y_test)
        results[name] = metrics
        
        # Log to table
        print(f"{name:<20} | {metrics['accuracy']:.3f} | {metrics['precision']:.3f} | {metrics['recall']:.3f} | {metrics['f1']:.3f} | {metrics['roc_auc']:.3f}")
        
        # Save confusion matrix plot
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {name}')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.savefig(os.path.join(RESULTS_DIR, f"cm_{name.lower().replace(' ', '_')}.png"))
        plt.close()
        
        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            best_model_name = name

    # Best Model Check
    print("-" * 70)
    print(f"✅ Best Model identified: {best_model_name}")
    
    # Assertions check
    passed = True
    best_metrics = results[best_model_name]
    for metric, threshold in THRESHOLDS.items():
        if best_metrics[metric] < threshold:
            print(f"❌ WARNING: {metric} ({best_metrics[metric]:.3f}) is below threshold ({threshold:.3f})")
            passed = False
        else:
            print(f"✅ {metric.capitalize()} target met!")
            
    # Save overall results
    with open(os.path.join(RESULTS_DIR, "model_performance.json"), "w") as f:
        json.dump({
            "results": results,
            "best_model": best_model_name,
            "thresholds": THRESHOLDS,
            "passed": passed
        }, f, indent=4)
        
    print("="*50)
    print(f"Results saved to {os.path.join(RESULTS_DIR, 'model_performance.json')}")
    print("="*50 + "\n")

if __name__ == "__main__":
    run_tests()
