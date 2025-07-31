import optuna
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import joblib
import time
import os

# --- Config ---
TRADE_LOG = "trades.csv"
SAVE_PATH = "model_isoforest_best.pkl"
ROLLING_WINDOW = 650

# --- Load recent trades ---
df = pd.read_csv(TRADE_LOG).dropna()
df = df.tail(ROLLING_WINDOW)

# --- Feature Selection (simplified, aligned with consumer) ---
X = df[["z_score", "price_change_pct", "time_gap_sec"]].values

# --- Optuna Objective ---
def objective(trial):
    n_estimators = trial.suggest_int("n_estimators", 50, 300)
    max_samples = trial.suggest_float("max_samples", 0.5, 1.0)
    contamination = trial.suggest_float("contamination", 0.002, 0.01)  # Lower range

    model = IsolationForest(
        n_estimators=n_estimators,
        max_samples=max_samples,
        contamination=contamination,
        random_state=42
    )
    preds = model.fit_predict(X)

    anomaly_count = np.sum(preds == -1)
    if anomaly_count == 0:
        return float("inf")  # Avoid degenerate model

    z_scores = df["z_score"].values
    separation = np.std(z_scores[preds == -1])

    return anomaly_count + (1.0 / (separation + 1e-6))  # Lower is better

# --- Run Tuning ---
start = time.time()
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=30)

# --- Train Final Model with Best Params ---
best = study.best_params
final_model = IsolationForest(
    n_estimators=best["n_estimators"],
    max_samples=best["max_samples"],
    contamination=best["contamination"],
    random_state=42
)
final_model.fit(X)
joblib.dump(final_model, SAVE_PATH)

print("✅ Best parameters:", study.best_params)
print("🧪 Best proxy score:", study.best_value)
print("⏱️ Time taken:", round(time.time() - start, 2), "seconds")
print(f"💾 Saved best model to {SAVE_PATH}")

# --- Optional: Log Tuning Output ---
with open("optuna_log.txt", "a") as f:
    f.write(f"\n---\n[{time.strftime('%Y-%m-%d %H:%M:%S')}]\n")
    f.write(f"Best params: {study.best_params}\n")
    f.write(f"Best score: {study.best_value:.4f}\n")
    f.write(f"Trials: {len(study.trials)}\n")
