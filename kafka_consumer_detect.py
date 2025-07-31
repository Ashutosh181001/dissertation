from kafka import KafkaConsumer
import json
import pandas as pd
import numpy as np
from datetime import datetime
import time
from sklearn.ensemble import IsolationForest
import joblib
import os

# --- Configs ---
KAFKA_TOPIC = "crypto_trades"
KAFKA_BROKER = "localhost:9092"
ROLLING_WINDOW = 650
Z_THRESHOLD = 3
MODEL_PATH = "model_isoforest.pkl"
BEST_MODEL_PATH = "model_isoforest_best.pkl"
RETRAIN_INTERVAL = 100

TRADE_LOG = "trades.csv"
ANOMALY_LOG = "anomalies.csv"
SLEEP = False

# --- Initialize model ---
buffer_for_training = []
counter = 0
is_model_fitted = False

if os.path.exists(BEST_MODEL_PATH):
    model = joblib.load(BEST_MODEL_PATH)
    is_model_fitted = True
    print("🌟 Loaded auto-tuned Isolation Forest model.")
elif os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    is_model_fitted = True
    print("📦 Loaded base Isolation Forest model.")
else:
    model = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
    print("🧪 Initialized new Isolation Forest model (untrained).")

# --- Kafka Consumer ---
consumer = KafkaConsumer(
    KAFKA_TOPIC,
    bootstrap_servers=KAFKA_BROKER,
    auto_offset_reset='latest',
    value_deserializer=lambda x: json.loads(x.decode('utf-8'))
)

# --- Rolling history storage ---
history_df = pd.DataFrame()

def compute_features(trade, history_df):
    features = {
        'timestamp': trade['timestamp'],
        'price': trade['price'],
        'quantity': trade['quantity'],
        'is_buyer_maker': trade['is_buyer_maker'],
        'injected': trade.get('injected', False)  # ✅ now included in trades.csv
    }

    if not history_df.empty:
        prev_price = history_df.iloc[-1]['price']
        features['price_change_pct'] = ((trade['price'] - prev_price) / prev_price) * 100
        prev_time = pd.to_datetime(history_df.iloc[-1]['timestamp'])
        curr_time = pd.to_datetime(trade['timestamp'])
        features['time_gap_sec'] = (curr_time - prev_time).total_seconds()
    else:
        features['price_change_pct'] = 0
        features['time_gap_sec'] = 0

    history_df = pd.concat([history_df, pd.DataFrame([trade])], ignore_index=True)
    history_df = history_df.tail(ROLLING_WINDOW)

    if len(history_df) >= ROLLING_WINDOW:
        prices = history_df['price'].astype(float)
        features['rolling_mean'] = prices.rolling(window=ROLLING_WINDOW).mean().iloc[-1]
        features['rolling_std'] = prices.rolling(window=ROLLING_WINDOW).std().iloc[-1]
        std = max(features['rolling_std'], 1e-6)
        features['z_score'] = (trade['price'] - features['rolling_mean']) / std
    else:
        features['rolling_mean'] = 0
        features['rolling_std'] = 0
        features['z_score'] = 0

    return features, history_df

def log_trade(features):
    pd.DataFrame([features]).to_csv(
        TRADE_LOG,
        mode='a',
        header=not os.path.exists(TRADE_LOG),
        index=False
    )

def log_anomaly(features, method):
    record = features.copy()
    if record.get('injected') == True:
        record['anomaly_type'] = method + "_injected"
    else:
        record['anomaly_type'] = method
    pd.DataFrame([record]).to_csv(
        ANOMALY_LOG,
        mode='a',
        header=not os.path.exists(ANOMALY_LOG),
        index=False
    )

# --- Main loop ---
print("🚀 Kafka Consumer started. Waiting for messages...")

for message in consumer:
    trade = message.value
    features, history_df = compute_features(trade, history_df)

    print(f"[{features['timestamp']}] Price: {features['price']} | Z: {features['z_score']:.2f} | Δ%: {features['price_change_pct']:.2f} | Δt: {features['time_gap_sec']}s")
    log_trade(features)

    if len(history_df) < ROLLING_WINDOW:
        continue

    z_score_flag = abs(features['z_score']) > Z_THRESHOLD
    if z_score_flag:
        print("🚨 Z-Score Anomaly Detected!")
        log_anomaly(features, "z_score")

    data_point = [
        features['z_score'],
        features['price_change_pct'],
        features['time_gap_sec']
    ]
    buffer_for_training.append(data_point)
    counter += 1

    if len(buffer_for_training) > ROLLING_WINDOW:
        recent_data = np.array(buffer_for_training[-ROLLING_WINDOW:])

        if counter % RETRAIN_INTERVAL == 0:
            model.fit(recent_data)
            joblib.dump(model, MODEL_PATH)
            is_model_fitted = True
            print("🔄 Isolation Forest model retrained.")

        if is_model_fitted:
            pred = model.predict([data_point])[0]
            if pred == -1 and z_score_flag:
                print("🚨 Isolation Forest Anomaly Detected (filtered)!")
                log_anomaly(features, "filtered_isoforest")

    if SLEEP:
        time.sleep(0.1)
