from kafka import KafkaConsumer

# Import the database manager to persist trades and anomalies
try:
    from database_manager import DatabaseManager

    # Initialize a single DatabaseManager instance
    db = DatabaseManager()
    USE_DATABASE = True
except Exception as e:
    # If the database module cannot be imported, fall back to CSV logging
    print(f"Warning: Could not import DatabaseManager ({e}). Falling back to CSV logs.")
    db = None
    USE_DATABASE = False

# Import alert manager and evaluation metrics
try:
    from alert_manager import AlertManager

    alert_manager = AlertManager()
    USE_ALERTS = True
except Exception as e:
    print(f"Warning: Could not import AlertManager ({e}). Alerts disabled.")
    alert_manager = None
    USE_ALERTS = False

try:
    from evaluation_metrics import AnomalyEvaluator

    evaluator = AnomalyEvaluator(window_minutes=60)
    USE_METRICS = True
except Exception as e:
    print(f"Warning: Could not import AnomalyEvaluator ({e}). Metrics disabled.")
    evaluator = None
    USE_METRICS = False

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
ROLLING_WINDOW = 1000
Z_THRESHOLD = 3.5
MODEL_PATH = "model_isoforest.pkl"
BEST_MODEL_PATH = "model_isoforest_best.pkl"
RETRAIN_INTERVAL = 1000
METRICS_UPDATE_INTERVAL = 500  # Update metrics every 50 trades

# Paths for CSV logs (used only when database logging is unavailable)
TRADE_LOG = "trades.csv"
ANOMALY_LOG = "anomalies.csv"
SLEEP = False

# --- Initialize model ---
buffer_for_training = []
counter = 0
metrics_counter = 0
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
history_df = pd.DataFrame(columns=['timestamp', 'price', 'quantity', 'is_buyer_maker'])

# Warm up the history using the last ROLLING_WINDOW trades.
# Prefer loading from the database if available; otherwise fall back to CSV.
try:
    if USE_DATABASE and db is not None:
        try:
            # Fetch the most recent ROLLING_WINDOW trades from the DB
            recent_trades = db.get_recent_trades(minutes=24 * 60, limit=ROLLING_WINDOW)
            if not recent_trades.empty:
                history_df = recent_trades[['timestamp', 'price', 'quantity', 'is_buyer_maker']].copy()
                history_df['price'] = history_df['price'].astype(float)
                history_df['quantity'] = history_df['quantity'].astype(float)
                history_df['is_buyer_maker'] = history_df['is_buyer_maker'].astype(int)
                print(f"📊 Loaded {len(history_df)} historical trades from DB for warm start")
        except Exception as e:
            # If DB read fails, fall back to CSV
            print(f"Could not load historical data from DB: {e}")

    if history_df.empty and os.path.exists(TRADE_LOG):
        # Load last N trades from CSV to initialize history
        existing_trades = pd.read_csv(TRADE_LOG)
        if not existing_trades.empty:
            # Take last ROLLING_WINDOW trades
            history_df = existing_trades[['timestamp', 'price', 'quantity', 'is_buyer_maker']].tail(
                ROLLING_WINDOW).copy()
            history_df['price'] = history_df['price'].astype(float)
            history_df['quantity'] = history_df['quantity'].astype(float)
            history_df['is_buyer_maker'] = history_df['is_buyer_maker'].astype(int)
            print(f"📊 Loaded {len(history_df)} historical trades from CSV for warm start")
except Exception as e:
    print(f"Could not load historical data: {e}")
    history_df = pd.DataFrame(columns=['timestamp', 'price', 'quantity', 'is_buyer_maker'])


def compute_features(trade, history_df):
    """Compute features with proper history management"""
    try:
        # Create feature dictionary with current trade data
        features = {
            'timestamp': trade['timestamp'],
            'price': float(trade['price']),
            'quantity': float(trade['quantity']),
            'is_buyer_maker': int(trade.get('is_buyer_maker', False)),
            'injected': int(trade.get('injected', False))
        }

        # IMPORTANT: Work with a copy of history_df to avoid modifying the original
        if history_df.empty:
            # Initialize with proper data types
            history_df = pd.DataFrame(columns=['timestamp', 'price', 'quantity', 'is_buyer_maker'])

        # Add current trade to history
        new_row = pd.DataFrame([{
            'timestamp': trade['timestamp'],
            'price': float(trade['price']),
            'quantity': float(trade['quantity']),
            'is_buyer_maker': int(trade.get('is_buyer_maker', False))
        }])

        history_df = pd.concat([history_df, new_row], ignore_index=True)

        # Keep only the most recent ROLLING_WINDOW trades
        if len(history_df) > ROLLING_WINDOW:
            history_df = history_df.iloc[-ROLLING_WINDOW:]

        # Calculate features based on history length
        history_length = len(history_df)

        if history_length > 1:
            # Price change features
            prev_price = float(history_df.iloc[-2]['price'])
            features['price_change_pct'] = ((features[
                                                 'price'] - prev_price) / prev_price) * 100 if prev_price > 0 else 0

            # Time gap features
            try:
                prev_time = pd.to_datetime(history_df.iloc[-2]['timestamp'])
                curr_time = pd.to_datetime(trade['timestamp'])
                features['time_gap_sec'] = max((curr_time - prev_time).total_seconds(), 0)
            except:
                features['time_gap_sec'] = 0

            # Volume features
            quantities = history_df['quantity'].astype(float)
            mean_quantity = quantities.mean()
            features['volume_ratio'] = features['quantity'] / mean_quantity if mean_quantity > 0 else 1

            # Buy/Sell pressure
            if history_length >= 10:
                recent_trades = history_df.tail(10)
                buy_volume = recent_trades[recent_trades['is_buyer_maker'] == False]['quantity'].astype(float).sum()
                sell_volume = recent_trades[recent_trades['is_buyer_maker'] == True]['quantity'].astype(float).sum()
                total_volume = buy_volume + sell_volume
                features['buy_pressure'] = buy_volume / total_volume if total_volume > 0 else 0.5
            else:
                features['buy_pressure'] = 0.5
        else:
            # First trade - set defaults
            features['price_change_pct'] = 0
            features['time_gap_sec'] = 0
            features['volume_ratio'] = 1
            features['buy_pressure'] = 0.5

        # Calculate rolling statistics only if we have enough data
        if history_length >= min(50, ROLLING_WINDOW):  # Need at least 50 trades for meaningful statistics
            prices = history_df['price'].astype(float)
            quantities = history_df['quantity'].astype(float)

            # Use the full history for rolling calculations
            window_size = min(history_length, ROLLING_WINDOW)

            # Rolling mean and std
            features['rolling_mean'] = float(prices.tail(window_size).mean())
            features['rolling_std'] = float(prices.tail(window_size).std())

            # Z-score
            if features['rolling_std'] > 0:
                features['z_score'] = (features['price'] - features['rolling_mean']) / features['rolling_std']
            else:
                features['z_score'] = 0

            # VWAP calculation
            recent_trades = history_df.tail(window_size)
            recent_prices = recent_trades['price'].astype(float)
            recent_quantities = recent_trades['quantity'].astype(float)
            total_value = (recent_prices * recent_quantities).sum()
            total_quantity = recent_quantities.sum()

            if total_quantity > 0:
                features['vwap'] = float(total_value / total_quantity)
                features['vwap_deviation'] = ((features['price'] - features['vwap']) / features['vwap']) * 100
            else:
                features['vwap'] = features['rolling_mean']
                features['vwap_deviation'] = 0

            # Price velocity
            if history_length >= 20:
                lookback = min(20, history_length)
                recent_data = history_df.tail(lookback)

                try:
                    timestamps = pd.to_datetime(recent_data['timestamp'])
                    time_span = (timestamps.iloc[-1] - timestamps.iloc[0]).total_seconds()

                    if time_span > 0:
                        price_change = float(recent_data['price'].iloc[-1]) - float(recent_data['price'].iloc[0])
                        features['price_velocity'] = price_change / time_span
                    else:
                        features['price_velocity'] = 0
                except:
                    features['price_velocity'] = 0
            else:
                features['price_velocity'] = 0

            # Volume spike detection
            if history_length >= 50:
                recent_volume_mean = quantities.tail(50).mean()
                features['volume_spike'] = features['quantity'] / recent_volume_mean if recent_volume_mean > 0 else 1
            else:
                features['volume_spike'] = 1

        else:
            # Not enough data for statistics - use current values
            features['rolling_mean'] = features['price']
            features['rolling_std'] = 0
            features['z_score'] = 0
            features['vwap'] = features['price']
            features['vwap_deviation'] = 0
            features['price_velocity'] = 0
            features['volume_spike'] = 1

        # Add contextual features
        try:
            dt = pd.to_datetime(trade['timestamp'])
            features['hour'] = dt.hour
            features['day_of_week'] = dt.dayofweek
            features['is_weekend'] = 1 if dt.dayofweek >= 5 else 0

            # Trading session
            hour = dt.hour
            if 23 <= hour or hour < 8:
                features['trading_session'] = 'asian'
            elif 7 <= hour < 16:
                features['trading_session'] = 'european'
            else:
                features['trading_session'] = 'us'
        except:
            features['hour'] = 0
            features['day_of_week'] = 0
            features['is_weekend'] = 0
            features['trading_session'] = 'unknown'

        return features, history_df

    except Exception as e:
        print(f"Error computing features: {e}")
        import traceback
        traceback.print_exc()

        # Return minimal valid features on error
        return {
            'timestamp': trade.get('timestamp', datetime.now().isoformat()),
            'price': float(trade.get('price', 0)),
            'quantity': float(trade.get('quantity', 0)),
            'is_buyer_maker': 0,
            'z_score': 0,
            'rolling_mean': float(trade.get('price', 0)),
            'rolling_std': 0,
            'price_change_pct': 0,
            'time_gap_sec': 0,
            'injected': 0,
            'volume_ratio': 1,
            'buy_pressure': 0.5,
            'vwap': float(trade.get('price', 0)),
            'vwap_deviation': 0,
            'price_velocity': 0,
            'volume_spike': 1,
            'hour': 0,
            'day_of_week': 0,
            'is_weekend': 0,
            'trading_session': 'unknown'
        }, history_df


def log_trade(features):
    """
    Persist a trade either to the database (preferred) or to CSV as a fallback.
    Returns the trade_id if using the database.
    """
    if USE_DATABASE and db is not None:
        try:
            return db.insert_trade(features)
        except Exception as e:
            # If database insertion fails, fall back to CSV logging
            print(f"Database trade insertion failed: {e}. Falling back to CSV.")
    # CSV fallback
    pd.DataFrame([features]).to_csv(
        TRADE_LOG,
        mode='a',
        header=not os.path.exists(TRADE_LOG),
        index=False
    )
    return None


def log_anomaly(trade_id, features, method):
    """
    Persist an anomaly either to the database (preferred) or to CSV as a fallback.
    Accepts the associated trade_id when available.
    """
    record = features.copy()
    # Determine anomaly type, appending '_injected' for synthetic anomalies
    anomaly_type = method + "_injected" if record.get('injected') else method
    record['anomaly_type'] = anomaly_type

    if USE_DATABASE and db is not None and trade_id is not None:
        try:
            # Insert anomaly into the database
            return db.insert_anomaly(trade_id, record, anomaly_type)
        except Exception as e:
            # If database insertion fails, fall back to CSV logging
            print(f"Database anomaly insertion failed: {e}. Falling back to CSV.")

    # CSV fallback
    pd.DataFrame([record]).to_csv(
        ANOMALY_LOG,
        mode='a',
        header=not os.path.exists(ANOMALY_LOG),
        index=False
    )
    return None


def record_detection_metrics(timestamp, model_name, detected, features, actual_anomaly=None):
    """Record detection for metrics evaluation"""
    if USE_METRICS and evaluator is not None:
        # Calculate confidence based on z-score and other factors
        z_score = abs(features.get('z_score', 0))
        volume_spike = features.get('volume_spike', 1)
        price_change = abs(features.get('price_change_pct', 0))

        # Simple confidence calculation (you can make this more sophisticated)
        confidence = min(1.0, (z_score / 6.0 + min(volume_spike, 5) / 10.0 + price_change / 10.0) / 3)

        # If this is an injected anomaly, we know the ground truth
        if features.get('injected'):
            actual_anomaly = True

        evaluator.record_detection(
            timestamp=timestamp,
            model_name=model_name,
            detected=detected,
            confidence=confidence,
            actual_anomaly=actual_anomaly
        )


def send_anomaly_alert(features, anomaly_types):
    """Send alert through alert manager with cooldown"""
    if USE_ALERTS and alert_manager is not None:
        # Prepare anomaly data for alert
        anomaly_data = {
            'timestamp': features['timestamp'],
            'anomaly_type': ', '.join(anomaly_types),
            'price': features['price'],
            'z_score': features.get('z_score', 0),
            'price_change_pct': features.get('price_change_pct', 0),
            'volume_spike': features.get('volume_spike', 1),
            'vwap_deviation': features.get('vwap_deviation', 0),
            'quantity': features.get('quantity', 0)
        }

        # Alert manager will handle cooldown internally
        alert_manager.send_alert(anomaly_data)


# --- Main loop ---
print("🚀 Kafka Consumer started. Waiting for messages...")

for message in consumer:
    trade = message.value
    features, history_df = compute_features(trade, history_df)

    print(
        f"[{features['timestamp']}] Price: {features['price']} | Z: {features.get('z_score', 0):.2f} | Δ%: {features.get('price_change_pct', 0):.2f} | Δt: {features.get('time_gap_sec', 0)}s")

    # Persist the trade and capture the returned trade_id when using the database
    trade_id = log_trade(features)

    if len(history_df) < ROLLING_WINDOW:
        continue

    # Track detected anomalies for this trade
    detected_anomalies = []

    # Z-score anomaly detection
    z_score_flag = abs(features['z_score']) > Z_THRESHOLD
    if z_score_flag:
        print("🚨 Z-Score Anomaly Detected!")
        log_anomaly(trade_id, features, "z_score")
        detected_anomalies.append("z_score")
        record_detection_metrics(features['timestamp'], "z_score", True, features)
    else:
        record_detection_metrics(features['timestamp'], "z_score", False, features)

    data_point = [
        features['z_score'],
        features['price_change_pct'],
        features['time_gap_sec']
    ]
    buffer_for_training.append(data_point)
    counter += 1
    metrics_counter += 1

    if len(buffer_for_training) > ROLLING_WINDOW:
        recent_data = np.array(buffer_for_training[-ROLLING_WINDOW:])

        if counter % RETRAIN_INTERVAL == 0:
            model.fit(recent_data)
            joblib.dump(model, MODEL_PATH)
            is_model_fitted = True
            print("🔄 Isolation Forest model retrained.")

        if is_model_fitted:
            pred = model.predict([data_point])[0]
            if pred == -1:
                if z_score_flag:
                    print("🚨 Isolation Forest Anomaly Detected (filtered)!")
                    log_anomaly(trade_id, features, "filtered_isoforest")
                    detected_anomalies.append("filtered_isoforest")
                    record_detection_metrics(features['timestamp'], "filtered_isoforest", True, features)
                else:
                    print("🚨 Isolation Forest Anomaly Detected!")
                    log_anomaly(trade_id, features, "isoforest")
                    detected_anomalies.append("isoforest")
                    record_detection_metrics(features['timestamp'], "isoforest", True, features)
            else:
                record_detection_metrics(features['timestamp'], "isoforest", False, features)

    # Send alert if anomalies were detected
    if detected_anomalies:
        send_anomaly_alert(features, detected_anomalies)

    # Periodically update metrics and save summary
    if metrics_counter >= METRICS_UPDATE_INTERVAL:
        if USE_METRICS and evaluator is not None:
            # Generate and save metrics report
            print("\n📊 Updating performance metrics...")
            metrics_report = evaluator.generate_report()
            print(metrics_report)

            # Save summary to JSON
            evaluator.save_summary()

            # Generate plots
            try:
                evaluator.plot_performance(save_path="evaluation_plots.png")
            except Exception as e:
                print(f"Could not generate plots: {e}")

            # Save metrics to database if available
            if USE_DATABASE and db is not None:
                metrics = evaluator.get_metrics()
                for model_name, model_metrics in metrics.items():
                    try:
                        db.save_model_performance(model_name, model_metrics)
                    except Exception as e:
                        print(f"Could not save metrics to database: {e}")

        metrics_counter = 0

    if SLEEP:
        time.sleep(0.1)