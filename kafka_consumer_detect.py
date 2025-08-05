"""
Enhanced Kafka Consumer with improved stability and consistent CSV format
"""

from kafka import KafkaConsumer
from kafka.errors import KafkaError
import json
import pandas as pd
import numpy as np
from datetime import datetime
import time
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
import joblib
import os
import warnings
import traceback
import signal
import sys

warnings.filterwarnings('ignore')

# Import our new modules with error handling
try:
    from evaluation_metrics import AnomalyEvaluator
    from alert_manager import AlertManager
    from database_manager import DatabaseManager
    HAS_MODULES = True
except ImportError as e:
    print(f"Warning: Some modules not available: {e}")
    HAS_MODULES = False

# --- Configs ---
KAFKA_TOPIC = "crypto_trades"
KAFKA_BROKER = "localhost:9092"
ROLLING_WINDOW = 650
Z_THRESHOLD = 3
RETRAIN_INTERVAL = 100

# File paths
TRADE_LOG = "trades.csv"
ANOMALY_LOG = "anomalies.csv"
MODEL_DIR = "models"
SLEEP = False

# CSV columns to maintain consistency
TRADE_CSV_COLUMNS = [
    'timestamp', 'price', 'quantity', 'is_buyer_maker',
    'z_score', 'rolling_mean', 'rolling_std',
    'price_change_pct', 'time_gap_sec', 'injected'
]

ANOMALY_CSV_COLUMNS = [
    'timestamp', 'price', 'quantity', 'z_score',
    'anomaly_type', 'model_votes'
]

# Create model directory if not exists
os.makedirs(MODEL_DIR, exist_ok=True)

# --- Initialize components with error handling ---
db = None
alert_manager = None
evaluator = None

if HAS_MODULES:
    try:
        db = DatabaseManager()
    except Exception as e:
        print(f"Warning: Database initialization failed: {e}")
        db = None

    try:
        alert_manager = AlertManager("alert_config.json")
    except Exception as e:
        print(f"Warning: Alert manager initialization failed: {e}")
        alert_manager = None

    try:
        evaluator = AnomalyEvaluator(window_minutes=60)
    except Exception as e:
        print(f"Warning: Evaluator initialization failed: {e}")
        evaluator = None

# Global flag for graceful shutdown
shutdown_flag = False

def signal_handler(sig, frame):
    """Handle shutdown signals gracefully"""
    global shutdown_flag
    print('\n🛑 Shutdown signal received. Closing gracefully...')
    shutdown_flag = True

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# --- Model Ensemble Class ---
class AnomalyEnsemble:
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.voting_threshold = 0.5

        # Initialize models
        self.models['isolation_forest'] = IsolationForest(
            n_estimators=150,
            contamination=0.01,
            random_state=42,
            n_jobs=-1
        )

        self.models['lof'] = LocalOutlierFactor(
            n_neighbors=20,
            contamination=0.01,
            novelty=True,
            n_jobs=-1
        )

        self.models['one_class_svm'] = OneClassSVM(
            kernel='rbf',
            gamma='auto',
            nu=0.01
        )

    def fit(self, X):
        """Fit all models in the ensemble"""
        try:
            X_scaled = self.scaler.fit_transform(X)

            for name, model in self.models.items():
                try:
                    model.fit(X_scaled)
                    print(f"✓ {name} fitted successfully")
                except Exception as e:
                    print(f"✗ Error fitting {name}: {e}")

            self.is_fitted = True
        except Exception as e:
            print(f"Error in ensemble fit: {e}")
            self.is_fitted = False

    def predict(self, X):
        """Get ensemble prediction using voting"""
        if not self.is_fitted:
            return 1

        try:
            X_scaled = self.scaler.transform(X.reshape(1, -1))
            votes = []

            for name, model in self.models.items():
                try:
                    # Convert numpy int64 to regular Python int
                    pred = int(model.predict(X_scaled)[0])
                    votes.append(1 if pred == -1 else 0)
                except:
                    votes.append(0)

            anomaly_ratio = sum(votes) / len(votes) if votes else 0
            return -1 if anomaly_ratio >= self.voting_threshold else 1
        except Exception as e:
            print(f"Error in ensemble predict: {e}")
            return 1

    def get_model_predictions(self, X):
        """Get individual model predictions for analysis"""
        if not self.is_fitted:
            return {}

        try:
            X_scaled = self.scaler.transform(X.reshape(1, -1))
            predictions = {}

            for name, model in self.models.items():
                try:
                    # Convert numpy int64 to regular Python int
                    pred = model.predict(X_scaled)[0]
                    predictions[name] = int(pred)  # Convert to Python int
                except:
                    predictions[name] = 1

            return predictions
        except Exception as e:
            print(f"Error getting model predictions: {e}")
            return {}

    def save(self, base_path):
        """Save all models and scaler"""
        try:
            joblib.dump(self.scaler, os.path.join(base_path, "ensemble_scaler.pkl"))
            for name, model in self.models.items():
                joblib.dump(model, os.path.join(base_path, f"ensemble_{name}.pkl"))
        except Exception as e:
            print(f"Error saving models: {e}")

    def load(self, base_path):
        """Load saved models if they exist"""
        try:
            scaler_path = os.path.join(base_path, "ensemble_scaler.pkl")
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
                self.is_fitted = True

            for name in self.models.keys():
                model_path = os.path.join(base_path, f"ensemble_{name}.pkl")
                if os.path.exists(model_path):
                    self.models[name] = joblib.load(model_path)
                    print(f"📦 Loaded {name} from disk")
        except Exception as e:
            print(f"Error loading models: {e}")

# --- Enhanced Feature Computation ---
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
    """Log to both CSV and database with error handling"""
    trade_id = None

    # Database insert
    if db is not None:
        try:
            trade_id = db.insert_trade(features)
        except Exception as e:
            print(f"Database insert error: {e}")

    # CSV logging - only selected columns for consistency
    try:
        csv_data = {col: features.get(col, 0) for col in TRADE_CSV_COLUMNS}
        pd.DataFrame([csv_data]).to_csv(
            TRADE_LOG,
            mode='a',
            header=not os.path.exists(TRADE_LOG),
            index=False,
            columns=TRADE_CSV_COLUMNS  # Ensure column order
        )
    except Exception as e:
        print(f"CSV logging error: {e}")

    return trade_id


# Add this function near the top of your kafka_consumer_detect.py file, after imports

def convert_numpy_types(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj


# Then update the log_anomaly function to use this helper:
def log_anomaly(trade_id, features, method, model_details=None):
    """Log anomaly with error handling"""
    try:
        # Convert any numpy types in model_details
        if model_details:
            model_details = convert_numpy_types(model_details)

        # Determine severity
        severity = 'medium'
        if alert_manager is not None:
            severity = alert_manager.determine_severity(features)

        # Database insert
        if db is not None and trade_id is not None:
            try:
                anomaly_id = db.insert_anomaly(trade_id, features, method, model_details, severity)
            except Exception as e:
                print(f"Database anomaly insert error: {e}")

        # Send alert
        if alert_manager is not None:
            anomaly_data = features.copy()
            anomaly_data['anomaly_type'] = method
            if model_details:
                anomaly_data['model_votes'] = model_details

            try:
                alert_manager.send_alert(anomaly_data, severity)
            except Exception as e:
                print(f"Alert sending error: {e}")

        # CSV logging - only selected columns
        csv_data = {
            'timestamp': features.get('timestamp', ''),
            'price': features.get('price', 0),
            'quantity': features.get('quantity', 0),
            'z_score': features.get('z_score', 0),
            'anomaly_type': method + "_injected" if features.get('injected') else method,
            'model_votes': json.dumps(model_details) if model_details else ''
        }

        pd.DataFrame([csv_data]).to_csv(
            ANOMALY_LOG,
            mode='a',
            header=not os.path.exists(ANOMALY_LOG),
            index=False,
            columns=ANOMALY_CSV_COLUMNS
        )
    except Exception as e:
        print(f"Error logging anomaly: {e}")

# --- Initialize with error handling ---
print("🚀 Starting Enhanced Kafka Consumer...")

# Initialize ensemble
try:
    ensemble = AnomalyEnsemble()
    ensemble.load(MODEL_DIR)
except Exception as e:
    print(f"Error initializing ensemble: {e}")
    ensemble = AnomalyEnsemble()

buffer_for_training = []
counter = 0
last_save_time = time.time()

# Initialize Kafka Consumer with retry logic
consumer = None
retry_count = 0
max_retries = 5

while retry_count < max_retries and not shutdown_flag:
    try:
        consumer = KafkaConsumer(
            KAFKA_TOPIC,
            bootstrap_servers=KAFKA_BROKER,
            auto_offset_reset='latest',
            value_deserializer=lambda x: json.loads(x.decode('utf-8')),
            consumer_timeout_ms=1000,  # 1 second timeout for checking shutdown
            session_timeout_ms=30000,
            heartbeat_interval_ms=10000,
            max_poll_records=10,
            enable_auto_commit=True,
            auto_commit_interval_ms=5000
        )
        print(f"✅ Connected to Kafka broker: {KAFKA_BROKER}")
        break
    except Exception as e:
        retry_count += 1
        print(f"❌ Failed to connect to Kafka (attempt {retry_count}/{max_retries}): {e}")
        if retry_count < max_retries:
            time.sleep(5)
        else:
            print("Failed to connect to Kafka after max retries. Exiting.")
            sys.exit(1)

# Rolling history storage
history_df = pd.DataFrame()

# Print initialization status
print(f"📊 Models in ensemble: {list(ensemble.models.keys())}")
if db:
    print(f"💾 Database: {db.db_path}")
if alert_manager:
    enabled_channels = [k for k, v in alert_manager.config.items() if v.get('enabled', False)]
    print(f"🔔 Alerts: Configured channels - {enabled_channels}")

# --- Main loop with error handling ---
print("Starting main processing loop...")

while not shutdown_flag:
    try:
        # Poll for messages
        messages = consumer.poll(timeout_ms=1000)

        if not messages:
            continue

        for topic_partition, records in messages.items():
            for message in records:
                try:
                    trade = message.value
                    features, history_df = compute_features(trade, history_df)

                    print(f"[{features['timestamp']}] Price: ${features['price']:.2f} | "
                          f"Z: {features['z_score']:.2f} | Vol: {features['quantity']:.4f}")

                    # Log trade and get ID
                    trade_id = log_trade(features)

                    if len(history_df) < ROLLING_WINDOW:
                        continue

                    # Track if this is an injected anomaly (ground truth)
                    is_actual_anomaly = bool(features.get('injected', False))
                    if is_actual_anomaly and evaluator is not None:
                        evaluator.add_ground_truth(features['timestamp'], True)

                    # Statistical anomaly detection
                    statistical_anomalies = []
                    statistical_detected = False

                    if abs(features['z_score']) > Z_THRESHOLD:
                        statistical_anomalies.append("z_score")
                        statistical_detected = True
                    if abs(features.get('vwap_deviation', 0)) > 2.0:
                        statistical_anomalies.append("vwap")
                        statistical_detected = True
                    if features.get('volume_spike', 1) > 3.0:
                        statistical_anomalies.append("volume")
                        statistical_detected = True

                    # Record detection for evaluation
                    if evaluator is not None:
                        if statistical_detected:
                            print(f"📊 Statistical anomalies: {', '.join(statistical_anomalies)}")
                            log_anomaly(trade_id, features, "_".join(statistical_anomalies))

                            confidence = max(
                                abs(features['z_score']) / Z_THRESHOLD,
                                abs(features.get('vwap_deviation', 0)) / 2.0,
                                features.get('volume_spike', 1) / 3.0
                            ) / 2

                            evaluator.record_detection(
                                features['timestamp'],
                                'statistical',
                                True,
                                min(confidence, 1.0),
                                is_actual_anomaly
                            )
                        else:
                            evaluator.record_detection(
                                features['timestamp'],
                                'statistical',
                                False,
                                0.0,
                                is_actual_anomaly
                            )

                    # ML-based detection
                    feature_vector = np.array([
                        features.get('z_score', 0),
                        features.get('price_change_pct', 0),
                        features.get('time_gap_sec', 0),
                        features.get('volume_ratio', 1),
                        features.get('buy_pressure', 0.5),
                        features.get('vwap_deviation', 0),
                        features.get('price_velocity', 0),
                        features.get('volume_spike', 1)
                    ])

                    buffer_for_training.append(feature_vector)
                    counter += 1

                    # ML detection
                    if len(buffer_for_training) > ROLLING_WINDOW:
                        recent_data = np.array(buffer_for_training[-ROLLING_WINDOW:])

                        # Retrain periodically
                        if counter % RETRAIN_INTERVAL == 0:
                            print("🔄 Retraining ensemble models...")
                            ensemble.fit(recent_data)
                            ensemble.save(MODEL_DIR)

                            # Generate evaluation report
                            if evaluator is not None:
                                print("\n📊 Current Model Performance:")
                                print(evaluator.generate_report())
                                evaluator.save_summary()

                                # Save to database
                                if db is not None:
                                    metrics = evaluator.get_metrics()
                                    for model_name, model_metrics in metrics.items():
                                        try:
                                            db.save_model_performance(model_name, model_metrics)
                                        except Exception as e:
                                            print(f"Error saving model performance: {e}")

                        if ensemble.is_fitted:
                            # Get predictions
                            ensemble_pred = ensemble.predict(feature_vector)

                            if ensemble_pred == -1:
                                model_predictions = ensemble.get_model_predictions(feature_vector)

                                # Calculate confidence
                                anomaly_votes = sum(1 for pred in model_predictions.values() if pred == -1)
                                confidence = anomaly_votes / len(model_predictions) if model_predictions else 0

                                print(f"🤖 ML Ensemble Anomaly Detected!")
                                print(f"   Model votes: {model_predictions}")
                                print(f"   Confidence: {confidence:.2f}")

                                # Log anomaly
                                log_anomaly(trade_id, features, "ensemble_ml", model_predictions)

                                # Record for evaluation
                                if evaluator is not None:
                                    evaluator.record_detection(
                                        features['timestamp'],
                                        'ensemble_ml',
                                        True,
                                        confidence,
                                        is_actual_anomaly
                                    )

                    # Generate plots periodically
                    if counter % 1000 == 0:
                        if evaluator is not None:
                            print("\n📈 Generating performance plots...")
                            try:
                                evaluator.plot_performance(f"evaluation_plots_{counter}.png")
                            except Exception as e:
                                print(f"Error generating plots: {e}")

                        # Print database statistics
                        if db is not None:
                            try:
                                stats = db.get_statistics(24)
                                print(f"\nDatabase statistics (last 24h):")
                                print(f"  Total trades: {stats['total_trades']}")
                                print(f"  Total anomalies: {stats['total_anomalies']}")
                                print(f"  Anomaly rate: {stats['anomaly_rate']:.2%}")
                            except Exception as e:
                                print(f"Error getting database stats: {e}")

                    # Periodic cleanup of old data
                    if time.time() - last_save_time > 3600:  # Every hour
                        if db is not None:
                            try:
                                db.cleanup_old_data(days_to_keep=7)
                                print("🧹 Cleaned up old data")
                            except Exception as e:
                                print(f"Error during cleanup: {e}")
                        last_save_time = time.time()

                    if SLEEP:
                        time.sleep(0.1)

                except Exception as e:
                    print(f"Error processing message: {e}")
                    traceback.print_exc()
                    continue

    except Exception as e:
        if shutdown_flag:
            break
        print(f"Error in main loop: {e}")
        traceback.print_exc()
        time.sleep(1)  # Brief pause before retrying

# Cleanup
print("\n🧹 Cleaning up...")
if consumer is not None:
    consumer.close()
print("✅ Consumer closed successfully")
print("👋 Goodbye!")