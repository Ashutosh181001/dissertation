"""
Database Manager for Crypto Anomaly Detection System

Provides efficient storage and retrieval using SQLite with proper indexing.
"""

import sqlite3
import pandas as pd
import json
from datetime import datetime, timedelta
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple
import logging
import os

logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    Manages SQLite database for trades and anomalies with optimized queries.
    """
    
    def __init__(self, db_path: str = 'trading_anomalies.db'):
        self.db_path = db_path
        self._init_database()
        
    def _init_database(self):
        """Initialize database tables and indexes"""
        with self.get_connection() as conn:
            # Trades table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    price REAL NOT NULL,
                    quantity REAL NOT NULL,
                    is_buyer_maker INTEGER,
                    z_score REAL,
                    rolling_mean REAL,
                    rolling_std REAL,
                    price_change_pct REAL,
                    time_gap_sec REAL,
                    volume_ratio REAL,
                    buy_pressure REAL,
                    vwap REAL,
                    vwap_deviation REAL,
                    price_velocity REAL,
                    volume_spike REAL,
                    -- Contextual features
                    hour INTEGER,
                    day_of_week INTEGER,
                    trading_session TEXT,
                    is_weekend INTEGER,
                    -- Metadata
                    injected INTEGER DEFAULT 0,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Anomalies table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS anomalies (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trade_id INTEGER,
                    timestamp DATETIME NOT NULL,
                    anomaly_type TEXT NOT NULL,
                    severity TEXT,
                    price REAL,
                    z_score REAL,
                    price_change_pct REAL,
                    volume_spike REAL,
                    vwap_deviation REAL,
                    model_votes TEXT,  -- JSON string
                    confidence REAL,
                    alerted INTEGER DEFAULT 0,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (trade_id) REFERENCES trades(id)
                )
            ''')
            
            # Model performance table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS model_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    model_name TEXT NOT NULL,
                    precision_score REAL,
                    recall_score REAL,
                    f1_score REAL,
                    true_positives INTEGER,
                    false_positives INTEGER,
                    false_negatives INTEGER,
                    true_negatives INTEGER,
                    total_predictions INTEGER,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Alert history table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS alert_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    anomaly_id INTEGER,
                    timestamp DATETIME NOT NULL,
                    severity TEXT,
                    channels TEXT,  -- JSON array
                    message TEXT,
                    success INTEGER DEFAULT 1,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (anomaly_id) REFERENCES anomalies(id)
                )
            ''')
            
            # Create indexes for performance
            conn.execute('CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(timestamp)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_trades_zscore ON trades(z_score)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_anomalies_timestamp ON anomalies(timestamp)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_anomalies_type ON anomalies(anomaly_type)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_model_perf_timestamp ON model_performance(timestamp)')
            
            conn.commit()
            
    @contextmanager
    def get_connection(self):
        """Context manager for database connections"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Enable column access by name
        try:
            yield conn
        finally:
            conn.close()
            
    def insert_trade(self, features: Dict) -> int:
        """
        Insert a trade record and return its ID.
        
        Parameters:
        -----------
        features: Dict
            Trade features from compute_features()
            
        Returns:
        --------
        int: The inserted trade ID
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Prepare values
            values = (
                features['timestamp'],
                features['price'],
                features['quantity'],
                features.get('is_buyer_maker', 0),
                features.get('z_score', 0),
                features.get('rolling_mean', 0),
                features.get('rolling_std', 0),
                features.get('price_change_pct', 0),
                features.get('time_gap_sec', 0),
                features.get('volume_ratio', 1),
                features.get('buy_pressure', 0.5),
                features.get('vwap', features['price']),
                features.get('vwap_deviation', 0),
                features.get('price_velocity', 0),
                features.get('volume_spike', 1),
                features.get('hour', 0),
                features.get('day_of_week', 0),
                features.get('trading_session', 'unknown'),
                features.get('is_weekend', 0),
                features.get('injected', 0)
            )
            
            cursor.execute('''
                INSERT INTO trades (
                    timestamp, price, quantity, is_buyer_maker,
                    z_score, rolling_mean, rolling_std,
                    price_change_pct, time_gap_sec, volume_ratio,
                    buy_pressure, vwap, vwap_deviation,
                    price_velocity, volume_spike,
                    hour, day_of_week, trading_session, is_weekend,
                    injected
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', values)
            
            conn.commit()
            return cursor.lastrowid
    
    def insert_anomaly(self, trade_id: int, features: Dict, anomaly_type: str, 
                      model_details: Optional[Dict] = None, severity: str = 'medium') -> int:
        """
        Insert an anomaly record.
        
        Parameters:
        -----------
        trade_id: int
            ID of the associated trade
        features: Dict
            Trade features
        anomaly_type: str
            Type of anomaly detected
        model_details: Dict, optional
            Model voting details
        severity: str
            Anomaly severity level
            
        Returns:
        --------
        int: The inserted anomaly ID
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            values = (
                trade_id,
                features['timestamp'],
                anomaly_type,
                severity,
                features['price'],
                features.get('z_score', 0),
                features.get('price_change_pct', 0),
                features.get('volume_spike', 1),
                features.get('vwap_deviation', 0),
                json.dumps(model_details) if model_details else None,
                None  # confidence can be calculated from model_details
            )
            
            cursor.execute('''
                INSERT INTO anomalies (
                    trade_id, timestamp, anomaly_type, severity,
                    price, z_score, price_change_pct, volume_spike,
                    vwap_deviation, model_votes, confidence
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', values)
            
            conn.commit()
            return cursor.lastrowid
    
    def get_recent_trades(self, minutes: int = 60, limit: Optional[int] = None) -> pd.DataFrame:
        """Get recent trades within the specified time window"""
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        
        query = '''
            SELECT * FROM trades
            WHERE timestamp >= ?
            ORDER BY timestamp DESC
        '''
        
        if limit:
            query += f' LIMIT {limit}'
            
        with self.get_connection() as conn:
            return pd.read_sql_query(query, conn, params=(cutoff_time.isoformat(),))
    
    def get_recent_anomalies(self, minutes: int = 60, anomaly_types: Optional[List[str]] = None) -> pd.DataFrame:
        """Get recent anomalies within the specified time window"""
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        
        query = '''
            SELECT a.*, t.quantity, t.rolling_mean, t.rolling_std
            FROM anomalies a
            JOIN trades t ON a.trade_id = t.id
            WHERE a.timestamp >= ?
        '''
        
        params = [cutoff_time.isoformat()]
        
        if anomaly_types:
            placeholders = ','.join(['?' for _ in anomaly_types])
            query += f' AND a.anomaly_type IN ({placeholders})'
            params.extend(anomaly_types)
            
        query += ' ORDER BY a.timestamp DESC'
        
        with self.get_connection() as conn:
            return pd.read_sql_query(query, conn, params=params)
    
    def save_model_performance(self, model_name: str, metrics: Dict):
        """Save model performance metrics"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO model_performance (
                    timestamp, model_name, precision_score, recall_score, f1_score,
                    true_positives, false_positives, false_negatives, true_negatives,
                    total_predictions
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now().isoformat(),
                model_name,
                metrics.get('precision', 0),
                metrics.get('recall', 0),
                metrics.get('f1_score', 0),
                metrics.get('true_positives', 0),
                metrics.get('false_positives', 0),
                metrics.get('false_negatives', 0),
                metrics.get('true_negatives', 0),
                metrics.get('total_predictions', 0)
            ))
            
            conn.commit()
    
    def get_model_performance_history(self, model_name: str, days: int = 7) -> pd.DataFrame:
        """Get model performance history"""
        cutoff_time = datetime.now() - timedelta(days=days)
        
        query = '''
            SELECT * FROM model_performance
            WHERE model_name = ? AND timestamp >= ?
            ORDER BY timestamp
        '''
        
        with self.get_connection() as conn:
            return pd.read_sql_query(query, conn, params=(model_name, cutoff_time.isoformat()))
    
    def get_statistics(self, hours: int = 24) -> Dict:
        """Get database statistics for the specified time period"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        with self.get_connection() as conn:
            # Total trades
            total_trades = conn.execute(
                'SELECT COUNT(*) FROM trades WHERE timestamp >= ?',
                (cutoff_time.isoformat(),)
            ).fetchone()[0]
            
            # Total anomalies
            total_anomalies = conn.execute(
                'SELECT COUNT(*) FROM anomalies WHERE timestamp >= ?',
                (cutoff_time.isoformat(),)
            ).fetchone()[0]
            
            # Anomalies by type
            anomaly_breakdown = conn.execute('''
                SELECT anomaly_type, COUNT(*) as count
                FROM anomalies
                WHERE timestamp >= ?
                GROUP BY anomaly_type
                ORDER BY count DESC
            ''', (cutoff_time.isoformat(),)).fetchall()
            
            # Average metrics
            avg_metrics = conn.execute('''
                SELECT 
                    AVG(price) as avg_price,
                    AVG(quantity) as avg_volume,
                    AVG(ABS(z_score)) as avg_z_score,
                    MAX(ABS(z_score)) as max_z_score,
                    AVG(volume_spike) as avg_volume_spike,
                    MAX(volume_spike) as max_volume_spike
                FROM trades
                WHERE timestamp >= ?
            ''', (cutoff_time.isoformat(),)).fetchone()
            
            return {
                'period_hours': hours,
                'total_trades': total_trades,
                'total_anomalies': total_anomalies,
                'anomaly_rate': total_anomalies / total_trades if total_trades > 0 else 0,
                'anomaly_breakdown': dict(anomaly_breakdown),
                'avg_price': avg_metrics['avg_price'],
                'avg_volume': avg_metrics['avg_volume'],
                'avg_z_score': avg_metrics['avg_z_score'],
                'max_z_score': avg_metrics['max_z_score'],
                'avg_volume_spike': avg_metrics['avg_volume_spike'],
                'max_volume_spike': avg_metrics['max_volume_spike']
            }
    
    def cleanup_old_data(self, days_to_keep: int = 30):
        """Remove data older than specified days"""
        cutoff_time = datetime.now() - timedelta(days=days_to_keep)
        
        with self.get_connection() as conn:
            # Delete old trades
            trades_deleted = conn.execute(
                'DELETE FROM trades WHERE timestamp < ?',
                (cutoff_time.isoformat(),)
            ).rowcount
            
            # Delete orphaned anomalies
            anomalies_deleted = conn.execute(
                'DELETE FROM anomalies WHERE timestamp < ?',
                (cutoff_time.isoformat(),)
            ).rowcount
            
            # Delete old performance metrics
            perf_deleted = conn.execute(
                'DELETE FROM model_performance WHERE timestamp < ?',
                (cutoff_time.isoformat(),)
            ).rowcount
            
            conn.commit()
            
            logger.info(f"Cleanup complete: {trades_deleted} trades, "
                       f"{anomalies_deleted} anomalies, {perf_deleted} performance records deleted")
            
            # Vacuum to reclaim space
            conn.execute('VACUUM')
    
    def export_to_csv(self, output_dir: str = 'exports'):
        """Export database tables to CSV files"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        tables = ['trades', 'anomalies', 'model_performance', 'alert_history']
        
        with self.get_connection() as conn:
            for table in tables:
                df = pd.read_sql_query(f'SELECT * FROM {table}', conn)
                output_file = os.path.join(output_dir, f'{table}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
                df.to_csv(output_file, index=False)
                print(f"Exported {len(df)} records from {table} to {output_file}")


# Integration helper
def migrate_from_csv_to_db(trades_csv: str = 'trades.csv', anomalies_csv: str = 'anomalies.csv'):
    """Migrate existing CSV data to database"""
    db = DatabaseManager()
    
    # Migrate trades
    if os.path.exists(trades_csv):
        print(f"Migrating trades from {trades_csv}...")
        trades_df = pd.read_csv(trades_csv)
        
        for _, row in trades_df.iterrows():
            features = row.to_dict()
            db.insert_trade(features)
        
        print(f"Migrated {len(trades_df)} trades")
    
    # Migrate anomalies
    if os.path.exists(anomalies_csv):
        print(f"Migrating anomalies from {anomalies_csv}...")
        anomalies_df = pd.read_csv(anomalies_csv)
        
        # Note: We won't have trade_id references, so we'll use NULL
        for _, row in anomalies_df.iterrows():
            features = row.to_dict()
            db.insert_anomaly(None, features, row['anomaly_type'])
        
        print(f"Migrated {len(anomalies_df)} anomalies")


# Example usage
if __name__ == "__main__":
    # Initialize database
    db = DatabaseManager()
    
    # Example: Insert a trade
    example_features = {
        'timestamp': datetime.now().isoformat(),
        'price': 50000.0,
        'quantity': 1.5,
        'z_score': 2.5,
        'rolling_mean': 49500.0,
        'volume_spike': 3.2
    }
    
    trade_id = db.insert_trade(example_features)
    print(f"Inserted trade with ID: {trade_id}")
    
    # Example: Insert an anomaly
    anomaly_id = db.insert_anomaly(
        trade_id, 
        example_features, 
        'z_score_volume',
        {'isolation_forest': -1, 'lof': 1}
    )
    print(f"Inserted anomaly with ID: {anomaly_id}")
    
    # Get statistics
    stats = db.get_statistics(24)
    print(f"\nDatabase statistics (last 24h):")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Example: Get recent anomalies
    recent_anomalies = db.get_recent_anomalies(60)
    print(f"\nRecent anomalies: {len(recent_anomalies)}")