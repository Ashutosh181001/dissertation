"""
Updated BTC/USDT anomaly detection dashboard.

This file is based on the provided dashboard code and updates the
database statistics display logic.  Database statistics are now shown
in the main area via a Streamlit fragment to ensure they persist
across partial refreshes and appear regardless of the selected chart
type.  Logs are still collected internally for debugging but are not
rendered in the UI.
"""

import os
from datetime import datetime, timedelta
import time
from typing import Dict, List, Optional

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import json

# Attempt to import the optional database manager.  If not available,
# default to CSV-only mode.
try:
    from database_manager import DatabaseManager  # type: ignore
    db = DatabaseManager()
    USE_DATABASE = True
except Exception:
    USE_DATABASE = False
    db = None

# File constants for CSV fallbacks
TRADE_LOG = "trades.csv"
ANOMALY_LOG = "anomalies.csv"

# Default refresh interval for the Live tab (in seconds).  This value is
# also used for other tabs when no explicit refresh interval is provided.
DEFAULT_LIVE_REFRESH = 5
# Default refresh interval for non-live tabs.  You can adjust this to reduce
# update frequency for longer windows if desired.
DEFAULT_TAB_REFRESH = DEFAULT_LIVE_REFRESH

# Time interval configuration
TIME_INTERVALS: Dict[str, Dict[str, Optional[int]]] = {
    "Live": {"minutes": 10, "refresh": DEFAULT_LIVE_REFRESH, "candle_interval": "1min"},
    "15m": {"minutes": 15, "refresh": 30, "candle_interval": "1min"},
    "1h": {"minutes": 60, "refresh": 60, "candle_interval": "1min"},
    "4h": {"minutes": 240, "refresh": 120, "candle_interval": "5min"},
    "1D": {"minutes": 1440, "refresh": 300, "candle_interval": "15min"},
    "1W": {"minutes": 10080, "refresh": 600, "candle_interval": "1h"},
}

# Maximum number of trades for fallback
MAX_BUFFER = 400


@st.cache_data(ttl=10)  # Reduced TTL for more frequent updates
def cached_load_trades(minutes: int, timestamp_key: str) -> pd.DataFrame:
    """Cached version of load_trades to reduce database hits"""
    logs = []
    # For live data (minutes <= 10), always allow fallback
    return load_trades(minutes, logs, allow_fallback=(minutes <= 15))

@st.cache_data(ttl=10)  # Reduced TTL for more frequent updates
def cached_load_anomalies(minutes: int, anomaly_types: List[str], timestamp_key: str) -> pd.DataFrame:
    """Cached version of load_anomalies"""
    logs = []
    return load_anomalies(minutes, logs, anomaly_types)

@st.cache_data(ttl=60)
def create_zscore_chart_cached(trades_df_hash: str, timestamps: List[str], z_scores: List[float]) -> go.Figure:
    """Cached Z-score chart creation"""
    z_fig = go.Figure()
    z_fig.add_trace(go.Scatter(
        x=timestamps,
        y=z_scores,
        mode='lines',
        name='Z-Score',
        line=dict(color='#2ca02c', width=2)
    ))
    z_fig.add_hline(y=3, line_dash="dash", line_color="red", annotation_text="Upper", annotation_position="top right")
    z_fig.add_hline(y=-3, line_dash="dash", line_color="red", annotation_text="Lower", annotation_position="bottom right")
    z_fig.add_hline(y=0, line_dash="dot", line_color="gray")
    z_fig.update_layout(height=300, template='plotly_dark', xaxis_title='', yaxis_title='Z-Score')
    return z_fig

def log_message(logs: List[str], message: str) -> None:
    """Internal logging helper.  Does not render to UI."""
    timestamp = datetime.utcnow().strftime("%H:%M:%S")
    logs.append(f"[{timestamp}] {message}")


def load_trades(minutes: int, logs: List[str], allow_fallback: bool = True) -> pd.DataFrame:
    """Load trades from database or CSV within the last `minutes`.

    If `allow_fallback` is True and no data is found for the requested
    window, the function will attempt to load a fixed number of recent
    trades to avoid empty displays.
    """
    df: pd.DataFrame = pd.DataFrame()

    # Try database first
    if USE_DATABASE and db is not None:
        try:
            # Pass the actual minutes to the database for efficient querying
            df = db.get_recent_trades(minutes=minutes, limit=None)
            if df is not None and not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'], errors="coerce")
                for col in ['price', 'quantity', 'z_score', 'rolling_mean', 'rolling_std', 'price_change_pct',
                            'time_gap_sec']:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                df.sort_values('timestamp', inplace=True)
                log_message(logs, f"Loaded {len(df)} trades from database")
                return df
        except Exception as e:
            log_message(logs, f"Database trade load error: {e}; falling back to CSV")

    # Try CSV fallback
    cutoff = datetime.utcnow() - timedelta(minutes=minutes)
    if os.path.exists(TRADE_LOG):
        try:
            trades = pd.read_csv(TRADE_LOG)
            if not trades.empty:
                trades['timestamp'] = pd.to_datetime(trades['timestamp'], errors="coerce")
                trades = trades[trades['timestamp'] >= cutoff]
                for col in ['price', 'quantity', 'z_score', 'rolling_mean', 'rolling_std', 'price_change_pct',
                            'time_gap_sec']:
                    if col in trades.columns:
                        trades[col] = pd.to_numeric(trades[col], errors='coerce')
                trades.sort_values('timestamp', inplace=True)
                log_message(logs, f"Loaded {len(trades)} trades from CSV")
                return trades
        except Exception as e:
            log_message(logs, f"CSV trade load error: {e}")

    # Fallback to recent trades if no data found and fallback is allowed
    if allow_fallback:
        if USE_DATABASE and db is not None:
            try:
                # Get recent trades without time restriction, but with limit
                fallback = db.get_recent_trades(minutes=None, limit=MAX_BUFFER)
                if fallback is not None and not fallback.empty:
                    fallback['timestamp'] = pd.to_datetime(fallback['timestamp'], errors="coerce")
                    for col in ['price', 'quantity', 'z_score', 'rolling_mean', 'rolling_std', 'price_change_pct',
                                'time_gap_sec']:
                        if col in fallback.columns:
                            fallback[col] = pd.to_numeric(fallback[col], errors='coerce')
                    fallback.sort_values('timestamp', inplace=True)
                    log_message(logs, f"Fallback: loaded {len(fallback)} recent trades from database")
                    return fallback
            except Exception as e:
                log_message(logs, f"Database fallback error: {e}")

        # CSV fallback
        if os.path.exists(TRADE_LOG):
            try:
                trades = pd.read_csv(TRADE_LOG)
                if not trades.empty:
                    trades['timestamp'] = pd.to_datetime(trades['timestamp'], errors="coerce")
                    for col in ['price', 'quantity', 'z_score', 'rolling_mean', 'rolling_std', 'price_change_pct',
                                'time_gap_sec']:
                        if col in trades.columns:
                            trades[col] = pd.to_numeric(trades[col], errors='coerce')
                    trades.sort_values('timestamp', inplace=True)
                    trades = trades.tail(MAX_BUFFER).reset_index(drop=True)
                    log_message(logs, f"Fallback: loaded {len(trades)} trades from CSV tail")
                    return trades
            except Exception as e:
                log_message(logs, f"CSV fallback error: {e}")

    log_message(logs, "No trades available")
    return df


def load_anomalies(minutes: int, logs: List[str], anomaly_types: Optional[List[str]] = None) -> pd.DataFrame:
    """Load anomalies from database or CSV within the last `minutes`."""
    cutoff = datetime.utcnow() - timedelta(minutes=minutes)
    df: pd.DataFrame = pd.DataFrame()
    if USE_DATABASE and db is not None:
        try:
            df = db.get_recent_anomalies(minutes=minutes, anomaly_types=anomaly_types)
            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'], errors="coerce")
                df = df[df['timestamp'] >= cutoff]
                for col in ['price', 'z_score', 'quantity']:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                if anomaly_types:
                    df = df[df['anomaly_type'].isin(anomaly_types)]
                log_message(logs, f"Loaded {len(df)} anomalies from database")
                return df
        except Exception as e:
            log_message(logs, f"Database anomaly load error: {e}; falling back to CSV")
    if os.path.exists(ANOMALY_LOG):
        try:
            anomalies = pd.read_csv(ANOMALY_LOG)
            if not anomalies.empty:
                anomalies['timestamp'] = pd.to_datetime(anomalies['timestamp'], errors="coerce")
                anomalies = anomalies[anomalies['timestamp'] >= cutoff]
                for col in ['price', 'z_score', 'quantity']:
                    if col in anomalies.columns:
                        anomalies[col] = pd.to_numeric(anomalies[col], errors='coerce')
                if anomaly_types:
                    anomalies = anomalies[anomalies['anomaly_type'].isin(anomaly_types)]
                log_message(logs, f"Loaded {len(anomalies)} anomalies from CSV")
                return anomalies
        except Exception as e:
            log_message(logs, f"CSV anomaly load error: {e}")
    log_message(logs, "No anomalies available")
    return df


def aggregate_to_candlesticks(trades_df: pd.DataFrame, interval: str) -> pd.DataFrame:
    """Aggregate trades into OHLCV candlesticks."""
    if trades_df.empty:
        return pd.DataFrame()
    df = trades_df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp').sort_index()
    try:
        ohlc = df['price'].resample(interval).agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'})
        volume = df['quantity'].resample(interval).sum()
        volume.name = 'volume'
        trade_count = df['price'].resample(interval).count()
        trade_count.name = 'trade_count'
        csticks = pd.concat([ohlc, volume, trade_count], axis=1)
        csticks = csticks[csticks['trade_count'] > 0].dropna().reset_index()
        return csticks
    except Exception:
        return pd.DataFrame()


def load_performance_metrics() -> Dict:
    """Load performance metrics from evaluation_summary.json if present."""
    try:
        if os.path.exists("evaluation_summary.json"):
            with open("evaluation_summary.json", 'r') as f:
                return json.load(f)
    except Exception:
        pass
    return {}


def load_database_stats() -> Dict:
    """Load database statistics for the last 24 hours."""
    if USE_DATABASE and db is not None:
        try:
            return db.get_statistics(hours=24)
        except Exception:
            pass
    return {}


def create_metrics_row(trades_df: pd.DataFrame, anomalies_df: pd.DataFrame, interval_name: str) -> None:
    """Render a row of key metrics."""
    if trades_df.empty:
        st.warning("No trade data available for metrics calculation")
        return
    col1, col2, col3, col4, col5 = st.columns(5)
    try:
        current_price = float(trades_df.iloc[-1]['price'])
        first_price = float(trades_df.iloc[0]['price']) if len(trades_df) > 0 else current_price
        price_change = ((current_price - first_price) / first_price) * 100 if first_price != 0 else 0.0
        with col1:
            st.metric("Current Price", f"${current_price:,.2f}", f"{price_change:+.2f}%")
        high_price = float(trades_df['price'].max())
        low_price = float(trades_df['price'].min())
        with col2:
            st.metric(f"{interval_name} High", f"${high_price:,.2f}")
        with col3:
            st.metric(f"{interval_name} Low", f"${low_price:,.2f}")
        # Volume calculation
        if 'quantity' in trades_df.columns:
            total_volume = float(trades_df['quantity'].sum())
        elif 'volume' in trades_df.columns:
            total_volume = float(trades_df['volume'].sum())
        else:
            total_volume = 0.0
        with col4:
            st.metric(f"{interval_name} Volume", f"{total_volume:.4f} BTC")
        anomaly_count = len(anomalies_df) if not anomalies_df.empty else 0
        with col5:
            st.metric("Anomalies", anomaly_count)
    except Exception as e:
        st.error(f"Error calculating metrics: {e}")


def create_candlestick_chart(candlesticks: pd.DataFrame, trades_df: pd.DataFrame, anomalies_df: pd.DataFrame,
                             show_indicators: Dict[str, bool], interval_name: str) -> go.Figure:
    """Create a candlestick chart with optional moving average and volume."""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3], subplot_titles=('Price', 'Volume'))
    if not candlesticks.empty:
        valid = candlesticks.dropna(subset=['open', 'high', 'low', 'close'])
        if not valid.empty:
            fig.add_trace(go.Candlestick(x=valid['timestamp'], open=valid['open'], high=valid['high'], low=valid['low'], close=valid['close'], name='Price', increasing=dict(line=dict(color='#26a69a', width=1)), decreasing=dict(line=dict(color='#ef5350', width=1))), row=1, col=1)
            if show_indicators.get('volume', False):
                colors = ['#26a69a' if row['close'] >= row['open'] else '#ef5350' for _, row in valid.iterrows()]
                fig.add_trace(go.Bar(x=valid['timestamp'], y=valid['volume'], marker_color=colors, name='Volume', opacity=0.7), row=2, col=1)
    else:
        if not trades_df.empty:
            fig.add_trace(go.Scatter(x=trades_df['timestamp'], y=trades_df['price'], mode='lines', name='Price', line=dict(color='#f0b90b', width=2)), row=1, col=1)
    # Moving average indicator
    if show_indicators.get('rolling_mean') and not trades_df.empty and 'rolling_mean' in trades_df.columns:
        rm_data = trades_df[['timestamp', 'rolling_mean']].dropna()
        if not rm_data.empty:
            fig.add_trace(go.Scatter(x=rm_data['timestamp'], y=rm_data['rolling_mean'], name='MA', line=dict(color='#ffa726', width=2), opacity=0.8), row=1, col=1)
    # Anomaly markers
    if not anomalies_df.empty:
        fig.add_trace(go.Scatter(x=anomalies_df['timestamp'], y=anomalies_df['price'], mode='markers', name='Anomalies', marker=dict(symbol='triangle-up', size=12, color='#ff1744', line=dict(width=2, color='white')), text=[f"Type: {t}<br>Z: {z:.2f}" for t, z in zip(anomalies_df['anomaly_type'], anomalies_df['z_score'])], hoverinfo='text+x+y'), row=1, col=1)
    # Layout
    fig.update_layout(template='plotly_dark', height=700, showlegend=True, hovermode='x unified', margin=dict(l=50, r=50, t=40, b=40), xaxis_rangeslider_visible=False, plot_bgcolor='#0e1117', paper_bgcolor='#0e1117', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    fig.update_xaxes(gridcolor='rgba(128,128,128,0.2)', showgrid=True)
    fig.update_yaxes(gridcolor='rgba(128,128,128,0.2)', showgrid=True, side='right', title_text="Price (USDT)", row=1, col=1)
    fig.update_yaxes(gridcolor='rgba(128,128,128,0.2)', showgrid=True, side='right', title_text="Volume (BTC)", row=2, col=1)
    return fig


def create_line_chart(trades_df: pd.DataFrame, anomalies_df: pd.DataFrame, show_indicators: Dict[str, bool]) -> go.Figure:
    """Create a line chart with optional moving average and anomalies."""
    fig = go.Figure()
    if not trades_df.empty:
        fig.add_trace(go.Scatter(x=trades_df['timestamp'], y=trades_df['price'], mode='lines', name='Price', line=dict(color='#f0b90b', width=2)))
        if show_indicators.get('rolling_mean') and 'rolling_mean' in trades_df.columns:
            rm_data = trades_df[['timestamp', 'rolling_mean']].dropna()
            if not rm_data.empty:
                fig.add_trace(go.Scatter(x=rm_data['timestamp'], y=rm_data['rolling_mean'], mode='lines', name='MA', line=dict(color='#ffa726', width=2, dash='dash')))
    if not anomalies_df.empty:
        fig.add_trace(go.Scatter(x=anomalies_df['timestamp'], y=anomalies_df['price'], mode='markers', name='Anomalies', marker=dict(symbol='triangle-up', size=12, color='#ff1744'), text=[f"Type: {t}<br>Z: {z:.2f}" for t, z in zip(anomalies_df['anomaly_type'], anomalies_df['z_score'])], hoverinfo='text+x+y'))
    fig.update_layout(template='plotly_dark', height=600, showlegend=True, hovermode='x unified', margin=dict(l=50, r=50, t=30, b=20), plot_bgcolor='#0e1117', paper_bgcolor='#0e1117', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), xaxis=dict(gridcolor='rgba(128,128,128,0.2)'), yaxis=dict(gridcolor='rgba(128,128,128,0.2)', side='right', title='Price (USDT)'))
    return fig


def display_interval(interval_name: str, interval_config: Dict[str, Optional[int]], chart_type: str,
                     show_indicators: Dict[str, bool], selected_anomaly_types: List[str], logs: List[str]) -> None:
    """Display a single tab's content."""
    minutes = interval_config.get("minutes", 60) or 60
    # Use the specific refresh interval or fall back to DEFAULT_TAB_REFRESH
    refresh_interval = interval_config.get("refresh") or DEFAULT_TAB_REFRESH
    allow_fallback = (interval_name == "Live")

    frag_decorator = st.fragment(run_every=refresh_interval)

    @frag_decorator
    def render_interval() -> None:
        # Create cache key based on current time and refresh interval
        current_time = datetime.utcnow()

        # For live data, use shorter cache periods
        if interval_name == "Live":
            cache_key_str = current_time.strftime("%Y%m%d_%H%M%S")[:-1]  # Remove last digit for 10-second buckets
        elif refresh_interval <= 60:
            cache_key_str = current_time.strftime("%Y%m%d_%H%M")  # 1-minute buckets
        else:
            cache_minutes = refresh_interval // 60
            cache_key = current_time.replace(second=0, microsecond=0)
            cache_key = cache_key.replace(minute=(cache_key.minute // cache_minutes) * cache_minutes)
            cache_key_str = cache_key.strftime("%Y%m%d_%H%M")

        # Load data with caching
        try:
            trades_df = cached_load_trades(minutes, cache_key_str)
            anomalies_df = cached_load_anomalies(minutes, selected_anomaly_types, cache_key_str)
        except Exception:
            # Fallback to non-cached if caching fails
            trades_df = load_trades(minutes, logs, allow_fallback=allow_fallback)
            anomalies_df = load_anomalies(minutes, logs, selected_anomaly_types)

        if trades_df.empty:
            st.warning(f"No trade data available for {interval_name}")
            st.info("Ensure that your data source is running and contains data for the selected time window.")
            return

        create_metrics_row(trades_df, anomalies_df, interval_name)
        st.markdown("<br>", unsafe_allow_html=True)

        try:
            if chart_type == "Candlestick":
                candlesticks = aggregate_to_candlesticks(trades_df,
                                                         interval_config.get("candle_interval", "1min") or "1min")
                fig_main = create_candlestick_chart(candlesticks, trades_df, anomalies_df, show_indicators,
                                                    interval_name)
            else:
                fig_main = create_line_chart(trades_df, anomalies_df, show_indicators)

            # Add unique key for main chart
            st.plotly_chart(fig_main, use_container_width=True, key=f"main_chart_{interval_name}_{chart_type}")

        except Exception as e:
            st.error(f"Error creating chart: {e}")
            st.info("Displaying basic price line chart as fallback.")
            import plotly.express as px
            fallback_fig = px.line(trades_df, x='timestamp', y='price', title=f'BTC/USDT Price - {interval_name}')
            fallback_fig.update_layout(template='plotly_dark')
            # Add unique key for fallback chart
            st.plotly_chart(fallback_fig, use_container_width=True, key=f"fallback_chart_{interval_name}")

        # Z-score chart with caching and unique key
        st.subheader("📉 Z-Score Over Time")
        if 'z_score' in trades_df.columns and not trades_df['z_score'].isna().all():
            try:
                # Prepare data for cached chart creation
                clean_df = trades_df[['timestamp', 'z_score']].dropna()
                timestamps = clean_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S').tolist()
                z_scores = clean_df['z_score'].tolist()
                df_hash = f"{len(clean_df)}_{interval_name}_{cache_key_str}"

                z_fig = create_zscore_chart_cached(df_hash, timestamps, z_scores)
                st.plotly_chart(z_fig, use_container_width=True, key=f"zscore_chart_{interval_name}")
            except Exception:
                # Fallback to non-cached version
                z_fig = go.Figure()
                z_fig.add_trace(
                    go.Scatter(x=trades_df['timestamp'], y=trades_df['z_score'], mode='lines', name='Z-Score',
                               line=dict(color='#2ca02c', width=2)))
                z_fig.add_hline(y=3, line_dash="dash", line_color="red", annotation_text="Upper",
                                annotation_position="top right")
                z_fig.add_hline(y=-3, line_dash="dash", line_color="red", annotation_text="Lower",
                                annotation_position="bottom right")
                z_fig.add_hline(y=0, line_dash="dot", line_color="gray")
                z_fig.update_layout(height=300, template='plotly_dark', xaxis_title='', yaxis_title='Z-Score')
                st.plotly_chart(z_fig, use_container_width=True, key=f"zscore_chart_{interval_name}")
        else:
            st.info("Z-score data not available.")

        # Anomalies table (limit to 20 rows for performance)
        if not anomalies_df.empty:
            st.subheader("🚨 Recent Anomalies")
            display_df = anomalies_df.copy().head(20)  # Limit rows for performance
            display_df['timestamp'] = display_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
            if 'price' in display_df.columns:
                display_df['price'] = display_df['price'].apply(lambda x: f"${x:,.2f}")
            if 'z_score' in display_df.columns:
                display_df['z_score'] = display_df['z_score'].apply(lambda x: f"{x:.2f}")
            cols = ['timestamp', 'anomaly_type', 'price', 'z_score']
            display_df = display_df[cols]
            display_df.columns = ['Timestamp', 'Type', 'Price', 'Z-Score']
            st.dataframe(display_df, use_container_width=True)

        # Last update
        st.caption(f"Last updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")

    render_interval()


def main() -> None:
    """Run the Streamlit application."""
    st.set_page_config(page_title="BTC/USDT Unified Dashboard", layout="wide", page_icon="📊", initial_sidebar_state="collapsed")
    st.markdown("""
        <style>
        .stApp { background-color: #0b0e11; }
        .stTabs [data-baseweb="tab-list"] { gap: 2px; background-color: #1e2329; padding: 4px; border-radius: 4px; }
        .stTabs [data-baseweb="tab"] { background-color: transparent; color: #848e9c; padding: 8px 16px; font-weight: 500; }
        .stTabs [aria-selected="true"] { background-color: #2b3139; color: #f0b90b; border-radius: 4px; }
        div[data-testid="metric-container"] { background-color: #1e2329; border: 1px solid #2b3139; padding: 10px 15px; border-radius: 8px; margin: 5px 0; }
        </style>
    """, unsafe_allow_html=True)
    st.markdown("# 📊 BTC/USDT Unified Dashboard")
    logs: List[str] = []
    with st.sidebar:
        st.header("⚙️ Settings")
        chart_type = st.radio("Chart Type", ["Candlestick", "Line"], index=0)
        st.subheader("Indicators")
        show_ma = st.checkbox("Moving Average", value=True)
        show_volume = st.checkbox("Show Volume", value=True)
        st.subheader("Anomaly Filters")
        all_anoms_df = load_anomalies(10080, logs, None)
        if not all_anoms_df.empty and 'anomaly_type' in all_anoms_df.columns:
            all_types = sorted(all_anoms_df['anomaly_type'].dropna().unique())
            selected_types = st.multiselect("Select anomaly types", all_types, default=all_types)
        else:
            selected_types = []
        st.subheader("Additional Information")
        show_perf = st.checkbox("Show performance metrics", value=True)
        # Note: database statistics will be displayed automatically below
    tabs = st.tabs(list(TIME_INTERVALS.keys()))
    for i, (interval_name, interval_config) in enumerate(TIME_INTERVALS.items()):
        with tabs[i]:
            display_interval(interval_name, interval_config, chart_type, {"rolling_mean": show_ma, "volume": show_volume}, selected_types, logs)
    if show_perf:
        st.header("📊 Model Performance Metrics")
        perf_metrics = load_performance_metrics()
        if perf_metrics and 'metrics' in perf_metrics:
            rows = []
            for model, m in perf_metrics['metrics'].items():
                rows.append({'Model': model, 'Precision': f"{m.get('precision', 0):.2%}", 'Recall': f"{m.get('recall', 0):.2%}", 'F1 Score': f"{m.get('f1_score', 0):.2%}", 'Total Detections': m.get('total_detections', 0)})
            if rows:
                metrics_df = pd.DataFrame(rows)
                st.dataframe(metrics_df, use_container_width=True)
                if 'generated_at' in perf_metrics:
                    st.caption(f"Metrics generated at {perf_metrics['generated_at']}")
        else:
            st.info("Performance metrics not available yet.")
    # Database statistics fragment displayed below
    @st.fragment(run_every=60)  # Refresh every minute
    def show_database_stats() -> None:
        if USE_DATABASE:
            st.header("💾 Database Statistics (24h)")
            stats = load_database_stats()
            if stats:
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric("Total Trades", f"{stats.get('total_trades', 0):,}")
                    st.metric("Average Price", f"${stats.get('avg_price', 0):,.2f}")
                with c2:
                    st.metric("Total Anomalies", f"{stats.get('total_anomalies', 0):,}")
                    st.metric("Max |Z-Score|", f"{stats.get('max_z_score', 0):.2f}")
                with c3:
                    st.metric("Anomaly Rate", f"{stats.get('anomaly_rate', 0) * 100:.2f}%")
                    st.metric("Max Volume Spike", f"{stats.get('max_volume_spike', 0):.1f}x")

                if 'anomaly_breakdown' in stats and stats['anomaly_breakdown']:
                    breakdown_df = pd.DataFrame.from_dict(stats['anomaly_breakdown'], orient='index',
                                                          columns=['Count']).reset_index()
                    breakdown_df.columns = ['Type', 'Count']
                    import plotly.express as px
                    fig_breakdown = px.bar(breakdown_df, x='Type', y='Count', title='Anomalies by Type (24h)')
                    fig_breakdown.update_layout(template='plotly_dark')
                    # Add unique key for breakdown chart
                    st.plotly_chart(fig_breakdown, use_container_width=True, key="anomaly_breakdown_chart")
            else:
                st.info("Database statistics not available.")

        show_database_stats()


if __name__ == "__main__":
    main()