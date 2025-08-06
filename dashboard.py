"""
Combined BTC/USDT anomaly detection dashboard.

This script merges the reliability and accuracy of the original anomaly
detection dashboard with the modern interface and features from the newer
prototype.  It continues to prioritise database-backed data sources
wherever possible (with a CSV fallback) and logs which source was used.

Features:
  - Tabs for multiple time windows (Live, 15 minutes, 1 hour, 4 hours,
    1 day, 1 week) with automatic refresh for the Live view.
  - Choice of candlestick or simple line charts with optional moving
    averages (rolling_mean) and volume bars.
  - Anomaly markers and a table of recent anomalies with filtering by
    anomaly type.
  - Key metrics including current price, interval high/low, volume and
    anomaly count.
  - Optional performance metrics and database statistics from the
    original dashboard when available.
  - Lightweight logging of data sources (database vs CSV) presented in
    the sidebar for transparency.

The goal of this file is to provide a stable and accurate dashboard
while incorporating the improved user interface of the newer prototype.
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
# default to CSV-only mode.  We catch broad exceptions here because the
# user might not have the module installed or configured.
try:
    from database_manager import DatabaseManager  # type: ignore
    db = DatabaseManager()
    USE_DATABASE = True
except Exception:
    USE_DATABASE = False
    db = None

# File constants used when falling back to CSV.  These names mirror
# those used in the legacy dashboard so existing logs can still be
# consumed.
TRADE_LOG = "trades.csv"
ANOMALY_LOG = "anomalies.csv"

# Default refresh interval for the Live tab (in seconds).  Other tabs
# refresh on user interaction only.
DEFAULT_LIVE_REFRESH = 5

# Time interval configuration.  Each entry defines the number of
# minutes of history to load and, optionally, a resampling interval for
# candlestick generation.  The Live view automatically refreshes.
TIME_INTERVALS: Dict[str, Dict[str, Optional[int]]] = {
    "Live": {"minutes": 10, "refresh": DEFAULT_LIVE_REFRESH, "candle_interval": "1min"},
    "15m": {"minutes": 15, "refresh": None, "candle_interval": "1min"},
    "1h": {"minutes": 60, "refresh": None, "candle_interval": "1min"},
    "4h": {"minutes": 240, "refresh": None, "candle_interval": "5min"},
    "1D": {"minutes": 1440, "refresh": None, "candle_interval": "15min"},
    "1W": {"minutes": 10080, "refresh": None, "candle_interval": "1H"},
}

# Maximum number of trades to load as a fallback when the requested
# interval contains no data.  This mirrors the behaviour of the
# original dashboard which always shows the most recent trades.  If
# there are fewer than this number of trades available, the entire
# dataset is returned.
MAX_BUFFER = 400


def log_message(logs: List[str], message: str) -> None:
    """Append a message to the log list with a timestamp.

    Parameters
    ----------
    logs : list
        A list of strings to which the message will be appended.
    message : str
        The message to append.
    """
    timestamp = datetime.utcnow().strftime("%H:%M:%S")
    logs.append(f"[{timestamp}] {message}")


def load_trades(minutes: int, logs: List[str]) -> pd.DataFrame:
    """Load trades from the database or CSV within the last `minutes`.

    This function attempts to load trades from the configured database
    first.  If that fails, it falls back to reading from a CSV file.
    Each numeric column is converted to a numeric dtype to avoid
    mixed-type issues.  The returned DataFrame is sorted by timestamp.

    Parameters
    ----------
    minutes : int
        Number of minutes of history to load.
    logs : list
        A list used for logging which data source was used.

    Returns
    -------
    pd.DataFrame
        DataFrame containing trade data with columns at least
        [timestamp, price, quantity] and optionally z_score,
        rolling_mean, rolling_std, price_change_pct, time_gap_sec.
    """
    # Use database if available
    cutoff = datetime.utcnow() - timedelta(minutes=minutes)
    df: pd.DataFrame = pd.DataFrame()
    if USE_DATABASE and db is not None:
        try:
            df = db.get_recent_trades(minutes=minutes, limit=None)
            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'], errors="coerce")
                # Filter by time on database side just in case
                df = df[df['timestamp'] >= cutoff]
                # Convert numeric columns
                for col in [
                    'price', 'quantity', 'z_score', 'rolling_mean',
                    'rolling_std', 'price_change_pct', 'time_gap_sec'
                ]:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                df.sort_values('timestamp', inplace=True)
                log_message(logs, f"Loaded {len(df)} trades from database")
                return df
        except Exception as e:
            # Log error and fall through to CSV
            log_message(logs, f"Database trade load error: {e}; falling back to CSV")

    # Fallback to CSV
    if os.path.exists(TRADE_LOG):
        try:
            trades = pd.read_csv(TRADE_LOG)
            if not trades.empty:
                trades['timestamp'] = pd.to_datetime(trades['timestamp'], errors="coerce")
                trades = trades[trades['timestamp'] >= cutoff]
                # Convert numeric columns
                for col in [
                    'price', 'quantity', 'z_score', 'rolling_mean',
                    'rolling_std', 'price_change_pct', 'time_gap_sec'
                ]:
                    if col in trades.columns:
                        trades[col] = pd.to_numeric(trades[col], errors='coerce')
                trades.sort_values('timestamp', inplace=True)
                log_message(logs, f"Loaded {len(trades)} trades from CSV")
                return trades
        except Exception as e:
            log_message(logs, f"CSV trade load error: {e}")
    # If nothing found, attempt a broader fallback similar to the legacy dashboard.
    # Try retrieving a fixed number of recent trades from the database or CSV.
    if USE_DATABASE and db is not None:
        try:
            fallback_df = db.get_recent_trades(minutes=60, limit=MAX_BUFFER)
            if fallback_df is not None and not fallback_df.empty:
                fallback_df['timestamp'] = pd.to_datetime(fallback_df['timestamp'], errors="coerce")
                for col in [
                    'price', 'quantity', 'z_score', 'rolling_mean',
                    'rolling_std', 'price_change_pct', 'time_gap_sec'
                ]:
                    if col in fallback_df.columns:
                        fallback_df[col] = pd.to_numeric(fallback_df[col], errors='coerce')
                fallback_df.sort_values('timestamp', inplace=True)
                log_message(logs, f"Fallback: loaded {len(fallback_df)} trades from database")
                return fallback_df
        except Exception as e:
            log_message(logs, f"Database fallback error: {e}")
    # CSV fallback for recent trades
    if os.path.exists(TRADE_LOG):
        try:
            trades = pd.read_csv(TRADE_LOG)
            if not trades.empty:
                trades['timestamp'] = pd.to_datetime(trades['timestamp'], errors="coerce")
                for col in [
                    'price', 'quantity', 'z_score', 'rolling_mean',
                    'rolling_std', 'price_change_pct', 'time_gap_sec'
                ]:
                    if col in trades.columns:
                        trades[col] = pd.to_numeric(trades[col], errors='coerce')
                trades.sort_values('timestamp', inplace=True)
                trades = trades.tail(MAX_BUFFER).reset_index(drop=True)
                log_message(logs, f"Fallback: loaded {len(trades)} trades from CSV tail")
                return trades
        except Exception as e:
            log_message(logs, f"CSV fallback error: {e}")
    # If still no data, log and return empty
    log_message(logs, "No trades available")
    return df


def load_anomalies(minutes: int, logs: List[str], anomaly_types: Optional[List[str]] = None) -> pd.DataFrame:
    """Load anomalies from the database or CSV within the last `minutes`.

    Parameters
    ----------
    minutes : int
        Number of minutes of history to load.
    logs : list
        A list used for logging which data source was used.
    anomaly_types : list, optional
        List of anomaly types to filter on.  If None, all anomalies
        within the window are returned.

    Returns
    -------
    pd.DataFrame
        DataFrame of anomalies with at least [timestamp, price,
        anomaly_type, z_score].
    """
    cutoff = datetime.utcnow() - timedelta(minutes=minutes)
    df: pd.DataFrame = pd.DataFrame()
    if USE_DATABASE and db is not None:
        try:
            df = db.get_recent_anomalies(minutes=minutes, anomaly_types=anomaly_types)
            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'], errors="coerce")
                df = df[df['timestamp'] >= cutoff]
                # Convert numeric columns
                for col in ['price', 'z_score', 'quantity']:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                if anomaly_types:
                    df = df[df['anomaly_type'].isin(anomaly_types)]
                log_message(logs, f"Loaded {len(df)} anomalies from database")
                return df
        except Exception as e:
            log_message(logs, f"Database anomaly load error: {e}; falling back to CSV")
    # Fallback to CSV
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
    """Aggregate trades into OHLCV candlesticks.

    This function resamples the trade DataFrame using pandas resample
    semantics.  Only candles with at least one trade are returned.

    Parameters
    ----------
    trades_df : pd.DataFrame
        Trades DataFrame with at least columns ['timestamp', 'price',
        'quantity'].
    interval : str
        Resampling rule (e.g., '1min', '5min', '15min', '1H').

    Returns
    -------
    pd.DataFrame
        Candlestick data with columns ['timestamp', 'open', 'high',
        'low', 'close', 'volume', 'trade_count'].
    """
    if trades_df.empty:
        return pd.DataFrame()
    # Ensure timestamp is datetime and set as index
    df = trades_df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')
    df = df.sort_index()
    try:
        ohlc = df['price'].resample(interval).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last'
        })
        volume = df['quantity'].resample(interval).sum()
        volume.name = 'volume'
        trade_count = df['price'].resample(interval).count()
        trade_count.name = 'trade_count'
        candlesticks = pd.concat([ohlc, volume, trade_count], axis=1)
        candlesticks = candlesticks[candlesticks['trade_count'] > 0]
        candlesticks = candlesticks.dropna()
        return candlesticks.reset_index()
    except Exception:
        # On error, return empty to avoid crashing
        return pd.DataFrame()


def load_performance_metrics() -> Dict:
    """Load latest performance metrics from evaluation summary JSON.

    Returns an empty dict if the file is missing or malformed.
    """
    try:
        if os.path.exists("evaluation_summary.json"):
            with open("evaluation_summary.json", 'r') as f:
                return json.load(f)
    except Exception:
        pass
    return {}


def load_database_stats() -> Dict:
    """Load database statistics via the database manager.

    Returns an empty dict when the database is unavailable or an error occurs.
    """
    if USE_DATABASE and db is not None:
        try:
            return db.get_statistics(hours=24)
        except Exception:
            pass
    return {}


def create_metrics_row(trades_df: pd.DataFrame, anomalies_df: pd.DataFrame, interval_name: str) -> None:
    """Render a row of key metrics into the Streamlit app.

    This function calculates the current price, percentage change over
    the interval, high/low values, total volume, and anomaly count.
    It gracefully handles missing data.
    """
    if trades_df.empty:
        st.warning("No trade data available for metrics calculation")
        return
    # Use five columns for metrics similar to the new prototype
    col1, col2, col3, col4, col5 = st.columns(5)
    try:
        current_price = float(trades_df.iloc[-1]['price'])
        # Price change relative to the first trade in the window
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
        total_volume = float(trades_df['quantity'].sum())
        with col4:
            st.metric(f"{interval_name} Volume", f"{total_volume:.4f} BTC")
        anomaly_count = len(anomalies_df) if not anomalies_df.empty else 0
        with col5:
            st.metric("Anomalies", anomaly_count)
    except Exception as e:
        st.error(f"Error calculating metrics: {e}")


def create_candlestick_chart(candlesticks: pd.DataFrame, trades_df: pd.DataFrame,
                             anomalies_df: pd.DataFrame, show_indicators: Dict[str, bool],
                             interval_name: str) -> go.Figure:
    """Create a candlestick chart with volume and optional indicators.

    The chart uses a two-row layout: candlesticks on top and volume
    below.  Anomaly markers are overlaid on the price chart.  A
    rolling_mean indicator can be drawn if requested.
    """
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.7, 0.3],
        subplot_titles=('Price', 'Volume')
    )
    # Draw candlesticks if available; otherwise fall back to simple line
    if not candlesticks.empty:
        valid = candlesticks.dropna(subset=['open', 'high', 'low', 'close'])
        if not valid.empty:
            fig.add_trace(
                go.Candlestick(
                    x=valid['timestamp'],
                    open=valid['open'],
                    high=valid['high'],
                    low=valid['low'],
                    close=valid['close'],
                    name='Price',
                    increasing=dict(line=dict(color='#26a69a', width=1)),
                    decreasing=dict(line=dict(color='#ef5350', width=1))
                ),
                row=1, col=1
            )
            # Volume bars (only displayed if the user has enabled volume in show_indicators)
            if show_indicators.get('volume', False):
                colors = [
                    '#26a69a' if row['close'] >= row['open'] else '#ef5350'
                    for _, row in valid.iterrows()
                ]
                fig.add_trace(
                    go.Bar(
                        x=valid['timestamp'],
                        y=valid['volume'],
                        marker_color=colors,
                        name='Volume',
                        opacity=0.7
                    ),
                    row=2, col=1
                )
    else:
        # If no candlesticks, use simple line from trades
        if not trades_df.empty:
            fig.add_trace(
                go.Scatter(
                    x=trades_df['timestamp'],
                    y=trades_df['price'],
                    mode='lines',
                    name='Price',
                    line=dict(color='#f0b90b', width=2)
                ),
                row=1, col=1
            )
    # Rolling mean indicator
    if show_indicators.get("rolling_mean") and not trades_df.empty and 'rolling_mean' in trades_df.columns:
        rolling_data = trades_df[['timestamp', 'rolling_mean']].dropna()
        if not rolling_data.empty:
            fig.add_trace(
                go.Scatter(
                    x=rolling_data['timestamp'],
                    y=rolling_data['rolling_mean'],
                    name='MA',
                    line=dict(color='#ffa726', width=2),
                    opacity=0.8
                ),
                row=1, col=1
            )
    # Anomaly markers
    if not anomalies_df.empty:
        fig.add_trace(
            go.Scatter(
                x=anomalies_df['timestamp'],
                y=anomalies_df['price'],
                mode='markers',
                name='Anomalies',
                marker=dict(symbol='triangle-up', size=12, color='#ff1744', line=dict(width=2, color='white')),
                text=[f"Type: {t}<br>Z: {z:.2f}" for t, z in zip(anomalies_df['anomaly_type'], anomalies_df['z_score'])],
                hoverinfo='text+x+y'
            ),
            row=1, col=1
        )
    # Layout adjustments
    fig.update_layout(
        template='plotly_dark',
        height=700,
        showlegend=True,
        hovermode='x unified',
        margin=dict(l=50, r=50, t=40, b=40),
        xaxis_rangeslider_visible=False,
        plot_bgcolor='#0e1117',
        paper_bgcolor='#0e1117',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    # Gridlines and axis labels
    fig.update_xaxes(gridcolor='rgba(128,128,128,0.2)', showgrid=True)
    fig.update_yaxes(gridcolor='rgba(128,128,128,0.2)', showgrid=True, side='right', title_text="Price (USDT)", row=1, col=1)
    fig.update_yaxes(gridcolor='rgba(128,128,128,0.2)', showgrid=True, side='right', title_text="Volume (BTC)", row=2, col=1)
    return fig


def create_line_chart(trades_df: pd.DataFrame, anomalies_df: pd.DataFrame, show_indicators: Dict[str, bool]) -> go.Figure:
    """Create a simple line chart with optional moving average and anomalies."""
    fig = go.Figure()
    if not trades_df.empty:
        fig.add_trace(
            go.Scatter(
                x=trades_df['timestamp'],
                y=trades_df['price'],
                mode='lines',
                name='Price',
                line=dict(color='#f0b90b', width=2)
            )
        )
        if show_indicators.get("rolling_mean") and 'rolling_mean' in trades_df.columns:
            rolling_data = trades_df[['timestamp', 'rolling_mean']].dropna()
            if not rolling_data.empty:
                fig.add_trace(
                    go.Scatter(
                        x=rolling_data['timestamp'],
                        y=rolling_data['rolling_mean'],
                        mode='lines',
                        name='MA',
                        line=dict(color='#ffa726', width=2, dash='dash')
                    )
                )
    if not anomalies_df.empty:
        fig.add_trace(
            go.Scatter(
                x=anomalies_df['timestamp'],
                y=anomalies_df['price'],
                mode='markers',
                name='Anomalies',
                marker=dict(symbol='triangle-up', size=12, color='#ff1744'),
                text=[f"Type: {t}<br>Z: {z:.2f}" for t, z in zip(anomalies_df['anomaly_type'], anomalies_df['z_score'])],
                hoverinfo='text+x+y'
            )
        )
    fig.update_layout(
        template='plotly_dark',
        height=600,
        showlegend=True,
        hovermode='x unified',
        margin=dict(l=50, r=50, t=30, b=20),
        plot_bgcolor='#0e1117',
        paper_bgcolor='#0e1117',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(gridcolor='rgba(128,128,128,0.2)'),
        yaxis=dict(gridcolor='rgba(128,128,128,0.2)', side='right', title='Price (USDT)')
    )
    return fig


def display_interval(interval_name: str, interval_config: Dict[str, Optional[int]], chart_type: str,
                    show_indicators: Dict[str, bool], selected_anomaly_types: List[str], logs: List[str]) -> None:
    """Display metrics, charts, anomalies and other information for a given interval.

    All visible content for an interval is wrapped in a Streamlit fragment to
    support timed auto-refresh (for the Live tab) without scrolling the
    entire page back to the top.  When run_every is None the fragment
    behaves normally (no automatic refresh).
    """
    minutes = interval_config.get("minutes", 60) or 60
    refresh_interval = interval_config.get("refresh")

    # Define a fragment that encapsulates all UI elements for this tab.  The
    # fragment will automatically rerun every `refresh_interval` seconds
    # (for Live) while preserving scroll position.
    if refresh_interval:
        frag_decorator = st.fragment(run_every=refresh_interval)
    else:
        frag_decorator = st.fragment

    @frag_decorator
    def render_interval() -> None:
        # Load trades and anomalies inside the fragment to update on refresh
        trades_df = load_trades(minutes, logs)
        anomalies_df = load_anomalies(minutes, logs, selected_anomaly_types)
        if trades_df.empty:
            st.warning(f"No trade data available for {interval_name}")
            st.info("Ensure that your data source is running and contains data for the selected time window.")
            return
        # Metrics row
        create_metrics_row(trades_df, anomalies_df, interval_name)
        st.markdown("<br>", unsafe_allow_html=True)
        # Main chart (candlestick or line)
        try:
            if chart_type == "Candlestick":
                candle_interval = interval_config.get("candle_interval", "1min") or "1min"
                candlesticks = aggregate_to_candlesticks(trades_df, candle_interval)
                fig_main = create_candlestick_chart(candlesticks, trades_df, anomalies_df, show_indicators, interval_name)
            else:
                fig_main = create_line_chart(trades_df, anomalies_df, show_indicators)
            st.plotly_chart(fig_main, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating chart: {e}")
            st.info("Displaying basic price line chart as fallback.")
            import plotly.express as px
            fallback_fig = px.line(trades_df, x='timestamp', y='price', title=f'BTC/USDT Price - {interval_name}')
            fallback_fig.update_layout(template='plotly_dark')
            st.plotly_chart(fallback_fig, use_container_width=True)
        # Additional charts: Z-score and Volume in two columns
        col_z, col_vol = st.columns(2)
        # Z-score chart
        with col_z:
            st.subheader("📉 Z-Score Over Time")
            if 'z_score' in trades_df.columns and not trades_df['z_score'].isna().all():
                z_fig = go.Figure()
                z_fig.add_trace(go.Scatter(
                    x=trades_df['timestamp'],
                    y=trades_df['z_score'],
                    mode='lines',
                    name='Z-Score',
                    line=dict(color='#2ca02c', width=2)
                ))
                # Threshold lines
                z_fig.add_hline(y=3, line_dash="dash", line_color="red", annotation_text="Upper", annotation_position="top right")
                z_fig.add_hline(y=-3, line_dash="dash", line_color="red", annotation_text="Lower", annotation_position="bottom right")
                z_fig.add_hline(y=0, line_dash="dot", line_color="gray")
                z_fig.update_layout(height=300, template='plotly_dark', xaxis_title='', yaxis_title='Z-Score')
                st.plotly_chart(z_fig, use_container_width=True)
            else:
                st.info("Z-score data not available.")
        # Volume chart
        with col_vol:
            if show_indicators.get('volume', False):
                st.subheader("📊 Trading Volume")
                if 'quantity' in trades_df.columns and not trades_df['quantity'].isna().all():
                    v_fig = go.Figure()
                    v_fig.add_trace(go.Bar(
                        x=trades_df['timestamp'],
                        y=trades_df['quantity'],
                        name='Volume',
                        marker_color='lightblue'
                    ))
                    v_fig.update_layout(height=300, template='plotly_dark', xaxis_title='', yaxis_title='Volume')
                    st.plotly_chart(v_fig, use_container_width=True)
                else:
                    st.info("Volume data not available.")
        # Anomalies table
        if not anomalies_df.empty:
            st.subheader("🚨 Recent Anomalies")
            display_df = anomalies_df.copy()
            display_df['timestamp'] = display_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
            if 'price' in display_df.columns:
                display_df['price'] = display_df['price'].apply(lambda x: f"${x:,.2f}")
            if 'z_score' in display_df.columns:
                display_df['z_score'] = display_df['z_score'].apply(lambda x: f"{x:.2f}")
            cols = ['timestamp', 'anomaly_type', 'price', 'z_score']
            display_df = display_df[cols]
            display_df.columns = ['Timestamp', 'Type', 'Price', 'Z-Score']
            display_df = display_df.head(20)
            st.dataframe(display_df, use_container_width=True)
        # Data information and last update
        with st.expander("📊 Data Information", expanded=False):
            c1, c2 = st.columns(2)
            with c1:
                st.write(f"**Trades loaded:** {len(trades_df)}")
                if not trades_df.empty:
                    st.write(f"**Time range:** {trades_df['timestamp'].min()} to {trades_df['timestamp'].max()}")
            with c2:
                st.write(f"**Anomalies:** {len(anomalies_df)}")
                if not anomalies_df.empty:
                    st.write(f"**Anomaly types:** {', '.join(anomalies_df['anomaly_type'].unique())}")
        st.caption(f"Last updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")
    # Call the fragment function to render the content
    render_interval()


def main() -> None:
    """Run the Streamlit dashboard application."""
    # Configure the Streamlit page
    st.set_page_config(
        page_title="BTC/USDT Unified Dashboard",
        layout="wide",
        page_icon="📊",
        initial_sidebar_state="collapsed"
    )
    # Apply custom CSS for a dark theme inspired by the new prototype
    st.markdown("""
        <style>
        .stApp { background-color: #0b0e11; }
        .stTabs [data-baseweb="tab-list"] { gap: 2px; background-color: #1e2329; padding: 4px; border-radius: 4px; }
        .stTabs [data-baseweb="tab"] { background-color: transparent; color: #848e9c; padding: 8px 16px; font-weight: 500; }
        .stTabs [aria-selected="true"] { background-color: #2b3139; color: #f0b90b; border-radius: 4px; }
        div[data-testid="metric-container"] { background-color: #1e2329; border: 1px solid #2b3139; padding: 10px 15px; border-radius: 8px; margin: 5px 0; }
        </style>
    """, unsafe_allow_html=True)
    # Title
    st.markdown("# 📊 BTC/USDT Unified Dashboard")
    # Sidebar controls
    logs: List[str] = []
    with st.sidebar:
        st.header("⚙️ Settings")
        # Chart type selector
        chart_type = st.radio("Chart Type", ["Candlestick", "Line"], index=0)
        # Indicators
        st.subheader("Indicators")
        show_ma = st.checkbox("Moving Average", value=True)
        # Volume indicator for candlesticks
        show_volume = st.checkbox("Show Volume", value=True)
        # Anomaly filtering
        st.subheader("Anomaly Filters")
        # Pre-load all anomaly types for the entire period (7 days)
        all_anoms_df = load_anomalies(10080, logs, None)
        if not all_anoms_df.empty and 'anomaly_type' in all_anoms_df.columns:
            all_types = sorted(all_anoms_df['anomaly_type'].dropna().unique())
            selected_types = st.multiselect("Select anomaly types", all_types, default=all_types)
        else:
            selected_types = []
        # Performance and DB stats toggles
        st.subheader("Additional Information")
        show_perf = st.checkbox("Show performance metrics", value=True)
        show_db_stats = st.checkbox("Show database statistics", value=USE_DATABASE)
        # Log display
        st.subheader("Logs")
        # Logs will be populated during processing; display a placeholder here
        log_area = st.empty()
    # Create tabs for each time interval
    tabs = st.tabs(list(TIME_INTERVALS.keys()))
    for i, (interval_name, interval_config) in enumerate(TIME_INTERVALS.items()):
        with tabs[i]:
            display_interval(
                interval_name,
                interval_config,
                chart_type,
                {"rolling_mean": show_ma, "volume": show_volume},
                selected_types,
                logs
            )
    # After rendering intervals, display performance metrics and DB stats
    if show_perf:
        st.header("📊 Model Performance Metrics")
        perf_metrics = load_performance_metrics()
        if perf_metrics and 'metrics' in perf_metrics:
            metrics_list = []
            for model, m in perf_metrics['metrics'].items():
                metrics_list.append({
                    'Model': model,
                    'Precision': f"{m.get('precision', 0):.2%}",
                    'Recall': f"{m.get('recall', 0):.2%}",
                    'F1 Score': f"{m.get('f1_score', 0):.2%}",
                    'Total Detections': m.get('total_detections', 0)
                })
            if metrics_list:
                metrics_df = pd.DataFrame(metrics_list)
                st.dataframe(metrics_df, use_container_width=True)
                if 'generated_at' in perf_metrics:
                    st.caption(f"Metrics generated at {perf_metrics['generated_at']}")
        else:
            st.info("Performance metrics not available yet.")
    if show_db_stats and USE_DATABASE:
        st.header("💾 Database Statistics (24h)")
        stats = load_database_stats()
        if stats:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Trades", f"{stats.get('total_trades', 0):,}")
                st.metric("Average Price", f"${stats.get('avg_price', 0):,.2f}")
            with col2:
                st.metric("Total Anomalies", f"{stats.get('total_anomalies', 0):,}")
                st.metric("Max |Z-Score|", f"{stats.get('max_z_score', 0):.2f}")
            with col3:
                st.metric("Anomaly Rate", f"{stats.get('anomaly_rate', 0) * 100:.2f}%")
                st.metric("Max Volume Spike", f"{stats.get('max_volume_spike', 0):.1f}x")
            # Anomaly breakdown bar chart
            if 'anomaly_breakdown' in stats and stats['anomaly_breakdown']:
                breakdown_df = pd.DataFrame.from_dict(stats['anomaly_breakdown'], orient='index', columns=['Count']).reset_index()
                breakdown_df.columns = ['Type', 'Count']
                import plotly.express as px
                fig_breakdown = px.bar(breakdown_df, x='Type', y='Count', title='Anomalies by Type (24h)')
                fig_breakdown.update_layout(template='plotly_dark')
                st.plotly_chart(fig_breakdown, use_container_width=True)
        else:
            st.info("Database statistics not available.")
    # Display logs collected during processing
    with st.sidebar:
        if logs:
            log_area.text("\n".join(logs))
        else:
            log_area.text("No logs generated yet.")


if __name__ == "__main__":
    main()