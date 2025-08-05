"""
Enhanced BTC/USDT anomaly detection dashboard with database support.

This dashboard provides real-time visualization of:
- Price movements with rolling mean
- Z-score trends
- Detected anomalies
- Performance metrics
- Database statistics
"""

import os
from datetime import datetime, timedelta
from typing import List

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import json

# Import database manager
try:
    from database_manager import DatabaseManager

    db = DatabaseManager()
    USE_DATABASE = True
except:
    USE_DATABASE = False
    print("Warning: Database manager not available, using CSV files only")

# -----------------------------------------------------------------------------
# Configuration constants
TRADE_LOG = "trades.csv"
ANOMALY_LOG = "anomalies.csv"
MAX_BUFFER = 400  # number of most recent trades to display
DEFAULT_REFRESH = 5  # default refresh interval in seconds


def load_recent_trades(max_buffer: int) -> pd.DataFrame:
    """Load the most recent trades from database or CSV."""
    # Try database first if available
    if USE_DATABASE:
        try:
            df = db.get_recent_trades(minutes=60, limit=max_buffer)
            if not df.empty:
                # Ensure timestamp is datetime
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                return df.sort_values('timestamp', ascending=True).tail(max_buffer)
        except Exception as e:
            st.sidebar.warning(f"Database read error: {e}")

    # Fallback to CSV
    if not os.path.exists(TRADE_LOG):
        return pd.DataFrame(columns=["timestamp", "price", "rolling_mean", "z_score", "quantity"])

    # Read trades CSV, skipping malformed rows if necessary
    try:
        trades = pd.read_csv(TRADE_LOG)
    except pd.errors.ParserError:
        # When the CSV has inconsistent columns (e.g. old and new formats mixed), skip bad lines
        trades = pd.read_csv(TRADE_LOG, engine='python', on_bad_lines='skip')
    if trades.empty:
        return trades

    trades["timestamp"] = pd.to_datetime(trades["timestamp"], errors="coerce")
    trades.dropna(subset=["timestamp"], inplace=True)

    # Convert numeric columns to proper types
    numeric_columns = ['price', 'quantity', 'z_score', 'rolling_mean', 'rolling_std',
                       'price_change_pct', 'time_gap_sec']
    for col in numeric_columns:
        if col in trades.columns:
            trades[col] = pd.to_numeric(trades[col], errors='coerce').fillna(0)

    return trades.tail(max_buffer).reset_index(drop=True)


def load_anomalies() -> pd.DataFrame:
    """Load anomaly events from database or CSV."""
    # Try database first if available
    if USE_DATABASE:
        try:
            df = db.get_recent_anomalies(minutes=60)
            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                return df.sort_values('timestamp', ascending=False)
        except Exception as e:
            st.sidebar.warning(f"Database anomaly read error: {e}")

    # Fallback to CSV
    if not os.path.exists(ANOMALY_LOG):
        return pd.DataFrame(columns=["timestamp", "price", "z_score", "quantity", "anomaly_type"])

    # Read anomalies CSV, skipping malformed rows if necessary
    try:
        anomalies = pd.read_csv(ANOMALY_LOG)
    except pd.errors.ParserError:
        anomalies = pd.read_csv(ANOMALY_LOG, engine='python', on_bad_lines='skip')
    if anomalies.empty:
        return anomalies

    anomalies["timestamp"] = pd.to_datetime(anomalies["timestamp"], errors="coerce")
    anomalies.dropna(subset=["timestamp"], inplace=True)

    # Convert numeric columns to proper types
    numeric_columns = ['price', 'quantity', 'z_score', 'volume_spike']
    for col in numeric_columns:
        if col in anomalies.columns:
            anomalies[col] = pd.to_numeric(anomalies[col], errors='coerce').fillna(0)

    return anomalies.reset_index(drop=True)


def get_anomaly_types(anomalies: pd.DataFrame) -> List[str]:
    """Return a sorted list of unique anomaly types."""
    if anomalies.empty or "anomaly_type" not in anomalies.columns:
        return []
    return sorted(anomalies["anomaly_type"].dropna().unique().tolist())


def load_performance_metrics() -> dict:
    """Load latest performance metrics from evaluation summary."""
    try:
        if os.path.exists("evaluation_summary.json"):
            with open("evaluation_summary.json", 'r') as f:
                return json.load(f)
    except:
        pass
    return {}


def load_database_stats() -> dict:
    """Load database statistics."""
    if USE_DATABASE:
        try:
            return db.get_statistics(hours=24)
        except:
            pass
    return {}


def main() -> None:
    """Entrypoint for the Streamlit dashboard."""
    # Configure the page
    st.set_page_config(
        page_title="BTC Anomaly Dashboard",
        layout="wide",
        page_icon="📊"
    )

    # Title and header
    st.title("📊 BTC/USDT Anomaly Detection Dashboard")
    st.markdown("Real-time monitoring of cryptocurrency trading anomalies")

    # Sidebar controls
    st.sidebar.header("⚙️ Settings")

    # Refresh rate control
    refresh_rate: int = int(
        st.sidebar.slider("Auto-refresh interval (seconds)", 1, 30, DEFAULT_REFRESH)
    )

    # Time window selection
    time_window = st.sidebar.selectbox(
        "Time window",
        ["Last 15 minutes", "Last 30 minutes", "Last 60 minutes", "Last 2 hours"],
        index=2
    )

    # Pre-load anomalies for filter
    anomalies_for_filter = load_anomalies()
    anomaly_types = get_anomaly_types(anomalies_for_filter)
    selected_types = st.sidebar.multiselect(
        "Anomaly types to show",
        anomaly_types,
        default=anomaly_types
    )

    # Display mode
    st.sidebar.header("📊 Display Options")
    show_volume = st.sidebar.checkbox("Show volume chart", value=True)
    show_metrics = st.sidebar.checkbox("Show performance metrics", value=True)
    show_db_stats = st.sidebar.checkbox("Show database statistics", value=USE_DATABASE)

    # Download section
    st.sidebar.header("💾 Downloads")

    # Initialize session state
    if "trades" not in st.session_state:
        st.session_state["trades"] = pd.DataFrame()
    if "anomalies" not in st.session_state:
        st.session_state["anomalies"] = pd.DataFrame()

    # Download buttons
    if not st.session_state["trades"].empty:
        csv = st.session_state["trades"].to_csv(index=False).encode("utf-8")
        st.sidebar.download_button(
            "📥 Download Trades CSV",
            csv,
            file_name=f"trades_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )

    if not st.session_state["anomalies"].empty:
        csv_anom = st.session_state["anomalies"].to_csv(index=False).encode("utf-8")
        st.sidebar.download_button(
            "📥 Download Anomalies CSV",
            csv_anom,
            file_name=f"anomalies_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )

    # -------------------------------------------------------------------------
    # Main dashboard fragment with auto-refresh
    @st.fragment(run_every=refresh_rate)
    def update_charts() -> None:
        # Load latest data
        trades = load_recent_trades(MAX_BUFFER)
        anomalies = load_anomalies()

        # Save to session state
        st.session_state.trades = trades
        st.session_state.anomalies = anomalies

        # Create columns for layout
        if show_metrics or show_db_stats:
            metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)

            # Display key metrics
            if not trades.empty:
                current_price = trades.iloc[-1]['price']
                price_change = ((current_price - trades.iloc[0]['price']) / trades.iloc[0]['price']) * 100
                avg_z_score = trades['z_score'].abs().mean() if 'z_score' in trades.columns else 0

                metrics_col1.metric(
                    "Current Price",
                    f"${current_price:,.2f}",
                    f"{price_change:+.2f}%"
                )

                metrics_col2.metric(
                    "Avg |Z-Score|",
                    f"{avg_z_score:.2f}"
                )

                # Anomaly count
                recent_anomaly_count = len(anomalies) if not anomalies.empty else 0
                metrics_col3.metric(
                    "Recent Anomalies",
                    recent_anomaly_count
                )

                # Anomaly rate
                if show_db_stats:
                    db_stats = load_database_stats()
                    if db_stats and db_stats.get('total_trades', 0) > 0:
                        anomaly_rate = db_stats.get('anomaly_rate', 0) * 100
                        metrics_col4.metric(
                            "Anomaly Rate (24h)",
                            f"{anomaly_rate:.2f}%"
                        )

        # Price chart with rolling mean
        st.subheader("📈 BTC/USDT Price with Rolling Mean")
        if not trades.empty:
            fig_price = go.Figure()

            # Add price line
            fig_price.add_trace(go.Scatter(
                x=trades['timestamp'],
                y=trades['price'],
                mode='lines',
                name='Price',
                line=dict(color='#1f77b4', width=2)
            ))

            # Add rolling mean if available
            if 'rolling_mean' in trades.columns:
                fig_price.add_trace(go.Scatter(
                    x=trades['timestamp'],
                    y=trades['rolling_mean'],
                    mode='lines',
                    name='Rolling Mean',
                    line=dict(color='#ff7f0e', width=2, dash='dash')
                ))

            # Add anomaly markers
            if not anomalies.empty and selected_types:
                filtered_anomalies = anomalies[anomalies['anomaly_type'].isin(selected_types)]
                if not filtered_anomalies.empty:
                    fig_price.add_trace(go.Scatter(
                        x=filtered_anomalies['timestamp'],
                        y=filtered_anomalies['price'],
                        mode='markers',
                        name='Anomalies',
                        marker=dict(
                            color='red',
                            size=10,
                            symbol='circle-open',
                            line=dict(width=2)
                        )
                    ))

            fig_price.update_layout(
                height=400,
                xaxis_tickformat='%H:%M:%S',
                template='plotly_white',
                hovermode='x unified'
            )
            st.plotly_chart(fig_price, use_container_width=True)
        else:
            st.info("No trade data available.")

        # Create two columns for Z-score and Volume
        col1, col2 = st.columns(2)

        with col1:
            # Z-score chart
            st.subheader("📉 Z-Score Over Time")
            if not trades.empty and "z_score" in trades.columns:
                fig_zscore = go.Figure()

                # Add z-score line
                fig_zscore.add_trace(go.Scatter(
                    x=trades['timestamp'],
                    y=trades['z_score'],
                    mode='lines',
                    name='Z-Score',
                    line=dict(color='#2ca02c', width=2)
                ))

                # Add threshold lines
                fig_zscore.add_hline(y=3, line_dash="dash", line_color="red",
                                     annotation_text="Upper Threshold")
                fig_zscore.add_hline(y=-3, line_dash="dash", line_color="red",
                                     annotation_text="Lower Threshold")
                fig_zscore.add_hline(y=0, line_dash="dot", line_color="gray")

                fig_zscore.update_layout(
                    height=300,
                    xaxis_tickformat='%H:%M:%S',
                    template='plotly_white'
                )
                st.plotly_chart(fig_zscore, use_container_width=True)
            else:
                st.info("Z-score data not available.")

        with col2:
            # Volume chart
            if show_volume:
                st.subheader("📊 Trading Volume")
                if not trades.empty and "quantity" in trades.columns:
                    fig_volume = go.Figure()

                    fig_volume.add_trace(go.Bar(
                        x=trades['timestamp'],
                        y=trades['quantity'],
                        name='Volume',
                        marker_color='lightblue'
                    ))

                    fig_volume.update_layout(
                        height=300,
                        xaxis_tickformat='%H:%M:%S',
                        template='plotly_white'
                    )
                    st.plotly_chart(fig_volume, use_container_width=True)
                else:
                    st.info("Volume data not available.")

        # Anomalies section
        st.subheader("🚨 Recent Anomalies")

        # Parse time window
        window_minutes = {
            "Last 15 minutes": 15,
            "Last 30 minutes": 30,
            "Last 60 minutes": 60,
            "Last 2 hours": 120
        }.get(time_window, 60)

        now = datetime.utcnow()
        recent_window = now - timedelta(minutes=window_minutes)

        if not anomalies.empty:
            recent_anomalies = anomalies[
                (anomalies["timestamp"] >= recent_window) &
                (anomalies["anomaly_type"].isin(selected_types))
                ]
        else:
            recent_anomalies = pd.DataFrame()

        if not recent_anomalies.empty:
            # Anomaly scatter plot
            fig_anom = px.scatter(
                recent_anomalies,
                x="timestamp",
                y="price",
                color="anomaly_type",
                symbol="anomaly_type",
                hover_data=["z_score", "quantity"],
                title=f"Anomalies ({time_window})",
                color_discrete_sequence=px.colors.qualitative.Set1
            )

            fig_anom.update_layout(
                height=350,
                xaxis_tickformat='%H:%M:%S',
                template='plotly_white'
            )
            st.plotly_chart(fig_anom, use_container_width=True)

            # Anomaly table
            st.subheader("📋 Anomaly Details")
            display_cols = ['timestamp', 'anomaly_type', 'price', 'z_score', 'volume_spike']
            available_cols = [col for col in display_cols if col in recent_anomalies.columns]

            # Format the dataframe for display
            display_df = recent_anomalies[available_cols].copy()
            display_df['timestamp'] = display_df['timestamp'].dt.strftime('%H:%M:%S')
            if 'price' in display_df.columns:
                display_df['price'] = display_df['price'].apply(lambda x: f"${x:,.2f}")
            if 'z_score' in display_df.columns:
                display_df['z_score'] = display_df['z_score'].apply(lambda x: f"{x:.2f}")
            if 'volume_spike' in display_df.columns:
                display_df['volume_spike'] = display_df['volume_spike'].apply(lambda x: f"{x:.1f}x")

            st.dataframe(display_df.tail(20), use_container_width=True)
        else:
            st.success(f"✅ No anomalies detected in {time_window.lower()}.")

        # Performance metrics section
        if show_metrics:
            st.subheader("📊 Model Performance Metrics")
            perf_metrics = load_performance_metrics()

            if perf_metrics and 'metrics' in perf_metrics:
                metrics_data = perf_metrics['metrics']

                # Create metrics table
                metrics_rows = []
                for model, metrics in metrics_data.items():
                    metrics_rows.append({
                        'Model': model,
                        'Precision': f"{metrics.get('precision', 0):.2%}",
                        'Recall': f"{metrics.get('recall', 0):.2%}",
                        'F1 Score': f"{metrics.get('f1_score', 0):.2%}",
                        'Total Detections': metrics.get('total_detections', 0)
                    })

                if metrics_rows:
                    metrics_df = pd.DataFrame(metrics_rows)
                    st.dataframe(metrics_df, use_container_width=True)

                    # Last updated
                    if 'generated_at' in perf_metrics:
                        st.caption(f"Last updated: {perf_metrics['generated_at']}")
            else:
                st.info("Performance metrics not yet available. They will appear after processing more trades.")

        # Database statistics
        if show_db_stats and USE_DATABASE:
            st.subheader("💾 Database Statistics (24h)")
            db_stats = load_database_stats()

            if db_stats:
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Total Trades", f"{db_stats.get('total_trades', 0):,}")
                    st.metric("Avg Price", f"${db_stats.get('avg_price', 0):,.2f}")

                with col2:
                    st.metric("Total Anomalies", f"{db_stats.get('total_anomalies', 0):,}")
                    st.metric("Max |Z-Score|", f"{db_stats.get('max_z_score', 0):.2f}")

                with col3:
                    st.metric("Anomaly Rate", f"{db_stats.get('anomaly_rate', 0) * 100:.2f}%")
                    st.metric("Max Volume Spike", f"{db_stats.get('max_volume_spike', 0):.1f}x")

                # Anomaly breakdown
                if 'anomaly_breakdown' in db_stats and db_stats['anomaly_breakdown']:
                    st.subheader("Anomaly Breakdown")
                    breakdown_df = pd.DataFrame.from_dict(
                        db_stats['anomaly_breakdown'],
                        orient='index',
                        columns=['Count']
                    ).reset_index()
                    breakdown_df.columns = ['Type', 'Count']

                    fig_breakdown = px.bar(
                        breakdown_df,
                        x='Type',
                        y='Count',
                        title='Anomalies by Type (24h)'
                    )
                    st.plotly_chart(fig_breakdown, use_container_width=True)

    # Call the fragment function
    update_charts()

    # Footer
    st.markdown("---")
    st.caption("BTC/USDT Anomaly Detection System | Real-time monitoring with machine learning")


if __name__ == "__main__":
    main()
