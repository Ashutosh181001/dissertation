"""
Improved BTC/USDT anomaly detection dashboard using Streamlit fragments.

This app is functionally similar to the original Streamlit dashboard, but
the charts and data update asynchronously inside a fragment.  By placing
the expensive I/O and plotting logic in a fragment decorated with
``@st.fragment(run_every=…)``, only that part of the page refreshes on
a timer.  The rest of the UI (title, sidebar controls and download
buttons) remains static, so the user never sees the page flash or a
long-running spinner when new trades arrive.

To run this app locally, install Streamlit version >= 1.37.0 and
Plotly.  Then execute ``streamlit run improved_dashboard.py``.  The app
will watch the ``trades.csv`` and ``anomalies.csv`` files in the same
directory and update charts in real time according to the selected
refresh interval.

Refer to the Streamlit documentation on fragments for more details
about how fragment reruns work and how to control the refresh interval
with ``run_every``【365608622546187†L181-L198】【551314107550257†L180-L186】.
"""

import os
from datetime import datetime, timedelta
from typing import List

import pandas as pd
import plotly.express as px
import streamlit as st


# -----------------------------------------------------------------------------
# Configuration constants
#
# These constants control file locations and the size of the rolling buffer of
# live trades displayed.  Adjust them as needed to suit your data and
# environment.
TRADE_LOG = "trades.csv"
ANOMALY_LOG = "anomalies.csv"
MAX_BUFFER = 400            # number of most recent trades to display
DEFAULT_REFRESH = 5          # default refresh interval in seconds


def load_recent_trades(max_buffer: int) -> pd.DataFrame:
    """Load the most recent trades from ``TRADE_LOG``.

    Parameters
    ----------
    max_buffer: int
        The maximum number of rows to return.  Older rows are discarded.

    Returns
    -------
    pd.DataFrame
        A DataFrame of trades with a parsed ``timestamp`` column.  If the
        file does not exist or is empty, an empty DataFrame with the expected
        columns is returned.
    """
    if not os.path.exists(TRADE_LOG):
        # Ensure the DataFrame has the necessary columns even when empty
        return pd.DataFrame(columns=["timestamp", "price", "rolling_mean", "z_score", "quantity"])
    trades = pd.read_csv(TRADE_LOG)
    if trades.empty:
        return trades
    # Parse timestamps and drop invalid rows
    trades["timestamp"] = pd.to_datetime(trades["timestamp"], errors="coerce")
    trades.dropna(subset=["timestamp"], inplace=True)
    # Keep only the latest ``max_buffer`` rows
    return trades.tail(max_buffer).reset_index(drop=True)


def load_anomalies() -> pd.DataFrame:
    """Load anomaly events from ``ANOMALY_LOG``.

    Returns
    -------
    pd.DataFrame
        A DataFrame of anomalies with a parsed ``timestamp`` column.  If the
        file does not exist or is empty, an empty DataFrame is returned.
    """
    if not os.path.exists(ANOMALY_LOG):
        return pd.DataFrame(columns=["timestamp", "price", "z_score", "quantity", "anomaly_type"])
    anomalies = pd.read_csv(ANOMALY_LOG)
    if anomalies.empty:
        return anomalies
    anomalies["timestamp"] = pd.to_datetime(anomalies["timestamp"], errors="coerce")
    anomalies.dropna(subset=["timestamp"], inplace=True)
    return anomalies.reset_index(drop=True)


def get_anomaly_types(anomalies: pd.DataFrame) -> List[str]:
    """Return a sorted list of unique anomaly types.

    Parameters
    ----------
    anomalies: pd.DataFrame
        The anomalies DataFrame.  If empty, an empty list is returned.

    Returns
    -------
    List[str]
        Unique anomaly type names sorted alphabetically.
    """
    if anomalies.empty or "anomaly_type" not in anomalies.columns:
        return []
    return sorted(anomalies["anomaly_type"].dropna().unique().tolist())


def main() -> None:
    """Entrypoint for the Streamlit dashboard."""
    # Configure the page before any elements are drawn
    st.set_page_config(page_title="BTC Anomaly Dashboard", layout="wide")
    st.title("📊 BTC/USDT Anomaly Detection Dashboard")

    # Sidebar controls
    st.sidebar.header("Settings")
    # Users can adjust how frequently the live charts refresh.  The minimum
    # interval is 1 second and the maximum is 30 seconds.  We cap the default
    # to ``DEFAULT_REFRESH`` defined above.
    refresh_rate: int = int(
        st.sidebar.slider("Auto-refresh interval (seconds)", 1, 30, DEFAULT_REFRESH)
    )

    # Pre-load anomalies once to populate the multiselect.  We don't need to
    # refresh this list at the fragment rate because anomaly types change
    # infrequently.  If new anomaly types appear, the user can rerun the app.
    anomalies_for_filter = load_anomalies()
    anomaly_types = get_anomaly_types(anomalies_for_filter)
    selected_types = st.sidebar.multiselect(
        "Anomaly types to show", anomaly_types, default=anomaly_types
    )

    st.sidebar.header("Downloads")
    # We populate the download buttons after the first fragment run using
    # session state.  If the user loads the page for the first time and clicks
    # download before the fragment runs, no file will be offered.
    if "trades" in st.session_state and not st.session_state["trades"].empty:
        csv = st.session_state["trades"].to_csv(index=False).encode("utf-8")
        st.sidebar.download_button(
            "Download Trades CSV",
            csv,
            file_name="trades.csv",
            mime="text/csv",
        )
    if "anomalies" in st.session_state and not st.session_state["anomalies"].empty:
        csv_anom = st.session_state["anomalies"].to_csv(index=False).encode("utf-8")
        st.sidebar.download_button(
            "Download Anomalies CSV",
            csv_anom,
            file_name="anomalies.csv",
            mime="text/csv",
        )

    # -------------------------------------------------------------------------
    # Define the fragment that loads data and updates the charts
    #
    # We decorate the ``update_charts`` function with ``@st.fragment``.  The
    # ``run_every`` argument tells Streamlit to rerun only this function at
    # ``refresh_rate`` second intervals while the session is active【551314107550257†L180-L186】.  All
    # other elements on the page remain static between fragment reruns.  This
    # prevents the whole page from flashing or showing a spinner when new data
    # arrives.
    @st.fragment(run_every=refresh_rate)
    def update_charts() -> None:
        # Read the latest trades and anomalies from disk
        trades = load_recent_trades(MAX_BUFFER)
        anomalies = load_anomalies()

        # Save to session state so the download buttons outside the fragment
        # have access to the most recent data
        st.session_state.trades = trades
        st.session_state.anomalies = anomalies

        # Price chart with rolling mean
        st.subheader("📈 BTC/USDT Price with Rolling Mean")
        if not trades.empty:
            fig_price = px.line(
                trades,
                x="timestamp",
                y=["price", "rolling_mean"],
                labels={"value": "Price", "timestamp": "Time"},
                title="Live Price Chart",
            )
            fig_price.update_layout(
                height=350, xaxis_tickformat="%H:%M:%S", template="plotly_white"
            )
            st.plotly_chart(fig_price, use_container_width=True)
        else:
            st.info("No trade data available.")

        # Z-score line chart
        st.subheader("📉 Z-Score Over Time")
        if not trades.empty and "z_score" in trades.columns:
            fig_zscore = px.line(
                trades,
                x="timestamp",
                y="z_score",
                labels={"z_score": "Z-score", "timestamp": "Time"},
                title="Z-Score Trend",
            )
            fig_zscore.update_layout(
                height=250, xaxis_tickformat="%H:%M:%S", template="plotly_white"
            )
            st.plotly_chart(fig_zscore, use_container_width=True)
        else:
            st.info("Z-score data not available in trades.")

        # Filter anomalies to the last 60 minutes and selected types
        st.subheader("🚨 Anomalies in Last 60 Minutes")
        now = datetime.utcnow()
        recent_window = now - timedelta(minutes=60)

        if not anomalies.empty:
            recent_anomalies = anomalies[
                (anomalies["timestamp"] >= recent_window)
                & (anomalies["anomaly_type"].isin(selected_types))
            ]
        else:
            recent_anomalies = pd.DataFrame()

        if not recent_anomalies.empty:
            fig_anom = px.scatter(
                recent_anomalies,
                x="timestamp",
                y="price",
                color="anomaly_type",
                symbol="anomaly_type",
                hover_data=["z_score", "quantity"],
                title="Persistent Anomalies (Last 60 min)",
            )
            fig_anom.update_layout(
                height=300, xaxis_tickformat="%H:%M:%S", template="plotly_white"
            )
            st.plotly_chart(fig_anom, use_container_width=True)
            # Show the last 20 anomalies in a table
            st.dataframe(recent_anomalies.tail(20), use_container_width=True)
        else:
            st.info("✅ No anomalies detected in the last 60 minutes.")

    # Call the fragment function.  Because the ``run_every`` parameter is
    # specified in the decorator, simply calling the function will schedule
    # periodic reruns at the given interval.  When the user changes
    # ``refresh_rate`` via the slider, the script is rerun and the fragment
    # redecorated with the new interval.
    update_charts()


if __name__ == "__main__":
    main()