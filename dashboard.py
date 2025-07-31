import streamlit as st
import pandas as pd
import os
import time
import plotly.express as px
from streamlit_autorefresh import st_autorefresh

# --- Config ---
TRADE_LOG = "trades.csv"
ANOMALY_LOG = "anomalies.csv"
REFRESH_INTERVAL = 5

st.set_page_config(page_title="BTC Anomaly Monitor", layout="wide")
st.title("📊 BTC/USDT Anomaly Detection Dashboard")

st.sidebar.header("View Mode")
view_mode = st.sidebar.radio("Select mode:", ["Live", "Historical"])

if view_mode == "Historical":
    interval = st.sidebar.selectbox("Group by", ["Hourly", "Daily"])
else:
    auto_refresh = st.sidebar.checkbox("Auto-refresh", value=True)
    refresh_rate = st.sidebar.slider("Refresh every (s)", 2, 30, REFRESH_INTERVAL)

@st.cache_data(ttl=5)
def load_data():
    trades = pd.read_csv(TRADE_LOG, parse_dates=["timestamp"]) if os.path.exists(TRADE_LOG) else pd.DataFrame()
    anomalies = pd.read_csv(ANOMALY_LOG, parse_dates=["timestamp"]) if os.path.exists(ANOMALY_LOG) else pd.DataFrame()
    return trades, anomalies

trades, anomalies = load_data()

if trades.empty:
    st.error("No trade data found.")
    st.stop()

trades.sort_values("timestamp", inplace=True)

# --- Auto-detect anomaly types ---
anomaly_types = anomalies["anomaly_type"].unique().tolist() if not anomalies.empty else []
selected_anomalies = st.sidebar.multiselect("Anomaly types to show", anomaly_types, default=anomaly_types)

# --- LIVE MODE ---
if view_mode == "Live":
    trades_live = trades.tail(400)
    time_window = trades_live["timestamp"].min(), trades_live["timestamp"].max()

    anomalies_filtered = anomalies[
        (anomalies["timestamp"] >= time_window[0]) &
        (anomalies["timestamp"] <= time_window[1]) &
        (anomalies["anomaly_type"].isin(selected_anomalies))
    ]

    st.subheader("📈 Live Price Chart with Rolling Mean")
    fig = px.line(
        trades_live,
        x="timestamp",
        y=["price", "rolling_mean"],
        labels={"value": "Price", "timestamp": "Time"},
        title="BTC/USDT Price & Rolling Mean"
    )
    fig.update_yaxes(range=[
        trades_live["price"].min() * 0.998,
        trades_live["price"].max() * 1.002
    ])

    marker_styles = {
        "z_score": {"color": "red", "symbol": "x"},
        "filtered_isoforest": {"color": "blue", "symbol": "x"},
        "isolation_forest": {"color": "purple", "symbol": "x"},
        "z_score_injected": {"color": "orange", "symbol": "star"},
        "filtered_isoforest_injected": {"color": "gold", "symbol": "star"},
        "isolation_forest_injected": {"color": "lightgreen", "symbol": "star"}
    }

    for anomaly_type in selected_anomalies:
        subset = anomalies_filtered[anomalies_filtered["anomaly_type"] == anomaly_type]
        if not subset.empty:
            style = marker_styles.get(anomaly_type, {"color": "gray", "symbol": "x"})
            fig.add_scatter(
                x=subset["timestamp"],
                y=subset["price"],
                mode="markers",
                name=anomaly_type,
                marker=dict(
                    size=10,
                    symbol=style["symbol"],
                    color=style["color"]
                ),
                customdata=subset[["z_score", "anomaly_type"]],
                hovertemplate="Time: %{x}<br>Price: %{y:.2f}<br>Z-score: %{customdata[0]:.2f}<br>Type: %{customdata[1]}"
            )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📉 Z-Score Trend")
    st.line_chart(trades_live.set_index("timestamp")[["z_score"]])

    if not anomalies_filtered.empty:
        st.subheader("🚨 Recent Detected Anomalies")
        st.dataframe(anomalies_filtered.tail(20), use_container_width=True)

        # === NEW: Persistent Anomaly Chart ===
        st.subheader("🛰 Persistent Anomaly Markers (last 60 minutes)")
        anomaly_window = pd.Timestamp.utcnow() - pd.Timedelta(minutes=60)
        recent_anomalies = anomalies[
            (anomalies["timestamp"] >= anomaly_window) &
            (anomalies["anomaly_type"].isin(selected_anomalies))
        ].copy()

        if not recent_anomalies.empty:
            anomaly_chart = px.scatter(
                recent_anomalies,
                x="timestamp",
                y="price",
                color="anomaly_type",
                title="Persistent Anomaly Timeline",
                symbol="anomaly_type",
                hover_data=["z_score", "quantity", "anomaly_type"]
            )
            st.plotly_chart(anomaly_chart, use_container_width=True)
        else:
            st.info("No persistent anomalies in the past hour.")

        def convert_df(df):
            return df.to_csv(index=False).encode("utf-8")

        st.download_button("⬇️ Download Anomalies CSV", convert_df(anomalies), "anomalies.csv", "text/csv")
        st.download_button("⬇️ Download Trades CSV", convert_df(trades), "trades.csv", "text/csv")


    if auto_refresh:
        st_autorefresh(interval=refresh_rate * 1000, key="refresh")

# --- HISTORICAL MODE (unchanged) ---
if view_mode == "Historical":
    trades_historical = trades.copy()
    trades_historical["timestamp"] = pd.to_datetime(
        trades_historical["timestamp"], format='mixed', errors='coerce'
    )
    trades_historical.dropna(subset=["timestamp"], inplace=True)
    trades_historical.set_index("timestamp", inplace=True)

    rule = "1H" if interval == "Hourly" else "1D"
    grouped = trades_historical.resample(rule).agg({
        "price": "mean",
        "quantity": "sum",
        "z_score": "mean"
    }).dropna()

    st.subheader(f"⏳ {interval} Price Trend")
    st.line_chart(grouped[["price"]])

    st.subheader(f"📉 {interval} Z-Score Trend")
    st.line_chart(grouped[["z_score"]])

    st.subheader(f"📦 {interval} Volume")
    st.bar_chart(grouped[["quantity"]])
