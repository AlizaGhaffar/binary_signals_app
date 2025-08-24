import streamlit as st
import pandas as pd
import numpy as np
import ta
import plotly.graph_objects as go
import random

# ----------------- CONFIG -----------------
st.set_page_config(page_title="Binary Signal Generator", layout="wide")
st.title("📊 Binary Options Signal Generator")
st.write("Generate CALL/PUT signals for EUR/USD OTC with multiple strategies.")

# ----------------- SIDEBAR -----------------
st.sidebar.header("⚙️ Settings")
timeframe = st.sidebar.selectbox("Select Timeframe", ["5 sec", "10 sec", "1 min"])
strategy = st.sidebar.multiselect(
    "Select Indicators",
    ["RSI", "EMA Crossover", "MACD", "Bollinger Bands"],
    default=["RSI", "EMA Crossover"]
)
simulate = st.sidebar.checkbox("Simulate Data", value=True)
uploaded_file = st.sidebar.file_uploader("Upload CSV (must include 'close')", type=["csv"])

# ----------------- LOAD DATA -----------------
if simulate or uploaded_file is None:
    np.random.seed(42)
    prices = np.cumsum(np.random.randn(1000) * 0.0001) + 1.1000
    df = pd.DataFrame({'close': prices})
else:
    df = pd.read_csv(uploaded_file)
    if 'close' not in df.columns:
        st.error("CSV must include a 'close' column.")
        st.stop()

# ----------------- CALCULATE INDICATORS -----------------
if "RSI" in strategy:
    df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
if "EMA Crossover" in strategy:
    df['ema_fast'] = df['close'].ewm(span=5, adjust=False).mean()
    df['ema_slow'] = df['close'].ewm(span=20, adjust=False).mean()
if "MACD" in strategy:
    macd = ta.trend.MACD(df['close'])
    df['macd'] = macd.macd()
    df['signal_line'] = macd.macd_signal()
if "Bollinger Bands" in strategy:
    bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
    df['bb_high'] = bb.bollinger_hband()
    df['bb_low'] = bb.bollinger_lband()

# ----------------- GENERATE SIGNALS -----------------
signals = []
for i in range(len(df)):
    if i < 20:
        signals.append((None, None))
        continue

    conditions = []

    # RSI condition
    if "RSI" in strategy and not pd.isna(df['rsi'].iloc[i]):
        if df['rsi'].iloc[i] < 30:
            conditions.append("CALL")
        elif df['rsi'].iloc[i] > 70:
            conditions.append("PUT")

    # EMA crossover condition
    if "EMA Crossover" in strategy and not pd.isna(df['ema_fast'].iloc[i]):
        if df['ema_fast'].iloc[i] > df['ema_slow'].iloc[i]:
            conditions.append("CALL")
        elif df['ema_fast'].iloc[i] < df['ema_slow'].iloc[i]:
            conditions.append("PUT")

    # MACD condition
    if "MACD" in strategy and not pd.isna(df['macd'].iloc[i]):
        if df['macd'].iloc[i] > df['signal_line'].iloc[i]:
            conditions.append("CALL")
        elif df['macd'].iloc[i] < df['signal_line'].iloc[i]:
            conditions.append("PUT")

    # Bollinger Bands
    if "Bollinger Bands" in strategy and not pd.isna(df['bb_high'].iloc[i]):
        if df['close'].iloc[i] < df['bb_low'].iloc[i]:
            conditions.append("CALL")
        elif df['close'].iloc[i] > df['bb_high'].iloc[i]:
            conditions.append("PUT")

    # Decision
    if len(conditions) == 0:
        signals.append((None, None))
    else:
        if conditions.count("CALL") > conditions.count("PUT"):
            final_signal = "CALL"
        elif conditions.count("PUT") > conditions.count("CALL"):
            final_signal = "PUT"
        else:
            final_signal = None

        confidence = int((conditions.count(final_signal) / len(strategy)) * 100)
        signals.append((final_signal, confidence))

df['signal'] = [s[0] for s in signals]
df['confidence'] = [s[1] for s in signals]

# ----------------- DISPLAY LATEST SIGNALS -----------------
st.subheader("📌 Latest Signals")
latest_signals = df.dropna(subset=['signal']).tail(10)
st.table(latest_signals[['close', 'signal', 'confidence']])

# ----------------- PLOT PRICE WITH SIGNALS -----------------
st.subheader("📈 Price Chart with Signals")
fig = go.Figure()
fig.add_trace(go.Scatter(y=df['close'], mode='lines', name='Price'))

for idx, row in df.iterrows():
    if row['signal'] == 'CALL':
        fig.add_trace(go.Scatter(x=[idx], y=[row['close']], mode='markers',
                                 marker=dict(color='green', size=10), name='CALL'))
    elif row['signal'] == 'PUT':
        fig.add_trace(go.Scatter(x=[idx], y=[row['close']], mode='markers',
                                 marker=dict(color='red', size=10), name='PUT'))

st.plotly_chart(fig, use_container_width=True)