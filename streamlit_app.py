import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from ta.trend import EMAIndicator, MACD
from ta.momentum import RSIIndicator

# ---------------------------
# Fetch stock data safely
# ---------------------------
def fetch_stock_data(symbol):
    end = datetime.now()
    start = end - timedelta(days=10)
    df = yf.download(symbol, interval='5m', start=start, end=end, prepost=True)
    if df.empty:
        return None
    df.index = pd.to_datetime(df.index)
    return df

# ---------------------------
# Add indicators safely
# ---------------------------
def add_indicators(df):
    if 'Close' not in df.columns or df['Close'].dropna().empty:
        return None
    
    df['EMA9'] = EMAIndicator(close=df['Close'], window=9).ema_indicator()
    df['RSI'] = RSIIndicator(close=df['Close']).rsi()
    df['MACD'] = MACD(close=df['Close']).macd()
    df['Volatility'] = df['Close'].rolling(window=10).std()
    return df.dropna()

# ---------------------------
# Simple weighted score prediction
# ---------------------------
def compute_score(df):
    def normalize(series):
        return (series - series.min()) / (series.max() - series.min() + 1e-9)

    ema_norm = normalize(df['EMA9'])
    rsi_norm = df['RSI'] / 100
    macd_norm = normalize(df['MACD'])
    vol_norm = normalize(df['Volatility'])
    
    score = (
        0.3 * ema_norm.iloc[-1] +
        0.3 * rsi_norm.iloc[-1] +
        0.3 * macd_norm.iloc[-1] +
        0.1 * (1 - vol_norm.iloc[-1])  # less volatility better
    )
    return score

# ---------------------------
# Convert score to verbal and % prediction
# ---------------------------
def verbal_and_price_change(score):
    # Map score (0 to 1) roughly to a predicted % price change range -1.5% to +1.5%
    predicted_change = (score - 0.5) * 3  # range roughly -1.5% to +1.5%
    
    if score > 0.65:
        verbal = "📈 Likely to perform well tomorrow morning."
    elif score < 0.35:
        verbal = "📉 May underperform tomorrow morning."
    else:
        verbal = "⚖️ May remain neutral or range-bound tomorrow morning."
    
    return verbal, predicted_change

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Simple Stock Morning Predictor", layout="centered")
st.title("🔎 Simple Morning Stock Predictor")

symbol = st.text_input("Enter Stock/ETF Symbol (e.g. TSLA, AAPL, GLD)").strip().upper()

if symbol:
    with st.spinner(f"Fetching data for {symbol}..."):
        df = fetch_stock_data(symbol)
        if df is None:
            st.error(f"No data found for symbol '{symbol}'. Please check the symbol and try again.")
        else:
            df = add_indicators(df)
            if df is None or df.empty:
                st.error("Insufficient data to calculate indicators. Try another symbol or check market hours.")
            else:
                score = compute_score(df)
                verbal, predicted_change = verbal_and_price_change(score)
                
                st.subheader(f"Prediction for {symbol}:")
                st.metric("Morning Profitability Score", f"{score:.2f} / 1.00")
                st.write(verbal)
                st.write(f"📊 **Predicted Price Change (6:30–9:30 AM PT):** {predicted_change:.2%}")

                # Optional: Show last 5 rows of indicator values for transparency
                st.markdown("### Recent Indicator Values")
                st.dataframe(df[['EMA9', 'RSI', 'MACD', 'Volatility']].tail(5))
