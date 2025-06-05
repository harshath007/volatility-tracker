import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from ta.trend import EMAIndicator, MACD
from ta.momentum import RSIIndicator


# ---------------------------
# Data Fetching with fallback
# ---------------------------
def fetch_stock_data(symbol):
    end = datetime.now()
    start = end - timedelta(days=10)

    # Try 5-minute interval first
    df = yf.download(symbol, interval='5m', start=start, end=end, prepost=True)
    if df is None or df.empty or 'Close' not in df.columns or df['Close'].dropna().empty:
        # Fallback to daily data
        df = yf.download(symbol, interval='1d', start=start, end=end)
        if df is None or df.empty or 'Close' not in df.columns or df['Close'].dropna().empty:
            return None, "No valid data found for symbol in intraday or daily intervals."
        else:
            return df, "Using daily data fallback."
    return df, "Using 5-minute intraday data."

# ---------------------------
# Add Technical Indicators
# ---------------------------
def add_indicators(df):
    # Ensure 'Close' column exists and no NaNs
    df = df.dropna(subset=['Close'])
    df['EMA9'] = EMAIndicator(close=df['Close'], window=9).ema_indicator()
    df['RSI'] = RSIIndicator(close=df['Close']).rsi()
    df['MACD'] = MACD(close=df['Close']).macd()
    df.dropna(inplace=True)
    return df

# ---------------------------
# Calculate morning price change (6:30–9:30 AM PT)
# For daily data fallback, use previous close to next open change
# ---------------------------
def get_morning_price_change(df, interval='5m'):
    df = df.copy()
    df.index = df.index.tz_localize(None)

    if interval == '5m':
        # Select rows between 06:30 and 09:30 PT (PT = UTC-7/8 depending on DST; assume UTC-7 for simplicity)
        # Yahoo data is in UTC by default. PT = UTC-7 during DST, UTC-8 otherwise.
        # To simplify, treat index as UTC and convert 06:30-09:30 PT to UTC:
        # PT + 7 hours = UTC
        # So morning session in UTC is 13:30 to 16:30
        morning_df = df.between_time('13:30', '16:30')
        if morning_df.empty:
            return None
        open_price = morning_df.iloc[0]['Open']
        close_price = morning_df.iloc[-1]['Close']
        return (close_price - open_price) / open_price
    else:
        # For daily data, approximate morning change by (Open - Previous Close)/Previous Close
        if len(df) < 2:
            return None
        prev_close = df['Close'].iloc[-2]
        today_open = df['Open'].iloc[-1]
        return (today_open - prev_close) / prev_close

# ---------------------------
# Composite score from indicators
# ---------------------------
def compute_score(df):
    # Normalize indicators from last available row
    last = df.iloc[-1]
    # Simple normalization based on indicator ranges
    norm_ema = (last['EMA9'] - df['EMA9'].min()) / (df['EMA9'].max() - df['EMA9'].min()) if df['EMA9'].max() != df['EMA9'].min() else 0.5
    norm_rsi = last['RSI'] / 100
    norm_macd = (last['MACD'] - df['MACD'].min()) / (df['MACD'].max() - df['MACD'].min()) if df['MACD'].max() != df['MACD'].min() else 0.5

    # Weighted sum
    score = 0.4 * norm_ema + 0.3 * norm_rsi + 0.3 * norm_macd
    return score

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Stock Morning Predictor", layout="wide")
st.title("🔍 Simple Morning Stock Predictor")

symbol = st.text_input("Enter Stock/ETF Symbol (e.g. AAPL, TSLA, GLD)").upper().strip()

if symbol:
    with st.spinner("Fetching data and calculating prediction..."):
        df, msg = fetch_stock_data(symbol)
        if df is None:
            st.error(msg)
        else:
            st.info(msg)
            try:
                df = add_indicators(df)
            except Exception as e:
                st.error(f"Error processing indicators: {e}")
                st.stop()

            interval = '5m' if '5m' in msg else '1d'
            morning_change = get_morning_price_change(df, interval=interval)

            if morning_change is None:
                st.warning("Could not calculate morning price change for this data.")
            else:
                score = compute_score(df)
                st.subheader(f"Prediction for {symbol}")
                st.metric("Predicted Morning Price Change", f"{morning_change * 100:.2f}%")
                
                # Verbal prediction based on score
                if score > 0.65:
                    st.success("📈 This stock is likely to rise tomorrow morning.")
                elif score < 0.35:
                    st.warning("📉 This stock is likely to fall tomorrow morning.")
                else:
                    st.info("⚖️ This stock may remain neutral or range-bound tomorrow morning.")

                # Optional: show recent data
                if st.checkbox("Show recent data and indicators"):
                    st.write(df.tail(10))
