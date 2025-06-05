import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from ta.trend import EMAIndicator, MACD
from ta.momentum import RSIIndicator

# Fetch data with fallback and validation
def fetch_stock_data(symbol):
    end = datetime.now()
    start = end - timedelta(days=10)

    df = yf.download(symbol, interval='5m', start=start, end=end, prepost=True)

    # Flatten columns if MultiIndex
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    if df is None or df.empty or 'Close' not in df.columns or df['Close'].dropna().empty:
        df = yf.download(symbol, interval='1d', start=start, end=end)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if df is None or df.empty or 'Close' not in df.columns or df['Close'].dropna().empty:
            return None, "No valid data found for symbol."
        else:
            return df, "Using daily data fallback."
    return df, "Using 5-minute intraday data."


# Add technical indicators safely
def add_indicators(df):
    if 'Close' not in df.columns or df['Close'].dropna().empty:
        raise ValueError("DataFrame missing valid 'Close' column.")

    df = df.dropna(subset=['Close'])
    df['EMA9'] = EMAIndicator(close=df['Close'], window=9).ema_indicator()
    df['RSI'] = RSIIndicator(close=df['Close']).rsi()
    df['MACD'] = MACD(close=df['Close']).macd()
    df.dropna(inplace=True)
    return df

# Get morning price change depending on interval
def get_morning_price_change(df, interval='5m'):
    df = df.copy()
    df.index = df.index.tz_localize(None)

    if interval == '5m':
        # Morning 6:30-9:30 PT = 13:30-16:30 UTC
        morning_df = df.between_time('13:30', '16:30')
        if morning_df.empty:
            return None
        open_price = morning_df.iloc[0]['Open']
        close_price = morning_df.iloc[-1]['Close']
        return (close_price - open_price) / open_price
    else:
        if len(df) < 2:
            return None
        prev_close = df['Close'].iloc[-2]
        today_open = df['Open'].iloc[-1]
        return (today_open - prev_close) / prev_close

# Compute composite score
def compute_score(df):
    last = df.iloc[-1]
    norm_ema = (last['EMA9'] - df['EMA9'].min()) / (df['EMA9'].max() - df['EMA9'].min()) if df['EMA9'].max() != df['EMA9'].min() else 0.5
    norm_rsi = last['RSI'] / 100
    norm_macd = (last['MACD'] - df['MACD'].min()) / (df['MACD'].max() - df['MACD'].min()) if df['MACD'].max() != df['MACD'].min() else 0.5

    score = 0.4 * norm_ema + 0.3 * norm_rsi + 0.3 * norm_macd
    return score

# Streamlit UI
st.set_page_config(page_title="Stock Morning Predictor", layout="wide")
st.title("🔍 Simple Morning Stock Predictor")

symbol = st.text_input("Enter Stock/ETF Symbol (e.g. AAPL, TSLA, GLD)").upper().strip()

if symbol:
    with st.spinner("Fetching data and calculating prediction..."):
        df, msg = fetch_stock_data(symbol)

        if df is None:
            st.error(msg)
            st.stop()

        st.info(msg)
        st.write(f"Data shape: {df.shape}")
        st.write("Columns:", df.columns.tolist())

        if 'Close' not in df.columns or df['Close'].dropna().empty:
            st.error("No valid 'Close' prices available in the data.")
            st.stop()

        try:
            df = add_indicators(df)
        except Exception as e:
            st.error(f"Error processing indicators: {e}")
            st.stop()

        interval = '5m' if '5-minute' in msg or '5m' in msg else '1d'
        morning_change = get_morning_price_change(df, interval=interval)

        if morning_change is None:
            st.warning("Could not calculate morning price change for this data.")
        else:
            score = compute_score(df)
            st.subheader(f"Prediction for {symbol}")
            st.metric("Predicted Morning Price Change", f"{morning_change * 100:.2f}%")

            if score > 0.65:
                st.success("📈 This stock is likely to rise tomorrow morning.")
            elif score < 0.35:
                st.warning("📉 This stock is likely to fall tomorrow morning.")
            else:
                st.info("⚖️ This stock may remain neutral or range-bound tomorrow morning.")

            if st.checkbox("Show recent data and indicators"):
                st.write(df.tail(10))
