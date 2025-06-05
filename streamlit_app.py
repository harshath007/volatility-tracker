import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from ta.trend import EMAIndicator, MACD
from ta.momentum import RSIIndicator

def fetch_stock_data(symbol):
    end = datetime.now()
    start = end - timedelta(days=10)
    df = yf.download(symbol, interval='5m', start=start, end=end, prepost=True)
    if df is None or df.empty:
        return None
    if 'Close' not in df.columns or df['Close'].dropna().empty:
        return None
    return df

def add_indicators(df):
    # Defensive check
    if 'Close' not in df.columns:
        raise ValueError("Dataframe missing 'Close' column")

    df = df.dropna(subset=['Close'])
    df = df.sort_index()
    if not df.index.is_unique:
        df = df[~df.index.duplicated(keep='first')]

    # Calculate indicators
    df['EMA9'] = EMAIndicator(close=df['Close'], window=9).ema_indicator()
    df['RSI'] = RSIIndicator(close=df['Close']).rsi()
    df['MACD'] = MACD(close=df['Close']).macd()
    df['Volatility'] = df['Close'].rolling(window=10).std()

    return df.dropna()

def compute_score(df):
    # Normalize indicators safely
    def norm(series):
        min_val = series.min()
        max_val = series.max()
        denom = max_val - min_val
        return (series.iloc[-1] - min_val) / (denom if denom != 0 else 1)

    norm_ema = norm(df['EMA9'])
    norm_rsi = df['RSI'].iloc[-1] / 100
    norm_macd = norm(df['MACD'])
    norm_vol = norm(df['Volatility'])

    score = 0.3 * norm_ema + 0.3 * norm_rsi + 0.3 * norm_macd + 0.1 * (1 - norm_vol)
    return score

def get_target(df):
    try:
        df = df.tz_localize(None)
    except Exception:
        pass  # if already naive datetime

    try:
        morning_open = df.between_time('06:30', '06:30')['Open']
        morning_close = df.between_time('09:30', '09:30')['Close']
        if morning_open.empty or morning_close.empty:
            return None
        open_price = morning_open.iloc[-1]
        close_price = morning_close.iloc[-1]
        return (close_price - open_price) / open_price
    except Exception:
        return None

st.set_page_config(page_title="Stock Morning Predictor", layout="wide")
st.title("🔍 Simple Morning Stock Predictor")

symbol = st.text_input("Enter Stock/ETF Symbol (e.g. AAPL, TSLA, GLD)")

if symbol:
    with st.spinner("Fetching data..."):
        df = fetch_stock_data(symbol.upper())
        if df is None:
            st.error("No valid data found for this symbol in the past 10 days or missing required columns.")
        else:
            try:
                df = add_indicators(df)
            except Exception as e:
                st.error(f"Error processing indicators: {e}")
            else:
                if df.empty:
                    st.error("Insufficient data after processing to compute indicators.")
                else:
                    score = compute_score(df)
                    target = get_target(df)

                    st.subheader(f"📊 Prediction for {symbol.upper()}")
                    st.metric("Predicted Morning Price Change", f"{(score - 0.5)*2*100:.2f}% (approx.)")

                    if score > 0.65:
                        st.success("📈 This stock is likely to gain in tomorrow morning's volatility period.")
                    elif score < 0.35:
                        st.warning("📉 This stock may decline in tomorrow morning's volatility period.")
                    else:
                        st.info("⚖️ This stock is expected to be relatively neutral tomorrow morning.")

                    if target is not None:
                        st.write(f"**Actual historical target morning change (last available):** {target*100:.2f}%")
                    else:
                        st.write("**No historical morning target data available for comparison.**")
