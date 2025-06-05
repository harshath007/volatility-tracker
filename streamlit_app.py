import streamlit as st
import yfinance as yf
import pandas as pd
from ta.trend import EMAIndicator, MACD
from ta.momentum import RSIIndicator

st.set_page_config(page_title="Stock Predictor", layout="centered")
st.title("📈 Stock Prediction App")

# User Input
symbol = st.text_input("Enter Stock/ETF Symbol (e.g. AAPL, TSLA, GLD)", value="AAPL").upper()

# Fortune 500 Ticker List (top examples)
fortune_500_tickers = [
    'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'TSLA', 'BRK-B', 'JNJ', 'V', 'WMT',
    'JPM', 'PG', 'MA', 'NVDA', 'UNH', 'HD', 'DIS', 'BAC', 'XOM', 'VZ'
]

with st.expander("📌 Fortune 500 Tickers"):
    st.write(fortune_500_tickers)

# Fetch historical intraday data
@st.cache_data

def load_data(symbol):
    try:
        df = yf.download(symbol, interval='30m', period='10d', progress=False)
        df.dropna(inplace=True)
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        return df
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        return pd.DataFrame()

def add_indicators(df):
    try:
        df = df.copy()
        df['EMA9'] = EMAIndicator(close=df['Close'], window=9).ema_indicator()
        df['EMA21'] = EMAIndicator(close=df['Close'], window=21).ema_indicator()
        df['RSI'] = RSIIndicator(close=df['Close'], window=14).rsi()
        macd = MACD(close=df['Close'])
        df['MACD'] = macd.macd()
        df['Signal'] = macd.macd_signal()
        return df.dropna()
    except Exception as e:
        st.error(f"Error processing indicators: {e}")
        return df

# Simple scoring model for prediction
def predict_movement(df):
    last = df.iloc[-1]
    score = 0
    if last['EMA9'] > last['EMA21']: score += 1
    if last['RSI'] < 30: score += 1
    if last['MACD'] > last['Signal']: score += 1

    pct_change = ((last['Close'] - df.iloc[-2]['Close']) / df.iloc[-2]['Close']) * 100
    prediction = "Rise" if score >= 2 else "Fall"
    return prediction, pct_change

# Load and process data
df = load_data(symbol)
if not df.empty:
    df = add_indicators(df)
    st.write(f"Using 30-minute intraday data.\n\nData shape: {df.shape}")
    st.dataframe(df.tail())

    prediction, change = predict_movement(df)
    st.subheader("📊 Prediction")
    st.write(f"**Predicted Movement:** {prediction}")
    st.write(f"**Predicted Price Change (next interval):** {change:.2f}%")

# Example: Prediction for multiple stocks (Fortune 500 scoring)
def batch_predict(symbols):
    records = []
    for sym in symbols:
        df = load_data(sym)
        if not df.empty:
            df = add_indicators(df)
            if not df.empty:
                pred, change = predict_movement(df)
                records.append({"Symbol": sym, "Predicted_Change": change, "Direction": pred})
    return pd.DataFrame(records)

st.markdown("---")
st.subheader("🔝 Top 10 Fortune 500 Predictions")

if st.button("Run Fortune 500 Predictions"):
    preds_df = batch_predict(fortune_500_tickers)
    if not preds_df.empty:
        top_10 = preds_df.sort_values(by='Predicted_Change', ascending=False).head(10)
        st.table(top_10.reset_index(drop=True))
    else:
        st.write("No predictions available.")
