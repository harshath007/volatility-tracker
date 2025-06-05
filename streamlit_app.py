import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from ta.trend import EMAIndicator, MACD
from ta.momentum import RSIIndicator
from pytrends.request import TrendReq
from transformers import pipeline
import requests
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import holidays

# Set page style
st.set_page_config("📈 Stock Mood", layout="wide")
st.markdown("## 🌅 Morning Stock Performance Forecaster")

# Utility Functions
def fetch_stock_data(symbol):
    end = datetime.now()
    start = end - timedelta(days=5)
    df = yf.download(symbol, interval='5m', start=start, end=end, prepost=True)
    if df.empty:
        return pd.DataFrame()
    df.index = df.index.tz_convert('US/Pacific') if df.index.tz else df.index.tz_localize('UTC').tz_convert('US/Pacific')
    return df.dropna()

def add_indicators(df):
    df['EMA9'] = EMAIndicator(close=df['Close'], window=9).ema_indicator()
    df['RSI'] = RSIIndicator(close=df['Close']).rsi()
    df['MACD'] = MACD(close=df['Close']).macd()
    df['Volatility'] = df['Close'].rolling(window=10).std()
    return df.dropna()

def get_google_trend_score(keyword):
    try:
        pytrends = TrendReq(hl='en-US', tz=360)
        pytrends.build_payload([keyword], cat=0, timeframe='now 7-d')
        data = pytrends.interest_over_time()
        return int(data[keyword].iloc[-1]) if not data.empty else 0
    except:
        return 0

def fetch_headlines(symbol):
    try:
        url = f"https://finance.yahoo.com/quote/{symbol}?p={symbol}"
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(r.text, 'html.parser')
        return [x.text.strip() for x in soup.find_all('h3') if x.text.strip()][:5]
    except:
        return [f"Failed to fetch news for {symbol}"]

def get_sentiment_score(headlines):
    try:
        classifier = pipeline("sentiment-analysis", model="ProsusAI/finbert")
        results = classifier(headlines)
        score = sum((r["score"] if r["label"] == "positive" else -r["score"]) for r in results)
        return round(score / len(results), 3)
    except:
        return 0.0

def compute_composite_score(df, trend_score, sentiment_score):
    df = df.copy()
    df['norm_EMA'] = (df['EMA9'] - df['EMA9'].min()) / (df['EMA9'].max() - df['EMA9'].min())
    df['norm_RSI'] = df['RSI'] / 100
    df['norm_MACD'] = (df['MACD'] - df['MACD'].min()) / (df['MACD'].max() - df['MACD'].min())
    df['norm_Volatility'] = (df['Volatility'] - df['Volatility'].min()) / (df['Volatility'].max() - df['Volatility'].min())

    score = (
        0.25 * df['norm_EMA'].iloc[-1] +
        0.2 * df['norm_RSI'].iloc[-1] +
        0.2 * df['norm_MACD'].iloc[-1] +
        0.1 * (1 - df['norm_Volatility'].iloc[-1]) +
        0.15 * trend_score / 100 +
        0.1 * sentiment_score
    )
    return round(score * 100, 2)

def plot_indicators(df):
    fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    axs[0].plot(df.index, df['Close'], label='Close', color='cyan')
    axs[0].plot(df.index, df['EMA9'], label='EMA9', color='magenta')
    axs[0].legend()
    axs[0].set_title('Close & EMA9')

    axs[1].plot(df.index, df['RSI'], color='orange')
    axs[1].axhline(70, color='red', linestyle='--')
    axs[1].axhline(30, color='green', linestyle='--')
    axs[1].set_title('RSI')

    axs[2].plot(df.index, df['MACD'], color='purple')
    axs[2].axhline(0, color='black', linestyle='--')
    axs[2].set_title('MACD')

    plt.tight_layout()
    st.pyplot(fig)

# Holiday check
if datetime.today().date() in holidays.US():
    st.warning("📅 Today is a U.S. market holiday.")

# Main App
symbol = st.text_input("Enter Stock/ETF Symbol (e.g. AAPL, QQQ, TSLA):").upper()

if symbol:
    try:
        df = fetch_stock_data(symbol)
        if df.empty:
            st.error("No data found. The symbol may be invalid or market may be closed.")
        else:
            df = add_indicators(df)
            trend = get_google_trend_score(symbol)
            headlines = fetch_headlines(symbol)
            sentiment = get_sentiment_score(headlines)
            score = compute_composite_score(df, trend, sentiment)

            st.metric("📊 Composite Score", f"{score}%")
            st.metric("📈 Google Trend", trend)
            st.metric("📰 News Sentiment", sentiment)

            st.subheader("🗞️ Headlines")
            for h in headlines:
                st.write("- " + h)

            st.subheader("📉 Technical Indicator Chart")
            plot_indicators(df)
    except Exception as e:
        st.error(f"❌ Error: {e}")
