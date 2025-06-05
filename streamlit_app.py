import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from ta.trend import EMAIndicator, MACD
from ta.momentum import RSIIndicator
from pytrends.request import TrendReq
from transformers import pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor

# ---------------------------
# Data Fetching
# ---------------------------

def fetch_stock_data(symbol):
    end = datetime.now()
    start = end - timedelta(days=10)
    df = yf.download(symbol, interval='5m', start=start, end=end, prepost=True)
    df.dropna(inplace=True)
    return df

# ---------------------------
# Technical Indicators
# ---------------------------

def add_indicators(df):
    df['EMA9'] = EMAIndicator(close=df['Close'], window=9).ema_indicator()
    df['RSI'] = RSIIndicator(close=df['Close']).rsi()
    df['MACD'] = MACD(close=df['Close']).macd()
    df['Volatility'] = df['Close'].rolling(window=10).std()
    return df.dropna()

# ---------------------------
# Google Trends
# ---------------------------

def get_google_trend_score(keyword):
    pytrends = TrendReq(hl='en-US', tz=360)
    pytrends.build_payload([keyword], cat=0, timeframe='now 7-d', geo='', gprop='')
    data = pytrends.interest_over_time()
    if not data.empty:
        return data[keyword].iloc[-1]
    return 0

# ---------------------------
# FinBERT Sentiment
# ---------------------------

def get_sentiment_score(keyword):
    headlines = [
        f"{keyword} stock price rises after earnings beat",
        f"{keyword} faces investigation over compliance",
    ]
    classifier = pipeline("sentiment-analysis", model="ProsusAI/finbert")
    results = classifier(headlines)
    sentiment_score = 0
    for r in results:
        if r['label'] == 'positive':
            sentiment_score += r['score']
        elif r['label'] == 'negative':
            sentiment_score -= r['score']
    return sentiment_score / len(results)

# ---------------------------
# Label: 6:30–9:30 AM PT price change
# ---------------------------

def get_target(df):
    df.index = df.index.tz_localize(None)
    morning_open = df.between_time('06:30', '06:30')['Open']
    morning_close = df.between_time('09:30', '09:30')['Close']
    if morning_open.empty or morning_close.empty:
        return None
    open_price = morning_open.iloc[-1]
    close_price = morning_close.iloc[-1]
    return (close_price - open_price) / open_price

# ---------------------------
# Strict Weighted Formula
# ---------------------------

def compute_weighted_score(df):
    # Normalize features
    df['norm_EMA'] = (df['EMA9'] - df['EMA9'].min()) / (df['EMA9'].max() - df['EMA9'].min())
    df['norm_RSI'] = df['RSI'] / 100  # Already 0-100
    df['norm_MACD'] = (df['MACD'] - df['MACD'].min()) / (df['MACD'].max() - df['MACD'].min())
    df['norm_Volatility'] = (df['Volatility'] - df['Volatility'].min()) / (df['Volatility'].max() - df['Volatility'].min())
    
    # Define fixed weights
    weights = {
        'EMA': 0.2,
        'RSI': 0.2,
        'MACD': 0.2,
        'Volatility': 0.1,
        'TrendScore': 0.15,
        'SentimentScore': 0.15
    }

    df['CompositeScore'] = (
        weights['EMA'] * df['norm_EMA'] +
        weights['RSI'] * df['norm_RSI'] +
        weights['MACD'] * df['norm_MACD'] +
        weights['Volatility'] * (1 - df['norm_Volatility']) +  # less volatile preferred
        weights['TrendScore'] * df['TrendScore'].mean() / 100 +
        weights['SentimentScore'] * df['SentimentScore'].mean()
    )
    return df['CompositeScore'].iloc[-1]

# ---------------------------
# Streamlit Search-based UI
# ---------------------------

st.set_page_config(page_title="Stock Morning Predictor", layout="wide")
st.title("🔍 Search-Based Morning Stock Predictor")

symbol = st.text_input("Search a Stock/ETF Symbol (e.g. TSLA, AAPL, GLD)")

if symbol:
    with st.spinner("Fetching data and generating prediction..."):
        df = fetch_stock_data(symbol)
        df = add_indicators(df)
        trend_score = get_google_trend_score(symbol)
        sentiment_score = get_sentiment_score(symbol)
        label = get_target(df)

    if label is None:
        st.error("Couldn't find complete 6:30–9:30 AM PT data. Try another symbol or wait for market open.")
    else:
        df['TrendScore'] = trend_score
        df['SentimentScore'] = sentiment_score
        composite_score = compute_weighted_score(df)

        st.subheader(f"📊 Prediction for {symbol.upper()}")
        st.metric("Predicted Morning Profitability Score", f"{composite_score * 100:.2f}%")
        if composite_score > 0.6:
            st.success("📈 This stock is likely to perform well tomorrow morning.")
        elif composite_score < 0.4:
            st.warning("📉 This stock may underperform in tomorrow's early session.")
        else:
            st.info("⚖️ This stock may remain neutral or range-bound.")
