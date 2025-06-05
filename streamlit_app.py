import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from ta.trend import EMAIndicator, MACD
from ta.momentum import RSIIndicator
from pytrends.request import TrendReq
from transformers import pipeline

# Fetch stock data - last 10 days, 5m interval, including pre/post market
def fetch_stock_data(symbol):
    end = datetime.now()
    start = end - timedelta(days=10)
    df = yf.download(symbol, interval='5m', start=start, end=end, prepost=True)
    df.dropna(inplace=True)
    return df

# Calculate indicators
def add_indicators(df):
    df['EMA9'] = EMAIndicator(close=df['Close'], window=9).ema_indicator()
    df['RSI'] = RSIIndicator(close=df['Close']).rsi()
    df['MACD'] = MACD(close=df['Close']).macd()
    df['Volatility'] = df['Close'].rolling(window=10).std()
    return df.dropna()

# Get Google Trends score
def get_google_trend_score(keyword):
    pytrends = TrendReq(hl='en-US', tz=360)
    try:
        pytrends.build_payload([keyword], timeframe='now 7-d')
        data = pytrends.interest_over_time()
        if not data.empty:
            return data[keyword].iloc[-1]
    except:
        pass
    return 0

# Get FinBERT sentiment score (simplified with two sample headlines)
def get_sentiment_score(keyword):
    classifier = pipeline("sentiment-analysis", model="ProsusAI/finbert", device=-1)
    headlines = [
        f"{keyword} stock rises after positive earnings",
        f"{keyword} faces regulatory scrutiny",
    ]
    results = classifier(headlines)
    score = 0
    for r in results:
        if r['label'] == 'positive':
            score += r['score']
        elif r['label'] == 'negative':
            score -= r['score']
    return score / len(results)

# Calculate next morning (6:30-9:30 PT) price change as label
def get_target_price_change(df):
    df.index = df.index.tz_localize(None)
    try:
        open_price = df.between_time('06:30', '06:30')['Open'][-1]
        close_price = df.between_time('09:30', '09:30')['Close'][-1]
        return (close_price - open_price) / open_price
    except:
        return None

# Combine scores in weighted sum to predict next morning price change
def predict_morning_change(df, trend_score, sentiment_score):
    # Normalize indicators to 0-1 scale
    def norm(series):
        return (series - series.min()) / (series.max() - series.min() + 1e-9)
    
    ema = norm(df['EMA9']).iloc[-1]
    rsi = df['RSI'].iloc[-1] / 100
    macd = norm(df['MACD']).iloc[-1]
    vol = 1 - norm(df['Volatility']).iloc[-1]  # Less volatility favored

    # Fixed weights - simple and intuitive
    weights = {
        'EMA9': 0.25,
        'RSI': 0.25,
        'MACD': 0.2,
        'Volatility': 0.1,
        'Trend': 0.1,
        'Sentiment': 0.1
    }

    score = (ema * weights['EMA9'] +
             rsi * weights['RSI'] +
             macd * weights['MACD'] +
             vol * weights['Volatility'] +
             (trend_score / 100) * weights['Trend'] +
             sentiment_score * weights['Sentiment'])

    return score

# Streamlit UI
st.set_page_config(page_title="Simple Morning Stock Predictor", layout="centered")
st.title("🔎 Simple Morning Stock Predictor")

symbol = st.text_input("Enter Stock/ETF Symbol (e.g. TSLA, AAPL, GLD)").upper()

if symbol:
    with st.spinner("Fetching data and calculating prediction..."):
        df = fetch_stock_data(symbol)
        if df.empty:
            st.error("No data found for this symbol. Try another one.")
        else:
            df = add_indicators(df)
            trend_score = get_google_trend_score(symbol)
            sentiment_score = get_sentiment_score(symbol)
            label = get_target_price_change(df)

            if label is None:
                st.warning("Market not open or insufficient data for 6:30-9:30 AM PT period.")
            else:
                pred_score = predict_morning_change(df, trend_score, sentiment_score)

                st.subheader(f"Prediction for {symbol} (Next Morning 6:30-9:30 AM PT)")
                st.metric("Predicted Morning Score", f"{pred_score:.2f}")

                if pred_score > 0.6:
                    st.success("Likely to increase in price.")
                elif pred_score < 0.4:
                    st.error("Likely to decrease in price.")
                else:
                    st.info("Likely to stay neutral or range-bound.")

                st.write(f"Historical actual next-morning price change (last day): {label:.2%}")

