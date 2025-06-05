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
import holidays
import requests
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt

# ---------------------------
# Data Fetching
# ---------------------------
def fetch_stock_data(symbol):
    end = datetime.now()
    start = end - timedelta(days=14)
    df = yf.download(symbol, interval='5m', start=start, end=end, prepost=True)
    df.dropna(inplace=True)
    df.index = df.index.tz_convert('US/Pacific') if df.index.tz is not None else df.index.tz_localize('UTC').tz_convert('US/Pacific')
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
# Real Headlines + Sentiment
# ---------------------------
def fetch_real_headlines(symbol):
    url = f"https://finance.yahoo.com/quote/{symbol}?p={symbol}"
    response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
    soup = BeautifulSoup(response.content, 'html.parser')
    headlines = [tag.text for tag in soup.find_all('h3')][:5]
    return headlines if headlines else [f"No recent news found for {symbol}"]

def get_sentiment_score_from_headlines(headlines):
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
    morning_open = df.between_time('06:30', '06:30')['Open']
    morning_close = df.between_time('09:30', '09:30')['Close']
    if morning_open.empty or morning_close.empty:
        return None
    open_price = morning_open.iloc[-1]
    close_price = morning_close.iloc[-1]
    return (close_price - open_price) / open_price

# ---------------------------
# Composite Scoring
# ---------------------------
def compute_weighted_score(df, trend_score, sentiment_score):
    df['norm_EMA'] = (df['EMA9'] - df['EMA9'].min()) / (df['EMA9'].max() - df['EMA9'].min())
    df['norm_RSI'] = df['RSI'] / 100
    df['norm_MACD'] = (df['MACD'] - df['MACD'].min()) / (df['MACD'].max() - df['MACD'].min())
    df['norm_Volatility'] = (df['Volatility'] - df['Volatility'].min()) / (df['Volatility'].max() - df['Volatility'].min())

    weights = {
        'EMA': 0.2,
        'RSI': 0.2,
        'MACD': 0.2,
        'Volatility': 0.1,
        'TrendScore': 0.15,
        'SentimentScore': 0.15
    }

    score = (
        weights['EMA'] * df['norm_EMA'].iloc[-1] +
        weights['RSI'] * df['norm_RSI'].iloc[-1] +
        weights['MACD'] * df['norm_MACD'].iloc[-1] +
        weights['Volatility'] * (1 - df['norm_Volatility'].iloc[-1]) +
        weights['TrendScore'] * trend_score / 100 +
        weights['SentimentScore'] * sentiment_score
    )
    return score

# ---------------------------
# Model Training
# ---------------------------
def train_model(df):
    df['Target'] = df['Close'].shift(-3)
    df.dropna(inplace=True)
    features = df[['EMA9', 'RSI', 'MACD', 'Volatility']].values
    target = df['Target'].values.ravel()

    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, shuffle=False)
    model = XGBRegressor(n_estimators=100, learning_rate=0.1)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return model, preds[-1], r2_score(y_test, preds), mean_squared_error(y_test, preds)


# ---------------------------
# Plotting
# ---------------------------
def plot_indicators(df):
    fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.patch.set_facecolor('#111')

    axs[0].plot(df.index, df['Close'], label='Close', color='cyan')
    axs[0].plot(df.index, df['EMA9'], label='EMA9', color='magenta')
    axs[0].legend()
    axs[0].set_title('Close & EMA9')

    axs[1].plot(df.index, df['RSI'], color='orange')
    axs[1].axhline(70, color='red', linestyle='--')
    axs[1].axhline(30, color='green', linestyle='--')
    axs[1].set_title('RSI')

    axs[2].plot(df.index, df['MACD'], color='magenta')
    axs[2].axhline(0, color='black', linestyle='--')
    axs[2].set_title('MACD')

    plt.tight_layout()
    st.pyplot(fig)

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Morning Stock Predictor", layout="wide")

with st.container():
    st.markdown("""
        <style>
        .main {
            background-color: #0e1117;
            color: #ffffff;
        }
        .stMetric { font-size: 1.3em; }
        .stAlert { font-size: 1.1em; }
        .stTextInput > div > div > input {
            background-color: #1e222a;
            color: white;
        }
        </style>
        """, unsafe_allow_html=True)

    st.title("\U0001F4B0 Morning Stock Performance Forecaster")
    symbol = st.text_input("Enter Stock/ETF Symbol (e.g. TSLA, AAPL, QQQ)")

    us_holidays = holidays.US()
    today = datetime.now().date()
    if today in us_holidays or today.weekday() >= 5:
        st.warning(f"⚠️ Today ({today.strftime('%A')}) is a market holiday or weekend.")

    if symbol:
        with st.spinner("Analyzing market data, trends, and headlines..."):
            try:
                df = fetch_stock_data(symbol)
                if df.empty:
                    st.error("No data fetched. The symbol may be invalid or market is closed.")
                else:
                    df = add_indicators(df)
                    trend_score = float(get_google_trend_score(symbol))
                    headlines = fetch_real_headlines(symbol)
                    sentiment_score = float(get_sentiment_score_from_headlines(headlines))
                    label = get_target(df)
                    model, model_pred, r2, mse = train_model(df)

                    if label is None:
                        st.error("Couldn't find complete 6:30–9:30 AM PT data. Try another symbol or wait for market open.")
                    else:
                        composite_score = compute_weighted_score(df, trend_score, sentiment_score)

                        st.subheader(f"\U0001F4CA Prediction for {symbol.upper()}")
                        st.metric("Composite Profitability Score", f"{composite_score * 100:.2f}%")
                        st.metric("6:30–9:30 AM Price Change", f"{label * 100:.2f}%")
                        st.metric("XGBoost Forecast (Next Close)", f"${model_pred:.2f}")
                        st.metric("Model R² Score", f"{r2:.3f}")

                        st.info("**Top Headlines**")
                        for h in headlines:
                            st.markdown(f"- {h}")

                        plot_indicators(df)

            except Exception as e:
                st.error(f"Error during prediction: {e}")
