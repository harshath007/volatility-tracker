import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from ta.trend import EMAIndicator, MACD, ADXIndicator, CCIIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.momentum import RSIIndicator
from ta.volume import OnBalanceVolumeIndicator
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob

# Download required nltk data
nltk.download('vader_lexicon')

# Initialize VADER sentiment analyzer
vader = SentimentIntensityAnalyzer()

st.set_page_config(page_title="Volatility & Sentiment Tracker (No APIs)", layout="wide")
st.title("📈 Volatility & Sentiment Tracker (No API Required)")

symbol = st.text_input("Enter Stock/ETF Symbol (e.g. AAPL, TSLA)", "AAPL").upper().strip()
if not symbol:
    st.stop()

@st.cache_data(ttl=1800)
def load_data(ticker):
    end = datetime.now()
    start = end - timedelta(days=365)
    df = yf.download(ticker, start=start, end=end, interval='1d')
    df.dropna(subset=['Close'], inplace=True)

    if df.empty:
        return pd.DataFrame()  # return empty if no data

    close = df['Close'].astype(float).dropna()
    high = df['High'].astype(float).dropna()
    low = df['Low'].astype(float).dropna()
    volume = df['Volume'].astype(float).dropna()

    try:
        df['EMA9'] = EMAIndicator(close, window=9).ema_indicator()
        df['RSI'] = RSIIndicator(close, window=14).rsi()
        df['MACD'] = MACD(close).macd()
        df['ADX'] = ADXIndicator(high, low, close).adx()
        df['CCI'] = CCIIndicator(high, low, close).cci()
        bb = BollingerBands(close, window=20, window_dev=2)
        df['BB_width'] = (bb.bollinger_hband() - bb.bollinger_lband()) / close
        df['ATR'] = AverageTrueRange(high, low, close).average_true_range()
        df['OBV'] = OnBalanceVolumeIndicator(close, volume).on_balance_volume()
        df['Volatility'] = close.pct_change().rolling(10).std() * np.sqrt(252)
    except Exception as e:
        st.error(f"Error calculating technical indicators: {e}")
        return pd.DataFrame()  # Return empty to signal failure

    return df


df = load_data(symbol)
if df.empty:
    st.error("No data found for this symbol.")
    st.stop()

# Dummy sentiment source: use latest 10 headlines from Yahoo Finance (free scraping)
@st.cache_data(ttl=1800)
def fetch_headlines(ticker):
    import requests
    from bs4 import BeautifulSoup

    url = f"https://finance.yahoo.com/quote/{ticker}?p={ticker}"
    try:
        r = requests.get(url, headers={"User-Agent":"Mozilla/5.0"})
        soup = BeautifulSoup(r.text, 'html.parser')
        # Look for headlines under "Latest News" section
        headlines = []
        for item in soup.select('h3'):
            txt = item.get_text()
            if len(txt) > 10:
                headlines.append(txt)
        return headlines[:10]
    except Exception:
        return []

headlines = fetch_headlines(symbol)
if not headlines:
    st.warning("Could not fetch recent news headlines for sentiment. Using sample sentences.")
    headlines = [
        "Stock rallies on strong earnings report.",
        "Market uncertainty affects tech stocks.",
        "Analysts upgrade the company outlook."
    ]

# Calculate sentiment scores
vader_scores = [vader.polarity_scores(h)['compound'] for h in headlines]
textblob_scores = [TextBlob(h).sentiment.polarity for h in headlines]

avg_vader = np.mean(vader_scores)
avg_textblob = np.mean(textblob_scores)
composite_sentiment = np.mean([avg_vader, avg_textblob])

# Volatility calculations
daily_vol = df['Volatility'].iloc[-1]
weekly_vol = df['Close'].pct_change().rolling(5).std().iloc[-1] * np.sqrt(252)
monthly_vol = df['Close'].pct_change().rolling(21).std().iloc[-1] * np.sqrt(252)

# UI Tabs
tabs = st.tabs(["Volatility", "Sentiment", "Indicators & Data"])

with tabs[0]:
    st.header(f"Volatility Forecasts for {symbol}")
    st.metric("Daily Volatility", f"{daily_vol*100:.2f} %")
    st.metric("Weekly Volatility", f"{weekly_vol*100:.2f} %")
    st.metric("Monthly Volatility", f"{monthly_vol*100:.2f} %")
    st.line_chart(df['Volatility'])

with tabs[1]:
    st.header(f"Sentiment Analysis from Latest Headlines")
    st.write("Sample Headlines:")
    for h in headlines:
        st.write(f"- {h}")
    st.metric("VADER Sentiment", f"{avg_vader:+.2f}")
    st.metric("TextBlob Sentiment", f"{avg_textblob:+.2f}")
    st.metric("Composite Sentiment", f"{composite_sentiment:+.2f}")
    if composite_sentiment > 0.05:
        st.success("Overall Sentiment: Positive 👍")
    elif composite_sentiment < -0.05:
        st.error("Overall Sentiment: Negative 👎")
    else:
        st.info("Overall Sentiment: Neutral ⚖️")

with tabs[2]:
    st.header("Latest Indicators & Data")
    st.write(df.tail(10)[[
        'Close', 'EMA9', 'RSI', 'MACD', 'ADX', 'CCI', 'BB_width', 'ATR', 'OBV', 'Volatility'
    ]])

st.caption("Volatility calculated from historic prices. Sentiment from free VADER and TextBlob NLP methods applied to latest headlines scraped from Yahoo Finance.")

