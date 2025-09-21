import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import json
from datetime import datetime, timedelta
from pandas.tseries.offsets import BDay

# Page config
st.set_page_config(page_title="Stock Price Predictor", page_icon="📈", layout="wide")

# OpenAI/OpenRouter configuration
from openai import OpenAI
MODEL = "mistralai/mistral-7b-instruct:free"

# Initialize session state for API key validation
if 'api_key_validated' not in st.session_state:
    st.session_state.api_key_validated = False

# Sidebar for API Key Configuration
st.sidebar.title("Configuration")
api_key = st.sidebar.text_input("Enter your OpenRouter API Key:", type="password", help="Get your API key from openrouter.ai")
submit_api = st.sidebar.button("Submit API Key")

def validate_api_key(api_key):
    try:
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
        completion = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": "Test"}],
            max_tokens=10
        )
        return True
    except:
        return False

if submit_api:
    if not api_key:
        st.sidebar.warning("Please enter your OpenRouter API Key to use AI analysis.")
    elif not validate_api_key(api_key):
        st.sidebar.error("Invalid OpenRouter API key. Please check and try again.")
    else:
        st.sidebar.success("API Key validated successfully!")
        st.session_state.api_key_validated = True

if not st.session_state.api_key_validated:
    st.title("Stock Price Predictor")
    st.info("Enter your API key in the sidebar and click 'Submit API Key' to get started.")
    st.stop()

if not api_key:
    st.sidebar.warning("Please enter your OpenRouter API Key to use AI analysis.")
    st.title("Stock Price Predictor")
    st.info("Enter your API key in the sidebar to get started.")
    st.stop()
elif not validate_api_key(api_key):
    st.sidebar.error("Invalid OpenRouter API key. Please check and try again.")
    st.stop()

def get_ai_analysis(stock_data, fundamentals, technical_indicators, ticker):
    """
    Send data to OpenRouter for AI analysis and suggestions using OpenAI client.
    """
    if len(stock_data) < 31:
        return "Error: Insufficient historical data for analysis (minimum 31 days required)."
    
    prompt = f"""
    You are a financial analyst expert in stock market prediction. Format your response using markdown with bold headers and key points.
    
    Stock Ticker: {ticker}
    
    Historical Price Data (last 30 days summary):
    - Current Price: {stock_data['Close'].iloc[-1]:.2f}
    - 5-Day Change: {((stock_data['Close'].iloc[-1] - stock_data['Close'].iloc[-6]) / stock_data['Close'].iloc[-6] * 100):.2f}%
    - 30-Day Change: {((stock_data['Close'].iloc[-1] - stock_data['Close'].iloc[-31]) / stock_data['Close'].iloc[-31] * 100):.2f}%
    - Volume Trend: Average volume {stock_data['Volume'].tail(10).mean():.0f}, Recent: {stock_data['Volume'].iloc[-1]:.0f}
    
    Technical Indicators:
    - 50-Day SMA: {technical_indicators['SMA_50'].iloc[-1]:.2f}
    - 200-Day SMA: {technical_indicators['SMA_200'].iloc[-1]:.2f}
    - RSI (14): {technical_indicators['RSI'].iloc[-1]:.2f} (Overbought >70, Oversold <30)
    - MACD: {technical_indicators['MACD'].iloc[-1]:.4f}, Signal: {technical_indicators['MACD_Signal'].iloc[-1]:.4f}
    
    Fundamental Data:
    - Market Cap: {fundamentals.get('marketCap', 'N/A')}
    - P/E Ratio: {fundamentals.get('trailingPE', 'N/A')}
    - EPS: {fundamentals.get('trailingEps', 'N/A')}
    - Dividend Yield: {fundamentals.get('dividendYield', 'N/A') * 100 if fundamentals.get('dividendYield') else 'N/A'}%
    - Beta: {fundamentals.get('beta', 'N/A')}
    
    Analyze the technical and fundamental aspects. Format your response as follows:

    **Technical Analysis:**
    [Your analysis here with key points in bold]

    **Fundamental Analysis:**
    [Your analysis here with key points in bold]

    **7-Day Price Prediction:**
    - Outlook: [bullish/bearish/neutral]
    - Target Range: [range]
    - Key Drivers: [main factors]

    **Recommendation:**
    - Position: [Buy/Sell/Hold]
    - Key Reasoning: [main points in bold]
    
    Keep response concise, professional, and evidence-based. Make sure to use bold markdown (**) for headers and key points.
    """
    
    try:
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
        
        completion = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.7,
            extra_headers={
                "HTTP-Referer": "localhost:8501",  # Streamlit default port
                "X-Title": "Stock Price Predictor",
            }
        )
        
        return completion.choices[0].message.content
    except Exception as e:
        return f"Error in AI analysis: {str(e)}"

def fetch_stock_data(ticker, period="1y"):
    """
    Fetch stock data using yfinance.
    """
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
        if hist.empty:
            return None, None
        info = stock.info
        return hist, info
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {str(e)}")
        return None, None

def compute_technical_indicators(hist):
    """
    Compute technical indicators.
    """
    if len(hist) < 200:
        return pd.DataFrame()
    
    df = hist.copy()
    
    # Simple Moving Averages
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-10)  # Prevent division by zero
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp1 = df['Close'].ewm(span=12).mean()
    exp2 = df['Close'].ewm(span=26).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
    
    # Check for valid data
    required_columns = ['SMA_50', 'SMA_200', 'RSI', 'MACD', 'MACD_Signal']
    if df[required_columns].tail(1).isna().any().any():
        return pd.DataFrame()
    
    return df

def predict_price_range(hist):
    """
    Simple statistical price range prediction for visualization (7-day horizon).
    """
    last_price = hist['Close'].iloc[-1]
    volatility = hist['Close'].pct_change().tail(30).std() * np.sqrt(7)
    upper_bound = last_price * (1 + volatility)
    lower_bound = last_price * (1 - volatility)
    return lower_bound, upper_bound

# Main app
st.title("📈 Stock Price Predictor")
st.markdown("Enter a stock ticker and click 'Analyze' to view technicals, fundamentals, AI-driven suggestions, and predicted price trends.")

if 'last_ticker' not in st.session_state:
    st.session_state.last_ticker = None

# Input fields in separate lines
ticker = st.text_input("Stock Ticker (e.g., AAPL):").upper()
period = st.selectbox("Data Period:", ["3mo", "6mo", "1y", "2y"], index=2)
analyze_button = st.button("Analyze Stock", key="analyze_button")

if analyze_button and ticker:
    if ticker != st.session_state.last_ticker:
        st.session_state.last_ticker = ticker
        st.cache_data.clear()
    
    progress = st.progress(0)
    with st.spinner(f"Fetching data for {ticker}..."):
        hist, info = fetch_stock_data(ticker, period)
        progress.progress(33)
    
    if hist is not None and info is not None:
        with st.spinner("Computing technical indicators..."):
            tech = compute_technical_indicators(hist)
            progress.progress(66)
        
        if not tech.empty:
            with st.spinner("Generating AI analysis..."):
                ai_response = get_ai_analysis(hist.tail(31), info, tech, ticker)
                progress.progress(100)
            
            if ai_response.startswith("Error"):
                st.error(ai_response)
                st.stop()
            
            # Predict price range
            lower_bound, upper_bound = predict_price_range(hist)
            
            # Tabs
            tab1, tab2, tab3 = st.tabs(["AI Analysis & Suggestions", "Price Chart with Prediction", "Technical & Fundamental Data"])
            
            with tab1:
                st.subheader(f"AI Analysis for {ticker}")
                st.markdown(ai_response, unsafe_allow_html=True)
            
            with tab2:
                st.subheader(f"{ticker} Price Chart with Predicted Range")
                fig = make_subplots(rows=3, cols=1, shared_xaxes=True, 
                                    vertical_spacing=0.05, 
                                    subplot_titles=('Price with Predicted Range', 'Volume', 'RSI'),
                                    row_heights=[0.5, 0.2, 0.3])
                
                # Candlestick
                fig.add_trace(go.Candlestick(x=hist.index,
                                            open=hist['Open'], high=hist['High'],
                                            low=hist['Low'], close=hist['Close'],
                                            name='Price'), row=1, col=1)
                
                # SMAs
                fig.add_trace(go.Scatter(x=tech.index, y=tech['SMA_50'], name='SMA 50', line=dict(color='blue')), row=1, col=1)
                fig.add_trace(go.Scatter(x=tech.index, y=tech['SMA_200'], name='SMA 200', line=dict(color='red')), row=1, col=1)
                
                # Predicted price range
                future_dates = [hist.index[-1] + BDay(i) for i in range(1, 8)]
                fig.add_trace(go.Scatter(x=future_dates, y=[upper_bound]*7, name='Upper Bound', 
                                        line=dict(color='green', dash='dash')), row=1, col=1)
                fig.add_trace(go.Scatter(x=future_dates, y=[lower_bound]*7, name='Lower Bound', 
                                        line=dict(color='red', dash='dash')), row=1, col=1)
                fig.add_trace(go.Scatter(x=future_dates + [future_dates[0]], 
                                        y=[upper_bound]*7 + [lower_bound], 
                                        fill='toself', fillcolor='rgba(0, 255, 0, 0.1)', 
                                        line=dict(color='rgba(255,255,255,0)'), showlegend=False), row=1, col=1)
                
                # Volume
                fig.add_trace(go.Bar(x=hist.index, y=hist['Volume'], name='Volume'), row=2, col=1)
                
                # RSI
                fig.add_trace(go.Scatter(x=tech.index, y=tech['RSI'], name='RSI (14)', line=dict(color='purple')), row=3, col=1)
                fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
                
                fig.update_layout(
                    xaxis_rangeslider_visible=False,
                    height=600,
                    showlegend=True,
                    xaxis=dict(range=[hist.index[0], future_dates[-1] + BDay(1)]),
                )
                st.plotly_chart(fig, use_container_width=True)
                
                st.write(f"**Predicted 7-Day Price Range:** ₹{lower_bound:.2f} - ₹{upper_bound:.2f}")
            
            with tab3:
                st.subheader("Technical Indicators")
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric("Current Price", f"₹{hist['Close'].iloc[-1]:.2f}")
                    st.metric("50-Day SMA", f"₹{tech['SMA_50'].iloc[-1]:.2f}")
                with col_b:
                    st.metric("RSI (14)", f"{tech['RSI'].iloc[-1]:.2f}")
                    st.metric("MACD", f"{tech['MACD'].iloc[-1]:.4f}")
                with col_c:
                    st.metric("200-Day SMA", f"₹{tech['SMA_200'].iloc[-1]:.2f}")
                    st.metric("MACD Signal", f"{tech['MACD_Signal'].iloc[-1]:.4f}")
                
                st.subheader("Fundamental Metrics")
                col_d, col_e, col_f = st.columns(3)
                with col_d:
                    market_cap = info.get('marketCap', None)
                    st.metric("Market Cap", f"₹{market_cap:,.0f}" if market_cap else "N/A")
                with col_e:
                    pe = info.get('trailingPE', None)
                    st.metric("P/E Ratio", f"{pe:.2f}" if isinstance(pe, (int, float)) else "N/A")
                with col_f:
                    beta = info.get('beta', None)
                    st.metric("Beta", f"{beta:.2f}" if isinstance(beta, (int, float)) else "N/A")
                
                eps = info.get('trailingEps', None)
                dividend = info.get('dividendYield', None)
                st.write(f"**EPS:** {eps if eps is not None else 'N/A'}")
                st.write(f"**Dividend Yield:** {f'{dividend * 100:.2f}%' if dividend else 'N/A'}")
        
        else:
            st.warning("Insufficient valid data for technical indicators. Try a longer period or a different ticker.")
    else:
        st.error("Could not fetch data. Please check the ticker symbol.")
elif analyze_button and not ticker:
    st.error("Please enter a stock ticker before analyzing.")

# Footer
st.markdown("---")
st.caption("This app uses yfinance for data and OpenRouter for AI analysis. Predictions are AI-generated and not financial advice.")