import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import yfinance as yf

# ---------- LOAD MODEL ----------
model = joblib.load("sarimax_model.pkl")

# ---------- PAGE SETTINGS ----------
st.set_page_config(
    page_title="Gold Price Prediction",
    page_icon="💰",
    layout="wide"
)

# ---------- CSS STYLING ----------
st.markdown("""
<style>

/* Glow Title */
.glow-title{
    text-align:center;
    padding:40px;
    border-radius:15px;
    position:relative;
}

.glow-title::before{
    content:"";
    position:absolute;
    top:-20px;
    left:50%;
    transform:translateX(-50%);
    width:600px;
    height:200px;
    background:radial-gradient(circle, rgba(255,215,0,0.35) 0%, rgba(255,215,0,0.15) 40%, transparent 70%);
    filter:blur(60px);
    z-index:-1;
}

.glow-title h1{
    color:#FFD700;
    font-size:48px;
    font-weight:700;
}

.glow-title p{
    color:#e5e7eb;
    font-size:18px;
}

/* Slider Styling */
.stSlider > div > div > div > div {
    transition: all 0.3s ease-in-out;
}

.stSlider > div > div > div {
    background: linear-gradient(90deg,#FFD700,#FFA500);
    height: 6px;
    border-radius: 10px;
}

.stSlider > div > div > div > div > div {
    background-color: gold;
    border: 3px solid white;
    width: 20px;
    height: 20px;
    box-shadow: 0px 0px 10px gold;
}

.stSlider > div > div > div > div > div:hover {
    transform: scale(1.3);
    box-shadow: 0px 0px 15px gold;
}

</style>
""", unsafe_allow_html=True)

# ---------- HEADER ----------
st.markdown("""
<div class="glow-title">
<h1>Gold Price Prediction Dashboard</h1>
<p>Predict future gold prices using financial market indicators</p>
</div>
""", unsafe_allow_html=True)

st.divider()

# ---------- LAYOUT ----------
col1, col2 = st.columns([2,1])

with col1:
    st.subheader("Project Overview")
    st.write("""
This dashboard predicts **Gold Prices** using a **SARIMAX Model**.

Factors affecting gold price:
• Silver Price  
• Crude Oil Price  
• S&P500 Index  
• Forecast Horizon  
""")

with col2:
    st.info("""
**Model:** SARIMAX  
**Data:** Yahoo Finance  
**Type:** Time Series Forecast
""")

st.divider()

# ---------- SIDEBAR ----------
@st.cache_data
def get_live_data():
    gold = yf.download("GC=F", period="1d", interval="1m")
    silver = yf.download("SI=F", period="1d", interval="1m")
    oil = yf.download("CL=F", period="1d", interval="1m")
    sp500 = yf.download("^GSPC", period="1d", interval="1m")

    return {
        "gold": gold["Close"].iloc[-1],
        "silver": silver["Close"].iloc[-1],
        "oil": oil["Close"].iloc[-1],
        "sp500": sp500["Close"].iloc[-1]
    }

live_data = get_live_data()

st.sidebar.header("📡 Live Market Indicators")

silver = st.sidebar.slider(
    "Silver Price",
    20.0, 100.0,
    float(live_data["silver"])
)

oil = st.sidebar.slider(
    "Oil Price",
    50.0, 120.0,
    float(live_data["oil"])
)

sp500 = st.sidebar.slider(
    "S&P500 Index",
    4000.0, 7000.0,
    float(live_data["sp500"])
)

# ---------- FUTURE INPUT ----------
future_days = st.sidebar.slider("Forecast Days", 1, 30, 7)
future_silver = np.linspace(silver*0.98, silver*1.02, future_days)
future_oil = np.linspace(oil*0.98, oil*1.02, future_days)
future_sp500 = np.linspace(sp500*0.99, sp500*1.01, future_days)

exog_future = pd.DataFrame({
    "Silver": future_silver,
    "Oil": future_oil,
    "SP500": future_sp500
})

# ---------- FORECAST ----------
forecast = model.forecast(steps=future_days, exog=exog_future)

# ---------- GRAPH ----------
# ---------- GRAPH (TRAIN + ACTUAL + FORECAST) ----------
# ---------- LOAD DATA ----------
data = pd.read_excel("gold_prediction_dataset.xlsx")

# Make sure date column exists (change if needed)
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Target column (change if needed)
target = data['Gold']

# Train-test split
train = target[:-30]
test = target[-30:]
fig, ax = plt.subplots(figsize=(10,5))



# Plot forecast line
ax.plot(forecast.index, forecast, label="SARIMAX Predicted", color="green")

# Dot placeholder (empty initially)
dot, = ax.plot([], [], 'ro', markersize=8, label="Predicted Point")

ax.set_title("Gold Price Prediction using SARIMAX")
ax.legend()

graph_placeholder = st.pyplot(fig)

# ---------- BUTTON ACTION ----------
if st.button("🔮 Predict Gold Price"):

    predicted_price = forecast.iloc[0]
    predicted_index = forecast.index[0]

    # Add red dot on predicted point
    ax.plot(predicted_index, predicted_price, 'ro', markersize=10)

    st.pyplot(fig)  # update graph

    # Show prediction card
    st.markdown(f"""
    <div style="
        background: linear-gradient(90deg,#FFD700,#FFC300);
        padding:30px;
        border-radius:15px;
        text-align:center;
        margin-top:20px;
        box-shadow:0px 5px 15px rgba(0,0,0,0.2);
    ">
        <h2 style="color:black;">Predicted Gold Price</h2>
        <h1 style="color:black; font-size:50px;">
        ${predicted_price:.2f}
        </h1>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    col1.metric("Max Price", f"${forecast.max():.2f}")
    col2.metric("Min Price", f"${forecast.min():.2f}")
# ---------- PREDICTION ----------
if st.button("🔮 Predict Gold Price"):

    predicted_price = forecast.iloc[0]

    st.markdown(f"""
    <div style="
        background: linear-gradient(90deg,#FFD700,#FFC300);
        padding:30px;
        border-radius:15px;
        text-align:center;
        margin-top:20px;
        box-shadow:0px 5px 15px rgba(0,0,0,0.2);
    ">
        <h2 style="color:black;">Predicted Gold Price</h2>
        <h1 style="color:black; font-size:50px;">
        ${predicted_price:.2f}
        </h1>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    col1.metric("Max Price", f"${forecast.max():.2f}")
    col2.metric("Min Price", f"${forecast.min():.2f}")

# ---------- MODEL INFO ----------
st.header("Model Explanation")

st.write("""
SARIMAX is an advanced time-series model that includes external variables.

It improves accuracy by using:
• Gold history  
• Silver trends  
• Oil prices  
• Stock market data  
""")

st.divider()
st.caption("Gold Price Prediction Project 🚀")
