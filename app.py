import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Load trained model
try:
    model = joblib.load("sarimax_model.pkl")
except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.stop()

# Page settings
st.set_page_config(
    page_title="Gold Price Prediction",
    page_icon="💰",
    layout="wide"
)

st.markdown("""
<style>

/* Title container */
.glow-title{
    text-align:center;
    padding:40px;
    border-radius:15px;
    position:relative;
}

/* Glow behind title */
.glow-title::before{
    content:"";
    position:absolute;
    top:-20px;
    left:50%;
    transform:translateX(-50%);
    width:600px;
    height:200px;

    background:radial-gradient(
        circle,
        rgba(255,215,0,0.35) 0%,
        rgba(255,215,0,0.15) 40%,
        transparent 70%
    );

    filter:blur(60px);
    z-index:-1;
}

/* Title styling */
.glow-title h1{
    color:#FFD700;
    font-size:48px;
    font-weight:700;
}

/* Subtitle */
.glow-title p{
    color:#e5e7eb;
    font-size:18px;
}

</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>

/* Smooth slider movement */
.stSlider > div > div > div > div {
    transition: all 0.3s ease-in-out;
}

/* Slider track */
.stSlider > div > div > div {
    background: linear-gradient(90deg,#FFD700,#FFA500);
    height: 6px;
    border-radius: 10px;
}

/* Slider thumb */
.stSlider > div > div > div > div > div {
    background-color: gold;
    border: 3px solid white;
    width: 20px;
    height: 20px;
    box-shadow: 0px 0px 10px gold;
}

/* Hover glow animation */
.stSlider > div > div > div > div > div:hover {
    transform: scale(1.3);
    box-shadow: 0px 0px 15px gold;
}

/* Smooth number update */
.stSlider span {
    transition: all 0.2s ease-in-out;
}

</style>
""", unsafe_allow_html=True)


# ---------- Centered Header ----------
st.markdown("""
<div class="glow-title">

<h1>Gold Price Prediction Dashboard</h1>

<p>Predict future gold prices using financial market indicators</p>

</div>
""", unsafe_allow_html=True)

st.divider()

# ---------- Layout ----------
col1, col2 = st.columns([2,1])

with col1:

    st.subheader("Project Overview")

    st.write("""
This interactive dashboard predicts **Gold Prices** using a trained **SARIMAX Time-Series Model**.

Gold prices are influenced by several economic indicators:

• Silver Price  
• Crude Oil Price  
• S&P500 Index  
• Forecast Horizon

Adjust the sliders in the sidebar to simulate different **market conditions**.
The graph will automatically update based on your inputs.
""")

with col2:

    st.info("""
**Model Used:** SARIMAX  
**Data Source:** Yahoo Finance  
**Prediction Type:** Time Series Forecast
""")

st.divider()

# ---------- Sidebar ----------
st.sidebar.image("https://media.tenor.com/ggczMlGicXAAAAAC/gold-coins.gif")
st.sidebar.header("Market Indicators")

silver = st.sidebar.slider("Silver Price", 20.0, 100.0, 30.0)
oil = st.sidebar.slider("Oil Price", 50.0, 120.0, 70.0)
sp500 = st.sidebar.slider("S&P500 Index", 4000.0, 7000.0, 5000.0)

future_days = st.sidebar.slider("Forecast Days", 10, 120, 30)

st.sidebar.markdown("---")
st.sidebar.write("Adjust values to simulate market changes.")

# ---------- Generate Future Inputs ----------
future_silver = np.linspace(silver*0.98, silver*1.02, future_days)
future_oil = np.linspace(oil*0.98, oil*1.02, future_days)
future_sp500 = np.linspace(sp500*0.99, sp500*1.01, future_days)

exog_future = pd.DataFrame({
    "Silver": future_silver,
    "Oil": future_oil,
    "SP500": future_sp500
})

# ---------- Forecast ----------
forecast = model.forecast(steps=future_days, exog=exog_future)

# ---------- Graph Section ----------
st.subheader("Future Gold Price Forecast")

fig, ax = plt.subplots(figsize=(10,5))

ax.plot(
    forecast.values,
    color="gold",
    linewidth=3,
    marker="o",
    label="Predicted Gold Price"
)

ax.set_title("Gold Price Forecast")
ax.set_xlabel("Future Days")
ax.set_ylabel("Gold Price (USD)")
ax.legend()

st.pyplot(fig)

# ---------- Metrics ----------
predict_button = st.button("🔮 Predict Gold Price")

if predict_button:

    with st.spinner("Calculating Gold Price Prediction..."):
        forecast = model.forecast(steps=future_days, exog=exog_future)

    predicted_price = forecast.iloc[0]

    st.markdown(
    f"""
    <div style="
        background: linear-gradient(90deg,#FFD700,#FFC300);
        padding:30px;
        border-radius:15px;
        text-align:center;
        margin-top:20px;
        box-shadow:0px 5px 15px rgba(0,0,0,0.2);
        ">
        <h2 style="color:black;"> Predicted Gold Price </h2>
        <h1 style="color:black; font-size:50px;">
        ${predicted_price:.2f}
        </h1>
    </div>
    """,
    unsafe_allow_html=True
    )

    st.divider()

    col1, col2 = st.columns(2)

    col1.metric("Maximum Forecast Price", f"${forecast.max():.2f}")
    col2.metric("Minimum Forecast Price", f"${forecast.min():.2f}")

    # Animated graph
    fig, ax = plt.subplots(figsize=(10,5))

    for i in range(len(forecast)):
        ax.clear()
        ax.plot(
            forecast[:i],
            color="gold",
            linewidth=3,
            marker="o"
        )
        ax.set_title("Future Gold Price Forecast")
        ax.set_xlabel("Future Days")
        ax.set_ylabel("Gold Price (USD)")
        st.pyplot(fig)

# ---------- Model Explanation ----------
st.header("Model Explanation")

st.write("""
### SARIMAX Model

SARIMAX (Seasonal AutoRegressive Integrated Moving Average with Exogenous Variables) 
is an advanced time-series forecasting model.

Unlike traditional ARIMA models, SARIMAX can incorporate **external economic indicators**, 
which significantly improves forecasting performance.

The model uses:

• Historical gold prices  
• Silver market trends  
• Oil price fluctuations  
• Stock market performance

After testing multiple models including:

- ARIMA
- Random Forest
- XGBoost

SARIMAX produced the **best prediction accuracy**, making it the final deployed model.
""")

st.divider()

st.caption("Major Project — Gold Price Prediction using Time Series Modeling")
