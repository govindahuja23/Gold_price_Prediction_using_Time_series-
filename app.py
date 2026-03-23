import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# ---------- LOAD MODEL ----------
try:
    model = joblib.load("sarimax_model.pkl")
except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.stop()

# ---------- PAGE SETTINGS ----------
st.set_page_config(
    page_title="Gold Price Prediction",
    page_icon="💰",
    layout="wide"
)

# ---------- HEADER ----------
st.title("💰 Gold Price Prediction Dashboard")
st.write("Predict future gold prices using SARIMAX model")

st.divider()

# ---------- SIDEBAR ----------
st.sidebar.header("Market Indicators")

silver = st.sidebar.slider("Silver Price", 20.0, 100.0, 30.0)
oil = st.sidebar.slider("Oil Price", 50.0, 120.0, 70.0)
sp500 = st.sidebar.slider("S&P500 Index", 4000.0, 7000.0, 5000.0)

future_days = st.sidebar.slider("Forecast Days", 10, 120, 30)

# ---------- FUTURE INPUT ----------
future_silver = np.linspace(silver*0.98, silver*1.02, future_days)
future_oil = np.linspace(oil*0.98, oil*1.02, future_days)
future_sp500 = np.linspace(sp500*0.99, sp500*1.01, future_days)

exog_future = pd.DataFrame({
    "Silver": future_silver,
    "Oil": future_oil,
    "SP500": future_sp500
})

# ---------- FORECAST ----------
future_forecast = model.forecast(steps=future_days, exog=exog_future)

# ---------- LOAD DATA ----------
st.subheader("📊 Gold Price Prediction using SARIMAX")

data = pd.read_excel("gold_prediction_dataset.xlsx")

# Fix CSV-style data
data = data.iloc[:,0].str.split(",", expand=True)
data.columns = ["Date","Gold","Silver","Oil","SP500"]

# Convert types
data["Date"] = pd.to_datetime(data["Date"])
data.set_index("Date", inplace=True)

data["Gold"] = pd.to_numeric(data["Gold"])

# ---------- TRAIN TEST SPLIT ----------
series = data["Gold"]

split = int(len(series) * 0.8)

train = series[:split]
test = series[split:]

# ---------- FORECAST (HISTORICAL) ----------
forecast = model.forecast(steps=len(test))

# Fix index (REMOVE ZIG-ZAG)
forecast = pd.Series(forecast, index=test.index)

# ---------- PLOT ----------
fig, ax = plt.subplots(figsize=(10,5))

ax.plot(train.index, train, label="Train", color="blue")
ax.plot(test.index, test, label="Actual", color="orange")
ax.plot(forecast.index, forecast, label="SARIMAX Predicted", color="green")

ax.set_title("Gold Price Prediction using SARIMAX")
ax.set_xlabel("Date")
ax.set_ylabel("Gold Price (USD)")
ax.legend()
ax.grid(alpha=0.3)

st.pyplot(fig)

# ---------- PREDICTION BUTTON ----------
if st.button("🔮 Predict Future Gold Price"):

    predicted_price = future_forecast.iloc[0]

    st.success(f"Predicted Gold Price: ${predicted_price:.2f}")

    st.write("Forecast Summary:")
    st.write(f"Max Price: ${future_forecast.max():.2f}")
    st.write(f"Min Price: ${future_forecast.min():.2f}")

    # Simple future graph (NO zig-zag)
    fig2, ax2 = plt.subplots(figsize=(10,5))

    ax2.plot(range(len(future_forecast)), future_forecast, color="green")

    ax2.set_title("Future Gold Price Forecast")
    ax2.set_xlabel("Future Days")
    ax2.set_ylabel("Gold Price")

    st.pyplot(fig2)

# ---------- FOOTER ----------
st.divider()
st.caption("Gold Price Prediction using SARIMAX Model")
