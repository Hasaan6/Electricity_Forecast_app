import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from datetime import timedelta

# App title
st.title("🔌 Electricity Consumption Forecasting App")
st.markdown("Predict electricity usage using SARIMAX model (based on 2025 real-time simulated data)")

# Load data
@st.cache
def load_data():
    df = pd.read_csv("data.csv")
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    df.set_index('date', inplace=True)
    return df

df = load_data()

# Sidebar
st.sidebar.header("🔍 Options")
show_temp = st.sidebar.checkbox("Show Temperature")
show_household = st.sidebar.checkbox("Show Household Info")

# Show data
st.subheader("📊 Raw Data (2025)")
st.write(df.head())

# Optional plots
if show_temp:
    st.subheader("🌡️ Temperature Over Time")
    st.line_chart(df['temperature'])

if show_household:
    st.subheader("🏘️ Household Info Over Time")
    st.line_chart(df['household'])

# Plot electricity consumption
st.subheader("⚡ Electricity Consumption (kWh)")
st.line_chart(df['consumption'])

# SARIMAX Model
st.subheader("📈 Forecasting with SARIMAX")
try:
    model = SARIMAX(df['consumption'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 7))
    results = model.fit()

    # Forecast 30 days ahead
    forecast = results.get_forecast(steps=30)
    forecast_index = [df.index[-1] + timedelta(days=i) for i in range(1, 31)]
    forecast_series = pd.Series(forecast.predicted_mean.values, index=forecast_index)

    # Plot forecast
    fig, ax = plt.subplots()
    df['consumption'].plot(ax=ax, label="Historical")
    forecast_series.plot(ax=ax, label="Forecast", linestyle="--")
    ax.set_xlabel("Date")
    ax.set_ylabel("Electricity (kWh)")
    ax.set_title("30-Day Electricity Forecast")
    ax.legend()
    st.pyplot(fig)

    st.success("✅ Forecast generated successfully!")
except Exception as e:
    st.error(f"❌ Model failed: {e}")
