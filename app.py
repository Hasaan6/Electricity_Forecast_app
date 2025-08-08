import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import numpy as np

st.set_page_config(page_title="Electricity Forecast App", layout="wide")
st.title("🔌 Electricity Consumption Forecasting")

# Load data
df = pd.read_csv("data.csv", parse_dates=['date'])
df.set_index('date', inplace=True)

st.subheader("📊 Raw Data")
st.dataframe(df)

# Plot original consumption
st.subheader("📈 Electricity Consumption Over Time")
st.line_chart(df['consumption'])

# ARIMA Forecasting
st.subheader("🔮 Forecasted Consumption")

# Fit ARIMA model
model = ARIMA(df['consumption'], order=(1, 1, 1))
model_fit = model.fit()

# Forecast next 7 days
forecast_steps = 7
forecast = model_fit.forecast(steps=forecast_steps)

# Create future date index
last_date = df.index[-1]
future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_steps)

# Combine into forecast DataFrame
forecast_df = pd.DataFrame({'Forecast': forecast}, index=future_dates)

# Plot forecast
st.line_chart(forecast_df)

# Show forecast table
st.write("### Forecasted Values")
st.dataframe(forecast_df)
