import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings

warnings.filterwarnings("ignore")

# Title
st.title("Electricity Consumption Forecasting App")

# Load the dataset
df = pd.read_csv("data.csv")

# Convert date to datetime
df['date'] = pd.to_datetime(df['date'])

# Sort by date
df = df.sort_values('date')

# Set index
df.set_index('date', inplace=True)

# Show raw data
st.subheader("Raw Data")
st.write(df.head())

# Optional: Line chart of consumption
st.subheader("Electricity Consumption Over Time")
st.line_chart(df['consumption'])

# Optional: Show temperature trend
if 'temperature' in df.columns:
    st.subheader("Temperature Over Time")
    st.line_chart(df['temperature'])

# Train-test split
train = df['consumption'][:-14]
test = df['consumption'][-14:]

# Fit SARIMA model
model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7))
results = model.fit()

# Forecast next 14 days
forecast = results.predict(start=len(train), end=len(train) + len(test) - 1, dynamic=False)

# Plot forecast vs actual
st.subheader("Forecast vs Actual")
fig, ax = plt.subplots()
test.plot(ax=ax, label='Actual')
forecast.plot(ax=ax, label='Forecast')
plt.legend()
st.pyplot(fig)

# Optional: Forecast future (next 14 days beyond last date)
st.subheader("Forecast for Next 14 Days")
future_model = SARIMAX(df['consumption'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 7))
future_results = future_model.fit()
future_forecast = future_results.predict(start=len(df), end=len(df) + 13, dynamic=False)

st.line_chart(future_forecast)
