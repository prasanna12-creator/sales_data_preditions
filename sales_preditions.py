import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# --- STREAMLIT UI SETUP ---
st.set_page_config(page_title="Sales Forecasting", layout="wide")
st.title("📈 Predictive Analytics: Revenue Forecasting")
st.markdown("Using Holt-Winters Exponential Smoothing to forecast future sales trends based on historical data.")

# --- 1. DATA LOADING ---
file_path = r"C:\Users\prasa\OneDrive\Documents\sales_data"
files_list = os.listdir(file_path)

csv_file = None
for file in files_list:
    if file.endswith('.csv'):
        csv_file = os.path.join(file_path, file)
        break

@st.cache_data
def load_csv(filepath):
    if filepath is None:
        return pd.DataFrame()
    
    # Using encoding_errors='ignore' based on the issues you fixed previously!
    dataframe = pd.read_csv(filepath, encoding='utf-8', encoding_errors='ignore')
    
    # Parse the specific Sale_Date column
    # dayfirst=True is a safe fallback if your dates are formatted DD-MM-YYYY
    dataframe["Sale_Date"] = pd.to_datetime(dataframe["Sale_Date"], dayfirst=True, errors='coerce')
    
    # Drop rows where the date couldn't be parsed
    dataframe = dataframe.dropna(subset=['Sale_Date'])
    
    return dataframe

df = load_csv(csv_file)

if df.empty:
    st.error("No CSV found or data could not be loaded. Please check your folder path.")
    st.stop()

# --- 2. TIME-SERIES PREPROCESSING ---
st.header("1. Historical Data Preprocessing")

# We use your exact 'Sale_Date' and 'Sales_Amount' columns
# Group by Date to get total daily revenue
daily_sales = df.groupby('Sale_Date')['Sales_Amount'].sum().reset_index()

# Set the date as the index (mandatory for time-series algorithms)
daily_sales.set_index('Sale_Date', inplace=True)

# Resample to ensure every single day is accounted for. 
# If there were 0 sales on a Tuesday, this explicitly fills it with $0.
ts_data = daily_sales.resample('D').sum().fillna(0)

st.write("Aggregated Daily Sales Timeline:")
st.line_chart(ts_data['Sales_Amount'])

# --- 3. TRAIN / TEST SPLIT ---
# We hold back the final 30 days of data to test the model's accuracy
test_days = 30

# Safety check: ensure we have enough data to split
if len(ts_data) <= test_days:
    st.error(f"Not enough historical data to test. You need more than {test_days} days of data.")
    st.stop()

train_data = ts_data.iloc[:-test_days]
test_data = ts_data.iloc[-test_days:]

# --- 4. MODEL TRAINING & EVALUATION ---
st.header("2. Model Evaluation (Testing on the last 30 days of history)")

# Fit the Holt-Winters Exponential Smoothing model
# We add a tiny shift (0.01) to avoid divide-by-zero math errors on days with exactly $0 sales
model = ExponentialSmoothing(
    train_data['Sales_Amount'] + 0.01, 
    trend='add', 
    seasonal=None, 
    initialization_method="estimated"
)
fitted_model = model.fit()

# Predict the 30-day test period
predictions = fitted_model.forecast(steps=test_days)

# Calculate Accuracy (RMSE)
rmse = np.sqrt(mean_squared_error(test_data['Sales_Amount'], predictions))
st.metric(label="Model Error (RMSE)", value=f"${rmse:,.2f}", delta="Lower is better", delta_color="inverse")

# --- 5. FUTURE FORECASTING ---
st.header("3. Future Revenue Forecast (Next 30 Days)")

# Retrain the model on the ENTIRE dataset so it uses the most recent data to predict the actual future
final_model = ExponentialSmoothing(
    ts_data['Sales_Amount'] + 0.01, 
    trend='add', 
    seasonal=None, 
    initialization_method="estimated"
).fit()

# Predict 30 days into the unknown future
future_forecast = final_model.forecast(steps=30)
future_dates = pd.date_range(start=ts_data.index[-1] + pd.Timedelta(days=1), periods=30)
future_df = pd.DataFrame({'Forecast_Amount': future_forecast.values}, index=future_dates)

# --- 6. PLOTLY VISUALIZATION ---
fig = go.Figure()

# Add Historical Data Line (Blue)
fig.add_trace(go.Scatter(
    x=ts_data.index, y=ts_data['Sales_Amount'],
    mode='lines', name='Historical Actuals', line=dict(color='blue')
))

# Add Future Forecast Line (Orange & Dotted)
fig.add_trace(go.Scatter(
    x=future_df.index, y=future_df['Forecast_Amount'],
    mode='lines', name='Future Forecast', line=dict(color='orange', dash='dot')
))

fig.update_layout(
    title='30-Day Predictive Revenue Forecast',
    xaxis_title='Date',
    yaxis_title='Revenue ($)',
    hovermode='x unified'
)

st.plotly_chart(fig, use_container_width=True)

with st.expander("🔍 View Raw Future Forecast Data"):
    st.dataframe(future_df.style.format("${:,.2f}"))