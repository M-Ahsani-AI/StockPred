import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import yfinance as yf
import ta
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Fetch stock data based on the ticker, period, and interval
def fetch_stock_data(ticker, period, interval):
    end_date = datetime.now()
    if period == '1wk':
        start_date = end_date - timedelta(days=7)
        data = yf.download(ticker, start=start_date, end=end_date, interval=interval)
    else:
        data = yf.download(ticker, period=period, interval=interval)
    return data

# Process data to ensure it is timezone-aware and has the correct format
def process_data(data):
    if data.index.tzinfo is None:
        data.index = data.index.tz_localize('UTC')
    data.index = data.index.tz_convert('US/Eastern')
    data.reset_index(inplace=True)
    data.rename(columns={'Date': 'Datetime'}, inplace=True)
    return data

def add_technical_indicators(data):
    close_series = data['Close'].squeeze()
    data['SMA_20'] = ta.trend.sma_indicator(close_series, window=20)
    data['EMA_20'] = ta.trend.ema_indicator(close_series, window=20)
    return data

# Train Random Forest model
def train_random_forest(data, future_steps=7, test_split=0.3):
    close_prices = data['Close'].values
    features = np.arange(len(close_prices)).reshape(-1, 1)

    X_train, X_test, y_train, y_test = train_test_split(features, close_prices, test_size=test_split, shuffle=False)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train.ravel())  # Use ravel() to ensure y is 1D

    y_test_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_test_pred)
    mae = mean_absolute_error(y_test, y_test_pred)
    rmse = np.sqrt(mse)

    mean_actual = np.mean(y_test)
    if mean_actual != 0:
        mse_percent = (mse / mean_actual) * 100
        mae_percent = (mae / mean_actual) * 100
        rmse_percent = (rmse / mean_actual) * 100
    else:
        mse_percent = float('inf')
        mae_percent = float('inf')
        rmse_percent = float('inf')

    print(f"Evaluation RF Metrics on Test Set :")
    print(f"- Mean Squared Error (MSE) : {mse_percent:.2f}%")
    print(f"- Mean Absolute Error (MAE) : {mae_percent:.2f}%")
    print(f"- Root Mean Squared Error (RMSE) : {rmse_percent:.2f}%")

    last_feature = features[-1, 0]  # Extract scalar value explicitly
    future_features = np.arange(last_feature + 1, last_feature + 1 + future_steps).reshape(-1, 1)
    future_predictions = model.predict(future_features)
    return future_predictions, mse_percent, mae_percent, rmse_percent

# Set up Streamlit page layout
st.set_page_config(layout="wide")
st.title('Prediksi Harga Saham PT. Gudang Garam Menggunakan Algoritma Random Forest')

# Sidebar for user input parameters
st.sidebar.header('Chart Parameters')
ticker = st.sidebar.text_input('Ticker', 'GGRM.JK')
time_period = st.sidebar.selectbox('Time Period', ['1d', 'max'])
chart_type = st.sidebar.selectbox('Chart Type', ['Candlestick'])
indicators = st.sidebar.multiselect('Technical Indicators', ['SMA 20', 'EMA 20'])
show_predictions = st.sidebar.checkbox('Show Predictions', value=True)

# Mapping time period to intervals
interval_mapping = {
    '1d': '1m',
    '1wk': '30m',
    '1mo': '1d',
    '1y': '1wk',
    'max': '1wk'
}

# Main content area
if st.sidebar.button('Update'):
    data = fetch_stock_data(ticker, time_period, interval_mapping[time_period])
    if data.empty:
        st.error("No data found. Try a different ticker or time period.")
    else:
        data = process_data(data)

        future_predictions = []
        future_dates = []
        model_metrics = {}

        try:
            future_predictions, mse_percent, mae_percent, rmse_percent = train_random_forest(data, future_steps=7)
            future_dates = [data['Datetime'].iloc[-1] + timedelta(days=i) for i in range(1, len(future_predictions) + 1)]
            model_metrics = {
                'Mean Squared Error (%)': mse_percent,
                'Mean Absolute Error (%)': mae_percent,
                'Root Mean Squared Error (%)': rmse_percent
            }
        except Exception as e:
            st.error(f"Error in Random Forest model: {e}")

        fig = go.Figure()
        if chart_type == 'Candlestick':
            fig.add_trace(go.Candlestick(x=data['Datetime'],
                                         open=data['Open'],
                                         high=data['High'],
                                         low=data['Low'],
                                         close=data['Close']))
        else:
            fig = px.line(data, x='Datetime', y='Close')

        if show_predictions and len(future_predictions) > 0 and len(future_dates) > 0:
            fig.add_trace(go.Scatter(x=future_dates, y=future_predictions, mode='lines', name='RF Predictions'))

        fig.update_layout(title=f'{ticker} {time_period.upper()} Chart with Predictions',
                          xaxis_title='Time',
                          yaxis_title='Price (IDR)',
                          height=600)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader('Historical Data')
        st.dataframe(data[['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume']])

        if len(future_predictions) > 0 and len(future_dates) > 0:
            st.subheader('Random Forest Predictions')
            prediction_df = pd.DataFrame({'Date': future_dates, 'Predicted Close': future_predictions})
            st.dataframe(prediction_df)

            st.subheader('Random Forest Metrics')
            st.json(model_metrics)
