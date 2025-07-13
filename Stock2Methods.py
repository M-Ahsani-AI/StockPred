import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import ta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import LambdaCallback
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

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

# Add simple moving average (SMA) and exponential moving average (EMA) indicators
def add_technical_indicators(data):
    close_series = data['Close'].squeeze()
    data['SMA_20'] = ta.trend.sma_indicator(close_series, window=20)
    data['EMA_20'] = ta.trend.ema_indicator(close_series, window=20)
    return data

# Train LSTM model
def train_lstm(data, future_steps=7, epochs=50, test_split=0.3):
    close_prices = data['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    close_prices_scaled = scaler.fit_transform(close_prices)
    
    seq_length = min(60, len(close_prices_scaled) - 1)
    if len(close_prices_scaled) <= seq_length:
        raise ValueError("Not enough data for LSTM. Increase dataset size or choose a longer time period.")

    X, y = [], []
    for i in range(len(close_prices_scaled) - seq_length):
        X.append(close_prices_scaled[i:i + seq_length])
        y.append(close_prices_scaled[i + seq_length])
    X, y = np.array(X), np.array(y)

    if X.shape[0] == 0:
        raise ValueError("Not enough data to create sequences for LSTM. Adjust sequence length or dataset size.")
    
    train_size = int(len(X) * (1 - test_split))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    model = Sequential([
        LSTM(50, return_sequences=True, activation='tanh', recurrent_activation='sigmoid', input_shape=(X_train.shape[1], 1)),
        LSTM(50, return_sequences=False, activation='tanh', recurrent_activation='sigmoid'),
        Dense(25, activation='relu'),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')

    print_loss_callback = LambdaCallback(
        on_epoch_end=lambda epoch, logs: print(
            f"Epoch {epoch + 1}/{epochs} - "
            f"Loss: {logs.get('loss', 'N/A')}, "
            f"Validation Loss: {logs.get('val_loss', 'N/A')}"
        )
    )

    try:
        history = model.fit(X_train, y_train, epochs=epochs, batch_size=32, verbose=0, validation_data=(X_test, y_test), callbacks=[print_loss_callback])
        training_loss = history.history.get('loss', [-1])[-1]  # Default to -1 if 'loss' key is missing
        validation_loss = history.history.get('val_loss', [-1])[-1]  # Default to -1 if 'val_loss' key is missing
    except Exception as e:
        raise ValueError(f"LSTM training failed: {e}")
    
    y_test_pred = model.predict(X_test, verbose=0)
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

    print(f"Evaluation LSTM Metrics on Test Set :")
    print(f"- Mean Squared Error (MSE) : {mse_percent:.2f}%")
    print(f"- Mean Absolute Error (MAE) : {mae_percent:.2f}%")
    print(f"- Root Mean Squared Error (RMSE) : {rmse_percent:.2f}%")

    last_sequence = close_prices_scaled[-seq_length:]
    future_predictions = []
    for _ in range(future_steps):
        prediction = model.predict(last_sequence[np.newaxis, :, :], verbose=0)
        future_predictions.append(prediction[0, 0])
        last_sequence = np.append(last_sequence[1:], prediction, axis=0)
    
    future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
    return future_predictions.flatten(),  training_loss, validation_loss

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
    return future_predictions

# Set up Streamlit page layout
st.set_page_config(layout="wide")
st.title('Prediksi Harga Saham PT. Gudang Garam Menggunakan Algoritma LSTM dan Random Forest')

# Sidebar for user input parameters
st.sidebar.header('Chart Parameters')
ticker = st.sidebar.text_input('Ticker', 'GGRM.JK')
time_period = st.sidebar.selectbox('Time Period', ['1d', 'max'])
chart_type = st.sidebar.selectbox('Chart Type', ['Candlestick'])
indicators = st.sidebar.multiselect('Technical Indicators', ['SMA 20', 'EMA 20'])
selected_model = st.sidebar.selectbox('Model', ['LSTM', 'Random Forest'])
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
        data = add_technical_indicators(data)
        
        future_predictions = []
        future_dates = []
        model_metrics = {}

        if selected_model == 'LSTM':
            try:
                future_predictions, training_loss, validation_loss = train_lstm(data, epochs=50)
                future_dates = [data['Datetime'].iloc[-1] + timedelta(days=i) for i in range(1, len(future_predictions) + 1)]
                model_metrics = {'Training Loss': training_loss, 'Validation Loss': validation_loss}
            except ValueError as e:
                st.error(f"Error in LSTM model: {e}")
        elif selected_model == 'Random Forest':
            future_predictions = train_random_forest(data, future_steps=7)
            future_dates = [data['Datetime'].iloc[-1] + timedelta(days=i) for i in range(1, len(future_predictions) + 1)]

        
        fig = go.Figure()
        if chart_type == 'Candlestick':
            fig.add_trace(go.Candlestick(x=data['Datetime'],
                                         open=data['Open'],
                                         high=data['High'],
                                         low=data['Low'],
                                         close=data['Close']))
        else:
            fig = px.line(data, x='Datetime', y='Close')

        for indicator in indicators:
            if indicator == 'SMA 20':
                fig.add_trace(go.Scatter(x=data['Datetime'], y=data['SMA_20'], name='SMA 20'))
            elif indicator == 'EMA 20':
                fig.add_trace(go.Scatter(x=data['Datetime'], y=data['EMA_20'], name='EMA 20'))

        if show_predictions and len(future_predictions) > 0 and len(future_dates) > 0:
            fig.add_trace(go.Scatter(x=future_dates, y=future_predictions, mode='lines', name=f'{selected_model} Predictions'))

        fig.update_layout(title=f'{ticker} {time_period.upper()} Chart with Predictions',
                          xaxis_title='Time',
                          yaxis_title='Price (IDR)',
                          height=600)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader('Historical Data')
        st.dataframe(data[['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume']])

        if len(future_predictions) > 0 and len(future_dates) > 0:
            st.subheader(f'{selected_model} Predictions')
            prediction_df = pd.DataFrame({'Date': future_dates, 'Predicted Close': future_predictions})
            st.dataframe(prediction_df)
            
            st.subheader(f'{selected_model} Metrics')
            st.json(model_metrics)
