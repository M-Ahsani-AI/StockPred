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

def train_lstm(data, future_steps=7, epochs=50, test_split=0.3):
    close_prices = data['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    close_prices_scaled = scaler.fit_transform(close_prices)
    
    seq_length = min(60, len(close_prices_scaled) - 1)  # Ensure seq_length does not exceed dataset size
    if len(close_prices_scaled) <= seq_length:
        raise ValueError("Not enough data for LSTM. Increase dataset size or choose a longer time period.")

    X, y = [], []
    for i in range(len(close_prices_scaled) - seq_length):
        X.append(close_prices_scaled[i:i + seq_length])
        y.append(close_prices_scaled[i + seq_length])
    X, y = np.array(X), np.array(y)

    if X.shape[0] == 0:
        raise ValueError("Not enough data to create sequences for LSTM. Adjust sequence length or dataset size.")
    
    # Reshape X for LSTM (samples, timesteps, features)
    X = X.reshape(X.shape[0], X.shape[1], 1)  # LSTM expects 3D input

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

    try:
        history = model.fit(X_train, y_train, epochs=epochs, batch_size=32, verbose=0, validation_data=(X_test, y_test))
    except Exception as e:
        raise ValueError(f"LSTM training failed: {e}")
    
    y_test_pred = model.predict(X_test, verbose=0)
    mse = mean_squared_error(y_test, y_test_pred)
    mae = mean_absolute_error(y_test, y_test_pred)
    rmse = np.sqrt(mse) 

    return mse, mae, rmse, model, scaler, y_test_pred

def train_mlp(data, future_steps=7, epochs=50, test_split=0.3):
    close_prices = data['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    close_prices_scaled = scaler.fit_transform(close_prices)
    
    seq_length = min(60, len(close_prices_scaled) - 1)
    if len(close_prices_scaled) <= seq_length:
        raise ValueError("Not enough data for MLP. Increase dataset size or choose a longer time period.")

    X, y = [], []
    for i in range(len(close_prices_scaled) - seq_length):
        X.append(close_prices_scaled[i:i + seq_length])
        y.append(close_prices_scaled[i + seq_length])
    X, y = np.array(X), np.array(y)

    if X.shape[0] == 0:
        raise ValueError("Not enough data to create sequences for MLP. Adjust sequence length or dataset size.")
    
    # Reshape X for MLP (samples, features)
    X = X.reshape(X.shape[0], X.shape[1])  # MLP expects 2D input

    train_size = int(len(X) * (1 - test_split))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    model = Sequential([
        Dense(128, activation='relu', input_dim=X_train.shape[1]),
        Dense(64, activation='relu'),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')

    try:
        history = model.fit(X_train, y_train, epochs=epochs, batch_size=32, verbose=0, validation_data=(X_test, y_test))
    except Exception as e:
        raise ValueError(f"MLP training failed: {e}")
    
    y_test_pred = model.predict(X_test, verbose=0)
    mse = mean_squared_error(y_test, y_test_pred)
    mae = mean_absolute_error(y_test, y_test_pred)
    rmse = np.sqrt(mse) 

    return mse, mae, rmse, model, scaler, y_test_pred

# Set up Streamlit page layout
st.set_page_config(layout="wide")
st.title('Prediksi Harga Saham PT. Gudang Garam Menggunakan Algoritma LSTM dan MLP')

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
    # Fetch data
    data = fetch_stock_data(ticker, time_period, interval_mapping[time_period])
    if data.empty:
        st.error("No data found. Try a different ticker or time period.")
    else:
        # Process and add indicators
        data = process_data(data)
        data = add_technical_indicators(data)
        
        # Train LSTM and MLP and generate predictions
        try:
            mse_lstm, mae_lstm, rmse_lstm, model_lstm, scaler_lstm, y_test_pred_lstm = train_lstm(data)
            mse_mlp, mae_mlp, rmse_mlp, model_mlp, scaler_mlp, y_test_pred_mlp = train_mlp(data)
        except ValueError as e:
            st.error(f"Error: {e}")
        
        # Print comparison in the command prompt
        print(f"LSTM Model - MSE: {mse_lstm}, MAE: {mae_lstm}, RMSE: {rmse_lstm}")
        print(f"MLP Model - MSE: {mse_mlp}, MAE: {mae_mlp}, RMSE: {rmse_mlp}")
        
        # Plot stock price chart
        fig = go.Figure()
        if chart_type == 'Candlestick':
            fig.add_trace(go.Candlestick(x=data['Datetime'],
                                         open=data['Open'],
                                         high=data['High'],
                                         low=data['Low'],
                                         close=data['Close']))
        else:
            fig = px.line(data, x='Datetime', y='Close')

        # Add technical indicators
        for indicator in indicators:
            if indicator == 'SMA 20':
                fig.add_trace(go.Scatter(x=data['Datetime'], y=data['SMA_20'], name='SMA 20'))
            elif indicator == 'EMA 20':
                fig.add_trace(go.Scatter(x=data['Datetime'], y=data['EMA_20'], name='EMA 20'))

        # Add LSTM predictions
        if show_predictions:
            future_predictions_lstm = model_lstm.predict(data['Close'][-60:].values.reshape(1, -1))
            future_predictions_lstm = scaler_lstm.inverse_transform(future_predictions_lstm)
            fig.add_trace(go.Scatter(x=[data['Datetime'].iloc[-1] + timedelta(days=i) for i in range(1, 8)], 
                                     y=future_predictions_lstm.flatten(), mode='lines', name='LSTM Predictions'))
        
        # Add MLP predictions
        if show_predictions:
            future_predictions_mlp = model_mlp.predict(data['Close'][-60:].values.reshape(1, -1))
            future_predictions_mlp = scaler_mlp.inverse_transform(future_predictions_mlp)
            fig.add_trace(go.Scatter(x=[data['Datetime'].iloc[-1] + timedelta(days=i) for i in range(1, 8)], 
                                     y=future_predictions_mlp.flatten(), mode='lines', name='MLP Predictions'))

        # Format graph
        fig.update_layout(title=f'{ticker} {time_period.upper()} Chart with Predictions',
                          xaxis_title='Time',
                          yaxis_title='Price (IDR)',
                          height=600,
                          font=dict(  
                                family="Arial, sans-serif",  # Font type
                                size=14,  # Font size
                                color="#000000"  # Text color
                                ),
                            xaxis=dict(
                            title=dict(
                            text="Time",
                            font=dict(size=20, color="#000000")  # X-axis label size
                                ),
                                tickfont=dict(color="black"),
                                gridcolor="black"
                            ),
                            yaxis=dict(
                            title=dict(
                            text="Price (IDR)",
                            font=dict(size=20, color="#000000")  # Y-axis label size
                                ),
                                tickfont=dict(color="black"),
                                gridcolor="black"
                            )
                        )
        st.plotly_chart(fig, use_container_width=True)

        # Display historical data
        st.subheader('Historical Data')
        st.dataframe(data[['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume']])

        # Prepare predictions data for display in a DataFrame
        predictions_df = pd.DataFrame({
            'Datetime': data['Datetime'].iloc[-len(y_test_pred_lstm):],
            'LSTM Predictions': scaler_lstm.inverse_transform(y_test_pred_lstm.reshape(-1, 1)).flatten(),
            'MLP Predictions': scaler_mlp.inverse_transform(y_test_pred_mlp.reshape(-1, 1)).flatten()
        })

        # Display predictions in a DataFrame
        st.subheader('Model Predictions')
        st.dataframe(predictions_df)

        # Display comparison of both models
        st.subheader('Model Performance Comparison')
        st.write(f"LSTM Model - MSE: {mse_lstm}, MAE: {mae_lstm}, RMSE: {rmse_lstm}")
        st.write(f"MLP Model - MSE: {mse_mlp}, MAE: {mae_mlp}, RMSE: {rmse_mlp}")
