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

def train_lstm(data, future_steps=8, epochs=50, test_split=0.2):
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
    
    train_size = int(len(X) * (1 - test_split))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    model = Sequential([
        LSTM(50, return_sequences=True, activation='tanh', recurrent_activation='sigmoid', input_shape=(X_train.shape[1], 1)),
        LSTM(50, return_sequences=False, activation='tanh', recurrent_activation='sigmoid'),
        Dense(25, activation='relu'),  # Added activation function
        Dense(1, activation='linear')  # Output layer with linear activation for regression
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Callback to print loss at each epoch
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
    
    # Calculate MSE, MAE, and R-squared on test set
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

    print(f"Evaluation Metrics on Test Set :")
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
    return future_predictions.flatten(), training_loss, validation_loss


# Set up Streamlit page layout
st.set_page_config(layout="wide")
st.title('Prediksi Harga Saham PT. Gudang Garam Menggunakan Algoritma Long Short-Term Memory (LSTM)')

# Sidebar for user input parameters
st.sidebar.header('Chart Parameters')
ticker = st.sidebar.text_input('Ticker', 'GGRM.JK')
time_period = st.sidebar.selectbox('Time Period', ['1mo', '3mo', '6mo', '1y', '5y', 'max'])
chart_type = st.sidebar.selectbox('Chart Type', ['Candlestick'])
indicators = st.sidebar.multiselect('Technical Indicators', ['SMA 20', 'EMA 20'])
show_predictions = st.sidebar.checkbox('Show LSTM Predictions', value=True)

# Mapping time period to intervals
interval_mapping = {
    '1mo': '1d',
    '3mo': '1d',
    '6mo': '1d',
    '1y': '1d',
    '5y': '1wk',
    'max': '1mo'
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
        
        # Train LSTM and generate predictions
        if show_predictions:
            try:
                future_predictions, training_loss, validation_loss = train_lstm(data, epochs=50)  # Epochs fixed in code
                future_dates = [data['Datetime'].iloc[-1] + timedelta(days=i) for i in range(1, len(future_predictions) + 1)]
            except ValueError as e:
                st.error(f"Error in LSTM model: {e}")
                show_predictions = False
                future_predictions = []
                future_dates = []
                
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
        if show_predictions and len(future_predictions) > 0 and len(future_dates) > 0:
            fig.add_trace(go.Scatter(x=future_dates, y=future_predictions, mode='lines', name='LSTM Predictions'))

        # Format graph
        fig.update_layout(title=f'{ticker} {time_period.upper()} Chart with Predictions',
                          xaxis_title='Time',
                          yaxis_title='Price (IDR)',
                          height=600,
                          font=dict(  
                                family="Arial, sans-serif",  # Jenis font
                                size=14,  # Ukuran font
                                color="#000000"  # Warna teks
                                ),
                            xaxis=dict(
                            title=dict(
                            text="Time",
                            font=dict(size=20, color="#000000")  # Ukuran teks untuk label sumbu X
                                ),
                                tickfont=dict(color="black"),
                                gridcolor="black"
                            ),
                            yaxis=dict(
                            title=dict(
                            text="Price (IDR)",
                            font=dict(size=20, color="#000000")  # Ukuran teks untuk label sumbu Y
                                ),
                                tickfont=dict(color="black"),
                                gridcolor="black"
                            )
                        )
        st.plotly_chart(fig, use_container_width=True)

        # Display historical data
        st.subheader('Historical Data')
        st.dataframe(data[['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume']])

        if show_predictions and len(future_predictions) > 0 and len(future_dates) > 0:
            # Format future predictions with commas
            formatted_predictions = ['{:,.2f}'.format(pred) for pred in future_predictions]
    
            # Display LSTM predictions
            st.subheader('LSTM Predictions')
            prediction_df = pd.DataFrame({'Date': future_dates, 'Predicted Close': formatted_predictions})
            st.dataframe(prediction_df)