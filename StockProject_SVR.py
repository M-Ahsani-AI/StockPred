import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import ta
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
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

# Train the SVR model and predict future stock prices
def train_svr(data, future_steps=7, test_split=0.3):
    close_prices = data['Close'].values.reshape(-1, 1)  # Ensure column vector for prices
    scaler = MinMaxScaler()
    close_prices_scaled = scaler.fit_transform(close_prices)
    
    seq_length = min(60, len(close_prices_scaled) - 1)  # Ensure seq_length does not exceed dataset size
    if len(close_prices_scaled) <= seq_length:
        raise ValueError("Not enough data for SVR. Increase dataset size or choose a longer time period.")

    X, y = [], []
    for i in range(len(close_prices_scaled) - seq_length):
        X.append(close_prices_scaled[i:i + seq_length])  # Append sequences of past prices
        y.append(close_prices_scaled[i + seq_length])  # Append the next price as the target
    X, y = np.array(X), np.array(y)

    # Reshape X to be 2D (samples, features)
    X = X.reshape(X.shape[0], -1)  # Flatten the sequence into a 2D array for SVR
    y = y.ravel()  # Flatten y to be 1D for compatibility with SVR

    if X.shape[0] == 0:
        raise ValueError("Not enough data to create sequences for SVR. Adjust sequence length or dataset size.")
    
    # Split data into train and test sets
    train_size = int(len(X) * (1 - test_split))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # SVR model
    svr = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.01)
    svr.fit(X_train, y_train)

    # Calculate MSE, MAE, and RMSE on test set
    y_test_pred = svr.predict(X_test)
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

    # Future prediction
    last_sequence = close_prices_scaled[-seq_length:]
    future_predictions = []
    for _ in range(future_steps):
        prediction = svr.predict(last_sequence.reshape(1, -1))  # Flatten sequence for prediction
        future_predictions.append(prediction[0])
        last_sequence = np.append(last_sequence[1:], prediction, axis=0)
    
    future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
    return future_predictions.flatten(), mse_percent, mae_percent, rmse_percent


# Set up Streamlit page layout
st.set_page_config(layout="wide")
st.title('Prediksi Harga Saham PT. Gudang Garam Menggunakan Algoritma Support Vector Regression (SVR)')

# Sidebar for user input parameters
st.sidebar.header('Chart Parameters')
ticker = st.sidebar.text_input('Ticker', 'GGRM.JK')
time_period = st.sidebar.selectbox('Time Period', ['1d', 'max'])
chart_type = st.sidebar.selectbox('Chart Type', ['Candlestick'])
indicators = st.sidebar.multiselect('Technical Indicators', ['SMA 20', 'EMA 20'])
show_predictions = st.sidebar.checkbox('Show SVR Predictions', value=True)

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
        
        # Train SVR and generate predictions
        if show_predictions:
            try:
                future_predictions, mse_percent, mae_percent, rmse_percent = train_svr(data)  # Using SVR model
                future_dates = [data['Datetime'].iloc[-1] + timedelta(days=i) for i in range(1, len(future_predictions) + 1)]
            except ValueError as e:
                st.error(f"Error in SVR model: {e}")
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

        # Add SVR predictions
        if show_predictions and len(future_predictions) > 0 and len(future_dates) > 0:
            fig.add_trace(go.Scatter(x=future_dates, y=future_predictions, mode='lines', name='SVR Predictions'))

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
    
            # Display SVR predictions
            st.subheader('SVR Predictions')
            prediction_df = pd.DataFrame({'Date': future_dates, 'Predicted Close': formatted_predictions})
            st.dataframe(prediction_df)
