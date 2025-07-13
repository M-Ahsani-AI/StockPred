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
from tensorflow.keras.layers import Dense
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

def train_ann(data, future_steps=7, epochs=50, test_split=0.3):
    # Separate close prices (target variable) and features
    close_prices = data['Close'].values.reshape(-1, 1)
    feature_data = data[['Close', 'SMA_20', 'EMA_20']].values  # Features: Close, SMA, EMA
    
    # Scalers for features and target (Close)
    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()

    # Scale the features and target separately
    features_scaled = feature_scaler.fit_transform(feature_data)
    close_prices_scaled = target_scaler.fit_transform(close_prices)
    
    # Prepare the data for training
    X = features_scaled[:-future_steps]  # Input features (excluding last `future_steps`)
    y = close_prices_scaled[future_steps:]  # Target values (future close prices)

    # Split into training and testing sets
    train_size = int(len(X) * (1 - test_split))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Build the ANN model
    model = Sequential([
        Dense(64, activation='relu', input_dim=X_train.shape[1]),
        Dense(32, activation='relu'),
        Dense(1, activation='linear')  # Output layer with linear activation for regression
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')

    try:
        history = model.fit(X_train, y_train, epochs=epochs, batch_size=32, verbose=0, validation_data=(X_test, y_test))
        training_loss = history.history['loss'][-1]  # Training loss at last epoch
        validation_loss = history.history['val_loss'][-1]  # Validation loss at last epoch
    except Exception as e:
        raise ValueError(f"ANN training failed: {e}")
    
    # Calculate MSE, MAE, and RMSE on the test set
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

    # Predict future prices using the trained model
    last_features = features_scaled[-future_steps:]  # Last `future_steps` rows of features
    future_predictions_scaled = model.predict(last_features)  # Get predictions in scaled form
    future_predictions = target_scaler.inverse_transform(future_predictions_scaled)  # Reverse scale

    return future_predictions.flatten(), training_loss, validation_loss

# Set up Streamlit page layout
st.set_page_config(layout="wide")
st.title('Prediksi Harga Saham PT. Gudang Garam Menggunakan Algoritma Artificial Neural Network (ANN)')

# Sidebar for user input parameters
st.sidebar.header('Chart Parameters')
ticker = st.sidebar.text_input('Ticker', 'GGRM.JK')
time_period = st.sidebar.selectbox('Time Period', ['1d', 'max'])
chart_type = st.sidebar.selectbox('Chart Type', ['Candlestick'])
indicators = st.sidebar.multiselect('Technical Indicators', ['SMA 20', 'EMA 20'])
show_predictions = st.sidebar.checkbox('Show ANN Predictions', value=True)

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
        
        # Train ANN and generate predictions
        if show_predictions:
            try:
                future_predictions, training_loss, validation_loss = train_ann(data, epochs=50)  # Epochs fixed in code
                future_dates = [data['Datetime'].iloc[-1] + timedelta(days=i) for i in range(1, len(future_predictions) + 1)]
            except ValueError as e:
                st.error(f"Error in ANN model: {e}")
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

        # Add ANN predictions
        if show_predictions and len(future_predictions) > 0 and len(future_dates) > 0:
            fig.add_trace(go.Scatter(x=future_dates, y=future_predictions, mode='lines', name='ANN Predictions'))

        # Format graph
        fig.update_layout(title=f'{ticker} {time_period.upper()} Chart with Predictions',
                          xaxis_title='Time',
                          yaxis_title='Price (IDR)',
                          height=600)
        st.plotly_chart(fig, use_container_width=True)

        # Display historical data
        st.subheader('Historical Data')
        st.dataframe(data[['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume']])

        if show_predictions and len(future_predictions) > 0 and len(future_dates) > 0:
            # Format future predictions with commas
            formatted_predictions = ['{:,.2f}'.format(pred) for pred in future_predictions]
    
            # Display ANN predictions
            st.subheader('ANN Predictions')
            prediction_df = pd.DataFrame({'Date': future_dates, 'Predicted Close': formatted_predictions})
            st.dataframe(prediction_df)
