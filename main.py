import pandas as pd
import numpy as np
import math
import plotly.graph_objects as go
from datetime import datetime
import time
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from flask import Flask, render_template
import yfinance as yf  

app = Flask(__name__)

def get_stock_data():
    ticker = "GGRM.JK"  
    start_date = "2021-11-01"
    end_date = "2024-11-30"

    data = yf.download(ticker, start=start_date, end=end_date, interval="1d")

    if data.empty:
        raise Exception(f"Data tidak ditemukan untuk ticker {ticker}. Pastikan simbolnya benar.")

    df = data[['Close']]
    return df

def preprocess_data(df):
    data = df[['Close']]
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    training_data_len = int(np.ceil(len(scaled_data) * 0.7))
    train_data = scaled_data[:training_data_len, :]

    x_train, y_train = [], []
    for i in range(60, len(train_data)):
        x_train.append(train_data[i-60:i, 0])
        y_train.append(train_data[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    return x_train, y_train, scaler, scaled_data, training_data_len

def split_train_test(scaled_data, training_data_len, data):
    test_data = scaled_data[training_data_len-60:, :]
    x_test, y_test = [], data['Close'][training_data_len:].values

    for i in range(60, len(test_data)):
        x_test.append(test_data[i-60:i, 0])

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    return x_test, y_test

def build_lstm_model(x_train):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dense(units=25))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_model(model, x_train, y_train):
    model.fit(x_train, y_train, batch_size=32, epochs=20)

def evaluate_model(model, x_test, y_test, scaler):
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)
    rmse = math.sqrt(mean_squared_error(y_test, predictions))
    return predictions, rmse

def predict_next_day(model, scaled_data, scaler):
    last_60_days = scaled_data[-60:]
    x_pred = [last_60_days]
    x_pred = np.array(x_pred)
    x_pred = np.reshape(x_pred, (x_pred.shape[0], x_pred.shape[1], 1))
    predicted_price = model.predict(x_pred)
    predicted_price = scaler.inverse_transform(predicted_price)
    return predicted_price

def plot_results(df, predictions, training_data_len):
    train = df[:training_data_len]
    valid = df[training_data_len:]
    valid = valid.iloc[:len(predictions)]
    valid['Predictions'] = predictions

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=train.index, y=train['Close'],
        mode='lines',
        name='Data Latih',
        line=dict(color='blue', width=2)
    ))
    fig.add_trace(go.Scatter(
        x=valid.index, y=valid['Close'],
        mode='lines',
        name='Data Validasi',
        line=dict(color='red', width=2)
    ))
    fig.add_trace(go.Scatter(
        x=valid.index, y=valid['Predictions'],
        mode='lines',
        name='Prediksi',
        line=dict(color='green', width=2)
    ))
    fig.update_layout(
        title='Prediksi Harga Saham PT Gudang Garam',
        xaxis_title=' Tanggal',
        yaxis_title='Harga Saham',
        legend=dict(x=0, y=1),
        xaxis=dict(showline=True, zeroline=False),
        yaxis=dict(showline=True, zeroline=False)
    )
    return fig.to_html(full_html=False)  # Mengembalikan HTML tanpa tag <html>

@app.route('/')
def index():
    try:
        df = get_stock_data()
        x_train, y_train, scaler, scaled_data, training_data_len = preprocess_data(df)
        model = build_lstm_model(x_train)
        train_model(model, x_train, y_train)
        x_test, y_test = split_train_test(scaled_data, training_data_len, df)
        predictions, rmse = evaluate_model(model, x_test, y_test, scaler)
        next_day_prediction = predict_next_day(model, scaled_data, scaler)
        plot_html = plot_results(df, predictions, training_data_len)
        return render_template('index.html', plot=plot_html, next_day_prediction=next_day_prediction[0][0], rmse=rmse)
    except Exception as e:
        return render_template('error.html', error_message=str(e))

if __name__ == '__main__':
    app.run(debug=True)