#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Import the required libraries
import warnings as w
w.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout


# In[2]:


#Fetching the stock data
def fetch_stock_data(ticker="AAPL", start="2015-01-01", end="2025-01-01"):
    df = yf.download(ticker, start=start, end=end)
    df = df[['Open','High','Low','Close','Volume']].dropna()
    df = df.asfreq('B', method='ffill')
    df.to_csv(f"{ticker}_raw.csv")
    print(f"Raw data saved: {ticker}_raw.csv")
    return df


# In[3]:


#Preprocessing the data
def preprocess_data(df):
    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)
    df_scaled.to_csv("processed_stock.csv")
    print("Processed data saved: processed_stock.csv")
    return df_scaled, scaler


# In[4]:


#Arima Forecast visualization
def arima_forecast(df, column='Close', order=(5,1,0), steps=30):
    model = ARIMA(df[column], order=order)
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=steps)
    
    plt.figure(figsize=(10,5))
    plt.plot(df[column], label="Actual")
    plt.plot(forecast, label="ARIMA Forecast", color='red')
    plt.title("ARIMA Forecast")
    plt.legend()
    plt.show()
    return forecast


# In[5]:


#Prophet Forecast visualization
def prophet_forecast(df, column='Close', periods=30):
    df_prophet = df[[column]].reset_index()
    df_prophet.columns = ['ds','y']
    
    model = Prophet(daily_seasonality=True)
    model.fit(df_prophet)
    
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    
    model.plot(forecast)
    plt.title("Prophet Forecast")
    plt.show()
    
    return forecast[['ds','yhat','yhat_lower','yhat_upper']]


# In[6]:


#Creating the data sequences
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data)-seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)


# In[7]:


#LSTM Forecast visualization
def lstm_forecast(df, column='Close', seq_length=50, epochs=20, batch_size=32, forecast_steps=30):
    data = df[column].values.reshape(-1,1)
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)
    
    X, y = create_sequences(data_scaled, seq_length)
    
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(seq_length,1)))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    
    model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=1)
    
    forecast_input = data_scaled[-seq_length:]
    forecast = []
    for _ in range(forecast_steps):
        pred = model.predict(forecast_input.reshape(1, seq_length, 1))
        forecast.append(pred[0,0])
        forecast_input = np.append(forecast_input[1:], pred[0,0])
    
    forecast = scaler.inverse_transform(np.array(forecast).reshape(-1,1))
    
    plt.figure(figsize=(10,5))
    plt.plot(df.index, df[column], label="Actual")
    future_index = pd.date_range(df.index[-1]+pd.Timedelta(days=1), periods=forecast_steps)
    plt.plot(future_index, forecast, label="LSTM Forecast", color='green')
    plt.title("LSTM Forecast")
    plt.legend()
    plt.show()
    
    return forecast


# In[8]:


#Main() Function
if __name__ == "__main__":
    ticker = "AAPL"
    raw_df = fetch_stock_data(ticker)
    processed_df, scaler = preprocess_data(raw_df)
    
    print("Running ARIMA Forecast...")
    arima_forecast(processed_df)
    
    print("Running Prophet Forecast...")
    prophet_forecast(processed_df)
    
    print("Running LSTM Forecast...")
    lstm_forecast(processed_df)

