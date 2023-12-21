import streamlit as st
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import SimpleRNN, LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import time
from sklearn.metrics import mean_squared_error

# Record start time
start_time = time.time()

st.title("Deep Learning Kelompok 6")
uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    # Preprocessing
    length_data = len(data)
    split_ratio = 0.7
    length_train = round(length_data * split_ratio)
    train_data = data[:length_train].iloc[:, :2]
    validation_data = data[length_train:].iloc[:, :2]

    dataset_train = train_data.Open.values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset_train_scaled = scaler.fit_transform(dataset_train)

    X_train, y_train = [], []
    time_step = 50
    for i in range(time_step, length_train):
        X_train.append(dataset_train_scaled[i - time_step:i, 0])
        y_train.append(dataset_train_scaled[i, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    y_train = np.reshape(y_train, (y_train.shape[0], 1))

    # Creating RNN model
    regressor = Sequential()
    regressor.add(SimpleRNN(units=50, activation="tanh", return_sequences=True, input_shape=(X_train.shape[1], 1)))
    regressor.add(Dropout(0.2))
    regressor.add(SimpleRNN(units=50, activation="tanh", return_sequences=True))
    regressor.add(Dropout(0.2))
    regressor.add(SimpleRNN(units=50, activation="tanh", return_sequences=True))
    regressor.add(Dropout(0.2))
    regressor.add(SimpleRNN(units=50))
    regressor.add(Dropout(0.2))
    regressor.add(Dense(units=1))
    regressor.compile(optimizer="adam", loss="mean_squared_error", metrics=["accuracy"])
    history = regressor.fit(X_train, y_train, epochs=50, batch_size=32)

    # Creating LSTM Model
    model_lstm = Sequential()
    model_lstm.add(LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model_lstm.add(LSTM(64, return_sequences=False))
    model_lstm.add(Dense(32))
    model_lstm.add(Dense(1))
    model_lstm.compile(loss="mean_squared_error", optimizer="adam", metrics=["accuracy"])
    history2 = model_lstm.fit(X_train, y_train, epochs=10, batch_size=10)

    # Prepare validation data
    dataset_val = validation_data.Open.values.reshape(-1, 1)
    dataset_val_scaled = scaler.transform(dataset_val)
    X_val, y_val = [], []
    for i in range(time_step, len(dataset_val_scaled)):
        X_val.append(dataset_val_scaled[i - time_step:i, 0])
        y_val.append(dataset_val_scaled[i, 0])
    X_val, y_val = np.array(X_val), np.array(y_val)
    X_val = np.reshape(X_val, (X_val.shape[0], X_val.shape[1], 1))
    y_val = np.reshape(y_val, (y_val.shape[0], 1))

    # Evaluate models on validation set
    RNN_val_pred = scaler.inverse_transform(regressor.predict(X_val))
    LSTM_val_pred = scaler.inverse_transform(model_lstm.predict(X_val))

    RNN_val_mse = mean_squared_error(scaler.inverse_transform(y_val), RNN_val_pred)
    LSTM_val_mse = mean_squared_error(scaler.inverse_transform(y_val), LSTM_val_pred)
    # Calculate execution time
    execution_time = time.time() - start_time
   # Display future price predictions, accuracy, and execution time
    st.write("Simple RNN, Open price prediction for future:", RNN_val_pred[-1][0])
    st.write("LSTM prediction, Open price prediction for future:", LSTM_val_pred[-1][0])

    st.write("Simple RNN, Validation set Mean Squared Error (MSE):", RNN_val_mse)
    st.write("LSTM, Validation set Mean Squared Error (MSE):", LSTM_val_mse)

    st.write("Execution Time:", execution_time, "seconds")
