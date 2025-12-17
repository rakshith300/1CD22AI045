"""
Improved LSTM Implementation
- Bidirectional LSTM
- Cleaner training pipeline
- Stable evaluation & visualization
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import os
import urllib.request
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Bidirectional, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import plot_model

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

# -----------------------------
# Dataset Download
# -----------------------------
URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv"
FILE_NAME = "airline-passengers.csv"

if not os.path.exists(FILE_NAME):
    urllib.request.urlretrieve(URL, FILE_NAME)

data = pd.read_csv(FILE_NAME)

# -----------------------------
# Data Preparation
# -----------------------------
values = data.iloc[:, 1].values.astype("float32").reshape(-1, 1)

plt.figure(figsize=(10,4))
plt.plot(values)
plt.title("International Airline Passengers")
plt.xlabel("Time")
plt.ylabel("Passengers")
plt.show()

scaler = MinMaxScaler()
values = scaler.fit_transform(values)

train_size = int(len(values) * 0.75)
train, test = values[:train_size], values[train_size:]

# -----------------------------
# Sequence Generator
# -----------------------------
def create_sequences(data, steps):
    X, y = [], []
    for i in range(len(data) - steps):
        X.append(data[i:i+steps, 0])
        y.append(data[i+steps, 0])
    return np.array(X), np.array(y)

TIME_STEPS = 12

trainX, trainY = create_sequences(train, TIME_STEPS)
testX, testY = create_sequences(test, TIME_STEPS)

trainX = trainX.reshape((trainX.shape[0], TIME_STEPS, 1))
testX = testX.reshape((testX.shape[0], TIME_STEPS, 1))

# -----------------------------
# Model Definition
# -----------------------------
model = Sequential([
    Bidirectional(LSTM(64, return_sequences=True), input_shape=(TIME_STEPS, 1)),
    Dropout(0.2),
    Bidirectional(LSTM(32)),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1)
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae']
)

model.summary()

# -----------------------------
# Training
# -----------------------------
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

history = model.fit(
    trainX, trainY,
    epochs=80,
    batch_size=8,
    validation_split=0.1,
    callbacks=[early_stop],
    verbose=1
)

plot_model(model, to_file="lstm_model.png", show_shapes=True)

# -----------------------------
# Prediction & Evaluation
# -----------------------------
train_pred = model.predict(trainX)
test_pred = model.predict(testX)

train_pred = scaler.inverse_transform(train_pred)
test_pred = scaler.inverse_transform(test_pred)

trainY_inv = scaler.inverse_transform(trainY.reshape(-1,1))
testY_inv = scaler.inverse_transform(testY.reshape(-1,1))

print("\nPerformance Metrics")
print("Train RMSE:", math.sqrt(mean_squared_error(trainY_inv, train_pred)))
print("Test RMSE :", math.sqrt(mean_squared_error(testY_inv, test_pred)))
print("Train MAE :", mean_absolute_error(trainY_inv, train_pred))
print("Test MAE  :", mean_absolute_error(testY_inv, test_pred))

# -----------------------------
# Visualization
# -----------------------------
plt.figure(figsize=(12,5))
plt.plot(scaler.inverse_transform(values), label="Actual")
plt.plot(range(TIME_STEPS, TIME_STEPS + len(train_pred)), train_pred, label="Train Prediction")
plt.plot(range(TIME_STEPS + len(train_pred),
               TIME_STEPS + len(train_pred) + len(test_pred)), test_pred, label="Test Prediction")
plt.legend()
plt.title("LSTM Passenger Prediction")
plt.show()

# -----------------------------
# Save Model
# -----------------------------
model.save("improved_lstm_model.h5")
print("\nModel saved successfully.")
