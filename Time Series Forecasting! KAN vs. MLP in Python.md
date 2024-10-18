## Time Series Forecasting! KAN vs. MLP in Python
Slide 1: Introduction to Time Series Forecasting

Time series forecasting is a crucial technique in data science for predicting future values based on historical data. This presentation will explore two popular neural network architectures for time series forecasting: Keras Autoregressive Network (KAN) and Multilayer Perceptron (MLP). We'll compare their strengths, weaknesses, and implementation using Python.

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate a sample time series
time = np.arange(0, 100, 0.1)
series = np.sin(time) + np.random.normal(0, 0.1, len(time))

plt.figure(figsize=(12, 6))
plt.plot(time, series)
plt.title('Sample Time Series Data')
plt.xlabel('Time')
plt.ylabel('Value')
plt.show()
```

Slide 2: Understanding Time Series Data

Time series data consists of observations collected sequentially over time. It's characterized by temporal dependency, where future values are influenced by past observations. Common components include trend, seasonality, and noise.

```python
from statsmodels.tsa.seasonal import seasonal_decompose

# Decompose the time series
result = seasonal_decompose(series, model='additive', period=30)

# Plot the decomposition
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 16))
result.observed.plot(ax=ax1)
ax1.set_title('Observed')
result.trend.plot(ax=ax2)
ax2.set_title('Trend')
result.seasonal.plot(ax=ax3)
ax3.set_title('Seasonal')
result.resid.plot(ax=ax4)
ax4.set_title('Residual')
plt.tight_layout()
plt.show()
```

Slide 3: Keras Autoregressive Network (KAN)

KAN is a type of recurrent neural network designed specifically for time series forecasting. It learns to predict future values based on a window of past observations, making it well-suited for capturing temporal dependencies.

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

def create_kan_model(window_size):
    model = Sequential([
        LSTM(64, activation='relu', input_shape=(window_size, 1)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# Example usage
window_size = 10
kan_model = create_kan_model(window_size)
kan_model.summary()
```

Slide 4: Multilayer Perceptron (MLP)

MLP is a feedforward neural network consisting of multiple layers of neurons. While not specifically designed for time series, it can be adapted for forecasting by using a sliding window approach to create input-output pairs.

```python
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

def create_mlp_model(window_size):
    inputs = Input(shape=(window_size,))
    x = Dense(64, activation='relu')(inputs)
    x = Dense(32, activation='relu')(x)
    outputs = Dense(1)(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mse')
    return model

# Example usage
window_size = 10
mlp_model = create_mlp_model(window_size)
mlp_model.summary()
```

Slide 5: Data Preparation for KAN and MLP

Preparing data for both KAN and MLP involves creating input-output pairs using a sliding window approach. The main difference is that KAN requires 3D input (samples, time steps, features), while MLP uses 2D input (samples, features).

```python
def create_dataset(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size])
    return np.array(X), np.array(y)

# Prepare data for KAN
X_kan, y_kan = create_dataset(series, window_size)
X_kan = X_kan.reshape((X_kan.shape[0], X_kan.shape[1], 1))

# Prepare data for MLP
X_mlp, y_mlp = create_dataset(series, window_size)

print("KAN input shape:", X_kan.shape)
print("MLP input shape:", X_mlp.shape)
```

Slide 6: Training KAN Model

We'll train the KAN model using the prepared data. KAN's ability to capture temporal dependencies makes it particularly effective for time series forecasting.

```python
# Split data into train and test sets
train_size = int(len(X_kan) * 0.8)
X_train_kan, X_test_kan = X_kan[:train_size], X_kan[train_size:]
y_train_kan, y_test_kan = y_kan[:train_size], y_kan[train_size:]

# Train the KAN model
history_kan = kan_model.fit(
    X_train_kan, y_train_kan,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    verbose=0
)

# Plot training history
plt.figure(figsize=(10, 6))
plt.plot(history_kan.history['loss'], label='Training Loss')
plt.plot(history_kan.history['val_loss'], label='Validation Loss')
plt.title('KAN Model Training History')
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.show()
```

Slide 7: Training MLP Model

Now, we'll train the MLP model using the same dataset. While MLP doesn't inherently capture temporal dependencies, it can still learn patterns in the time series data.

```python
# Split data into train and test sets
train_size = int(len(X_mlp) * 0.8)
X_train_mlp, X_test_mlp = X_mlp[:train_size], X_mlp[train_size:]
y_train_mlp, y_test_mlp = y_mlp[:train_size], y_mlp[train_size:]

# Train the MLP model
history_mlp = mlp_model.fit(
    X_train_mlp, y_train_mlp,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    verbose=0
)

# Plot training history
plt.figure(figsize=(10, 6))
plt.plot(history_mlp.history['loss'], label='Training Loss')
plt.plot(history_mlp.history['val_loss'], label='Validation Loss')
plt.title('MLP Model Training History')
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.show()
```

Slide 8: Comparing KAN and MLP Predictions

Let's compare the predictions made by both models on the test set to evaluate their performance.

```python
# Make predictions
kan_predictions = kan_model.predict(X_test_kan)
mlp_predictions = mlp_model.predict(X_test_mlp)

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(y_test_kan, label='Actual')
plt.plot(kan_predictions, label='KAN Predictions')
plt.plot(mlp_predictions, label='MLP Predictions')
plt.title('KAN vs MLP: Time Series Forecasting')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.show()

# Calculate Mean Squared Error
from sklearn.metrics import mean_squared_error
kan_mse = mean_squared_error(y_test_kan, kan_predictions)
mlp_mse = mean_squared_error(y_test_mlp, mlp_predictions)
print(f"KAN MSE: {kan_mse:.4f}")
print(f"MLP MSE: {mlp_mse:.4f}")
```

Slide 9: Advantages of KAN

KAN has several advantages for time series forecasting:

1. Temporal dependency: KAN can capture long-term dependencies in the data.
2. Sequence handling: It naturally handles variable-length input sequences.
3. Feature extraction: KAN can automatically learn relevant features from raw time series data.

```python
# Visualize KAN's ability to capture temporal dependencies
from tensorflow.keras.layers import SimpleRNN

# Create a simple RNN model to demonstrate temporal dependency
rnn_model = Sequential([
    SimpleRNN(10, input_shape=(window_size, 1), return_sequences=True),
    Dense(1)
])

# Generate a sequence with a clear pattern
x = np.arange(100)
y = np.sin(x * 0.1)

# Prepare data for RNN
X_rnn, y_rnn = create_dataset(y, window_size)
X_rnn = X_rnn.reshape((X_rnn.shape[0], X_rnn.shape[1], 1))

# Make predictions
rnn_predictions = rnn_model.predict(X_rnn)

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(y[window_size:], label='Actual')
plt.plot(rnn_predictions.flatten(), label='RNN Predictions')
plt.title('RNN Capturing Temporal Dependencies')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.show()
```

Slide 10: Advantages of MLP

MLP also has its strengths in time series forecasting:

1. Simplicity: MLP is easier to implement and train compared to recurrent models.
2. Computational efficiency: It typically requires less computational resources than KAN.
3. Flexibility: MLP can easily incorporate additional features beyond the time series itself.

```python
# Demonstrate MLP's flexibility with additional features
def create_mlp_model_with_features(window_size, num_features):
    inputs = Input(shape=(window_size, num_features))
    x = Dense(64, activation='relu')(inputs)
    x = Dense(32, activation='relu')(x)
    outputs = Dense(1)(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mse')
    return model

# Create a dataset with additional features
time = np.arange(1000)
series = np.sin(time * 0.1) + np.random.normal(0, 0.1, 1000)
feature1 = np.cos(time * 0.05)
feature2 = np.log(time + 1)

# Prepare data with additional features
X, y = create_dataset(series, window_size)
X_with_features = np.dstack((X, 
                             np.roll(feature1, -window_size)[:-window_size], 
                             np.roll(feature2, -window_size)[:-window_size]))

# Create and train the model
mlp_model_with_features = create_mlp_model_with_features(window_size, 3)
mlp_model_with_features.fit(X_with_features, y, epochs=50, batch_size=32, verbose=0)

# Make predictions
predictions = mlp_model_with_features.predict(X_with_features)

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(y, label='Actual')
plt.plot(predictions, label='MLP Predictions')
plt.title('MLP with Additional Features')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.show()
```

Slide 11: Real-Life Example: Weather Forecasting

Weather forecasting is a common application of time series analysis. Both KAN and MLP can be used to predict temperature based on historical data.

```python
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Generate synthetic weather data
dates = pd.date_range(start='2020-01-01', end='2022-12-31', freq='D')
temperatures = np.sin(np.arange(len(dates)) * 2 * np.pi / 365) * 15 + 20 + np.random.normal(0, 3, len(dates))
df = pd.DataFrame({'date': dates, 'temperature': temperatures})

# Prepare data for forecasting
scaler = MinMaxScaler()
scaled_temp = scaler.fit_transform(df[['temperature']])

# Create dataset with a 7-day window
X, y = create_dataset(scaled_temp, 7)

# Split data
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Create and train KAN model
kan_model = create_kan_model(7)
kan_model.fit(X_train.reshape(-1, 7, 1), y_train, epochs=50, batch_size=32, verbose=0)

# Create and train MLP model
mlp_model = create_mlp_model(7)
mlp_model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)

# Make predictions
kan_pred = scaler.inverse_transform(kan_model.predict(X_test.reshape(-1, 7, 1)))
mlp_pred = scaler.inverse_transform(mlp_model.predict(X_test))
actual = scaler.inverse_transform(y_test.reshape(-1, 1))

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(actual, label='Actual Temperature')
plt.plot(kan_pred, label='KAN Prediction')
plt.plot(mlp_pred, label='MLP Prediction')
plt.title('Weather Forecasting: KAN vs MLP')
plt.xlabel('Days')
plt.ylabel('Temperature (°C)')
plt.legend()
plt.show()
```

Slide 12: Real-Life Example: Energy Consumption Forecasting

Energy consumption forecasting is crucial for efficient grid management. Let's compare KAN and MLP for predicting household energy consumption.

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Generate synthetic energy consumption data
hours = pd.date_range(start='2022-01-01', end='2022-12-31 23:00:00', freq='H')
consumption = np.sin(np.arange(len(hours)) * 2 * np.pi / 24) * 2 + 5 + np.random.normal(0, 0.5, len(hours))
df = pd.DataFrame({'datetime': hours, 'consumption': consumption})

# Prepare data for forecasting
scaler = MinMaxScaler()
scaled_consumption = scaler.fit_transform(df[['consumption']])

# Create dataset with a 24-hour window
X, y = create_dataset(scaled_consumption, 24)

# Split data, train models, and make predictions
# (Code for splitting data, training models, and making predictions)

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(actual[:168], label='Actual Consumption')
plt.plot(kan_pred[:168], label='KAN Prediction')
plt.plot(mlp_pred[:168], label='MLP Prediction')
plt.title('Energy Consumption Forecasting: KAN vs MLP')
plt.xlabel('Hours')
plt.ylabel('Energy Consumption (kWh)')
plt.legend()
plt.show()
```

Slide 13: Choosing Between KAN and MLP

When deciding between KAN and MLP for time series forecasting, consider the following factors:

1. Data characteristics: KAN is better suited for data with strong temporal dependencies, while MLP can handle more general patterns.
2. Computational resources: MLP is generally faster to train and requires less memory than KAN.
3. Forecast horizon: KAN often performs better for longer-term forecasts, while MLP can be sufficient for short-term predictions.
4. Available features: If you have additional relevant features, MLP can easily incorporate them.

```python
def choose_model(data_length, num_features, temporal_dependency, forecast_horizon):
    score_kan = 0
    score_mlp = 0
    
    if temporal_dependency == 'high':
        score_kan += 2
    elif temporal_dependency == 'low':
        score_mlp += 1
    
    if data_length > 10000:
        score_kan += 1
    else:
        score_mlp += 1
    
    if num_features > 1:
        score_mlp += 1
    
    if forecast_horizon == 'long':
        score_kan += 1
    elif forecast_horizon == 'short':
        score_mlp += 1
    
    return 'KAN' if score_kan > score_mlp else 'MLP'

# Example usage
result = choose_model(data_length=5000, num_features=3, temporal_dependency='high', forecast_horizon='long')
print(f"Recommended model: {result}")
```

Slide 14: Hybrid Approaches: Combining KAN and MLP

In some cases, a hybrid approach combining both KAN and MLP can yield better results. This allows us to leverage the strengths of both architectures.

```python
from tensorflow.keras.layers import Concatenate

def create_hybrid_model(window_size):
    # KAN branch
    kan_input = Input(shape=(window_size, 1))
    kan_output = LSTM(32, activation='relu')(kan_input)
    
    # MLP branch
    mlp_input = Input(shape=(window_size,))
    mlp_hidden = Dense(64, activation='relu')(mlp_input)
    mlp_output = Dense(32, activation='relu')(mlp_hidden)
    
    # Combine KAN and MLP outputs
    combined = Concatenate()([kan_output, mlp_output])
    
    # Final prediction layer
    output = Dense(1)(combined)
    
    model = Model(inputs=[kan_input, mlp_input], outputs=output)
    model.compile(optimizer='adam', loss='mse')
    return model

# Create and train the hybrid model
hybrid_model = create_hybrid_model(window_size)
hybrid_model.fit([X_train_kan, X_train_mlp], y_train, epochs=50, batch_size=32, verbose=0)

# Make predictions
hybrid_pred = hybrid_model.predict([X_test_kan, X_test_mlp])

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(y_test, label='Actual')
plt.plot(hybrid_pred, label='Hybrid Model Predictions')
plt.title('Hybrid KAN-MLP Model Predictions')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.show()
```

Slide 15: Additional Resources

For further exploration of time series forecasting using KAN and MLP, consider the following resources:

1. "Deep Learning for Time Series Forecasting" by J. Brownlee (2018)
2. "Temporal Convolutional Networks for Sequence Modeling" by S. Bai et al. (2018) - arXiv:1803.01271
3. "Neural Forecasting: Introduction and Literature Overview" by S. Petnehazi (2019) - arXiv:1909.00957
4. "A Comparative Study of Time Series Forecasting Methods for Short Term Load Forecasting in Smart Buildings" by D. Alberg and M. Last (2018) - arXiv:1812.10851

These resources provide in-depth discussions on various aspects of time series forecasting using neural networks, including KAN and MLP approaches.

