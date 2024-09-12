## Time Series Anomaly Detection with Spatial-Temporal Normality Learning
Slide 1: Introduction to Time Series Anomaly Detection

Time series anomaly detection is a crucial task in various domains, including IoT, cybersecurity, and industrial monitoring. Spatial-Temporal Normality Learning (STEN) is an advanced technique that combines spatial and temporal information to identify anomalies in time series data. This presentation will explore STEN and its implementation using Python.

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate a sample time series with an anomaly
np.random.seed(42)
time = np.arange(100)
normal_data = np.sin(time * 0.1) + np.random.normal(0, 0.1, 100)
anomaly = np.zeros(100)
anomaly[60:70] = 2  # Introduce an anomaly

time_series = normal_data + anomaly

plt.figure(figsize=(12, 6))
plt.plot(time, time_series)
plt.title("Time Series with Anomaly")
plt.xlabel("Time")
plt.ylabel("Value")
plt.show()
```

Slide 2: Understanding Spatial-Temporal Normality

STEN considers both spatial and temporal aspects of data to detect anomalies. Spatial normality refers to the relationship between different variables or features at a given time point, while temporal normality focuses on the patterns and trends over time for each variable.

```python
import pandas as pd

# Create a sample multivariate time series
dates = pd.date_range(start='2023-01-01', periods=100, freq='H')
df = pd.DataFrame({
    'timestamp': dates,
    'temperature': np.sin(np.arange(100) * 0.1) + np.random.normal(0, 0.1, 100),
    'humidity': np.cos(np.arange(100) * 0.1) + np.random.normal(0, 0.1, 100),
    'pressure': np.tan(np.arange(100) * 0.05) + np.random.normal(0, 0.1, 100)
})

print(df.head())

# Visualize spatial relationships
plt.figure(figsize=(10, 6))
plt.scatter(df['temperature'], df['humidity'], c=df['pressure'], cmap='viridis')
plt.colorbar(label='Pressure')
plt.xlabel('Temperature')
plt.ylabel('Humidity')
plt.title('Spatial Relationship between Variables')
plt.show()
```

Slide 3: STEN Architecture

STEN typically employs a deep learning architecture, often based on autoencoders or recurrent neural networks (RNNs). The model learns to reconstruct normal patterns in both spatial and temporal dimensions, allowing it to identify deviations as anomalies.

```python
import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dense, RepeatVector, TimeDistributed

# Define STEN model architecture
def create_sten_model(input_shape):
    inputs = Input(shape=input_shape)
    encoded = LSTM(64, activation='relu')(inputs)
    repeated = RepeatVector(input_shape[0])(encoded)
    decoded = LSTM(64, activation='relu', return_sequences=True)(repeated)
    outputs = TimeDistributed(Dense(input_shape[1]))(decoded)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mse')
    return model

# Example usage
input_shape = (10, 3)  # 10 time steps, 3 features
model = create_sten_model(input_shape)
model.summary()
```

Slide 4: Data Preprocessing for STEN

Before applying STEN, it's crucial to preprocess the time series data. This involves normalization, handling missing values, and creating sliding windows for temporal context.

```python
from sklearn.preprocessing import MinMaxScaler

# Normalize the data
scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(df[['temperature', 'humidity', 'pressure']])

# Create sliding windows
def create_sequences(data, seq_length):
    sequences = []
    for i in range(len(data) - seq_length + 1):
        seq = data[i:i+seq_length]
        sequences.append(seq)
    return np.array(sequences)

seq_length = 10
X = create_sequences(normalized_data, seq_length)

print("Shape of input sequences:", X.shape)
```

Slide 5: Training the STEN Model

Training the STEN model involves using the preprocessed data to learn normal patterns. The model is trained to reconstruct the input sequences, minimizing the reconstruction error for normal data.

```python
# Split data into train and test sets
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]

# Train the model
model = create_sten_model((seq_length, 3))
history = model.fit(
    X_train, X_train,
    epochs=50,
    batch_size=32,
    validation_split=0.1,
    shuffle=False
)

# Plot training history
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Training History')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
```

Slide 6: Anomaly Detection with STEN

Once trained, the STEN model can detect anomalies by comparing the reconstruction error of new data points to a threshold. Points with high reconstruction errors are flagged as potential anomalies.

```python
# Predict on test data
X_pred = model.predict(X_test)

# Calculate reconstruction error
mse = np.mean(np.power(X_test - X_pred, 2), axis=(1, 2))

# Set threshold for anomaly detection
threshold = np.mean(mse) + 2 * np.std(mse)

# Identify anomalies
anomalies = mse > threshold

# Visualize results
plt.figure(figsize=(12, 6))
plt.plot(mse)
plt.axhline(y=threshold, color='r', linestyle='--', label='Threshold')
plt.title('Reconstruction Error')
plt.xlabel('Sample')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.show()

print(f"Number of anomalies detected: {np.sum(anomalies)}")
```

Slide 7: Real-life Example: Environmental Monitoring

STEN can be applied to environmental monitoring systems to detect unusual patterns in sensor data. For instance, in a smart city project, sensors collect data on air quality, temperature, and noise levels.

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Simulated environmental data
np.random.seed(42)
dates = pd.date_range(start='2023-01-01', periods=1000, freq='H')
df = pd.DataFrame({
    'timestamp': dates,
    'temperature': np.sin(np.arange(1000) * 0.02) + np.random.normal(20, 5, 1000),
    'humidity': np.cos(np.arange(1000) * 0.02) + np.random.normal(60, 10, 1000),
    'air_quality': np.random.normal(50, 10, 1000)
})

# Introduce anomalies
df.loc[500:520, 'air_quality'] += 100  # Sudden spike in air pollution

# Preprocess data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df[['temperature', 'humidity', 'air_quality']])

# Create sequences
def create_sequences(data, seq_length):
    sequences = []
    for i in range(len(data) - seq_length + 1):
        seq = data[i:i+seq_length]
        sequences.append(seq)
    return np.array(sequences)

seq_length = 24  # Use 24 hours of data to predict the next hour
X = create_sequences(scaled_data, seq_length)

# Train STEN model
model = Sequential([
    LSTM(64, activation='relu', input_shape=(seq_length, 3), return_sequences=True),
    LSTM(32, activation='relu', return_sequences=False),
    Dense(3)
])
model.compile(optimizer='adam', loss='mse')
model.fit(X[:-1], scaled_data[seq_length:], epochs=50, batch_size=32, validation_split=0.2, verbose=0)

# Detect anomalies
predictions = model.predict(X)
mse = np.mean(np.power(X[:, -1, :] - predictions, 2), axis=1)
threshold = np.mean(mse) + 2 * np.std(mse)
anomalies = mse > threshold

# Visualize results
plt.figure(figsize=(12, 6))
plt.plot(df['timestamp'][seq_length:], df['air_quality'][seq_length:], label='Actual')
plt.scatter(df['timestamp'][seq_length:][anomalies], df['air_quality'][seq_length:][anomalies], color='red', label='Anomaly')
plt.title('Air Quality Monitoring with Anomaly Detection')
plt.xlabel('Time')
plt.ylabel('Air Quality Index')
plt.legend()
plt.show()
```

Slide 8: Real-life Example: Network Traffic Analysis

STEN can be employed in network security to detect unusual patterns in network traffic, potentially indicating cyber attacks or network issues.

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Simulated network traffic data
np.random.seed(42)
dates = pd.date_range(start='2023-01-01', periods=1000, freq='5min')
df = pd.DataFrame({
    'timestamp': dates,
    'incoming_traffic': np.random.poisson(100, 1000),
    'outgoing_traffic': np.random.poisson(80, 1000),
    'active_connections': np.random.poisson(50, 1000)
})

# Introduce anomalies
df.loc[800:820, 'incoming_traffic'] *= 5  # Sudden spike in incoming traffic

# Preprocess data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df[['incoming_traffic', 'outgoing_traffic', 'active_connections']])

# Create sequences
def create_sequences(data, seq_length):
    sequences = []
    for i in range(len(data) - seq_length + 1):
        seq = data[i:i+seq_length]
        sequences.append(seq)
    return np.array(sequences)

seq_length = 12  # Use 1 hour of data to predict the next 5 minutes
X = create_sequences(scaled_data, seq_length)

# Train STEN model
model = Sequential([
    LSTM(64, activation='relu', input_shape=(seq_length, 3), return_sequences=True),
    LSTM(32, activation='relu', return_sequences=False),
    Dense(3)
])
model.compile(optimizer='adam', loss='mse')
model.fit(X[:-1], scaled_data[seq_length:], epochs=50, batch_size=32, validation_split=0.2, verbose=0)

# Detect anomalies
predictions = model.predict(X)
mse = np.mean(np.power(X[:, -1, :] - predictions, 2), axis=1)
threshold = np.mean(mse) + 2 * np.std(mse)
anomalies = mse > threshold

# Visualize results
plt.figure(figsize=(12, 6))
plt.plot(df['timestamp'][seq_length:], df['incoming_traffic'][seq_length:], label='Actual')
plt.scatter(df['timestamp'][seq_length:][anomalies], df['incoming_traffic'][seq_length:][anomalies], color='red', label='Anomaly')
plt.title('Network Traffic Monitoring with Anomaly Detection')
plt.xlabel('Time')
plt.ylabel('Incoming Traffic (packets)')
plt.legend()
plt.show()
```

Slide 9: Handling Seasonal and Trend Components

Many time series exhibit seasonal patterns and long-term trends. STEN can be enhanced to handle these components by incorporating techniques like seasonal decomposition or using more complex architectures.

```python
from statsmodels.tsa.seasonal import seasonal_decompose

# Generate sample data with trend and seasonality
np.random.seed(42)
time = np.arange(1000)
trend = 0.02 * time
seasonality = 10 * np.sin(2 * np.pi * time / 365.25)
noise = np.random.normal(0, 1, 1000)
data = trend + seasonality + noise

# Perform seasonal decomposition
result = seasonal_decompose(data, model='additive', period=365)

# Visualize components
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

# Use residuals for anomaly detection
residuals = result.resid.dropna().values
scaler = MinMaxScaler()
scaled_residuals = scaler.fit_transform(residuals.reshape(-1, 1))

# Create sequences and train STEN model on residuals
X = create_sequences(scaled_residuals, seq_length=24)
model = create_sten_model((24, 1))
model.fit(X[:-1], X[:-1], epochs=50, batch_size=32, validation_split=0.2, verbose=0)

# Detect anomalies in residuals
predictions = model.predict(X)
mse = np.mean(np.power(X - predictions, 2), axis=(1, 2))
threshold = np.mean(mse) + 2 * np.std(mse)
anomalies = mse > threshold

plt.figure(figsize=(12, 6))
plt.plot(residuals, label='Residuals')
plt.scatter(np.where(anomalies)[0], residuals[anomalies], color='red', label='Anomaly')
plt.title('Anomaly Detection on Residuals')
plt.xlabel('Time')
plt.ylabel('Residual Value')
plt.legend()
plt.show()
```

Slide 10: Handling Multivariate Time Series

STEN can be extended to handle multivariate time series, where multiple variables are observed simultaneously. This is particularly useful in complex systems where anomalies may manifest across multiple dimensions.

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Generate multivariate time series data
np.random.seed(42)
dates = pd.date_range(start='2023-01-01', periods=1000, freq='H')
df = pd.DataFrame({
    'timestamp': dates,
    'temperature': np.sin(np.arange(1000) * 0.02) + np.random.normal(20, 2, 1000),
    'humidity': np.cos(np.arange(1000) * 0.02) + np.random.normal(60, 5, 1000),
    'pressure': np.random.normal(1013, 5, 1000),
    'wind_speed': np.abs(np.random.normal(0, 5, 1000))
})

# Introduce correlated anomalies
df.loc[500:510, ['temperature', 'humidity']] += [10, -20]

# Preprocess data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df[['temperature', 'humidity', 'pressure', 'wind_speed']])

# Create sequences
def create_sequences(data, seq_length):
    sequences = []
    for i in range(len(data) - seq_length + 1):
        seq = data[i:i+seq_length]
        sequences.append(seq)
    return np.array(sequences)

seq_length = 24
X = create_sequences(scaled_data, seq_length)

# Train STEN model
model = Sequential([
    LSTM(64, activation='relu', input_shape=(seq_length, 4), return_sequences=True),
    LSTM(32, activation='relu', return_sequences=False),
    Dense(4)
])
model.compile(optimizer='adam', loss='mse')
model.fit(X[:-1], scaled_data[seq_length:], epochs=50, batch_size=32, validation_split=0.2, verbose=0)

# Detect anomalies
predictions = model.predict(X)
mse = np.mean(np.power(X[:, -1, :] - predictions, 2), axis=1)
threshold = np.mean(mse) + 2 * np.std(mse)
anomalies = mse > threshold

# Visualize results
plt.figure(figsize=(12, 6))
plt.plot(df['timestamp'][seq_length:], df['temperature'][seq_length:], label='Temperature')
plt.plot(df['timestamp'][seq_length:], df['humidity'][seq_length:], label='Humidity')
plt.scatter(df['timestamp'][seq_length:][anomalies], 
            df['temperature'][seq_length:][anomalies], 
            color='red', marker='x', s=50, label='Anomaly')
plt.title('Multivariate Time Series Anomaly Detection')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.show()
```

Slide 11: Feature Importance in STEN

Understanding which features contribute most to anomaly detection can provide valuable insights. Techniques like SHAP (SHapley Additive exPlanations) can be used to interpret STEN models and identify the most influential variables.

```python
import shap
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense

# Create a simplified STEN model for interpretation
input_shape = (seq_length, 4)
inputs = Input(shape=input_shape)
lstm = LSTM(32, activation='relu')(inputs)
outputs = Dense(4)(lstm)
model = Model(inputs=inputs, outputs=outputs)
model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(X, scaled_data[seq_length:], epochs=50, batch_size=32, verbose=0)

# Create an explainer
explainer = shap.DeepExplainer(model, X[:100])

# Calculate SHAP values
shap_values = explainer.shap_values(X[100:110])

# Visualize feature importance
shap.summary_plot(shap_values[0], X[100:110], feature_names=['Temperature', 'Humidity', 'Pressure', 'Wind Speed'])
```

Slide 12: Ensemble Methods for STEN

Ensemble methods can improve the robustness and accuracy of STEN models by combining multiple models or techniques. This approach can help capture different aspects of the data and reduce false positives.

```python
from sklearn.ensemble import IsolationForest
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# STEN model
sten_model = Sequential([
    LSTM(64, activation='relu', input_shape=(seq_length, 4), return_sequences=True),
    LSTM(32, activation='relu', return_sequences=False),
    Dense(4)
])
sten_model.compile(optimizer='adam', loss='mse')
sten_model.fit(X, scaled_data[seq_length:], epochs=50, batch_size=32, verbose=0)

# Isolation Forest model
iso_forest = IsolationForest(contamination=0.1, random_state=42)
iso_forest.fit(scaled_data)

# Combine predictions
sten_mse = np.mean(np.power(X[:, -1, :] - sten_model.predict(X), 2), axis=1)
sten_anomalies = sten_mse > np.mean(sten_mse) + 2 * np.std(sten_mse)

iso_anomalies = iso_forest.predict(scaled_data) == -1

# Ensemble anomaly detection
ensemble_anomalies = sten_anomalies[seq_length:] & iso_anomalies[seq_length:]

# Visualize results
plt.figure(figsize=(12, 6))
plt.plot(df['timestamp'][seq_length:], df['temperature'][seq_length:], label='Temperature')
plt.scatter(df['timestamp'][seq_length:][ensemble_anomalies], 
            df['temperature'][seq_length:][ensemble_anomalies], 
            color='red', marker='x', s=50, label='Ensemble Anomaly')
plt.title('Ensemble Anomaly Detection')
plt.xlabel('Time')
plt.ylabel('Temperature')
plt.legend()
plt.show()
```

Slide 13: Challenges and Future Directions

While STEN is a powerful technique for time series anomaly detection, it faces several challenges:

1. Handling concept drift and evolving patterns in time series data
2. Balancing model complexity with interpretability
3. Dealing with extremely rare events or black swan scenarios
4. Adapting to different types of anomalies (point, contextual, and collective anomalies)

Future research directions include:

1. Incorporating transfer learning techniques to adapt STEN models across different domains
2. Developing self-supervised learning approaches for STEN to leverage unlabeled data
3. Exploring federated learning for privacy-preserving anomaly detection in distributed systems
4. Integrating explainable AI techniques to improve model interpretability and trust

```python
# Pseudocode for an adaptive STEN model

class AdaptiveSTEN:
    def __init__(self, input_shape, learning_rate):
        self.model = create_sten_model(input_shape)
        self.learning_rate = learning_rate
    
    def detect_anomalies(self, data):
        predictions = self.model.predict(data)
        errors = calculate_reconstruction_error(data, predictions)
        return errors > self.calculate_threshold(errors)
    
    def update_model(self, new_data):
        self.model.fit(new_data, new_data, epochs=1, batch_size=32)
    
    def calculate_threshold(self, errors):
        return np.mean(errors) + 2 * np.std(errors)
    
    def adapt_to_concept_drift(self, data_stream):
        for batch in data_stream:
            anomalies = self.detect_anomalies(batch)
            self.update_model(batch[~anomalies])  # Update model with non-anomalous data
            yield anomalies

# Usage
adaptive_sten = AdaptiveSTEN(input_shape=(24, 4), learning_rate=0.001)
for anomalies in adaptive_sten.adapt_to_concept_drift(data_stream):
    process_anomalies(anomalies)
```

Slide 14: Additional Resources

For those interested in diving deeper into STEN and related techniques, here are some valuable resources:

1. "Deep Learning for Time Series Forecasting" by N. Laptev, J. Yosinski, L. E. Li, and S. Smyl (2017). ArXiv:1701.01887 \[cs.LG\] [https://arxiv.org/abs/1701.01887](https://arxiv.org/abs/1701.01887)
2. "Anomaly Detection in Time Series: A Comprehensive Evaluation" by G. Bontempi, S. Ben Taieb, and Y.-A. Le Borgne (2021). ArXiv:2103.16236 \[cs.LG\] [https://arxiv.org/abs/2103.16236](https://arxiv.org/abs/2103.16236)
3. "A Survey on Deep Learning for Time Series Forecasting" by B. Lim and S. Zohren (2020). ArXiv:2004.13408 \[cs.LG\] [https://arxiv.org/abs/2004.13408](https://arxiv.org/abs/2004.13408)

These papers provide in-depth discussions on various aspects of time series analysis, deep learning approaches, and anomaly detection techniques, which can complement the understanding of STEN and its applications.

