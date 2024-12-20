## Spatial-Temporal Normality Learning for Time Series Anomaly Detection
Slide 1: Introduction to Spatial-Temporal Normality Learning (STEN)

Spatial-Temporal Normality Learning (STEN) is a powerful technique for detecting anomalies in time series data. It combines temporal and spatial dimensions to identify unusual patterns or behaviors. STEN is particularly useful in various domains, such as IoT sensor networks, network traffic analysis, and environmental monitoring.

```python
import numpy as np
import matplotlib.pyplot as plt

# Simulating a normal time series
np.random.seed(42)
time = np.arange(100)
normal_series = np.sin(time * 0.1) + np.random.normal(0, 0.1, 100)

# Introducing an anomaly
anomaly_index = 70
normal_series[anomaly_index] += 2

plt.figure(figsize=(10, 5))
plt.plot(time, normal_series)
plt.axvline(x=anomaly_index, color='r', linestyle='--', label='Anomaly')
plt.title('Time Series with Anomaly')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.show()
```

Slide 2: Temporal Dimension in STEN

The temporal dimension in STEN focuses on the sequential nature of time series data. It captures patterns and dependencies across different time steps, allowing the model to learn normal temporal behavior and identify deviations.

```python
import pandas as pd

# Creating a time series dataset
dates = pd.date_range(start='2023-01-01', end='2023-01-10', freq='H')
values = np.sin(np.arange(len(dates)) * 0.1) + np.random.normal(0, 0.1, len(dates))
df = pd.DataFrame({'timestamp': dates, 'value': values})

# Extracting temporal features
df['hour'] = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.dayofweek

print(df.head())
```

Slide 3: Spatial Dimension in STEN

The spatial dimension in STEN considers the relationships between different variables or sensors in a multivariate time series. It helps capture correlations and dependencies across multiple data streams, enabling the detection of anomalies that may not be apparent when examining each variable independently.

```python
import seaborn as sns

# Simulating multivariate time series data
np.random.seed(42)
n_sensors = 5
n_timestamps = 100
data = np.random.randn(n_timestamps, n_sensors)

# Introducing correlations between sensors
data[:, 1] = data[:, 0] * 0.8 + np.random.randn(n_timestamps) * 0.2
data[:, 2] = data[:, 1] * 0.7 + np.random.randn(n_timestamps) * 0.3

# Visualizing correlations
corr_matrix = np.corrcoef(data.T)
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix of Sensor Data')
plt.show()
```

Slide 4: Order Prediction-based Temporal Normality Learning (OTN)

OTN is a component of STEN that focuses on learning temporal patterns by predicting the order of events or values in a time series. It helps identify anomalies by detecting unexpected sequences or temporal dependencies.

```python
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Prepare data for OTN
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(normal_series.reshape(-1, 1))

# Create sequences for order prediction
seq_length = 10
X, y = [], []
for i in range(len(scaled_data) - seq_length):
    X.append(scaled_data[i:i+seq_length])
    y.append(scaled_data[i+seq_length])

X = np.array(X)
y = np.array(y)

# Build and train OTN model
model = Sequential([
    LSTM(50, activation='relu', input_shape=(seq_length, 1)),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=50, verbose=0)

# Predict next value
last_sequence = scaled_data[-seq_length:].reshape(1, seq_length, 1)
predicted = model.predict(last_sequence)
print(f"Predicted next value: {scaler.inverse_transform(predicted)[0][0]:.2f}")
```

Slide 5: Distance Prediction-based Spatial Normality Learning (DSN)

DSN is another component of STEN that focuses on learning spatial relationships between different variables or sensors. It predicts the distances or similarities between data points in the spatial dimension, helping to identify anomalies that deviate from normal spatial patterns.

```python
from sklearn.metrics.pairwise import euclidean_distances

# Simulating multivariate sensor data
n_sensors = 5
n_timestamps = 100
sensor_data = np.random.randn(n_timestamps, n_sensors)

# Calculate pairwise distances between sensors
distances = euclidean_distances(sensor_data.T)

# Visualize distance matrix
plt.figure(figsize=(8, 6))
sns.heatmap(distances, annot=True, cmap='viridis')
plt.title('Pairwise Distances Between Sensors')
plt.xlabel('Sensor ID')
plt.ylabel('Sensor ID')
plt.show()

# Predict distance for a new data point
new_data_point = np.random.randn(1, n_sensors)
predicted_distances = euclidean_distances(new_data_point, sensor_data.T)
print("Predicted distances for new data point:", predicted_distances[0])
```

Slide 6: Combining OTN and DSN in STEN

STEN combines the strengths of OTN and DSN to create a comprehensive anomaly detection framework. By considering both temporal and spatial dimensions, STEN can detect complex anomalies that may be missed by traditional methods.

```python
import tensorflow as tf

# Simplified STEN model combining OTN and DSN
class STENModel(tf.keras.Model):
    def __init__(self, seq_length, n_sensors):
        super(STENModel, self).__init__()
        self.lstm = LSTM(50, activation='relu', input_shape=(seq_length, n_sensors))
        self.dense_temporal = Dense(n_sensors)
        self.dense_spatial = Dense(n_sensors * (n_sensors - 1) // 2)
    
    def call(self, inputs):
        x = self.lstm(inputs)
        temporal_output = self.dense_temporal(x)
        spatial_output = self.dense_spatial(x)
        return temporal_output, spatial_output

# Create and compile the model
seq_length = 10
n_sensors = 5
model = STENModel(seq_length, n_sensors)
model.compile(optimizer='adam', loss=['mse', 'mse'])

# Generate dummy data and train the model
X = np.random.randn(100, seq_length, n_sensors)
y_temporal = np.random.randn(100, n_sensors)
y_spatial = np.random.randn(100, n_sensors * (n_sensors - 1) // 2)
model.fit(X, [y_temporal, y_spatial], epochs=10, verbose=0)

print("STEN model trained successfully")
```

Slide 7: Anomaly Score Calculation

The anomaly score in STEN is typically calculated by combining the prediction errors from both the temporal (OTN) and spatial (DSN) components. A higher anomaly score indicates a higher likelihood of an anomaly.

```python
def calculate_anomaly_score(temporal_error, spatial_error, alpha=0.5):
    """
    Calculate the anomaly score using a weighted combination of temporal and spatial errors.
    
    :param temporal_error: Error from the OTN component
    :param spatial_error: Error from the DSN component
    :param alpha: Weight for temporal error (1 - alpha for spatial error)
    :return: Anomaly score
    """
    return alpha * temporal_error + (1 - alpha) * spatial_error

# Simulate prediction errors
temporal_errors = np.abs(np.random.randn(100))
spatial_errors = np.abs(np.random.randn(100))

# Calculate anomaly scores
anomaly_scores = [calculate_anomaly_score(te, se) for te, se in zip(temporal_errors, spatial_errors)]

# Visualize anomaly scores
plt.figure(figsize=(10, 5))
plt.plot(anomaly_scores)
plt.title('Anomaly Scores')
plt.xlabel('Time')
plt.ylabel('Anomaly Score')
plt.show()

print(f"Mean anomaly score: {np.mean(anomaly_scores):.2f}")
print(f"Max anomaly score: {np.max(anomaly_scores):.2f}")
```

Slide 8: Evaluation Metrics - Area Under the Receiver Operating Characteristic Curve (AUC-ROC)

AUC-ROC is a common metric for evaluating the performance of anomaly detection models. It measures the model's ability to distinguish between normal and anomalous data points across different threshold values.

```python
from sklearn.metrics import roc_curve, roc_auc_score
import numpy as np

# Simulate true labels and predicted probabilities
np.random.seed(42)
y_true = np.random.randint(0, 2, 1000)
y_pred = np.random.rand(1000)

# Calculate ROC curve and AUC
fpr, tpr, thresholds = roc_curve(y_true, y_pred)
auc_roc = roc_auc_score(y_true, y_pred)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc_roc:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--', label='Random classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.show()

print(f"AUC-ROC score: {auc_roc:.2f}")
```

Slide 9: Evaluation Metrics - Area Under the Precision-Recall Curve (AUC-PR)

AUC-PR is another important metric for anomaly detection, especially when dealing with imbalanced datasets where anomalies are rare. It focuses on the trade-off between precision and recall at different threshold values.

```python
from sklearn.metrics import precision_recall_curve, average_precision_score

# Calculate precision-recall curve and AUC-PR
precision, recall, _ = precision_recall_curve(y_true, y_pred)
auc_pr = average_precision_score(y_true, y_pred)

# Plot precision-recall curve
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, label=f'PR curve (AUC = {auc_pr:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.show()

print(f"AUC-PR score: {auc_pr:.2f}")
```

Slide 10: Evaluation Metrics - Best F1 Score

The F1 score is the harmonic mean of precision and recall. The best F1 score is the highest F1 score achieved across different threshold values, providing a balanced measure of the model's performance.

```python
from sklearn.metrics import f1_score

def find_best_f1_score(y_true, y_pred_proba):
    thresholds = np.linspace(0, 1, 100)
    f1_scores = []
    
    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        f1 = f1_score(y_true, y_pred)
        f1_scores.append(f1)
    
    best_f1 = max(f1_scores)
    best_threshold = thresholds[np.argmax(f1_scores)]
    
    return best_f1, best_threshold

# Find best F1 score and corresponding threshold
best_f1, best_threshold = find_best_f1_score(y_true, y_pred)

print(f"Best F1 score: {best_f1:.2f}")
print(f"Best threshold: {best_threshold:.2f}")

# Plot F1 scores for different thresholds
thresholds = np.linspace(0, 1, 100)
f1_scores = [f1_score(y_true, (y_pred >= t).astype(int)) for t in thresholds]

plt.figure(figsize=(8, 6))
plt.plot(thresholds, f1_scores)
plt.axvline(x=best_threshold, color='r', linestyle='--', label=f'Best threshold: {best_threshold:.2f}')
plt.xlabel('Threshold')
plt.ylabel('F1 Score')
plt.title('F1 Score vs. Threshold')
plt.legend()
plt.show()
```

Slide 11: Real-Life Example - Network Traffic Anomaly Detection

STEN can be applied to network traffic analysis to detect unusual patterns or potential security threats. In this example, we'll simulate network traffic data and use STEN to identify anomalies.

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Simulate network traffic data
np.random.seed(42)
n_samples = 1000
timestamps = pd.date_range(start='2023-01-01', periods=n_samples, freq='5T')
packet_count = np.random.poisson(lam=100, size=n_samples)
byte_count = packet_count * np.random.randint(100, 1500, size=n_samples)
unique_ips = np.random.randint(10, 100, size=n_samples)

# Introduce anomalies
anomaly_indices = [200, 500, 800]
packet_count[anomaly_indices] *= 10
byte_count[anomaly_indices] *= 15
unique_ips[anomaly_indices] *= 5

# Create DataFrame
df = pd.DataFrame({
    'timestamp': timestamps,
    'packet_count': packet_count,
    'byte_count': byte_count,
    'unique_ips': unique_ips
})

# Normalize features
scaler = StandardScaler()
normalized_data = scaler.fit_transform(df[['packet_count', 'byte_count', 'unique_ips']])

# Simple anomaly detection using Z-score
z_scores = np.abs(normalized_data).mean(axis=1)
threshold = 3
anomalies = z_scores > threshold

# Visualize results
plt.figure(figsize=(12, 6))
plt.plot(df['timestamp'], z_scores, label='Anomaly Score')
plt.axhline(y=threshold, color='r', linestyle='--', label='Threshold')
plt.scatter(df['timestamp'][anomalies], z_scores[anomalies], color='red', label='Detected Anomalies')
plt.title('Network Traffic Anomaly Detection')
plt.xlabel('Timestamp')
plt.ylabel('Anomaly Score')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print(f"Number of detected anomalies: {anomalies.sum()}")
```

Slide 12: Real-Life Example - Environmental Monitoring

STEN can be applied to environmental monitoring to detect unusual patterns in sensor data. This example simulates temperature and humidity data from multiple sensors and applies a simplified version of STEN to identify anomalies.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Simulate environmental sensor data
np.random.seed(42)
n_sensors = 5
n_samples = 1000
timestamps = pd.date_range(start='2023-01-01', periods=n_samples, freq='H')

# Generate normal patterns with daily and seasonal variations
time = np.arange(n_samples)
base_temp = 20 + 5 * np.sin(2 * np.pi * time / (24 * 365)) + 2 * np.sin(2 * np.pi * time / 24)
base_humidity = 60 + 10 * np.sin(2 * np.pi * time / (24 * 365)) - 5 * np.sin(2 * np.pi * time / 24)

# Create sensor data with some variations
temperatures = np.array([base_temp + np.random.normal(0, 1, n_samples) for _ in range(n_sensors)]).T
humidities = np.array([base_humidity + np.random.normal(0, 2, n_samples) for _ in range(n_sensors)]).T

# Introduce anomalies
anomaly_indices = [200, 500, 800]
temperatures[anomaly_indices] += np.random.uniform(5, 10, size=(len(anomaly_indices), n_sensors))
humidities[anomaly_indices] += np.random.uniform(-20, 20, size=(len(anomaly_indices), n_sensors))

# Combine data
data = np.concatenate([temperatures, humidities], axis=1)

# Normalize data
scaler = StandardScaler()
normalized_data = scaler.fit_transform(data)

# Simple anomaly detection using Mahalanobis distance
def mahalanobis_distance(x, mean, cov):
    diff = x - mean
    return np.sqrt(diff.dot(np.linalg.inv(cov)).dot(diff))

mean = np.mean(normalized_data, axis=0)
cov = np.cov(normalized_data.T)
anomaly_scores = np.array([mahalanobis_distance(x, mean, cov) for x in normalized_data])

# Detect anomalies
threshold = np.percentile(anomaly_scores, 99)
anomalies = anomaly_scores > threshold

# Visualize results
plt.figure(figsize=(12, 6))
plt.plot(timestamps, anomaly_scores, label='Anomaly Score')
plt.axhline(y=threshold, color='r', linestyle='--', label='Threshold')
plt.scatter(timestamps[anomalies], anomaly_scores[anomalies], color='red', label='Detected Anomalies')
plt.title('Environmental Monitoring Anomaly Detection')
plt.xlabel('Timestamp')
plt.ylabel('Anomaly Score')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print(f"Number of detected anomalies: {anomalies.sum()}")
```

Slide 13: Challenges and Limitations of STEN

While STEN is a powerful technique for time series anomaly detection, it faces several challenges and limitations:

1. High computational complexity for large-scale datasets
2. Sensitivity to hyperparameter tuning
3. Difficulty in handling concept drift and evolving normal patterns
4. Challenges in interpreting the root cause of detected anomalies

To address these issues, researchers continue to develop improved variants of STEN and hybrid approaches that combine STEN with other machine learning techniques.

```python
# Pseudocode for an adaptive STEN algorithm

class AdaptiveSTEN:
    def __init__(self, window_size, update_frequency):
        self.window_size = window_size
        self.update_frequency = update_frequency
        self.model = initialize_sten_model()
        self.data_buffer = []
    
    def detect_anomalies(self, new_data):
        anomaly_scores = self.model.compute_anomaly_scores(new_data)
        self.data_buffer.extend(new_data)
        
        if len(self.data_buffer) >= self.update_frequency:
            self.update_model()
        
        return anomaly_scores
    
    def update_model(self):
        recent_data = self.data_buffer[-self.window_size:]
        self.model.retrain(recent_data)
        self.data_buffer = []

# Usage
adaptive_sten = AdaptiveSTEN(window_size=1000, update_frequency=100)
for batch in data_stream:
    anomaly_scores = adaptive_sten.detect_anomalies(batch)
    # Process anomaly scores
```

Slide 14: Future Directions and Research Opportunities

The field of time series anomaly detection using STEN continues to evolve. Some promising research directions include:

1. Incorporating attention mechanisms to improve temporal and spatial feature learning
2. Developing unsupervised STEN variants for scenarios with limited labeled data
3. Exploring transfer learning approaches to adapt STEN models across different domains
4. Integrating explainable AI techniques to enhance the interpretability of STEN results
5. Investigating the application of STEN in edge computing environments for real-time anomaly detection

```python
# Pseudocode for a STEN model with attention mechanism

import tensorflow as tf

class STENWithAttention(tf.keras.Model):
    def __init__(self, seq_length, n_sensors):
        super(STENWithAttention, self).__init__()
        self.lstm = tf.keras.layers.LSTM(64, return_sequences=True)
        self.attention = tf.keras.layers.Attention()
        self.dense = tf.keras.layers.Dense(1)
    
    def call(self, inputs):
        lstm_output = self.lstm(inputs)
        attention_output = self.attention([lstm_output, lstm_output])
        return self.dense(attention_output)

# Usage
seq_length = 100
n_sensors = 5
model = STENWithAttention(seq_length, n_sensors)
model.compile(optimizer='adam', loss='mse')

# Train the model
# model.fit(X_train, y_train, epochs=100, validation_data=(X_val, y_val))
```

Slide 15: Additional Resources

For those interested in diving deeper into STEN and related techniques for time series anomaly detection, here are some valuable resources:

1. "Robust Time Series Anomaly Detection with Spatial-Temporal Normality Learning" by Xu et al. (2023) - ArXiv: [https://arxiv.org/abs/2303.08850](https://arxiv.org/abs/2303.08850)
2. "A Comprehensive Survey on Graph Anomaly Detection with Deep Learning" by Ma et al. (2021) - ArXiv: [https://arxiv.org/abs/2106.07178](https://arxiv.org/abs/2106.07178)
3. "Time Series Anomaly Detection: A Survey" by Braei and Wagner (2022) - ArXiv: [https://arxiv.org/abs/2101.02666](https://arxiv.org/abs/2101.02666)

These papers provide in-depth discussions on STEN and related techniques, as well as broader overviews of time series anomaly detection methods.

