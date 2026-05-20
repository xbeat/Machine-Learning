## Anomaly Detection in Time Series Data with Python
Slide 1: Introduction to Anomaly Detection in Time Series Data

Anomaly detection in time series data is a crucial task in various fields, from monitoring industrial processes to analyzing environmental changes. This technique helps identify unusual patterns or events that deviate significantly from expected behavior. In this presentation, we'll explore how to perform anomaly detection using Python, focusing on practical examples and actionable insights.

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate a sample time series with an anomaly
np.random.seed(42)
time = np.arange(0, 100, 0.1)
signal = np.sin(time) + np.random.normal(0, 0.1, len(time))
signal[800:830] += 2  # Introduce an anomaly

plt.figure(figsize=(12, 6))
plt.plot(time, signal)
plt.title('Time Series with Anomaly')
plt.xlabel('Time')
plt.ylabel('Signal')
plt.show()
```

Slide 2: Statistical Approaches: Moving Average

One of the simplest methods for anomaly detection is using a moving average. This technique calculates the average of a fixed-size window of data points and compares each point to this average. If a point deviates significantly from the moving average, it's flagged as an anomaly.

```python
import numpy as np
import matplotlib.pyplot as plt

def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size), 'valid') / window_size

# Generate sample data
np.random.seed(42)
time = np.arange(1000)
signal = np.sin(time * 0.05) + np.random.normal(0, 0.5, 1000)
signal[700:720] += 5  # Introduce an anomaly

# Calculate moving average
window_size = 50
ma = moving_average(signal, window_size)

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(time, signal, label='Original Signal')
plt.plot(time[window_size-1:], ma, label='Moving Average', color='red')
plt.title('Time Series with Moving Average')
plt.xlabel('Time')
plt.ylabel('Signal')
plt.legend()
plt.show()
```

Slide 3: Detecting Anomalies with Moving Average

To detect anomalies using the moving average method, we can set a threshold based on the standard deviation of the signal. Points that deviate from the moving average by more than a certain number of standard deviations are considered anomalies.

```python
import numpy as np
import matplotlib.pyplot as plt

def detect_anomalies(signal, ma, threshold=2):
    std = np.std(signal)
    anomalies = np.abs(signal - ma) > threshold * std
    return anomalies

# Using the same data from the previous slide
window_size = 50
ma = moving_average(signal, window_size)

# Detect anomalies
anomalies = detect_anomalies(signal[window_size-1:], ma)

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(time, signal, label='Original Signal')
plt.plot(time[window_size-1:], ma, label='Moving Average', color='red')
plt.scatter(time[window_size-1:][anomalies], signal[window_size-1:][anomalies], 
            color='green', label='Anomalies')
plt.title('Anomaly Detection using Moving Average')
plt.xlabel('Time')
plt.ylabel('Signal')
plt.legend()
plt.show()
```

Slide 4: Z-Score Method

The Z-score method is another statistical approach for anomaly detection. It measures how many standard deviations away a data point is from the mean. This method is particularly useful when the data follows a normal distribution.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def z_score_anomalies(data, threshold=3):
    z_scores = np.abs(stats.zscore(data))
    return z_scores > threshold

# Generate sample data
np.random.seed(42)
time = np.arange(1000)
signal = np.random.normal(0, 1, 1000)
signal[800:820] = 5  # Introduce anomalies

# Detect anomalies using Z-score
anomalies = z_score_anomalies(signal)

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(time, signal, label='Signal')
plt.scatter(time[anomalies], signal[anomalies], color='red', label='Anomalies')
plt.title('Anomaly Detection using Z-Score Method')
plt.xlabel('Time')
plt.ylabel('Signal')
plt.legend()
plt.show()
```

Slide 5: Seasonal Decomposition

Many time series exhibit seasonal patterns. Seasonal decomposition helps separate a time series into trend, seasonal, and residual components. Anomalies can then be detected in the residual component.

```python
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# Generate sample seasonal data
np.random.seed(42)
time = np.arange(1000)
trend = 0.01 * time
seasonal = 5 * np.sin(2 * np.pi * time / 100)
residual = np.random.normal(0, 1, 1000)
signal = trend + seasonal + residual
signal[800:820] += 10  # Introduce anomalies

# Perform seasonal decomposition
result = seasonal_decompose(signal, model='additive', period=100)

# Plot results
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 16))
result.observed.plot(ax=ax1)
ax1.set_title('Original Signal')
result.trend.plot(ax=ax2)
ax2.set_title('Trend')
result.seasonal.plot(ax=ax3)
ax3.set_title('Seasonal')
result.resid.plot(ax=ax4)
ax4.set_title('Residual')
plt.tight_layout()
plt.show()
```

Slide 6: Detecting Anomalies in Residuals

After seasonal decomposition, we can apply anomaly detection techniques to the residual component. This approach helps identify anomalies that are not part of the regular seasonal pattern or overall trend.

```python
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# Using the same seasonal data from the previous slide
result = seasonal_decompose(signal, model='additive', period=100)

# Detect anomalies in residuals using Z-score method
residual_anomalies = z_score_anomalies(result.resid, threshold=3)

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(time, result.resid, label='Residual')
plt.scatter(time[residual_anomalies], result.resid[residual_anomalies], 
            color='red', label='Anomalies')
plt.title('Anomaly Detection in Residual Component')
plt.xlabel('Time')
plt.ylabel('Residual')
plt.legend()
plt.show()
```

Slide 7: Machine Learning Approach: Isolation Forest

Isolation Forest is an unsupervised machine learning algorithm that's particularly effective for anomaly detection. It works by isolating anomalies in the data rather than profiling normal points.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

# Generate sample data
np.random.seed(42)
time = np.arange(1000).reshape(-1, 1)
signal = np.sin(time * 0.05) + np.random.normal(0, 0.2, (1000, 1))
signal[800:820] += 5  # Introduce anomalies

# Train Isolation Forest
clf = IsolationForest(contamination=0.01, random_state=42)
clf.fit(signal)

# Predict anomalies
anomalies = clf.predict(signal) == -1

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(time, signal, label='Signal')
plt.scatter(time[anomalies], signal[anomalies], color='red', label='Anomalies')
plt.title('Anomaly Detection using Isolation Forest')
plt.xlabel('Time')
plt.ylabel('Signal')
plt.legend()
plt.show()
```

Slide 8: Real-life Example: Temperature Monitoring

Let's consider a real-life example of temperature monitoring in a manufacturing process. We'll generate synthetic data to simulate temperature readings over time and detect anomalies that could indicate equipment malfunction or process issues.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

# Generate synthetic temperature data
np.random.seed(42)
time = np.arange(1000)
temperature = 20 + 5 * np.sin(2 * np.pi * time / 100) + np.random.normal(0, 1, 1000)
temperature[700:720] += 15  # Simulate equipment malfunction

# Reshape data for Isolation Forest
X = temperature.reshape(-1, 1)

# Train Isolation Forest
clf = IsolationForest(contamination=0.02, random_state=42)
clf.fit(X)

# Predict anomalies
anomalies = clf.predict(X) == -1

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(time, temperature, label='Temperature')
plt.scatter(time[anomalies], temperature[anomalies], color='red', label='Anomalies')
plt.title('Temperature Monitoring with Anomaly Detection')
plt.xlabel('Time (hours)')
plt.ylabel('Temperature (°C)')
plt.legend()
plt.show()
```

Slide 9: Real-life Example: Network Traffic Analysis

Another practical application of anomaly detection is in network traffic analysis. We'll simulate network traffic data and use the Z-score method to identify potential security threats or unusual network behavior.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Generate synthetic network traffic data
np.random.seed(42)
time = np.arange(1000)
traffic = np.random.poisson(50, 1000)  # Normal traffic
traffic[800:820] = np.random.poisson(200, 20)  # Simulate traffic spike

# Detect anomalies using Z-score method
def z_score_anomalies(data, threshold=3):
    z_scores = np.abs(stats.zscore(data))
    return z_scores > threshold

anomalies = z_score_anomalies(traffic, threshold=3)

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(time, traffic, label='Network Traffic')
plt.scatter(time[anomalies], traffic[anomalies], color='red', label='Anomalies')
plt.title('Network Traffic Analysis with Anomaly Detection')
plt.xlabel('Time (minutes)')
plt.ylabel('Traffic Volume (packets/min)')
plt.legend()
plt.show()
```

Slide 10: Handling Multiple Variables: Multivariate Anomaly Detection

In many real-world scenarios, we need to consider multiple variables simultaneously. Multivariate anomaly detection techniques can help identify anomalies in complex, multi-dimensional data.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.covariance import EllipticEnvelope

# Generate multivariate data
np.random.seed(42)
n_samples = 1000
n_outliers = 50
n_features = 2

# Generate normal data
X = np.random.randn(n_samples - n_outliers, n_features)

# Generate outliers
outliers = np.random.uniform(low=-4, high=4, size=(n_outliers, n_features))
X = np.r_[X, outliers]

# Fit the Elliptic Envelope model
ee = EllipticEnvelope(contamination=0.05, random_state=42)
ee.fit(X)

# Predict anomalies
y_pred = ee.predict(X)

# Plot results
plt.figure(figsize=(10, 8))
plt.scatter(X[y_pred == 1, 0], X[y_pred == 1, 1], c='blue', label='Normal')
plt.scatter(X[y_pred == -1, 0], X[y_pred == -1, 1], c='red', label='Anomalies')
plt.title('Multivariate Anomaly Detection')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()
```

Slide 11: Time Series Forecasting for Anomaly Detection

Combining time series forecasting with anomaly detection can be powerful. We can use forecasting models to predict expected values and then identify anomalies as significant deviations from these predictions.

```python
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# Generate sample data
np.random.seed(42)
time = np.arange(1000)
signal = np.sin(time * 0.05) + np.random.normal(0, 0.2, 1000)
signal[900:920] += 2  # Introduce anomalies

# Fit ARIMA model
model = ARIMA(signal[:800], order=(1, 1, 1))
results = model.fit()

# Make predictions
predictions = results.forecast(steps=200)

# Calculate prediction intervals
pred_int = results.get_forecast(steps=200).conf_int()

# Detect anomalies
anomalies = (signal[800:] < pred_int[:, 0]) | (signal[800:] > pred_int[:, 1])

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(time[:800], signal[:800], label='Training Data')
plt.plot(time[800:], signal[800:], label='Actual Data')
plt.plot(time[800:], predictions, label='Forecast', color='red')
plt.fill_between(time[800:], pred_int[:, 0], pred_int[:, 1], color='pink', alpha=0.3)
plt.scatter(time[800:][anomalies], signal[800:][anomalies], color='green', label='Anomalies')
plt.title('Time Series Forecasting and Anomaly Detection')
plt.xlabel('Time')
plt.ylabel('Signal')
plt.legend()
plt.show()
```

Slide 12: Ensemble Methods for Robust Anomaly Detection

Combining multiple anomaly detection techniques can lead to more robust and accurate results. We'll demonstrate an ensemble approach using different methods we've covered.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.ensemble import IsolationForest

def z_score_anomalies(data, threshold=3):
    return np.abs(stats.zscore(data)) > threshold

def moving_average_anomalies(data, window_size=20, threshold=2):
    ma = np.convolve(data, np.ones(window_size), 'valid') / window_size
    residuals = data[window_size-1:] - ma
    return np.abs(residuals) > threshold * np.std(residuals)

# Generate sample data
np.random.seed(42)
time = np.arange(1000)
signal = np.sin(time * 0.05) + np.random.normal(0, 0.2, 1000)
signal[800:820] += 3  # Introduce anomalies

# Apply different methods
z_score_result = z_score_anomalies(signal)
ma_result = np.pad(moving_average_anomalies(signal), (19, 0), 'constant')
iso_forest = IsolationForest(contamination=0.02, random_state=42)
iso_forest_result = iso_forest.fit_predict(signal.reshape(-1, 1)) == -1

# Combine results (majority voting)
ensemble_result = ((z_score_result.astype(int) + 
                    ma_result.astype(int) + 
                    iso_forest_result.astype(int)) >= 2)

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(time, signal, label='Signal')
plt.scatter(time[ensemble_result], signal[ensemble_result], color='red', label='Ensemble Anomalies')
plt.title('Ensemble Anomaly Detection')
plt.xlabel('Time')
plt.ylabel('Signal')
plt.legend()
plt.show()
```

Slide 13: Evaluating Anomaly Detection Performance

Assessing the performance of anomaly detection algorithms is crucial. We'll explore common metrics and visualization techniques to evaluate our models.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

# Generate data with known anomalies
np.random.seed(42)
time = np.arange(1000)
signal = np.sin(time * 0.05) + np.random.normal(0, 0.2, 1000)
true_anomalies = np.zeros(1000, dtype=bool)
true_anomalies[800:820] = True
signal[true_anomalies] += 3

# Detect anomalies using a simple threshold
threshold = np.mean(signal) + 2 * np.std(signal)
detected_anomalies = signal > threshold

# Calculate metrics
cm = confusion_matrix(true_anomalies, detected_anomalies)
precision = precision_score(true_anomalies, detected_anomalies)
recall = recall_score(true_anomalies, detected_anomalies)
f1 = f1_score(true_anomalies, detected_anomalies)

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(time, signal, label='Signal')
plt.axhline(y=threshold, color='r', linestyle='--', label='Threshold')
plt.scatter(time[detected_anomalies], signal[detected_anomalies], color='red', label='Detected Anomalies')
plt.scatter(time[true_anomalies], signal[true_anomalies], color='green', marker='x', s=100, label='True Anomalies')
plt.title(f'Anomaly Detection Evaluation (Precision: {precision:.2f}, Recall: {recall:.2f}, F1: {f1:.2f})')
plt.xlabel('Time')
plt.ylabel('Signal')
plt.legend()
plt.show()

# Print confusion matrix
print("Confusion Matrix:")
print(cm)
```

Slide 14: Challenges and Considerations in Anomaly Detection

Anomaly detection in time series data comes with various challenges. We'll discuss some key considerations and potential solutions.

1. Concept Drift: Time series data may evolve over time, making static models less effective. Solution: Use adaptive algorithms or regularly retrain models.
2. Seasonal Patterns: Complex seasonal patterns can be mistaken for anomalies. Solution: Apply seasonal decomposition or use domain knowledge to model seasonality.
3. Imbalanced Data: Anomalies are typically rare, leading to class imbalance issues. Solution: Use appropriate evaluation metrics and techniques like oversampling or undersampling.
4. Multivariate Time Series: Dealing with multiple interdependent variables can be complex. Solution: Use dimensionality reduction techniques or multivariate anomaly detection algorithms.
5. Real-time Detection: Some applications require immediate anomaly detection. Solution: Implement streaming algorithms or use efficient online learning methods.


Slide 15: Challenges and Considerations in Anomaly Detection

```python
# Pseudocode for handling concept drift
def adaptive_anomaly_detection(data_stream):
    model = initialize_model()
    for data_point in data_stream:
        prediction = model.predict(data_point)
        if is_anomaly(prediction):
            report_anomaly(data_point)
        model.update(data_point)
```

Slide 16: Additional Resources

For those interested in diving deeper into anomaly detection in time series data, here are some valuable resources:

1. "Outlier Detection for Temporal Data" by Gupta et al. (2014) ArXiv link: [https://arxiv.org/abs/1401.3665](https://arxiv.org/abs/1401.3665)
2. "A Survey of Deep Learning Techniques for Anomaly Detection in Time Series Data" by Aljohani et al. (2023) ArXiv link: [https://arxiv.org/abs/2305.18415](https://arxiv.org/abs/2305.18415)
3. "Time Series Anomaly Detection; A Survey" by Braei and Wagner (2020) ArXiv link: [https://arxiv.org/abs/2004.00433](https://arxiv.org/abs/2004.00433)

These papers provide comprehensive overviews of various techniques and recent advancements in the field of time series anomaly detection.

