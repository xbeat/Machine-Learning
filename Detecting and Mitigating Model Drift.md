## Detecting and Mitigating Model Drift
Slide 1: Understanding Model Drift

Model drift is a crucial concept in machine learning, referring to the degradation of a model's performance over time. This occurs when the statistical properties of the target variable change in unforeseen ways, causing the predictions to become less accurate. It's essential to recognize and address drift to maintain model effectiveness.

```python
import numpy as np
import matplotlib.pyplot as plt

# Simulate model performance over time
np.random.seed(42)
time = np.arange(100)
performance = 0.9 - 0.003 * time + np.random.normal(0, 0.02, 100)

plt.figure(figsize=(10, 6))
plt.plot(time, performance)
plt.title('Model Performance Over Time')
plt.xlabel('Time')
plt.ylabel('Performance Metric')
plt.show()
```

Slide 2: Types of Drift

There are four main types of drift in machine learning:

1. Concept Drift: Changes in the underlying relationships between features and outcomes.
2. Data Drift: Changes observed in the data distribution.
3. Feature Drift: Changes in the distribution of model's input features.
4. Label Drift: Changes in the model's output distribution.

Understanding these types helps in identifying and addressing drift effectively.

```python
import pandas as pd
from scipy import stats

def detect_drift(old_data, new_data):
    drift_types = ['Concept', 'Data', 'Feature', 'Label']
    results = {}
    
    for column in old_data.columns:
        _, p_value = stats.ks_2samp(old_data[column], new_data[column])
        results[column] = p_value < 0.05
    
    return {drift: any(results.values()) for drift in drift_types}

# Example usage
old_data = pd.DataFrame(np.random.randn(1000, 4), columns=['A', 'B', 'C', 'D'])
new_data = pd.DataFrame(np.random.randn(1000, 4) + 0.5, columns=['A', 'B', 'C', 'D'])

print(detect_drift(old_data, new_data))
```

Slide 3: Detecting Drift

Detecting drift involves monitoring your model's performance and analyzing data distributions. Common methods include:

1. Tracking performance metrics like accuracy or precision over time.
2. Comparing new data distributions to the training set.
3. Using statistical tests to measure divergence between distributions.

```python
from scipy import stats

def detect_distribution_drift(reference, current, threshold=0.05):
    _, p_value = stats.ks_2samp(reference, current)
    return p_value < threshold

# Example: Detecting drift in a feature
reference_data = np.random.normal(0, 1, 1000)
current_data = np.random.normal(0.5, 1, 1000)  # Slight shift in mean

drift_detected = detect_distribution_drift(reference_data, current_data)
print(f"Drift detected: {drift_detected}")
```

Slide 4: Performance Metrics for Drift Detection

Monitoring performance metrics is crucial for early drift detection. Common metrics include accuracy, precision, recall, and F1-score. Plotting these metrics over time can reveal trends and sudden changes indicative of drift.

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import matplotlib.pyplot as plt

def plot_performance_metrics(y_true, y_pred_over_time):
    metrics = {
        'Accuracy': accuracy_score,
        'Precision': precision_score,
        'Recall': recall_score,
        'F1-score': f1_score
    }
    
    plt.figure(figsize=(12, 8))
    for metric_name, metric_func in metrics.items():
        scores = [metric_func(y_true, y_pred) for y_pred in y_pred_over_time]
        plt.plot(scores, label=metric_name)
    
    plt.title('Performance Metrics Over Time')
    plt.xlabel('Time')
    plt.ylabel('Score')
    plt.legend()
    plt.show()

# Example usage
np.random.seed(42)
y_true = np.random.randint(2, size=100)
y_pred_over_time = [np.random.randint(2, size=100) for _ in range(50)]

plot_performance_metrics(y_true, y_pred_over_time)
```

Slide 5: Statistical Tests for Drift Detection

Statistical tests can provide a more rigorous approach to detecting drift. The Kolmogorov-Smirnov test is commonly used to compare distributions and identify significant changes.

```python
from scipy import stats
import numpy as np

def ks_test_drift(reference_data, new_data, alpha=0.05):
    statistic, p_value = stats.ks_2samp(reference_data, new_data)
    drift_detected = p_value < alpha
    return drift_detected, p_value

# Example usage
np.random.seed(42)
reference_data = np.random.normal(0, 1, 1000)
new_data_no_drift = np.random.normal(0, 1, 1000)
new_data_with_drift = np.random.normal(0.5, 1, 1000)

print("No drift case:")
print(ks_test_drift(reference_data, new_data_no_drift))

print("\nDrift case:")
print(ks_test_drift(reference_data, new_data_with_drift))
```

Slide 6: Visualizing Data Drift

Visualizing data distributions can help identify drift. Techniques like histograms and kernel density estimation (KDE) plots allow for easy comparison between reference and new data.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def plot_distribution_comparison(reference_data, new_data, feature_name):
    plt.figure(figsize=(10, 6))
    
    kde_ref = stats.gaussian_kde(reference_data)
    kde_new = stats.gaussian_kde(new_data)
    
    x_range = np.linspace(min(reference_data.min(), new_data.min()),
                          max(reference_data.max(), new_data.max()), 100)
    
    plt.plot(x_range, kde_ref(x_range), label='Reference Data')
    plt.plot(x_range, kde_new(x_range), label='New Data')
    
    plt.title(f'Distribution Comparison: {feature_name}')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.legend()
    plt.show()

# Example usage
np.random.seed(42)
reference_data = np.random.normal(0, 1, 1000)
new_data = np.random.normal(0.5, 1.2, 1000)

plot_distribution_comparison(reference_data, new_data, 'Example Feature')
```

Slide 7: Handling Drift - Retraining

One common approach to handling drift is retraining the model periodically with new data. This helps the model adapt to changing patterns in the data.

```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def retrain_model(X, y, test_size=0.2):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return model, accuracy

# Example usage
np.random.seed(42)
X = np.random.rand(1000, 5)
y = (X[:, 0] + X[:, 1] > 1).astype(int)

initial_model, initial_accuracy = retrain_model(X, y)
print(f"Initial model accuracy: {initial_accuracy:.4f}")

# Simulate drift by changing the data distribution
X_new = np.random.rand(1000, 5) * 1.5
y_new = (X_new[:, 0] + X_new[:, 1] > 1.5).astype(int)

retrained_model, retrained_accuracy = retrain_model(X_new, y_new)
print(f"Retrained model accuracy: {retrained_accuracy:.4f}")
```

Slide 8: Handling Drift - Adaptive Learning

Adaptive learning involves continuously updating the model as new data becomes available. This approach allows the model to adapt to gradual changes in data distribution.

```python
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.linear_model import SGDClassifier

class AdaptiveClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, alpha=0.01):
        self.alpha = alpha
        self.model = SGDClassifier(alpha=alpha, loss='log', random_state=42)
    
    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)
        self.model.fit(X, y)
        return self
    
    def partial_fit(self, X, y):
        if not hasattr(self, 'classes_'):
            self.classes_ = unique_labels(y)
        self.model.partial_fit(X, y, classes=self.classes_)
        return self
    
    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)
        return self.model.predict(X)

# Example usage
np.random.seed(42)
X = np.random.rand(1000, 5)
y = (X[:, 0] + X[:, 1] > 1).astype(int)

model = AdaptiveClassifier()
model.fit(X[:800], y[:800])

# Simulate streaming data
for i in range(800, 1000, 10):
    model.partial_fit(X[i:i+10], y[i:i+10])

print("Final model accuracy:", accuracy_score(y[800:], model.predict(X[800:])))
```

Slide 9: Handling Drift - Ensemble Methods

Ensemble methods can be effective in handling drift by combining multiple models. This approach can help maintain performance even when some models become less accurate due to drift.

```python
from sklearn.ensemble import VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

def create_ensemble():
    models = [
        ('dt', DecisionTreeClassifier(random_state=42)),
        ('nb', GaussianNB()),
        ('lr', LogisticRegression(random_state=42))
    ]
    return VotingClassifier(estimators=models, voting='soft')

# Example usage
np.random.seed(42)
X = np.random.rand(1000, 5)
y = (X[:, 0] + X[:, 1] > 1).astype(int)

X_train, X_test = X[:800], X[800:]
y_train, y_test = y[:800], y[800:]

ensemble = create_ensemble()
ensemble.fit(X_train, y_train)

print("Ensemble accuracy:", accuracy_score(y_test, ensemble.predict(X_test)))

# Compare with individual models
for name, model in ensemble.named_estimators_.items():
    model.fit(X_train, y_train)
    print(f"{name} accuracy:", accuracy_score(y_test, model.predict(X_test)))
```

Slide 10: Real-Life Example - Weather Prediction

Consider a weather prediction model that forecasts temperature. Over time, climate change may cause gradual shifts in temperature patterns, leading to concept drift.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Simulate temperature data with gradual warming trend
np.random.seed(42)
years = np.arange(1900, 2024)
base_temp = 15 + np.random.normal(0, 1, len(years))
warming_trend = 0.01 * (years - 1900)
temperatures = base_temp + warming_trend

# Train a model on early data
model = LinearRegression()
train_years = years[:100]
train_temps = temperatures[:100]
model.fit(train_years.reshape(-1, 1), train_temps)

# Predict temperatures for all years
predicted_temps = model.predict(years.reshape(-1, 1))

plt.figure(figsize=(12, 6))
plt.scatter(years, temperatures, alpha=0.5, label='Actual Temperatures')
plt.plot(years, predicted_temps, color='red', label='Model Predictions')
plt.title('Temperature Predictions with Concept Drift')
plt.xlabel('Year')
plt.ylabel('Temperature (°C)')
plt.legend()
plt.show()

# Calculate error over time
error = np.abs(temperatures - predicted_temps)
plt.figure(figsize=(12, 6))
plt.plot(years, error)
plt.title('Prediction Error Over Time')
plt.xlabel('Year')
plt.ylabel('Absolute Error (°C)')
plt.show()
```

Slide 11: Real-Life Example - Image Classification

In image classification, data drift can occur when new types of images are introduced. For example, a model trained on daytime images might struggle with nighttime images.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def generate_image_features(num_samples, night=False):
    features = np.random.rand(num_samples, 3)  # RGB values
    if night:
        features *= 0.5  # Darker images
    return features

def plot_images(day_images, night_images):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.imshow(day_images[:100].reshape(10, 10, 3))
    ax1.set_title('Day Images')
    ax2.imshow(night_images[:100].reshape(10, 10, 3))
    ax2.set_title('Night Images')
    plt.show()

# Generate data
np.random.seed(42)
X_day = generate_image_features(1000)
y_day = (X_day.sum(axis=1) > 1.5).astype(int)
X_night = generate_image_features(1000, night=True)
y_night = (X_night.sum(axis=1) > 0.75).astype(int)

# Train model on day images
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_day, y_day)

# Evaluate on day and night images
print("Accuracy on day images:", accuracy_score(y_day, model.predict(X_day)))
print("Accuracy on night images:", accuracy_score(y_night, model.predict(X_night)))

# Visualize the images
plot_images(X_day, X_night)
```

Slide 12: Monitoring and Alerting

Implementing a robust monitoring and alerting system is crucial for detecting drift early. This system should track key performance metrics and data statistics, triggering alerts when significant changes are detected.

```python
import numpy as np
from scipy import stats

class DriftMonitor:
    def __init__(self, reference_data, threshold=0.05):
        self.reference_data = reference_data
        self.threshold = threshold
    
    def check_drift(self, new_data):
        _, p_value = stats.ks_2samp(self.reference_data, new_data)
        return p_value < self.threshold

# Example usage
np.random.seed(42)
reference_data = np.random.normal(0, 1, 1000)
monitor = DriftMonitor(reference_data)

# Simulate new data arrivals
for _ in range(5):
    new_data = np.random.normal(np.random.uniform(-0.5, 0.5), 1, 1000)
    if monitor.check_drift(new_data):
        print("Alert: Drift detected!")
    else:
        print("No significant drift detected.")
```

Slide 13: Continuous Learning and Adaptation

To stay ahead of drift, implement a continuous learning pipeline. This involves regularly updating your model with new data, validating its performance, and deploying improved versions.

```python
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import SGDClassifier

class ContinuousLearningClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, alpha=0.01):
        self.model = SGDClassifier(alpha=alpha, loss='log', random_state=42)
        self.performance_history = []
    
    def partial_fit(self, X, y, classes=None):
        self.model.partial_fit(X, y, classes=classes)
        self.performance_history.append(self.model.score(X, y))
        return self
    
    def predict(self, X):
        return self.model.predict(X)

# Example usage
clf = ContinuousLearningClassifier()
for _ in range(10):
    X_batch = np.random.rand(100, 5)
    y_batch = (X_batch.sum(axis=1) > 2.5).astype(int)
    clf.partial_fit(X_batch, y_batch, classes=[0, 1])

print("Performance history:", clf.performance_history)
```

Slide 14: Drift Prevention Strategies

Preventing drift involves proactive measures such as regular data quality checks, feature engineering to capture relevant trends, and maintaining a diverse and representative dataset.

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_and_check_data(data, expected_columns):
    # Check for missing columns
    missing_columns = set(expected_columns) - set(data.columns)
    if missing_columns:
        raise ValueError(f"Missing columns: {missing_columns}")
    
    # Check for unexpected data types
    for column in expected_columns:
        if data[column].dtype != 'float64':
            print(f"Warning: Column {column} is not float64")
    
    # Normalize numerical features
    scaler = StandardScaler()
    data[expected_columns] = scaler.fit_transform(data[expected_columns])
    
    return data

# Example usage
expected_columns = ['feature1', 'feature2', 'feature3']
data = pd.DataFrame({
    'feature1': [1.0, 2.0, 3.0],
    'feature2': [0.1, 0.2, 0.3],
    'feature3': [10, 20, 30]  # Integer instead of float
})

processed_data = preprocess_and_check_data(data, expected_columns)
print(processed_data)
```

Slide 15: Additional Resources

For those interested in diving deeper into model drift and performance maintenance, consider exploring these resources:

1. "Concept Drift Adaptation Techniques in Distributed Environments" (arXiv:1801.04977)
2. "A Survey on Concept Drift Adaptation" (arXiv:1308.2397)
3. "Learning under Concept Drift: A Review" (arXiv:2004.05785)

These papers provide comprehensive overviews of drift detection and adaptation techniques in machine learning.

