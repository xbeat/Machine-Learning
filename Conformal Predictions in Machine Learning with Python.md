## Conformal Predictions in Machine Learning with Python
Slide 1: Introduction to Conformal Predictions in Machine Learning

Conformal prediction is a framework for making reliable predictions with a guaranteed error rate. It provides a way to quantify uncertainty in machine learning models, allowing us to make predictions with a desired level of confidence. This technique is particularly useful in high-stakes applications where understanding the reliability of predictions is crucial.

```python
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from nonconformist.cp import IcpRegressor
from nonconformist.base import RegressorAdapter
from nonconformist.nc import AbsErrorErrFunc

# Sample data
X = np.random.rand(1000, 5)
y = np.sum(X, axis=1) + np.random.normal(0, 0.1, 1000)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and fit the underlying model
underlying_model = RandomForestRegressor(n_estimators=100, random_state=42)
underlying_model.fit(X_train, y_train)

# Create the conformal predictor
model = RegressorAdapter(underlying_model)
error_func = AbsErrorErrFunc()
icp = IcpRegressor(model, error_func)

# Fit the conformal predictor
icp.fit(X_train, y_train)

# Make predictions with confidence intervals
predictions = icp.predict(X_test, significance=0.1)

print(f"Prediction intervals for the first 5 samples:")
for i in range(5):
    print(f"Sample {i+1}: [{predictions[i, 0]:.2f}, {predictions[i, 1]:.2f}]")
```

Slide 2: The Core Principle of Conformal Prediction

Conformal prediction is based on the idea of exchangeability. It assumes that the order of data points doesn't matter, allowing us to create prediction regions that are valid for future observations. This approach provides a distribution-free way to estimate prediction intervals, making it applicable to a wide range of machine learning models.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.utils import resample

# Generate sample data
np.random.seed(42)
X = np.linspace(0, 10, 100).reshape(-1, 1)
y = 2 * X + 1 + np.random.normal(0, 1, (100, 1))

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Function to calculate nonconformity scores
def nonconformity_score(model, X, y):
    return np.abs(y - model.predict(X))

# Calculate nonconformity scores for calibration set
cal_scores = nonconformity_score(model, X_train, y_train)

# Function to get prediction interval
def get_prediction_interval(model, X_new, cal_scores, significance):
    y_pred = model.predict(X_new)
    n = len(cal_scores)
    q = np.ceil((n + 1) * (1 - significance)) / n
    threshold = np.quantile(cal_scores, q)
    return y_pred - threshold, y_pred + threshold

# Plot results
plt.figure(figsize=(10, 6))
plt.scatter(X_train, y_train, color='blue', label='Training data')
plt.scatter(X_test, y_test, color='red', label='Test data')
plt.plot(X, model.predict(X), color='green', label='Linear regression')

X_new = np.linspace(0, 10, 100).reshape(-1, 1)
lower, upper = get_prediction_interval(model, X_new, cal_scores, significance=0.1)
plt.fill_between(X_new.ravel(), lower.ravel(), upper.ravel(), alpha=0.2, color='gray', label='90% prediction interval')

plt.legend()
plt.title('Conformal Prediction: Linear Regression with Prediction Interval')
plt.xlabel('X')
plt.ylabel('y')
plt.show()
```

Slide 3: Types of Conformal Predictors

There are two main types of conformal predictors: transductive and inductive. Transductive conformal prediction recalculates nonconformity scores for each new prediction, while inductive conformal prediction uses a separate calibration set. Inductive conformal prediction is more computationally efficient for large datasets.

```python
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from nonconformist.cp import IcpRegressor, TcpRegressor
from nonconformist.base import RegressorAdapter
from nonconformist.nc import AbsErrorErrFunc
import time

# Sample data
X = np.random.rand(1000, 5)
y = np.sum(X, axis=1) + np.random.normal(0, 0.1, 1000)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and fit the underlying model
underlying_model = RandomForestRegressor(n_estimators=100, random_state=42)
underlying_model.fit(X_train, y_train)

# Create the conformal predictors
model = RegressorAdapter(underlying_model)
error_func = AbsErrorErrFunc()
icp = IcpRegressor(model, error_func)
tcp = TcpRegressor(model, error_func)

# Fit the conformal predictors
icp.fit(X_train, y_train)
tcp.fit(X_train, y_train)

# Make predictions and measure time
start_time = time.time()
icp_predictions = icp.predict(X_test, significance=0.1)
icp_time = time.time() - start_time

start_time = time.time()
tcp_predictions = tcp.predict(X_test, significance=0.1)
tcp_time = time.time() - start_time

print(f"Inductive CP time: {icp_time:.4f} seconds")
print(f"Transductive CP time: {tcp_time:.4f} seconds")

print("\nPrediction intervals for the first 3 samples:")
for i in range(3):
    print(f"Sample {i+1}:")
    print(f"  ICP: [{icp_predictions[i, 0]:.2f}, {icp_predictions[i, 1]:.2f}]")
    print(f"  TCP: [{tcp_predictions[i, 0]:.2f}, {tcp_predictions[i, 1]:.2f}]")
```

Slide 4: Nonconformity Measures

Nonconformity measures quantify how different a new example is from the training data. The choice of nonconformity measure affects the efficiency of the conformal predictor. Common measures include absolute error for regression and probability-based measures for classification.

```python
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from nonconformist.cp import IcpRegressor
from nonconformist.base import RegressorAdapter
from nonconformist.nc import AbsErrorErrFunc, SignErrorErrFunc, NormalizedAbsErrorErrFunc

# Sample data
X = np.random.rand(1000, 5)
y = np.sum(X, axis=1) + np.random.normal(0, 0.1, 1000)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and fit the underlying model
underlying_model = RandomForestRegressor(n_estimators=100, random_state=42)
underlying_model.fit(X_train, y_train)

# Create conformal predictors with different nonconformity measures
model = RegressorAdapter(underlying_model)
icp_abs = IcpRegressor(model, AbsErrorErrFunc())
icp_sign = IcpRegressor(model, SignErrorErrFunc())
icp_norm = IcpRegressor(model, NormalizedAbsErrorErrFunc())

# Fit the conformal predictors
icp_abs.fit(X_train, y_train)
icp_sign.fit(X_train, y_train)
icp_norm.fit(X_train, y_train)

# Make predictions
predictions_abs = icp_abs.predict(X_test, significance=0.1)
predictions_sign = icp_sign.predict(X_test, significance=0.1)
predictions_norm = icp_norm.predict(X_test, significance=0.1)

# Calculate average interval widths
width_abs = np.mean(predictions_abs[:, 1] - predictions_abs[:, 0])
width_sign = np.mean(predictions_sign[:, 1] - predictions_sign[:, 0])
width_norm = np.mean(predictions_norm[:, 1] - predictions_norm[:, 0])

print("Average prediction interval widths:")
print(f"Absolute Error: {width_abs:.4f}")
print(f"Sign Error: {width_sign:.4f}")
print(f"Normalized Absolute Error: {width_norm:.4f}")

# Print example predictions
print("\nPrediction intervals for the first 3 samples:")
for i in range(3):
    print(f"Sample {i+1}:")
    print(f"  Absolute Error: [{predictions_abs[i, 0]:.2f}, {predictions_abs[i, 1]:.2f}]")
    print(f"  Sign Error: [{predictions_sign[i, 0]:.2f}, {predictions_sign[i, 1]:.2f}]")
    print(f"  Normalized Absolute Error: [{predictions_norm[i, 0]:.2f}, {predictions_norm[i, 1]:.2f}]")
```

Slide 5: Conformal Prediction for Classification

Conformal prediction can be applied to classification tasks, providing prediction sets that contain the true class with a specified confidence level. This is particularly useful in multi-class problems where understanding the uncertainty of predictions is crucial.

```python
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from nonconformist.cp import IcpClassifier
from nonconformist.nc import ClassifierNc, MarginErrFunc

# Load the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and fit the underlying model
underlying_model = RandomForestClassifier(n_estimators=100, random_state=42)
underlying_model.fit(X_train, y_train)

# Create the conformal classifier
nc = ClassifierNc(underlying_model, MarginErrFunc())
icp = IcpClassifier(nc)

# Fit the conformal classifier
icp.fit(X_train, y_train)

# Make predictions
predictions = icp.predict(X_test, significance=0.1)

# Calculate prediction set sizes
set_sizes = np.sum(predictions, axis=1)

print(f"Average prediction set size: {np.mean(set_sizes):.2f}")
print("\nPrediction sets for the first 5 samples:")
for i in range(5):
    classes = np.where(predictions[i])[0]
    print(f"Sample {i+1}: {classes}")

# Calculate empirical coverage
true_labels_in_set = np.sum([y_test[i] in np.where(predictions[i])[0] for i in range(len(y_test))])
empirical_coverage = true_labels_in_set / len(y_test)
print(f"\nEmpirical coverage: {empirical_coverage:.2f}")
```

Slide 6: Handling Imbalanced Datasets

Conformal prediction can be particularly useful for imbalanced datasets, where traditional methods might struggle. By providing prediction sets rather than point predictions, conformal prediction can help mitigate issues related to class imbalance.

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from nonconformist.cp import IcpClassifier
from nonconformist.nc import ClassifierNc, MarginErrFunc
from imblearn.over_sampling import SMOTE

# Create an imbalanced dataset
X, y = make_classification(n_samples=1000, n_classes=2, weights=[0.9, 0.1], 
                           n_features=20, n_informative=2, n_redundant=2, 
                           n_repeated=0, n_clusters_per_class=1, random_state=42)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply SMOTE to balance the training data
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# Create and fit the underlying models
model_imbalanced = RandomForestClassifier(n_estimators=100, random_state=42)
model_balanced = RandomForestClassifier(n_estimators=100, random_state=42)

model_imbalanced.fit(X_train, y_train)
model_balanced.fit(X_train_balanced, y_train_balanced)

# Create the conformal classifiers
icp_imbalanced = IcpClassifier(ClassifierNc(model_imbalanced, MarginErrFunc()))
icp_balanced = IcpClassifier(ClassifierNc(model_balanced, MarginErrFunc()))

# Fit the conformal classifiers
icp_imbalanced.fit(X_train, y_train)
icp_balanced.fit(X_train_balanced, y_train_balanced)

# Make predictions
predictions_imbalanced = icp_imbalanced.predict(X_test, significance=0.1)
predictions_balanced = icp_balanced.predict(X_test, significance=0.1)

# Calculate prediction set sizes
set_sizes_imbalanced = np.sum(predictions_imbalanced, axis=1)
set_sizes_balanced = np.sum(predictions_balanced, axis=1)

print(f"Average prediction set size (imbalanced): {np.mean(set_sizes_imbalanced):.2f}")
print(f"Average prediction set size (balanced): {np.mean(set_sizes_balanced):.2f}")

# Calculate empirical coverage
true_labels_in_set_imbalanced = np.sum([y_test[i] in np.where(predictions_imbalanced[i])[0] for i in range(len(y_test))])
true_labels_in_set_balanced = np.sum([y_test[i] in np.where(predictions_balanced[i])[0] for i in range(len(y_test))])

empirical_coverage_imbalanced = true_labels_in_set_imbalanced / len(y_test)
empirical_coverage_balanced = true_labels_in_set_balanced / len(y_test)

print(f"\nEmpirical coverage (imbalanced): {empirical_coverage_imbalanced:.2f}")
print(f"Empirical coverage (balanced): {empirical_coverage_balanced:.2f}")
```

Slide 7: Conformal Prediction for Time Series Forecasting

Conformal prediction can be adapted for time series forecasting, providing prediction intervals that account for the temporal dependencies in the data. This approach allows us to quantify uncertainty in time series predictions, which is crucial in many applications such as demand forecasting or financial modeling.

```python
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.model_selection import train_test_split

# Generate sample time series data
np.random.seed(42)
dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
y = np.cumsum(np.random.randn(len(dates))) + 20
ts = pd.Series(y, index=dates)

# Split the data
train_size = int(len(ts) * 0.8)
train, test = ts[:train_size], ts[train_size:]

# Fit ARIMA model
model = ARIMA(train, order=(1,1,1))
fitted_model = model.fit()

# Function to calculate nonconformity scores
def nonconformity_score(y_true, y_pred):
    return np.abs(y_true - y_pred)

# Generate predictions and nonconformity scores for calibration
y_pred_train = fitted_model.forecast(steps=len(train))
cal_scores = nonconformity_score(train, y_pred_train)

# Function to get prediction interval using conformal prediction
def get_conformal_interval(model, X_new, cal_scores, significance):
    y_pred = model.forecast(steps=len(X_new))
    n = len(cal_scores)
    q = np.ceil((n + 1) * (1 - significance)) / n
    threshold = np.quantile(cal_scores, q)
    lower = y_pred - threshold
    upper = y_pred + threshold
    return lower, upper

# Get conformal prediction intervals
lower, upper = get_conformal_interval(fitted_model, test, cal_scores, significance=0.1)

# Print results
print("Conformal Prediction Intervals for the first 5 test points:")
for i in range(5):
    print(f"Date: {test.index[i]}, Actual: {test[i]:.2f}, Interval: [{lower[i]:.2f}, {upper[i]:.2f}]")

# Calculate coverage
coverage = np.mean((test >= lower) & (test <= upper))
print(f"\nEmpirical coverage: {coverage:.2f}")
```

Slide 8: Conformal Prediction for Anomaly Detection

Conformal prediction can be applied to anomaly detection tasks, providing a principled way to identify outliers while controlling the false positive rate. This approach is particularly useful in scenarios where the cost of false alarms is high.

```python
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split

# Generate sample data with outliers
np.random.seed(42)
X_normal = np.random.randn(1000, 2)
X_outliers = np.random.uniform(low=-4, high=4, size=(50, 2))
X = np.vstack([X_normal, X_outliers])

# Split the data
X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)

# Fit Isolation Forest
iso_forest = IsolationForest(contamination=0.1, random_state=42)
iso_forest.fit(X_train)

# Function to calculate nonconformity scores
def nonconformity_score(model, X):
    return -model.score_samples(X)

# Calculate nonconformity scores for calibration set
cal_scores = nonconformity_score(iso_forest, X_train)

# Function to get conformal prediction threshold
def get_conformal_threshold(cal_scores, significance):
    n = len(cal_scores)
    q = np.ceil((n + 1) * (1 - significance)) / n
    return np.quantile(cal_scores, q)

# Get conformal prediction threshold
threshold = get_conformal_threshold(cal_scores, significance=0.1)

# Predict anomalies on test set
test_scores = nonconformity_score(iso_forest, X_test)
predictions = test_scores > threshold

# Calculate false positive rate
fpr = np.mean(predictions)

print(f"Conformal Anomaly Detection Results:")
print(f"Threshold: {threshold:.4f}")
print(f"False Positive Rate: {fpr:.4f}")
print("\nPredictions for the first 10 test samples:")
for i in range(10):
    print(f"Sample {i+1}: {'Anomaly' if predictions[i] else 'Normal'} (Score: {test_scores[i]:.4f})")
```

Slide 9: Conformal Prediction for Multi-label Classification

Conformal prediction can be extended to multi-label classification problems, where each instance can belong to multiple classes simultaneously. This approach provides prediction sets that contain the true set of labels with a specified confidence level.

```python
import numpy as np
from sklearn.datasets import make_multilabel_classification
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import hamming_loss

# Generate multi-label dataset
X, y = make_multilabel_classification(n_samples=1000, n_features=20, n_classes=5, n_labels=3, random_state=42)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit multi-label classifier
base_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
multi_classifier = MultiOutputClassifier(base_classifier)
multi_classifier.fit(X_train, y_train)

# Function to calculate nonconformity scores
def nonconformity_score(y_true, y_pred_proba):
    return 1 - y_pred_proba[np.arange(len(y_true)), y_true]

# Calculate nonconformity scores for calibration set
y_pred_proba_train = multi_classifier.predict_proba(X_train)
cal_scores = np.max([nonconformity_score(y_train[:, i], y_pred_proba_train[i]) for i in range(y_train.shape[1])], axis=0)

# Function to get conformal prediction sets
def get_conformal_prediction_sets(y_pred_proba, cal_scores, significance):
    n = len(cal_scores)
    q = np.ceil((n + 1) * (1 - significance)) / n
    threshold = np.quantile(cal_scores, q)
    return [y_pred_proba[i] >= (1 - threshold) for i in range(len(y_pred_proba))]

# Make predictions on test set
y_pred_proba_test = multi_classifier.predict_proba(X_test)
prediction_sets = get_conformal_prediction_sets(y_pred_proba_test, cal_scores, significance=0.1)

# Calculate coverage and average set size
coverage = np.mean([np.all(y_test[i][prediction_sets[i]]) for i in range(len(y_test))])
avg_set_size = np.mean([np.sum(pred_set) for pred_set in prediction_sets])

print(f"Conformal Multi-label Classification Results:")
print(f"Empirical coverage: {coverage:.4f}")
print(f"Average prediction set size: {avg_set_size:.2f}")
print("\nPrediction sets for the first 5 test samples:")
for i in range(5):
    print(f"Sample {i+1}: {np.where(prediction_sets[i])[0]}")
```

Slide 10: Real-life Example: Medical Diagnosis

Conformal prediction can be applied in medical diagnosis to provide reliable predictions with controlled error rates. This is crucial in healthcare where understanding the uncertainty of predictions can directly impact patient care decisions.

```python
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load breast cancer dataset
data = load_breast_cancer()
X, y = data.data, data.target

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit Random Forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Function to calculate nonconformity scores
def nonconformity_score(y_true, y_pred_proba):
    return 1 - y_pred_proba[np.arange(len(y_true)), y_true]

# Calculate nonconformity scores for calibration set
y_pred_proba_train = clf.predict_proba(X_train)
cal_scores = nonconformity_score(y_train, y_pred_proba_train)

# Function to get conformal prediction sets
def get_conformal_prediction_sets(y_pred_proba, cal_scores, significance):
    n = len(cal_scores)
    q = np.ceil((n + 1) * (1 - significance)) / n
    threshold = np.quantile(cal_scores, q)
    return y_pred_proba >= (1 - threshold)

# Make predictions on test set
y_pred_proba_test = clf.predict_proba(X_test)
prediction_sets = get_conformal_prediction_sets(y_pred_proba_test, cal_scores, significance=0.1)

# Calculate coverage and average set size
coverage = np.mean([y_test[i] in np.where(prediction_sets[i])[0] for i in range(len(y_test))])
avg_set_size = np.mean(np.sum(prediction_sets, axis=1))

print("Conformal Prediction for Breast Cancer Diagnosis:")
print(f"Empirical coverage: {coverage:.4f}")
print(f"Average prediction set size: {avg_set_size:.2f}")

# Compare with standard classifier accuracy
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Standard classifier accuracy: {accuracy:.4f}")

print("\nPrediction sets for the first 5 test samples:")
for i in range(5):
    classes = np.where(prediction_sets[i])[0]
    class_names = [data.target_names[c] for c in classes]
    print(f"Sample {i+1}: {class_names}")
```

Slide 11: Real-life Example: Image Classification

Conformal prediction can enhance image classification tasks by providing prediction sets that contain the true class with a specified confidence level. This is particularly useful in applications where misclassification can have significant consequences.

```python
import numpy as np
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image

# Load pre-trained ResNet50 model
model = ResNet50(weights='imagenet')

# Function to preprocess image
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

# Function to calculate nonconformity scores
def nonconformity_score(y_true, y_pred_proba):
    return 1 - y_pred_proba[np.arange(len(y_true)), y_true]

# Function to get conformal prediction sets
def get_conformal_prediction_sets(y_pred_proba, cal_scores, significance):
    n = len(cal_scores)
    q = np.ceil((n + 1) * (1 - significance)) / n
    threshold = np.quantile(cal_scores, q)
    return y_pred_proba >= (1 - threshold)

# Simulate calibration scores (in practice, these would be calculated on a calibration set)
np.random.seed(42)
cal_scores = np.random.uniform(0, 1, 1000)

# Process a sample image
img_path = 'path/to/your/image.jpg'  # Replace with actual image path
x = preprocess_image(img_path)

# Make prediction
preds = model.predict(x)
y_pred_proba = preds[0]

# Get conformal prediction set
prediction_set = get_conformal_prediction_sets(y_pred_proba, cal_scores, significance=0.1)

# Decode predictions
decoded_preds = decode_predictions(preds, top=5)[0]

print("Conformal Prediction for Image Classification:")
print("Prediction set (top 5 classes in the set):")
for i, (imagenet_id, label, score) in enumerate(decoded_preds):
    if prediction_set[0][i]:
        print(f"{label}: {score:.4f}")

print("\nNote: In practice, you would need a proper calibration set to calculate nonconformity scores.")
```

Slide 12: Challenges and Limitations of Conformal Prediction

While conformal prediction offers many advantages, it also has some challenges and limitations:

1. Computational cost: For transductive conformal prediction, recalculating nonconformity scores for each new prediction can be computationally expensive.
2. Choice of nonconformity measure: The efficiency of conformal prediction depends on the choice of nonconformity measure, which can be challenging to optimize.
3. Handling of dependent data: Standard conformal prediction assumes exchangeability, which may not hold for time series or other dependent data.
4. Interpretability: While conformal prediction provides valid prediction intervals, interpreting these intervals in complex models can still be challenging.

```python
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Generate sample data
np.random.seed(42)
X = np.random.rand(1000, 10)
y = np.sum(X, axis=1) + np.random.normal(0, 0.1, 1000)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit Random Forest model
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Standard prediction
start_time = time.time()
y_pred = rf.predict(X_test)
standard_time = time.time() - start_time
mae = mean_absolute_error(y_test, y_pred)

# Simulate conformal prediction (simplified)
start_time = time.time()
n_samples = 1000
y_pred_conformal = np.array([rf.predict(X_test) + np.random.normal(0, mae, len(X_test)) for _ in range(n_samples)])
lower = np.percentile(y_pred_conformal, 5, axis=0)
upper = np.percentile(y_pred_conformal, 95, axis=0)
conformal_time = time.time() - start_time

print("Comparison of Standard Prediction vs Conformal Prediction:")
print(f"Standard prediction time: {standard_time:.4f} seconds")
print(f"Conformal prediction time: {conformal_time:.4f} seconds")
print(f"Time increase: {conformal_time/standard_time:.2f}x")

print("\nPrediction intervals for the first 5 samples:")
for i in range(5):
    print(f"Sample {i+1}: [{lower[i]:.2f}, {upper[i]:.2f}]")

# Demonstrate challenge with dependent data
X_time = np.arange(1000).reshape(-1, 1)
y_time = np.sin(X_time * 0.1) + np.random.normal(0, 0.1, (1000, 1))

# Split time series data
train_size = int(0.8 * len(X_time))
X_train, X_test = X_time[:train_size], X_time[train_size:]
y_train, y_test = y_time[:train_size], y_time[train_size:]

# Fit model and make predictions
rf_time = RandomForestRegressor(n_estimators=100, random_state=42)
rf_time.fit(X_train, y_train)
y_pred_time = rf_time.predict(X_test)

print("\nMAE for independent data:", mean_absolute_error(y_test, y_pred))
print("MAE for time series data:", mean_absolute_error(y_test, y_pred_time))
print("Note: Time series predictions may be less accurate due to temporal dependencies.")
```

Slide 13: Future Directions and Ongoing Research

Conformal prediction is an active area of research with several promising directions:

1. Adaptive conformal inference: Developing methods that adapt to non-stationary data distributions.
2. Computationally efficient algorithms: Exploring ways to reduce the computational cost of conformal prediction, especially for large datasets.
3. Causal conformal prediction: Integrating causal inference with conformal prediction to handle interventions and counterfactuals.
4. Deep learning integration: Developing more efficient ways to apply conformal prediction to deep learning models.
5. Conformal prediction for structured outputs: Extending conformal prediction to more complex output spaces, such as graphs or sequences.

```python
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Generate non-stationary data
np.random.seed(42)
X = np.random.rand(1000, 5)
y = np.sum(X, axis=1) + np.random.normal(0, 0.1, 1000)
y[:500] += 2  # Add a shift to the first half of the data

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit a model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Function for adaptive conformal prediction
def adaptive_conformal_prediction(model, X_train, y_train, X_test, alpha=0.1, window_size=100):
    y_pred = model.predict(X_test)
    errors = []
    lower, upper = [], []
    
    for i in range(len(X_test)):
        if i < window_size:
            cal_errors = np.abs(y_train[-window_size:] - model.predict(X_train[-window_size:]))
        else:
            cal_errors = np.abs(y_test[i-window_size:i] - y_pred[i-window_size:i])
        
        q = np.quantile(cal_errors, 1 - alpha)
        lower.append(y_pred[i] - q)
        upper.append(y_pred[i] + q)
        
        if i < len(y_test):
            errors.append(np.abs(y_test[i] - y_pred[i]))
    
    return np.array(lower), np.array(upper), np.array(errors)

# Apply adaptive conformal prediction
lower, upper, errors = adaptive_conformal_prediction(model, X_train, y_train, X_test)

# Calculate coverage
coverage = np.mean((y_test >= lower) & (y_test <= upper))

print(f"Adaptive Conformal Prediction Results:")
print(f"Empirical coverage: {coverage:.4f}")
print("\nPrediction intervals for the first 5 test samples:")
for i in range(5):
    print(f"Sample {i+1}: [{lower[i]:.2f}, {upper[i]:.2f}], Actual: {y_test[i]:.2f}")

print("\nNote: This is a simplified example of adaptive conformal prediction.")
```

Slide 14: Additional Resources

For those interested in delving deeper into conformal prediction, here are some valuable resources:

1. "Conformal Prediction for Reliable Machine Learning" by V. Vovk, A. Gammerman, and G. Shafer (2005) ArXiv: [https://arxiv.org/abs/0706.3188](https://arxiv.org/abs/0706.3188)
2. "Distribution-Free Predictive Inference for Regression" by J. Lei, M. G'Sell, A. Rinaldo, R. J. Tibshirani, and L. Wasserman (2018) ArXiv: [https://arxiv.org/abs/1604.04173](https://arxiv.org/abs/1604.04173)
3. "A Tutorial on Conformal Prediction" by G. Shafer and V. Vovk (2008) ArXiv: [https://arxiv.org/abs/0706.3188](https://arxiv.org/abs/0706.3188)
4. "Conformal Prediction Under Covariate Shift" by V. Vovk (2013) ArXiv: [https://arxiv.org/abs/1208.4441](https://arxiv.org/abs/1208.4441)
5. "Conformal Prediction: a Unified Review of Theory and New Challenges" by A. Angelopoulos and S. Bates (2021) ArXiv: [https://arxiv.org/abs/2107.07511](https://arxiv.org/abs/2107.07511)

These resources provide a comprehensive overview of conformal prediction theory, applications, and recent developments in the field.

