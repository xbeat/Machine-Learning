## Conformal Prediction Measuring ML Model Confidence

Slide 1: Introduction to Conformal Prediction

Conformal prediction is a statistical method that provides a measure of confidence for individual predictions in machine learning models. Unlike traditional approaches that focus on overall model accuracy, conformal prediction aims to quantify uncertainty for each specific prediction. This technique is particularly valuable in high-stakes scenarios where understanding the reliability of individual predictions is crucial.

```python
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

# Load the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split the data into training and calibration sets
X_train, X_cal, y_train, y_cal = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest classifier
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# Function to calculate conformal scores
def conformal_score(model, X, y_true):
    probas = model.predict_proba(X)
    return 1 - probas[np.arange(len(y_true)), y_true]

# Calculate conformal scores for the calibration set
cal_scores = conformal_score(clf, X_cal, y_cal)

print(f"Calibration scores shape: {cal_scores.shape}")
print(f"Calibration scores range: [{cal_scores.min():.4f}, {cal_scores.max():.4f}]")
```

Slide 2: The Need for Confidence in Predictions

Traditional machine learning approaches often focus on aggregate performance metrics, such as accuracy or F1-score. However, these metrics don't provide insight into the confidence of individual predictions. In critical applications, such as medical diagnosis or autonomous driving, understanding the reliability of each prediction is crucial for making informed decisions.

```python
import matplotlib.pyplot as plt

# Generate some example predictions and true labels
np.random.seed(42)
predictions = np.random.rand(100)
true_labels = (np.random.rand(100) > 0.5).astype(int)

# Calculate accuracy
accuracy = np.mean(predictions.round() == true_labels)

plt.figure(figsize=(10, 6))
plt.scatter(range(100), predictions, c=true_labels, cmap='coolwarm')
plt.axhline(y=0.5, color='r', linestyle='--')
plt.title(f'Model Predictions vs True Labels (Accuracy: {accuracy:.2f})')
plt.xlabel('Sample Index')
plt.ylabel('Prediction Probability')
plt.colorbar(label='True Label')
plt.show()

print(f"Overall accuracy: {accuracy:.2f}")
print(f"But what about the confidence in each individual prediction?")
```

Slide 3: Conformal Prediction: A Solution for Confidence

Conformal prediction addresses the need for individual prediction confidence by providing a mathematically rigorous framework. It works with any machine learning model and offers a way to generate prediction sets or intervals with a guaranteed error rate. This approach shifts the focus from point predictions to sets of predictions with a specified confidence level.

```python
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression

# Generate a synthetic regression dataset
X, y = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest regressor
regressor = RandomForestRegressor(random_state=42)
regressor.fit(X_train, y_train)

# Make predictions
y_pred = regressor.predict(X_test)

# Calculate prediction intervals (simple example, not conformal yet)
y_std = np.std(y_train)
lower_bound = y_pred - 1.96 * y_std
upper_bound = y_pred + 1.96 * y_std

print(f"Prediction interval coverage: {np.mean((y_test >= lower_bound) & (y_test <= upper_bound)):.2f}")
```

Slide 4: The Conformal Prediction Process

The conformal prediction process involves several key steps:

1.  Splitting the data into training and calibration sets
2.  Training a model on the training data
3.  Computing conformal scores for the calibration set
4.  Determining a threshold based on the desired confidence level
5.  Applying the threshold to new predictions to generate prediction sets or intervals

```python
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer

# Load the breast cancer dataset
data = load_breast_cancer()
X, y = data.data, data.target

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_cal, y_train, y_cal = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

# Train the model
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# Compute conformal scores
def conformal_score(model, X, y):
    probas = model.predict_proba(X)
    return 1 - probas[np.arange(len(y)), y]

cal_scores = conformal_score(clf, X_cal, y_cal)

# Set confidence level and find threshold
alpha = 0.1
threshold = np.percentile(cal_scores, (1 - alpha) * 100)

print(f"Conformal score threshold at {1-alpha:.0%} confidence: {threshold:.4f}")
```

Slide 5: Constructing Prediction Sets

For classification tasks, conformal prediction generates prediction sets rather than single-class predictions. These sets contain all classes that meet the confidence threshold. This approach provides a more nuanced view of the model's uncertainty for each prediction.

```python
# Function to generate prediction sets
def prediction_set(model, X, threshold):
    probas = model.predict_proba(X)
    return [set(np.where(1 - p <= threshold)[0]) for p in probas]

# Generate prediction sets for test data
test_sets = prediction_set(clf, X_test, threshold)

# Evaluate coverage and set sizes
correct_coverage = sum(y_test[i] in s for i, s in enumerate(test_sets)) / len(y_test)
avg_set_size = np.mean([len(s) for s in test_sets])

print(f"Empirical coverage: {correct_coverage:.2f}")
print(f"Average prediction set size: {avg_set_size:.2f}")

# Visualize some prediction sets
for i in range(5):
    print(f"Sample {i+1}: True class = {y_test[i]}, Prediction set = {test_sets[i]}")
```

Slide 6: Conformal Prediction for Regression

In regression tasks, conformal prediction produces prediction intervals rather than point estimates. These intervals provide a range of values within which the true value is expected to fall, with a specified confidence level.

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression

# Generate synthetic regression data
X, y = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_cal, y_train, y_cal = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

# Train the model
reg = RandomForestRegressor(random_state=42)
reg.fit(X_train, y_train)

# Compute conformal scores for regression
def conformal_score_reg(model, X, y):
    y_pred = model.predict(X)
    return np.abs(y - y_pred)

cal_scores = conformal_score_reg(reg, X_cal, y_cal)

# Set confidence level and find threshold
alpha = 0.1
threshold = np.percentile(cal_scores, (1 - alpha) * 100)

# Generate prediction intervals
y_pred = reg.predict(X_test)
lower = y_pred - threshold
upper = y_pred + threshold

# Evaluate coverage
coverage = np.mean((y_test >= lower) & (y_test <= upper))

print(f"Empirical coverage: {coverage:.2f}")
print(f"Average interval width: {np.mean(upper - lower):.2f}")
```

Slide 7: Advantages of Conformal Prediction

Conformal prediction offers several key advantages:

1.  Model-agnostic: It can be applied to any machine learning model.
2.  Theoretically grounded: It provides rigorous guarantees on the error rate.
3.  Flexibility: It can be used for both classification and regression tasks.
4.  Interpretability: The prediction sets or intervals are easy to understand and communicate.

Slide 8: Advantages of Conformal Prediction

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split

def apply_conformal_prediction(X, y, model, task='classification'):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_cal, y_train, y_cal = train_test_split(X_train, y_train, test_size=0.25, random_state=42)
    
    model.fit(X_train, y_train)
    
    if task == 'classification':
        scores = 1 - model.predict_proba(X_cal)[np.arange(len(y_cal)), y_cal]
    else:  # regression
        scores = np.abs(y_cal - model.predict(X_cal))
    
    threshold = np.percentile(scores, 90)  # 90% confidence level
    
    if task == 'classification':
        pred_sets = [set(np.where(1 - p <= threshold)[0]) for p in model.predict_proba(X_test)]
        coverage = np.mean([y_test[i] in s for i, s in enumerate(pred_sets)])
    else:  # regression
        y_pred = model.predict(X_test)
        lower, upper = y_pred - threshold, y_pred + threshold
        coverage = np.mean((y_test >= lower) & (y_test <= upper))
    
    return coverage

# Generate datasets
X_clf, y_clf = make_classification(n_samples=1000, n_features=20, random_state=42)
X_reg, y_reg = make_regression(n_samples=1000, n_features=20, random_state=42)

# Apply conformal prediction
clf_coverage = apply_conformal_prediction(X_clf, y_clf, RandomForestClassifier(random_state=42))
reg_coverage = apply_conformal_prediction(X_reg, y_reg, RandomForestRegressor(random_state=42), task='regression')

print(f"Classification coverage: {clf_coverage:.2f}")
print(f"Regression coverage: {reg_coverage:.2f}")
```

Slide 9: Conformal Scores: The Heart of the Method

Conformal scores measure how "unusual" a prediction is compared to the calibration data. For classification, a common score is 1 minus the predicted probability of the true class. For regression, it's often the absolute difference between the prediction and the true value. The choice of scoring function can significantly impact the performance of conformal prediction.

Slide 10: Conformal Scores: The Heart of the Method

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

# Generate a synthetic dataset
X, y = make_classification(n_samples=1000, n_features=20, n_classes=3, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a classifier
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# Compute conformal scores
def conformal_score(model, X, y):
    probas = model.predict_proba(X)
    return 1 - probas[np.arange(len(y)), y]

scores = conformal_score(clf, X_test, y_test)

# Visualize the distribution of scores
plt.figure(figsize=(10, 6))
plt.hist(scores, bins=30, edgecolor='black')
plt.title('Distribution of Conformal Scores')
plt.xlabel('Conformal Score')
plt.ylabel('Frequency')
plt.axvline(np.percentile(scores, 90), color='r', linestyle='--', label='90th percentile')
plt.legend()
plt.show()

print(f"Mean conformal score: {np.mean(scores):.4f}")
print(f"90th percentile score: {np.percentile(scores, 90):.4f}")
```

Slide 11: Calibration Set: Size and Selection

The calibration set plays a crucial role in conformal prediction. Its size impacts the reliability of conformal score estimates, while balancing the need for sufficient training data. Typically, 10-20% of the data is reserved for calibration. Cross-validation strategies can be employed to maximize data utilization and improve the robustness of the calibration process.

Slide 12: Calibration Set: Size and Selection

```python
import numpy as np
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

# Generate a synthetic dataset
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)

def evaluate_calibration_size(X, y, sizes):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    results = []
    
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        for size in sizes:
            n_cal = int(len(X_train) * size)
            X_train_, X_cal = X_train[:-n_cal], X_train[-n_cal:]
            y_train_, y_cal = y_train[:-n_cal], y_train[-n_cal:]
            
            clf = RandomForestClassifier(random_state=42)
            clf.fit(X_train_, y_train_)
            
            scores = 1 - clf.predict_proba(X_cal)[np.arange(len(y_cal)), y_cal]
            threshold = np.percentile(scores, 90)
            
            test_scores = 1 - clf.predict_proba(X_test)[np.arange(len(y_test)), y_test]
            coverage = np.mean(test_scores <= threshold)
            
            results.append((size, coverage))
    
    return results

# Evaluate different calibration set sizes
sizes = [0.1, 0.2, 0.3, 0.4]
results = evaluate_calibration_size(X, y, sizes)

# Print average coverage for each calibration set size
for size in sizes:
    coverages = [cov for s, cov in results if s == size]
    print(f"Size: {size*100:.0f}%, Avg Coverage: {np.mean(coverages):.4f}")
```

Slide 13: Handling Non-Conformity: Alternative Scoring Methods

While the default scoring method (1 - probability of true class) works well for many scenarios, alternative non-conformity measures can be more appropriate for certain tasks or model types. These alternatives can improve the efficiency and interpretability of conformal prediction sets.

Slide 14: Handling Non-Conformity: Alternative Scoring Methods

```python
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

# Generate synthetic data
X, y = make_classification(n_samples=1000, n_features=20, n_classes=3, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a classifier
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# Define different non-conformity scoring functions
def inverse_probability(probas, true_class):
    return 1 - probas[np.arange(len(true_class)), true_class]

def margin(probas, true_class):
    sorted_probas = np.sort(probas, axis=1)
    return sorted_probas[:, -1] - probas[np.arange(len(true_class)), true_class]

def entropy(probas, true_class):
    return -np.sum(probas * np.log(probas + 1e-10), axis=1)

# Compute scores using different methods
test_probas = clf.predict_proba(X_test)
inv_prob_scores = inverse_probability(test_probas, y_test)
margin_scores = margin(test_probas, y_test)
entropy_scores = entropy(test_probas, y_test)

# Compare score distributions
for name, scores in [("Inverse Probability", inv_prob_scores), 
                     ("Margin", margin_scores), 
                     ("Entropy", entropy_scores)]:
    print(f"{name} - Mean: {np.mean(scores):.4f}, Std: {np.std(scores):.4f}")
```

Slide 15: Real-Life Example: Medical Diagnosis

Conformal prediction can be particularly valuable in medical diagnosis, where understanding the confidence of a prediction is crucial. Let's consider a simplified example of predicting the risk of heart disease.

```python
from sklearn.datasets import load_heart_disease
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Load the heart disease dataset
X, y = load_heart_disease(return_X_y=True)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_cal, y_train, y_cal = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

# Train the model
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# Compute conformal scores
def conformal_score(model, X, y):
    probas = model.predict_proba(X)
    return 1 - probas[np.arange(len(y)), y]

cal_scores = conformal_score(clf, X_cal, y_cal)

# Set confidence level and find threshold
alpha = 0.1
threshold = np.percentile(cal_scores, (1 - alpha) * 100)

# Generate prediction sets for test data
test_probas = clf.predict_proba(X_test)
test_sets = [set(np.where(1 - p <= threshold)[0]) for p in test_probas]

# Evaluate results
coverage = sum(y_test[i] in s for i, s in enumerate(test_sets)) / len(y_test)
avg_set_size = np.mean([len(s) for s in test_sets])

print(f"Empirical coverage: {coverage:.2f}")
print(f"Average prediction set size: {avg_set_size:.2f}")

# Example predictions
for i in range(5):
    print(f"Patient {i+1}: True risk = {y_test[i]}, Prediction set = {test_sets[i]}")
```

Slide 16: Real-Life Example: Environmental Monitoring

Conformal prediction can be applied to regression tasks in environmental monitoring, such as predicting air quality index (AQI) with uncertainty estimates. This information is valuable for public health decisions and policy-making.

```python
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression

# Generate synthetic AQI data
X, y = make_regression(n_samples=1000, n_features=10, noise=5, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_cal, y_train, y_cal = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

# Train the model
reg = RandomForestRegressor(random_state=42)
reg.fit(X_train, y_train)

# Compute conformal scores
def conformal_score_reg(model, X, y):
    y_pred = model.predict(X)
    return np.abs(y - y_pred)

cal_scores = conformal_score_reg(reg, X_cal, y_cal)

# Set confidence level and find threshold
alpha = 0.1
threshold = np.percentile(cal_scores, (1 - alpha) * 100)

# Generate prediction intervals
y_pred = reg.predict(X_test)
lower = y_pred - threshold
upper = y_pred + threshold

# Evaluate coverage and interval width
coverage = np.mean((y_test >= lower) & (y_test <= upper))
avg_width = np.mean(upper - lower)

print(f"Empirical coverage: {coverage:.2f}")
print(f"Average interval width: {avg_width:.2f}")

# Example predictions
for i in range(5):
    print(f"Location {i+1}: True AQI = {y_test[i]:.1f}, Prediction interval = [{lower[i]:.1f}, {upper[i]:.1f}]")
```

Slide 17: Challenges and Limitations of Conformal Prediction

While conformal prediction offers many advantages, it's important to be aware of its limitations:

1.  Assumes exchangeability of data points
2.  May produce wide prediction intervals in some cases
3.  Computational overhead for large datasets
4.  Sensitivity to the choice of conformity score

Slide 18: Challenges and Limitations of Conformal Prediction

```python
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression

# Generate non-stationary data
np.random.seed(42)
X = np.random.rand(1000, 5)
y = 2 * X[:, 0] + np.sin(4 * np.pi * X[:, 1]) + 0.5 * np.random.randn(1000)
y[:500] += 5  # Add a shift to the first half of the data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_cal, y_train, y_cal = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

# Train model and compute conformal scores
reg = RandomForestRegressor(random_state=42)
reg.fit(X_train, y_train)

def conformal_score(model, X, y):
    return np.abs(y - model.predict(X))

cal_scores = conformal_score(reg, X_cal, y_cal)
threshold = np.percentile(cal_scores, 90)

# Generate prediction intervals
y_pred = reg.predict(X_test)
lower = y_pred - threshold
upper = y_pred + threshold

# Evaluate coverage and interval width
coverage = np.mean((y_test >= lower) & (y_test <= upper))
avg_width = np.mean(upper - lower)

print(f"Coverage: {coverage:.2f}")
print(f"Average interval width: {avg_width:.2f}")

# Check for potential issues
print(f"Max interval width: {np.max(upper - lower):.2f}")
print(f"Min interval width: {np.min(upper - lower):.2f}")
print(f"Interval width std: {np.std(upper - lower):.2f}")
```

Slide 19: Future Directions and Ongoing Research

Conformal prediction is an active area of research with several exciting directions:

1.  Adaptive conformal inference for handling distribution shifts
2.  Conformal prediction for time series and sequential data
3.  Integration with deep learning and uncertainty quantification in neural networks
4.  Efficient algorithms for large-scale and online conformal prediction

Slide 20: Future Directions and Ongoing Research

```python
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Simulate a time series with a trend
np.random.seed(42)
t = np.arange(1000)
y = 0.02 * t + np.sin(0.1 * t) + 0.5 * np.random.randn(1000)

# Split into training and test sets
t_train, t_test, y_train, y_test = train_test_split(t.reshape(-1, 1), y, test_size=0.2, random_state=42)

# Train a model
reg = RandomForestRegressor(random_state=42)
reg.fit(t_train, y_train)

# Simple adaptive conformal prediction
def adaptive_conformal_prediction(model, X_train, y_train, X_test, window_size=100):
    predictions = []
    intervals = []
    
    for i in range(len(X_test)):
        y_pred = model.predict(X_test[i].reshape(1, -1))[0]
        
        # Use a sliding window for calibration
        start = max(0, i - window_size)
        X_cal = X_train[start:i+start]
        y_cal = y_train[start:i+start]
        
        errors = np.abs(y_cal - model.predict(X_cal))
        threshold = np.percentile(errors, 90)
        
        predictions.append(y_pred)
        intervals.append((y_pred - threshold, y_pred + threshold))
    
    return np.array(predictions), np.array(intervals)

# Apply adaptive conformal prediction
y_pred, intervals = adaptive_conformal_prediction(reg, t_train, y_train, t_test)

# Evaluate results
coverage = np.mean((y_test >= intervals[:, 0]) & (y_test <= intervals[:, 1]))
avg_width = np.mean(intervals[:, 1] - intervals[:, 0])

print(f"Adaptive Conformal Prediction:")
print(f"Coverage: {coverage:.2f}")
print(f"Average interval width: {avg_width:.2f}")
```

Slide 21: Additional Resources

For those interested in diving deeper into conformal prediction, here are some valuable resources:

1.  "A Tutorial on Conformal Prediction" by Shafer and Vovk (2008) ArXiv: [https://arxiv.org/abs/0706.3188](https://arxiv.org/abs/0706.3188)
2.  "Distribution-Free Predictive Inference for Regression" by Lei et al. (2018) ArXiv: [https://arxiv.org/abs/1604.04173](https://arxiv.org/abs/1604.04173)
3.  "Conformal Prediction Under Covariate Shift" by Tibshirani et al. (2019) ArXiv: [https://arxiv.org/abs/1904.06019](https://arxiv.org/abs/1904.06019)
4.  "Conformalized Quantile Regression" by Romano et al. (2019) ArXiv: [https://arxiv.org/abs/1905.03222](https://arxiv.org/abs/1905.03222)

These papers provide a solid theoretical foundation and explore advanced topics in conformal prediction, including applications to various machine learning tasks and scenarios.

