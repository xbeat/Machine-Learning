## Overcoming Zero-Inflated Regression Challenges
Slide 1: Understanding Zero-Inflated Data

Zero-inflated datasets contain an excessive number of zero values in the target variable, which can significantly impact regression modeling performance. These scenarios commonly occur in real-world applications like count data, healthcare costs, or environmental measurements where zero observations are frequent and meaningful.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Generate sample zero-inflated data
np.random.seed(42)
n_samples = 1000
zeros = np.zeros(600)
non_zeros = np.random.exponential(scale=2.0, size=400)
y = np.concatenate([zeros, non_zeros])
X = np.random.normal(size=n_samples)

# Visualize distribution
plt.figure(figsize=(10, 6))
plt.hist(y, bins=50)
plt.title('Zero-Inflated Distribution')
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.show()
```

Slide 2: Traditional Regression Limitations

Standard regression models struggle with zero-inflated data because they assume continuous output distributions. Linear regression, for instance, can only predict exact zeros when the regression line crosses the origin, leading to poor predictions for zero-heavy datasets.

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Fit traditional linear regression
X = X.reshape(-1, 1)
reg = LinearRegression()
reg.fit(X, y)

# Predict and evaluate
y_pred = reg.predict(X)
mse = mean_squared_error(y, y_pred)

print(f"MSE with traditional regression: {mse:.4f}")
print(f"Number of exact zero predictions: {np.sum(y_pred == 0)}")
```

Slide 3: Zero-Inflated Model Architecture

The hybrid approach combines classification and regression models to handle zero-inflated data more effectively. The classifier determines if the output should be zero, while the regression model predicts non-zero values, creating a two-stage prediction pipeline.

```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

class ZeroInflatedRegressor:
    def __init__(self, classifier, regressor):
        self.classifier = classifier
        self.regressor = regressor
        
    def fit(self, X, y):
        # Create binary labels for classifier
        y_binary = (y != 0).astype(int)
        
        # Fit classifier
        self.classifier.fit(X, y_binary)
        
        # Fit regressor on non-zero data
        mask = y != 0
        self.regressor.fit(X[mask], y[mask])
        
        return self
```

Slide 4: Source Code for Zero-Inflated Model Implementation

```python
    def predict(self, X):
        # Get classification predictions
        zero_pred = self.classifier.predict(X)
        
        # Get regression predictions
        reg_pred = self.regressor.predict(X)
        
        # Combine predictions
        final_pred = np.where(zero_pred == 0, 0, reg_pred)
        
        return final_pred

# Initialize models
clf = RandomForestClassifier(n_estimators=100)
reg = LinearRegression()
zero_inf_model = ZeroInflatedRegressor(clf, reg)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

Slide 5: Model Training and Evaluation

A comprehensive evaluation framework compares traditional regression against the zero-inflated approach. The metrics focus on both overall prediction accuracy and the model's ability to correctly identify zero values in the target variable.

```python
# Train both models
reg_simple = LinearRegression()
reg_simple.fit(X_train, y_train)
zero_inf_model.fit(X_train, y_train)

# Predictions
y_pred_simple = reg_simple.predict(X_test)
y_pred_zeroinf = zero_inf_model.predict(X_test)

# Evaluation metrics
def evaluate_predictions(y_true, y_pred, model_name):
    mse = mean_squared_error(y_true, y_pred)
    zero_accuracy = np.mean((y_true == 0) == (y_pred == 0))
    print(f"\n{model_name} Results:")
    print(f"MSE: {mse:.4f}")
    print(f"Zero Prediction Accuracy: {zero_accuracy:.4f}")

evaluate_predictions(y_test, y_pred_simple, "Simple Regression")
evaluate_predictions(y_test, y_pred_zeroinf, "Zero-Inflated Model")
```

Slide 6: Real-World Example - Medical Costs Analysis

Healthcare cost analysis often involves zero-inflated data where many patients incur no costs while others generate significant expenses. This example demonstrates handling real medical claims data with a zero-inflated regression approach.

```python
# Simulate medical claims data
np.random.seed(42)
n_patients = 1000

# Features: age, bmi, smoker
age = np.random.normal(45, 15, n_patients)
bmi = np.random.normal(28, 5, n_patients)
smoker = np.random.binomial(1, 0.2, n_patients)

# Generate zero-inflated costs
prob_zero = 0.4
mask = np.random.binomial(1, 1-prob_zero, n_patients)
base_costs = np.exp(0.02*age + 0.1*bmi + 1.5*smoker + np.random.normal(0, 0.5, n_patients))
costs = mask * base_costs

X_medical = np.column_stack([age, bmi, smoker])
```

Slide 7: Source Code for Medical Costs Model

```python
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor

# Preprocess features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_medical)

# Split data
X_train, X_test, costs_train, costs_test = train_test_split(
    X_scaled, costs, test_size=0.2, random_state=42
)

# Initialize models
medical_clf = RandomForestClassifier(n_estimators=100)
medical_reg = GradientBoostingRegressor(n_estimators=100)
medical_model = ZeroInflatedRegressor(medical_clf, medical_reg)

# Train and evaluate
medical_model.fit(X_train, costs_train)
costs_pred = medical_model.predict(X_test)

# Calculate metrics
zero_accuracy = np.mean((costs_test == 0) == (costs_pred == 0))
mse = mean_squared_error(costs_test, costs_pred)
print(f"Zero Prediction Accuracy: {zero_accuracy:.4f}")
print(f"MSE: {mse:.4f}")
```

Slide 8: Advanced Performance Metrics

Zero-inflated models require specialized metrics beyond traditional regression measures. We implement custom evaluation metrics that account for both the binary classification performance and the continuous prediction accuracy.

```python
def zero_inflated_metrics(y_true, y_pred):
    # Binary metrics for zero vs non-zero
    true_zeros = y_true == 0
    pred_zeros = y_pred == 0
    
    precision = np.sum((true_zeros) & (pred_zeros)) / np.sum(pred_zeros)
    recall = np.sum((true_zeros) & (pred_zeros)) / np.sum(true_zeros)
    f1 = 2 * (precision * recall) / (precision + recall)
    
    # Continuous metrics for non-zero values
    mask_nonzero = ~true_zeros
    if np.sum(mask_nonzero) > 0:
        rmse_nonzero = np.sqrt(mean_squared_error(
            y_true[mask_nonzero], 
            y_pred[mask_nonzero]
        ))
    else:
        rmse_nonzero = np.nan
        
    return {
        'zero_precision': precision,
        'zero_recall': recall,
        'zero_f1': f1,
        'rmse_nonzero': rmse_nonzero
    }
```

Slide 9: Model Diagnostics and Visualization

Effective visualization techniques help understand the performance characteristics of zero-inflated models, particularly in identifying where the model succeeds or fails in predicting zeros versus continuous values.

```python
def plot_zero_inflated_diagnostics(y_true, y_pred):
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Actual vs Predicted
    plt.subplot(131)
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([0, max(y_true)], [0, max(y_true)], 'r--')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Actual vs Predicted')
    
    # Plot 2: Zero prediction distribution
    plt.subplot(132)
    true_zeros = y_true == 0
    pred_near_zero = np.abs(y_pred) < 0.1
    
    categories = ['True Zero\nPred Zero', 'True Zero\nPred Non-Zero',
                 'True Non-Zero\nPred Zero', 'True Non-Zero\nPred Non-Zero']
    counts = [
        np.sum(true_zeros & pred_near_zero),
        np.sum(true_zeros & ~pred_near_zero),
        np.sum(~true_zeros & pred_near_zero),
        np.sum(~true_zeros & ~pred_near_zero)
    ]
    
    plt.bar(categories, counts)
    plt.xticks(rotation=45)
    plt.title('Zero Prediction Analysis')
    
    plt.tight_layout()
    plt.show()
```

Slide 10: Environmental Data Example

Environmental monitoring data often exhibits zero-inflation, particularly in pollution or rainfall measurements. This example implements a zero-inflated model for daily precipitation prediction.

```python
# Simulate daily precipitation data
n_days = 1000

# Features: temperature, humidity, pressure
temperature = np.random.normal(20, 8, n_days)
humidity = np.random.normal(60, 15, n_days)
pressure = np.random.normal(1013, 5, n_days)

# Generate zero-inflated precipitation
prob_rain = 1 / (1 + np.exp(-(0.1*(humidity-60) - 0.05*(pressure-1013))))
rain_mask = np.random.binomial(1, prob_rain, n_days)
rain_amount = np.exp(0.05*humidity - 0.01*pressure + 
                    np.random.normal(0, 0.5, n_days)) * rain_mask

X_weather = np.column_stack([temperature, humidity, pressure])
```

Slide 11: Source Code for Environmental Model

```python
from sklearn.neural_network import MLPRegressor

# Preprocess environmental data
scaler_weather = StandardScaler()
X_weather_scaled = scaler_weather.fit_transform(X_weather)

# Split environmental data
X_weather_train, X_weather_test, rain_train, rain_test = train_test_split(
    X_weather_scaled, rain_amount, test_size=0.2, random_state=42
)

# Initialize specialized models for weather
weather_clf = RandomForestClassifier(n_estimators=150)
weather_reg = MLPRegressor(hidden_layer_sizes=(100, 50),
                          max_iter=1000,
                          random_state=42)
weather_model = ZeroInflatedRegressor(weather_clf, weather_reg)

# Train model
weather_model.fit(X_weather_train, rain_train)

# Generate predictions
rain_pred = weather_model.predict(X_weather_test)

# Calculate specialized metrics for precipitation
metrics = zero_inflated_metrics(rain_test, rain_pred)
print("\nPrecipitation Model Metrics:")
for metric, value in metrics.items():
    print(f"{metric}: {value:.4f}")
```

Slide 12: Model Comparison and Statistical Testing

A rigorous comparison between traditional and zero-inflated approaches requires statistical testing to validate performance improvements. This implementation uses cross-validation and statistical tests to compare models.

```python
from sklearn.model_selection import KFold
from scipy import stats

def compare_models(X, y, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Storage for metrics
    traditional_mse = []
    zeroinf_mse = []
    
    for train_idx, test_idx in kf.split(X):
        # Split data
        X_fold_train, X_fold_test = X[train_idx], X[test_idx]
        y_fold_train, y_fold_test = y[train_idx], y[test_idx]
        
        # Train traditional model
        trad_model = LinearRegression()
        trad_model.fit(X_fold_train, y_fold_train)
        trad_pred = trad_model.predict(X_fold_test)
        traditional_mse.append(mean_squared_error(y_fold_test, trad_pred))
        
        # Train zero-inflated model
        zi_model = ZeroInflatedRegressor(
            RandomForestClassifier(n_estimators=100),
            LinearRegression()
        )
        zi_model.fit(X_fold_train, y_fold_train)
        zi_pred = zi_model.predict(X_fold_test)
        zeroinf_mse.append(mean_squared_error(y_fold_test, zi_pred))
    
    # Perform statistical test
    t_stat, p_value = stats.ttest_rel(traditional_mse, zeroinf_mse)
    
    return {
        'traditional_mse_mean': np.mean(traditional_mse),
        'zeroinf_mse_mean': np.mean(zeroinf_mse),
        't_statistic': t_stat,
        'p_value': p_value
    }
```

Slide 13: Handling Imbalanced Zero-Inflation

When the proportion of zeros is extremely high, additional techniques like weighted sampling or SMOTE can improve model performance. This implementation shows how to handle severe zero-inflation.

```python
from imblearn.over_sampling import SMOTE

def train_balanced_zeroinf_model(X, y, sampling_strategy=0.5):
    # Create binary labels
    y_binary = (y != 0).astype(int)
    
    # Apply SMOTE to balance classes
    smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42)
    X_resampled, y_binary_resampled = smote.fit_resample(X, y_binary)
    
    # Create corresponding continuous values
    y_resampled = np.zeros_like(y_binary_resampled, dtype=float)
    non_zero_mask = y_binary_resampled == 1
    
    # Keep original non-zero values and generate synthetic ones
    original_non_zeros = y[y != 0]
    synthetic_non_zeros = np.random.choice(
        original_non_zeros,
        size=np.sum(non_zero_mask) - len(original_non_zeros)
    )
    
    y_resampled[non_zero_mask] = np.concatenate([
        original_non_zeros, synthetic_non_zeros
    ])
    
    return X_resampled, y_resampled
```

Slide 14: Additional Resources

*   "Zero-Inflated Regression Models for Count Data with Applications" - [https://arxiv.org/abs/1611.09321](https://arxiv.org/abs/1611.09321)
*   "A Comparative Study of Zero-Inflated Models" - [https://arxiv.org/abs/1712.08947](https://arxiv.org/abs/1712.08947)
*   "Deep Learning Approaches for Zero-Inflated Data" - [https://arxiv.org/abs/2003.00652](https://arxiv.org/abs/2003.00652)
*   Suggested Google Search Terms:
    *   "Zero-inflated regression python implementation"
    *   "Handling excess zeros in machine learning"
    *   "Two-part models for zero-inflated data"

