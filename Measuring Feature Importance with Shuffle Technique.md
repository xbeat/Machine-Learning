## Measuring Feature Importance with Shuffle Technique
Slide 1: Understanding Shuffle Feature Importance

Shuffle Feature Importance is a model-agnostic technique that measures feature importance by randomly permuting individual features and observing the impact on model performance. This approach provides insights into feature relevance without requiring model retraining.

```python
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor

def calculate_feature_importance(model, X_test, y_test, feature_name, n_iterations=10):
    # Calculate baseline performance
    baseline_pred = model.predict(X_test)
    baseline_score = mean_squared_error(y_test, baseline_pred)
    
    importance_scores = []
    
    # Perform multiple iterations of shuffling
    for _ in range(n_iterations):
        # Create a copy of test data
        X_test_shuffled = X_test.copy()
        # Shuffle the specific feature
        X_test_shuffled[feature_name] = np.random.permutation(X_test_shuffled[feature_name])
        
        # Calculate new performance
        new_pred = model.predict(X_test_shuffled)
        new_score = mean_squared_error(y_test, new_pred)
        
        # Calculate importance as performance drop
        importance = new_score - baseline_score
        importance_scores.append(importance)
    
    return np.mean(importance_scores)
```

Slide 2: Basic Implementation of Shuffle Importance

This implementation demonstrates how to apply shuffle feature importance to a random forest model using a sample dataset. The code includes data preparation, model training, and importance calculation for each feature.

```python
# Sample data preparation
from sklearn.datasets import make_regression

# Generate synthetic data
X, y = make_regression(n_samples=1000, n_features=5, random_state=42)
X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(5)])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)

# Calculate importance for each feature
feature_importance = {}
for feature in X.columns:
    importance = calculate_feature_importance(rf_model, X_test, y_test, feature)
    feature_importance[feature] = importance

# Sort features by importance
sorted_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
```

Slide 3: Visualization of Feature Importance

Creating clear visualizations helps interpret the results of shuffle feature importance analysis. This implementation uses seaborn and matplotlib to create informative bar plots of feature importance scores.

```python
import seaborn as sns
import matplotlib.pyplot as plt

def plot_feature_importance(importance_dict, title="Feature Importance Scores"):
    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(importance_dict.values()), 
                y=list(importance_dict.keys()),
                palette="viridis")
    
    plt.title(title)
    plt.xlabel("Importance Score (MSE Increase)")
    plt.ylabel("Features")
    
    # Add value labels on bars
    for i, v in enumerate(importance_dict.values()):
        plt.text(v, i, f'{v:.4f}', va='center')
    
    plt.tight_layout()
    return plt

# Plot feature importance
plot_feature_importance(sorted_importance)
plt.show()
```

Slide 4: Handling Categorical Features

Categorical features require special handling when implementing shuffle importance. This implementation shows how to properly shuffle categorical variables while maintaining their distribution characteristics.

```python
def calculate_categorical_importance(model, X_test, y_test, cat_feature, n_iterations=10):
    baseline_pred = model.predict(X_test)
    baseline_score = mean_squared_error(y_test, baseline_pred)
    importance_scores = []
    
    for _ in range(n_iterations):
        X_test_shuffled = X_test.copy()
        # Preserve category distribution while shuffling
        categories = X_test_shuffled[cat_feature].unique()
        for category in categories:
            mask = X_test_shuffled[cat_feature] == category
            indices = np.where(mask)[0]
            X_test_shuffled.loc[indices, cat_feature] = np.random.permutation(
                X_test_shuffled.loc[indices, cat_feature])
        
        new_pred = model.predict(X_test_shuffled)
        new_score = mean_squared_error(y_test, new_pred)
        importance_scores.append(new_score - baseline_score)
    
    return np.mean(importance_scores)
```

Slide 5: Real-world Example - Customer Churn Prediction

A practical implementation using a telecom customer churn dataset demonstrates shuffle importance in action. This example shows data preprocessing and importance calculation for customer retention analysis.

```python
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load telecom churn dataset
df = pd.DataFrame({
    'monthly_charges': np.random.uniform(20, 100, 1000),
    'tenure': np.random.randint(1, 72, 1000),
    'contract_type': np.random.choice(['Month-to-month', '1 year', '2 year'], 1000),
    'internet_service': np.random.choice(['DSL', 'Fiber optic', 'No'], 1000),
    'churn': np.random.choice([0, 1], 1000, p=[0.7, 0.3])
})

# Preprocess categorical variables
le = LabelEncoder()
df['contract_type_encoded'] = le.fit_transform(df['contract_type'])
df['internet_service_encoded'] = le.fit_transform(df['internet_service'])

# Prepare features and target
features = ['monthly_charges', 'tenure', 'contract_type_encoded', 'internet_service_encoded']
X = df[features]
y = df['churn']

# Train model and calculate importance
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# Calculate importance for each feature
churn_importance = {feature: calculate_feature_importance(clf, X_test, y_test, feature)
                   for feature in features}
```

Slide 6: Cross-validation Implementation

Implementing shuffle importance with cross-validation provides more robust importance estimates by reducing the impact of data splitting variability.

```python
from sklearn.model_selection import KFold
import numpy as np

def cross_validated_shuffle_importance(X, y, model, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    feature_importance = {col: [] for col in X.columns}
    
    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Train model
        model.fit(X_train, y_train)
        
        # Calculate importance for each feature
        for feature in X.columns:
            importance = calculate_feature_importance(model, X_test, y_test, feature)
            feature_importance[feature].append(importance)
    
    # Calculate mean importance across folds
    mean_importance = {feature: np.mean(scores) 
                      for feature, scores in feature_importance.items()}
    
    return mean_importance

# Example usage
cv_importance = cross_validated_shuffle_importance(X, y, RandomForestClassifier())
```

Slide 7: Parallel Implementation for Large Datasets

For large datasets, parallel processing can significantly reduce computation time. This implementation uses multiprocessing to calculate feature importance scores concurrently.

```python
from multiprocessing import Pool
from functools import partial

def parallel_shuffle_importance(model, X, y, n_jobs=-1):
    def _single_feature_importance(feature, X_test, y_test, model):
        return feature, calculate_feature_importance(model, X_test, y_test, feature)
    
    # Prepare partial function with fixed arguments
    partial_importance = partial(_single_feature_importance, 
                               X_test=X, 
                               y_test=y, 
                               model=model)
    
    # Calculate importance scores in parallel
    with Pool(n_jobs) as pool:
        results = pool.map(partial_importance, X.columns)
    
    return dict(results)

# Example usage
from sklearn.ensemble import GradientBoostingRegressor
model = GradientBoostingRegressor()
model.fit(X_train, y_train)

parallel_importance = parallel_shuffle_importance(model, X_test, y_test)
```

Slide 8: Feature Importance Confidence Intervals

Computing confidence intervals for feature importance scores helps assess the reliability of importance estimates and feature ranking stability.

```python
def calculate_importance_confidence_intervals(model, X, y, n_bootstrap=100, conf_level=0.95):
    n_samples = len(X)
    feature_importance_bootstrap = {col: [] for col in X.columns}
    
    for _ in range(n_bootstrap):
        # Bootstrap sampling
        indices = np.random.choice(n_samples, n_samples, replace=True)
        X_boot, y_boot = X.iloc[indices], y.iloc[indices]
        
        # Calculate importance for bootstrap sample
        importance = parallel_shuffle_importance(model, X_boot, y_boot)
        
        for feature, score in importance.items():
            feature_importance_bootstrap[feature].append(score)
    
    # Calculate confidence intervals
    confidence_intervals = {}
    for feature, scores in feature_importance_bootstrap.items():
        lower = np.percentile(scores, (1 - conf_level) / 2 * 100)
        upper = np.percentile(scores, (1 + conf_level) / 2 * 100)
        confidence_intervals[feature] = (lower, upper)
    
    return confidence_intervals
```

Slide 9: Handling Correlated Features

This implementation addresses the challenge of correlated features in shuffle importance analysis by identifying and clustering correlated features before importance calculation.

```python
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.stats import spearmanr

def handle_correlated_features(X, threshold=0.8):
    # Calculate correlation matrix
    corr_matrix = np.abs(spearmanr(X)[0])
    
    # Convert correlation matrix to distance matrix
    distance_matrix = 1 - corr_matrix
    
    # Perform hierarchical clustering
    linkage_matrix = linkage(distance_matrix, method='complete')
    clusters = fcluster(linkage_matrix, t=1-threshold, criterion='distance')
    
    # Select representative features from each cluster
    representative_features = []
    for cluster_id in np.unique(clusters):
        cluster_features = X.columns[clusters == cluster_id]
        if len(cluster_features) == 1:
            representative_features.append(cluster_features[0])
        else:
            # Select feature with highest variance in cluster
            variances = X[cluster_features].var()
            representative_features.append(variances.idxmax())
    
    return representative_features, dict(zip(X.columns, clusters))

# Example usage
representative_features, cluster_mapping = handle_correlated_features(X)
importance_uncorrelated = calculate_feature_importance(model, 
                                                     X_test[representative_features], 
                                                     y_test,
                                                     representative_features[0])
```

Slide 10: Time Series Implementation

Adapting shuffle importance for time series data requires special consideration to maintain temporal dependencies while shuffling features.

```python
def time_series_shuffle_importance(model, X, y, feature, window_size=10):
    def sliding_window_shuffle(series, window_size):
        shuffled_series = series.copy()
        n_windows = len(series) // window_size
        
        for i in range(n_windows):
            start_idx = i * window_size
            end_idx = start_idx + window_size
            window = shuffled_series[start_idx:end_idx]
            shuffled_series[start_idx:end_idx] = np.random.permutation(window)
        
        return shuffled_series

    baseline_pred = model.predict(X)
    baseline_score = mean_squared_error(y, baseline_pred)
    
    X_shuffled = X.copy()
    X_shuffled[feature] = sliding_window_shuffle(X_shuffled[feature], window_size)
    
    shuffled_pred = model.predict(X_shuffled)
    shuffled_score = mean_squared_error(y, shuffled_pred)
    
    return shuffled_score - baseline_score

# Example with time series data
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Generate sample time series data
dates = pd.date_range(start='2023-01-01', periods=1000, freq='D')
df_ts = pd.DataFrame({
    'date': dates,
    'feature1': np.random.normal(0, 1, 1000),
    'feature2': np.random.normal(0, 1, 1000),
    'target': np.random.normal(0, 1, 1000)
})

# Calculate time series importance
X_ts = df_ts[['feature1', 'feature2']]
y_ts = df_ts['target']
model_ts = RandomForestRegressor().fit(X_ts, y_ts)

ts_importance = {feature: time_series_shuffle_importance(model_ts, X_ts, y_ts, feature)
                for feature in X_ts.columns}
```

Slide 11: Multi-output Implementation

Extending shuffle importance to handle multiple output variables requires aggregating importance scores across all outputs.

```python
def multi_output_shuffle_importance(model, X, y, feature, aggregation='mean'):
    """
    Calculate feature importance for multi-output models
    
    Parameters:
    aggregation: str, {'mean', 'max', 'min'}
        Method to aggregate importance across outputs
    """
    baseline_pred = model.predict(X)
    baseline_scores = [mean_squared_error(y[:, i], baseline_pred[:, i])
                      for i in range(y.shape[1])]
    
    X_shuffled = X.copy()
    X_shuffled[feature] = np.random.permutation(X_shuffled[feature])
    
    shuffled_pred = model.predict(X_shuffled)
    shuffled_scores = [mean_squared_error(y[:, i], shuffled_pred[:, i])
                      for i in range(y.shape[1])]
    
    importance_per_output = [s2 - s1 for s1, s2 
                           in zip(baseline_scores, shuffled_scores)]
    
    if aggregation == 'mean':
        return np.mean(importance_per_output)
    elif aggregation == 'max':
        return np.max(importance_per_output)
    elif aggregation == 'min':
        return np.min(importance_per_output)
    else:
        raise ValueError("Invalid aggregation method")

# Example usage with multi-output regression
X_multi, y_multi = make_regression(n_samples=1000, 
                                 n_features=5, 
                                 n_targets=3, 
                                 random_state=42)
X_multi = pd.DataFrame(X_multi, columns=[f'feature_{i}' for i in range(5)])

model_multi = RandomForestRegressor()
model_multi.fit(X_multi, y_multi)

multi_importance = {feature: multi_output_shuffle_importance(model_multi, 
                                                           X_multi, 
                                                           y_multi, 
                                                           feature)
                   for feature in X_multi.columns}
```

Slide 12: Feature Importance Stability Analysis

This implementation assesses the stability of feature importance rankings across multiple random seeds and data subsets to ensure reliable feature selection.

```python
def analyze_importance_stability(X, y, model_class, n_iterations=50, sample_fraction=0.8):
    feature_rankings = {col: [] for col in X.columns}
    importance_values = {col: [] for col in X.columns}
    
    for i in range(n_iterations):
        # Subsample data
        indices = np.random.choice(len(X), 
                                 size=int(len(X) * sample_fraction), 
                                 replace=False)
        X_subset = X.iloc[indices]
        y_subset = y.iloc[indices]
        
        # Train model and calculate importance
        model = model_class(random_state=i)
        model.fit(X_subset, y_subset)
        importance = {col: calculate_feature_importance(model, X_subset, y_subset, col)
                     for col in X.columns}
        
        # Store rankings and values
        ranked_features = sorted(importance.items(), 
                               key=lambda x: x[1], 
                               reverse=True)
        for rank, (feature, value) in enumerate(ranked_features):
            feature_rankings[feature].append(rank)
            importance_values[feature].append(value)
    
    # Calculate stability metrics
    stability_metrics = {}
    for feature in X.columns:
        stability_metrics[feature] = {
            'mean_rank': np.mean(feature_rankings[feature]),
            'rank_std': np.std(feature_rankings[feature]),
            'mean_importance': np.mean(importance_values[feature]),
            'importance_std': np.std(importance_values[feature]),
            'rank_range': np.ptp(feature_rankings[feature])
        }
    
    return stability_metrics
```

Slide 13: Real-world Example - Credit Risk Assessment

Implementing shuffle importance for credit risk assessment using actual lending data demonstrates the practical application in financial modeling.

```python
# Generate sample credit data
np.random.seed(42)
n_samples = 1000

credit_data = pd.DataFrame({
    'income': np.random.normal(50000, 20000, n_samples),
    'credit_score': np.random.normal(700, 50, n_samples),
    'debt_ratio': np.random.uniform(0.1, 0.6, n_samples),
    'employment_length': np.random.randint(0, 30, n_samples),
    'default': np.random.binomial(1, 0.15, n_samples)
})

# Preprocess features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_credit = credit_data.drop('default', axis=1)
X_credit_scaled = pd.DataFrame(scaler.fit_transform(X_credit),
                             columns=X_credit.columns)
y_credit = credit_data['default']

# Train model and calculate importance
from sklearn.ensemble import GradientBoostingClassifier
credit_model = GradientBoostingClassifier(random_state=42)
credit_model.fit(X_credit_scaled, y_credit)

credit_importance = {}
for feature in X_credit_scaled.columns:
    importance = calculate_feature_importance(credit_model, 
                                           X_credit_scaled, 
                                           y_credit, 
                                           feature)
    credit_importance[feature] = importance

# Visualize results with confidence intervals
confidence_intervals = calculate_importance_confidence_intervals(
    credit_model, X_credit_scaled, y_credit)

# Plot with error bars
plt.figure(figsize=(10, 6))
features = list(credit_importance.keys())
values = list(credit_importance.values())
errors = [(confidence_intervals[f][1] - confidence_intervals[f][0])/2 
          for f in features]

plt.errorbar(values, range(len(features)), 
            xerr=errors, fmt='o', capsize=5)
plt.yticks(range(len(features)), features)
plt.xlabel('Importance Score')
plt.title('Credit Risk Feature Importance with 95% Confidence Intervals')
plt.tight_layout()
```

Slide 14: Additional Resources

*   "Feature Importance Analysis for Deep Neural Networks" - [https://arxiv.org/abs/1905.02639](https://arxiv.org/abs/1905.02639)
*   "Understanding Feature Importance Stability in High-Dimensional Data" - [https://arxiv.org/abs/2010.09219](https://arxiv.org/abs/2010.09219)
*   "A Survey of Methods for Model-Agnostic Feature Importance" - [https://arxiv.org/abs/2010.13070](https://arxiv.org/abs/2010.13070)
*   For more detailed implementations and examples, search for:
    *   "Permutation Importance Methods in Scikit-learn"
    *   "Feature Importance Stability Analysis in Python"
    *   "Model-Agnostic Feature Selection Techniques"

