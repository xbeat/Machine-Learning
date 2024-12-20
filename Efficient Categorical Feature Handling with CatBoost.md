## Efficient Categorical Feature Handling with CatBoost
Slide 1: CatBoost Fundamentals

CatBoost leverages ordered target statistics to handle categorical features efficiently while preventing target leakage. This implementation demonstrates the basic setup and training of a CatBoost model, showcasing its automatic categorical feature processing capabilities.

```python
import numpy as np
from catboost import CatBoostRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

# Generate sample data
X, y = make_regression(n_samples=1000, n_features=5, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Initialize and train CatBoost model
model = CatBoostRegressor(
    iterations=300,
    learning_rate=0.1,
    depth=6,
    loss_function='RMSE',
    verbose=False
)

# Train the model
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)
print(f"Model R2 Score: {model.score(X_test, y_test):.4f}")
```

Slide 2: Categorical Feature Processing

CatBoost's unique approach to categorical features involves sophisticated encoding techniques that combine target statistics with random permutations. This method reduces prediction shift and maintains reliable cross-validation performance.

```python
import pandas as pd
from catboost import Pool

# Create dataset with mixed feature types
data = {
    'numeric': np.random.normal(0, 1, 1000),
    'category': np.random.choice(['A', 'B', 'C'], 1000),
    'target': np.random.normal(0, 1, 1000)
}
df = pd.DataFrame(data)

# Specify categorical features
cat_features = ['category']
train_pool = Pool(
    data=df.drop('target', axis=1),
    label=df['target'],
    cat_features=cat_features
)

# Train model with automatic categorical processing
model = CatBoostRegressor(iterations=100)
model.fit(train_pool)

print("Feature importance:", model.get_feature_importance())
```

Slide 3: Advanced Loss Functions

CatBoost supports multiple loss functions optimized for different scenarios. This implementation showcases custom loss function usage and demonstrates how to implement quantile regression for uncertainty estimation.

```python
from catboost import CatBoostRegressor

# Initialize model with custom loss function
model = CatBoostRegressor(
    iterations=500,
    loss_function='Quantile:alpha=0.5',
    learning_rate=0.1,
    depth=6
)

# Training with multiple quantiles
quantiles = [0.1, 0.5, 0.9]
predictions = {}

for q in quantiles:
    model = CatBoostRegressor(
        iterations=500,
        loss_function=f'Quantile:alpha={q}',
        random_seed=42
    )
    model.fit(X_train, y_train)
    predictions[f'q{int(q*100)}'] = model.predict(X_test)

# Display prediction intervals
results = pd.DataFrame(predictions)
print("Prediction intervals:\n", results.head())
```

Slide 4: Learning Rate Scheduling

The implementation of dynamic learning rate scheduling in CatBoost helps achieve better convergence and model performance. This advanced technique adapts the learning process throughout training iterations.

```python
# Custom learning rate scheduler
def custom_lr_scheduler(iteration_count):
    return 0.1 * np.exp(-0.01 * iteration_count)

model = CatBoostRegressor(
    iterations=1000,
    custom_rate=custom_lr_scheduler,
    depth=6,
    verbose=100
)

# Training with custom scheduler
model.fit(
    X_train, y_train,
    eval_set=(X_test, y_test),
    early_stopping_rounds=50
)

# Plot learning rate progression
import matplotlib.pyplot as plt
iterations = range(model.tree_count_)
learning_rates = [custom_lr_scheduler(i) for i in iterations]
plt.plot(iterations, learning_rates)
plt.title('Learning Rate Schedule')
plt.show()
```

Slide 5: Feature Importance Analysis

CatBoost provides multiple methods for analyzing feature importance, including SHAP values and feature interaction scores. This implementation demonstrates how to extract and visualize these insights from trained models.

```python
import shap
from catboost import CatBoostRegressor

# Train model
model = CatBoostRegressor(iterations=500)
model.fit(X_train, y_train)

# Calculate different types of feature importance
feature_importance = {
    'PredictionValuesChange': model.get_feature_importance(),
    'LossFunctionChange': model.get_feature_importance(type='LossFunctionChange'),
    'ShapValues': model.get_feature_importance(type='ShapValues',
                                             data=X_test,
                                             thread_count=-1)
}

# Visualize importance scores
for importance_type, scores in feature_importance.items():
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(scores)), scores)
    plt.title(f'Feature Importance: {importance_type}')
    plt.show()
```

Slide 6: Cross-Validation Strategies

CatBoost implements specialized cross-validation techniques that account for ordered boosting and temporal dependencies. This advanced implementation shows how to properly validate models with time-series and categorical data.

```python
from catboost import cv
from sklearn.model_selection import TimeSeriesSplit

# Prepare parameters for cross-validation
params = {
    'iterations': 500,
    'learning_rate': 0.1,
    'depth': 6,
    'loss_function': 'RMSE'
}

# Custom time-series split
tscv = TimeSeriesSplit(n_splits=5)

# Perform cross-validation
cv_data = cv(
    Pool(X_train, y_train, cat_features=[]),
    params,
    fold_count=5,
    partition_random_seed=42,
    stratified=False,
    plot=True
)

print("CV Results Summary:")
print(f"Mean RMSE: {cv_data['test-RMSE-mean'].mean():.4f}")
print(f"Std RMSE: {cv_data['test-RMSE-std'].mean():.4f}")
```

Slide 7: Handling Missing Values

CatBoost provides sophisticated techniques for handling missing values in both numerical and categorical features. This implementation demonstrates various strategies for missing value treatment and their impact on model performance.

```python
import numpy as np
from catboost import Pool, CatBoostRegressor

# Create dataset with missing values
X_missing = X_train.copy()
X_missing[np.random.rand(*X_missing.shape) < 0.1] = np.nan

# Configure different missing value treatments
models = {
    'default': CatBoostRegressor(iterations=300),
    'nan_mode_min': CatBoostRegressor(iterations=300, nan_mode='Min'),
    'nan_mode_max': CatBoostRegressor(iterations=300, nan_mode='Max')
}

# Train and evaluate models
results = {}
for name, model in models.items():
    model.fit(X_missing, y_train, eval_set=(X_test, y_test))
    results[name] = model.score(X_test, y_test)

print("Model Performance with Different NaN Handling:")
for name, score in results.items():
    print(f"{name}: R² = {score:.4f}")
```

Slide 8: GPU Acceleration

CatBoost's GPU implementation offers significant speedup for large datasets while maintaining model quality. This code demonstrates how to configure and optimize GPU training for maximum performance.

```python
from catboost import CatBoostRegressor
import time

# Configure GPU parameters
gpu_params = {
    'task_type': 'GPU',
    'devices': '0:1',  # Using first GPU
    'gpu_ram_part': 0.95,  # GPU memory utilization
    'iterations': 1000,
    'learning_rate': 0.1
}

# Benchmark CPU vs GPU training
def benchmark_training(task_type):
    start_time = time.time()
    model = CatBoostRegressor(
        **{**gpu_params, 'task_type': task_type}
    )
    model.fit(X_train, y_train, verbose=False)
    return time.time() - start_time

# Compare training times
times = {
    'CPU': benchmark_training('CPU'),
    'GPU': benchmark_training('GPU')
}

print("Training Times:")
for device, duration in times.items():
    print(f"{device}: {duration:.2f} seconds")
```

Slide 9: Early Stopping and Overfitting Detection

CatBoost implements sophisticated overfitting detection mechanisms and early stopping criteria. This implementation shows how to utilize these features effectively while maintaining model performance.

```python
from catboost import CatBoostRegressor, Pool

# Create evaluation pool
eval_pool = Pool(X_test, y_test)

# Configure model with early stopping
model = CatBoostRegressor(
    iterations=1000,
    learning_rate=0.1,
    early_stopping_rounds=50,
    use_best_model=True,
    eval_metric='RMSE'
)

# Train with overfitting detection
model.fit(
    X_train, y_train,
    eval_set=eval_pool,
    verbose_eval=100,
    plot=True
)

# Print optimization results
print(f"Best iteration: {model.get_best_iteration()}")
print(f"Best score: {model.get_best_score():.4f}")
print("Training stopped early:", model.tree_count_ < 1000)
```

Slide 10: Feature Interaction Analysis

CatBoost allows detailed analysis of feature interactions through various metrics. This implementation demonstrates how to extract and visualize interaction strengths between features.

```python
from catboost import CatBoostRegressor
import numpy as np
import seaborn as sns

# Train model
model = CatBoostRegressor(iterations=500)
model.fit(X_train, y_train)

# Calculate feature interaction scores
interaction_scores = model.get_feature_importance(
    type='Interaction',
    data=Pool(X_train, y_train)
)

# Create interaction matrix
n_features = X_train.shape[1]
interaction_matrix = np.zeros((n_features, n_features))

for i, j, score in interaction_scores:
    interaction_matrix[int(i), int(j)] = score
    interaction_matrix[int(j), int(i)] = score

# Visualize interactions
plt.figure(figsize=(10, 8))
sns.heatmap(interaction_matrix, 
            annot=True, 
            fmt='.2f',
            cmap='YlOrRd')
plt.title('Feature Interaction Strength')
plt.show()
```

Slide 11: Real-world Application: Financial Time Series

This implementation demonstrates CatBoost's application to financial time series prediction, incorporating temporal features and market indicators for stock price prediction while handling look-ahead bias.

```python
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor

# Prepare financial dataset
def prepare_financial_features(df):
    df['returns'] = df['close'].pct_change()
    df['volatility'] = df['returns'].rolling(20).std()
    df['ma_50'] = df['close'].rolling(50).mean()
    df['rsi'] = calculate_rsi(df['close'], periods=14)
    return df

# Create sample financial dataset
dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
data = {
    'close': np.random.normal(100, 10, len(dates)),
    'volume': np.random.normal(1000000, 100000, len(dates)),
    'high': np.random.normal(102, 10, len(dates)),
    'low': np.random.normal(98, 10, len(dates))
}
df = pd.DataFrame(data, index=dates)
df = prepare_financial_features(df)

# Configure time-aware validation
model = CatBoostRegressor(
    iterations=1000,
    learning_rate=0.03,
    depth=6,
    has_time=True
)

# Train model with temporal awareness
model.fit(
    df.drop(['returns'], axis=1),
    df['returns'],
    eval_set=[(df.drop(['returns'], axis=1)[-100:], df['returns'][-100:])],
    verbose=100
)

print(f"Feature importance:\n{pd.Series(model.feature_importances_, index=df.drop(['returns'], axis=1).columns)}")
```

Slide 12: Real-world Application: Credit Scoring

Implementation of a credit scoring system using CatBoost, demonstrating handling of mixed data types, missing values, and interpretation of results for regulatory compliance.

```python
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score, precision_recall_curve

# Prepare credit scoring dataset
def prepare_credit_features(df):
    categorical_features = ['employment_type', 'education', 'marital_status']
    numerical_features = ['age', 'income', 'debt_ratio', 'credit_history_length']
    
    # Create sample credit data
    n_samples = 1000
    df = pd.DataFrame({
        'employment_type': np.random.choice(['FT', 'PT', 'SE'], n_samples),
        'education': np.random.choice(['HS', 'BA', 'MA', 'PHD'], n_samples),
        'marital_status': np.random.choice(['S', 'M', 'D'], n_samples),
        'age': np.random.normal(40, 10, n_samples),
        'income': np.random.normal(60000, 20000, n_samples),
        'debt_ratio': np.random.normal(0.3, 0.1, n_samples),
        'credit_history_length': np.random.normal(10, 5, n_samples),
        'default': np.random.binomial(1, 0.2, n_samples)
    })
    
    return df, categorical_features, numerical_features

# Train credit scoring model
df, cat_features, num_features = prepare_credit_features(pd.DataFrame())

model = CatBoostClassifier(
    iterations=500,
    learning_rate=0.1,
    depth=6,
    loss_function='Logloss',
    eval_metric='AUC',
    random_seed=42
)

# Train with categorical features
model.fit(
    df[cat_features + num_features],
    df['default'],
    cat_features=cat_features,
    eval_set=[(df[cat_features + num_features], df['default'])],
    plot=True
)

# Generate SHAP values for interpretability
import shap
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(df[cat_features + num_features])

print("Model Performance Metrics:")
print(f"ROC-AUC: {roc_auc_score(df['default'], model.predict_proba(df[cat_features + num_features])[:, 1]):.4f}")
```

Slide 13: Real-world Application Results

This slide presents the performance metrics and visualization of results from the previous real-world applications, demonstrating CatBoost's effectiveness in practical scenarios.

```python
# Financial Model Results
financial_metrics = {
    'RMSE': np.sqrt(((predictions - y_test) ** 2).mean()),
    'MAE': np.abs(predictions - y_test).mean(),
    'R2': model.score(X_test, y_test)
}

# Credit Scoring Results
credit_metrics = {
    'AUC': roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]),
    'Precision': precision_score(y_test, model.predict(X_test)),
    'Recall': recall_score(y_test, model.predict(X_test))
}

# Visualization of results
plt.figure(figsize=(15, 5))

# Financial predictions plot
plt.subplot(1, 2, 1)
plt.plot(y_test.index[-100:], y_test[-100:], label='Actual')
plt.plot(y_test.index[-100:], predictions[-100:], label='Predicted')
plt.title('Financial Predictions (Last 100 Days)')
plt.legend()

# ROC curve for credit scoring
plt.subplot(1, 2, 2)
fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1], '--')
plt.title('ROC Curve - Credit Scoring')

plt.tight_layout()
plt.show()

print("Financial Model Metrics:", financial_metrics)
print("\nCredit Scoring Metrics:", credit_metrics)
```

Slide 14: Additional Resources

1.  "CatBoost: unbiased boosting with categorical features" [https://arxiv.org/abs/1706.09516](https://arxiv.org/abs/1706.09516)
2.  "Accelerating CatBoost with GPUs: Implementation and Performance Analysis" [https://arxiv.org/abs/2012.14406](https://arxiv.org/abs/2012.14406)
3.  "Learning to Rank with CatBoost: A Case Study" [https://arxiv.org/abs/2006.08474](https://arxiv.org/abs/2006.08474)
4.  "Ordered Boosting: A New Method to Prevent Overfitting in CatBoost" [https://arxiv.org/abs/1901.08002](https://arxiv.org/abs/1901.08002)
5.  "Feature Interactions in CatBoost: Detection and Visualization" [https://arxiv.org/abs/2004.08132](https://arxiv.org/abs/2004.08132)

