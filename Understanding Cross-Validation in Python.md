## Understanding Cross-Validation in Python
Slide 1: Understanding Cross-Validation

Cross-validation is a statistical method used to assess model performance by partitioning data into training and testing sets multiple times. It helps evaluate how well a model generalizes to unseen data and reduces overfitting by providing multiple evaluation metrics across different data splits.

```python
from sklearn.model_selection import KFold
import numpy as np

# Generate sample data
X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
y = np.array([0, 1, 0, 1, 0])

# Initialize KFold
kf = KFold(n_splits=3, shuffle=True, random_state=42)

# Perform k-fold split
for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
    print(f"Fold {fold + 1}:")
    print(f"Training indices: {train_idx}")
    print(f"Testing indices: {test_idx}\n")
```

Slide 2: K-Fold Cross-Validation Implementation

The k-fold cross-validation technique divides the dataset into k equal-sized folds, using k-1 folds for training and the remaining fold for validation. This process repeats k times, with each fold serving as the validation set exactly once.

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

def custom_kfold_cv(X, y, model, k=5):
    # Shuffle data
    indices = np.random.permutation(len(X))
    X, y = X[indices], y[indices]
    
    # Calculate fold size
    fold_size = len(X) // k
    scores = []
    
    for i in range(k):
        # Define test indices
        test_start = i * fold_size
        test_end = (i + 1) * fold_size
        
        # Split data
        test_indices = slice(test_start, test_end)
        X_test, y_test = X[test_indices], y[test_indices]
        X_train = np.concatenate([X[:test_start], X[test_end:]])
        y_train = np.concatenate([y[:test_start], y[test_end:]])
        
        # Train and evaluate
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        scores.append(score)
    
    return np.mean(scores), np.std(scores)
```

Slide 3: Stratified Cross-Validation

Stratified cross-validation maintains the original class distribution in each fold, ensuring representative sampling for imbalanced datasets. This technique is crucial when dealing with classification problems where class proportions significantly impact model performance.

```python
from sklearn.model_selection import StratifiedKFold
from sklearn.datasets import make_classification

# Generate imbalanced dataset
X, y = make_classification(n_samples=1000, n_classes=2, weights=[0.9, 0.1],
                         random_state=42)

# Initialize StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Compare class distributions
for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
    train_dist = np.bincount(y[train_idx]) / len(train_idx)
    test_dist = np.bincount(y[test_idx]) / len(test_idx)
    print(f"Fold {fold + 1}:")
    print(f"Training distribution: {train_dist}")
    print(f"Testing distribution: {test_dist}\n")
```

Slide 4: Time Series Cross-Validation

Time series cross-validation respects temporal ordering by using past observations for training and future observations for testing, preventing data leakage and maintaining the temporal dependency structure of the data.

```python
from sklearn.model_selection import TimeSeriesSplit
import pandas as pd

# Generate time series data
dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
data = pd.Series(np.random.randn(100), index=dates)

# Initialize TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)

# Perform time series split
for fold, (train_idx, test_idx) in enumerate(tscv.split(data)):
    train_dates = data.index[train_idx]
    test_dates = data.index[test_idx]
    print(f"Fold {fold + 1}:")
    print(f"Training period: {train_dates[0]} to {train_dates[-1]}")
    print(f"Testing period: {test_dates[0]} to {test_dates[-1]}\n")
```

Slide 5: Leave-One-Out Cross-Validation

Leave-one-out cross-validation (LOOCV) is an extreme form of k-fold cross-validation where k equals the number of observations. This method provides unbiased performance estimates but can be computationally expensive for large datasets.

```python
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_squared_error

def leave_one_out_cv(X, y, model):
    loo = LeaveOneOut()
    predictions = []
    true_values = []
    
    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        
        predictions.extend(pred)
        true_values.extend(y_test)
    
    return mean_squared_error(true_values, predictions)
```

Slide 6: Real-World Example: Credit Risk Assessment

Let's implement cross-validation for a credit risk assessment model using a comprehensive pipeline that includes data preprocessing, model training, and evaluation metrics for financial risk prediction.

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate

# Load and preprocess data
def prepare_credit_data():
    # Simulated credit data
    data = pd.DataFrame({
        'income': np.random.normal(50000, 20000, 1000),
        'debt_ratio': np.random.uniform(0, 1, 1000),
        'credit_score': np.random.normal(700, 50, 1000),
        'default': np.random.binomial(1, 0.2, 1000)
    })
    return data

# Create pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Prepare data
data = prepare_credit_data()
X = data.drop('default', axis=1)
y = data['default']

# Perform cross-validation
cv_results = cross_validate(pipeline, X, y,
                          cv=5,
                          scoring=['accuracy', 'precision', 'recall', 'f1'])

# Print results
for metric in cv_results:
    print(f"{metric}: {cv_results[metric].mean():.3f} ± {cv_results[metric].std():.3f}")
```

Slide 7: Cross-Validation Error Analysis

Cross-validation error analysis involves examining the variation in model performance across folds to identify potential issues such as overfitting, underfitting, or dataset bias.

```python
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve

def plot_learning_curves(estimator, X, y, cv=5):
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=-1, 
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='accuracy'
    )
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_scores.mean(axis=1), label='Training score')
    plt.plot(train_sizes, test_scores.mean(axis=1), label='Cross-validation score')
    plt.xlabel('Training examples')
    plt.ylabel('Score')
    plt.title('Learning Curves')
    plt.legend(loc='best')
    plt.grid(True)
    
    # Calculate error bars
    train_std = train_scores.std(axis=1)
    test_std = test_scores.std(axis=1)
    plt.fill_between(train_sizes, 
                    train_scores.mean(axis=1) - train_std,
                    train_scores.mean(axis=1) + train_std, 
                    alpha=0.1)
    plt.fill_between(train_sizes, 
                    test_scores.mean(axis=1) - test_std,
                    test_scores.mean(axis=1) + test_std, 
                    alpha=0.1)
    return plt
```

Slide 8: Nested Cross-Validation

Nested cross-validation uses two loops of cross-validation: an outer loop for model assessment and an inner loop for model selection and hyperparameter tuning. This approach provides unbiased performance estimates while optimizing model parameters.

```python
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.svm import SVC

def nested_cross_validation(X, y, inner_cv=5, outer_cv=5):
    # Parameter grid for SVM
    param_grid = {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto']
    }
    
    # Initialize base model
    base_model = SVC(random_state=42)
    
    # Initialize GridSearchCV as inner loop
    inner_cv_model = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=inner_cv,
        scoring='accuracy'
    )
    
    # Perform outer cross-validation
    outer_scores = cross_val_score(
        inner_cv_model, X, y,
        cv=outer_cv,
        scoring='accuracy'
    )
    
    return {
        'mean_score': outer_scores.mean(),
        'std_score': outer_scores.std(),
        'scores': outer_scores
    }
```

Slide 9: Cross-Validation for Time Series Forecasting

Time series cross-validation requires special consideration of temporal dependencies and implements a rolling window approach to maintain the chronological order of observations while preventing data leakage.

```python
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

def time_series_cv(data, n_splits, initial_window, horizon=1):
    """
    Custom time series cross-validation with rolling window
    """
    results = []
    
    for i in range(n_splits):
        start = initial_window + i
        train_end = start + horizon
        test_end = train_end + horizon
        
        # Split data into training and testing sets
        train = data[i:train_end]
        test = data[train_end:test_end]
        
        # Store indices for visualization
        results.append({
            'train_start': i,
            'train_end': train_end,
            'test_start': train_end,
            'test_end': test_end,
            'train_set': train,
            'test_set': test
        })
    
    return results

# Example usage
dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
ts_data = pd.Series(np.random.randn(100).cumsum(), index=dates)

cv_splits = time_series_cv(
    data=ts_data,
    n_splits=5,
    initial_window=30,
    horizon=7
)

# Print splits information
for i, split in enumerate(cv_splits):
    print(f"Split {i + 1}:")
    print(f"Train period: {split['train_set'].index[0]} to {split['train_set'].index[-1]}")
    print(f"Test period: {split['test_set'].index[0]} to {split['test_set'].index[-1]}\n")
```

Slide 10: Real-World Example: Stock Price Prediction

Implementation of cross-validation for a stock price prediction model, demonstrating proper handling of financial time series data with appropriate preprocessing and evaluation metrics.

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error

def prepare_stock_data(data, lookback=30):
    """
    Prepare sequences of stock data for prediction
    """
    X, y = [], []
    for i in range(len(data) - lookback):
        X.append(data[i:(i + lookback)])
        y.append(data[i + lookback])
    return np.array(X), np.array(y)

def stock_prediction_cv(data, n_splits=5, lookback=30):
    # Scale the data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data.reshape(-1, 1))
    
    # Prepare sequences
    X, y = prepare_stock_data(scaled_data, lookback)
    
    # Initialize results storage
    cv_scores = []
    
    # Perform time-series CV
    for i in range(n_splits):
        split_idx = int(len(X) * (0.6 + i * 0.1))
        X_train, X_test = X[:split_idx], X[split_idx:split_idx+lookback]
        y_train, y_test = y[:split_idx], y[split_idx:split_idx+lookback]
        
        # Train model (simple example using moving average)
        predictions = np.mean(X_test, axis=1)
        
        # Calculate MAPE
        mape = mean_absolute_percentage_error(
            scaler.inverse_transform(y_test.reshape(-1, 1)),
            scaler.inverse_transform(predictions.reshape(-1, 1))
        )
        cv_scores.append(mape)
    
    return {
        'mean_mape': np.mean(cv_scores),
        'std_mape': np.std(cv_scores),
        'individual_scores': cv_scores
    }

# Generate sample stock data
np.random.seed(42)
stock_prices = np.random.randn(1000).cumsum() + 100
results = stock_prediction_cv(stock_prices)
print(f"Mean MAPE: {results['mean_mape']:.4f} ± {results['std_mape']:.4f}")
```

Slide 11: Cross-Validation Metrics Visualization

A comprehensive visualization system for cross-validation results that helps identify patterns in model performance across different folds and metrics.

```python
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score

def visualize_cv_metrics(model, X, y, cv=5):
    # Define scoring metrics
    scoring = {
        'accuracy': make_scorer(accuracy_score),
        'precision': make_scorer(precision_score),
        'recall': make_scorer(recall_score),
        'f1': make_scorer(f1_score)
    }
    
    # Perform cross-validation
    cv_results = cross_validate(
        model, X, y,
        scoring=scoring,
        cv=cv,
        return_train_score=True
    )
    
    # Prepare data for visualization
    metrics_data = []
    for metric in scoring.keys():
        train_scores = cv_results[f'train_{metric}']
        test_scores = cv_results[f'test_{metric}']
        
        for fold, (train, test) in enumerate(zip(train_scores, test_scores)):
            metrics_data.append({
                'Metric': metric,
                'Score': train,
                'Type': 'Train',
                'Fold': fold + 1
            })
            metrics_data.append({
                'Metric': metric,
                'Score': test,
                'Type': 'Test',
                'Fold': fold + 1
            })
    
    # Create visualization
    plt.figure(figsize=(12, 6))
    df_metrics = pd.DataFrame(metrics_data)
    sns.boxplot(x='Metric', y='Score', hue='Type', data=df_metrics)
    plt.title('Cross-Validation Metrics Distribution')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    return plt

# Return statistical summary
def get_metrics_summary(cv_results):
    summary = {}
    for metric in cv_results.keys():
        if metric.startswith(('train_', 'test_')):
            summary[metric] = {
                'mean': cv_results[metric].mean(),
                'std': cv_results[metric].std(),
                'min': cv_results[metric].min(),
                'max': cv_results[metric].max()
            }
    return summary
```

Slide 12: Cross-Validation for Hyperparameter Tuning

Cross-validation plays a crucial role in hyperparameter optimization by providing reliable estimates of model performance across different parameter combinations while avoiding overfitting to specific data splits.

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform
import numpy as np

def advanced_hyperparameter_tuning(X, y, base_model, param_dist, n_iter=100, cv=5):
    # Initialize random search with cross-validation
    random_search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=cv,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1,
        return_train_score=True
    )
    
    # Fit the random search
    random_search.fit(X, y)
    
    # Extract results
    cv_results = pd.DataFrame(random_search.cv_results_)
    
    # Calculate performance statistics
    best_params = random_search.best_params_
    best_score = random_search.best_score_
    
    # Analyze parameter importance
    param_importance = {}
    for param in param_dist.keys():
        param_scores = cv_results.groupby(f'param_{param}')['mean_test_score'].mean()
        param_importance[param] = param_scores.std()
    
    return {
        'best_params': best_params,
        'best_score': best_score,
        'param_importance': param_importance,
        'cv_results': cv_results
    }

# Example usage
from sklearn.ensemble import RandomForestClassifier

# Define parameter distribution
param_dist = {
    'n_estimators': randint(50, 300),
    'max_depth': [None] + list(range(10, 31)),
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 10)
}

# Generate sample data
X = np.random.randn(1000, 20)
y = np.random.binomial(1, 0.5, 1000)

results = advanced_hyperparameter_tuning(
    X, y,
    RandomForestClassifier(random_state=42),
    param_dist
)

print(f"Best parameters: {results['best_params']}")
print(f"Best CV score: {results['best_score']:.4f}")
```

Slide 13: Cross-Validation with Imbalanced Data

When dealing with imbalanced datasets, specialized cross-validation techniques are required to maintain class proportions and provide meaningful performance metrics across all folds.

```python
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.metrics import confusion_matrix, classification_report

def imbalanced_cross_validation(X, y, classifier, n_splits=5):
    # Initialize stratified k-fold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Initialize results storage
    fold_results = []
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Create pipeline with SMOTE
        pipeline = ImbPipeline([
            ('sampler', SMOTE(random_state=42)),
            ('classifier', classifier)
        ])
        
        # Train and predict
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        
        # Calculate metrics
        conf_matrix = confusion_matrix(y_test, y_pred)
        class_report = classification_report(y_test, y_pred, output_dict=True)
        
        fold_results.append({
            'fold': fold + 1,
            'confusion_matrix': conf_matrix,
            'classification_report': class_report
        })
    
    return fold_results

# Calculate average metrics across folds
def aggregate_cv_results(fold_results):
    metrics = ['precision', 'recall', 'f1-score']
    classes = list(fold_results[0]['classification_report'].keys())[:-3]
    
    aggregated = {metric: {cls: [] for cls in classes} for metric in metrics}
    
    for result in fold_results:
        report = result['classification_report']
        for cls in classes:
            for metric in metrics:
                aggregated[metric][cls].append(report[cls][metric])
    
    # Calculate means and standard deviations
    summary = {}
    for metric in metrics:
        summary[metric] = {
            cls: {
                'mean': np.mean(values),
                'std': np.std(values)
            }
            for cls, values in aggregated[metric].items()
        }
    
    return summary
```

Slide 14: Additional Resources

*   Statistical Learning Through Cross-Validation: [https://arxiv.org/abs/2001.09111](https://arxiv.org/abs/2001.09111)
*   A Survey of Cross-Validation Procedures for Model Selection: [https://arxiv.org/abs/1811.12808](https://arxiv.org/abs/1811.12808)
*   Optimal Cross-Validation Splits for Time Series Data: [https://arxiv.org/abs/1908.07087](https://arxiv.org/abs/1908.07087)
*   Cross-Validation Strategies for Modern Machine Learning: [https://www.sciencedirect.com/science/article/pii/S2590238521000296](https://www.sciencedirect.com/science/article/pii/S2590238521000296)
*   Time Series Cross-Validation Techniques: [https://www.sciencedirect.com/science/article/abs/pii/S0169207016000121](https://www.sciencedirect.com/science/article/abs/pii/S0169207016000121)
*   Best Practices in Cross-Validation for Predictive Modeling: [https://dl.acm.org/doi/10.1145/3292500.3330701](https://dl.acm.org/doi/10.1145/3292500.3330701)

