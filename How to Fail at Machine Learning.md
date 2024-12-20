## How to Fail at Machine Learning
Slide 1: Data Leakage - The Silent Model Killer

Data leakage occurs when information from outside the training dataset influences the model development process, leading to overoptimistic performance metrics and poor generalization. This fundamental mistake happens when features contain information about the target that wouldn't be available during real-world predictions.

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Wrong way: scaling before splitting
data = pd.DataFrame(np.random.randn(1000, 4), columns=['f1', 'f2', 'f3', 'target'])
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)  # Leakage: test data influences scaling
X_train, X_test, y_train, y_test = train_test_split(scaled_data[:,:3], scaled_data[:,3])

# Correct way: scale after splitting
X_train, X_test, y_train, y_test = train_test_split(data.drop('target',axis=1), data['target'])
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Fit only on training data
X_test_scaled = scaler.transform(X_test)  # Apply same transformation to test
```

Slide 2: Target Encoding Gone Wrong

Target encoding is a powerful feature engineering technique that can easily lead to overfitting when implemented incorrectly. Using the entire dataset for encoding instead of proper cross-validation creates a deceptive performance boost that won't generalize.

```python
import pandas as pd
from sklearn.model_selection import KFold

# Wrong way: encoding using all data
def naive_target_encoding(df, cat_col, target_col):
    means = df.groupby(cat_col)[target_col].mean()
    return df[cat_col].map(means)

# Correct way: cross-fold target encoding
def proper_target_encoding(df, cat_col, target_col, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    df['encoded'] = np.nan
    
    for train_idx, val_idx in kf.split(df):
        means = df.iloc[train_idx].groupby(cat_col)[target_col].mean()
        df.loc[val_idx, 'encoded'] = df.loc[val_idx, cat_col].map(means)
    
    return df['encoded']
```

Slide 3: P-Hacking Through Multiple Testing

P-hacking involves repeatedly testing different hypotheses or model variations until achieving statistically significant results. This practice invalidates the fundamental assumptions of statistical testing and leads to false discoveries.

```python
import numpy as np
from scipy import stats

# Wrong way: testing multiple hypotheses without correction
def p_hacking_example(n_experiments=1000):
    significant_results = 0
    alpha = 0.05
    
    for _ in range(n_experiments):
        # Generate random data with no real effect
        control = np.random.normal(100, 15, 30)
        treatment = np.random.normal(100, 15, 30)
        
        # Keep testing until finding "significant" result
        t_stat, p_value = stats.ttest_ind(control, treatment)
        if p_value < alpha:
            significant_results += 1
    
    print(f"False positive rate: {significant_results/n_experiments:.2%}")
    # Output: ~5% of tests will be "significant" by chance

# Correct way: Using Bonferroni correction
def proper_multiple_testing(n_experiments=1000):
    alpha_corrected = 0.05 / n_experiments  # Bonferroni correction
    # ... rest of the analysis
```

Slide 4: Time Series Cross-Validation Mistakes

When dealing with time series data, using traditional k-fold cross-validation leads to data leakage by allowing the model to peek into the future. This creates unrealistic performance estimates and models that fail in production.

```python
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

# Wrong way: regular k-fold CV for time series
def wrong_ts_validation(df):
    kf = KFold(n_splits=5, shuffle=True)  # Shuffling breaks temporal order
    for train_idx, val_idx in kf.split(df):
        # This allows future data to influence past predictions
        pass

# Correct way: time series CV
def proper_ts_validation(df):
    tscv = TimeSeriesSplit(n_splits=5)
    for train_idx, val_idx in tscv.split(df):
        # Maintains temporal order
        train_data = df.iloc[train_idx]
        val_data = df.iloc[val_idx]
        assert train_data.index.max() < val_data.index.min()
```

Slide 5: Feature Selection Bias

The common mistake of performing feature selection on the entire dataset before splitting leads to information leakage and overoptimistic model performance. This creates a false sense of model capability and poor generalization.

```python
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# Wrong way: feature selection before split
X = df.drop('target', axis=1)
y = df['target']
selector = SelectKBest(f_classif, k=10)
X_selected = selector.fit_transform(X, y)  # Leakage!
X_train, X_test, y_train, y_test = train_test_split(X_selected, y)

# Correct way: feature selection within pipeline
pipeline = Pipeline([
    ('feature_selection', SelectKBest(f_classif, k=10)),
    ('classifier', RandomForestClassifier())
])
X_train, X_test, y_train, y_test = train_test_split(X, y)
pipeline.fit(X_train, y_train)
```

Slide 6: Sample Weights Manipulation

Improper handling of sample weights can dramatically skew model performance and lead to biased predictions. When weights are applied incorrectly during training or validation, the model's understanding of the true data distribution becomes distorted.

```python
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Wrong way: arbitrary weight assignment
def wrong_weight_handling(X, y):
    # Arbitrarily increasing weights for minority class
    weights = np.ones(len(y))
    weights[y == 1] = 10  # Arbitrary weight boost
    
    model = RandomForestClassifier()
    model.fit(X, y, sample_weight=weights)
    return model

# Correct way: balanced weight calculation
def proper_weight_handling(X, y):
    # Calculate balanced weights based on class distribution
    class_weights = dict(zip(
        np.unique(y),
        len(y) / (len(np.unique(y)) * np.bincount(y))
    ))
    
    model = RandomForestClassifier(class_weight=class_weights)
    model.fit(X, y)
    return model
```

Slide 7: Correlation vs Causation Fallacy

A common pitfall in ML is confusing correlation with causation, leading to models that capture spurious relationships. This implementation demonstrates how seemingly correlated features might not have any causal relationship with the target variable.

```python
import numpy as np
import pandas as pd
from scipy import stats

# Generate synthetic data with spurious correlation
np.random.seed(42)
n_samples = 1000

# Two completely independent processes
time = np.linspace(0, 100, n_samples)
ice_cream_sales = 1000 + 100 * np.sin(time/10) + np.random.normal(0, 10, n_samples)
shark_attacks = 10 + 5 * np.sin(time/10) + np.random.normal(0, 1, n_samples)

# Calculate correlation
correlation = stats.pearsonr(ice_cream_sales, shark_attacks)

print(f"Correlation coefficient: {correlation[0]:.3f}")
print(f"P-value: {correlation[1]:.3e}")

# Demonstrate why this is misleading
df = pd.DataFrame({
    'ice_cream_sales': ice_cream_sales,
    'shark_attacks': shark_attacks,
    'temperature': 25 + 10 * np.sin(time/10) + np.random.normal(0, 2, n_samples)
})

# The real causal factor (temperature) affects both variables
print("\nPartial correlations controlling for temperature:")
partial_corr = df.pcorr()
print(partial_corr['ice_cream_sales']['shark_attacks'])
```

Slide 8: Hyperparameter Tuning Contamination

Incorrect hyperparameter optimization can lead to severe overfitting when validation data leaks into the tuning process. This implementation shows the dramatic difference between proper and improper tuning approaches.

```python
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Wrong way: using test data in hyperparameter tuning
def contaminated_tuning(X_train, X_test, y_train, y_test):
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30]
    }
    
    # Wrong: Including test data in the search
    X_all = np.vstack([X_train, X_test])
    y_all = np.concatenate([y_train, y_test])
    
    grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5)
    grid_search.fit(X_all, y_all)  # Data leakage!
    return grid_search.best_params_

# Correct way: nested cross-validation
def proper_tuning(X_train, y_train):
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30]
    }
    
    # Outer CV for performance estimation
    outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)
    # Inner CV for hyperparameter tuning
    inner_cv = KFold(n_splits=3, shuffle=True, random_state=42)
    
    clf = RandomForestClassifier()
    grid_search = GridSearchCV(clf, param_grid, cv=inner_cv)
    
    nested_scores = cross_val_score(grid_search, X_train, y_train, cv=outer_cv)
    return nested_scores.mean(), nested_scores.std()
```

Slide 9: Imbalanced Data Mishandling

Improper handling of imbalanced datasets leads to deceptive model performance metrics and poor minority class prediction. This implementation demonstrates common pitfalls and their solutions.

```python
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# Wrong way: naive accuracy metrics on imbalanced data
def wrong_imbalanced_handling(X, y):
    model = RandomForestClassifier()
    model.fit(X, y)
    
    # Misleading accuracy on imbalanced data
    y_pred = model.predict(X)
    print("Naive accuracy:", (y_pred == y).mean())  # Could be high but meaningless

# Correct way: proper resampling and metrics
def proper_imbalanced_handling(X, y):
    # Create pipeline with SMOTE and classifier
    pipeline = ImbPipeline([
        ('sampler', SMOTE(random_state=42)),
        ('classifier', RandomForestClassifier())
    ])
    
    # Use stratified k-fold CV
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Evaluate using appropriate metrics
    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_val)
        
        print("\nClassification Report:")
        print(classification_report(y_val, y_pred))
```

Slide 10: Training-Serving Skew

Training-serving skew occurs when the model's training environment differs from the production environment, leading to unexpected behavior. This implementation demonstrates how subtle differences in data preprocessing can cause significant performance degradation.

```python
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

# Wrong way: inconsistent preprocessing
def wrong_preprocessing_pipeline():
    # Training time
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    model.fit(X_train_scaled, y_train)
    
    # Saving model without scaler parameters
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    # Serving time - different scaling!
    new_scaler = StandardScaler()  # Wrong: New scaler parameters
    X_prod_scaled = new_scaler.fit_transform(X_prod)  # Wrong: Fitting on prod data

# Correct way: consistent preprocessing pipeline
class PreprocessingPipeline:
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = None
    
    def fit(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        self.model = RandomForestClassifier()
        self.model.fit(X_scaled, y)
    
    def predict(self, X):
        X_scaled = self.scaler.transform(X)  # Using same scaling parameters
        return self.model.predict(X_scaled)
    
    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)
```

Slide 11: Feature Engineering Data Leakage

Feature engineering operations that incorporate future information create unrealistic model performance. This implementation shows how temporal features must be carefully constructed to avoid look-ahead bias.

```python
import pandas as pd
import numpy as np

# Wrong way: future-looking features
def wrong_feature_engineering(df):
    # Wrong: Using future information to create features
    df['rolling_mean'] = df['value'].rolling(window=7, center=True).mean()  # Uses future values
    df['global_mean'] = df['value'].mean()  # Uses all data including future
    return df

# Correct way: time-aware feature engineering
def proper_feature_engineering(df):
    # Use expanding window instead of centered rolling
    df['rolling_mean'] = df['value'].expanding(min_periods=1).mean()
    
    # Calculate cumulative features
    df['cumulative_mean'] = df['value'].expanding().mean()
    df['cumulative_std'] = df['value'].expanding().std()
    
    # Lag features (using only past information)
    for lag in [1, 7, 30]:
        df[f'lag_{lag}'] = df['value'].shift(lag)
    
    return df

# Example usage with time series data
dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
df = pd.DataFrame({
    'date': dates,
    'value': np.random.normal(0, 1, len(dates))
}).set_index('date')

correct_features = proper_feature_engineering(df.copy())
```

Slide 12: Cross-Validation in Time Series Forecasting

Implementing cross-validation for time series data requires special attention to temporal dependencies. Using traditional cross-validation methods leads to data leakage and unrealistic performance estimates.

```python
from sklearn.model_selection import TimeSeriesSplit
import numpy as np

class TimeSeriesValidator:
    def __init__(self, n_splits=5, gap=0):
        self.tscv = TimeSeriesSplit(n_splits=n_splits, gap=gap)
        self.performances = []
    
    def validate(self, X, y, model):
        forecasts = []
        actuals = []
        
        for train_idx, test_idx in self.tscv.split(X):
            # Ensure temporal order
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # Train model only on historical data
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Store results
            forecasts.extend(y_pred)
            actuals.extend(y_test)
            
            # Calculate performance metrics
            mae = np.mean(np.abs(y_test - y_pred))
            self.performances.append(mae)
        
        return {
            'mae': np.mean(self.performances),
            'mae_std': np.std(self.performances),
            'forecasts': forecasts,
            'actuals': actuals
        }

# Example usage
validator = TimeSeriesValidator(n_splits=5, gap=7)  # 7-day gap between train/test
results = validator.validate(X, y, model)
```

Slide 13: Results Misrepresentation

Improper reporting of model results can lead to misleading conclusions about model performance. This implementation demonstrates how to properly evaluate and present model results.

```python
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

class ModelEvaluator:
    def __init__(self, model, X_train, X_test, y_train, y_test):
        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
    def evaluate(self):
        # Train performance
        train_pred = self.model.predict(self.X_train)
        train_proba = self.model.predict_proba(self.X_train)
        
        # Test performance
        test_pred = self.model.predict(self.X_test)
        test_proba = self.model.predict_proba(self.X_test)
        
        return {
            'train': {
                'confusion_matrix': confusion_matrix(self.y_train, train_pred),
                'classification_report': classification_report(self.y_train, train_pred),
                'predictions': train_pred,
                'probabilities': train_proba
            },
            'test': {
                'confusion_matrix': confusion_matrix(self.y_test, test_pred),
                'classification_report': classification_report(self.y_test, test_pred),
                'predictions': test_pred,
                'probabilities': test_proba
            }
        }
    
    def plot_results(self, results):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot confusion matrices
        sns.heatmap(results['train']['confusion_matrix'], annot=True, ax=ax1)
        ax1.set_title('Training Confusion Matrix')
        
        sns.heatmap(results['test']['confusion_matrix'], annot=True, ax=ax2)
        ax2.set_title('Test Confusion Matrix')
        
        plt.tight_layout()
        return fig

# Example usage
evaluator = ModelEvaluator(model, X_train, X_test, y_train, y_test)
results = evaluator.evaluate()
fig = evaluator.plot_results(results)
```

Slide 14: Additional Resources

*   Papers Related to ML Pitfalls and Best Practices:
    *   [https://arxiv.org/abs/2003.08119](https://arxiv.org/abs/2003.08119) "Common Pitfalls and Best Practices in Machine Learning Research"
    *   [https://arxiv.org/abs/1906.00900](https://arxiv.org/abs/1906.00900) "A Survey on Data Collection for Machine Learning"
    *   [https://arxiv.org/abs/1811.12808](https://arxiv.org/abs/1811.12808) "Hidden Technical Debt in Machine Learning Systems"
*   Recommended Reading:
    *   Google's ML Best Practices: [https://developers.google.com/machine-learning/guides/rules-of-ml](https://developers.google.com/machine-learning/guides/rules-of-ml)
    *   Microsoft's Responsible AI Principles: [https://www.microsoft.com/en-us/ai/responsible-ai](https://www.microsoft.com/en-us/ai/responsible-ai)
    *   Papers With Code Reproducibility Guide: [https://paperswithcode.com/reproducibility-checklist](https://paperswithcode.com/reproducibility-checklist)

