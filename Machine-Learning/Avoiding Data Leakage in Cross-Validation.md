## Avoiding Data Leakage in Cross-Validation
Slide 1: Cross-Validation Data Leakage

Data leakage occurs when information from the validation set influences model training, leading to overoptimistic performance estimates. This common pitfall happens when preprocessing steps like scaling or feature selection are performed before splitting the data.

```python
# Incorrect approach - data leakage
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import numpy as np

X = np.random.randn(1000, 20)
y = np.random.randint(0, 2, 1000)

# Wrong: scaling before splitting
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kf = KFold(n_splits=5)
for train_idx, val_idx in kf.split(X_scaled):
    X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
    # This leads to data leakage as validation data influenced scaling
```

Slide 2: Proper Cross-Validation Pipeline

A correct implementation ensures that data preprocessing occurs within each fold, maintaining strict separation between training and validation data to prevent information leakage and obtain unbiased performance estimates.

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

# Correct approach
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression())
])

# Preprocessing happens inside each fold
scores = cross_val_score(pipeline, X, y, cv=5)
print(f"Cross-validation scores: {scores}")
print(f"Mean CV score: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")
```

Slide 3: Stratification in Imbalanced Datasets

Imbalanced datasets require stratified sampling to maintain class distributions across folds. Regular k-fold cross-validation can lead to highly variable performance metrics when class proportions differ significantly between folds.

```python
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import numpy as np

# Create imbalanced dataset
X = np.random.randn(1000, 10)
y = np.concatenate([np.ones(50), np.zeros(950)])  # 5% positive class

# Stratified K-Fold maintains class proportions
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
f1_scores = []

for train_idx, val_idx in skf.split(X, y):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    # Check class distribution in each fold
    print(f"Training set class distribution: {np.bincount(y_train.astype(int))}")
    print(f"Validation set class distribution: {np.bincount(y_val.astype(int))}")
```

Slide 4: Time Series Cross-Validation

Traditional random cross-validation fails for time series data as it violates temporal dependencies. Time series split ensures that future data points are not used to predict past events.

```python
from sklearn.model_selection import TimeSeriesSplit
import pandas as pd
import numpy as np

# Generate time series data
dates = pd.date_range(start='2023-01-01', periods=1000, freq='D')
X = np.random.randn(1000, 5)
y = np.random.randn(1000)

tscv = TimeSeriesSplit(n_splits=5)
for train_idx, val_idx in tscv.split(X):
    print(f"Training set: {dates[train_idx[0]]} to {dates[train_idx[-1]]}")
    print(f"Validation set: {dates[val_idx[0]]} to {dates[val_idx[-1]]}\n")
```

Slide 5: Nested Cross-Validation

When performing hyperparameter tuning, nested cross-validation provides unbiased performance estimates by separating model selection from evaluation using inner and outer loops.

```python
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.svm import SVC

# Outer cross-validation loop
outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)
inner_cv = KFold(n_splits=3, shuffle=True, random_state=42)

# Parameter grid for tuning
param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['rbf', 'linear']
}

# Inner loop for model selection
clf = GridSearchCV(SVC(), param_grid, cv=inner_cv)

# Outer loop for performance estimation
nested_scores = cross_val_score(clf, X, y, cv=outer_cv)
print(f"Nested CV score: {nested_scores.mean():.3f} (+/- {nested_scores.std() * 2:.3f})")
```

Slide 6: Cross-Validation with Custom Splitting Criteria

Sometimes standard splitting strategies don't meet specific domain requirements. Custom splitters allow implementing business logic or domain-specific constraints while maintaining proper cross-validation principles.

```python
from sklearn.model_selection import BaseCrossValidator
import numpy as np

class CustomSplitter(BaseCrossValidator):
    def __init__(self, n_splits=5):
        self.n_splits = n_splits
        
    def split(self, X, y=None, groups=None):
        n_samples = len(X)
        # Custom logic: split based on data characteristics
        indices = np.arange(n_samples)
        
        for i in range(self.n_splits):
            # Example: exclude samples with specific characteristics
            mask = self._get_custom_mask(X, i)
            train_idx = indices[~mask]
            val_idx = indices[mask]
            yield train_idx, val_idx
            
    def _get_custom_mask(self, X, fold):
        # Implement custom splitting logic
        return np.random.rand(len(X)) < 0.2
    
    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

# Usage
custom_cv = CustomSplitter(n_splits=5)
scores = cross_val_score(pipeline, X, y, cv=custom_cv)
```

Slide 7: Group Cross-Validation

Handling grouped data requires special attention to prevent information leakage between related samples. Group k-fold ensures that samples from the same group stay together during splitting.

```python
from sklearn.model_selection import GroupKFold
import numpy as np

# Generate grouped data
X = np.random.randn(100, 4)
y = np.random.randint(0, 2, 100)
groups = np.repeat(range(20), 5)  # 20 groups, 5 samples each

group_cv = GroupKFold(n_splits=5)
for fold, (train_idx, val_idx) in enumerate(group_cv.split(X, y, groups)):
    # Verify group separation
    train_groups = set(groups[train_idx])
    val_groups = set(groups[val_idx])
    print(f"Fold {fold + 1}:")
    print(f"Training groups: {len(train_groups)}")
    print(f"Validation groups: {len(val_groups)}")
    print(f"Intersection: {train_groups.intersection(val_groups)}\n")
```

Slide 8: Memory Efficient Cross-Validation

Large datasets require memory-efficient cross-validation implementations. This approach uses generators and batch processing to handle datasets that don't fit in memory.

```python
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import KFold

class MemoryEfficientCV:
    def __init__(self, estimator, batch_size=1000):
        self.estimator = estimator
        self.batch_size = batch_size
        
    def batch_generator(self, X, y, indices):
        for i in range(0, len(indices), self.batch_size):
            batch_idx = indices[i:i + self.batch_size]
            yield X[batch_idx], y[batch_idx]
    
    def evaluate(self, X, y, cv=5):
        kf = KFold(n_splits=cv, shuffle=True)
        scores = []
        
        for train_idx, val_idx in kf.split(X):
            # Train model using batches
            for X_batch, y_batch in self.batch_generator(X, y, train_idx):
                self.estimator.partial_fit(X_batch, y_batch)
            
            # Evaluate using batches
            batch_scores = []
            for X_batch, y_batch in self.batch_generator(X, y, val_idx):
                score = self.estimator.score(X_batch, y_batch)
                batch_scores.append(score)
            
            scores.append(np.mean(batch_scores))
        
        return np.array(scores)
```

Slide 9: Cross-Validation for Time Series Forecasting

Time series forecasting requires specialized cross-validation techniques that preserve temporal order and handle multiple prediction horizons appropriately.

```python
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

class TimeSeriesForecaster(BaseEstimator, RegressorMixin):
    def __init__(self, window_size=10, horizon=5):
        self.window_size = window_size
        self.horizon = horizon
    
    def create_sequences(self, X):
        sequences = []
        targets = []
        for i in range(len(X) - self.window_size - self.horizon + 1):
            sequences.append(X[i:(i + self.window_size)])
            targets.append(X[i + self.window_size:i + self.window_size + self.horizon])
        return np.array(sequences), np.array(targets)
    
    def rolling_window_cv(self, X, n_splits=5):
        total_size = len(X)
        fold_size = total_size // n_splits
        
        for i in range(n_splits):
            train_end = total_size - (n_splits - i) * fold_size
            val_end = train_end + fold_size
            
            train_data = X[:train_end]
            val_data = X[train_end:val_end]
            
            X_train, y_train = self.create_sequences(train_data)
            X_val, y_val = self.create_sequences(val_data)
            
            yield X_train, X_val, y_train, y_val
```

Slide 10: Cross-Validation Performance Metrics

Cross-validation performance analysis requires careful consideration of appropriate metrics and their statistical properties. This implementation focuses on robust performance estimation.

```python
from sklearn.metrics import make_scorer
from scipy import stats
import numpy as np

class RobustCVMetrics:
    def __init__(self, cv_scores, confidence=0.95):
        self.cv_scores = np.array(cv_scores)
        self.confidence = confidence
    
    def compute_statistics(self):
        mean = np.mean(self.cv_scores)
        std = np.std(self.cv_scores, ddof=1)
        
        # Confidence interval using t-distribution
        n = len(self.cv_scores)
        t_value = stats.t.ppf((1 + self.confidence) / 2, n - 1)
        ci = t_value * std / np.sqrt(n)
        
        return {
            'mean': mean,
            'std': std,
            'ci_lower': mean - ci,
            'ci_upper': mean + ci,
            'cv_scores': self.cv_scores
        }
    
    def statistical_comparison(self, other_cv_scores):
        # Paired t-test between two sets of CV scores
        t_stat, p_value = stats.ttest_rel(self.cv_scores, other_cv_scores)
        return {'t_statistic': t_stat, 'p_value': p_value}

# Example usage
cv_results = RobustCVMetrics([0.85, 0.82, 0.87, 0.84, 0.86])
stats = cv_results.compute_statistics()
print(f"Mean CV score: {stats['mean']:.3f} ({stats['ci_lower']:.3f}, {stats['ci_upper']:.3f})")
```

Slide 11: Feature Selection in Cross-Validation

Feature selection must be performed independently within each cross-validation fold to prevent data leakage. This implementation shows how to properly integrate feature selection into the cross-validation pipeline.

```python
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

class NestedFeatureSelector:
    def __init__(self, n_features=10):
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('feature_selector', SelectKBest(score_func=f_classif, k=n_features)),
            ('classifier', RandomForestClassifier())
        ])
        
    def nested_cv_with_feature_selection(self, X, y, outer_cv, inner_cv):
        outer_scores = []
        selected_features_per_fold = []
        
        for train_idx, test_idx in outer_cv.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Fit pipeline on training data
            self.pipeline.fit(X_train, y_train)
            
            # Get selected feature indices
            selector = self.pipeline.named_steps['feature_selector']
            selected_features = np.where(selector.get_support())[0]
            selected_features_per_fold.append(selected_features)
            
            # Evaluate on test set
            score = self.pipeline.score(X_test, y_test)
            outer_scores.append(score)
        
        return {
            'cv_scores': outer_scores,
            'mean_score': np.mean(outer_scores),
            'std_score': np.std(outer_scores),
            'feature_stability': self._compute_feature_stability(selected_features_per_fold)
        }
    
    def _compute_feature_stability(self, selected_features_per_fold):
        # Calculate feature selection stability across folds
        n_folds = len(selected_features_per_fold)
        feature_counts = {}
        
        for fold_features in selected_features_per_fold:
            for feature in fold_features:
                feature_counts[feature] = feature_counts.get(feature, 0) + 1
        
        return {k: v/n_folds for k, v in feature_counts.items()}
```

Slide 12: Cross-Validation with Unbalanced Time Windows

Special handling is required when dealing with time series data that has varying importance across different time periods. This implementation addresses temporal weighting in cross-validation.

```python
import numpy as np
from sklearn.model_selection import BaseTimeSeriesSplitter

class WeightedTimeSeriesSplit:
    def __init__(self, n_splits=5, weight_function=None):
        self.n_splits = n_splits
        self.weight_function = weight_function or (lambda t: 1.0)
    
    def split(self, X, y=None):
        n_samples = len(X)
        indices = np.arange(n_samples)
        
        # Generate time-based weights
        weights = np.array([self.weight_function(t) for t in range(n_samples)])
        
        # Calculate split points based on cumulative weights
        total_weight = np.sum(weights)
        weight_per_fold = total_weight / self.n_splits
        cumsum_weights = np.cumsum(weights)
        
        split_points = []
        current_weight = weight_per_fold
        
        for _ in range(self.n_splits - 1):
            split_idx = np.searchsorted(cumsum_weights, current_weight)
            split_points.append(split_idx)
            current_weight += weight_per_fold
        
        # Generate train/test indices
        for i in range(len(split_points)):
            if i == 0:
                train_idx = indices[:split_points[i]]
                test_idx = indices[split_points[i]:split_points[i+1] if i+1 < len(split_points) else None]
            else:
                train_idx = indices[:split_points[i]]
                test_idx = indices[split_points[i]:split_points[i+1] if i+1 < len(split_points) else None]
            
            yield train_idx, test_idx
            
    def get_n_splits(self):
        return self.n_splits

# Example usage with exponential weighting
def exponential_weight(t, decay=0.1):
    return np.exp(-decay * t)

weighted_cv = WeightedTimeSeriesSplit(n_splits=5, 
                                    weight_function=exponential_weight)
```

Slide 13: Additional Resources

*   Efficient Cross-Validation for Neural Networks: [https://arxiv.org/abs/1904.01334](https://arxiv.org/abs/1904.01334)
*   Nested Cross-Validation When Selecting Classifiers is Overzealous for Most Practical Applications: [https://arxiv.org/abs/2005.12496](https://arxiv.org/abs/2005.12496)
*   A Survey on Cross-Validation: [https://arxiv.org/abs/1811.12808](https://arxiv.org/abs/1811.12808)
*   Time Series Cross-Validation Techniques: [http://www.google.com/search?q=time+series+cross+validation+techniques](http://www.google.com/search?q=time+series+cross+validation+techniques)
*   Feature Selection in Cross-Validation: [https://www.sciencedirect.com/topics/computer-science/cross-validation-feature-selection](https://www.sciencedirect.com/topics/computer-science/cross-validation-feature-selection)

