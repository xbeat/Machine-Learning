## Cross-Validation Techniques for Reliable Model Prediction
Slide 1: Cross-Validation Fundamentals

Cross-validation is a statistical method for assessing model performance and generalization ability by partitioning data into training and testing sets. This implementation demonstrates the basic framework for performing cross-validation using scikit-learn compatible estimators.

```python
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from typing import Tuple

def basic_cross_validation(X: np.ndarray, y: np.ndarray, 
                         test_size: float = 0.2, 
                         random_state: int = 42) -> Tuple[float, float]:
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Example using a simple classifier (replace with any sklearn estimator)
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(random_state=random_state)
    
    # Train and evaluate
    model.fit(X_train, y_train)
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    return train_score, test_score

# Example usage
X = np.random.randn(1000, 10)  # 1000 samples, 10 features
y = np.random.randint(0, 2, 1000)  # Binary classification
train_acc, test_acc = basic_cross_validation(X, y)
print(f"Training accuracy: {train_acc:.3f}")
print(f"Testing accuracy: {test_acc:.3f}")
```

Slide 2: Leave-One-Out Cross-Validation Implementation

Leave-One-Out Cross-Validation (LOOCV) is an exhaustive method where each data point serves as the test set exactly once. This implementation showcases the manual computation of LOOCV without using sklearn's built-in functions.

```python
import numpy as np
from sklearn.base import clone
from sklearn.linear_model import LinearRegression

def manual_loocv(X: np.ndarray, y: np.ndarray, model) -> float:
    n_samples = X.shape[0]
    errors = np.zeros(n_samples)
    
    for i in range(n_samples):
        # Create train/test indices
        test_idx = [i]
        train_idx = list(set(range(n_samples)) - {i})
        
        # Split data
        X_train = X[train_idx]
        X_test = X[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]
        
        # Train model and predict
        model_clone = clone(model)
        model_clone.fit(X_train, y_train)
        pred = model_clone.predict(X_test)
        
        # Calculate error
        errors[i] = (pred - y_test) ** 2
        
    return np.mean(errors)  # Return MSE

# Example usage
X = np.random.randn(100, 3)
y = 2*X[:, 0] + 3*X[:, 1] - X[:, 2] + np.random.randn(100)*0.1
model = LinearRegression()
mse = manual_loocv(X, y, model)
print(f"LOOCV Mean Squared Error: {mse:.6f}")
```

Slide 3: K-Fold Cross-Validation Implementation

K-Fold Cross-Validation divides the dataset into k equal-sized folds, using each fold as a test set once. This implementation demonstrates how to manually perform k-fold cross-validation while tracking various performance metrics.

```python
import numpy as np
from sklearn.base import clone
from sklearn.metrics import mean_squared_error, r2_score
from typing import Dict, List

def k_fold_cv(X: np.ndarray, y: np.ndarray, 
              model, k: int = 5) -> Dict[str, List[float]]:
    n_samples = X.shape[0]
    fold_size = n_samples // k
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    
    metrics = {
        'mse': [],
        'r2': [],
        'fold': []
    }
    
    for fold in range(k):
        # Create fold indices
        start_idx = fold * fold_size
        end_idx = start_idx + fold_size
        test_idx = indices[start_idx:end_idx]
        train_idx = np.concatenate([
            indices[:start_idx],
            indices[end_idx:]
        ])
        
        # Split data
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Train and evaluate
        model_clone = clone(model)
        model_clone.fit(X_train, y_train)
        y_pred = model_clone.predict(X_test)
        
        # Calculate metrics
        metrics['mse'].append(mean_squared_error(y_test, y_pred))
        metrics['r2'].append(r2_score(y_test, y_pred))
        metrics['fold'].append(fold)
    
    return metrics

# Example usage
from sklearn.linear_model import Ridge
X = np.random.randn(1000, 5)
y = np.sum(X, axis=1) + np.random.randn(1000) * 0.1
model = Ridge(alpha=1.0)
results = k_fold_cv(X, y, model, k=5)

print("K-Fold Cross-Validation Results:")
for fold in range(5):
    print(f"Fold {fold+1}:")
    print(f"  MSE: {results['mse'][fold]:.6f}")
    print(f"  R2: {results['r2'][fold]:.6f}")
```

Slide 4: Stratified K-Fold Cross-Validation

Stratified K-Fold maintains the same proportion of samples for each class as in the whole dataset, which is crucial for imbalanced classification problems. This implementation shows how to perform stratified k-fold cross-validation manually.

```python
import numpy as np
from collections import Counter
from sklearn.base import clone
from sklearn.metrics import accuracy_score

def stratified_k_fold(X: np.ndarray, y: np.ndarray, 
                     model, k: int = 5) -> dict:
    # Get class distributions
    class_counts = Counter(y)
    class_indices = {c: np.where(y == c)[0] for c in class_counts}
    
    # Calculate samples per fold for each class
    fold_sizes = {c: len(indices) // k 
                 for c, indices in class_indices.items()}
    
    results = {'accuracy': [], 'fold': []}
    
    for fold in range(k):
        test_idx = []
        # Get stratified test indices
        for class_label, indices in class_indices.items():
            start_idx = fold * fold_sizes[class_label]
            end_idx = start_idx + fold_sizes[class_label]
            test_idx.extend(indices[start_idx:end_idx])
        
        # Create train indices
        all_idx = set(range(len(y)))
        train_idx = list(all_idx - set(test_idx))
        
        # Split data
        X_train = X[train_idx]
        X_test = X[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]
        
        # Train and evaluate
        model_clone = clone(model)
        model_clone.fit(X_train, y_train)
        y_pred = model_clone.predict(X_test)
        
        # Store results
        results['accuracy'].append(accuracy_score(y_test, y_pred))
        results['fold'].append(fold)
    
    return results

# Example usage
from sklearn.svm import SVC
# Create imbalanced dataset
X = np.random.randn(1000, 4)
y = np.concatenate([np.zeros(800), np.ones(200)])
model = SVC(kernel='rbf', random_state=42)
results = stratified_k_fold(X, y, model, k=5)

print("Stratified K-Fold Results:")
for fold in range(5):
    print(f"Fold {fold+1} Accuracy: {results['accuracy'][fold]:.4f}")
print(f"Mean Accuracy: {np.mean(results['accuracy']):.4f}")
```

Slide 5: Time Series Cross-Validation

Time series cross-validation requires maintaining temporal order of observations. This implementation demonstrates a rolling window approach where future data points are never used to predict past values.

```python
import numpy as np
from typing import Tuple, List
from sklearn.metrics import mean_squared_error

def time_series_cv(X: np.ndarray, y: np.ndarray, 
                  window_size: int, 
                  horizon: int = 1) -> List[Tuple[np.ndarray, np.ndarray]]:
    
    n_samples = len(X)
    splits = []
    
    # Generate splits maintaining temporal order
    for i in range(window_size, n_samples - horizon + 1):
        train_idx = range(i - window_size, i)
        test_idx = range(i, i + horizon)
        splits.append((train_idx, test_idx))
    
    return splits

def rolling_window_cv(X: np.ndarray, y: np.ndarray, 
                     model, window_size: int) -> dict:
    results = {
        'test_predictions': [],
        'test_actual': [],
        'mse': []
    }
    
    splits = time_series_cv(X, y, window_size)
    
    for train_idx, test_idx in splits:
        # Split data
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Fit and predict
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Store results
        results['test_predictions'].extend(y_pred)
        results['test_actual'].extend(y_test)
        results['mse'].append(mean_squared_error(y_test, y_pred))
    
    return results

# Example usage
from sklearn.linear_model import LinearRegression
# Generate time series data
t = np.linspace(0, 100, 1000)
X = np.column_stack([t, np.sin(0.1*t)])
y = 2*np.sin(0.1*t) + 0.5*t + np.random.randn(1000)*0.1

model = LinearRegression()
results = rolling_window_cv(X, y, model, window_size=100)

print(f"Average MSE: {np.mean(results['mse']):.6f}")
print(f"MSE Standard Deviation: {np.std(results['mse']):.6f}")
```

Slide 6: Cross-Validation Performance Metrics

This implementation focuses on computing comprehensive performance metrics across different cross-validation folds, including precision, recall, F1-score, and ROC curves for classification problems.

```python
import numpy as np
from sklearn.metrics import (precision_score, recall_score, 
                           f1_score, roc_curve, auc)
from sklearn.model_selection import KFold
from typing import Dict, List

class CVMetrics:
    def __init__(self, X: np.ndarray, y: np.ndarray, 
                 model, n_folds: int = 5):
        self.X = X
        self.y = y
        self.model = model
        self.n_folds = n_folds
        self.metrics = self._compute_metrics()
    
    def _compute_metrics(self) -> Dict[str, List[float]]:
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        metrics = {
            'precision': [],
            'recall': [],
            'f1': [],
            'roc_auc': [],
            'fpr': [],
            'tpr': []
        }
        
        for train_idx, test_idx in kf.split(self.X):
            X_train, X_test = self.X[train_idx], self.X[test_idx]
            y_train, y_test = self.y[train_idx], self.y[test_idx]
            
            # Train model
            self.model.fit(X_train, y_train)
            
            # Get predictions and probabilities
            y_pred = self.model.predict(X_test)
            y_prob = self.model.predict_proba(X_test)[:, 1]
            
            # Compute metrics
            metrics['precision'].append(precision_score(y_test, y_pred))
            metrics['recall'].append(recall_score(y_test, y_pred))
            metrics['f1'].append(f1_score(y_test, y_pred))
            
            # Compute ROC curve
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            metrics['roc_auc'].append(auc(fpr, tpr))
            metrics['fpr'].append(fpr)
            metrics['tpr'].append(tpr)
        
        return metrics
    
    def print_summary(self):
        print("Cross-Validation Performance Metrics:")
        for metric in ['precision', 'recall', 'f1', 'roc_auc']:
            values = self.metrics[metric]
            print(f"{metric.upper()}:")
            print(f"  Mean: {np.mean(values):.4f}")
            print(f"  Std: {np.std(values):.4f}")
            print(f"  Min: {np.min(values):.4f}")
            print(f"  Max: {np.max(values):.4f}")

# Example usage
from sklearn.ensemble import RandomForestClassifier
# Generate binary classification dataset
X = np.random.randn(1000, 10)
y = (X[:, 0] + X[:, 1] > 0).astype(int)

model = RandomForestClassifier(n_estimators=100, random_state=42)
cv_metrics = CVMetrics(X, y, model)
cv_metrics.print_summary()
```

Slide 7: Nested Cross-Validation

Nested cross-validation implements an inner loop for model selection and an outer loop for model assessment, providing unbiased estimation of the model's generalization error.

```python
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from typing import Dict, List, Tuple

def nested_cv(X: np.ndarray, y: np.ndarray, 
             model_class, param_grid: Dict,
             outer_splits: int = 5, 
             inner_splits: int = 3) -> Dict:
    
    outer_cv = KFold(n_splits=outer_splits, shuffle=True, random_state=42)
    results = {
        'outer_scores': [],
        'best_params': [],
        'inner_scores': []
    }
    
    for outer_train_idx, outer_test_idx in outer_cv.split(X):
        X_train_outer = X[outer_train_idx]
        y_train_outer = y[outer_train_idx]
        X_test_outer = X[outer_test_idx]
        y_test_outer = y[outer_test_idx]
        
        # Inner cross-validation for model selection
        inner_cv = KFold(n_splits=inner_splits, shuffle=True, random_state=42)
        best_score = float('inf')
        best_params = None
        inner_scores = []
        
        for params in _param_grid_iterator(param_grid):
            cv_scores = []
            for inner_train_idx, inner_val_idx in inner_cv.split(X_train_outer):
                X_train_inner = X_train_outer[inner_train_idx]
                y_train_inner = y_train_outer[inner_train_idx]
                X_val_inner = X_train_outer[inner_val_idx]
                y_val_inner = y_train_outer[inner_val_idx]
                
                # Train model with current parameters
                model = model_class(**params)
                model.fit(X_train_inner, y_train_inner)
                y_pred_inner = model.predict(X_val_inner)
                score = mean_squared_error(y_val_inner, y_pred_inner)
                cv_scores.append(score)
            
            mean_score = np.mean(cv_scores)
            inner_scores.append(mean_score)
            
            if mean_score < best_score:
                best_score = mean_score
                best_params = params
        
        # Train final model with best parameters
        final_model = model_class(**best_params)
        final_model.fit(X_train_outer, y_train_outer)
        y_pred_outer = final_model.predict(X_test_outer)
        outer_score = mean_squared_error(y_test_outer, y_pred_outer)
        
        results['outer_scores'].append(outer_score)
        results['best_params'].append(best_params)
        results['inner_scores'].append(inner_scores)
    
    return results

def _param_grid_iterator(param_grid: Dict) -> List[Dict]:
    """Helper function to iterate over parameter grid"""
    import itertools
    keys = param_grid.keys()
    values = param_grid.values()
    for instance in itertools.product(*values):
        yield dict(zip(keys, instance))

# Example usage
from sklearn.ensemble import RandomForestRegressor
# Generate regression dataset
X = np.random.randn(500, 5)
y = np.sum(X**2, axis=1) + np.random.randn(500) * 0.1

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7],
    'min_samples_split': [2, 5]
}

results = nested_cv(X, y, RandomForestRegressor, param_grid)

print("Nested Cross-Validation Results:")
print(f"Mean Outer Score (MSE): {np.mean(results['outer_scores']):.6f}")
print("\nBest Parameters per Outer Fold:")
for i, params in enumerate(results['best_params']):
    print(f"Fold {i+1}: {params}")
```

Slide 8: Bootstrap Cross-Validation

Bootstrap cross-validation uses random sampling with replacement to create multiple training sets. This implementation demonstrates the bootstrap technique with error estimation and confidence intervals.

```python
import numpy as np
from sklearn.base import clone
from typing import Dict, List
from sklearn.metrics import mean_squared_error
import scipy.stats as stats

class BootstrapCV:
    def __init__(self, model, n_bootstrap: int = 1000):
        self.model = model
        self.n_bootstrap = n_bootstrap
        self.bootstrap_scores = []
        self.bootstrap_predictions = []
        
    def fit_predict(self, X: np.ndarray, y: np.ndarray) -> Dict:
        n_samples = X.shape[0]
        
        for _ in range(self.n_bootstrap):
            # Generate bootstrap indices
            bootstrap_indices = np.random.choice(
                n_samples, size=n_samples, replace=True
            )
            oob_indices = list(set(range(n_samples)) - set(bootstrap_indices))
            
            # Split data
            X_boot = X[bootstrap_indices]
            y_boot = y[bootstrap_indices]
            X_oob = X[oob_indices]
            y_oob = y[oob_indices]
            
            # Train model
            model_clone = clone(self.model)
            model_clone.fit(X_boot, y_boot)
            
            # Make predictions on out-of-bag samples
            y_pred = model_clone.predict(X_oob)
            
            # Store results
            self.bootstrap_scores.append(
                mean_squared_error(y_oob, y_pred)
            )
            self.bootstrap_predictions.append((oob_indices, y_pred))
            
        return self._compute_statistics()
    
    def _compute_statistics(self) -> Dict:
        scores = np.array(self.bootstrap_scores)
        ci = stats.norm.interval(
            0.95, 
            loc=scores.mean(), 
            scale=scores.std()
        )
        
        return {
            'mean_mse': scores.mean(),
            'std_mse': scores.std(),
            'ci_lower': ci[0],
            'ci_upper': ci[1],
            'bootstrap_scores': scores
        }

# Example usage
from sklearn.linear_model import ElasticNet

# Generate dataset
np.random.seed(42)
X = np.random.randn(1000, 10)
y = 2*X[:, 0] + 3*X[:, 1] - X[:, 2] + np.random.randn(1000)*0.1

# Initialize model and bootstrap CV
model = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)
bootstrap_cv = BootstrapCV(model, n_bootstrap=1000)
results = bootstrap_cv.fit_predict(X, y)

print("Bootstrap Cross-Validation Results:")
print(f"Mean MSE: {results['mean_mse']:.6f}")
print(f"Standard Deviation: {results['std_mse']:.6f}")
print(f"95% Confidence Interval: [{results['ci_lower']:.6f}, {results['ci_upper']:.6f}]")
```

Slide 9: Cross-Validation for Time Series with Multiple Steps Ahead

This implementation focuses on evaluating model performance for multi-step time series forecasting while maintaining temporal dependencies and preventing data leakage.

```python
import numpy as np
from typing import Tuple, List, Dict
from sklearn.base import clone
from sklearn.metrics import mean_squared_error

class MultiStepTimeSeriesCV:
    def __init__(self, model, n_splits: int = 5, 
                 forecast_horizon: int = 3):
        self.model = model
        self.n_splits = n_splits
        self.forecast_horizon = forecast_horizon
        self.results = []
        
    def create_features(self, data: np.ndarray, 
                       lookback: int) -> Tuple[np.ndarray, np.ndarray]:
        """Create features using sliding window approach"""
        X, y = [], []
        for i in range(len(data) - lookback - self.forecast_horizon + 1):
            X.append(data[i:(i + lookback)])
            y.append(data[(i + lookback):(i + lookback + self.forecast_horizon)])
        return np.array(X), np.array(y)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict:
        n_samples = len(X)
        split_size = n_samples // self.n_splits
        
        for i in range(self.n_splits):
            start_idx = i * split_size
            end_idx = start_idx + split_size
            
            # Create train-test split
            X_train = X[:start_idx]
            y_train = y[:start_idx]
            X_test = X[start_idx:end_idx]
            y_test = y[start_idx:end_idx]
            
            if len(X_train) == 0:
                continue
                
            # Train model
            model_clone = clone(self.model)
            model_clone.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model_clone.predict(X_test)
            
            # Calculate errors for each forecast step
            errors = []
            for step in range(self.forecast_horizon):
                mse = mean_squared_error(
                    y_test[:, step], 
                    y_pred[:, step]
                )
                errors.append(mse)
            
            self.results.append({
                'fold': i,
                'errors': errors,
                'mean_error': np.mean(errors)
            })
            
        return self._summarize_results()
    
    def _summarize_results(self) -> Dict:
        all_errors = np.array([r['errors'] for r in self.results])
        mean_errors_per_step = np.mean(all_errors, axis=0)
        std_errors_per_step = np.std(all_errors, axis=0)
        
        return {
            'mean_errors_per_step': mean_errors_per_step,
            'std_errors_per_step': std_errors_per_step,
            'overall_mean_error': np.mean(mean_errors_per_step),
            'overall_std_error': np.mean(std_errors_per_step)
        }

# Example usage
from sklearn.neural_network import MLPRegressor

# Generate synthetic time series data
t = np.linspace(0, 100, 1000)
y = np.sin(0.1*t) + 0.5*np.sin(0.2*t) + np.random.randn(1000)*0.1

# Create features and targets
lookback = 10
X, y = MultiStepTimeSeriesCV(None).create_features(y, lookback)

# Initialize model and cross-validation
model = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000)
cv = MultiStepTimeSeriesCV(model, n_splits=5, forecast_horizon=3)
results = cv.evaluate(X, y)

print("\nMulti-step Time Series Cross-Validation Results:")
print("Mean Errors per Step:")
for step, (mean_err, std_err) in enumerate(zip(
    results['mean_errors_per_step'], 
    results['std_errors_per_step']
)):
    print(f"Step {step+1}: {mean_err:.6f} ± {std_err:.6f}")
print(f"\nOverall Mean Error: {results['overall_mean_error']:.6f}")
```

Slide 10: Cross-Validation for Imbalanced Datasets

This implementation focuses on handling imbalanced datasets using stratified sampling and specialized metrics, ensuring proper evaluation of model performance across minority and majority classes.

```python
import numpy as np
from collections import Counter
from sklearn.base import clone
from sklearn.metrics import (precision_recall_fscore_support,
                           confusion_matrix, roc_auc_score)
from typing import Dict, List

class ImbalancedCV:
    def __init__(self, model, n_splits: int = 5, 
                 min_class_size: int = None):
        self.model = model
        self.n_splits = n_splits
        self.min_class_size = min_class_size
        self.results = []
        
    def stratified_split(self, X: np.ndarray, y: np.ndarray) -> List[tuple]:
        """Create stratified folds maintaining class ratios"""
        class_indices = {c: np.where(y == c)[0] for c in np.unique(y)}
        fold_size = min(len(indices) for indices in class_indices.values()) // self.n_splits
        
        splits = []
        for fold in range(self.n_splits):
            test_idx = []
            for indices in class_indices.values():
                np.random.shuffle(indices)
                start_idx = fold * fold_size
                end_idx = start_idx + fold_size
                test_idx.extend(indices[start_idx:end_idx])
            
            train_idx = list(set(range(len(y))) - set(test_idx))
            splits.append((train_idx, test_idx))
            
        return splits
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict:
        splits = self.stratified_split(X, y)
        
        for fold, (train_idx, test_idx) in enumerate(splits):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Train model
            model_clone = clone(self.model)
            model_clone.fit(X_train, y_train)
            
            # Get predictions
            y_pred = model_clone.predict(X_test)
            y_prob = model_clone.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_test, y_pred, average=None
            )
            conf_matrix = confusion_matrix(y_test, y_pred)
            auc_score = roc_auc_score(y_test, y_prob)
            
            # Calculate per-class metrics
            class_metrics = {}
            for i, class_label in enumerate(np.unique(y)):
                tp = conf_matrix[i, i]
                fp = conf_matrix[:, i].sum() - tp
                fn = conf_matrix[i, :].sum() - tp
                tn = conf_matrix.sum() - (tp + fp + fn)
                
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                class_metrics[class_label] = {
                    'precision': precision[i],
                    'recall': recall[i],
                    'f1': f1[i],
                    'specificity': specificity
                }
            
            self.results.append({
                'fold': fold,
                'class_metrics': class_metrics,
                'auc': auc_score,
                'confusion_matrix': conf_matrix
            })
            
        return self._summarize_results()
    
    def _summarize_results(self) -> Dict:
        n_classes = len(self.results[0]['class_metrics'])
        summary = {
            'per_class': {},
            'overall': {
                'auc_mean': np.mean([r['auc'] for r in self.results]),
                'auc_std': np.std([r['auc'] for r in self.results])
            }
        }
        
        metrics = ['precision', 'recall', 'f1', 'specificity']
        for class_label in range(n_classes):
            class_summary = {}
            for metric in metrics:
                values = [r['class_metrics'][class_label][metric] 
                         for r in self.results]
                class_summary[f'{metric}_mean'] = np.mean(values)
                class_summary[f'{metric}_std'] = np.std(values)
            summary['per_class'][class_label] = class_summary
            
        return summary

# Example usage
from sklearn.ensemble import GradientBoostingClassifier

# Generate imbalanced dataset
np.random.seed(42)
n_samples = 1000
X = np.random.randn(n_samples, 5)
# Create imbalanced classes (90% class 0, 10% class 1)
y = np.zeros(n_samples)
y[:100] = 1

model = GradientBoostingClassifier(n_estimators=100, random_state=42)
cv = ImbalancedCV(model, n_splits=5)
results = cv.evaluate(X, y)

print("Imbalanced Dataset Cross-Validation Results:\n")
print("Overall AUC:")
print(f"Mean: {results['overall']['auc_mean']:.4f}")
print(f"Std: {results['overall']['auc_std']:.4f}\n")

for class_label, metrics in results['per_class'].items():
    print(f"Class {class_label} Metrics:")
    for metric in ['precision', 'recall', 'f1', 'specificity']:
        mean = metrics[f'{metric}_mean']
        std = metrics[f'{metric}_std']
        print(f"{metric.capitalize()}: {mean:.4f} ± {std:.4f}")
    print()
```

Slide 11: Cross-Validation with Feature Selection

This implementation combines cross-validation with feature selection, ensuring that feature selection is performed independently within each fold to prevent data leakage.

```python
import numpy as np
from sklearn.base import clone
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from typing import Dict, List, Tuple

class FeatureSelectionCV:
    def __init__(self, model, n_features: int, 
                 n_splits: int = 5):
        self.model = model
        self.n_features = n_features
        self.n_splits = n_splits
        self.results = []
        self.selected_features = []
        
    def select_features(self, X: np.ndarray, 
                       y: np.ndarray) -> Tuple[np.ndarray, List[int]]:
        """Select top k features using F-scores"""
        selector = SelectKBest(score_func=f_classif, k=self.n_features)
        X_selected = selector.fit_transform(X, y)
        selected_indices = np.where(selector.get_support())[0]
        return X_selected, selected_indices
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict:
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        
        for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Perform feature selection
            X_train_selected, selected_features = self.select_features(
                X_train, y_train
            )
            X_test_selected = X_test[:, selected_features]
            
            # Train model
            model_clone = clone(self.model)
            model_clone.fit(X_train_selected, y_train)
            
            # Evaluate
            train_score = model_clone.score(X_train_selected, y_train)
            test_score = model_clone.score(X_test_selected, y_test)
            
            self.results.append({
                'fold': fold,
                'train_score': train_score,
                'test_score': test_score,
                'selected_features': selected_features
            })
            self.selected_features.append(selected_features)
            
        return self._summarize_results()
    
    def _summarize_results(self) -> Dict:
        # Calculate feature selection stability
        feature_counts = np.zeros(X.shape[1])
        for features in self.selected_features:
            feature_counts[features] += 1
        stability = feature_counts / len(self.selected_features)
        
        return {
            'train_score_mean': np.mean([r['train_score'] 
                                       for r in self.results]),
            'train_score_std': np.std([r['train_score'] 
                                     for r in self.results]),
            'test_score_mean': np.mean([r['test_score'] 
                                      for r in self.results]),
            'test_score_std': np.std([r['test_score'] 
                                    for r in self.results]),
            'feature_stability': stability
        }

# Example usage
from sklearn.svm import SVC

# Generate dataset with irrelevant features
np.random.seed(42)
n_samples = 1000
n_features = 20
X = np.random.randn(n_samples, n_features)
# Only first 5 features are relevant
y = (X[:, :5].sum(axis=1) > 0).astype(int)

model = SVC(kernel='rbf', probability=True)
cv = FeatureSelectionCV(model, n_features=5, n_splits=5)
results = cv.evaluate(X, y)

print("Feature Selection Cross-Validation Results:\n")
print("Model Performance:")
print(f"Train Score: {results['train_score_mean']:.4f} ± {results['train_score_std']:.4f}")
print(f"Test Score: {results['test_score_mean']:.4f} ± {results['test_score_std']:.4f}\n")

print("Feature Stability (selection frequency):")
for i, stability in enumerate(results['feature_stability']):
    if stability > 0:
        print(f"Feature {i}: {stability:.2f}")
```

Slide 12: Custom Scoring Metrics for Cross-Validation

This implementation demonstrates how to create and use custom scoring metrics in cross-validation, particularly useful for domain-specific evaluation criteria.

```python
import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import KFold
from typing import Callable, Dict, List
from functools import partial

class CustomScoringCV:
    def __init__(self, model: BaseEstimator, metrics: Dict[str, Callable],
                 n_splits: int = 5):
        self.model = model
        self.metrics = metrics
        self.n_splits = n_splits
        self.results = []
        
    def custom_score(self, y_true: np.ndarray, y_pred: np.ndarray,
                    y_prob: np.ndarray = None) -> Dict[str, float]:
        """Calculate all custom metrics"""
        scores = {}
        for name, metric in self.metrics.items():
            if 'prob' in name.lower() and y_prob is not None:
                scores[name] = metric(y_true, y_prob)
            else:
                scores[name] = metric(y_true, y_pred)
        return scores
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict:
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        
        for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Train model
            model_clone = clone(self.model)
            model_clone.fit(X_train, y_train)
            
            # Get predictions
            y_pred = model_clone.predict(X_test)
            y_prob = None
            if hasattr(model_clone, 'predict_proba'):
                y_prob = model_clone.predict_proba(X_test)
            
            # Calculate metrics
            fold_scores = self.custom_score(y_test, y_pred, y_prob)
            fold_scores['fold'] = fold
            self.results.append(fold_scores)
            
        return self._summarize_results()
    
    def _summarize_results(self) -> Dict:
        summary = {}
        for metric in self.metrics.keys():
            values = [r[metric] for r in self.results]
            summary[f'{metric}_mean'] = np.mean(values)
            summary[f'{metric}_std'] = np.std(values)
        return summary

# Custom metric definitions
def weighted_accuracy(y_true: np.ndarray, y_pred: np.ndarray,
                     weights: Dict[int, float]) -> float:
    """Accuracy weighted by class importance"""
    correct = y_true == y_pred
    weighted_correct = sum(correct * weights[y] for y in np.unique(y_true))
    weighted_total = sum(weights[y] for y in y_true)
    return weighted_correct / weighted_total

def top_k_accuracy(y_true: np.ndarray, y_prob: np.ndarray,
                  k: int = 3) -> float:
    """Accuracy considering top k predictions"""
    top_k_pred = np.argsort(y_prob, axis=1)[:, -k:]
    return np.mean([y in pred for y, pred in zip(y_true, top_k_pred)])

# Example usage
from sklearn.ensemble import RandomForestClassifier

# Generate multi-class dataset
np.random.seed(42)
n_samples = 1000
n_features = 10
n_classes = 5
X = np.random.randn(n_samples, n_features)
y = np.random.randint(0, n_classes, n_samples)

# Define custom metrics
class_weights = {i: (i + 1) / n_classes for i in range(n_classes)}
metrics = {
    'weighted_acc': partial(weighted_accuracy, weights=class_weights),
    'top_3_acc': partial(top_k_accuracy, k=3),
}

model = RandomForestClassifier(n_estimators=100, random_state=42)
cv = CustomScoringCV(model, metrics)
results = cv.evaluate(X, y)

print("Custom Scoring Cross-Validation Results:\n")
for metric, mean_value in results.items():
    if 'mean' in metric:
        base_metric = metric.replace('_mean', '')
        std_value = results[base_metric + '_std']
        print(f"{base_metric}:")
        print(f"  Mean: {mean_value:.4f}")
        print(f"  Std:  {std_value:.4f}\n")
```

Slide 13: Cross-Validation for Model Calibration

This implementation focuses on evaluating and improving probability calibration through cross-validation, essential for applications requiring reliable probability estimates.

```python
import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.calibration import calibration_curve
from sklearn.model_selection import KFold
from typing import Dict, List, Tuple

class CalibrationCV:
    def __init__(self, model: BaseEstimator, n_splits: int = 5,
                 n_bins: int = 10):
        self.model = model
        self.n_splits = n_splits
        self.n_bins = n_bins
        self.results = []
        
    def compute_calibration_metrics(self, y_true: np.ndarray,
                                  y_prob: np.ndarray) -> Dict:
        """Compute calibration curve and related metrics"""
        prob_true, prob_pred = calibration_curve(
            y_true, y_prob, n_bins=self.n_bins
        )
        
        # Calculate Expected Calibration Error (ECE)
        ece = np.abs(prob_true - prob_pred).mean()
        
        # Calculate Maximum Calibration Error (MCE)
        mce = np.abs(prob_true - prob_pred).max()
        
        # Calculate Brier Score
        brier = np.mean((y_prob - y_true) ** 2)
        
        return {
            'prob_true': prob_true,
            'prob_pred': prob_pred,
            'ece': ece,
            'mce': mce,
            'brier': brier
        }
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict:
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        
        for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Train model
            model_clone = clone(self.model)
            model_clone.fit(X_train, y_train)
            
            # Get probability predictions
            y_prob = model_clone.predict_proba(X_test)[:, 1]
            
            # Calculate calibration metrics
            metrics = self.compute_calibration_metrics(y_test, y_prob)
            metrics['fold'] = fold
            self.results.append(metrics)
            
        return self._summarize_results()
    
    def _summarize_results(self) -> Dict:
        summary = {}
        metrics = ['ece', 'mce', 'brier']
        
        for metric in metrics:
            values = [r[metric] for r in self.results]
            summary[f'{metric}_mean'] = np.mean(values)
            summary[f'{metric}_std'] = np.std(values)
        
        # Average calibration curve
        prob_true = np.mean([r['prob_true'] for r in self.results], axis=0)
        prob_pred = np.mean([r['prob_pred'] for r in self.results], axis=0)
        summary['calibration_curve'] = {
            'prob_true': prob_true,
            'prob_pred': prob_pred
        }
        
        return summary

# Example usage
from sklearn.linear_model import LogisticRegression

# Generate binary classification dataset with probability structure
np.random.seed(42)
n_samples = 1000
X = np.random.randn(n_samples, 5)
# Create probabilities with some miscalibration
logits = X[:, 0] + 0.5 * X[:, 1]
probs = 1 / (1 + np.exp(-logits))
y = (probs > np.random.rand(n_samples)).astype(int)

model = LogisticRegression()
cv = CalibrationCV(model)
results = cv.evaluate(X, y)

print("Calibration Cross-Validation Results:\n")
print("Expected Calibration Error (ECE):")
print(f"  Mean: {results['ece_mean']:.4f}")
print(f"  Std:  {results['ece_std']:.4f}\n")

print("Maximum Calibration Error (MCE):")
print(f"  Mean: {results['mce_mean']:.4f}")
print(f"  Std:  {results['mce_std']:.4f}\n")

print("Brier Score:")
print(f"  Mean: {results['brier_mean']:.4f}")
print(f"  Std:  {results['brier_std']:.4f}")
```

Slide 14: Computational Efficiency in Cross-Validation

This implementation focuses on optimizing cross-validation for large datasets using parallel processing and efficient data handling techniques.

```python
import numpy as np
from sklearn.base import clone
from sklearn.model_selection import KFold
from typing import Dict, List
from concurrent.futures import ProcessPoolExecutor
import time
from functools import partial

class EfficientCV:
    def __init__(self, model, n_splits: int = 5, 
                 n_jobs: int = -1, chunk_size: int = None):
        self.model = model
        self.n_splits = n_splits
        self.n_jobs = n_jobs if n_jobs > 0 else None
        self.chunk_size = chunk_size
        self.results = []
        
    def _process_fold(self, fold_data: tuple) -> Dict:
        """Process a single fold"""
        fold, (train_idx, test_idx), X, y = fold_data
        
        # Extract fold data
        if self.chunk_size:
            X_train = self._process_in_chunks(X[train_idx])
            X_test = self._process_in_chunks(X[test_idx])
        else:
            X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        start_time = time.time()
        
        # Train model
        model_clone = clone(self.model)
        model_clone.fit(X_train, y_train)
        
        # Get predictions
        y_pred = model_clone.predict(X_test)
        train_score = model_clone.score(X_train, y_train)
        test_score = model_clone.score(X_test, y_test)
        
        processing_time = time.time() - start_time
        
        return {
            'fold': fold,
            'train_score': train_score,
            'test_score': test_score,
            'processing_time': processing_time,
            'train_size': len(train_idx),
            'test_size': len(test_idx)
        }
    
    def _process_in_chunks(self, X: np.ndarray) -> np.ndarray:
        """Process data in chunks to manage memory"""
        if self.chunk_size is None:
            return X
            
        chunks = []
        for i in range(0, len(X), self.chunk_size):
            chunk = X[i:i + self.chunk_size]
            chunks.append(chunk)
        return np.vstack(chunks)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict:
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        fold_data = [
            (fold, split, X, y) 
            for fold, split in enumerate(kf.split(X))
        ]
        
        if self.n_jobs != 1:
            # Parallel processing
            with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
                self.results = list(executor.map(self._process_fold, fold_data))
        else:
            # Sequential processing
            self.results = [self._process_fold(data) for data in fold_data]
            
        return self._summarize_results()
    
    def _summarize_results(self) -> Dict:
        summary = {
            'train_score_mean': np.mean([r['train_score'] 
                                       for r in self.results]),
            'train_score_std': np.std([r['train_score'] 
                                     for r in self.results]),
            'test_score_mean': np.mean([r['test_score'] 
                                      for r in self.results]),
            'test_score_std': np.std([r['test_score'] 
                                    for r in self.results]),
            'avg_processing_time': np.mean([r['processing_time'] 
                                          for r in self.results]),
            'total_processing_time': sum(r['processing_time'] 
                                       for r in self.results)
        }
        return summary

# Example usage
from sklearn.ensemble import RandomForestClassifier
import psutil

# Generate large dataset
np.random.seed(42)
n_samples = 100000
n_features = 50
X = np.random.randn(n_samples, n_features)
y = (X[:, 0] + X[:, 1] > 0).astype(int)

# Configure chunk size based on available memory
available_memory = psutil.virtual_memory().available
chunk_size = min(10000, n_samples // 10)  # Adjust based on your system

model = RandomForestClassifier(n_estimators=100, random_state=42)
cv = EfficientCV(model, n_splits=5, n_jobs=4, chunk_size=chunk_size)
results = cv.evaluate(X, y)

print("Efficient Cross-Validation Results:\n")
print("Model Performance:")
print(f"Train Score: {results['train_score_mean']:.4f} ± {results['train_score_std']:.4f}")
print(f"Test Score: {results['test_score_mean']:.4f} ± {results['test_score_std']:.4f}\n")

print("Processing Time Statistics:")
print(f"Average Time per Fold: {results['avg_processing_time']:.2f} seconds")
print(f"Total Processing Time: {results['total_processing_time']:.2f} seconds")
```

Slide 15: Additional Resources

1.  "A Survey of Cross-Validation Procedures for Model Selection"  
    [https://arxiv.org/abs/0907.4728](https://arxiv.org/abs/0907.4728)
2.  "Cross-Validation Strategies for Data with Temporal, Spatial, Hierarchical, or Phylogenetic Structure"  
    [https://arxiv.org/abs/1809.09121](https://arxiv.org/abs/1809.09121)
3.  "Understanding and Improving Cross-Validation for Deep Learning"  
    [https://arxiv.org/abs/1906.02267](https://arxiv.org/abs/1906.02267)
4.  "Nested Cross Validation When Selecting Classifiers is Overzealous for Most Practical Applications"  
    [https://arxiv.org/abs/2005.02114](https://arxiv.org/abs/2005.02114)
5.  "Bootstrap Methods for Time Series"  
    [https://arxiv.org/abs/1104.0831](https://arxiv.org/abs/1104.0831)

Note: These are example URLs. Since I cannot access external content, you should verify these URLs and papers independently.

