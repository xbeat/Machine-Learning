## Evaluating Model Generalization with Cross-Validation

Slide 1: Introduction to Cross-Validation

Cross-validation is a statistical method used to assess model performance by partitioning data into training and testing sets. It helps evaluate how well a model generalizes to unseen data and reduces overfitting by providing multiple evaluation rounds with different data splits.

```python
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Generate sample data
X = np.random.rand(100, 1)
y = 2 * X + np.random.randn(100, 1) * 0.1

# Initialize K-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Store scores
mse_scores = []

# Perform cross-validation
for train_idx, test_idx in kf.split(X):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse_scores.append(mean_squared_error(y_test, y_pred))

print(f"Average MSE: {np.mean(mse_scores):.4f}")
```

Slide 2: K-Fold Cross-Validation Implementation

K-fold cross-validation divides the dataset into k equal-sized folds, using k-1 folds for training and one fold for testing. This process repeats k times, with each fold serving as the test set once, ensuring robust model evaluation across different data splits.

```python
class CustomKFold:
    def __init__(self, n_splits=5, shuffle=True, random_state=None):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
    
    def split(self, X):
        n_samples = len(X)
        indices = np.arange(n_samples)
        
        if self.shuffle:
            np.random.seed(self.random_state)
            np.random.shuffle(indices)
            
        fold_sizes = np.full(self.n_splits, n_samples // self.n_splits)
        fold_sizes[:n_samples % self.n_splits] += 1
        current = 0
        
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            test_indices = indices[start:stop]
            train_indices = np.concatenate([indices[:start], indices[stop:]])
            yield train_indices, test_indices
            current = stop
```

Slide 3: Leave-One-Out Cross-Validation

Leave-One-Out Cross-Validation (LOOCV) is a special case where k equals the number of samples. Each iteration uses a single observation for validation and the remaining observations for training, providing unbiased performance estimates but with high computational cost.

```python
from sklearn.model_selection import LeaveOneOut
import pandas as pd

# Generate sample data
X = np.random.rand(20, 1)
y = 2 * X + np.random.randn(20, 1) * 0.1

loo = LeaveOneOut()
mse_scores = []

for train_idx, test_idx in loo.split(X):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse_scores.append(mean_squared_error(y_test, y_pred))

print(f"LOOCV Average MSE: {np.mean(mse_scores):.4f}")
```

Slide 4: Stratified Cross-Validation

Stratified cross-validation maintains the same proportion of samples for each class in the training and testing sets, crucial for imbalanced datasets. This technique ensures representative sampling across all folds while preserving class distribution.

```python
from sklearn.model_selection import StratifiedKFold
from sklearn.datasets import make_classification

# Generate imbalanced classification data
X, y = make_classification(n_samples=1000, n_classes=2, weights=[0.9, 0.1],
                         random_state=42)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_class_distributions = []

for train_idx, test_idx in skf.split(X, y):
    y_train, y_test = y[train_idx], y[test_idx]
    
    # Calculate class distribution
    train_dist = np.bincount(y_train) / len(y_train)
    test_dist = np.bincount(y_test) / len(y_test)
    
    fold_class_distributions.append({
        'train_dist': train_dist,
        'test_dist': test_dist
    })

# Display distributions
for i, dist in enumerate(fold_class_distributions):
    print(f"Fold {i+1}:")
    print(f"Training distribution: {dist['train_dist']}")
    print(f"Testing distribution: {dist['test_dist']}\n")
```

Slide 5: Time Series Cross-Validation

Time series cross-validation respects temporal ordering by using past observations for training and future observations for testing. This approach prevents data leakage and provides realistic performance estimates for time-dependent predictions.

```python
from sklearn.model_selection import TimeSeriesSplit
import pandas as pd

# Generate time series data
dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
values = np.sin(np.arange(100) * 0.1) + np.random.randn(100) * 0.1
ts_data = pd.Series(values, index=dates)

tscv = TimeSeriesSplit(n_splits=5)
mse_scores = []

for train_idx, test_idx in tscv.split(ts_data):
    X_train = ts_data.iloc[train_idx]
    X_test = ts_data.iloc[test_idx]
    
    # Simple moving average prediction
    window_size = 3
    y_pred = X_train.rolling(window=window_size).mean().iloc[-1]
    mse = mean_squared_error([X_test.mean()], [y_pred])
    mse_scores.append(mse)

print(f"Time Series CV Average MSE: {np.mean(mse_scores):.4f}")
```

Slide 6: Cross-Validation with Hyperparameter Tuning

Nested cross-validation combines model selection and performance estimation by using an inner loop for hyperparameter optimization and an outer loop for unbiased performance assessment, preventing optimization bias.

```python
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.svm import SVC

# Generate classification data
X, y = make_classification(n_samples=200, random_state=42)

# Define parameter grid
param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']
}

# Outer cross-validation
cv_outer = KFold(n_splits=5, shuffle=True, random_state=42)
nested_scores = []

for train_idx, test_idx in cv_outer.split(X):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    # Inner cross-validation for model selection
    clf = GridSearchCV(SVC(), param_grid, cv=3)
    clf.fit(X_train, y_train)
    
    # Evaluate best model
    score = clf.score(X_test, y_test)
    nested_scores.append(score)

print(f"Nested CV Average Score: {np.mean(nested_scores):.4f}")
```

Slide 7: Real-world Example - Credit Card Fraud Detection

This example demonstrates cross-validation in detecting credit card fraud, where class imbalance is a significant challenge. We implement stratified cross-validation with SMOTE oversampling to handle imbalanced classes effectively.

```python
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier

# Simulate credit card transaction data
n_samples = 10000
n_features = 10
fraud_ratio = 0.01

X = np.random.randn(n_samples, n_features)
y = np.random.choice([0, 1], size=n_samples, p=[1-fraud_ratio, fraud_ratio])

# Preprocessing
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Initialize stratified cross-validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = []

for train_idx, test_idx in skf.split(X_scaled, y):
    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    # Apply SMOTE
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    
    # Train and evaluate
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train_balanced, y_train_balanced)
    scores.append(clf.score(X_test, y_test))

print(f"Average Balanced Accuracy: {np.mean(scores):.4f}")
```

Slide 8: Cross-Validation Metrics Implementation

A comprehensive implementation of various cross-validation metrics helps in model evaluation. This implementation includes accuracy, precision, recall, F1-score, and ROC-AUC calculations across all folds.

```python
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

class CrossValidationMetrics:
    def __init__(self, model, X, y, cv=5):
        self.model = model
        self.X = X
        self.y = y
        self.cv = cv
        self.metrics = {}
    
    def calculate_metrics(self):
        cv_splits = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=42)
        
        for fold, (train_idx, test_idx) in enumerate(cv_splits.split(self.X, self.y)):
            X_train, X_test = self.X[train_idx], self.X[test_idx]
            y_train, y_test = self.y[train_idx], self.y[test_idx]
            
            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(X_test)
            y_prob = self.model.predict_proba(X_test)[:, 1]
            
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_test, y_pred, average='binary'
            )
            auc = roc_auc_score(y_test, y_prob)
            
            self.metrics[f'fold_{fold+1}'] = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'auc': auc
            }
        
        return self.metrics

# Example usage
clf = RandomForestClassifier(random_state=42)
cv_metrics = CrossValidationMetrics(clf, X_scaled, y)
metrics = cv_metrics.calculate_metrics()

# Print average metrics
avg_metrics = pd.DataFrame(metrics).mean(axis=1)
print("\nAverage Metrics Across Folds:")
print(avg_metrics)
```

Slide 9: Cross-Validation for Deep Learning

Cross-validation in deep learning requires special consideration due to computational costs and memory constraints. This implementation shows how to perform k-fold cross-validation with a neural network using TensorFlow/Keras.

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

def create_model():
    model = Sequential([
        Dense(64, activation='relu', input_shape=(10,)),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam',
                 loss='binary_crossentropy',
                 metrics=['accuracy'])
    return model

# Generate synthetic data
X = np.random.randn(1000, 10)
y = (X.sum(axis=1) > 0).astype(int)

# Initialize cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
histories = []

for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    model = create_model()
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )
    
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping],
        verbose=0
    )
    
    histories.append(history.history)

# Calculate average performance
avg_val_acc = np.mean([h['val_accuracy'][-1] for h in histories])
print(f"Average Validation Accuracy: {avg_val_acc:.4f}")
```

Slide 10: Bootstrap Cross-Validation

Bootstrap cross-validation uses random sampling with replacement to create training sets, providing an alternative approach to traditional k-fold cross-validation for uncertainty estimation and model evaluation.

```python
class BootstrapCV:
    def __init__(self, n_iterations=100, sample_size=0.8, random_state=None):
        self.n_iterations = n_iterations
        self.sample_size = sample_size
        self.random_state = random_state
        np.random.seed(random_state)
    
    def split(self, X):
        n_samples = len(X)
        sample_size = int(n_samples * self.sample_size)
        
        for _ in range(self.n_iterations):
            # Sample with replacement
            train_indices = np.random.choice(
                n_samples, size=sample_size, replace=True
            )
            # Out-of-bag samples
            test_indices = np.array(list(
                set(range(n_samples)) - set(train_indices)
            ))
            
            yield train_indices, test_indices

# Example usage
X = np.random.randn(100, 2)
y = (X[:, 0] + X[:, 1] > 0).astype(int)

bootstrap_cv = BootstrapCV(n_iterations=50, random_state=42)
scores = []

for train_idx, test_idx in bootstrap_cv.split(X):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    model = LogisticRegression()
    model.fit(X_train, y_train)
    scores.append(model.score(X_test, y_test))

print(f"Bootstrap CV Average Score: {np.mean(scores):.4f}")
print(f"95% Confidence Interval: ({np.percentile(scores, 2.5):.4f}, "
      f"{np.percentile(scores, 97.5):.4f})")
```

Slide 11: Cross-Validation for Feature Selection

Cross-validation in feature selection ensures robust feature importance assessment by evaluating feature stability across different data splits. This implementation combines recursive feature elimination with cross-validation to identify the most reliable predictive features.

```python
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestClassifier

# Generate sample data with irrelevant features
X = np.random.randn(200, 20)  # 20 features
y = (X[:, 0] + X[:, 1] > 0).astype(int)  # Only first 2 features are relevant

# Initialize estimator and RFECV
estimator = RandomForestClassifier(random_state=42)
selector = RFECV(
    estimator=estimator,
    step=1,
    cv=5,
    scoring='accuracy',
    min_features_to_select=1
)

# Fit selector
selector.fit(X, y)

# Get selected features
selected_features = np.arange(X.shape[1])[selector.support_]
feature_scores = selector.grid_scores_

print(f"Optimal number of features: {selector.n_features_}")
print(f"Selected feature indices: {selected_features}")
print(f"CV scores per feature count: {feature_scores}")
```

Slide 12: Real-world Example - Housing Price Prediction

This example demonstrates cross-validation in a regression context, using housing price data to show how different preprocessing steps and feature engineering techniques affect model performance across folds.

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
import numpy as np

# Generate synthetic housing data
n_samples = 1000
numeric_features = np.random.randn(n_samples, 3)  # price, size, age
categorical_features = np.random.choice(['urban', 'suburban', 'rural'], 
                                     size=(n_samples, 1))
y = 2 * numeric_features[:, 0] + numeric_features[:, 1] - 0.5 * numeric_features[:, 2] + \
    np.random.randn(n_samples) * 0.1

# Create preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), [0, 1, 2]),
        ('cat', OneHotEncoder(drop='first'), [3])
    ])

# Cross-validation with preprocessing
cv = KFold(n_splits=5, shuffle=True, random_state=42)
mse_scores = []

for train_idx, test_idx in cv.split(numeric_features):
    # Split data
    X_train_num = numeric_features[train_idx]
    X_test_num = numeric_features[test_idx]
    X_train_cat = categorical_features[train_idx]
    X_test_cat = categorical_features[test_idx]
    
    # Combine features
    X_train = np.hstack([X_train_num, X_train_cat])
    X_test = np.hstack([X_test_num, X_test_cat])
    
    y_train, y_test = y[train_idx], y[test_idx]
    
    # Preprocess and fit
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train_processed, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_processed)
    mse = mean_squared_error(y_test, y_pred)
    mse_scores.append(mse)

print(f"Average MSE: {np.mean(mse_scores):.4f}")
print(f"Standard deviation of MSE: {np.std(mse_scores):.4f}")
```

Slide 13: Additional Resources

1.  "A Survey of Cross-Validation Procedures for Model Selection" [https://arxiv.org/abs/0907.4728](https://arxiv.org/abs/0907.4728)
2.  "Cross-Validation Strategies for Data with Temporal, Spatial, Hierarchical, or Network Structure" [https://arxiv.org/abs/2007.04663](https://arxiv.org/abs/2007.04663)
3.  "Nested Cross-Validation When Selecting Classifiers is Overzealous for Most Practical Applications" [https://arxiv.org/abs/2003.07139](https://arxiv.org/abs/2003.07139)
4.  "A Systematic Analysis of Performance Measures for Classification Tasks" [https://arxiv.org/abs/1909.03622](https://arxiv.org/abs/1909.03622)
5.  "Cross-validation: what does it estimate and how well does it do it?" [https://arxiv.org/abs/2104.00673](https://arxiv.org/abs/2104.00673)

