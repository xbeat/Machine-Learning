## Ensemble Learning Combining Models for Robust Predictions
Slide 1: Introduction to Ensemble Learning Fundamentals

Ensemble learning combines multiple models to create a more robust prediction system. This fundamental approach leverages the concept of model aggregation, where individual weak learners collaborate to form a stronger predictive model that outperforms single models in accuracy and reliability.

```python
# Basic ensemble framework
class EnsembleModel:
    def __init__(self, models):
        self.models = models
    
    def predict(self, X):
        # Get predictions from all models
        predictions = np.array([model.predict(X) for model in self.models])
        # Return majority vote for classification
        return np.mean(predictions, axis=0)

# Example usage
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification

# Generate sample data
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2)

# Create base models
base_models = [DecisionTreeClassifier(max_depth=3) for _ in range(3)]
ensemble = EnsembleModel(base_models)
```

Slide 2: Bootstrap Aggregating (Bagging) Implementation

Bagging involves training multiple models on different bootstrap samples of the original dataset. This technique reduces variance and helps prevent overfitting by creating diverse training sets through random sampling with replacement.

```python
import numpy as np
from sklearn.tree import DecisionTreeClassifier

class BaggingClassifier:
    def __init__(self, n_estimators=10, sample_size=1.0):
        self.n_estimators = n_estimators
        self.sample_size = sample_size
        self.models = []
        
    def fit(self, X, y):
        n_samples = int(len(X) * self.sample_size)
        
        for _ in range(self.n_estimators):
            # Bootstrap sampling
            indices = np.random.choice(len(X), size=n_samples, replace=True)
            X_sample = X[indices]
            y_sample = y[indices]
            
            # Train model on bootstrap sample
            model = DecisionTreeClassifier()
            model.fit(X_sample, y_sample)
            self.models.append(model)
            
    def predict(self, X):
        predictions = np.array([model.predict(X) for model in self.models])
        return np.mean(predictions, axis=0)
```

Slide 3: Advanced Bagging with Out-of-Bag Error Estimation

Out-of-Bag (OOB) error estimation provides an unbiased estimate of the generalization error without requiring a separate validation set. This technique utilizes samples not used during the bootstrap process for each base model.

```python
class BaggingWithOOB:
    def __init__(self, n_estimators=10):
        self.n_estimators = n_estimators
        self.models = []
        self.oob_score_ = None
    
    def fit(self, X, y):
        n_samples = len(X)
        predictions = np.zeros((n_samples,))
        n_predictions = np.zeros((n_samples,))
        
        for _ in range(self.n_estimators):
            # Bootstrap sampling
            indices = np.random.choice(n_samples, n_samples, replace=True)
            oob_indices = list(set(range(n_samples)) - set(indices))
            
            model = DecisionTreeClassifier()
            model.fit(X[indices], y[indices])
            self.models.append(model)
            
            # OOB predictions
            if len(oob_indices) > 0:
                predictions[oob_indices] += model.predict(X[oob_indices])
                n_predictions[oob_indices] += 1
                
        # Calculate OOB score
        valid_indices = n_predictions > 0
        self.oob_score_ = np.mean((predictions[valid_indices] / 
                                 n_predictions[valid_indices]) == y[valid_indices])
```

Slide 4: Gradient Boosting Implementation

Gradient Boosting builds an ensemble by training each new model to correct the errors of previous models. This sequential approach focuses on reducing the residual errors through gradient descent optimization.

```python
import numpy as np
from sklearn.tree import DecisionTreeRegressor

class GradientBoostingRegressor:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.models = []
        
    def fit(self, X, y):
        # Initialize predictions with zeros
        current_predictions = np.zeros_like(y)
        
        for _ in range(self.n_estimators):
            # Calculate residuals
            residuals = y - current_predictions
            
            # Fit a new model on residuals
            model = DecisionTreeRegressor(max_depth=self.max_depth)
            model.fit(X, residuals)
            
            # Update predictions
            current_predictions += self.learning_rate * model.predict(X)
            self.models.append(model)
            
    def predict(self, X):
        predictions = np.zeros(len(X))
        for model in self.models:
            predictions += self.learning_rate * model.predict(X)
        return predictions
```

Slide 5: AdaBoost Implementation from Scratch

AdaBoost (Adaptive Boosting) iteratively adjusts instance weights based on classification errors. This algorithm gives more importance to misclassified samples in subsequent iterations, forcing the model to focus on harder examples.

```python
import numpy as np
from sklearn.tree import DecisionTreeClassifier

class AdaBoostClassifier:
    def __init__(self, n_estimators=50):
        self.n_estimators = n_estimators
        self.models = []
        self.alphas = []
        
    def fit(self, X, y):
        n_samples = len(X)
        weights = np.ones(n_samples) / n_samples
        
        for _ in range(self.n_estimators):
            # Train weak learner
            model = DecisionTreeClassifier(max_depth=1)
            model.fit(X, y, sample_weight=weights)
            predictions = model.predict(X)
            
            # Calculate weighted error
            error = np.sum(weights * (predictions != y))
            alpha = 0.5 * np.log((1 - error) / error)
            
            # Update weights
            weights *= np.exp(-alpha * y * predictions)
            weights /= np.sum(weights)
            
            self.models.append(model)
            self.alphas.append(alpha)
    
    def predict(self, X):
        predictions = sum(alpha * model.predict(X) 
                        for alpha, model in zip(self.alphas, self.models))
        return np.sign(predictions)
```

Slide 6: Stacking Ensemble Implementation

Stacking combines multiple base models by training a meta-model on their predictions. This advanced ensemble technique learns the optimal way to weight each model's contribution based on their performance patterns.

```python
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression

class StackingEnsemble:
    def __init__(self, base_models, meta_model=None, n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model or LogisticRegression()
        self.n_folds = n_folds
        
    def fit(self, X, y):
        # Generate meta-features
        meta_features = np.zeros((X.shape[0], len(self.base_models)))
        kf = KFold(n_splits=self.n_folds, shuffle=True)
        
        # Train base models and create meta-features
        for i, model in enumerate(self.base_models):
            for train_idx, val_idx in kf.split(X):
                # Train on fold
                model.fit(X[train_idx], y[train_idx])
                # Predict on validation fold
                meta_features[val_idx, i] = model.predict_proba(X[val_idx])[:, 1]
        
        # Train meta model
        self.meta_model.fit(meta_features, y)
        
        # Retrain base models on full dataset
        for model in self.base_models:
            model.fit(X, y)
            
    def predict(self, X):
        meta_features = np.column_stack([
            model.predict_proba(X)[:, 1] for model in self.base_models
        ])
        return self.meta_model.predict(meta_features)
```

Slide 7: Real-world Application - Credit Card Fraud Detection

Ensemble learning excels in fraud detection due to its ability to capture complex patterns and handle imbalanced datasets. This implementation combines multiple models to achieve robust fraud detection capabilities.

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Load and preprocess data
def prepare_fraud_detection_data(df):
    # Scale numerical features
    scaler = StandardScaler()
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    
    # Split features and target
    X = df.drop('Class', axis=1)
    y = df['Class']
    return train_test_split(X, y, test_size=0.2, stratify=y)

# Create ensemble for fraud detection
class FraudDetectionEnsemble:
    def __init__(self):
        self.rf = RandomForestClassifier(n_estimators=100)
        self.gb = GradientBoostingClassifier(n_estimators=100)
        self.ada = AdaBoostClassifier(n_estimators=50)
        
    def fit(self, X, y):
        self.rf.fit(X, y)
        self.gb.fit(X, y)
        self.ada.fit(X, y)
        
    def predict_proba(self, X):
        predictions = np.column_stack([
            self.rf.predict_proba(X)[:, 1],
            self.gb.predict_proba(X)[:, 1],
            self.ada.predict_proba(X)[:, 1]
        ])
        return np.mean(predictions, axis=1)

# Usage example
X_train, X_test, y_train, y_test = prepare_fraud_detection_data(fraud_df)
model = FraudDetectionEnsemble()
model.fit(X_train, y_train)
y_pred = (model.predict_proba(X_test) > 0.5).astype(int)
print(classification_report(y_test, y_pred))
```

Slide 8: Voting Ensemble with Weighted Decisions

Voting ensembles combine predictions from multiple models through majority voting or weighted averaging. This implementation allows for both hard and soft voting with customizable model weights based on individual model performance.

```python
class WeightedVotingEnsemble:
    def __init__(self, models, weights=None, voting='soft'):
        self.models = models
        self.weights = weights if weights else [1] * len(models)
        self.voting = voting
        
    def fit(self, X, y):
        # Train all base models
        for model in self.models:
            model.fit(X, y)
            
    def predict(self, X):
        if self.voting == 'hard':
            predictions = np.array([model.predict(X) for model in self.models])
            weighted_votes = np.zeros_like(predictions[0])
            
            for pred, weight in zip(predictions, self.weights):
                weighted_votes += weight * (pred == 1)
                
            return (weighted_votes >= sum(self.weights)/2).astype(int)
        else:
            probas = np.array([model.predict_proba(X) for model in self.models])
            avg_proba = np.average(probas, weights=self.weights, axis=0)
            return (avg_proba[:, 1] >= 0.5).astype(int)

# Example usage
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

# Create base models
models = [
    SVC(probability=True),
    RandomForestClassifier(n_estimators=100),
    GaussianNB()
]

# Initialize ensemble with custom weights
weights = [0.4, 0.4, 0.2]  # Based on model performance
ensemble = WeightedVotingEnsemble(models, weights=weights, voting='soft')
```

Slide 9: Time Series Ensemble with Historical Window

Time series forecasting benefits from ensemble methods by combining multiple models that capture different temporal patterns. This implementation uses a sliding window approach with various base models.

```python
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor

class TimeSeriesEnsemble:
    def __init__(self, window_size=10):
        self.window_size = window_size
        self.models = {
            'linear': Ridge(),
            'rf': RandomForestRegressor(n_estimators=100),
            'ma': None  # Moving average doesn't require training
        }
        
    def create_sequences(self, data):
        X, y = [], []
        for i in range(len(data) - self.window_size):
            X.append(data[i:(i + self.window_size)])
            y.append(data[i + self.window_size])
        return np.array(X), np.array(y)
    
    def fit(self, data):
        X, y = self.create_sequences(data)
        
        # Train models
        self.models['linear'].fit(X, y)
        self.models['rf'].fit(X, y)
        
    def predict(self, data):
        if len(data) < self.window_size:
            raise ValueError("Insufficient data for prediction")
            
        X = data[-self.window_size:].reshape(1, -1)
        
        # Get predictions from each model
        predictions = {
            'linear': self.models['linear'].predict(X)[0],
            'rf': self.models['rf'].predict(X)[0],
            'ma': np.mean(data[-self.window_size:])
        }
        
        # Combine predictions with weighted average
        weights = {'linear': 0.3, 'rf': 0.5, 'ma': 0.2}
        final_prediction = sum(pred * weights[model] 
                             for model, pred in predictions.items())
        
        return final_prediction

# Example usage
import numpy as np

# Generate sample time series data
np.random.seed(42)
time_series = np.cumsum(np.random.normal(0, 1, 1000))

# Train and predict
model = TimeSeriesEnsemble(window_size=10)
train_size = int(len(time_series) * 0.8)
model.fit(time_series[:train_size])

# Make predictions
predictions = []
for i in range(train_size, len(time_series) - 10):
    pred = model.predict(time_series[i:i+10])
    predictions.append(pred)
```

Slide 10: Advanced Gradient Boosting with Loss Functions

Gradient boosting can be customized with different loss functions to optimize for specific objectives. This implementation demonstrates how to incorporate custom loss functions and their gradients into the boosting process.

```python
import numpy as np
from scipy.special import expit  # For logistic function

class AdvancedGradientBoosting:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.models = []
        
    def _logistic_loss(self, y_true, y_pred):
        """Custom logistic loss function"""
        y_pred = expit(y_pred)  # Convert to probabilities
        return -np.mean(y_true * np.log(y_pred) + 
                       (1 - y_true) * np.log(1 - y_pred))
    
    def _gradient(self, y_true, y_pred):
        """Compute gradient of logistic loss"""
        y_pred = expit(y_pred)
        return y_pred - y_true
    
    def fit(self, X, y):
        # Initialize predictions
        F = np.zeros(len(y))
        
        for _ in range(self.n_estimators):
            # Calculate negative gradient
            gradient = -self._gradient(y, F)
            
            # Fit base model to gradient
            model = DecisionTreeRegressor(max_depth=self.max_depth)
            model.fit(X, gradient)
            
            # Update predictions
            update = self.learning_rate * model.predict(X)
            F += update
            
            self.models.append(model)
            
    def predict_proba(self, X):
        # Get raw scores
        F = np.zeros(len(X))
        for model in self.models:
            F += self.learning_rate * model.predict(X)
        # Convert to probabilities
        return expit(F)

# Example usage with custom evaluation
def evaluate_model(y_true, y_pred_proba, threshold=0.5):
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {'precision': precision, 'recall': recall, 'f1': f1}
```

Slide 11: Random Forest with Feature Importance Analysis

This implementation extends the random forest algorithm to include detailed feature importance analysis and visualization capabilities, helping identify the most influential variables in the model.

```python
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

class AdvancedRandomForest:
    def __init__(self, n_estimators=100, max_features='sqrt', max_depth=None):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.max_depth = max_depth
        self.trees = []
        self.feature_importance_ = None
        
    def _bootstrap_sample(self, X, y):
        n_samples = X.shape[0]
        idxs = np.random.choice(n_samples, size=n_samples, replace=True)
        return X[idxs], y[idxs]
    
    def fit(self, X, y):
        self.n_features_ = X.shape[1]
        self.feature_importance_ = np.zeros(self.n_features_)
        
        if self.max_features == 'sqrt':
            self.max_features_ = int(np.sqrt(self.n_features_))
        
        for _ in range(self.n_estimators):
            # Bootstrap sample
            X_sample, y_sample = self._bootstrap_sample(X, y)
            
            # Train tree
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                max_features=self.max_features_
            )
            tree.fit(X_sample, y_sample)
            
            # Accumulate feature importance
            self.feature_importance_ += tree.feature_importances_
            self.trees.append(tree)
            
        # Normalize feature importance
        self.feature_importance_ /= self.n_estimators
        
    def plot_feature_importance(self, feature_names=None, top_n=10):
        if feature_names is None:
            feature_names = [f'Feature {i}' for i in range(self.n_features_)]
            
        # Sort importances
        indices = np.argsort(self.feature_importance_)[::-1][:top_n]
        
        plt.figure(figsize=(10, 6))
        plt.title('Feature Importances')
        plt.bar(range(top_n), 
                self.feature_importance_[indices],
                align='center')
        plt.xticks(range(top_n), 
                  [feature_names[i] for i in indices], 
                  rotation=45)
        plt.tight_layout()
        return plt.gcf()

# Example usage with feature importance analysis
X = np.random.randn(1000, 20)  # 20 features
y = (X[:, 0] + X[:, 1] * 2 + np.random.randn(1000) > 0).astype(int)

model = AdvancedRandomForest(n_estimators=100)
model.fit(X, y)

# Plot feature importance
feature_names = [f'Feature_{i}' for i in range(20)]
model.plot_feature_importance(feature_names=feature_names)
```

Slide 12: Real-world Application - Credit Risk Assessment

Credit risk assessment requires robust prediction models that can handle complex relationships in financial data. This ensemble implementation combines multiple models to predict credit default probability.

```python
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, precision_recall_curve

class CreditRiskEnsemble:
    def __init__(self):
        self.scalers = {}
        self.models = {
            'rf': RandomForestClassifier(n_estimators=200, class_weight='balanced'),
            'gb': GradientBoostingClassifier(n_estimators=100, learning_rate=0.1),
            'lr': LogisticRegression(class_weight='balanced')
        }
        self.weights = {'rf': 0.4, 'gb': 0.4, 'lr': 0.2}
        
    def preprocess_features(self, X, train=True):
        numeric_features = X.select_dtypes(include=['float64', 'int64']).columns
        
        if train:
            for feature in numeric_features:
                self.scalers[feature] = StandardScaler()
                X[feature] = self.scalers[feature].fit_transform(X[[feature]])
        else:
            for feature in numeric_features:
                X[feature] = self.scalers[feature].transform(X[[feature]])
        
        return X
    
    def fit(self, X, y):
        X_processed = self.preprocess_features(X.copy(), train=True)
        
        # Train each model
        for name, model in self.models.items():
            model.fit(X_processed, y)
            
    def predict_proba(self, X):
        X_processed = self.preprocess_features(X.copy(), train=False)
        predictions = np.zeros((len(X), 2))
        
        for name, model in self.models.items():
            predictions += self.weights[name] * model.predict_proba(X_processed)
            
        return predictions / sum(self.weights.values())
    
    def evaluate(self, X, y):
        y_pred_proba = self.predict_proba(X)[:, 1]
        
        # Calculate metrics
        auc_score = roc_auc_score(y, y_pred_proba)
        precisions, recalls, thresholds = precision_recall_curve(y, y_pred_proba)
        
        # Find optimal threshold
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx]
        
        return {
            'auc_score': auc_score,
            'optimal_threshold': optimal_threshold,
            'best_f1': f1_scores[optimal_idx],
            'precision_at_best': precisions[optimal_idx],
            'recall_at_best': recalls[optimal_idx]
        }

# Example usage
def load_and_prepare_credit_data(df):
    # Prepare features
    categorical_features = df.select_dtypes(include=['object']).columns
    df = pd.get_dummies(df, columns=categorical_features)
    
    # Split target
    X = df.drop('default', axis=1)
    y = df['default']
    
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Train and evaluate
X_train, X_test, y_train, y_test = load_and_prepare_credit_data(credit_df)
model = CreditRiskEnsemble()
model.fit(X_train, y_train)
metrics = model.evaluate(X_test, y_test)
```

Slide 13: Extreme Gradient Boosting with Early Stopping

This implementation showcases an advanced XGBoost-style gradient boosting approach with early stopping and custom evaluation metrics for optimal model performance.

```python
class ExtremeGradientBoosting:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3,
                 early_stopping_rounds=10, eval_metric='logloss'):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.early_stopping_rounds = early_stopping_rounds
        self.eval_metric = eval_metric
        self.models = []
        self.best_iteration = None
        
    def _calculate_metric(self, y_true, y_pred):
        if self.eval_metric == 'logloss':
            pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
            return -np.mean(y_true * np.log(pred) + 
                          (1 - y_true) * np.log(1 - pred))
        elif self.eval_metric == 'rmse':
            return np.sqrt(np.mean((y_true - y_pred) ** 2))
            
    def fit(self, X, y, eval_set=None):
        F = np.zeros(len(y))
        best_score = float('inf')
        rounds_without_improve = 0
        
        eval_scores = []
        if eval_set is not None:
            X_val, y_val = eval_set
            F_val = np.zeros(len(y_val))
        
        for i in range(self.n_estimators):
            # Calculate gradients
            grad = self._calculate_gradients(y, F)
            
            # Train base learner
            model = DecisionTreeRegressor(max_depth=self.max_depth)
            model.fit(X, grad)
            
            # Update predictions
            update = self.learning_rate * model.predict(X)
            F += update
            self.models.append(model)
            
            # Early stopping check
            if eval_set is not None:
                F_val += self.learning_rate * model.predict(X_val)
                val_score = self._calculate_metric(y_val, self._sigmoid(F_val))
                eval_scores.append(val_score)
                
                if val_score < best_score:
                    best_score = val_score
                    self.best_iteration = i
                    rounds_without_improve = 0
                else:
                    rounds_without_improve += 1
                    
                if rounds_without_improve >= self.early_stopping_rounds:
                    self.models = self.models[:self.best_iteration + 1]
                    break
    
    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def _calculate_gradients(self, y_true, F):
        y_pred = self._sigmoid(F)
        return y_pred - y_true
    
    def predict_proba(self, X):
        F = np.zeros(len(X))
        for model in self.models:
            F += self.learning_rate * model.predict(X)
        proba = self._sigmoid(F)
        return np.vstack([1 - proba, proba]).T

# Example usage with early stopping
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
model = ExtremeGradientBoosting(early_stopping_rounds=5)
model.fit(X_train, y_train, eval_set=(X_val, y_val))
```

Slide 14: Additional Resources

*   "XGBoost: A Scalable Tree Boosting System"
    *   [https://arxiv.org/abs/1603.02754](https://arxiv.org/abs/1603.02754)
*   "LightGBM: A Highly Efficient Gradient Boosting Decision Tree"
    *   [https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree](https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree)
*   "Deep Forest: Towards an Alternative to Deep Neural Networks"
    *   [https://arxiv.org/abs/1702.08835](https://arxiv.org/abs/1702.08835)
*   "On Calibration of Modern Neural Networks"
    *   [https://arxiv.org/abs/1706.04599](https://arxiv.org/abs/1706.04599)
*   "Gradient Boosting Machines: A Tutorial"
    *   Search on Google Scholar for comprehensive tutorials on gradient boosting machines

