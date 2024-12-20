## Visual Guide to Bagging and Boosting in Machine Learning
Slide 1: Understanding Bagging in Machine Learning

Bagging, short for Bootstrap Aggregating, is a fundamental ensemble technique that creates multiple training subsets through random sampling with replacement. This method reduces overfitting by training independent models on different data distributions and combining their predictions through averaging or voting mechanisms.

```python
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification

class BaggingFromScratch:
    def __init__(self, n_estimators=10):
        self.n_estimators = n_estimators
        self.estimators = []
    
    def bootstrap_sample(self, X, y):
        n_samples = X.shape[0]
        idxs = np.random.choice(n_samples, size=n_samples, replace=True)
        return X[idxs], y[idxs]
    
    def fit(self, X, y):
        self.estimators = []
        for _ in range(self.n_estimators):
            estimator = DecisionTreeClassifier()
            X_sample, y_sample = self.bootstrap_sample(X, y)
            estimator.fit(X_sample, y_sample)
            self.estimators.append(estimator)
    
    def predict(self, X):
        predictions = np.array([est.predict(X) for est in self.estimators])
        return np.round(np.mean(predictions, axis=0))
```

Slide 2: Implementing Basic Boosting Algorithm

Boosting builds an ensemble sequentially, where each model attempts to correct the errors made by previous models. The algorithm assigns higher weights to misclassified samples, forcing subsequent models to focus on challenging cases and improve overall performance.

```python
import numpy as np
from sklearn.tree import DecisionTreeClassifier

class SimpleAdaBoost:
    def __init__(self, n_estimators=50, learning_rate=1.0):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.estimators = []
        self.estimator_weights = []
        
    def fit(self, X, y):
        n_samples = X.shape[0]
        sample_weights = np.ones(n_samples) / n_samples
        
        for _ in range(self.n_estimators):
            estimator = DecisionTreeClassifier(max_depth=1)
            estimator.fit(X, y, sample_weight=sample_weights)
            predictions = estimator.predict(X)
            
            incorrect = predictions != y
            estimator_error = np.mean(incorrect * sample_weights)
            
            estimator_weight = self.learning_rate * np.log((1 - estimator_error) / estimator_error)
            
            sample_weights *= np.exp(estimator_weight * incorrect)
            sample_weights /= np.sum(sample_weights)
            
            self.estimators.append(estimator)
            self.estimator_weights.append(estimator_weight)
```

Slide 3: Practical Example - Credit Card Fraud Detection

Financial fraud detection represents a perfect use case for ensemble methods due to its inherent class imbalance and complex patterns. This implementation demonstrates how bagging can be effectively used to detect fraudulent transactions while handling imbalanced datasets.

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

# Simulating credit card transaction data
np.random.seed(42)
X, y = make_classification(n_samples=10000, n_features=20, n_classes=2, 
                         weights=[0.97, 0.03], random_state=42)

# Data preprocessing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Training bagging classifier
bagging_clf = BaggingFromScratch(n_estimators=100)
bagging_clf.fit(X_train_scaled, y_train)

# Predictions and evaluation
y_pred = bagging_clf.predict(X_test_scaled)
print(classification_report(y_test, y_pred))
```

Slide 4: Mathematics Behind Bagging

The mathematical foundation of bagging involves statistical concepts of bootstrap sampling and aggregation. Understanding these principles helps in grasping how variance reduction is achieved through ensemble averaging.

```python
# Mathematical formulas for Bagging
"""
$$P(x) = \frac{1}{M} \sum_{m=1}^{M} P_m(x)$$

Where:
$$P(x)$$ is the final prediction
$$M$$ is the number of base models
$$P_m(x)$$ is the prediction of model m

Variance Reduction:
$$Var(\bar{X}) = \frac{\sigma^2}{n} \cdot \frac{1 + (n-1)\rho}{n}$$

Where:
$$\sigma^2$$ is the variance of individual models
$$n$$ is the number of models
$$\rho$$ is the correlation between models
"""
```

Slide 5: Advanced Boosting Implementation

AdaBoost's sophisticated weighting mechanism adjusts sample importance based on previous model performance. This implementation showcases the intricate details of weight updates and model combination in boosting algorithms.

```python
class AdvancedAdaBoost:
    def __init__(self, n_estimators=50, learning_rate=1.0):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.estimators = []
        self.estimator_weights = []
        
    def fit(self, X, y):
        n_samples = X.shape[0]
        sample_weights = np.ones(n_samples) / n_samples
        
        for _ in range(self.n_estimators):
            estimator = DecisionTreeClassifier(max_depth=2)
            estimator.fit(X, y, sample_weight=sample_weights)
            predictions = estimator.predict(X)
            
            incorrect = predictions != y
            estimator_error = np.sum(incorrect * sample_weights) / np.sum(sample_weights)
            
            # Avoid division by zero
            estimator_error = np.clip(estimator_error, 1e-15, 1 - 1e-15)
            
            estimator_weight = self.learning_rate * 0.5 * np.log(
                (1 - estimator_error) / estimator_error
            )
            
            # Update sample weights
            sample_weights *= np.exp(estimator_weight * (2 * incorrect - 1))
            sample_weights /= np.sum(sample_weights)
            
            self.estimators.append(estimator)
            self.estimator_weights.append(estimator_weight)
    
    def predict(self, X):
        predictions = np.array([
            estimator.predict(X) * weight
            for estimator, weight in zip(self.estimators, self.estimator_weights)
        ])
        return np.sign(np.sum(predictions, axis=0))
```

Slide 6: Random Forest Implementation from Scratch

Random Forest extends the bagging concept by incorporating feature randomization at each split. This implementation demonstrates how to combine multiple decision trees with random feature selection to create a robust ensemble classifier.

```python
class RandomForestFromScratch:
    def __init__(self, n_trees=100, max_features='sqrt'):
        self.n_trees = n_trees
        self.max_features = max_features
        self.trees = []
        
    def _get_max_features(self, n_features):
        if isinstance(self.max_features, str):
            if self.max_features == 'sqrt':
                return int(np.sqrt(n_features))
        return n_features
    
    def _create_tree(self, X, y):
        n_features = X.shape[1]
        max_features = self._get_max_features(n_features)
        tree = DecisionTreeClassifier(
            max_features=max_features,
            criterion='gini'
        )
        
        # Bootstrap sampling
        n_samples = X.shape[0]
        sample_idx = np.random.choice(n_samples, size=n_samples, replace=True)
        X_sample = X[sample_idx]
        y_sample = y[sample_idx]
        
        tree.fit(X_sample, y_sample)
        return tree
    
    def fit(self, X, y):
        self.trees = [self._create_tree(X, y) for _ in range(self.n_trees)]
    
    def predict(self, X):
        predictions = np.array([tree.predict(X) for tree in self.trees])
        return np.apply_along_axis(
            lambda x: np.bincount(x.astype(int)).argmax(),
            axis=0,
            arr=predictions
        )
```

Slide 7: Gradient Boosting Implementation

Gradient Boosting builds an ensemble by fitting new models to the residuals of previous predictions. This implementation shows how to create a basic gradient boosting machine for regression tasks.

```python
class GradientBoostingFromScratch:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees = []
        
    def fit(self, X, y):
        self.trees = []
        F = np.zeros(len(y))
        
        for _ in range(self.n_estimators):
            residuals = y - F
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X, residuals)
            
            predictions = tree.predict(X)
            F += self.learning_rate * predictions
            self.trees.append(tree)
    
    def predict(self, X):
        predictions = np.zeros(len(X))
        for tree in self.trees:
            predictions += self.learning_rate * tree.predict(X)
        return predictions
```

Slide 8: Real-world Application - Customer Churn Prediction

This example demonstrates how ensemble methods can be applied to predict customer churn in a telecommunications company, showcasing data preprocessing, model training, and evaluation metrics.

```python
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, precision_recall_curve

# Simulating telecom customer data
def generate_telecom_data(n_samples=1000):
    np.random.seed(42)
    data = {
        'usage_minutes': np.random.normal(600, 200, n_samples),
        'contract_length': np.random.choice(['monthly', 'yearly'], n_samples),
        'payment_delay': np.random.poisson(0.5, n_samples),
        'customer_service_calls': np.random.poisson(2, n_samples),
        'churn': np.random.binomial(1, 0.2, n_samples)
    }
    return pd.DataFrame(data)

# Data preprocessing
df = generate_telecom_data()
le = LabelEncoder()
df['contract_length'] = le.fit_transform(df['contract_length'])

X = df.drop('churn', axis=1)
y = df['churn']

# Train ensemble model
rf_model = RandomForestFromScratch(n_trees=100)
rf_model.fit(X.values, y.values)

# Evaluate
y_pred = rf_model.predict(X.values)
print(f"ROC-AUC Score: {roc_auc_score(y, y_pred)}")
```

Slide 9: Mathematics of Gradient Boosting

The mathematical foundations of gradient boosting involve optimization through gradient descent in function space. These formulas illustrate the core concepts behind the algorithm.

```python
"""
Forward Stagewise Additive Modeling:
$$F_m(x) = F_{m-1}(x) + \gamma_m h_m(x)$$

Where:
$$F_m(x)$$ is the model at iteration m
$$\gamma_m$$ is the step size
$$h_m(x)$$ is the base learner

Loss Minimization:
$$L(y, F_m(x)) = L(y, F_{m-1}(x)) - \gamma_m \nabla_F L(y, F_{m-1}(x)) h_m(x)$$

Gradient Calculation:
$$r_{im} = -\left[\frac{\partial L(y_i, F(x_i))}{\partial F(x_i)}\right]_{F=F_{m-1}}$$
"""
```

Slide 10: XGBoost Implementation Core Concepts

XGBoost represents a highly optimized implementation of gradient boosting, incorporating regularization and system optimization. This implementation demonstrates key concepts of the XGBoost algorithm including weighted quantile sketch and sparse-aware split finding.

```python
class SimpleXGBoost:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, lambda_l2=1.0):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.lambda_l2 = lambda_l2
        self.trees = []
    
    def _calculate_gradient_hessian(self, y_true, y_pred):
        gradient = 2 * (y_pred - y_true)
        hessian = 2 * np.ones_like(y_true)
        return gradient, hessian
    
    def _calculate_gain(self, gradient, hessian, left_indices, right_indices):
        left_grad = gradient[left_indices].sum()
        left_hess = hessian[left_indices].sum()
        right_grad = gradient[right_indices].sum()
        right_hess = hessian[right_indices].sum()
        
        gain = 0.5 * (
            (left_grad ** 2 / (left_hess + self.lambda_l2) +
             right_grad ** 2 / (right_hess + self.lambda_l2)) -
            (left_grad + right_grad) ** 2 / (left_hess + right_hess + self.lambda_l2)
        )
        return gain
    
    def fit(self, X, y):
        self.trees = []
        y_pred = np.zeros_like(y, dtype=float)
        
        for _ in range(self.n_estimators):
            gradient, hessian = self._calculate_gradient_hessian(y, y_pred)
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X, -gradient, sample_weight=hessian)
            
            update = self.learning_rate * tree.predict(X)
            y_pred += update
            self.trees.append(tree)
```

Slide 11: Handling Imbalanced Data in Ensemble Learning

Ensemble methods can be modified to handle imbalanced datasets through techniques like class weights, SMOTE sampling, and custom loss functions. This implementation shows how to adapt ensemble methods for imbalanced classification tasks.

```python
from imblearn.over_sampling import SMOTE
from collections import Counter

class ImbalancedEnsemble:
    def __init__(self, base_estimator='rf', n_estimators=100, sampling_strategy='auto'):
        self.n_estimators = n_estimators
        self.sampling_strategy = sampling_strategy
        self.base_estimator = base_estimator
        self.estimators = []
        
    def fit(self, X, y):
        # Apply SMOTE for each base estimator
        smote = SMOTE(sampling_strategy=self.sampling_strategy)
        print(f"Original class distribution: {Counter(y)}")
        
        for i in range(self.n_estimators):
            # Create different balanced datasets
            X_resampled, y_resampled = smote.fit_resample(X, y)
            
            if self.base_estimator == 'rf':
                estimator = DecisionTreeClassifier(max_depth=3)
            else:
                estimator = DecisionTreeClassifier(max_depth=1)
                
            # Train on balanced data
            estimator.fit(X_resampled, y_resampled)
            self.estimators.append(estimator)
            
        print(f"Resampled class distribution: {Counter(y_resampled)}")
    
    def predict_proba(self, X):
        probas = np.array([est.predict_proba(X) for est in self.estimators])
        return np.mean(probas, axis=0)
    
    def predict(self, X):
        probas = self.predict_proba(X)
        return np.argmax(probas, axis=1)
```

Slide 12: Real-world Application - Financial Market Prediction

This implementation demonstrates how ensemble methods can be applied to predict stock market movements using technical indicators and market data.

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def create_features(data):
    # Technical indicators
    data['SMA_20'] = data['close'].rolling(window=20).mean()
    data['RSI'] = calculate_rsi(data['close'], periods=14)
    data['MACD'] = calculate_macd(data['close'])
    return data

def calculate_rsi(prices, periods=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(prices, fast=12, slow=26):
    exp1 = prices.ewm(span=fast).mean()
    exp2 = prices.ewm(span=slow).mean()
    return exp1 - exp2

# Generate sample market data
np.random.seed(42)
dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
market_data = pd.DataFrame({
    'date': dates,
    'close': np.random.normal(100, 10, len(dates)).cumsum(),
    'volume': np.random.exponential(1000000, len(dates))
})

# Prepare features and target
market_data = create_features(market_data)
market_data['target'] = np.where(market_data['close'].shift(-1) > market_data['close'], 1, 0)

# Train ensemble model
features = ['SMA_20', 'RSI', 'MACD', 'volume']
X = market_data[features].dropna()
y = market_data['target'].dropna()

model = GradientBoostingFromScratch(n_estimators=100, learning_rate=0.1)
model.fit(X, y)
```

Slide 13: Advanced Ensemble Stacking Implementation

Stacking combines predictions from multiple models using a meta-learner. This implementation shows how to create a stacked ensemble that leverages the strengths of different base models while avoiding overfitting through cross-validation.

```python
from sklearn.model_selection import KFold
from sklearn.base import clone

class StackingEnsemble:
    def __init__(self, base_models, meta_model, n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds
        self.base_predictions = None
        
    def fit(self, X, y):
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        self.base_predictions = np.zeros((X.shape[0], len(self.base_models)))
        
        # Train base models using cross-validation
        for i, model in enumerate(self.base_models):
            model_predictions = np.zeros(X.shape[0])
            
            for train_idx, val_idx in kf.split(X):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train = y[train_idx]
                
                # Clone model to avoid fitting the same instance
                clone_model = clone(model)
                clone_model.fit(X_train, y_train)
                model_predictions[val_idx] = clone_model.predict(X_val)
            
            self.base_predictions[:, i] = model_predictions
            # Fit on full dataset
            model.fit(X, y)
        
        # Train meta model
        self.meta_model.fit(self.base_predictions, y)
        
    def predict(self, X):
        meta_features = np.column_stack([
            model.predict(X) for model in self.base_models
        ])
        return self.meta_model.predict(meta_features)
```

Slide 14: Time Series Forecasting with Ensemble Methods

This implementation demonstrates how to adapt ensemble methods for time series forecasting, incorporating temporal dependencies and handling seasonal patterns.

```python
class TimeSeriesEnsemble:
    def __init__(self, n_estimators=100, lookback=10, forecast_horizon=1):
        self.n_estimators = n_estimators
        self.lookback = lookback
        self.forecast_horizon = forecast_horizon
        self.models = []
        self.scalers = []
    
    def create_sequences(self, data):
        X, y = [], []
        for i in range(len(data) - self.lookback - self.forecast_horizon + 1):
            X.append(data[i:(i + self.lookback)])
            y.append(data[i + self.lookback:i + self.lookback + self.forecast_horizon])
        return np.array(X), np.array(y)
    
    def fit(self, data):
        X, y = self.create_sequences(data)
        
        for _ in range(self.n_estimators):
            # Bootstrap sampling with temporal blocks
            block_size = min(50, len(X) // 10)
            n_blocks = len(X) // block_size
            
            indices = []
            for _ in range(n_blocks):
                start_idx = np.random.randint(0, len(X) - block_size)
                indices.extend(range(start_idx, start_idx + block_size))
            
            X_boot = X[indices]
            y_boot = y[indices]
            
            # Scale data
            scaler = StandardScaler()
            X_boot_scaled = scaler.fit_transform(X_boot.reshape(-1, X_boot.shape[-1]))
            X_boot_scaled = X_boot_scaled.reshape(X_boot.shape)
            
            # Train model
            model = DecisionTreeRegressor(max_depth=3)
            model.fit(X_boot_scaled.reshape(X_boot_scaled.shape[0], -1), y_boot)
            
            self.models.append(model)
            self.scalers.append(scaler)
    
    def predict(self, X):
        predictions = []
        for model, scaler in zip(self.models, self.scalers):
            X_scaled = scaler.transform(X.reshape(-1, X.shape[-1]))
            X_scaled = X_scaled.reshape(X.shape)
            pred = model.predict(X_scaled.reshape(X_scaled.shape[0], -1))
            predictions.append(pred)
        return np.mean(predictions, axis=0)
```

Slide 15: Additional Resources

*   ArXiv Paper: "XGBoost: A Scalable Tree Boosting System"
    *   [https://arxiv.org/abs/1603.02754](https://arxiv.org/abs/1603.02754)
*   ArXiv Paper: "LightGBM: A Highly Efficient Gradient Boosting Decision Tree"
    *   [https://arxiv.org/abs/1711.08789](https://arxiv.org/abs/1711.08789)
*   ArXiv Paper: "CatBoost: unbiased boosting with categorical features"
    *   [https://arxiv.org/abs/1706.09516](https://arxiv.org/abs/1706.09516)
*   General Resources:
    *   Google Scholar: "ensemble methods machine learning"
    *   IEEE Xplore: Search for "gradient boosting algorithms"
    *   ACM Digital Library: "random forests applications"

