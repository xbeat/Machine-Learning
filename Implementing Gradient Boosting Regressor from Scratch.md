## Implementing Gradient Boosting Regressor from Scratch
Slide 1: Understanding Gradient Boosting Core Concepts

Gradient Boosting is an ensemble learning method that builds a strong predictive model by combining multiple weak learners sequentially. Each new model attempts to correct the errors made by the previous models, leading to increasingly accurate predictions through gradient descent optimization.

```python
# Mathematical foundation of Gradient Boosting
# Loss function and gradient calculation
'''
Loss Function:
$$L(y, F(x)) = \frac{1}{2}(y - F(x))^2$$

Negative Gradient:
$$-\frac{\partial L(y, F(x))}{\partial F(x)} = y - F(x)$$

Boosting Update:
$$F_{m}(x) = F_{m-1}(x) + \nu \cdot h_m(x)$$
'''

import numpy as np
from typing import Tuple

class GradientBoostingBase:
    def __init__(self, learning_rate: float = 0.1):
        self.learning_rate = learning_rate
        self.models = []
    
    def compute_pseudo_residuals(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Calculate negative gradients (residuals) for MSE loss"""
        return y_true - y_pred
```

Slide 2: Decision Tree as Base Learner

The foundation of Gradient Boosting typically uses decision trees as weak learners. These trees are constrained in depth to prevent overfitting and maintain the "weak learner" property essential for boosting algorithms to work effectively.

```python
class DecisionNode:
    def __init__(self):
        self.feature_index = None
        self.threshold = None
        self.left = None
        self.right = None
        self.value = None
        
class DecisionTree:
    def __init__(self, max_depth: int = 3):
        self.max_depth = max_depth
        self.root = DecisionNode()
        
    def calculate_mse(self, y: np.ndarray) -> float:
        """Calculate Mean Squared Error for a node"""
        if len(y) == 0:
            return 0
        return np.mean((y - np.mean(y)) ** 2)
    
    def find_best_split(self, X: np.ndarray, y: np.ndarray) -> Tuple[int, float]:
        best_feature = None
        best_threshold = None
        best_score = float('inf')
        
        for feature in range(X.shape[1]):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask
                
                score = (self.calculate_mse(y[left_mask]) * sum(left_mask) +
                        self.calculate_mse(y[right_mask]) * sum(right_mask)) / len(y)
                
                if score < best_score:
                    best_score = score
                    best_feature = feature
                    best_threshold = threshold
                    
        return best_feature, best_threshold
```

Slide 3: Building the Tree Structure

The recursive tree-building process forms the backbone of our gradient boosting implementation. Each split decision aims to minimize the loss function while respecting the maximum depth constraint to prevent overfitting.

```python
def build_tree(self, X: np.ndarray, y: np.ndarray, depth: int = 0) -> DecisionNode:
    node = DecisionNode()
    
    # Base cases: max depth reached or pure node
    if depth >= self.max_depth or len(np.unique(y)) == 1:
        node.value = np.mean(y)
        return node
    
    # Find best split
    feature_index, threshold = self.find_best_split(X, y)
    
    if feature_index is None:  # No improvement possible
        node.value = np.mean(y)
        return node
    
    # Create child nodes
    left_mask = X[:, feature_index] <= threshold
    right_mask = ~left_mask
    
    node.feature_index = feature_index
    node.threshold = threshold
    node.left = self.build_tree(X[left_mask], y[left_mask], depth + 1)
    node.right = self.build_tree(X[right_mask], y[right_mask], depth + 1)
    
    return node

def predict_single(self, node: DecisionNode, x: np.ndarray) -> float:
    if node.value is not None:
        return node.value
    
    if x[node.feature_index] <= node.threshold:
        return self.predict_single(node.left, x)
    return self.predict_single(node.right, x)
```

Slide 4: Gradient Boosting Implementation Core

The core implementation combines the base learners sequentially, with each new model focusing on the residuals from previous predictions. The learning rate controls the contribution of each new tree to the ensemble.

```python
class GradientBoostingRegressor:
    def __init__(self, n_estimators: int = 100, learning_rate: float = 0.1, 
                 max_depth: int = 3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees = []
        self.initial_prediction = None
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        # Initialize prediction with mean value
        self.initial_prediction = np.mean(y)
        current_predictions = np.full_like(y, self.initial_prediction, dtype=float)
        
        for _ in range(self.n_estimators):
            # Calculate pseudo residuals
            residuals = y - current_predictions
            
            # Train new tree on residuals
            tree = DecisionTree(max_depth=self.max_depth)
            tree.fit(X, residuals)
            self.trees.append(tree)
            
            # Update predictions
            predictions = tree.predict(X)
            current_predictions += self.learning_rate * predictions
        
        return self
```

Slide 5: Model Training and Prediction Pipeline

A comprehensive training pipeline ensures proper model initialization, sequential tree building, and residual updates. The prediction mechanism aggregates contributions from all trees while applying the learning rate to maintain stable predictions.

```python
def predict(self, X: np.ndarray) -> np.ndarray:
    # Start with initial prediction
    predictions = np.full(X.shape[0], self.initial_prediction, dtype=float)
    
    # Add predictions from each tree
    for tree in self.trees:
        predictions += self.learning_rate * tree.predict(X)
    
    return predictions

def score(self, X: np.ndarray, y: np.ndarray) -> float:
    """Calculate R-squared score"""
    predictions = self.predict(X)
    ss_total = np.sum((y - np.mean(y)) ** 2)
    ss_residual = np.sum((y - predictions) ** 2)
    return 1 - (ss_residual / ss_total)
```

Slide 6: Loss Function and Gradient Computation

The loss function guides the boosting process by quantifying prediction errors. For regression, we implement Mean Squared Error (MSE) loss, while the gradients direct the learning process towards error minimization.

```python
class LossFunctions:
    @staticmethod
    def mse_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Mean Squared Error Loss
        $$L(y, \hat{y}) = \frac{1}{n}\sum_{i=1}^n (y_i - \hat{y}_i)^2$$
        """
        return np.mean((y_true - y_pred) ** 2)
    
    @staticmethod
    def mse_gradient(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Gradient of MSE Loss
        $$\frac{\partial L}{\partial \hat{y}} = -2(y - \hat{y})$$
        """
        return -2 * (y_true - y_pred)
```

Slide 7: Real-world Example: Housing Price Prediction

Implementing gradient boosting for predicting housing prices demonstrates its practical application. We'll use a subset of features from the California housing dataset to showcase the model's effectiveness.

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load and preprocess housing data
def prepare_housing_data():
    from sklearn.datasets import fetch_california_housing
    housing = fetch_california_housing()
    X = pd.DataFrame(housing.data, columns=housing.feature_names)
    y = housing.target
    
    # Split and scale data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test

# Train and evaluate model
X_train, X_test, y_train, y_test = prepare_housing_data()
gb_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1)
gb_model.fit(X_train, y_train)
```

Slide 8: Model Evaluation and Performance Metrics

A comprehensive evaluation framework helps assess model performance through various metrics. We implement multiple evaluation criteria to provide a thorough understanding of the model's predictive capabilities.

```python
class ModelEvaluator:
    @staticmethod
    def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        """Calculate multiple regression metrics"""
        mse = np.mean((y_true - y_pred) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_true - y_pred))
        r2 = 1 - (np.sum((y_true - y_pred) ** 2) / 
                  np.sum((y_true - np.mean(y_true)) ** 2))
        
        return {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2
        }

# Evaluate model performance
train_pred = gb_model.predict(X_train)
test_pred = gb_model.predict(X_test)

print("Training Metrics:", ModelEvaluator.calculate_metrics(y_train, train_pred))
print("Testing Metrics:", ModelEvaluator.calculate_metrics(y_test, test_pred))
```

Slide 9: Feature Importance Calculation

Feature importance analysis reveals which predictors contribute most significantly to the model's predictions. This implementation calculates importance scores based on the reduction in prediction error achieved by each feature across all trees.

```python
class FeatureImportance:
    def __init__(self, model, feature_names=None):
        self.model = model
        self.feature_names = feature_names
        self.importance_scores = None
        
    def calculate_importance(self) -> np.ndarray:
        """
        Calculate feature importance scores using variance reduction
        $$Importance_j = \sum_{t=1}^T \sum_{n \in N_t} w_n v_n I(v_n = j)$$
        """
        n_features = self.model.trees[0].n_features
        importance = np.zeros(n_features)
        
        for tree in self.model.trees:
            tree_importance = self._get_tree_importance(tree)
            importance += tree_importance
            
        # Normalize scores
        importance = importance / len(self.model.trees)
        self.importance_scores = importance
        return importance
    
    def plot_importance(self):
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(self.importance_scores)), self.importance_scores)
        if self.feature_names is not None:
            plt.xticks(range(len(self.importance_scores)), self.feature_names, rotation=45)
        plt.title('Feature Importance')
        plt.tight_layout()
        plt.show()
```

Slide 10: Hyperparameter Optimization

Optimizing model hyperparameters is crucial for achieving optimal performance. We implement a grid search with cross-validation to find the best combination of learning rate, number of estimators, and maximum tree depth.

```python
class GridSearchCV:
    def __init__(self, param_grid: dict, cv: int = 5):
        self.param_grid = param_grid
        self.cv = cv
        self.best_params_ = None
        self.best_score_ = float('-inf')
        
    def fit(self, X: np.ndarray, y: np.ndarray):
        from itertools import product
        param_combinations = [dict(zip(self.param_grid.keys(), v)) 
                            for v in product(*self.param_grid.values())]
        
        for params in param_combinations:
            scores = []
            # Perform k-fold cross-validation
            for fold in range(self.cv):
                # Create train/val split
                val_size = len(X) // self.cv
                val_start = fold * val_size
                val_end = (fold + 1) * val_size
                
                X_train = np.concatenate([X[:val_start], X[val_end:]])
                y_train = np.concatenate([y[:val_start], y[val_end:]])
                X_val = X[val_start:val_end]
                y_val = y[val_start:val_end]
                
                # Train and evaluate model
                model = GradientBoostingRegressor(**params)
                model.fit(X_train, y_train)
                score = model.score(X_val, y_val)
                scores.append(score)
            
            # Update best parameters if necessary
            mean_score = np.mean(scores)
            if mean_score > self.best_score_:
                self.best_score_ = mean_score
                self.best_params_ = params
```

Slide 11: Real-world Example: Stock Price Prediction

This comprehensive example demonstrates gradient boosting application for predicting stock prices using technical indicators and historical data. We implement feature engineering and time series cross-validation.

```python
import yfinance as yf
from typing import List, Tuple

class StockPricePredictor:
    def __init__(self, symbol: str, start_date: str, end_date: str):
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.data = None
        self.model = None
        
    def prepare_data(self) -> Tuple[np.ndarray, np.ndarray]:
        # Download stock data
        stock_data = yf.download(self.symbol, self.start_date, self.end_date)
        
        # Calculate technical indicators
        stock_data['SMA_20'] = stock_data['Close'].rolling(window=20).mean()
        stock_data['RSI'] = self.calculate_rsi(stock_data['Close'])
        stock_data['MACD'] = self.calculate_macd(stock_data['Close'])
        
        # Prepare features and target
        features = ['SMA_20', 'RSI', 'MACD', 'Volume']
        X = stock_data[features].values[20:]  # Skip NaN values
        y = stock_data['Close'].values[20:]
        
        return X, y
    
    @staticmethod
    def calculate_rsi(prices: np.ndarray, periods: int = 14) -> np.ndarray:
        """Calculate Relative Strength Index"""
        delta = np.diff(prices)
        gain = (delta > 0) * delta
        loss = (delta < 0) * -delta
        
        avg_gain = np.convolve(gain, np.ones(periods)/periods, mode='valid')
        avg_loss = np.convolve(loss, np.ones(periods)/periods, mode='valid')
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return np.pad(rsi, (periods, 0), mode='constant', constant_values=np.nan)
```

Slide 12: Advanced Feature Engineering

Feature engineering in gradient boosting requires careful consideration of interaction effects and non-linear relationships. This implementation showcases advanced feature transformation techniques specifically designed for boosting algorithms.

```python
class FeatureEngineer:
    def __init__(self, polynomial_degree: int = 2, interaction_only: bool = False):
        self.polynomial_degree = polynomial_degree
        self.interaction_only = interaction_only
        self.feature_names = None
        
    def generate_polynomial_features(self, X: np.ndarray) -> np.ndarray:
        """
        Generate polynomial and interaction features
        For features x1, x2: Creates x1^2, x2^2, x1*x2 if degree=2
        """
        n_samples, n_features = X.shape
        combinations = []
        
        for degree in range(2, self.polynomial_degree + 1):
            for indices in self._get_combinations(range(n_features), degree):
                new_feature = np.prod(X[:, indices], axis=1)
                combinations.append(new_feature.reshape(-1, 1))
        
        if len(combinations) > 0:
            return np.hstack([X] + combinations)
        return X
    
    def _get_combinations(self, items, degree):
        from itertools import combinations_with_replacement
        if self.interaction_only:
            from itertools import combinations
            return combinations(items, degree)
        return combinations_with_replacement(items, degree)
```

Slide 13: Early Stopping and Model Validation

Implementing early stopping prevents overfitting by monitoring validation performance during training. The implementation includes a flexible validation scheme with various stopping criteria.

```python
class EarlyStoppingGBR(GradientBoostingRegressor):
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3,
                 patience=5, min_delta=1e-4):
        super().__init__(n_estimators, learning_rate, max_depth)
        self.patience = patience
        self.min_delta = min_delta
        self.validation_scores = []
        
    def fit(self, X: np.ndarray, y: np.ndarray, 
            X_val: np.ndarray, y_val: np.ndarray) -> 'EarlyStoppingGBR':
        self.initial_prediction = np.mean(y)
        current_predictions = np.full_like(y, self.initial_prediction)
        val_predictions = np.full_like(y_val, self.initial_prediction)
        
        best_score = float('-inf')
        patience_counter = 0
        
        for i in range(self.n_estimators):
            # Train new tree
            residuals = y - current_predictions
            tree = DecisionTree(max_depth=self.max_depth)
            tree.fit(X, residuals)
            
            # Update predictions
            tree_preds = tree.predict(X)
            val_tree_preds = tree.predict(X_val)
            
            current_predictions += self.learning_rate * tree_preds
            val_predictions += self.learning_rate * val_tree_preds
            
            # Calculate validation score
            val_score = self._calculate_score(y_val, val_predictions)
            self.validation_scores.append(val_score)
            
            # Check for early stopping
            if val_score > best_score + self.min_delta:
                best_score = val_score
                patience_counter = 0
                self.trees.append(tree)
            else:
                patience_counter += 1
                
            if patience_counter >= self.patience:
                break
                
        return self
```

Slide 14: Additional Resources

*   "A New Learning Paradigm: Gradient Boosting" - [https://arxiv.org/abs/1901.02345](https://arxiv.org/abs/1901.02345)
*   "XGBoost: A Scalable Tree Boosting System" - [https://arxiv.org/abs/1603.02754](https://arxiv.org/abs/1603.02754)
*   "LightGBM: A Highly Efficient Gradient Boosting Decision Tree" - [https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree](https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree)
*   For advanced implementation details and research papers, search on Google Scholar: "gradient boosting optimization techniques"
*   Official documentation and tutorials: [https://scikit-learn.org/stable/modules/ensemble.html#gradient-boosting](https://scikit-learn.org/stable/modules/ensemble.html#gradient-boosting)

