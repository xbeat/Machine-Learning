## Why Boosting Models Use Decision Trees
Slide 1: Understanding Base Learners in Boosting

Boosting algorithms can work with various base learners, not just decision trees. This introductory implementation demonstrates how to create a simple boosting framework that accepts different types of base learners, showing the flexibility of the boosting concept.

```python
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR

class CustomBoostingRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, base_learner=None, n_estimators=100, learning_rate=0.1):
        self.base_learner = base_learner or DecisionTreeRegressor(max_depth=3)
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.models = []
        
    def fit(self, X, y):
        current_pred = np.zeros_like(y)
        for _ in range(self.n_estimators):
            residual = y - current_pred
            model = clone(self.base_learner).fit(X, residual)
            self.models.append(model)
            current_pred += self.learning_rate * model.predict(X)
        return self
```

Slide 2: Comparing Different Base Learners

The effectiveness of different base learners can be evaluated by implementing them in our boosting framework. Here we compare decision trees, linear regression, and SVR models as base learners on a synthetic dataset.

```python
# Generate synthetic dataset
np.random.seed(42)
X = np.random.randn(1000, 5)
y = 5 * X[:, 0] + np.sin(X[:, 1]) + np.random.randn(1000) * 0.1

# Initialize different boosting models
boosting_tree = CustomBoostingRegressor(
    base_learner=DecisionTreeRegressor(max_depth=3),
    n_estimators=100
)

boosting_linear = CustomBoostingRegressor(
    base_learner=LinearRegression(),
    n_estimators=100
)

boosting_svr = CustomBoostingRegressor(
    base_learner=SVR(kernel='rbf'),
    n_estimators=100
)

# Evaluate models using cross-validation
from sklearn.model_selection import cross_val_score
scores_tree = cross_val_score(boosting_tree, X, y, cv=5)
scores_linear = cross_val_score(boosting_linear, X, y, cv=5)
scores_svr = cross_val_score(boosting_svr, X, y, cv=5)
```

Slide 3: Tree-Based Boosting Implementation

Tree-based boosting models excel in handling complex data patterns. This implementation shows how to create a gradient boosting regressor specifically optimized for decision trees as base learners.

```python
class TreeBoostingRegressor:
    def __init__(self, max_depth=3, n_estimators=100, learning_rate=0.1):
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.trees = []
        
    def fit(self, X, y):
        self.F0 = np.mean(y)
        F = np.ones(len(y)) * self.F0
        
        for _ in range(self.n_estimators):
            residuals = y - F
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X, residuals)
            self.trees.append(tree)
            F += self.learning_rate * tree.predict(X)
        return self
    
    def predict(self, X):
        return self.F0 + self.learning_rate * sum(
            tree.predict(X) for tree in self.trees
        )
```

Slide 4: Handling Missing Values in Boosting

Tree-based boosting models naturally handle missing values through their splitting criteria. This implementation demonstrates how different base learners process datasets with missing values.

```python
import pandas as pd
from sklearn.impute import SimpleImputer

# Create dataset with missing values
X_missing = X.copy()
X_missing[np.random.rand(*X.shape) < 0.1] = np.nan

# Tree-based boosting (handles missing values naturally)
tree_boost = TreeBoostingRegressor()
tree_boost.fit(X_missing, y)

# Other base learners require imputation
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X_missing)

linear_boost = CustomBoostingRegressor(base_learner=LinearRegression())
linear_boost.fit(X_imputed, y)

# Compare performance
print(f"Tree MSE: {np.mean((y - tree_boost.predict(X_missing))**2)}")
print(f"Linear MSE: {np.mean((y - linear_boost.predict(X_imputed))**2)}")
```

Slide 5: Mathematical Foundation of Boosting

Understanding the mathematical principles behind boosting helps explain why trees work well as base learners. The following implementation shows the gradient boosting algorithm with its mathematical foundation.

```python
"""
Mathematical formulation:
$$F_m(x) = F_{m-1}(x) + \gamma_m h_m(x)$$
where:
$$h_m(x) = \arg\min_{h} \sum_{i=1}^n L(y_i, F_{m-1}(x_i) + h(x_i))$$
$$\gamma_m = \arg\min_{\gamma} \sum_{i=1}^n L(y_i, F_{m-1}(x_i) + \gamma h_m(x_i))$$
"""

class GradientBoostingMath:
    def __init__(self, n_estimators=100, learning_rate=0.1):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.trees = []
        
    def gradient(self, y_true, y_pred):
        return y_true - y_pred  # For MSE loss
        
    def fit(self, X, y):
        self.initial_prediction = np.mean(y)
        F = np.ones(len(y)) * self.initial_prediction
        
        for _ in range(self.n_estimators):
            gradient = self.gradient(y, F)
            tree = DecisionTreeRegressor(max_depth=3)
            tree.fit(X, gradient)
            self.trees.append(tree)
            F += self.learning_rate * tree.predict(X)
```

Slide 6: Feature Importance in Tree-Based Boosting

Tree-based boosting models provide natural feature importance calculations based on the reduction in loss function at each split. This implementation demonstrates how to extract and visualize feature importance scores.

```python
import matplotlib.pyplot as plt

class FeatureImportanceBooster:
    def __init__(self, n_estimators=100):
        self.n_estimators = n_estimators
        self.trees = []
        self.feature_importances_ = None
        
    def fit(self, X, y, feature_names=None):
        self.feature_names = feature_names or [f'Feature_{i}' for i in range(X.shape[1])]
        self.feature_importances_ = np.zeros(X.shape[1])
        
        F = np.zeros(len(y))
        for _ in range(self.n_estimators):
            tree = DecisionTreeRegressor(max_depth=3)
            tree.fit(X, y - F)
            self.trees.append(tree)
            self.feature_importances_ += tree.feature_importances_
            F += tree.predict(X)
            
        self.feature_importances_ /= self.n_estimators
        return self
    
    def plot_importance(self):
        plt.figure(figsize=(10, 6))
        sorted_idx = np.argsort(self.feature_importances_)
        pos = np.arange(sorted_idx.shape[0]) + .5
        plt.barh(pos, self.feature_importances_[sorted_idx])
        plt.yticks(pos, np.array(self.feature_names)[sorted_idx])
        plt.xlabel('Feature Importance')
        plt.title('Feature Importance in Boosting Model')
        plt.tight_layout()
        return plt.gcf()
```

Slide 7: Handling Categorical Variables

Tree-based boosting models can naturally handle categorical variables through one-hot encoding or label encoding. This implementation shows how different encodings affect boosting performance.

```python
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import pandas as pd

# Create synthetic categorical data
np.random.seed(42)
categorical_data = pd.DataFrame({
    'num_feature': np.random.randn(1000),
    'cat_feature1': np.random.choice(['A', 'B', 'C'], 1000),
    'cat_feature2': np.random.choice(['X', 'Y', 'Z', 'W'], 1000)
})

# Label encoding approach
label_encoder = LabelEncoder()
cat_label_encoded = categorical_data.copy()
for col in ['cat_feature1', 'cat_feature2']:
    cat_label_encoded[col] = label_encoder.fit_transform(
        categorical_data[col]
    )

# One-hot encoding approach
onehot_encoder = OneHotEncoder(sparse=False)
cat_cols = ['cat_feature1', 'cat_feature2']
cat_onehot = onehot_encoder.fit_transform(categorical_data[cat_cols])
cat_onehot_df = pd.DataFrame(
    cat_onehot,
    columns=onehot_encoder.get_feature_names_out(cat_cols)
)
cat_onehot_df['num_feature'] = categorical_data['num_feature']

# Compare boosting performance with different encodings
y = np.random.randn(1000)  # Target variable
tree_boost_label = TreeBoostingRegressor().fit(cat_label_encoded, y)
tree_boost_onehot = TreeBoostingRegressor().fit(cat_onehot_df, y)
```

Slide 8: Learning Rate Scheduling

Implementing dynamic learning rate scheduling can improve boosting model performance. This implementation shows how to create an adaptive learning rate scheme.

```python
class AdaptiveLearningRateBooster:
    def __init__(self, n_estimators=100, initial_lr=0.1):
        self.n_estimators = n_estimators
        self.initial_lr = initial_lr
        self.trees = []
        self.learning_rates = []
        
    def schedule_learning_rate(self, iteration):
        """Implements a cosine annealing learning rate schedule"""
        return self.initial_lr * (1 + np.cos(np.pi * iteration / self.n_estimators)) / 2
        
    def fit(self, X, y):
        F = np.zeros(len(y))
        for i in range(self.n_estimators):
            lr = self.schedule_learning_rate(i)
            self.learning_rates.append(lr)
            
            residuals = y - F
            tree = DecisionTreeRegressor(max_depth=3)
            tree.fit(X, residuals)
            self.trees.append(tree)
            
            F += lr * tree.predict(X)
        return self
    
    def plot_learning_rates(self):
        plt.plot(self.learning_rates)
        plt.xlabel('Iteration')
        plt.ylabel('Learning Rate')
        plt.title('Adaptive Learning Rate Schedule')
        return plt.gcf()
```

Slide 9: Early Stopping Implementation

Early stopping prevents overfitting by monitoring validation performance during training. This implementation shows how to incorporate early stopping in a boosting model with a custom patience mechanism.

```python
class EarlyStoppingBooster:
    def __init__(self, n_estimators=100, learning_rate=0.1, patience=10):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.patience = patience
        self.trees = []
        
    def fit(self, X, y, X_val, y_val):
        best_val_loss = float('inf')
        patience_counter = 0
        F_train = np.zeros(len(y))
        F_val = np.zeros(len(y_val))
        
        for i in range(self.n_estimators):
            # Fit new tree on residuals
            residuals = y - F_train
            tree = DecisionTreeRegressor(max_depth=3)
            tree.fit(X, residuals)
            
            # Update predictions
            F_train += self.learning_rate * tree.predict(X)
            F_val += self.learning_rate * tree.predict(X_val)
            
            # Calculate validation loss
            val_loss = np.mean((y_val - F_val) ** 2)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.trees.append(tree)
            else:
                patience_counter += 1
                
            if patience_counter >= self.patience:
                print(f"Early stopping at iteration {i}")
                break
        
        return self
```

Slide 10: Real-world Example - Housing Price Prediction

This practical implementation demonstrates boosting on the Boston Housing dataset, showing data preprocessing, model training, and performance evaluation.

```python
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load and prepare data
boston = load_boston()
X, y = boston.data, boston.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create and train model with different base learners
models = {
    'tree': CustomBoostingRegressor(
        base_learner=DecisionTreeRegressor(max_depth=3)
    ),
    'linear': CustomBoostingRegressor(
        base_learner=LinearRegression()
    ),
    'svr': CustomBoostingRegressor(
        base_learner=SVR(kernel='rbf')
    )
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    results[name] = {
        'mse': mean_squared_error(y_test, y_pred),
        'r2': r2_score(y_test, y_pred)
    }

# Print results
for name, metrics in results.items():
    print(f"{name.upper()} Results:")
    print(f"MSE: {metrics['mse']:.4f}")
    print(f"R2: {metrics['r2']:.4f}\n")
```

Slide 11: Advanced Base Learner Selection

This implementation demonstrates how to dynamically select the best base learner for each boosting iteration based on validation performance.

```python
class AdaptiveBaseLearnerBooster:
    def __init__(self, base_learners=None, n_estimators=100):
        self.base_learners = base_learners or [
            DecisionTreeRegressor(max_depth=3),
            DecisionTreeRegressor(max_depth=5),
            LinearRegression()
        ]
        self.n_estimators = n_estimators
        self.selected_models = []
        
    def fit(self, X, y):
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2
        )
        F_train = np.zeros(len(y_train))
        F_val = np.zeros(len(y_val))
        
        for _ in range(self.n_estimators):
            best_val_loss = float('inf')
            best_model = None
            
            # Try each base learner
            for base_model in self.base_learners:
                model = clone(base_model)
                model.fit(X_train, y_train - F_train)
                val_pred = model.predict(X_val)
                val_loss = mean_squared_error(y_val, F_val + val_pred)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model = model
            
            # Update with best model
            self.selected_models.append(best_model)
            F_train += best_model.predict(X_train)
            F_val += best_model.predict(X_val)
            
        return self
```

Slide 12: Performance Comparison Framework

This implementation provides a comprehensive framework for comparing different base learners in boosting, including metrics calculation and visualization of results across multiple datasets.

```python
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_validate
import seaborn as sns

class BoostingComparator:
    def __init__(self, base_learners_dict, n_estimators=100):
        self.base_learners = base_learners_dict
        self.n_estimators = n_estimators
        self.results = {}
        
    def evaluate(self, X, y, cv=5):
        scoring = {
            'mse': make_scorer(mean_squared_error),
            'r2': make_scorer(r2_score)
        }
        
        for name, base_learner in self.base_learners.items():
            booster = CustomBoostingRegressor(
                base_learner=base_learner,
                n_estimators=self.n_estimators
            )
            
            scores = cross_validate(
                booster, X, y,
                scoring=scoring,
                cv=cv,
                return_train_score=True
            )
            
            self.results[name] = {
                'test_mse': scores['test_mse'].mean(),
                'test_r2': scores['test_r2'].mean(),
                'train_mse': scores['train_mse'].mean(),
                'train_r2': scores['train_r2'].mean()
            }
            
    def plot_comparison(self):
        df_results = pd.DataFrame(self.results).T
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        sns.barplot(data=df_results, y=df_results.index, x='test_mse', ax=ax1)
        ax1.set_title('MSE Comparison')
        
        sns.barplot(data=df_results, y=df_results.index, x='test_r2', ax=ax2)
        ax2.set_title('R2 Comparison')
        
        plt.tight_layout()
        return fig
```

Slide 13: Real-world Example - Credit Risk Assessment

A practical implementation showing how different base learners perform in a credit risk prediction scenario, including handling of imbalanced classes and feature preprocessing.

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, precision_recall_curve

class CreditRiskBooster:
    def __init__(self, base_learner=None, n_estimators=100):
        self.base_learner = base_learner or DecisionTreeRegressor(max_depth=3)
        self.n_estimators = n_estimators
        self.scaler = StandardScaler()
        
    def preprocess_features(self, X):
        numeric_cols = X.select_dtypes(include=['float64', 'int64']).columns
        X_processed = X.copy()
        X_processed[numeric_cols] = self.scaler.fit_transform(X[numeric_cols])
        return X_processed
    
    def fit(self, X, y):
        X_processed = self.preprocess_features(X)
        self.booster = CustomBoostingRegressor(
            base_learner=self.base_learner,
            n_estimators=self.n_estimators
        )
        self.booster.fit(X_processed, y)
        return self
    
    def predict_proba(self, X):
        X_processed = self.preprocess_features(X)
        raw_predictions = self.booster.predict(X_processed)
        return 1 / (1 + np.exp(-raw_predictions))
    
    def evaluate(self, X_test, y_test):
        probs = self.predict_proba(X_test)
        auc_score = roc_auc_score(y_test, probs)
        precision, recall, _ = precision_recall_curve(y_test, probs)
        
        return {
            'auc_roc': auc_score,
            'precision': precision,
            'recall': recall
        }
```

Slide 14: Additional Resources

*   "XGBoost: A Scalable Tree Boosting System" - [https://arxiv.org/abs/1603.02754](https://arxiv.org/abs/1603.02754)
*   "LightGBM: A Highly Efficient Gradient Boosting Decision Tree" - [https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree](https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree)
*   "CatBoost: unbiased boosting with categorical features" - [https://arxiv.org/abs/1706.09516](https://arxiv.org/abs/1706.09516)
*   "On the Decision Theoretic Generalization of On-line Learning and an Application to Boosting" - [https://www.sciencedirect.com/science/article/abs/pii/S002200009791504X](https://www.sciencedirect.com/science/article/abs/pii/S002200009791504X)
*   Suggested Google Search: "Comparison of different base learners in gradient boosting algorithms"

