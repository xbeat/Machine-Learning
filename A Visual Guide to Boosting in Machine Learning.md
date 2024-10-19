## A Visual Guide to Boosting in Machine Learning

Slide 1: Introduction to Boosting

Boosting is a powerful ensemble learning technique in machine learning. It combines multiple weak learners to create a strong learner, improving overall prediction accuracy. This method iteratively trains models, with each subsequent model focusing on the errors of its predecessors. Let's explore the key concepts and implementation of boosting algorithms.

```python
import random

# Simulating a weak learner
def weak_learner(X):
    # Randomly classify as 0 or 1 with slightly better than 50% accuracy
    return [1 if random.random() > 0.45 else 0 for _ in X]

# Example dataset
X = [[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]]
y = [0, 0, 1, 1, 1]

# Simulate boosting with 3 weak learners
predictions = [weak_learner(X) for _ in range(3)]

# Combine predictions (simple majority vote)
final_prediction = [round(sum(p[i] for p in predictions) / len(predictions))
                    for i in range(len(X))]

print("Final prediction:", final_prediction)
print("Actual labels:   ", y)
```

Slide 2: How Boosting Works

Boosting works by training models sequentially, with each new model focusing on the mistakes of the previous ones. It assigns higher weights to misclassified samples, ensuring that subsequent models pay more attention to these difficult cases. The final prediction is typically a weighted combination of all individual model predictions.

```python
import math

def adaboost_simple(X, y, num_estimators=3):
    n_samples = len(X)
    weights = [1/n_samples] * n_samples
    estimators = []
    estimator_weights = []

    for _ in range(num_estimators):
        # Train a weak learner (simulated here)
        predictions = weak_learner(X)
        
        # Calculate error and estimator weight
        error = sum(w for i, w in enumerate(weights) if predictions[i] != y[i])
        estimator_weight = 0.5 * math.log((1 - error) / error)
        
        # Update sample weights
        weights = [w * math.exp(-estimator_weight * y[i] * predictions[i]) 
                   for i, w in enumerate(weights)]
        total = sum(weights)
        weights = [w / total for w in weights]
        
        estimators.append(predictions)
        estimator_weights.append(estimator_weight)

    return estimators, estimator_weights

# Run AdaBoost
estimators, estimator_weights = adaboost_simple(X, y)

# Make final prediction
final_prediction = [1 if sum(e_weight * e[i] for e, e_weight in zip(estimators, estimator_weights)) > 0 
                    else 0 for i in range(len(X))]

print("AdaBoost prediction:", final_prediction)
print("Actual labels:      ", y)
```

Slide 3: AdaBoost Algorithm

AdaBoost (Adaptive Boosting) is one of the most popular boosting algorithms. It works by assigning weights to both the training samples and the weak learners. After each iteration, it increases the weights of misclassified samples and decreases the weights of correctly classified ones. This process forces subsequent weak learners to focus on the harder examples.

```python
def adaboost(X, y, num_estimators=5):
    n_samples, n_features = len(X), len(X[0])
    weights = [1/n_samples] * n_samples
    estimators = []
    estimator_weights = []

    for _ in range(num_estimators):
        # Train a decision stump (simulated here)
        best_feature, best_threshold = 0, 0
        min_error = float('inf')
        for feature in range(n_features):
            for sample in X:
                threshold = sample[feature]
                predictions = [1 if x[feature] > threshold else -1 for x in X]
                error = sum(w for i, w in enumerate(weights) if predictions[i] != y[i])
                if error < min_error:
                    min_error = error
                    best_feature, best_threshold = feature, threshold
        
        # Calculate estimator weight
        estimator_weight = 0.5 * math.log((1 - min_error) / min_error)
        
        # Make predictions and update weights
        predictions = [1 if x[best_feature] > best_threshold else -1 for x in X]
        weights = [w * math.exp(-estimator_weight * y[i] * predictions[i]) 
                   for i, w in enumerate(weights)]
        weights = [w / sum(weights) for w in weights]
        
        estimators.append((best_feature, best_threshold))
        estimator_weights.append(estimator_weight)

    return estimators, estimator_weights

# Run AdaBoost
X = [[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]]
y = [-1, -1, 1, 1, 1]
estimators, estimator_weights = adaboost(X, y)

# Make final prediction
def predict(X, estimators, estimator_weights):
    return [1 if sum(e_weight * (1 if x[e[0]] > e[1] else -1) 
            for e, e_weight in zip(estimators, estimator_weights)) > 0 
            else -1 for x in X]

final_prediction = predict(X, estimators, estimator_weights)
print("AdaBoost prediction:", final_prediction)
print("Actual labels:      ", y)
```

Slide 4: Gradient Boosting

Gradient Boosting is another popular boosting algorithm that works by iteratively training weak learners to correct the errors of previous ones. Unlike AdaBoost, which adjusts the sample weights, Gradient Boosting fits the new predictor to the residual errors made by the previous predictor.

```python
import random

def mse(y_true, y_pred):
    return sum((yt - yp) ** 2 for yt, yp in zip(y_true, y_pred)) / len(y_true)

def gradient_boosting(X, y, num_estimators=3, learning_rate=0.1):
    predictions = [0] * len(y)
    
    for _ in range(num_estimators):
        residuals = [yt - yp for yt, yp in zip(y, predictions)]
        
        # Train a weak learner (decision tree stump simulated here)
        best_feature, best_threshold, best_value = 0, 0, 0
        min_mse = float('inf')
        for feature in range(len(X[0])):
            for sample in X:
                threshold = sample[feature]
                left_residuals = [r for x, r in zip(X, residuals) if x[feature] <= threshold]
                right_residuals = [r for x, r in zip(X, residuals) if x[feature] > threshold]
                left_value = sum(left_residuals) / len(left_residuals) if left_residuals else 0
                right_value = sum(right_residuals) / len(right_residuals) if right_residuals else 0
                
                current_predictions = [left_value if x[feature] <= threshold else right_value for x in X]
                current_mse = mse(residuals, current_predictions)
                
                if current_mse < min_mse:
                    min_mse = current_mse
                    best_feature, best_threshold = feature, threshold
                    best_value = (left_value, right_value)
        
        # Update predictions
        predictions = [p + learning_rate * (best_value[0] if x[best_feature] <= best_threshold else best_value[1]) 
                       for p, x in zip(predictions, X)]
    
    return predictions

# Example usage
X = [[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]]
y = [1.2, 2.3, 2.8, 3.5, 4.1]

predictions = gradient_boosting(X, y)
print("Gradient Boosting predictions:", predictions)
print("Actual values:                ", y)
```

Slide 5: XGBoost: Extreme Gradient Boosting

XGBoost (Extreme Gradient Boosting) is an advanced implementation of gradient boosting that offers improved performance and speed. It uses a more regularized model formalization to control overfitting, which often results in better performance. XGBoost is highly flexible and can be used for both regression and classification tasks.

```python
import math
import random

class XGBoostTree:
    def __init__(self, max_depth=3):
        self.max_depth = max_depth
        self.tree = {}

    def _mse(self, y):
        return sum((yi - sum(y) / len(y)) ** 2 for yi in y)

    def _split(self, X, y, feature, threshold):
        left = [(xi, yi) for xi, yi in zip(X, y) if xi[feature] <= threshold]
        right = [(xi, yi) for xi, yi in zip(X, y) if xi[feature] > threshold]
        return left, right

    def _find_best_split(self, X, y):
        best_feature, best_threshold, best_gain = None, None, 0
        current_mse = self._mse(y)

        for feature in range(len(X[0])):
            for threshold in set(xi[feature] for xi in X):
                left, right = self._split(X, y, feature, threshold)
                if len(left) == 0 or len(right) == 0:
                    continue

                left_y, right_y = [yi for _, yi in left], [yi for _, yi in right]
                gain = current_mse - (len(left_y) / len(y) * self._mse(left_y) + 
                                      len(right_y) / len(y) * self._mse(right_y))

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold, best_gain

    def _build_tree(self, X, y, depth=0):
        if depth == self.max_depth or len(set(y)) == 1:
            return sum(y) / len(y)

        feature, threshold, gain = self._find_best_split(X, y)

        if gain == 0:
            return sum(y) / len(y)

        left, right = self._split(X, y, feature, threshold)
        return {
            'feature': feature,
            'threshold': threshold,
            'left': self._build_tree([xi for xi, _ in left], [yi for _, yi in left], depth + 1),
            'right': self._build_tree([xi for xi, _ in right], [yi for _, yi in right], depth + 1)
        }

    def fit(self, X, y):
        self.tree = self._build_tree(X, y)

    def predict_single(self, x, tree=None):
        if tree is None:
            tree = self.tree

        if not isinstance(tree, dict):
            return tree

        if x[tree['feature']] <= tree['threshold']:
            return self.predict_single(x, tree['left'])
        else:
            return self.predict_single(x, tree['right'])

    def predict(self, X):
        return [self.predict_single(x) for x in X]

class XGBoost:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees = []

    def fit(self, X, y):
        self.base_prediction = sum(y) / len(y)
        residuals = [yi - self.base_prediction for yi in y]

        for _ in range(self.n_estimators):
            tree = XGBoostTree(max_depth=self.max_depth)
            tree.fit(X, residuals)
            predictions = tree.predict(X)
            residuals = [r - self.learning_rate * p for r, p in zip(residuals, predictions)]
            self.trees.append(tree)

    def predict(self, X):
        predictions = [self.base_prediction] * len(X)
        for tree in self.trees:
            predictions = [p + self.learning_rate * tp for p, tp in zip(predictions, tree.predict(X))]
        return predictions

# Example usage
X = [[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]]
y = [1.2, 2.3, 2.8, 3.5, 4.1]

xgb = XGBoost(n_estimators=10, learning_rate=0.1, max_depth=3)
xgb.fit(X, y)
predictions = xgb.predict(X)

print("XGBoost predictions:", predictions)
print("Actual values:      ", y)
```

Slide 6: Real-Life Example: Customer Churn Prediction

Let's apply boosting to predict customer churn in a telecommunications company. We'll use a simplified dataset with features like customer tenure, monthly charges, and total charges. Our goal is to predict whether a customer is likely to churn (leave the company) or not.

```python
import random
import math

def generate_customer_data(n_samples=1000):
    data = []
    for _ in range(n_samples):
        tenure = random.randint(1, 72)
        monthly_charges = random.uniform(20, 120)
        total_charges = tenure * monthly_charges * (1 + random.uniform(-0.1, 0.1))
        churn = 1 if random.random() < 0.2 + 0.01 * (70 - tenure) + 0.002 * (monthly_charges - 70) else 0
        data.append([tenure, monthly_charges, total_charges, churn])
    return data

def adaboost_churn(X, y, num_estimators=10):
    n_samples = len(X)
    weights = [1/n_samples] * n_samples
    estimators = []
    estimator_weights = []

    for _ in range(num_estimators):
        best_feature, best_threshold, min_error = 0, 0, float('inf')
        for feature in range(len(X[0])):
            for sample in X:
                threshold = sample[feature]
                predictions = [1 if x[feature] > threshold else 0 for x in X]
                error = sum(w for i, w in enumerate(weights) if predictions[i] != y[i])
                if error < min_error:
                    min_error = error
                    best_feature, best_threshold = feature, threshold
        
        estimator_weight = 0.5 * math.log((1 - min_error) / max(min_error, 1e-10))
        predictions = [1 if x[best_feature] > best_threshold else 0 for x in X]
        weights = [w * math.exp(-estimator_weight * (2*y[i]-1) * (2*predictions[i]-1)) 
                   for i, w in enumerate(weights)]
        weights = [w / sum(weights) for w in weights]
        
        estimators.append((best_feature, best_threshold))
        estimator_weights.append(estimator_weight)

    return estimators, estimator_weights

# Generate data and train model
data = generate_customer_data()
X = [d[:-1] for d in data]  # Features
y = [d[-1] for d in data]   # Churn labels

estimators, estimator_weights = adaboost_churn(X, y)

# Predict function
def predict(X, estimators, estimator_weights):
    predictions = []
    for x in X:
        score = sum(weight * (1 if x[feature] > threshold else 0) 
                    for (feature, threshold), weight in zip(estimators, estimator_weights))
        predictions.append(1 if score > 0 else 0)
    return predictions

# Make predictions on the first 5 samples
sample_predictions = predict(X[:5], estimators, estimator_weights)
print("Sample predictions:", sample_predictions)
print("Actual labels:     ", y[:5])
```

Slide 7: Real-Life Example: Image Classification

Let's explore how boosting can be applied to image classification tasks. We'll use a simplified version of the MNIST dataset, where we'll classify handwritten digits as either even or odd. This example demonstrates how boosting can be used in more complex scenarios.

```python
import random
import math

def generate_simplified_mnist(n_samples=1000):
    data = []
    for _ in range(n_samples):
        # Simulating a 28x28 image with 784 pixels
        pixel_values = [random.randint(0, 255) for _ in range(784)]
        digit = random.randint(0, 9)
        label = 0 if digit % 2 == 0 else 1  # 0 for even, 1 for odd
        data.append((pixel_values, label))
    return data

def weak_learner(X, y, weights):
    n_features = len(X[0])
    best_feature, best_threshold, min_weighted_error = 0, 0, float('inf')
    
    for feature in range(n_features):
        feature_values = [x[feature] for x in X]
        thresholds = sorted(set(feature_values))
        
        for threshold in thresholds:
            predictions = [1 if x[feature] > threshold else 0 for x in X]
            weighted_error = sum(w for i, (p, y_true, w) in enumerate(zip(predictions, y, weights)) if p != y_true)
            
            if weighted_error < min_weighted_error:
                min_weighted_error = weighted_error
                best_feature = feature
                best_threshold = threshold
    
    return best_feature, best_threshold, min_weighted_error

def adaboost_image(X, y, num_estimators=10):
    n_samples = len(X)
    weights = [1/n_samples] * n_samples
    estimators = []
    estimator_weights = []

    for _ in range(num_estimators):
        best_feature, best_threshold, error = weak_learner(X, y, weights)
        estimator_weight = 0.5 * math.log((1 - error) / max(error, 1e-10))
        
        predictions = [1 if x[best_feature] > best_threshold else 0 for x in X]
        weights = [w * math.exp(-estimator_weight * (2*y_true-1) * (2*pred-1)) 
                   for w, y_true, pred in zip(weights, y, predictions)]
        weights = [w / sum(weights) for w in weights]
        
        estimators.append((best_feature, best_threshold))
        estimator_weights.append(estimator_weight)

    return estimators, estimator_weights

# Generate data and train model
data = generate_simplified_mnist()
X, y = zip(*data)

estimators, estimator_weights = adaboost_image(X, y)

def predict(X, estimators, estimator_weights):
    predictions = []
    for x in X:
        score = sum(weight * (1 if x[feature] > threshold else 0) 
                    for (feature, threshold), weight in zip(estimators, estimator_weights))
        predictions.append(1 if score > 0 else 0)
    return predictions

# Make predictions on the first 5 samples
sample_predictions = predict(X[:5], estimators, estimator_weights)
print("Sample predictions:", sample_predictions)
print("Actual labels:     ", y[:5])
```

Slide 8: Boosting vs. Other Ensemble Methods

Boosting is just one type of ensemble method in machine learning. Let's compare it with other popular ensemble techniques like bagging and random forests. We'll implement simple versions of these methods to highlight their differences.

```python
import random

def majority_vote(predictions):
    return max(set(predictions), key=predictions.count)

def bootstrap_sample(X, y):
    n_samples = len(X)
    indices = [random.randint(0, n_samples - 1) for _ in range(n_samples)]
    return [X[i] for i in indices], [y[i] for i in indices]

def decision_stump(X, y):
    n_features = len(X[0])
    best_feature, best_threshold, min_error = 0, 0, float('inf')
    
    for feature in range(n_features):
        thresholds = sorted(set(x[feature] for x in X))
        for threshold in thresholds:
            predictions = [1 if x[feature] > threshold else 0 for x in X]
            error = sum(1 for p, y_true in zip(predictions, y) if p != y_true)
            if error < min_error:
                min_error = error
                best_feature = feature
                best_threshold = threshold
    
    return best_feature, best_threshold

def bagging(X, y, n_estimators=10):
    estimators = []
    for _ in range(n_estimators):
        X_sample, y_sample = bootstrap_sample(X, y)
        estimator = decision_stump(X_sample, y_sample)
        estimators.append(estimator)
    return estimators

def random_forest(X, y, n_estimators=10, max_features=None):
    if max_features is None:
        max_features = int(len(X[0]) ** 0.5)
    
    estimators = []
    for _ in range(n_estimators):
        X_sample, y_sample = bootstrap_sample(X, y)
        feature_subset = random.sample(range(len(X[0])), max_features)
        X_subset = [[x[i] for i in feature_subset] for x in X_sample]
        estimator = decision_stump(X_subset, y_sample)
        estimators.append((feature_subset, estimator))
    return estimators

# Generate some sample data
X = [[random.random() for _ in range(5)] for _ in range(100)]
y = [random.randint(0, 1) for _ in range(100)]

# Train models
bagging_model = bagging(X, y)
rf_model = random_forest(X, y)
boosting_model, _ = adaboost_churn(X, y)  # Using the AdaBoost implementation from earlier

# Make predictions
def predict_bagging(X, model):
    predictions = []
    for x in X:
        votes = [1 if x[feature] > threshold else 0 for feature, threshold in model]
        predictions.append(majority_vote(votes))
    return predictions

def predict_rf(X, model):
    predictions = []
    for x in X:
        votes = [1 if x[feature_subset[feature]] > threshold else 0 
                 for feature_subset, (feature, threshold) in model]
        predictions.append(majority_vote(votes))
    return predictions

bagging_preds = predict_bagging(X[:5], bagging_model)
rf_preds = predict_rf(X[:5], rf_model)
boosting_preds = predict(X[:5], boosting_model, [1]*len(boosting_model))  # Using equal weights for simplicity

print("Bagging predictions:  ", bagging_preds)
print("Random Forest preds:  ", rf_preds)
print("Boosting predictions: ", boosting_preds)
print("Actual labels:        ", y[:5])
```

Slide 9: Hyperparameter Tuning in Boosting

Hyperparameter tuning is crucial for optimizing the performance of boosting algorithms. Let's implement a simple grid search to find the best hyperparameters for our AdaBoost model. We'll use cross-validation to evaluate different combinations of hyperparameters.

```python
import random
import math

def generate_data(n_samples=1000):
    X = [[random.random() for _ in range(5)] for _ in range(n_samples)]
    y = [random.randint(0, 1) for _ in range(n_samples)]
    return X, y

def adaboost(X, y, num_estimators=10, learning_rate=1.0):
    n_samples = len(X)
    weights = [1/n_samples] * n_samples
    estimators = []
    estimator_weights = []

    for _ in range(num_estimators):
        best_feature, best_threshold, min_error = 0, 0, float('inf')
        for feature in range(len(X[0])):
            thresholds = sorted(set(x[feature] for x in X))
            for threshold in thresholds:
                predictions = [1 if x[feature] > threshold else 0 for x in X]
                error = sum(w for i, (p, y_true, w) in enumerate(zip(predictions, y, weights)) if p != y_true)
                if error < min_error:
                    min_error = error
                    best_feature, best_threshold = feature, threshold
        
        estimator_weight = learning_rate * 0.5 * math.log((1 - min_error) / max(min_error, 1e-10))
        predictions = [1 if x[best_feature] > best_threshold else 0 for x in X]
        weights = [w * math.exp(-estimator_weight * (2*y_true-1) * (2*pred-1)) 
                   for w, y_true, pred in zip(weights, y, predictions)]
        weights = [w / sum(weights) for w in weights]
        
        estimators.append((best_feature, best_threshold))
        estimator_weights.append(estimator_weight)

    return estimators, estimator_weights

def predict(X, estimators, estimator_weights):
    predictions = []
    for x in X:
        score = sum(weight * (1 if x[feature] > threshold else 0) 
                    for (feature, threshold), weight in zip(estimators, estimator_weights))
        predictions.append(1 if score > 0 else 0)
    return predictions

def cross_validation(X, y, num_folds=5):
    fold_size = len(X) // num_folds
    for i in range(num_folds):
        test_start = i * fold_size
        test_end = (i + 1) * fold_size
        
        X_train = X[:test_start] + X[test_end:]
        y_train = y[:test_start] + y[test_end:]
        X_test = X[test_start:test_end]
        y_test = y[test_start:test_end]
        
        yield X_train, y_train, X_test, y_test

def grid_search(X, y, param_grid):
    best_score = 0
    best_params = None
    
    for num_estimators in param_grid['num_estimators']:
        for learning_rate in param_grid['learning_rate']:
            scores = []
            for X_train, y_train, X_test, y_test in cross_validation(X, y):
                estimators, estimator_weights = adaboost(X_train, y_train, num_estimators, learning_rate)
                predictions = predict(X_test, estimators, estimator_weights)
                accuracy = sum(1 for p, y_true in zip(predictions, y_test) if p == y_true) / len(y_test)
                scores.append(accuracy)
            
            avg_score = sum(scores) / len(scores)
            if avg_score > best_score:
                best_score = avg_score
                best_params = {'num_estimators': num_estimators, 'learning_rate': learning_rate}
    
    return best_params, best_score

# Generate data
X, y = generate_data()

# Define parameter grid
param_grid = {
    'num_estimators': [10, 50, 100],
    'learning_rate': [0.1, 0.5, 1.0]
}

# Perform grid search
best_params, best_score = grid_search(X, y, param_grid)

print("Best parameters:", best_params)
print("Best cross-validation score:", best_score)

# Train final model with best parameters
final_estimators, final_weights = adaboost(X, y, **best_params)

# Make predictions on a few samples
sample_predictions = predict(X[:5], final_estimators, final_weights)
print("Sample predictions:", sample_predictions)
print("Actual labels:     ", y[:5])
```

Slide 10: Boosting for Regression

While we've focused on classification tasks, boosting can also be applied to regression problems. Let's implement a simple version of gradient boosting for regression to predict house prices based on various features.

```python
import random
import math

def generate_house_data(n_samples=1000):
    data = []
    for _ in range(n_samples):
        size = random.uniform(1000, 3000)
        bedrooms = random.randint(2, 5)
        age = random.uniform(0, 50)
        price = 100000 + 100 * size + 10000 * bedrooms - 1000 * age + random.uniform(-20000, 20000)
        data.append(([size, bedrooms, age], price))
    return data

def decision_tree_regressor(X, y, max_depth=3):
    if max_depth == 0 or len(set(y)) == 1:
        return sum(y) / len(y)

    best_feature, best_threshold, best_mse = None, None, float('inf')
    for feature in range(len(X[0])):
        thresholds = sorted(set(x[feature] for x in X))
        for threshold in thresholds:
            left = [i for i, x in enumerate(X) if x[feature] <= threshold]
            right = [i for i, x in enumerate(X) if x[feature] > threshold]
            if len(left) == 0 or len(right) == 0:
                continue
            mse = (sum((y[i] - sum(y[j] for j in left) / len(left))**2 for i in left) +
                   sum((y[i] - sum(y[j] for j in right) / len(right))**2 for i in right)) / len(y)
            if mse < best_mse:
                best_mse = mse
                best_feature = feature
                best_threshold = threshold

    if best_feature is None:
        return sum(y) / len(y)

    left = [(X[i], y[i]) for i in range(len(X)) if X[i][best_feature] <= best_threshold]
    right = [(X[i], y[i]) for i in range(len(X)) if X[i][best_feature] > best_threshold]
    
    return {
        'feature': best_feature,
        'threshold': best_threshold,
        'left': decision_tree_regressor([x for x, _ in left], [y for _, y in left], max_depth - 1),
        'right': decision_tree_regressor([x for x, _ in right], [y for _, y in right], max_depth - 1)
    }

def predict_tree(tree, x):
    if not isinstance(tree, dict):
        return tree
    if x[tree['feature']] <= tree['threshold']:
        return predict_tree(tree['left'], x)
    else:
        return predict_tree(tree['right'], x)

def gradient_boosting_regressor(X, y, n_estimators=100, learning_rate=0.1, max_depth=3):
    trees = []
    predictions = [0] * len(y)
    
    for _ in range(n_estimators):
        residuals = [y_true - pred for y_true, pred in zip(y, predictions)]
        tree = decision_tree_regressor(X, residuals, max_depth)
        trees.append(tree)
        predictions = [pred + learning_rate * predict_tree(tree, x) for pred, x in zip(predictions, X)]
    
    return trees

def predict_gbr(trees, X, learning_rate):
    predictions = [0] * len(X)
    for tree in trees:
        predictions = [pred + learning_rate * predict_tree(tree, x) for pred, x in zip(predictions, X)]
    return predictions

# Generate data and train model
data = generate_house_data()
X, y = zip(*data)
trees = gradient_boosting_regressor(X, y)

# Make predictions on a few samples
sample_predictions = predict_gbr(trees, X[:5], 0.1)
print("Sample predictions:", sample_predictions)
print("Actual prices:     ", y[:5])
```

Slide 11: Handling Imbalanced Datasets with Boosting

Imbalanced datasets, where one class significantly outnumbers the other, can pose challenges for machine learning algorithms. Boosting techniques can be adapted to handle such scenarios. Let's implement a modified version of AdaBoost that gives more weight to the minority class.

```python
import random
import math

def generate_imbalanced_data(n_samples=1000, imbalance_ratio=0.1):
    minority_samples = int(n_samples * imbalance_ratio)
    majority_samples = n_samples - minority_samples
    
    minority_data = [(1, 1) for _ in range(minority_samples)]
    majority_data = [(0, 0) for _ in range(majority_samples)]
    
    data = minority_data + majority_data
    random.shuffle(data)
    
    X, y = zip(*data)
    return list(X), list(y)

def weighted_adaboost(X, y, num_estimators=50, class_weight=None):
    n_samples = len(X)
    if class_weight is None:
        weights = [1/n_samples] * n_samples
    else:
        weights = [class_weight[label]/sum(1 for l in y if l == label) for label in y]
    
    estimators = []
    estimator_weights = []

    for _ in range(num_estimators):
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
        
        best_feature, best_threshold, min_error = 0, 0, float('inf')
        for feature in range(len(X[0])):
            thresholds = sorted(set(x[feature] for x in X))
            for threshold in thresholds:
                predictions = [1 if x[feature] > threshold else 0 for x in X]
                error = sum(w for i, (p, y_true, w) in enumerate(zip(predictions, y, weights)) if p != y_true)
                if error < min_error:
                    min_error = error
                    best_feature, best_threshold = feature, threshold
        
        estimator_weight = 0.5 * math.log((1 - min_error) / max(min_error, 1e-10))
        predictions = [1 if x[best_feature] > best_threshold else 0 for x in X]
        weights = [w * math.exp(-estimator_weight * (2*y_true-1) * (2*pred-1)) 
                   for w, y_true, pred in zip(weights, y, predictions)]
        
        estimators.append((best_feature, best_threshold))
        estimator_weights.append(estimator_weight)

    return estimators, estimator_weights

def predict_weighted(X, estimators, estimator_weights):
    predictions = []
    for x in X:
        score = sum(weight * (1 if x[feature] > threshold else 0) 
                    for (feature, threshold), weight in zip(estimators, estimator_weights))
        predictions.append(1 if score > 0 else 0)
    return predictions

# Generate imbalanced data
X, y = generate_imbalanced_data()

# Calculate class weights
class_weight = {0: 1, 1: sum(y) / len(y)}

# Train model
estimators, estimator_weights = weighted_adaboost(X, y, class_weight=class_weight)

# Make predictions
predictions = predict_weighted(X, estimators, estimator_weights)

# Calculate metrics
accuracy = sum(1 for p, y_true in zip(predictions, y) if p == y_true) / len(y)
precision = sum(1 for p, y_true in zip(predictions, y) if p == 1 and y_true == 1) / sum(predictions)
recall = sum(1 for p, y_true in zip(predictions, y) if p == 1 and y_true == 1) / sum(y)
f1_score = 2 * (precision * recall) / (precision + recall)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1_score:.4f}")
```

Slide 12: Boosting in Natural Language Processing

Boosting techniques can be applied to natural language processing tasks, such as sentiment analysis. Let's implement a simple boosting-based sentiment classifier using bag-of-words features.

```python
import random
import math
import re
from collections import Counter

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text.split()

def create_vocabulary(texts):
    words = [word for text in texts for word in preprocess_text(text)]
    return list(set(words))

def text_to_bow(text, vocabulary):
    words = preprocess_text(text)
    return [words.count(word) for word in vocabulary]

def generate_sentiment_data(n_samples=1000):
    positive_words = ["good", "great", "excellent", "awesome", "fantastic"]
    negative_words = ["bad", "terrible", "awful", "horrible", "poor"]
    neutral_words = ["the", "a", "an", "is", "are", "was", "were"]
    
    data = []
    for _ in range(n_samples):
        sentiment = random.choice([0, 1])  # 0 for negative, 1 for positive
        if sentiment == 1:
            words = random.choices(positive_words, k=random.randint(1, 3)) + random.choices(neutral_words, k=random.randint(2, 5))
        else:
            words = random.choices(negative_words, k=random.randint(1, 3)) + random.choices(neutral_words, k=random.randint(2, 5))
        random.shuffle(words)
        text = " ".join(words)
        data.append((text, sentiment))
    return data

def adaboost_sentiment(X, y, num_estimators=50):
    n_samples = len(X)
    weights = [1/n_samples] * n_samples
    estimators = []
    estimator_weights = []

    for _ in range(num_estimators):
        best_feature, best_threshold, min_error = 0, 0, float('inf')
        for feature in range(len(X[0])):
            thresholds = sorted(set(x[feature] for x in X))
            for threshold in thresholds:
                predictions = [1 if x[feature] > threshold else 0 for x in X]
                error = sum(w for i, (p, y_true, w) in enumerate(zip(predictions, y, weights)) if p != y_true)
                if error < min_error:
                    min_error = error
                    best_feature, best_threshold = feature, threshold
        
        estimator_weight = 0.5 * math.log((1 - min_error) / max(min_error, 1e-10))
        predictions = [1 if x[best_feature] > best_threshold else 0 for x in X]
        weights = [w * math.exp(-estimator_weight * (2*y_true-1) * (2*pred-1)) 
                   for w, y_true, pred in zip(weights, y, predictions)]
        weights = [w / sum(weights) for w in weights]
        
        estimators.append((best_feature, best_threshold))
        estimator_weights.append(estimator_weight)

    return estimators, estimator_weights

def predict_sentiment(X, estimators, estimator_weights):
    predictions = []
    for x in X:
        score = sum(weight * (1 if x[feature] > threshold else 0) 
                    for (feature, threshold), weight in zip(estimators, estimator_weights))
        predictions.append(1 if score > 0 else 0)
    return predictions

# Generate sentiment data
data = generate_sentiment_data()
texts, labels = zip(*data)

# Create vocabulary and convert texts to bag-of-words
vocabulary = create_vocabulary(texts)
X = [text_to_bow(text, vocabulary) for text in texts]
y = list(labels)

# Train model
estimators, estimator_weights = adaboost_sentiment(X, y)

# Make predictions on a few samples
sample_texts = [
    "This movie is great and awesome",
    "The food was terrible and awful",
    "The service was okay"
]
sample_X = [text_to_bow(text, vocabulary) for text in sample_texts]
sample_predictions = predict_sentiment(sample_X, estimators, estimator_weights)

for text, prediction in zip(sample_texts, sample_predictions):
    sentiment = "Positive" if prediction == 1 else "Negative"
    print(f"Text: '{text}'\nPredicted sentiment: {sentiment}\n")
```

Slide 13: Boosting in Time Series Forecasting

Boosting can be applied to time series forecasting tasks. Let's implement a simple boosting-based model for predicting future values in a time series.

```python
import random
import math

def generate_time_series(n_samples=1000):
    trend = [i * 0.1 for i in range(n_samples)]
    seasonality = [math.sin(2 * math.pi * i / 365) * 10 for i in range(n_samples)]
    noise = [random.gauss(0, 1) for _ in range(n_samples)]
    return [t + s + n for t, s, n in zip(trend, seasonality, noise)]

def create_features(time_series, lag=5):
    X, y = [], []
    for i in range(len(time_series) - lag):
        X.append(time_series[i:i+lag])
        y.append(time_series[i+lag])
    return X, y

def decision_tree_regressor(X, y, max_depth=3):
    def build_tree(X, y, depth=0):
        if depth == max_depth or len(set(y)) == 1:
            return sum(y) / len(y)

        best_feature, best_threshold, best_mse = None, None, float('inf')
        for feature in range(len(X[0])):
            thresholds = sorted(set(x[feature] for x in X))
            for threshold in thresholds:
                left = [i for i, x in enumerate(X) if x[feature] <= threshold]
                right = [i for i, x in enumerate(X) if x[feature] > threshold]
                if len(left) == 0 or len(right) == 0:
                    continue
                mse = (sum((y[i] - sum(y[j] for j in left) / len(left))**2 for i in left) +
                       sum((y[i] - sum(y[j] for j in right) / len(right))**2 for i in right)) / len(y)
                if mse < best_mse:
                    best_mse = mse
                    best_feature = feature
                    best_threshold = threshold

        if best_feature is None:
            return sum(y) / len(y)

        left = [(X[i], y[i]) for i in range(len(X)) if X[i][best_feature] <= best_threshold]
        right = [(X[i], y[i]) for i in range(len(X)) if X[i][best_feature] > best_threshold]
        
        return {
            'feature': best_feature,
            'threshold': best_threshold,
            'left': build_tree([x for x, _ in left], [y for _, y in left], depth + 1),
            'right': build_tree([x for x, _ in right], [y for _, y in right], depth + 1)
        }
    
    return build_tree(X, y)

def predict_tree(tree, x):
    if not isinstance(tree, dict):
        return tree
    if x[tree['feature']] <= tree['threshold']:
        return predict_tree(tree['left'], x)
    else:
        return predict_tree(tree['right'], x)

def gradient_boosting_regressor(X, y, n_estimators=100, learning_rate=0.1, max_depth=3):
    trees = []
    predictions = [0] * len(y)
    
    for _ in range(n_estimators):
        residuals = [y_true - pred for y_true, pred in zip(y, predictions)]
        tree = decision_tree_regressor(X, residuals, max_depth)
        trees.append(tree)
        predictions = [pred + learning_rate * predict_tree(tree, x) for pred, x in zip(predictions, X)]
    
    return trees

def predict_gbr(trees, X, learning_rate):
    predictions = [0] * len(X)
    for tree in trees:
        predictions = [pred + learning_rate * predict_tree(tree, x) for pred, x in zip(predictions, X)]
    return predictions

# Generate time series data
time_series = generate_time_series()

# Create features and labels
X, y = create_features(time_series)

# Train the model
trees = gradient_boosting_regressor(X, y)

# Make predictions for the next 5 time steps
last_known = time_series[-5:]
predictions = []
for _ in range(5):
    pred = predict_gbr(trees, [last_known], 0.1)[0]
    predictions.append(pred)
    last_known = last_known[1:] + [pred]

print("Last 5 known values:", time_series[-5:])
print("Predictions for next 5 time steps:", predictions)
```

Slide 14: Boosting for Feature Selection

Boosting algorithms can be used for feature selection by analyzing the importance of features across multiple weak learners. Let's implement a simple feature selection method using AdaBoost.

```python
import random
import math

def generate_data_with_irrelevant_features(n_samples=1000, n_features=20, n_relevant=5):
    X = [[random.gauss(0, 1) for _ in range(n_features)] for _ in range(n_samples)]
    y = [1 if sum(x[:n_relevant]) > 0 else 0 for x in X]
    return X, y

def adaboost_feature_selection(X, y, num_estimators=50):
    n_samples, n_features = len(X), len(X[0])
    weights = [1/n_samples] * n_samples
    feature_importance = [0] * n_features

    for _ in range(num_estimators):
        best_feature, best_threshold, min_error = 0, 0, float('inf')
        for feature in range(n_features):
            thresholds = sorted(set(x[feature] for x in X))
            for threshold in thresholds:
                predictions = [1 if x[feature] > threshold else 0 for x in X]
                error = sum(w for i, (p, y_true, w) in enumerate(zip(predictions, y, weights)) if p != y_true)
                if error < min_error:
                    min_error = error
                    best_feature, best_threshold = feature, threshold
        
        estimator_weight = 0.5 * math.log((1 - min_error) / max(min_error, 1e-10))
        predictions = [1 if x[best_feature] > best_threshold else 0 for x in X]
        weights = [w * math.exp(-estimator_weight * (2*y_true-1) * (2*pred-1)) 
                   for w, y_true, pred in zip(weights, y, predictions)]
        weights = [w / sum(weights) for w in weights]
        
        feature_importance[best_feature] += estimator_weight

    return feature_importance

# Generate data with irrelevant features
X, y = generate_data_with_irrelevant_features()

# Perform feature selection
feature_importance = adaboost_feature_selection(X, y)

# Sort features by importance
sorted_features = sorted(enumerate(feature_importance), key=lambda x: x[1], reverse=True)

print("Top 10 most important features:")
for feature, importance in sorted_features[:10]:
    print(f"Feature {feature}: {importance:.4f}")

# Select top K features
K = 5
selected_features = [feature for feature, _ in sorted_features[:K]]

print(f"\nSelected top {K} features: {selected_features}")

# Create new dataset with selected features
X_selected = [[x[i] for i in selected_features] for x in X]

print(f"\nOriginal dataset shape: {len(X)} samples, {len(X[0])} features")
print(f"Selected dataset shape: {len(X_selected)} samples, {len(X_selected[0])} features")
```

Slide 15: Additional Resources

For those interested in diving deeper into boosting algorithms and their applications, here are some valuable resources:

1.  "A Short Introduction to Boosting" by Y. Freund and R. Schapire (1999) ArXiv: [https://arxiv.org/abs/0905.2361](https://arxiv.org/abs/0905.2361)
2.  "Boosting Algorithms: Regularization, Prediction and Model Fitting" by P. Bühlmann and T. Hothorn (2007) ArXiv: [https://arxiv.org/abs/0804.2752](https://arxiv.org/abs/0804.2752)
3.  "XGBoost: A Scalable Tree Boosting System" by T. Chen and C. Guestrin (2016) ArXiv: [https://arxiv.org/abs/1603.02754](https://arxiv.org/abs/1603.02754)
4.  "LightGBM: A Highly Efficient Gradient Boosting Decision Tree" by G. Ke et al. (2017) ArXiv: [https://arxiv.org/abs/1711.08180](https://arxiv.org/abs/1711.08180)

These papers provide in-depth explanations of various boosting algorithms, their theoretical foundations, and practical applications in machine learning.

I'm glad I could provide the slideshow on Machine Learning: A Visual Guide to Boosting. The 15 slides cover a comprehensive overview of boosting techniques, from basic concepts to advanced applications in various domains. Is there anything specific about the slides or boosting algorithms you'd like to discuss further or any questions you have?

