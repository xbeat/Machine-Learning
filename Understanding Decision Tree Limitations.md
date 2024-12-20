## Understanding Decision Tree Limitations
Slide 1: Understanding Decision Tree Limitations

Traditional decision trees suffer from axis-parallel splits, meaning they can only create rectangular decision boundaries by splitting features along vertical or horizontal lines. This fundamental limitation makes them struggle with diagonal or non-linear decision boundaries.

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate diagonal separation data
np.random.seed(42)
n_points = 1000
X = np.random.uniform(0, 10, (n_points, 2))
y = (X[:, 1] > X[:, 0]).astype(int)

# Add noise around diagonal
noise_mask = np.abs(X[:, 1] - X[:, 0]) < 0.5
y[noise_mask] = np.random.binomial(1, 0.5, np.sum(noise_mask))

# Visualize limitations
plt.figure(figsize=(10, 6))
plt.scatter(X[y == 0, 0], X[y == 0, 1], label='Class 0')
plt.scatter(X[y == 1, 0], X[y == 1, 1], label='Class 1')
plt.axvline(x=5, color='r', linestyle='--', label='Vertical Split')
plt.axhline(y=5, color='g', linestyle='--', label='Horizontal Split')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.title('Traditional Decision Tree Splits')
plt.show()
```

Slide 2: Implementing Rotational Trees Base Class

The RotationalTree foundation requires a base class that handles data rotation and implements core decision tree functionality. This class will serve as the building block for our enhanced tree algorithms.

```python
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

class RotationalTreeBase(BaseEstimator, ClassifierMixin):
    def __init__(self, max_depth=5, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree_ = None
        
    def _rotate_data(self, X, angle):
        """Rotate feature space by given angle"""
        theta = np.radians(angle)
        rotation_matrix = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])
        return np.dot(X, rotation_matrix)
    
    def _calculate_optimal_rotation(self, X, y):
        """Find optimal rotation angle using grid search"""
        angles = np.linspace(0, 180, 36)  # Try 36 different angles
        best_score = float('-inf')
        best_angle = 0
        
        for angle in angles:
            X_rot = self._rotate_data(X, angle)
            score = self._calculate_split_score(X_rot, y)
            if score > best_score:
                best_score = score
                best_angle = angle
                
        return best_angle
```

Slide 3: Optimal Split Calculation

The effectiveness of rotational trees depends heavily on finding the optimal split point after rotation. This implementation uses information gain with Gini impurity as the splitting criterion for binary classification.

```python
def _calculate_split_score(self, X, y):
    """Calculate split score using Gini impurity"""
    def gini(y):
        if len(y) == 0:
            return 0
        p = np.bincount(y) / len(y)
        return 1 - np.sum(p ** 2)
    
    best_score = float('-inf')
    n_features = X.shape[1]
    
    for feature in range(n_features):
        thresholds = np.unique(X[:, feature])
        for threshold in thresholds:
            left_mask = X[:, feature] <= threshold
            right_mask = ~left_mask
            
            # Calculate weighted Gini impurity
            n_left = np.sum(left_mask)
            n_right = np.sum(right_mask)
            n_total = len(y)
            
            if n_left == 0 or n_right == 0:
                continue
                
            gini_left = gini(y[left_mask])
            gini_right = gini(y[right_mask])
            score = -((n_left/n_total) * gini_left + 
                     (n_right/n_total) * gini_right)
            
            if score > best_score:
                best_score = score
    
    return best_score
```

Slide 4: Tree Node Implementation

The Node class forms the fundamental structure of our rotational decision tree, storing split information, rotation angles, and prediction values. This implementation supports both binary and multi-class classification.

```python
class Node:
    def __init__(self):
        self.left = None
        self.right = None
        self.feature_index = None
        self.threshold = None
        self.rotation_angle = None
        self.prediction = None
        
    def is_leaf(self):
        return self.prediction is not None

class RotationalTreeNode(Node):
    def __init__(self, **kwargs):
        super().__init__()
        self.gain = None
        self.n_samples = None
        self.depth = None
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def to_dict(self):
        """Convert node to dictionary for visualization"""
        node_dict = {
            'rotation_angle': self.rotation_angle,
            'feature_index': self.feature_index,
            'threshold': self.threshold,
            'prediction': self.prediction,
            'n_samples': self.n_samples
        }
        if not self.is_leaf():
            node_dict['left'] = self.left.to_dict()
            node_dict['right'] = self.right.to_dict()
        return node_dict
```

Slide 5: Building the Rotational Tree

The core tree building process involves recursively creating nodes with optimal rotations and splits. This implementation includes early stopping criteria and handles the recursive nature of decision tree construction.

```python
def _build_tree(self, X, y, depth=0):
    node = RotationalTreeNode(depth=depth, n_samples=len(y))
    
    # Check stopping criteria
    if (depth >= self.max_depth or 
        len(y) < self.min_samples_split or 
        len(np.unique(y)) == 1):
        node.prediction = np.argmax(np.bincount(y))
        return node
    
    # Find optimal rotation
    rotation_angle = self._calculate_optimal_rotation(X, y)
    X_rotated = self._rotate_data(X, rotation_angle)
    
    # Find best split
    best_score = float('-inf')
    best_feature = None
    best_threshold = None
    
    for feature in range(X.shape[1]):
        thresholds = np.unique(X_rotated[:, feature])
        for threshold in thresholds:
            left_mask = X_rotated[:, feature] <= threshold
            score = self._calculate_split_score(X_rotated[left_mask], y[left_mask])
            if score > best_score:
                best_score = score
                best_feature = feature
                best_threshold = threshold
    
    # Create split
    node.rotation_angle = rotation_angle
    node.feature_index = best_feature
    node.threshold = best_threshold
    
    # Recursive splitting
    left_mask = X_rotated[:, best_feature] <= best_threshold
    node.left = self._build_tree(X[left_mask], y[left_mask], depth + 1)
    node.right = self._build_tree(X[~left_mask], y[~left_mask], depth + 1)
    
    return node
```

Slide 6: Prediction Implementation

The prediction process involves traversing the tree while applying the appropriate rotations at each node. This implementation handles both single sample and batch predictions efficiently.

```python
def predict(self, X):
    return np.array([self._predict_single(x) for x in X])

def _predict_single(self, x):
    node = self.tree_
    while not node.is_leaf():
        # Apply rotation
        x_rot = self._rotate_data(x.reshape(1, -1), node.rotation_angle)
        # Navigate tree
        if x_rot[0, node.feature_index] <= node.threshold:
            node = node.left
        else:
            node = node.right
    return node.prediction

def predict_proba(self, X):
    """Predict class probabilities for X"""
    predictions = []
    for x in X:
        node = self.tree_
        while not node.is_leaf():
            x_rot = self._rotate_data(x.reshape(1, -1), node.rotation_angle)
            if x_rot[0, node.feature_index] <= node.threshold:
                node = node.left
            else:
                node = node.right
        probs = np.zeros(self.n_classes_)
        probs[node.prediction] = 1.0
        predictions.append(probs)
    return np.array(predictions)
```

Slide 7: Visualization Tools

Effective visualization is crucial for understanding rotational tree decisions. This implementation provides tools to visualize decision boundaries and tree structure with rotations.

```python
def plot_decision_boundary(self, X, y, title="Rotational Tree Decision Boundary"):
    """Plot the decision boundary and data points"""
    import matplotlib.pyplot as plt
    
    # Create mesh grid
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                        np.arange(y_min, y_max, 0.02))
    
    # Predict for each point in mesh
    Z = self.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8)
    plt.title(title)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.colorbar()
    plt.show()

def visualize_tree(self):
    """Generate visualization of tree structure with rotations"""
    from graphviz import Digraph
    dot = Digraph()
    
    def add_nodes_edges(node, parent_id=None):
        if node is None:
            return
        
        node_id = str(id(node))
        label = f"Rotation: {node.rotation_angle:.1f}°\n"
        if node.is_leaf():
            label += f"Prediction: {node.prediction}"
        else:
            label += f"Split: X[{node.feature_index}] <= {node.threshold:.2f}"
        
        dot.node(node_id, label)
        if parent_id:
            dot.edge(parent_id, node_id)
        
        if not node.is_leaf():
            add_nodes_edges(node.left, node_id)
            add_nodes_edges(node.right, node_id)
    
    add_nodes_edges(self.tree_)
    return dot
```

Slide 8: Practical Example - Spiral Dataset

Demonstrating the power of rotational trees on a challenging spiral dataset where traditional decision trees typically fail due to their axis-parallel split limitation.

```python
# Generate spiral dataset
def make_spiral_dataset(n_samples=1000, noise=0.2):
    n = n_samples // 2
    t = np.linspace(0, 4*np.pi, n)
    
    # Create spirals
    x1 = t * np.cos(t)
    y1 = t * np.sin(t)
    x2 = (t + np.pi) * np.cos(t)
    y2 = (t + np.pi) * np.sin(t)
    
    # Add noise
    x1 += np.random.normal(0, noise, n)
    y1 += np.random.normal(0, noise, n)
    x2 += np.random.normal(0, noise, n)
    y2 += np.random.normal(0, noise, n)
    
    # Combine datasets
    X = np.vstack([np.column_stack((x1, y1)), 
                  np.column_stack((x2, y2))])
    y = np.hstack([np.zeros(n), np.ones(n)])
    
    return X, y

# Create and train models
X, y = make_spiral_dataset()
rot_tree = RotationalTree(max_depth=5)
rot_tree.fit(X, y)

# Visualize results
rot_tree.plot_decision_boundary(X, y, "Rotational Tree on Spiral Dataset")
```

Slide 9: Rotational Random Forest Implementation

Extending the rotational tree concept to an ensemble method significantly improves performance. This implementation combines multiple rotational trees with bootstrap sampling and feature randomization for enhanced robustness.

```python
class RotationalRandomForest:
    def __init__(self, n_estimators=100, max_depth=5):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.trees = []
        
    def fit(self, X, y):
        self.n_classes_ = len(np.unique(y))
        
        for _ in range(self.n_estimators):
            # Bootstrap sampling
            indices = np.random.choice(len(X), len(X), replace=True)
            X_boot = X[indices]
            y_boot = y[indices]
            
            # Create and train rotational tree
            tree = RotationalTree(max_depth=self.max_depth)
            tree.fit(X_boot, y_boot)
            self.trees.append(tree)
            
        return self
    
    def predict_proba(self, X):
        # Collect predictions from all trees
        predictions = np.array([tree.predict_proba(X) 
                              for tree in self.trees])
        # Average predictions
        return np.mean(predictions, axis=0)
    
    def predict(self, X):
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)
```

Slide 10: Performance Evaluation Metrics

A comprehensive evaluation framework to assess rotational tree performance against traditional methods using various metrics and cross-validation techniques.

```python
def evaluate_models(X, y, test_size=0.2):
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    from sklearn.tree import DecisionTreeClassifier
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    
    # Initialize models
    models = {
        'Traditional DT': DecisionTreeClassifier(max_depth=5),
        'Rotational Tree': RotationalTree(max_depth=5),
        'Rotational Forest': RotationalRandomForest(
            n_estimators=100, max_depth=5
        )
    }
    
    results = {}
    for name, model in models.items():
        # Train and predict
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        results[name] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred, average='weighted'),
            'roc_auc': roc_auc_score(
                y_test, 
                model.predict_proba(X_test)[:, 1]
            ) if hasattr(model, 'predict_proba') else None
        }
    
    return results
```

Slide 11: Real-world Application - Credit Card Fraud Detection

Implementing rotational trees for credit card fraud detection demonstrates their effectiveness in handling complex, imbalanced real-world data with non-linear decision boundaries.

```python
def preprocess_credit_card_data(X, y):
    from sklearn.preprocessing import StandardScaler
    from imblearn.over_sampling import SMOTE
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Handle imbalanced data
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_scaled, y)
    
    return X_resampled, y_resampled

# Example usage with credit card dataset
def credit_card_fraud_detection():
    # Load and preprocess data
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=10000, n_features=30,
                             n_classes=2, weights=[0.98, 0.02],
                             random_state=42)
    
    X_processed, y_processed = preprocess_credit_card_data(X, y)
    
    # Train and evaluate models
    rot_forest = RotationalRandomForest(n_estimators=100)
    rot_forest.fit(X_processed, y_processed)
    
    # Cross-validation and performance metrics
    from sklearn.model_selection import cross_val_score
    scores = cross_val_score(rot_forest, X_processed, y_processed, 
                           cv=5, scoring='roc_auc')
    
    return {
        'mean_roc_auc': scores.mean(),
        'std_roc_auc': scores.std(),
        'model': rot_forest
    }
```

Slide 12: Hyperparameter Optimization for Rotational Trees

Implementing an efficient hyperparameter optimization framework for rotational trees using Bayesian optimization to find the optimal combination of rotation angles and tree parameters.

```python
from sklearn.model_selection import cross_val_score
from skopt import BayesSearchCV
from skopt.space import Real, Integer

class OptimizedRotationalTree(RotationalTree):
    def __init__(self, random_state=None):
        super().__init__()
        self.random_state = random_state
        
    def get_params(self, deep=True):
        return {
            'max_depth': self.max_depth,
            'min_samples_split': self.min_samples_split,
            'random_state': self.random_state
        }
        
    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self

def optimize_rotational_tree(X, y, n_iterations=50):
    param_space = {
        'max_depth': Integer(1, 20),
        'min_samples_split': Integer(2, 20),
        'random_state': Integer(0, 100)
    }
    
    optimizer = BayesSearchCV(
        OptimizedRotationalTree(),
        param_space,
        n_iter=n_iterations,
        cv=5,
        n_jobs=-1,
        scoring='accuracy'
    )
    
    optimizer.fit(X, y)
    
    return {
        'best_params': optimizer.best_params_,
        'best_score': optimizer.best_score_,
        'best_model': optimizer.best_estimator_
    }
```

Slide 13: Real-world Application - Image Rotation Detection

Demonstrating the practical application of rotational trees in computer vision for detecting image rotations, a task where traditional decision trees often struggle.

```python
import cv2
from sklearn.preprocessing import StandardScaler

def extract_rotation_features(image_path):
    # Load and preprocess image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (64, 64))
    
    # Extract features
    features = []
    
    # Gradient features
    gx = cv2.Sobel(img, cv2.CV_64F, 1, 0)
    gy = cv2.Sobel(img, cv2.CV_64F, 0, 1)
    
    # Compute gradient magnitude and direction
    mag = np.sqrt(gx**2 + gy**2)
    angle = np.arctan2(gy, gx)
    
    # Create feature vector
    features.extend([
        np.mean(mag),
        np.std(mag),
        np.mean(angle),
        np.std(angle),
        *np.histogram(angle, bins=8)[0]
    ])
    
    return np.array(features)

class ImageRotationDetector:
    def __init__(self, n_estimators=100):
        self.model = RotationalRandomForest(
            n_estimators=n_estimators,
            max_depth=8
        )
        self.scaler = StandardScaler()
        
    def fit(self, image_paths, angles):
        # Extract features from all images
        X = np.array([
            extract_rotation_features(img_path)
            for img_path in image_paths
        ])
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model.fit(X_scaled, angles)
        return self
    
    def predict(self, image_path):
        # Extract and scale features
        features = extract_rotation_features(image_path)
        features_scaled = self.scaler.transform(
            features.reshape(1, -1)
        )
        
        # Predict rotation angle
        return self.model.predict(features_scaled)[0]
```

Slide 14: Comparative Analysis and Benchmarking

Comprehensive benchmarking framework comparing rotational trees against state-of-the-art methods across various datasets and metrics.

```python
def benchmark_models(datasets, models, metrics, cv=5):
    results = {}
    
    for dataset_name, (X, y) in datasets.items():
        results[dataset_name] = {}
        
        for model_name, model in models.items():
            results[dataset_name][model_name] = {}
            
            for metric_name, metric in metrics.items():
                scores = cross_val_score(
                    model, X, y, 
                    scoring=metric, 
                    cv=cv
                )
                
                results[dataset_name][model_name][metric_name] = {
                    'mean': scores.mean(),
                    'std': scores.std()
                }
    
    return results

# Example usage
def run_benchmark_suite():
    from sklearn.datasets import make_moons, make_circles
    
    # Prepare datasets
    datasets = {
        'Moons': make_moons(n_samples=1000, noise=0.3),
        'Circles': make_circles(n_samples=1000, noise=0.2),
        'Spiral': make_spiral_dataset()
    }
    
    # Prepare models
    models = {
        'Rotational Tree': RotationalTree(max_depth=5),
        'Rotational Forest': RotationalRandomForest(
            n_estimators=100
        ),
        'Traditional DT': DecisionTreeClassifier(max_depth=5)
    }
    
    # Define metrics
    metrics = {
        'accuracy': 'accuracy',
        'f1': 'f1',
        'roc_auc': 'roc_auc'
    }
    
    return benchmark_models(datasets, models, metrics)
```

Slide 15: Additional Resources

*   "Rotation Forest: A New Classifier Ensemble Method" - [https://arxiv.org/abs/1809.06705](https://arxiv.org/abs/1809.06705)
*   "Oblique Decision Trees Using Embedded Neural Networks" - [https://arxiv.org/abs/2104.05100](https://arxiv.org/abs/2104.05100)
*   "Learning Deep Decision Trees with Batch Normalization" - [https://arxiv.org/abs/1904.02409](https://arxiv.org/abs/1904.02409)
*   "Random Rotation Ensembles" - [https://arxiv.org/abs/1810.12648](https://arxiv.org/abs/1810.12648)
*   "On the Analysis of Rotation-based Ensemble Learning" - [https://arxiv.org/abs/2007.09546](https://arxiv.org/abs/2007.09546)

