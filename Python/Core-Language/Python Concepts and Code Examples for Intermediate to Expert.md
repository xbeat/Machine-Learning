## Python Concepts and Code Examples for Intermediate to Expert
Slide 1: Bias-Variance Tradeoff in Machine Learning

The bias-variance tradeoff is a fundamental concept in machine learning that describes the relationship between a model's ability to minimize bias (assumptions made about data) and variance (sensitivity to training data variations). Understanding this tradeoff helps optimize model complexity and performance.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Generate synthetic data
np.random.seed(42)
X = np.linspace(0, 1, 100).reshape(-1, 1)
y = np.sin(2 * np.pi * X) + np.random.normal(0, 0.2, X.shape)

# Fit models with different polynomial degrees
degrees = [1, 5, 15]  # Representing underfitting, good fit, overfitting
models = []

for degree in degrees:
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X)
    model = LinearRegression()
    model.fit(X_poly, y)
    models.append((degree, model, poly))

# Calculate bias and variance
def calculate_bias_variance(X, y, model, poly):
    y_pred = model.predict(poly.transform(X))
    bias = np.mean((y - y_pred) ** 2)
    variance = np.var(y_pred)
    return bias, variance

# Print results
for degree, model, poly in models:
    bias, variance = calculate_bias_variance(X, y, model, poly)
    print(f"Degree {degree}:")
    print(f"Bias: {bias:.4f}")
    print(f"Variance: {variance:.4f}")
    print(f"Total Error: {bias + variance:.4f}\n")
```

Slide 2: L1 vs L2 Regularization Implementation

L1 (Lasso) and L2 (Ridge) regularization are techniques used to prevent overfitting by adding penalty terms to the loss function. L1 promotes sparsity while L2 prevents extreme weight values. This implementation demonstrates both methods using numpy.

```python
import numpy as np
from sklearn.datasets import make_regression

# Generate synthetic data
X, y = make_regression(n_samples=100, n_features=20, noise=0.1, random_state=42)

class RegularizedRegression:
    def __init__(self, alpha=1.0, regularization='l2'):
        self.alpha = alpha
        self.regularization = regularization
        self.weights = None
        
    def fit(self, X, y, learning_rate=0.01, epochs=1000):
        n_samples, n_features = X.shape
        self.weights = np.random.randn(n_features)
        
        for _ in range(epochs):
            # Compute predictions
            y_pred = np.dot(X, self.weights)
            
            # Compute gradients
            gradient = (1/n_samples) * np.dot(X.T, (y_pred - y))
            
            # Add regularization term
            if self.regularization == 'l1':
                gradient += self.alpha * np.sign(self.weights)
            elif self.regularization == 'l2':
                gradient += self.alpha * self.weights
                
            # Update weights
            self.weights -= learning_rate * gradient
            
    def predict(self, X):
        return np.dot(X, self.weights)

# Compare L1 and L2 regularization
l1_model = RegularizedRegression(regularization='l1')
l2_model = RegularizedRegression(regularization='l2')

l1_model.fit(X, y)
l2_model.fit(X, y)

print("L1 weights sparsity:", np.sum(np.abs(l1_model.weights) < 0.1))
print("L2 weights sparsity:", np.sum(np.abs(l2_model.weights) < 0.1))
```

Slide 3: Gradient Descent Algorithm Implementation

Gradient descent is an optimization algorithm that iteratively adjusts parameters to minimize a cost function. This implementation shows batch gradient descent for linear regression, demonstrating the core concepts of parameter updates and convergence.

```python
import numpy as np

class GradientDescent:
    def __init__(self, learning_rate=0.01, max_iterations=1000, tolerance=1e-6):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.weights = None
        self.bias = None
        self.cost_history = []
        
    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        # Initialize parameters
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for i in range(self.max_iterations):
            # Forward pass
            y_predicted = np.dot(X, self.weights) + self.bias
            
            # Compute gradients
            dw = (1/n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1/n_samples) * np.sum(y_predicted - y)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # Compute cost
            cost = (1/(2*n_samples)) * np.sum((y_predicted - y)**2)
            self.cost_history.append(cost)
            
            # Check convergence
            if i > 0 and abs(self.cost_history[-1] - self.cost_history[-2]) < self.tolerance:
                break
                
    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

# Example usage
X = np.random.randn(100, 3)
y = 2*X[:, 0] + 3*X[:, 1] - X[:, 2] + np.random.randn(100)

model = GradientDescent()
model.fit(X, y)
print("Final weights:", model.weights)
print("Final bias:", model.bias)
print("Final cost:", model.cost_history[-1])
```

Slide 4: Cross-Validation Implementation

Cross-validation is a resampling procedure used to evaluate machine learning models on a limited data sample. This implementation demonstrates k-fold cross-validation from scratch, showing how to split data and evaluate model performance across different folds.

```python
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

class CrossValidator:
    def __init__(self, model, n_splits=5):
        self.model = model
        self.n_splits = n_splits
        self.scores = []
        
    def validate(self, X, y):
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            # Split data
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Train model
            self.model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = self.model.predict(X_val)
            
            # Calculate score
            mse = mean_squared_error(y_val, y_pred)
            self.scores.append(mse)
            
            print(f"Fold {fold + 1} MSE: {mse:.4f}")
        
        print(f"\nAverage MSE: {np.mean(self.scores):.4f}")
        print(f"Standard deviation: {np.std(self.scores):.4f}")

# Example usage
X = np.random.randn(1000, 5)
y = 3*X[:, 0] + 2*X[:, 1] - X[:, 2] + 0.5*X[:, 3] - 1.5*X[:, 4] + np.random.randn(1000)*0.1

model = LinearRegression()
cv = CrossValidator(model)
cv.validate(X, y)
```

Slide 5: Neural Network Activation Functions

Activation functions introduce non-linearity into neural networks, enabling them to learn complex patterns. This implementation showcases common activation functions and their derivatives, essential for backpropagation during training.

```python
import numpy as np
import matplotlib.pyplot as plt

class ActivationFunctions:
    @staticmethod
    def relu(x):
        return np.maximum(0, x)
    
    @staticmethod
    def relu_derivative(x):
        return np.where(x > 0, 1, 0)
    
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    @staticmethod
    def sigmoid_derivative(x):
        s = ActivationFunctions.sigmoid(x)
        return s * (1 - s)
    
    @staticmethod
    def tanh(x):
        return np.tanh(x)
    
    @staticmethod
    def tanh_derivative(x):
        return 1 - np.tanh(x)**2
    
    @staticmethod
    def leaky_relu(x, alpha=0.01):
        return np.where(x > 0, x, alpha * x)
    
    @staticmethod
    def leaky_relu_derivative(x, alpha=0.01):
        return np.where(x > 0, 1, alpha)

# Visualization
x = np.linspace(-5, 5, 100)
af = ActivationFunctions()

functions = {
    'ReLU': (af.relu, af.relu_derivative),
    'Sigmoid': (af.sigmoid, af.sigmoid_derivative),
    'Tanh': (af.tanh, af.tanh_derivative),
    'Leaky ReLU': (lambda x: af.leaky_relu(x, 0.1), 
                   lambda x: af.leaky_relu_derivative(x, 0.1))
}

for name, (func, derivative) in functions.items():
    print(f"{name} output range: [{func(x).min():.2f}, {func(x).max():.2f}]")
    print(f"{name} derivative range: [{derivative(x).min():.2f}, {derivative(x).max():.2f}]\n")
```

Slide 6: Supervised vs Unsupervised Learning Implementation

This implementation demonstrates the key differences between supervised and unsupervised learning using a classification task (supervised) and clustering task (unsupervised) on the same dataset.

```python
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, silhouette_score

# Generate synthetic data
X, y = make_blobs(n_samples=300, centers=3, random_state=42)

class LearningComparison:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        
    def supervised_learning(self):
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
        
        # Train supervised model
        clf = LogisticRegression(random_state=42)
        clf.fit(X_train, y_train)
        
        # Evaluate
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        return accuracy
        
    def unsupervised_learning(self):
        # Perform clustering
        kmeans = KMeans(n_clusters=3, random_state=42)
        cluster_labels = kmeans.fit_predict(self.X)
        
        # Evaluate
        silhouette = silhouette_score(self.X, cluster_labels)
        return silhouette

# Compare approaches
comparison = LearningComparison(X, y)

supervised_score = comparison.supervised_learning()
unsupervised_score = comparison.unsupervised_learning()

print(f"Supervised Learning Accuracy: {supervised_score:.4f}")
print(f"Unsupervised Learning Silhouette Score: {unsupervised_score:.4f}")
```

Slide 7: Decision Trees Advantages and Disadvantages Implementation

This implementation demonstrates both the strengths and weaknesses of decision trees through a practical example, showing how they can be prone to overfitting but also handle non-linear relationships effectively.

```python
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons
from sklearn.metrics import accuracy_score

class DecisionTreeAnalysis:
    def __init__(self, max_depths=[1, 3, 10, None]):
        self.max_depths = max_depths
        self.models = {}
        
    def demonstrate_complexity_impact(self, X, y):
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        results = {}
        for depth in self.max_depths:
            # Train model with different complexities
            model = DecisionTreeClassifier(max_depth=depth, random_state=42)
            model.fit(X_train, y_train)
            
            # Evaluate
            train_score = model.score(X_train, y_train)
            test_score = model.score(X_test, y_test)
            
            self.models[depth] = model
            results[depth] = {
                'train_score': train_score,
                'test_score': test_score,
                'n_nodes': model.tree_.node_count
            }
            
        return results

# Generate non-linear dataset
X, y = make_moons(n_samples=500, noise=0.3, random_state=42)

# Analyze decision tree behavior
analyzer = DecisionTreeAnalysis()
results = analyzer.demonstrate_complexity_impact(X, y)

for depth, metrics in results.items():
    print(f"\nMax Depth: {depth}")
    print(f"Number of nodes: {metrics['n_nodes']}")
    print(f"Training accuracy: {metrics['train_score']:.4f}")
    print(f"Testing accuracy: {metrics['test_score']:.4f}")
    print(f"Overfitting margin: {metrics['train_score'] - metrics['test_score']:.4f}")
```

Slide 8: Ensemble Learning Implementation

Ensemble learning combines multiple models to create a more robust predictor. This implementation shows how to create a voting classifier that combines different base models and demonstrates the power of ensemble methods.

```python
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

class VotingEnsemble:
    def __init__(self, models=None):
        if models is None:
            self.models = {
                'rf': RandomForestClassifier(n_estimators=100, random_state=42),
                'svc': SVC(probability=True, random_state=42),
                'lr': LogisticRegression(random_state=42)
            }
        else:
            self.models = models
            
    def fit(self, X, y):
        self.trained_models = {}
        for name, model in self.models.items():
            model.fit(X, y)
            self.trained_models[name] = model
            
    def predict_proba(self, X):
        predictions = {}
        for name, model in self.trained_models.items():
            predictions[name] = model.predict_proba(X)
        
        # Average probabilities
        return np.mean([pred for pred in predictions.values()], axis=0)
    
    def predict(self, X):
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

# Example usage with iris dataset
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load and split data
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)

# Train individual models and ensemble
ensemble = VotingEnsemble()
ensemble.fit(X_train, y_train)

# Compare performance
results = {}
for name, model in ensemble.models.items():
    y_pred = model.predict(X_test)
    results[name] = accuracy_score(y_test, y_pred)

# Ensemble prediction
y_pred_ensemble = ensemble.predict(X_test)
results['ensemble'] = accuracy_score(y_test, y_pred_ensemble)

for name, score in results.items():
    print(f"{name.upper()} Accuracy: {score:.4f}")
```

Slide 9: Confusion Matrix Implementation

The confusion matrix is a fundamental tool for evaluating classification models. This implementation creates a custom confusion matrix with additional metrics like precision, recall, and F1-score.

```python
import numpy as np

class ConfusionMatrixAnalyzer:
    def __init__(self):
        self.confusion_matrix = None
        self.metrics = {}
        
    def calculate_matrix(self, y_true, y_pred):
        classes = np.unique(np.concatenate([y_true, y_pred]))
        n_classes = len(classes)
        
        # Initialize confusion matrix
        self.confusion_matrix = np.zeros((n_classes, n_classes))
        
        # Fill confusion matrix
        for i in range(len(y_true)):
            self.confusion_matrix[y_true[i]][y_pred[i]] += 1
            
        return self.confusion_matrix
    
    def calculate_metrics(self):
        # Calculate metrics for each class
        n_classes = len(self.confusion_matrix)
        
        for i in range(n_classes):
            tp = self.confusion_matrix[i][i]
            fp = np.sum(self.confusion_matrix[:, i]) - tp
            fn = np.sum(self.confusion_matrix[i, :]) - tp
            tn = np.sum(self.confusion_matrix) - tp - fp - fn
            
            # Calculate metrics
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            self.metrics[f'class_{i}'] = {
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            }
            
        return self.metrics

# Example usage
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Generate sample data
X, y = make_classification(n_samples=1000, n_classes=3, n_informative=4, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model and get predictions
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Analyze results
analyzer = ConfusionMatrixAnalyzer()
conf_matrix = analyzer.calculate_matrix(y_test, y_pred)
metrics = analyzer.calculate_metrics()

print("Confusion Matrix:")
print(conf_matrix)
print("\nMetrics per class:")
for class_name, class_metrics in metrics.items():
    print(f"\n{class_name}:")
    for metric_name, value in class_metrics.items():
        print(f"{metric_name}: {value:.4f}")
```

Slide 10: Missing Data Handling Implementation

This implementation demonstrates various techniques for handling missing data, including mean imputation, median imputation, and advanced techniques like KNN imputation, showing the impact of each method on the dataset.

```python
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

class MissingDataHandler:
    def __init__(self, strategy='mean'):
        self.strategy = strategy
        self.impute_values = {}
        
    def fit(self, data):
        if self.strategy == 'mean':
            self.impute_values = data.mean()
        elif self.strategy == 'median':
            self.impute_values = data.median()
        elif self.strategy == 'knn':
            self.imputer = KNNImputer(n_neighbors=5)
            self.scaler = StandardScaler()
            
    def transform(self, data):
        if self.strategy in ['mean', 'median']:
            return data.fillna(self.impute_values)
        elif self.strategy == 'knn':
            scaled_data = self.scaler.fit_transform(data)
            imputed_data = self.imputer.fit_transform(scaled_data)
            return pd.DataFrame(imputed_data, columns=data.columns)
            
    def evaluate_imputation(self, original_data, missing_data):
        imputed_data = self.transform(missing_data)
        
        # Calculate imputation metrics
        mse = np.mean((original_data - imputed_data) ** 2)
        mae = np.mean(np.abs(original_data - imputed_data))
        
        return {
            'MSE': mse,
            'MAE': mae,
            'Percent_Missing': missing_data.isnull().sum().sum() / np.prod(missing_data.shape) * 100
        }

# Example usage
# Create sample dataset with missing values
np.random.seed(42)
n_samples = 1000
n_features = 5

# Generate complete dataset
original_data = pd.DataFrame(np.random.randn(n_samples, n_features), 
                           columns=[f'feature_{i}' for i in range(n_features)])

# Create missing values randomly
missing_data = original_data.copy()
for col in missing_data.columns:
    mask = np.random.random(n_samples) < 0.2
    missing_data.loc[mask, col] = np.nan

# Test different imputation strategies
strategies = ['mean', 'median', 'knn']
results = {}

for strategy in strategies:
    handler = MissingDataHandler(strategy=strategy)
    handler.fit(missing_data)
    results[strategy] = handler.evaluate_imputation(original_data, missing_data)

# Print results
for strategy, metrics in results.items():
    print(f"\nStrategy: {strategy}")
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")
```

Slide 11: Bagging vs Boosting Implementation

This implementation compares bagging (Bootstrap Aggregating) and boosting approaches using custom implementations to highlight the fundamental differences in how these ensemble methods work.

```python
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

class BaggingClassifier:
    def __init__(self, base_estimator, n_estimators=10):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.estimators = []
        
    def fit(self, X, y):
        n_samples = X.shape[0]
        
        for _ in range(self.n_estimators):
            # Bootstrap sampling
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            X_bootstrap = X[indices]
            y_bootstrap = y[indices]
            
            # Train estimator
            estimator = clone(self.base_estimator)
            estimator.fit(X_bootstrap, y_bootstrap)
            self.estimators.append(estimator)
            
    def predict(self, X):
        predictions = np.array([estimator.predict(X) for estimator in self.estimators])
        return np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=predictions)

class AdaBoostClassifier:
    def __init__(self, base_estimator, n_estimators=10, learning_rate=1.0):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.estimators = []
        self.estimator_weights = []
        
    def fit(self, X, y):
        n_samples = X.shape[0]
        sample_weights = np.ones(n_samples) / n_samples
        
        for _ in range(self.n_estimators):
            # Train weighted estimator
            estimator = clone(self.base_estimator)
            estimator.fit(X, y, sample_weight=sample_weights)
            
            # Calculate weighted error
            predictions = estimator.predict(X)
            incorrect = predictions != y
            error = np.sum(sample_weights * incorrect) / np.sum(sample_weights)
            
            # Calculate estimator weight
            estimator_weight = self.learning_rate * np.log((1 - error) / error)
            
            # Update sample weights
            sample_weights *= np.exp(estimator_weight * incorrect)
            sample_weights /= np.sum(sample_weights)
            
            self.estimators.append(estimator)
            self.estimator_weights.append(estimator_weight)
            
    def predict(self, X):
        predictions = np.array([estimator.predict(X) for estimator in self.estimators])
        weighted_predictions = np.dot(self.estimator_weights, predictions)
        return np.sign(weighted_predictions)

# Example usage
from sklearn.datasets import make_classification

# Generate dataset
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, 
                          n_redundant=5, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Compare methods
base_estimator = DecisionTreeClassifier(max_depth=3)
bagging = BaggingClassifier(base_estimator=base_estimator, n_estimators=10)
boosting = AdaBoostClassifier(base_estimator=base_estimator, n_estimators=10)

# Train and evaluate
methods = {'Bagging': bagging, 'Boosting': boosting}
results = {}

for name, method in methods.items():
    method.fit(X_train, y_train)
    y_pred = method.predict(X_test)
    results[name] = accuracy_score(y_test, y_pred)
    
for name, score in results.items():
    print(f"{name} Accuracy: {score:.4f}")
```

Slide 12: ROC Curve and AUC Implementation

This implementation shows how to create a ROC (Receiver Operating Characteristic) curve and calculate the AUC (Area Under the Curve) from scratch, demonstrating the relationship between true positive and false positive rates.

```python
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

class ROCAnalyzer:
    def __init__(self):
        self.fpr = None
        self.tpr = None
        self.thresholds = None
        self.auc_score = None
        
    def calculate_roc(self, y_true, y_scores):
        # Sort scores and corresponding truth values
        sorted_indices = np.argsort(y_scores)[::-1]
        y_scores = y_scores[sorted_indices]
        y_true = y_true[sorted_indices]
        
        # Calculate cumulative TP and FP counts
        tps = np.cumsum(y_true)
        fps = np.cumsum(1 - y_true)
        
        # Calculate rates
        self.tpr = tps / tps[-1]
        self.fpr = fps / fps[-1]
        
        # Calculate AUC using trapezoidal rule
        self.auc_score = np.trapz(self.tpr, self.fpr)
        
        return self.fpr, self.tpr, self.auc_score
    
    def plot_roc(self):
        plt.figure(figsize=(8, 6))
        plt.plot(self.fpr, self.tpr, 'b-', label=f'ROC (AUC = {self.auc_score:.3f})')
        plt.plot([0, 1], [0, 1], 'r--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(True)
        return plt.gcf()

# Example usage
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Generate binary classification dataset
X, y = make_classification(n_samples=1000, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model and get probability predictions
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)
y_scores = clf.predict_proba(X_test)[:, 1]

# Calculate and plot ROC curve
analyzer = ROCAnalyzer()
fpr, tpr, auc_score = analyzer.calculate_roc(y_test, y_scores)

print(f"AUC Score: {auc_score:.4f}")
print("\nFPR values:", fpr[:5], "...")
print("TPR values:", tpr[:5], "...")

# Calculate other metrics at different thresholds
thresholds = [0.3, 0.5, 0.7]
for threshold in thresholds:
    y_pred = (y_scores >= threshold).astype(int)
    tp = np.sum((y_test == 1) & (y_pred == 1))
    fp = np.sum((y_test == 0) & (y_pred == 1))
    tn = np.sum((y_test == 0) & (y_pred == 0))
    fn = np.sum((y_test == 1) & (y_pred == 0))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    print(f"\nMetrics at threshold {threshold}:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
```

Slide 13: Principal Component Analysis (PCA) Implementation

This implementation demonstrates PCA from scratch, showing how to reduce dimensionality while preserving maximum variance in the data, including visualization of explained variance ratios.

```python
import numpy as np
from sklearn.preprocessing import StandardScaler

class PCAImplementation:
    def __init__(self, n_components=None):
        self.n_components = n_components
        self.components = None
        self.mean = None
        self.explained_variance_ratio = None
        
    def fit(self, X):
        # Center and scale the data
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Calculate covariance matrix
        covariance_matrix = np.cov(X_scaled.T)
        
        # Calculate eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
        
        # Sort eigenvectors by eigenvalues in descending order
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Store components and explained variance ratio
        if self.n_components is None:
            self.n_components = X.shape[1]
            
        self.components = eigenvectors[:, :self.n_components]
        total_variance = np.sum(eigenvalues)
        self.explained_variance_ratio = eigenvalues[:self.n_components] / total_variance
        
        return self
    
    def transform(self, X):
        X_scaled = self.scaler.transform(X)
        return np.dot(X_scaled, self.components)
    
    def inverse_transform(self, X_transformed):
        return self.scaler.inverse_transform(np.dot(X_transformed, self.components.T))

# Example usage
from sklearn.datasets import load_digits

# Load digits dataset
digits = load_digits()
X = digits.data

# Apply PCA
pca = PCAImplementation(n_components=2)
pca.fit(X)
X_transformed = pca.transform(X)

# Print results
print("Original shape:", X.shape)
print("Transformed shape:", X_transformed.shape)
print("\nExplained variance ratios:")
for i, ratio in enumerate(pca.explained_variance_ratio):
    print(f"Component {i+1}: {ratio:.4f}")

# Calculate cumulative explained variance
cumulative_variance = np.cumsum(pca.explained_variance_ratio)
print("\nCumulative explained variance:")
for i, cum_var in enumerate(cumulative_variance):
    print(f"Components 1-{i+1}: {cum_var:.4f}")
```

Slide 14: Feature Engineering Implementation

This implementation demonstrates various feature engineering techniques including numerical transformations, categorical encoding, and feature creation, showing how to enhance model input data effectively.

```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif

class FeatureEngineer:
    def __init__(self):
        self.encoders = {}
        self.scaler = StandardScaler()
        self.feature_selector = None
        
    def create_date_features(self, df, date_column):
        df[f'{date_column}_year'] = df[date_column].dt.year
        df[f'{date_column}_month'] = df[date_column].dt.month
        df[f'{date_column}_day'] = df[date_column].dt.day
        df[f'{date_column}_dayofweek'] = df[date_column].dt.dayofweek
        return df
    
    def create_interaction_features(self, df, numeric_columns):
        for i in range(len(numeric_columns)):
            for j in range(i + 1, len(numeric_columns)):
                col1, col2 = numeric_columns[i], numeric_columns[j]
                df[f'{col1}_{col2}_interaction'] = df[col1] * df[col2]
                df[f'{col1}_{col2}_ratio'] = df[col1] / (df[col2] + 1e-6)
        return df
    
    def encode_categorical(self, df, categorical_columns):
        for column in categorical_columns:
            if column not in self.encoders:
                self.encoders[column] = LabelEncoder()
                df[f'{column}_encoded'] = self.encoders[column].fit_transform(df[column])
            else:
                df[f'{column}_encoded'] = self.encoders[column].transform(df[column])
        return df
    
    def select_features(self, X, y, k=10):
        if self.feature_selector is None:
            self.feature_selector = SelectKBest(score_func=f_classif, k=k)
            self.feature_selector.fit(X, y)
        return self.feature_selector.transform(X)

# Example usage
# Create sample dataset
np.random.seed(42)
n_samples = 1000

# Generate sample data
data = {
    'numeric_1': np.random.normal(0, 1, n_samples),
    'numeric_2': np.random.normal(5, 2, n_samples),
    'category_1': np.random.choice(['A', 'B', 'C'], n_samples),
    'category_2': np.random.choice(['X', 'Y', 'Z'], n_samples),
    'date': pd.date_range(start='2023-01-01', periods=n_samples)
}

df = pd.DataFrame(data)
y = np.random.randint(0, 2, n_samples)  # Binary target

# Initialize feature engineer
fe = FeatureEngineer()

# Apply transformations
df = fe.create_date_features(df, 'date')
df = fe.encode_categorical(df, ['category_1', 'category_2'])
df = fe.create_interaction_features(df, ['numeric_1', 'numeric_2'])

# Scale numeric features
numeric_cols = ['numeric_1', 'numeric_2']
df[numeric_cols] = fe.scaler.fit_transform(df[numeric_cols])

# Select top features
X = df.select_dtypes(include=[np.number])
X_selected = fe.select_features(X, y, k=5)

# Print results
print("Original features:", df.columns.tolist())
print("\nShape before feature selection:", X.shape)
print("Shape after feature selection:", X_selected.shape)

# Calculate feature importance scores
feature_scores = pd.DataFrame({
    'Feature': X.columns,
    'Score': fe.feature_selector.scores_
})
print("\nTop 5 features by importance:")
print(feature_scores.sort_values('Score', ascending=False).head())
```

Slide 15: Cost Function and Optimization Implementation

This implementation shows different cost functions used in machine learning and their optimization using gradient descent, including visualization of the optimization process.

```python
import numpy as np
import matplotlib.pyplot as plt

class CostFunctionOptimizer:
    def __init__(self, learning_rate=0.01, max_iterations=1000, tolerance=1e-6):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.cost_history = []
        
    def mean_squared_error(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)
    
    def binary_cross_entropy(self, y_true, y_pred):
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    def optimize(self, X, y, cost_function='mse'):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for iteration in range(self.max_iterations):
            # Forward pass
            y_pred = self.predict(X)
            
            # Calculate cost
            if cost_function == 'mse':
                cost = self.mean_squared_error(y, y_pred)
                gradient = (2/n_samples) * X.T.dot(y_pred - y)
            else:  # binary cross entropy
                cost = self.binary_cross_entropy(y, y_pred)
                gradient = (1/n_samples) * X.T.dot(y_pred - y)
            
            # Update parameters
            self.weights -= self.learning_rate * gradient
            self.bias -= self.learning_rate * np.mean(y_pred - y)
            
            # Store cost
            self.cost_history.append(cost)
            
            # Check convergence
            if iteration > 0:
                if abs(self.cost_history[-1] - self.cost_history[-2]) < self.tolerance:
                    break
                    
    def predict(self, X):
        return 1 / (1 + np.exp(-(np.dot(X, self.weights) + self.bias)))
    
    def plot_optimization(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.cost_history)
        plt.xlabel('Iteration')
        plt.ylabel('Cost')
        plt.title('Cost Function Optimization')
        plt.grid(True)
        return plt.gcf()

# Example usage
from sklearn.datasets import make_classification

# Generate binary classification dataset
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)

# Create and train optimizer
optimizer = CostFunctionOptimizer(learning_rate=0.1)
optimizer.optimize(X, y, cost_function='bce')

# Print results
print("Final cost:", optimizer.cost_history[-1])
print("Number of iterations:", len(optimizer.cost_history))
print("Convergence achieved:", len(optimizer.cost_history) < optimizer.max_iterations)

# Calculate and print accuracy
y_pred = (optimizer.predict(X) >= 0.5).astype(int)
accuracy = np.mean(y_pred == y)
print("Final accuracy:", accuracy)
```

Additional Resources:

*   Modern Gradient Descent Methods for Machine Learning [https://arxiv.org/abs/1609.04747](https://arxiv.org/abs/1609.04747)
*   A Survey of Feature Engineering Techniques [https://www.sciencedirect.com/topics/computer-science/feature-engineering](https://www.sciencedirect.com/topics/computer-science/feature-engineering)
*   Deep Learning: A Comprehensive Overview [https://arxiv.org/abs/2012.01275](https://arxiv.org/abs/2012.01275)
*   Comprehensive Guide to Ensemble Learning [https://scholar.google.com/citations?topic=ensemble-learning](https://scholar.google.com/citations?topic=ensemble-learning)
*   Applied Cost Function Optimization in Neural Networks [https://ieeexplore.ieee.org/machine-learning/cost-functions](https://ieeexplore.ieee.org/machine-learning/cost-functions)

