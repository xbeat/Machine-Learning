## Machine Learning Algorithms Handwritten Notes
Slide 1: Introduction to Machine Learning

Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without explicit programming. It uses statistical techniques to allow computers to find hidden patterns in data and make intelligent decisions based on these patterns.

```python
# Basic ML workflow example
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Generate synthetic data
X = np.random.randn(100, 1)
y = 2 * X + 1 + np.random.randn(100, 1) * 0.1

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create and train model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)
print(f"Model Score: {model.score(X_test, y_test):.4f}")
```

Slide 2: Types of Machine Learning

Understanding the fundamental types of machine learning is crucial for selecting the appropriate approach for different problems. The main categories include supervised learning, unsupervised learning, reinforcement learning, and semi-supervised learning, each serving distinct purposes.

```python
# Example demonstrating different ML types
from sklearn.cluster import KMeans  # Unsupervised
from sklearn.svm import SVC        # Supervised
import numpy as np

# Supervised Learning Example
X_supervised = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
y_supervised = np.array([0, 0, 1, 1])
supervised_model = SVC()
supervised_model.fit(X_supervised, y_supervised)

# Unsupervised Learning Example
X_unsupervised = np.array([[1, 2], [2, 2], [4, 5], [5, 4]])
unsupervised_model = KMeans(n_clusters=2)
clusters = unsupervised_model.fit_predict(X_unsupervised)

print("Supervised Prediction:", supervised_model.predict([[2.5, 3.5]]))
print("Unsupervised Clusters:", clusters)
```

Slide 3: Data Collection and Preprocessing

Data collection and preprocessing form the foundation of any machine learning project. This crucial step involves gathering relevant data, handling missing values, normalizing features, and preparing the dataset for model training.

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Load and preprocess data
def preprocess_data(data_path):
    # Read data
    df = pd.read_csv(data_path)
    
    # Handle missing values
    imputer = SimpleImputer(strategy='mean')
    df_numeric = df.select_dtypes(include=[np.number])
    df[df_numeric.columns] = imputer.fit_transform(df_numeric)
    
    # Normalize features
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(
        scaler.fit_transform(df_numeric),
        columns=df_numeric.columns
    )
    
    return df_scaled

# Example usage
# df = preprocess_data('dataset.csv')
# Print example of preprocessing steps
print("Example preprocessing steps:")
example_data = pd.DataFrame({
    'A': [1, np.nan, 3, 4],
    'B': [5, 6, np.nan, 8]
})
print("Original:\n", example_data)
print("\nProcessed:\n", preprocess_data(example_data))
```

Slide 4: Feature Engineering Techniques

Feature engineering transforms raw data into features that better represent the underlying problem, improving model performance. This process requires domain knowledge and creative approach to extract meaningful information from existing features.

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import LabelEncoder

def engineer_features(df):
    # Create interaction features
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    poly = PolynomialFeatures(degree=2, include_bias=False)
    poly_features = pd.DataFrame(
        poly.fit_transform(df[numeric_cols]),
        columns=poly.get_feature_names(numeric_cols)
    )
    
    # Create time-based features (example with datetime)
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df['day_of_week'] = df['date'].dt.dayofweek
        df['month'] = df['date'].dt.month
        df['year'] = df['date'].dt.year
    
    # Encode categorical variables
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        le = LabelEncoder()
        df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
    
    return df

# Example usage
example_df = pd.DataFrame({
    'numeric_feat': [1, 2, 3, 4],
    'category': ['A', 'B', 'A', 'C'],
    'date': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04']
})
engineered_df = engineer_features(example_df)
print("Engineered Features:\n", engineered_df.head())
```

Slide 5: Model Training and Cross-Validation

Model training is an iterative process requiring proper validation techniques to ensure generalization. Cross-validation helps assess model performance and prevent overfitting by evaluating the model on different data subsets.

```python
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import KFold

# Generate synthetic dataset
X, y = make_classification(n_samples=1000, n_features=20, 
                         n_informative=15, n_redundant=5,
                         random_state=42)

# Initialize model and cross-validation
model = RandomForestClassifier(n_estimators=100, random_state=42)
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

# Perform cross-validation
scores = cross_val_score(model, X, y, cv=kfold, scoring='accuracy')

print(f"Cross-validation scores: {scores}")
print(f"Mean CV score: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")

# Train final model
model.fit(X, y)
print(f"Final model score: {model.score(X, y):.4f}")
```

Slide 6: Overfitting and Underfitting Detection

Understanding model complexity and its relationship with bias-variance tradeoff is crucial for detecting and preventing overfitting and underfitting. These concepts are fundamental to achieving optimal model performance through proper regularization and model selection.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import learning_curve

def plot_learning_complexity(X, y, degrees=[1, 3, 10]):
    plt.figure(figsize=(15, 5))
    
    for i, degree in enumerate(degrees, 1):
        poly = PolynomialFeatures(degree=degree)
        X_poly = poly.fit_transform(X.reshape(-1, 1))
        
        model = LinearRegression()
        
        # Calculate learning curves
        train_sizes, train_scores, val_scores = learning_curve(
            model, X_poly, y, train_sizes=np.linspace(0.1, 1.0, 10),
            cv=5, scoring='neg_mean_squared_error'
        )
        
        train_scores_mean = -train_scores.mean(axis=1)
        val_scores_mean = -val_scores.mean(axis=1)
        
        plt.subplot(1, 3, i)
        plt.plot(train_sizes, train_scores_mean, label='Training error')
        plt.plot(train_sizes, val_scores_mean, label='Validation error')
        plt.title(f'Polynomial Degree {degree}')
        plt.xlabel('Training Examples')
        plt.ylabel('Mean Squared Error')
        plt.legend()
    
    plt.tight_layout()
    return plt

# Generate synthetic data
X = np.linspace(0, 1, 100)
y = np.sin(2 * np.pi * X) + np.random.normal(0, 0.2, X.shape)

# Plot learning curves for different complexities
plot = plot_learning_complexity(X, y)
plot.show()
```

Slide 7: Regularization Techniques Implementation

Regularization helps prevent overfitting by adding penalty terms to the model's loss function. The three main types - L1 (Lasso), L2 (Ridge), and Elastic Net combine different approaches to achieve optimal feature selection and coefficient shrinkage.

```python
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import numpy as np

def compare_regularization(X, y, alphas=[0.1, 1.0, 10.0]):
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Initialize results dictionary
    results = {
        'ridge': [],
        'lasso': [],
        'elastic': []
    }
    
    # Train and evaluate models with different alpha values
    for alpha in alphas:
        # Ridge Regression
        ridge = Ridge(alpha=alpha)
        ridge.fit(X_scaled, y)
        ridge_pred = ridge.predict(X_scaled)
        results['ridge'].append({
            'alpha': alpha,
            'mse': mean_squared_error(y, ridge_pred),
            'coef': ridge.coef_
        })
        
        # Lasso Regression
        lasso = Lasso(alpha=alpha)
        lasso.fit(X_scaled, y)
        lasso_pred = lasso.predict(X_scaled)
        results['lasso'].append({
            'alpha': alpha,
            'mse': mean_squared_error(y, lasso_pred),
            'coef': lasso.coef_
        })
        
        # Elastic Net
        elastic = ElasticNet(alpha=alpha, l1_ratio=0.5)
        elastic.fit(X_scaled, y)
        elastic_pred = elastic.predict(X_scaled)
        results['elastic'].append({
            'alpha': alpha,
            'mse': mean_squared_error(y, elastic_pred),
            'coef': elastic.coef_
        })
    
    return results

# Generate synthetic data
np.random.seed(42)
X = np.random.randn(100, 20)
y = X[:, 0] * 1 + X[:, 1] * 2 + np.random.randn(100) * 0.1

# Compare different regularization techniques
results = compare_regularization(X, y)

# Print results
for method, result_list in results.items():
    print(f"\n{method.upper()} Results:")
    for r in result_list:
        print(f"Alpha: {r['alpha']}, MSE: {r['mse']:.4f}")
        print(f"Non-zero coefficients: {np.sum(r['coef'] != 0)}")
```

Slide 8: Decision Trees from Scratch

Implementation of a decision tree classifier from scratch helps understand the fundamental concepts of tree-based learning, including splitting criteria, recursive partitioning, and pruning strategies.

```python
import numpy as np
from collections import Counter

class DecisionNode:
    def __init__(self, feature_idx=None, threshold=None, left=None, 
                 right=None, value=None):
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

class DecisionTreeFromScratch:
    def __init__(self, max_depth=5):
        self.max_depth = max_depth
        self.root = None
        
    def gini_impurity(self, y):
        counter = Counter(y)
        impurity = 1
        for count in counter.values():
            prob = count / len(y)
            impurity -= prob ** 2
        return impurity
    
    def find_best_split(self, X, y):
        best_gain = -1
        best_feature = None
        best_threshold = None
        
        current_impurity = self.gini_impurity(y)
        
        for feature in range(X.shape[1]):
            thresholds = np.unique(X[:, feature])
            
            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask
                
                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue
                
                left_impurity = self.gini_impurity(y[left_mask])
                right_impurity = self.gini_impurity(y[right_mask])
                
                n_left = np.sum(left_mask)
                n_right = np.sum(right_mask)
                n_total = len(y)
                
                gain = current_impurity - (
                    (n_left/n_total) * left_impurity + 
                    (n_right/n_total) * right_impurity
                )
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold
                    
        return best_feature, best_threshold, best_gain
    
    def _build_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))
        
        # Stopping criteria
        if (self.max_depth is not None and depth >= self.max_depth) or \
           n_classes == 1 or n_samples < 2:
            leaf_value = Counter(y).most_common(1)[0][0]
            return DecisionNode(value=leaf_value)
        
        # Find best split
        best_feature, best_threshold, best_gain = self.find_best_split(X, y)
        
        if best_gain == -1:
            leaf_value = Counter(y).most_common(1)[0][0]
            return DecisionNode(value=leaf_value)
        
        # Split the data
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask
        
        # Recursively build left and right subtrees
        left_subtree = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_subtree = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        
        return DecisionNode(
            feature_idx=best_feature,
            threshold=best_threshold,
            left=left_subtree,
            right=right_subtree
        )
    
    def fit(self, X, y):
        self.root = self._build_tree(X, y)
        
    def _predict_single(self, x, node):
        if node.value is not None:
            return node.value
        
        if x[node.feature_idx] <= node.threshold:
            return self._predict_single(x, node.left)
        return self._predict_single(x, node.right)
    
    def predict(self, X):
        return np.array([self._predict_single(x, self.root) for x in X])

# Example usage
X = np.random.randn(100, 2)
y = (X[:, 0] + X[:, 1] > 0).astype(int)

tree = DecisionTreeFromScratch(max_depth=3)
tree.fit(X, y)
predictions = tree.predict(X)
accuracy = np.mean(predictions == y)
print(f"Accuracy: {accuracy:.4f}")
```

Slide 9: Random Forest Implementation

Random Forests combine multiple decision trees to create a powerful ensemble model that reduces overfitting and improves generalization through bagging and random feature selection at each split.

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np

class RandomForestImplementation:
    def __init__(self, n_trees=100, max_depth=None, min_samples_split=2,
                 max_features='sqrt'):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.trees = []
        
    def bootstrap_sample(self, X, y):
        n_samples = X.shape[0]
        idxs = np.random.choice(n_samples, size=n_samples, replace=True)
        return X[idxs], y[idxs]
    
    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_trees):
            tree = RandomForestClassifier(
                n_estimators=1,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=self.max_features
            )
            # Get bootstrap sample
            X_sample, y_sample = self.bootstrap_sample(X, y)
            # Train tree on bootstrap sample
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)
    
    def predict(self, X):
        # Collect predictions from all trees
        tree_predictions = np.array([tree.predict(X) for tree in self.trees])
        # Return majority vote
        return np.array([
            np.bincount(predictions).argmax() 
            for predictions in tree_predictions.T
        ])

# Generate synthetic dataset
from sklearn.datasets import make_classification
X, y = make_classification(
    n_samples=1000, 
    n_features=20,
    n_informative=15,
    n_redundant=5,
    random_state=42
)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train and evaluate custom Random Forest
rf_custom = RandomForestImplementation(n_trees=100, max_depth=5)
rf_custom.fit(X_train, y_train)
custom_predictions = rf_custom.predict(X_test)

# Compare with sklearn implementation
rf_sklearn = RandomForestClassifier(
    n_estimators=100, max_depth=5, random_state=42
)
rf_sklearn.fit(X_train, y_train)
sklearn_predictions = rf_sklearn.predict(X_test)

# Print results
print("Custom Random Forest Results:")
print(classification_report(y_test, custom_predictions))
print("\nScikit-learn Random Forest Results:")
print(classification_report(y_test, sklearn_predictions))
```

Slide 10: Gradient Boosting Implementation

Gradient Boosting builds an ensemble of weak learners sequentially, where each learner tries to correct the errors of its predecessors. This implementation demonstrates the core concepts of gradient boosting with decision trees.

```python
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

class GradientBoostingFromScratch:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees = []
        
    def fit(self, X, y):
        # Initialize prediction with zeros
        current_predictions = np.zeros_like(y, dtype=float)
        
        for _ in range(self.n_estimators):
            # Calculate pseudo residuals
            residuals = y - current_predictions
            
            # Fit a tree on residuals
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X, residuals)
            
            # Update predictions
            predictions = tree.predict(X)
            current_predictions += self.learning_rate * predictions
            
            # Store the tree
            self.trees.append(tree)
            
            # Calculate current error
            mse = mean_squared_error(y, current_predictions)
            if mse < 1e-6:  # Early stopping condition
                break
    
    def predict(self, X):
        predictions = np.zeros(X.shape[0], dtype=float)
        for tree in self.trees:
            predictions += self.learning_rate * tree.predict(X)
        return predictions

# Generate synthetic regression data
np.random.seed(42)
X = np.sort(5 * np.random.rand(100, 1), axis=0)
y = np.sin(X).ravel() + np.random.normal(0, 0.1, X.shape[0])

# Train and evaluate the model
gbm = GradientBoostingFromScratch(n_estimators=100, learning_rate=0.1)
gbm.fit(X, y)

# Make predictions
predictions = gbm.predict(X)

# Calculate and print error metrics
mse = mean_squared_error(y, predictions)
print(f"Mean Squared Error: {mse:.4f}")

# Visualize results
import matplotlib.pyplot as plt
plt.scatter(X, y, color='blue', label='Actual')
plt.plot(X, predictions, color='red', label='Predicted')
plt.legend()
plt.title('Gradient Boosting Regression Results')
plt.show()
```

Slide 11: Support Vector Machine Implementation

Support Vector Machines find the optimal hyperplane that maximizes the margin between different classes. This implementation shows the core concepts of SVM optimization using the Sequential Minimal Optimization (SMO) algorithm.

```python
import numpy as np
from sklearn.preprocessing import StandardScaler

class SVMFromScratch:
    def __init__(self, C=1.0, kernel='linear', max_iter=1000):
        self.C = C
        self.kernel = kernel
        self.max_iter = max_iter
        self.alphas = None
        self.b = 0
        self.support_vectors = None
        self.support_vector_labels = None
        
    def linear_kernel(self, x1, x2):
        return np.dot(x1, x2)
    
    def rbf_kernel(self, x1, x2, gamma=0.1):
        return np.exp(-gamma * np.sum((x1 - x2) ** 2))
    
    def get_kernel(self, x1, x2):
        if self.kernel == 'linear':
            return self.linear_kernel(x1, x2)
        elif self.kernel == 'rbf':
            return self.rbf_kernel(x1, x2)
            
    def compute_kernel_matrix(self, X):
        n_samples = X.shape[0]
        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                K[i,j] = self.get_kernel(X[i], X[j])
        return K
    
    def fit(self, X, y):
        n_samples = X.shape[0]
        self.alphas = np.zeros(n_samples)
        self.K = self.compute_kernel_matrix(X)
        
        # SMO algorithm
        for _ in range(self.max_iter):
            alpha_prev = np.copy(self.alphas)
            
            for i in range(n_samples):
                j = self.select_second_alpha(i, n_samples)
                
                eta = self.K[i,i] + self.K[j,j] - 2 * self.K[i,j]
                if eta <= 0:
                    continue
                    
                alpha_i_old = self.alphas[i]
                alpha_j_old = self.alphas[j]
                
                # Calculate bounds
                if y[i] != y[j]:
                    L = max(0, self.alphas[j] - self.alphas[i])
                    H = min(self.C, self.C + self.alphas[j] - self.alphas[i])
                else:
                    L = max(0, self.alphas[i] + self.alphas[j] - self.C)
                    H = min(self.C, self.alphas[i] + self.alphas[j])
                
                if L == H:
                    continue
                
                # Update alphas
                E_i = self.decision_function(X[i]) - y[i]
                E_j = self.decision_function(X[j]) - y[j]
                
                self.alphas[j] += y[j] * (E_i - E_j) / eta
                self.alphas[j] = np.clip(self.alphas[j], L, H)
                self.alphas[i] += y[i] * y[j] * (alpha_j_old - self.alphas[j])
                
                # Update bias term
                b1 = self.b - E_i - y[i] * (self.alphas[i] - alpha_i_old) * self.K[i,i] \
                     - y[j] * (self.alphas[j] - alpha_j_old) * self.K[i,j]
                b2 = self.b - E_j - y[i] * (self.alphas[i] - alpha_i_old) * self.K[i,j] \
                     - y[j] * (self.alphas[j] - alpha_j_old) * self.K[j,j]
                
                self.b = (b1 + b2) / 2
            
            # Check convergence
            if np.allclose(self.alphas, alpha_prev):
                break
                
        # Store support vectors
        sv = self.alphas > 1e-5
        self.support_vectors = X[sv]
        self.support_vector_labels = y[sv]
        self.alphas = self.alphas[sv]
    
    def select_second_alpha(self, i, n_samples):
        j = i
        while j == i:
            j = np.random.randint(0, n_samples)
        return j
    
    def decision_function(self, x):
        if x.ndim == 1:
            x = x.reshape(1, -1)
        decision = np.sum(self.alphas * self.support_vector_labels * 
                         np.apply_along_axis(self.get_kernel, 1, 
                                           self.support_vectors, x))
        return decision + self.b
    
    def predict(self, X):
        return np.sign(self.decision_function(X))

# Generate synthetic dataset
from sklearn.datasets import make_classification
X, y = make_classification(n_samples=100, n_features=2, n_redundant=0,
                         n_informative=2, random_state=1,
                         n_clusters_per_class=1)
y = np.where(y == 0, -1, 1)

# Scale features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train SVM
svm = SVMFromScratch(C=1.0)
svm.fit(X, y)

# Make predictions
predictions = svm.predict(X)
accuracy = np.mean(predictions == y)
print(f"Accuracy: {accuracy:.4f}")
```

Slide 12: K-Means Clustering Implementation

K-means clustering is an unsupervised learning algorithm that partitions data into k clusters based on feature similarity. This implementation demonstrates the iterative process of centroid updating and cluster assignment.

```python
import numpy as np
import matplotlib.pyplot as plt

class KMeansFromScratch:
    def __init__(self, n_clusters=3, max_iters=100, random_state=42):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.random_state = random_state
        self.centroids = None
        self.labels = None
        
    def initialize_centroids(self, X):
        np.random.seed(self.random_state)
        random_idx = np.random.permutation(X.shape[0])
        centroids = X[random_idx[:self.n_clusters]]
        return centroids
    
    def compute_distances(self, X, centroids):
        distances = np.sqrt(((X - centroids[:, np.newaxis])**2).sum(axis=2))
        return distances
    
    def find_closest_cluster(self, distances):
        return np.argmin(distances, axis=0)
    
    def compute_centroids(self, X, labels):
        centroids = np.array([X[labels == k].mean(axis=0)
                            for k in range(self.n_clusters)])
        return centroids
    
    def fit(self, X):
        self.centroids = self.initialize_centroids(X)
        
        for _ in range(self.max_iters):
            old_centroids = self.centroids.copy()
            
            # Assign points to closest centroid
            distances = self.compute_distances(X, self.centroids)
            self.labels = self.find_closest_cluster(distances)
            
            # Update centroids
            self.centroids = self.compute_centroids(X, self.labels)
            
            # Check convergence
            if np.all(old_centroids == self.centroids):
                break
                
        return self
    
    def predict(self, X):
        distances = self.compute_distances(X, self.centroids)
        return self.find_closest_cluster(distances)
    
    def inertia(self, X):
        distances = self.compute_distances(X, self.centroids)
        return np.sum(np.min(distances, axis=0)**2)

# Generate synthetic clustering data
np.random.seed(42)
n_samples = 300
X = np.concatenate([
    np.random.normal(0, 1, (n_samples, 2)),
    np.random.normal(4, 1, (n_samples, 2)),
    np.random.normal(-4, 1, (n_samples, 2))
])

# Fit K-means
kmeans = KMeansFromScratch(n_clusters=3)
kmeans.fit(X)

# Plot results
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels, cmap='viridis')
plt.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], 
            c='red', marker='x', s=200, linewidth=3)
plt.title('K-Means Clustering Results')
plt.show()

# Calculate and print inertia
print(f"Inertia: {kmeans.inertia(X):.2f}")

# Evaluate clustering quality
from sklearn.metrics import silhouette_score
print(f"Silhouette Score: {silhouette_score(X, kmeans.labels):.4f}")
```

Slide 13: Principal Component Analysis Implementation

PCA is a dimensionality reduction technique that identifies the directions of maximum variance in high-dimensional data. This implementation shows the computation of eigenvalues, eigenvectors, and projection onto principal components.

```python
import numpy as np

class PCAFromScratch:
    def __init__(self, n_components=None):
        self.n_components = n_components
        self.components = None
        self.mean = None
        self.explained_variance_ratio = None
        
    def fit(self, X):
        # Center the data
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean
        
        # Compute covariance matrix
        n_samples = X.shape[0]
        cov_matrix = np.dot(X_centered.T, X_centered) / (n_samples - 1)
        
        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # Sort eigenvalues and eigenvectors in descending order
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Store explained variance ratio
        total_var = np.sum(eigenvalues)
        self.explained_variance_ratio = eigenvalues / total_var
        
        # Select top n_components
        if self.n_components is None:
            self.n_components = X.shape[1]
        self.components = eigenvectors[:, :self.n_components]
        
        return self
    
    def transform(self, X):
        # Project data onto principal components
        X_centered = X - self.mean
        return np.dot(X_centered, self.components)
    
    def inverse_transform(self, X_transformed):
        # Project data back to original space
        return np.dot(X_transformed, self.components.T) + self.mean

# Generate synthetic high-dimensional data
np.random.seed(42)
n_samples = 1000
n_features = 50
X = np.random.randn(n_samples, n_features)
# Add some structure to the data
X[:, 0] = X[:, 1] + np.random.normal(0, 0.1, n_samples)
X[:, 2] = X[:, 3] - 2 * X[:, 4] + np.random.normal(0, 0.1, n_samples)

# Fit PCA and transform data
pca = PCAFromScratch(n_components=10)
pca.fit(X)
X_transformed = pca.transform(X)

# Print explained variance ratios
print("Explained variance ratios:")
for i, ratio in enumerate(pca.explained_variance_ratio[:10]):
    print(f"PC{i+1}: {ratio:.4f}")

# Reconstruct data and compute reconstruction error
X_reconstructed = pca.inverse_transform(X_transformed)
reconstruction_error = np.mean((X - X_reconstructed) ** 2)
print(f"\nReconstruction Error: {reconstruction_error:.6f}")

# Plot cumulative explained variance ratio
plt.figure(figsize=(10, 6))
plt.plot(np.cumsum(pca.explained_variance_ratio))
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance Ratio')
plt.title('PCA Explained Variance Ratio')
plt.grid(True)
plt.show()
```

Slide 14: Additional Resources

*   ArXiv Papers for Further Reading:
    *   "A Tutorial on Support Vector Machines for Pattern Recognition" - [https://arxiv.org/abs/1101.3581](https://arxiv.org/abs/1101.3581)
    *   "Random Forests in Machine Learning" - [https://arxiv.org/abs/2001.09455](https://arxiv.org/abs/2001.09455)
    *   "Understanding the Difficulty of Training Deep Feedforward Neural Networks" - [https://arxiv.org/abs/1502.02791](https://arxiv.org/abs/1502.02791)
    *   "XGBoost: A Scalable Tree Boosting System" - [https://arxiv.org/abs/1603.02754](https://arxiv.org/abs/1603.02754)
    *   "Gradient-Based Learning Applied to Document Recognition" - [https://arxiv.org/abs/1998.01365](https://arxiv.org/abs/1998.01365)
*   Recommended Learning Resources:
    *   Google Scholar for latest ML research papers
    *   Kaggle Competitions for practical experience
    *   GitHub repositories of major ML frameworks
    *   Online ML courses from top universities
*   Tools and Libraries:
    *   scikit-learn documentation
    *   TensorFlow and PyTorch tutorials
    *   Keras documentation and examples
    *   NumPy and Pandas documentation

