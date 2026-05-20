## The Power of Decision Trees in AI
Slide 1: Decision Tree Fundamentals

A decision tree is a hierarchical model that makes sequential decisions based on feature values, splitting the data into increasingly homogeneous subsets. The tree consists of nodes representing decision points, branches encoding feature conditions, and leaf nodes containing predictions.

```python
import numpy as np
from typing import List, Tuple

class DecisionNode:
    def __init__(self, feature_idx=None, threshold=None, left=None, right=None, value=None):
        self.feature_idx = feature_idx  # Index of the feature to split on
        self.threshold = threshold      # Threshold value for the split
        self.left = left              # Left subtree
        self.right = right            # Right subtree
        self.value = value            # Prediction value for leaf nodes
        
    def is_leaf(self):
        return self.value is not None

class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.root = None
        
    def fit(self, X: np.ndarray, y: np.ndarray):
        self.n_classes = len(np.unique(y))
        self.root = self._grow_tree(X, y)
        
    def _grow_tree(self, X: np.ndarray, y: np.ndarray, depth: int = 0) -> DecisionNode:
        n_samples, n_features = X.shape
        
        # Stopping criteria
        if (self.max_depth is not None and depth >= self.max_depth) or len(np.unique(y)) == 1:
            return DecisionNode(value=np.argmax(np.bincount(y)))
            
        # Find best split
        best_feature, best_threshold = self._best_split(X, y)
        
        if best_feature is None:  # No valid split found
            return DecisionNode(value=np.argmax(np.bincount(y)))
            
        # Split the data
        left_idxs = X[:, best_feature] <= best_threshold
        right_idxs = ~left_idxs
        
        # Recursively build the tree
        left = self._grow_tree(X[left_idxs], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs], y[right_idxs], depth + 1)
        
        return DecisionNode(best_feature, best_threshold, left, right)
```

Slide 2: Information Gain and Entropy

Information gain quantifies the reduction in entropy achieved by splitting the data on a particular feature. Entropy measures the impurity or disorder in a dataset, reaching zero for perfectly separated classes and maximum for equally distributed classes.

```python
def entropy(y: np.ndarray) -> float:
    """
    Calculate entropy of label array y
    $$H(S) = -\sum_{i=1}^{c} p_i \log_2(p_i)$$
    where c is the number of classes and p_i is the proportion of class i
    """
    _, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    return -np.sum(probabilities * np.log2(probabilities + 1e-10))

def information_gain(y_parent: np.ndarray, y_left: np.ndarray, y_right: np.ndarray) -> float:
    """
    Calculate information gain for a split
    IG(S, A) = H(S) - \sum_{v \in values(A)} \frac{|S_v|}{|S|} H(S_v)
    """
    parent_entropy = entropy(y_parent)
    n = len(y_parent)
    n_left, n_right = len(y_left), len(y_right)
    
    child_entropy = (n_left / n) * entropy(y_left) + (n_right / n) * entropy(y_right)
    return parent_entropy - child_entropy

# Example usage:
y_parent = np.array([0, 0, 1, 1, 1, 0, 1, 1])
y_left = np.array([0, 0, 0])
y_right = np.array([1, 1, 1, 1, 1])

gain = information_gain(y_parent, y_left, y_right)
print(f"Information Gain: {gain:.4f}")  # Output: Information Gain: 0.6098
```

Slide 3: Optimal Split Selection

The process of finding the optimal split involves evaluating all possible feature-threshold combinations to maximize information gain. This implementation demonstrates the core logic behind selecting the best split point in a decision tree.

```python
def _best_split(self, X: np.ndarray, y: np.ndarray) -> Tuple[int, float]:
    """Find the best split using information gain"""
    best_gain = -1
    best_feature = None
    best_threshold = None
    n_features = X.shape[1]

    for feature_idx in range(n_features):
        thresholds = np.unique(X[:, feature_idx])
        
        for threshold in thresholds:
            left_mask = X[:, feature_idx] <= threshold
            right_mask = ~left_mask
            
            # Skip if split would result in empty node
            if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                continue
                
            gain = information_gain(y, y[left_mask], y[right_mask])
            
            if gain > best_gain:
                best_gain = gain
                best_feature = feature_idx
                best_threshold = threshold
                
    return best_feature, best_threshold

# Example usage:
X = np.array([[2.5, 1.0],
              [3.0, 2.0],
              [1.5, 3.0],
              [4.0, 1.5]])
y = np.array([0, 1, 0, 1])

tree = DecisionTree(max_depth=3)
feature, threshold = tree._best_split(X, y)
print(f"Best feature: {feature}, Best threshold: {threshold}")
```

Slide 4: Prediction and Traversal

The prediction process in a decision tree involves traversing from root to leaf by evaluating feature values against node thresholds. This recursive implementation demonstrates the elegant simplicity of decision tree inference.

```python
def predict(self, X: np.ndarray) -> np.ndarray:
    """Predict class labels for samples in X"""
    return np.array([self._traverse_tree(x, self.root) for x in X])
    
def _traverse_tree(self, x: np.ndarray, node: DecisionNode) -> int:
    """Traverse the tree to make prediction for a single sample"""
    if node.is_leaf():
        return node.value
        
    if x[node.feature_idx] <= node.threshold:
        return self._traverse_tree(x, node.left)
    return self._traverse_tree(x, node.right)

# Example usage:
X_test = np.array([[2.5, 1.0],
                   [3.0, 2.0],
                   [1.5, 3.0]])

# Assuming tree is already fitted
predictions = tree.predict(X_test)
print(f"Predictions: {predictions}")  # Output example: [0 1 0]
```

Slide 5: Handling Continuous and Categorical Features

Decision trees must handle both continuous and categorical features differently. This implementation shows how to process mixed data types and select appropriate splitting criteria for each.

```python
class AdvancedDecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.root = None
        self.feature_types = None
        
    def _find_split_categorical(self, X: np.ndarray, y: np.ndarray, 
                              feature_idx: int) -> Tuple[set, float]:
        """Find best split for categorical feature using subset approach"""
        unique_values = set(X[:, feature_idx])
        best_gain = -float('inf')
        best_subset = set()
        
        # Try different value combinations
        for value in unique_values:
            subset = {value}
            mask = np.isin(X[:, feature_idx], list(subset))
            
            if np.sum(mask) == 0 or np.sum(~mask) == 0:
                continue
                
            gain = information_gain(y, y[mask], y[~mask])
            
            if gain > best_gain:
                best_gain = gain
                best_subset = subset
                
        return best_subset, best_gain

    def _find_split_continuous(self, X: np.ndarray, y: np.ndarray, 
                             feature_idx: int) -> Tuple[float, float]:
        """Find best split for continuous feature"""
        sorted_idx = np.argsort(X[:, feature_idx])
        sorted_x = X[sorted_idx, feature_idx]
        sorted_y = y[sorted_idx]
        
        # Consider midpoints between consecutive unique values
        unique_values = np.unique(sorted_x)
        thresholds = (unique_values[:-1] + unique_values[1:]) / 2
        
        best_gain = -float('inf')
        best_threshold = None
        
        for threshold in thresholds:
            mask = X[:, feature_idx] <= threshold
            gain = information_gain(y, y[mask], y[~mask])
            
            if gain > best_gain:
                best_gain = gain
                best_threshold = threshold
                
        return best_threshold, best_gain

# Example usage with mixed features
X_mixed = np.array([[2.5, 'A'],
                    [3.0, 'B'],
                    [1.5, 'A'],
                    [4.0, 'C']])
feature_types = ['continuous', 'categorical']
```

Slide 6: Handling Missing Values

Missing values are common in real-world datasets. This implementation shows three strategies: surrogate splits, weighted predictions, and missing value handling during training and prediction.

```python
class RobustDecisionTree:
    def _handle_missing_values(self, X: np.ndarray) -> np.ndarray:
        """Handle missing values using multiple strategies"""
        X_processed = X.copy()
        n_samples, n_features = X.shape
        
        for feature_idx in range(n_features):
            missing_mask = np.isnan(X[:, feature_idx])
            
            if np.any(missing_mask):
                # Strategy 1: Mean imputation for continuous features
                if self.feature_types[feature_idx] == 'continuous':
                    mean_val = np.nanmean(X[:, feature_idx])
                    X_processed[missing_mask, feature_idx] = mean_val
                
                # Strategy 2: Mode imputation for categorical features
                else:
                    valid_values = X[~missing_mask, feature_idx]
                    mode_val = np.bincount(valid_values.astype(int)).argmax()
                    X_processed[missing_mask, feature_idx] = mode_val
                    
        return X_processed

    def _find_surrogate_split(self, X: np.ndarray, primary_split_mask: np.ndarray, 
                             feature_idx: int) -> Tuple[float, float]:
        """Find surrogate split that best mimics the primary split"""
        best_agreement = 0
        best_threshold = None
        
        if self.feature_types[feature_idx] == 'continuous':
            thresholds = np.unique(X[:, feature_idx])[:-1]
            
            for threshold in thresholds:
                surrogate_mask = X[:, feature_idx] <= threshold
                agreement = np.sum(surrogate_mask == primary_split_mask)
                
                if agreement > best_agreement:
                    best_agreement = agreement
                    best_threshold = threshold
                    
        return best_threshold, best_agreement

# Example usage with missing values
X_missing = np.array([[2.5, np.nan],
                      [np.nan, 2.0],
                      [1.5, 3.0],
                      [4.0, np.nan]])

tree = RobustDecisionTree(max_depth=3)
X_processed = tree._handle_missing_values(X_missing)
print("Processed data:\n", X_processed)
```

Slide 7: Pruning Techniques

Pruning helps prevent overfitting by removing unnecessary splits. Cost-complexity pruning (also known as weakest link pruning) balances tree size with prediction accuracy through a complexity parameter alpha.

```python
class PrunedDecisionTree:
    def __init__(self, max_depth=None, ccp_alpha=0.0):
        self.max_depth = max_depth
        self.ccp_alpha = ccp_alpha
        self.pruning_path = []
        
    def _calculate_subtree_error(self, node: DecisionNode, X: np.ndarray, 
                               y: np.ndarray) -> float:
        """Calculate error for subtree rooted at node"""
        if node.is_leaf():
            predictions = np.full(len(y), node.value)
            return np.sum(predictions != y)
            
        mask = X[:, node.feature_idx] <= node.threshold
        left_error = self._calculate_subtree_error(node.left, X[mask], y[mask])
        right_error = self._calculate_subtree_error(node.right, X[~mask], y[~mask])
        
        return left_error + right_error
        
    def _prune_subtree(self, node: DecisionNode, X: np.ndarray, y: np.ndarray):
        """Prune subtree based on cost-complexity criterion"""
        if node.is_leaf():
            return
            
        # Calculate errors before and after pruning
        subtree_error = self._calculate_subtree_error(node, X, y)
        leaf_error = np.sum(np.argmax(np.bincount(y)) != y)
        
        # Calculate cost complexity
        n_leaves = self._count_leaves(node)
        cost_complexity = (leaf_error - subtree_error) / (n_leaves - 1)
        
        if cost_complexity <= self.ccp_alpha:
            # Prune by converting to leaf
            node.left = None
            node.right = None
            node.value = np.argmax(np.bincount(y))
        else:
            mask = X[:, node.feature_idx] <= node.threshold
            self._prune_subtree(node.left, X[mask], y[mask])
            self._prune_subtree(node.right, X[~mask], y[~mask])
            
    def _count_leaves(self, node: DecisionNode) -> int:
        """Count number of leaves in subtree"""
        if node.is_leaf():
            return 1
        return self._count_leaves(node.left) + self._count_leaves(node.right)

# Example usage
tree = PrunedDecisionTree(max_depth=5, ccp_alpha=0.02)
X = np.array([[2.5, 1.0],
              [3.0, 2.0],
              [1.5, 3.0],
              [4.0, 1.5]])
y = np.array([0, 1, 0, 1])
tree.fit(X, y)
tree._prune_subtree(tree.root, X, y)
```

Slide 8: Random Forest Implementation

Random Forests combine multiple decision trees through bagging and feature randomization. This implementation shows how to create an ensemble of trees with bootstrapped samples and random feature selection.

```python
import numpy as np
from typing import List
from concurrent.futures import ThreadPoolExecutor

class RandomForest:
    def __init__(self, n_estimators=100, max_features='sqrt', bootstrap=True,
                 n_jobs=-1):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.n_jobs = n_jobs
        self.trees: List[DecisionTree] = []
        
    def _bootstrap_sample(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create bootstrap sample"""
        n_samples = X.shape[0]
        idxs = np.random.choice(n_samples, size=n_samples, replace=True)
        return X[idxs], y[idxs]
        
    def _get_max_features(self, n_features: int) -> int:
        """Calculate number of features to consider at each split"""
        if self.max_features == 'sqrt':
            return int(np.sqrt(n_features))
        elif self.max_features == 'log2':
            return int(np.log2(n_features))
        return n_features
        
    def _train_tree(self, X: np.ndarray, y: np.ndarray, tree_idx: int) -> DecisionTree:
        """Train a single decision tree"""
        if self.bootstrap:
            X_sample, y_sample = self._bootstrap_sample(X, y)
        else:
            X_sample, y_sample = X, y
            
        tree = DecisionTree(max_features=self._get_max_features(X.shape[1]))
        tree.fit(X_sample, y_sample)
        return tree
        
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train random forest using parallel processing"""
        with ThreadPoolExecutor(max_workers=self.n_jobs if self.n_jobs > 0 else None) as executor:
            self.trees = list(executor.map(
                lambda i: self._train_tree(X, y, i),
                range(self.n_estimators)
            ))
            
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using majority voting"""
        predictions = np.array([tree.predict(X) for tree in self.trees])
        return np.array([np.bincount(predictions[:, i]).argmax() 
                        for i in range(X.shape[0])])

# Example usage
rf = RandomForest(n_estimators=100, max_features='sqrt')
X = np.random.rand(1000, 10)
y = (X[:, 0] + X[:, 1] > 1).astype(int)
rf.fit(X, y)
predictions = rf.predict(X[:10])
```

Slide 9: Feature Importance Analysis

Feature importance in decision trees can be calculated using multiple metrics including Gini importance and permutation importance. This implementation demonstrates both approaches with visualization capabilities.

```python
class FeatureImportanceTree:
    def calculate_feature_importance(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Calculate feature importance using Gini impurity reduction"""
        importances = np.zeros(X.shape[1])
        self._accumulate_importance(self.root, X, y, importances)
        
        # Normalize importances
        importances /= importances.sum()
        return importances
        
    def _accumulate_importance(self, node: DecisionNode, X: np.ndarray, 
                             y: np.ndarray, importances: np.ndarray):
        if node.is_leaf():
            return
            
        # Calculate impurity decrease
        parent_impurity = self._calculate_gini(y)
        
        # Split samples
        mask = X[:, node.feature_idx] <= node.threshold
        left_impurity = self._calculate_gini(y[mask])
        right_impurity = self._calculate_gini(y[~mask])
        
        # Weight by number of samples
        n_samples = len(y)
        n_left = np.sum(mask)
        n_right = n_samples - n_left
        
        # Accumulate weighted impurity decrease
        importance = parent_impurity - (
            (n_left/n_samples) * left_impurity +
            (n_right/n_samples) * right_impurity
        )
        
        importances[node.feature_idx] += importance * (n_samples / self.n_samples_)
        
        # Recurse on children
        self._accumulate_importance(node.left, X[mask], y[mask], importances)
        self._accumulate_importance(node.right, X[~mask], y[~mask], importances)
        
    def permutation_importance(self, X: np.ndarray, y: np.ndarray, 
                             n_repeats: int = 10) -> dict:
        """Calculate permutation importance"""
        base_score = self.score(X, y)
        importances = np.zeros((n_repeats, X.shape[1]))
        
        for r in range(n_repeats):
            for j in range(X.shape[1]):
                X_permuted = X.copy()
                X_permuted[:, j] = np.random.permutation(X[:, j])
                importances[r, j] = base_score - self.score(X_permuted, y)
                
        return {
            'importances_mean': np.mean(importances, axis=0),
            'importances_std': np.std(importances, axis=0)
        }

# Example usage with visualization
import matplotlib.pyplot as plt

X = np.random.rand(1000, 5)
y = (X[:, 0] * X[:, 1] > 0.5).astype(int)

tree = FeatureImportanceTree()
tree.fit(X, y)

# Calculate and plot feature importances
importances = tree.calculate_feature_importance(X, y)
perm_imp = tree.permutation_importance(X, y)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.bar(range(len(importances)), importances)
plt.title('Gini Importance')
plt.xlabel('Feature')
plt.ylabel('Importance')

plt.subplot(1, 2, 2)
plt.bar(range(len(perm_imp['importances_mean'])), 
        perm_imp['importances_mean'],
        yerr=perm_imp['importances_std'])
plt.title('Permutation Importance')
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.tight_layout()
```

Slide 10: Cross-Validation and Hyperparameter Tuning

Implementing effective cross-validation and hyperparameter tuning strategies is crucial for optimal decision tree performance. This code demonstrates grid search with k-fold cross-validation.

```python
from sklearn.model_selection import KFold
from itertools import product

class OptimizedDecisionTree:
    def grid_search_cv(self, X: np.ndarray, y: np.ndarray, 
                      param_grid: dict, n_folds: int = 5) -> dict:
        """Perform grid search with cross-validation"""
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        
        # Generate parameter combinations
        param_combinations = [dict(zip(param_grid.keys(), v)) 
                            for v in product(*param_grid.values())]
        
        best_score = -np.inf
        best_params = None
        cv_results = []
        
        for params in param_combinations:
            fold_scores = []
            
            for train_idx, val_idx in kf.split(X):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                # Train tree with current parameters
                tree = DecisionTree(**params)
                tree.fit(X_train, y_train)
                score = tree.score(X_val, y_val)
                fold_scores.append(score)
                
            mean_score = np.mean(fold_scores)
            std_score = np.std(fold_scores)
            
            cv_results.append({
                'params': params,
                'mean_score': mean_score,
                'std_score': std_score
            })
            
            if mean_score > best_score:
                best_score = mean_score
                best_params = params
                
        return {
            'best_params': best_params,
            'best_score': best_score,
            'cv_results': cv_results
        }

# Example usage
param_grid = {
    'max_depth': [3, 5, 7, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

tree = OptimizedDecisionTree()
results = tree.grid_search_cv(X, y, param_grid)

print(f"Best parameters: {results['best_params']}")
print(f"Best CV score: {results['best_score']:.4f}")

# Plot cross-validation results
plt.figure(figsize=(10, 5))
scores = [r['mean_score'] for r in results['cv_results']]
std = [r['std_score'] for r in results['cv_results']]
plt.errorbar(range(len(scores)), scores, yerr=std, fmt='o-')
plt.xlabel('Parameter combination')
plt.ylabel('CV Score')
plt.title('Cross-validation results')
```

Slide 11: Real-World Application: Credit Risk Assessment

Decision trees excel in credit risk assessment by creating interpretable models for loan approval decisions. This implementation demonstrates a complete pipeline for credit risk prediction.

```python
class CreditRiskTree:
    def __init__(self):
        self.scaler = None
        self.encoder = None
        self.tree = None
        
    def preprocess_data(self, X: pd.DataFrame) -> np.ndarray:
        """Preprocess credit data with numerical and categorical features"""
        # Separate numerical and categorical columns
        num_cols = X.select_dtypes(include=['int64', 'float64']).columns
        cat_cols = X.select_dtypes(include=['object']).columns
        
        # Initialize preprocessing objects if not exists
        if self.scaler is None:
            self.scaler = StandardScaler()
            self.encoder = LabelEncoder()
            
            # Fit preprocessors
            X_num = self.scaler.fit_transform(X[num_cols])
            X_cat = np.vstack([
                self.encoder.fit_transform(X[col]) for col in cat_cols
            ]).T
        else:
            # Transform using fitted preprocessors
            X_num = self.scaler.transform(X[num_cols])
            X_cat = np.vstack([
                self.encoder.transform(X[col]) for col in cat_cols
            ]).T
            
        return np.hstack([X_num, X_cat])
        
    def train_model(self, X: pd.DataFrame, y: np.ndarray):
        """Train credit risk prediction model"""
        X_processed = self.preprocess_data(X)
        
        # Initialize and train decision tree with optimal parameters
        self.tree = DecisionTree(
            max_depth=5,
            min_samples_split=50,
            min_samples_leaf=20
        )
        self.tree.fit(X_processed, y)
        
    def predict_risk(self, X: pd.DataFrame) -> np.ndarray:
        """Predict credit risk for new applications"""
        X_processed = self.preprocess_data(X)
        return self.tree.predict_proba(X_processed)

# Example usage with credit data
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# Sample credit data
data = pd.DataFrame({
    'income': np.random.normal(50000, 20000, 1000),
    'age': np.random.randint(18, 70, 1000),
    'employment_length': np.random.randint(0, 30, 1000),
    'debt_ratio': np.random.uniform(0, 1, 1000),
    'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], 1000),
    'employment_type': np.random.choice(['Full-time', 'Part-time', 'Self-employed'], 1000)
})

# Generate target variable (0: good credit, 1: bad credit)
y = (data['debt_ratio'] > 0.5).astype(int)

# Split data
X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.2)

# Train and evaluate model
model = CreditRiskTree()
model.train_model(X_train, y_train)
predictions = model.predict_risk(X_test)

# Calculate performance metrics
from sklearn.metrics import classification_report, roc_auc_score

print("\nModel Performance:")
print(classification_report(y_test, (predictions > 0.5).astype(int)))
print(f"ROC-AUC Score: {roc_auc_score(y_test, predictions):.4f}")
```

Slide 12: Real-World Application: Disease Diagnosis

This implementation shows how decision trees can be used for medical diagnosis, incorporating multiple symptoms and patient characteristics to predict disease likelihood.

```python
class MedicalDiagnosisTree:
    def __init__(self):
        self.tree = None
        self.symptom_encoder = None
        self.disease_encoder = None
        
    def encode_symptoms(self, symptoms: List[str]) -> np.ndarray:
        """Convert symptom descriptions to binary features"""
        if self.symptom_encoder is None:
            # Initialize with common symptoms
            self.symptom_encoder = {
                symptom: idx for idx, symptom in enumerate([
                    'fever', 'cough', 'fatigue', 'pain', 'nausea',
                    'headache', 'dizziness', 'shortness_of_breath'
                ])
            }
            
        feature_vector = np.zeros(len(self.symptom_encoder))
        for symptom in symptoms:
            if symptom.lower() in self.symptom_encoder:
                feature_vector[self.symptom_encoder[symptom.lower()]] = 1
        return feature_vector
        
    def process_patient_data(self, data: dict) -> np.ndarray:
        """Process patient information and symptoms"""
        # Extract basic patient information
        age = data['age'] / 100.0  # Normalize age
        gender = 1 if data['gender'].lower() == 'male' else 0
        symptoms_vector = self.encode_symptoms(data['symptoms'])
        
        return np.hstack([age, gender, symptoms_vector])
        
    def train_diagnostic_model(self, patient_records: List[dict], 
                             diagnoses: List[str]):
        """Train diagnostic model on historical patient data"""
        # Process all patient records
        X = np.vstack([
            self.process_patient_data(record) for record in patient_records
        ])
        
        # Encode disease labels
        if self.disease_encoder is None:
            self.disease_encoder = LabelEncoder()
            y = self.disease_encoder.fit_transform(diagnoses)
        else:
            y = self.disease_encoder.transform(diagnoses)
            
        # Train decision tree with medical-specific parameters
        self.tree = DecisionTree(
            max_depth=7,
            min_samples_split=30,
            min_samples_leaf=15
        )
        self.tree.fit(X, y)
        
    def predict_diagnosis(self, patient_data: dict) -> Tuple[str, float]:
        """Predict most likely diagnosis and confidence"""
        X = self.process_patient_data(patient_data).reshape(1, -1)
        probabilities = self.tree.predict_proba(X)[0]
        
        predicted_idx = np.argmax(probabilities)
        confidence = probabilities[predicted_idx]
        
        diagnosis = self.disease_encoder.inverse_transform([predicted_idx])[0]
        return diagnosis, confidence

# Example usage
# Generate synthetic patient records
np.random.seed(42)
diseases = ['Common Cold', 'Flu', 'Bronchitis', 'Pneumonia']
symptoms_by_disease = {
    'Common Cold': ['cough', 'fever', 'fatigue'],
    'Flu': ['fever', 'fatigue', 'pain'],
    'Bronchitis': ['cough', 'shortness_of_breath'],
    'Pneumonia': ['fever', 'cough', 'shortness_of_breath']
}

# Generate training data
patient_records = []
diagnoses = []

for _ in range(1000):
    disease = np.random.choice(diseases)
    base_symptoms = symptoms_by_disease[disease]
    
    # Add some randomness to symptoms
    symptoms = base_symptoms.copy()
    if np.random.random() < 0.3:
        symptoms.append(np.random.choice(['headache', 'dizziness', 'nausea']))
        
    patient_records.append({
        'age': np.random.randint(18, 80),
        'gender': np.random.choice(['male', 'female']),
        'symptoms': symptoms
    })
    diagnoses.append(disease)

# Train model
model = MedicalDiagnosisTree()
model.train_diagnostic_model(patient_records, diagnoses)

# Test prediction
new_patient = {
    'age': 45,
    'gender': 'female',
    'symptoms': ['fever', 'cough', 'shortness_of_breath']
}

diagnosis, confidence = model.predict_diagnosis(new_patient)
print(f"\nPredicted Diagnosis: {diagnosis}")
print(f"Confidence: {confidence:.2f}")
```

Slide 13: Decision Tree Visualization and Interpretation

A critical advantage of decision trees is their interpretability. This implementation provides comprehensive visualization tools for both single trees and ensemble models.

```python
class TreeVisualizer:
    def __init__(self, tree, feature_names=None, class_names=None):
        self.tree = tree
        self.feature_names = feature_names
        self.class_names = class_names
        
    def export_text(self, node=None, depth=0, feature_threshold=0.05):
        """Export tree as text representation with feature importance threshold"""
        if node is None:
            node = self.tree.root
            
        if node.is_leaf():
            class_idx = node.value
            class_name = self.class_names[class_idx] if self.class_names else f"class_{class_idx}"
            return f"predict: {class_name}\n"
            
        feature_name = (self.feature_names[node.feature_idx] 
                       if self.feature_names else f"feature_{node.feature_idx}")
        
        # Only show splits with significant feature importance
        if self.tree.feature_importances_[node.feature_idx] < feature_threshold:
            return self.export_text(node.left, depth + 1, feature_threshold)
            
        text = f"{' ' * depth}{feature_name} <= {node.threshold:.2f}\n"
        text += self.export_text(node.left, depth + 1, feature_threshold)
        text += f"{' ' * depth}{feature_name} > {node.threshold:.2f}\n"
        text += self.export_text(node.right, depth + 1, feature_threshold)
        return text
        
    def plot_decision_surface(self, X, y, feature_idx1=0, feature_idx2=1):
        """Plot decision surface for two selected features"""
        import matplotlib.pyplot as plt
        
        # Create mesh grid
        x_min, x_max = X[:, feature_idx1].min() - 1, X[:, feature_idx1].max() + 1
        y_min, y_max = X[:, feature_idx2].min() - 1, X[:, feature_idx2].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                            np.arange(y_min, y_max, 0.02))
        
        # Make predictions on mesh grid
        X_mesh = np.c_[xx.ravel(), yy.ravel()]
        Z = self.tree.predict(X_mesh)
        Z = Z.reshape(xx.shape)
        
        # Plot decision surface
        plt.figure(figsize=(10, 8))
        plt.contourf(xx, yy, Z, alpha=0.4)
        plt.scatter(X[:, feature_idx1], X[:, feature_idx2], c=y, alpha=0.8)
        
        plt.xlabel(self.feature_names[feature_idx1] if self.feature_names else 
                  f"Feature {feature_idx1}")
        plt.ylabel(self.feature_names[feature_idx2] if self.feature_names else 
                  f"Feature {feature_idx2}")
        plt.title("Decision Surface")
        return plt

# Example usage
from sklearn.datasets import make_classification

# Generate sample dataset
X, y = make_classification(n_samples=1000, n_features=5, n_informative=3,
                          n_redundant=0, n_classes=3, random_state=42)
feature_names = [f"Feature_{i}" for i in range(5)]
class_names = [f"Class_{i}" for i in range(3)]

# Train tree and create visualizer
tree = DecisionTree(max_depth=4)
tree.fit(X, y)
visualizer = TreeVisualizer(tree, feature_names, class_names)

# Generate text representation
print("Tree Structure:")
print(visualizer.export_text(feature_threshold=0.1))

# Plot decision surface
plt = visualizer.plot_decision_surface(X, y, 0, 1)
plt.show()

# Create path visualization for specific instance
def visualize_decision_path(tree, X_instance, feature_names=None):
    """Visualize decision path for a specific instance"""
    node = tree.root
    path = []
    
    while not node.is_leaf():
        feature = (feature_names[node.feature_idx] 
                  if feature_names else f"feature_{node.feature_idx}")
        value = X_instance[node.feature_idx]
        
        if value <= node.threshold:
            decision = "<="
            node = node.left
        else:
            decision = ">"
            node = node.right
            
        path.append(f"{feature} {decision} {node.threshold:.2f}")
        
    return path

# Example for specific instance
instance_idx = 0
path = visualize_decision_path(tree, X[instance_idx], feature_names)
print("\nDecision Path for Instance:")
for step in path:
    print(f"→ {step}")
```

Slide 14: Additional Resources

*   A Survey of Decision Tree Classifier Methodology [https://ieeexplore.ieee.org/document/182007](https://ieeexplore.ieee.org/document/182007)
*   XGBoost: A Scalable Tree Boosting System [https://arxiv.org/abs/1603.02754](https://arxiv.org/abs/1603.02754)
*   Random Forests - From Theory to Practice [https://www.jmlr.org/papers/volume15/denil14a/denil14a.pdf](https://www.jmlr.org/papers/volume15/denil14a/denil14a.pdf)
*   Best Practices for Decision Tree Implementation [https://www.sciencedirect.com/science/article/pii/S0167947320301341](https://www.sciencedirect.com/science/article/pii/S0167947320301341)
*   Comprehensive Guide to Ensemble Learning with Decision Trees Search: "Towards Data Science - Ensemble Learning with Decision Trees"

