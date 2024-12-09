## Explainable AI Demystifying Complex Models
Slide 1: Introduction to SHAP Values

SHAP (SHapley Additive exPlanations) values provide a unified measure of feature importance based on game theoretic principles. These values represent the contribution of each feature to the prediction difference from the baseline, calculated by considering all possible feature combinations.

```python
# Basic SHAP implementation with scikit-learn and shap
import shap
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# Create synthetic dataset
X = np.random.random((100, 4))
y = X[:, 0] * 2 + X[:, 1] - X[:, 2] + np.random.random(100)

# Train model
model = RandomForestRegressor(n_estimators=100)
model.fit(X, y)

# Calculate SHAP values
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# Output example
print("SHAP values shape:", shap_values.shape)
print("First sample SHAP values:", shap_values[0])
```

Slide 2: TreeSHAP Implementation

TreeSHAP specifically optimizes SHAP value computation for tree-based models by leveraging the internal structure of decision trees. This implementation demonstrates the core algorithm for a single decision tree.

```python
def compute_tree_shap(tree, x, feature_names):
    def recurse(node, path_probability=1.0):
        if tree.is_leaf(node):
            return {feature: 0 for feature in feature_names}
        
        feature = tree.feature(node)
        threshold = tree.threshold(node)
        
        if x[feature] <= threshold:
            shap_values = recurse(tree.left_child(node), 
                                path_probability * tree.left_probability(node))
        else:
            shap_values = recurse(tree.right_child(node), 
                                path_probability * tree.right_probability(node))
            
        shap_values[feature_names[feature]] += path_probability * (
            tree.value(node.right) - tree.value(node.left))
        
        return shap_values
    
    return recurse(tree.root)
```

Slide 3: FastTreeSHAP v2 Implementation

FastTreeSHAP v2 leverages parallel computing and optimized data structures to accelerate SHAP value calculations. This implementation showcases the core components of the algorithm with vectorized operations.

```python
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from functools import partial

class FastTreeSHAPv2:
    def __init__(self, n_jobs=-1):
        self.n_jobs = n_jobs
    
    def _compute_weight_matrix(self, tree_structure):
        n_nodes = len(tree_structure)
        weights = np.zeros((n_nodes, n_nodes))
        
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                if tree_structure[j]['parent'] == i:
                    weights[i,j] = tree_structure[j]['weight']
        
        return weights
    
    def compute_shap_values(self, X, tree_model):
        partial_compute = partial(self._single_tree_shap, tree_model=tree_model)
        
        with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
            results = list(executor.map(partial_compute, X))
        
        return np.array(results)
```

Slide 4: SHAP Value Visualization

Understanding feature importance through visualization is crucial in XAI. This implementation creates comprehensive SHAP plots including summary plots, force plots, and dependency plots.

```python
import shap
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestRegressor

# Load and prepare data
boston = load_boston()
X, y = boston.data, boston.target
model = RandomForestRegressor(n_estimators=100).fit(X, y)

# Initialize SHAP explainer
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# Create visualization
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X, feature_names=boston.feature_names)
plt.title('SHAP Summary Plot')
plt.tight_layout()
plt.show()
```

Slide 5: Implementing Isolation Forest with SHAP

Isolation Forest combined with SHAP values provides powerful anomaly detection capabilities while maintaining interpretability. This implementation demonstrates how to integrate both techniques for explainable outlier detection.

```python
import numpy as np
from sklearn.ensemble import IsolationForest
import shap

# Create synthetic dataset with anomalies
np.random.seed(42)
X_normal = np.random.normal(0, 1, (1000, 3))
X_anomaly = np.random.normal(3, 1, (50, 3))
X = np.vstack([X_normal, X_anomaly])

# Train Isolation Forest
iso_forest = IsolationForest(n_estimators=100, contamination=0.1)
iso_forest.fit(X)

# Calculate SHAP values for anomaly detection
explainer = shap.TreeExplainer(iso_forest)
shap_values = explainer.shap_values(X)

# Print example anomaly explanation
print("SHAP values for first anomaly:", shap_values[-1])
```

Slide 6: Real-world Application: Credit Card Fraud Detection

A practical implementation of XAI for financial fraud detection, combining gradient boosting with SHAP explanations to create an interpretable fraud detection system that meets regulatory requirements.

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import shap

# Load and preprocess credit card data
def prepare_fraud_detection_model(data_path):
    # Load data (assuming CSV with transaction features)
    df = pd.read_csv(data_path)
    
    # Split features and target
    X = df.drop(['is_fraud', 'transaction_id'], axis=1)
    y = df['is_fraud']
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model
    model = XGBClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    
    # Calculate SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    
    return model, explainer, shap_values, X_test

# Example usage
model, explainer, shap_values, X_test = prepare_fraud_detection_model('fraud_data.csv')
```

Slide 7: SHAP Interaction Values

SHAP interaction values reveal how features work together to influence model predictions, providing deeper insights into feature relationships and complex patterns.

```python
import shap
import numpy as np
from sklearn.ensemble import RandomForestClassifier

def compute_shap_interactions(model, X):
    # Initialize SHAP explainer
    explainer = shap.TreeExplainer(model)
    
    # Calculate SHAP interaction values
    shap_interaction_values = explainer.shap_interaction_values(X)
    
    # Compute feature interaction strengths
    interaction_matrix = np.zeros((X.shape[1], X.shape[1]))
    for i in range(X.shape[1]):
        for j in range(X.shape[1]):
            interaction_matrix[i,j] = np.abs(
                shap_interaction_values[:, i, j]
            ).mean()
    
    return interaction_matrix, shap_interaction_values

# Example usage
model = RandomForestClassifier().fit(X_train, y_train)
interaction_matrix, interactions = compute_shap_interactions(model, X_test)
```

Slide 8: Local Interpretable Model-agnostic Explanations (LIME) Integration

This implementation combines SHAP with LIME to provide complementary explanations, offering both global and local interpretability for complex models.

```python
from lime import lime_tabular
import shap
import numpy as np

class HybridExplainer:
    def __init__(self, model, X_train):
        self.model = model
        self.shap_explainer = shap.TreeExplainer(model)
        self.lime_explainer = lime_tabular.LimeTabularExplainer(
            X_train,
            mode='regression',
            feature_names=feature_names
        )
    
    def explain_instance(self, x):
        # Get SHAP explanation
        shap_values = self.shap_explainer.shap_values(x)
        
        # Get LIME explanation
        lime_exp = self.lime_explainer.explain_instance(
            x, 
            self.model.predict
        )
        
        return {
            'shap_values': shap_values,
            'lime_explanation': lime_exp
        }

# Example usage
explainer = HybridExplainer(model, X_train)
explanation = explainer.explain_instance(X_test[0])
```

Slide 9: Optimization Techniques for TreeSHAP

This implementation focuses on memory optimization and computational efficiency for TreeSHAP calculations, particularly useful when dealing with large-scale datasets and deep tree ensembles.

```python
import numpy as np
from numba import jit
from concurrent.futures import ThreadPoolExecutor

class OptimizedTreeSHAP:
    def __init__(self, max_memory_gb=4):
        self.max_batch_size = self._calculate_batch_size(max_memory_gb)
    
    @staticmethod
    @jit(nopython=True)
    def _fast_tree_path(node_array, x, threshold_array):
        path = []
        node = 0
        while node_array[node] != -1:
            path.append(node)
            if x[threshold_array[node, 0]] <= threshold_array[node, 1]:
                node = node_array[node * 2 + 1]
            else:
                node = node_array[node * 2 + 2]
        return path
    
    def _calculate_batch_size(self, max_memory_gb):
        memory_bytes = max_memory_gb * 1024**3
        return int(memory_bytes / (self.feature_count * 8))
    
    def compute_shap(self, X, tree_ensemble):
        results = []
        for i in range(0, len(X), self.max_batch_size):
            batch = X[i:i + self.max_batch_size]
            batch_results = self._process_batch(batch, tree_ensemble)
            results.append(batch_results)
        return np.concatenate(results)
```

Slide 10: Feature Contribution Analysis

A comprehensive implementation for analyzing and visualizing feature contributions across different prediction scenarios, helping identify key drivers in model decisions.

```python
import pandas as pd
import numpy as np
import shap
from sklearn.preprocessing import StandardScaler

class FeatureContributionAnalyzer:
    def __init__(self, model, X_train):
        self.model = model
        self.scaler = StandardScaler().fit(X_train)
        self.explainer = shap.TreeExplainer(model)
        
    def analyze_contributions(self, X):
        # Calculate SHAP values
        shap_values = self.explainer.shap_values(X)
        
        # Calculate absolute contributions
        abs_contributions = np.abs(shap_values).mean(axis=0)
        
        # Calculate relative importance
        total_impact = abs_contributions.sum()
        relative_importance = abs_contributions / total_impact
        
        # Generate contribution summary
        summary = pd.DataFrame({
            'feature': X.columns,
            'absolute_contribution': abs_contributions,
            'relative_importance': relative_importance
        }).sort_values('absolute_contribution', ascending=False)
        
        return summary, shap_values
    
    def get_top_contributors(self, X, threshold=0.8):
        summary, _ = self.analyze_contributions(X)
        cumsum = summary['relative_importance'].cumsum()
        return summary[cumsum <= threshold]
```

Slide 11: Model-Agnostic SHAP Implementation

This implementation provides a generalized approach to computing SHAP values for any machine learning model, not just tree-based ones, using sampling-based approximation methods.

```python
import numpy as np
from itertools import combinations

class ModelAgnosticSHAP:
    def __init__(self, model, background_data, n_samples=1000):
        self.model = model
        self.background = background_data
        self.n_samples = n_samples
        self.n_features = background_data.shape[1]
    
    def _sample_coalitions(self):
        coalition_matrix = np.random.binomial(
            n=1, p=0.5, 
            size=(self.n_samples, self.n_features)
        )
        return coalition_matrix
    
    def explain_instance(self, x):
        coalitions = self._sample_coalitions()
        shap_values = np.zeros(self.n_features)
        
        for coalition in coalitions:
            # Create mixed instance
            mixed_instance = np.where(
                coalition[:, np.newaxis], 
                x, 
                self.background
            )
            
            # Calculate marginal contribution
            pred_with = self.model.predict(mixed_instance)
            pred_without = self.model.predict(self.background)
            
            # Update SHAP values
            contribution = pred_with - pred_without
            shap_values += coalition * contribution
            
        return shap_values / self.n_samples

# Example usage
explainer = ModelAgnosticSHAP(model, X_background)
instance_explanation = explainer.explain_instance(X_test[0])
```

Slide 12: Performance Metrics for SHAP Explanations

This implementation introduces quantitative metrics to evaluate the quality and reliability of SHAP explanations, including consistency scores and explanation stability measures.

```python
import numpy as np
from sklearn.metrics import r2_score
from scipy.stats import spearmanr

class SHAPEvaluator:
    def __init__(self, model, explainer):
        self.model = model
        self.explainer = explainer
        
    def explanation_stability(self, X, n_bootstrap=100):
        stability_scores = []
        base_shap = self.explainer.shap_values(X)
        
        for _ in range(n_bootstrap):
            # Bootstrap sample
            idx = np.random.choice(len(X), len(X))
            boot_shap = self.explainer.shap_values(X[idx])
            
            # Calculate correlation between explanations
            corr, _ = spearmanr(
                base_shap.flatten(), 
                boot_shap.flatten()
            )
            stability_scores.append(corr)
            
        return {
            'mean_stability': np.mean(stability_scores),
            'std_stability': np.std(stability_scores)
        }
    
    def explanation_fidelity(self, X):
        shap_values = self.explainer.shap_values(X)
        base_pred = self.model.predict(X)
        shap_pred = shap_values.sum(axis=1) + self.explainer.expected_value
        
        return r2_score(base_pred, shap_pred)
```

Slide 13: Real-world Application: Medical Diagnosis

Implementation of an interpretable diagnostic model using SHAP values to explain medical predictions while maintaining privacy and regulatory compliance.

```python
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import shap

class MedicalDiagnosisExplainer:
    def __init__(self, model_params=None):
        self.model = RandomForestClassifier(**(model_params or {}))
        self.feature_names = None
        self.explainer = None
        
    def fit(self, X, y, feature_names=None):
        self.feature_names = feature_names or [f'feature_{i}' for i in range(X.shape[1])]
        self.model.fit(X, y)
        self.explainer = shap.TreeExplainer(self.model)
        
    def explain_diagnosis(self, patient_data):
        # Calculate SHAP values
        shap_values = self.explainer.shap_values(patient_data)
        
        # Create explanation dictionary
        explanation = {
            'prediction': self.model.predict_proba(patient_data)[0],
            'feature_importance': dict(zip(
                self.feature_names,
                np.abs(shap_values[0] if isinstance(shap_values, list) else shap_values)
            )),
            'key_factors': self._get_key_factors(shap_values, patient_data)
        }
        
        return explanation
    
    def _get_key_factors(self, shap_values, patient_data, top_k=5):
        values = shap_values[0] if isinstance(shap_values, list) else shap_values
        idx = np.argsort(np.abs(values))[-top_k:]
        
        return {
            self.feature_names[i]: {
                'value': patient_data[0][i],
                'impact': values[i]
            } for i in idx
        }
```

Slide 14: Additional Resources

*   "A Unified Approach to Interpreting Model Predictions" - [https://arxiv.org/abs/1705.07874](https://arxiv.org/abs/1705.07874)
*   "Fast Tree-structured SHAP Computation" - [https://arxiv.org/abs/2102.13035](https://arxiv.org/abs/2102.13035)
*   "Consistent Individualized Feature Attribution for Tree Ensembles" - [https://arxiv.org/abs/1802.03888](https://arxiv.org/abs/1802.03888)
*   "Explaining Machine Learning Models: A Non-Technical Guide to Interpretable AI" - Available at Google Scholar
*   "TreeSHAP: Fast Parallel Tree Ensembles" - Search on Google Scholar for recent publications
*   "Optimizing SHAP Computation for Large-Scale Applications" - Available through academic databases

