## 5 Reasons for Feature Selection in Machine Learning
Slide 1: Dimensionality Reduction and Computational Efficiency

Feature selection significantly reduces computational complexity and training time by eliminating irrelevant or redundant features. This process becomes crucial when dealing with high-dimensional datasets where the curse of dimensionality can severely impact model performance and resource utilization.

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.feature_selection import SelectKBest, f_classif
from time import time

# Generate synthetic dataset with redundant features
X, y = make_classification(n_samples=1000, n_features=100, 
                         n_informative=10, random_state=42)

# Measure training time with all features
t0 = time()
selected_features = SelectKBest(score_func=f_classif, k=10)
X_selected = selected_features.fit_transform(X, y)

print(f"Original features: {X.shape[1]}")
print(f"Selected features: {X_selected.shape[1]}")
print(f"Feature selection time: {time() - t0:.3f} seconds")
```

Slide 2: Variance Thresholding Implementation

Understanding feature variance helps identify constant or near-constant features that provide minimal discriminative power. This implementation demonstrates how to remove low-variance features using a custom threshold approach with numpy.

```python
import numpy as np

def variance_threshold_selector(X, threshold=0.01):
    # Calculate variance of each feature
    variances = np.var(X, axis=0)
    
    # Create boolean mask for features above threshold
    mask = variances > threshold
    
    # Return selected features and their indices
    return X[:, mask], np.where(mask)[0]

# Example usage
X = np.random.randn(100, 20)
X[:, 5] = 0.1  # Create low-variance feature

X_selected, selected_indices = variance_threshold_selector(X)
print(f"Original features: {X.shape[1]}")
print(f"Features after variance threshold: {X_selected.shape[1]}")
print(f"Removed feature indices: {np.where(~np.in1d(range(X.shape[1]), selected_indices))[0]}")
```

Slide 3: Correlation-Based Feature Selection

High correlation between features indicates redundancy in the dataset. This implementation uses correlation matrix analysis to identify and remove highly correlated features while retaining the most informative ones based on their correlation with the target variable.

```python
import pandas as pd
import numpy as np
from scipy.stats import spearmanr

def correlation_selector(X, y, threshold=0.7):
    # Calculate correlation matrix
    corr_matrix = pd.DataFrame(X).corr().abs()
    
    # Calculate correlation with target
    target_corr = np.array([abs(spearmanr(X[:, i], y)[0]) for i in range(X.shape[1])])
    
    # Find features to remove
    features_to_remove = set()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if corr_matrix.iloc[i, j] > threshold:
                # Remove feature with lower correlation to target
                if target_corr[i] < target_corr[j]:
                    features_to_remove.add(i)
                else:
                    features_to_remove.add(j)
    
    # Select features
    selected_features = [i for i in range(X.shape[1]) if i not in features_to_remove]
    return X[:, selected_features], selected_features

# Example usage
X, y = make_classification(n_samples=1000, n_features=20, 
                         n_informative=10, random_state=42)
X_selected, selected_features = correlation_selector(X, y)
print(f"Selected features: {selected_features}")
print(f"Reduced feature shape: {X_selected.shape}")
```

Slide 4: Recursive Feature Elimination

A sophisticated approach that iteratively constructs the model, ranks features by importance, and removes the least important ones. This implementation showcases RFE with cross-validation to determine the optimal number of features.

```python
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

def recursive_feature_elimination(X, y, step=1):
    # Initialize model
    model = LogisticRegression(random_state=42)
    
    # Initialize RFE
    rfe = RFE(estimator=model, n_features_to_select=1, step=step)
    
    # Fit RFE
    rfe.fit(X, y)
    
    # Get ranking and support mask
    ranking = rfe.ranking_
    support = rfe.support_
    
    # Calculate cross-validation scores for different feature counts
    scores = []
    features_counts = range(1, X.shape[1] + 1)
    
    for n_features in features_counts:
        rfe = RFE(estimator=model, n_features_to_select=n_features, step=1)
        score = cross_val_score(estimator=model, X=X, y=y, cv=5)
        scores.append(np.mean(score))
    
    # Get optimal number of features
    optimal_n_features = features_counts[np.argmax(scores)]
    
    return optimal_n_features, ranking, support, scores

# Example usage
X, y = make_classification(n_samples=1000, n_features=20, 
                         n_informative=10, random_state=42)
opt_features, ranking, support, scores = recursive_feature_elimination(X, y)
print(f"Optimal number of features: {opt_features}")
print(f"Feature ranking: {ranking}")
```

Slide 5: Information Gain and Mutual Information

Information theory metrics provide powerful insights into feature relevance by measuring the mutual information between features and target variables. This implementation demonstrates both entropy-based and mutual information-based feature selection.

```python
from sklearn.feature_selection import mutual_info_classif
import numpy as np
from scipy.stats import entropy

def information_based_selection(X, y, method='mutual_info', k=5):
    if method == 'mutual_info':
        # Calculate mutual information scores
        mi_scores = mutual_info_classif(X, y)
    else:
        # Calculate information gain
        mi_scores = np.array([
            entropy(y) - conditional_entropy(X[:, i], y)
            for i in range(X.shape[1])
        ])
    
    # Select top k features
    top_features = np.argsort(mi_scores)[-k:]
    
    return X[:, top_features], top_features, mi_scores

def conditional_entropy(x, y):
    # Calculate conditional entropy H(Y|X)
    y_unique = np.unique(y)
    x_unique = np.unique(x)
    
    cond_entropy = 0
    for x_val in x_unique:
        p_x = np.mean(x == x_val)
        y_given_x = y[x == x_val]
        if len(y_given_x) > 0:
            cond_entropy += p_x * entropy(y_given_x)
    
    return cond_entropy

# Example usage
X, y = make_classification(n_samples=1000, n_features=20,
                         n_informative=10, random_state=42)
X_selected, selected_features, scores = information_based_selection(X, y)
print(f"Selected features: {selected_features}")
print(f"Feature scores: {scores}")
```

Slide 6: Lasso Regularization for Feature Selection

Lasso (L1) regularization inherently performs feature selection by driving coefficients of less important features to exactly zero. This implementation demonstrates how to use Lasso regression to identify and select the most relevant features while handling multicollinearity.

```python
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
import numpy as np

def lasso_feature_selection(X, y, cv=5):
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Fit LassoCV
    lasso = LassoCV(cv=cv, random_state=42, max_iter=10000)
    lasso.fit(X_scaled, y)
    
    # Get selected features
    selected_features = np.where(lasso.coef_ != 0)[0]
    feature_importance = np.abs(lasso.coef_)
    
    return X[:, selected_features], selected_features, feature_importance

# Example usage with synthetic data
X, y = make_classification(n_samples=1000, n_features=20,
                         n_informative=10, random_state=42)
X_selected, selected_features, importance = lasso_feature_selection(X, y)

print(f"Selected features: {selected_features}")
print(f"Feature importance: {importance}")
print(f"Optimal alpha: {lasso.alpha_:.4f}")
```

Slide 7: Statistical Testing for Feature Selection

Statistical tests provide a rigorous framework for evaluating feature significance. This implementation uses multiple statistical methods including Chi-square, ANOVA, and Mann-Whitney U tests for different types of features.

```python
from scipy.stats import chi2_contingency, f_oneway, mannwhitneyu
import numpy as np
import pandas as pd

def statistical_feature_selection(X, y, method='anova', alpha=0.05):
    p_values = []
    selected_features = []
    
    for i in range(X.shape[1]):
        if method == 'anova':
            # ANOVA test for numerical features
            _, p_value = f_oneway(*[X[y == label, i] 
                                  for label in np.unique(y)])
        elif method == 'chi2':
            # Chi-square test for categorical features
            contingency = pd.crosstab(X[:, i], y)
            _, p_value, _, _ = chi2_contingency(contingency)
        elif method == 'mannwhitney':
            # Mann-Whitney U test for binary classification
            _, p_value = mannwhitneyu(X[y == 0, i], X[y == 1, i])
            
        p_values.append(p_value)
        if p_value < alpha:
            selected_features.append(i)
    
    return X[:, selected_features], selected_features, p_values

# Example usage
X, y = make_classification(n_samples=1000, n_features=20,
                         n_informative=10, random_state=42)
X_selected, selected_feats, p_vals = statistical_feature_selection(X, y)

print(f"Selected features: {selected_feats}")
print(f"P-values: {np.round(p_vals, 4)}")
```

Slide 8: Random Forest Feature Importance

Random Forests provide built-in feature importance measures through mean decrease in impurity or permutation importance. This implementation demonstrates both approaches and their differences in feature selection.

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
import numpy as np

def random_forest_feature_selection(X, y, importance_type='mdi', 
                                  threshold=0.05):
    # Initialize and fit Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    
    if importance_type == 'mdi':
        # Mean Decrease Impurity importance
        importance = rf.feature_importances_
    else:
        # Permutation importance
        result = permutation_importance(rf, X, y, n_repeats=10,
                                      random_state=42)
        importance = result.importances_mean
    
    # Select features above threshold
    selected_features = np.where(importance > threshold)[0]
    
    return (X[:, selected_features], selected_features, 
            importance[selected_features])

# Example usage
X, y = make_classification(n_samples=1000, n_features=20,
                         n_informative=10, random_state=42)
X_selected, selected_feats, importance = random_forest_feature_selection(X, y)

print(f"Selected features: {selected_feats}")
print(f"Feature importance: {np.round(importance, 4)}")
```

Slide 9: Recursive Feature Addition

Instead of eliminating features, this approach starts with an empty feature set and recursively adds the most important features until a stopping criterion is met. This implementation includes cross-validation for optimal feature subset selection.

```python
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
import numpy as np

def recursive_feature_addition(X, y, estimator=None, cv=5):
    if estimator is None:
        estimator = LogisticRegression(random_state=42)
    
    n_features = X.shape[1]
    selected_features = []
    current_score = 0
    
    # Score tracking
    scores_history = []
    
    while len(selected_features) < n_features:
        best_score = 0
        best_feature = None
        
        # Try adding each feature
        for feature in range(n_features):
            if feature not in selected_features:
                features_to_try = selected_features + [feature]
                X_subset = X[:, features_to_try]
                
                # Calculate cross-validation score
                scores = cross_val_score(estimator, X_subset, y, cv=cv)
                score = np.mean(scores)
                
                if score > best_score:
                    best_score = score
                    best_feature = feature
        
        # Add best feature if it improves score
        if best_score > current_score:
            selected_features.append(best_feature)
            current_score = best_score
            scores_history.append(current_score)
        else:
            break
    
    return (X[:, selected_features], selected_features, 
            scores_history)

# Example usage
X, y = make_classification(n_samples=1000, n_features=20,
                         n_informative=10, random_state=42)
X_selected, selected_feats, scores = recursive_feature_addition(X, y)

print(f"Selected features: {selected_feats}")
print(f"Score history: {np.round(scores, 4)}")
```

Slide 10: Real-world Example - Credit Card Fraud Detection

This comprehensive example demonstrates feature selection in a credit card fraud detection scenario, implementing multiple selection methods and comparing their effectiveness in identifying fraudulent transactions.

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

def credit_fraud_feature_selection(X, y):
    # Initialize results dictionary
    results = {}
    
    # 1. Variance Threshold
    def variance_selector(X, threshold=0.01):
        variances = np.var(X, axis=0)
        return variances > threshold
    
    var_mask = variance_selector(X)
    X_var = X[:, var_mask]
    results['variance'] = {'mask': var_mask, 'X': X_var}
    
    # 2. Correlation Analysis
    corr_matrix = np.corrcoef(X.T)
    high_corr_features = set()
    for i in range(len(corr_matrix)):
        for j in range(i):
            if abs(corr_matrix[i, j]) > 0.95:
                high_corr_features.add(i)
    
    corr_mask = [i not in high_corr_features for i in range(X.shape[1])]
    X_corr = X[:, corr_mask]
    results['correlation'] = {'mask': corr_mask, 'X': X_corr}
    
    # 3. Random Forest Importance
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    importance_mask = rf.feature_importances_ > np.mean(rf.feature_importances_)
    X_rf = X[:, importance_mask]
    results['rf_importance'] = {'mask': importance_mask, 'X': X_rf}
    
    return results

# Example usage with synthetic fraud data
np.random.seed(42)
n_samples = 10000
n_features = 30

# Generate synthetic credit card transaction data
X = np.random.randn(n_samples, n_features)
# Add some correlated features
X[:, 5] = X[:, 0] * 0.9 + np.random.randn(n_samples) * 0.1
X[:, 6] = X[:, 1] * 0.95 + np.random.randn(n_samples) * 0.05
# Generate fraud labels (1% fraud rate)
y = np.random.choice([0, 1], size=n_samples, p=[0.99, 0.01])

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Apply feature selection
results = credit_fraud_feature_selection(X_train, y_train)

# Evaluate each method
for method, data in results.items():
    clf = RandomForestClassifier(random_state=42)
    clf.fit(data['X'], y_train)
    y_pred = clf.predict(X_test[:, data['mask']])
    print(f"\nResults for {method}:")
    print(classification_report(y_test, y_pred))
```

Slide 11: Real-world Example - Gene Expression Analysis

This implementation showcases feature selection in genomics, where high-dimensional gene expression data requires efficient feature selection to identify relevant genes for disease classification.

```python
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFdr, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

def gene_expression_feature_selection(X, y, method='combined'):
    """
    Feature selection for gene expression data using multiple methods
    """
    results = {}
    
    # 1. Statistical Testing with FDR control
    fdr_selector = SelectFdr(f_classif, alpha=0.05)
    X_fdr = fdr_selector.fit_transform(X, y)
    results['fdr'] = {
        'mask': fdr_selector.get_support(),
        'X': X_fdr,
        'pvalues': fdr_selector.pvalues_
    }
    
    # 2. Stability Selection
    def stability_selection(X, y, n_iterations=50):
        feature_counts = np.zeros(X.shape[1])
        for _ in range(n_iterations):
            # Bootstrap sample
            indices = np.random.choice(len(X), size=len(X), replace=True)
            X_boot, y_boot = X[indices], y[indices]
            
            # Random Forest on bootstrap sample
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(X_boot, y_boot)
            
            # Count important features
            important_features = rf.feature_importances_ > np.mean(rf.feature_importances_)
            feature_counts += important_features
        
        return feature_counts / n_iterations > 0.5
    
    stability_mask = stability_selection(X, y)
    X_stability = X[:, stability_mask]
    results['stability'] = {'mask': stability_mask, 'X': X_stability}
    
    # 3. Combined approach
    if method == 'combined':
        combined_mask = results['fdr']['mask'] & results['stability']['mask']
        X_combined = X[:, combined_mask]
        results['combined'] = {'mask': combined_mask, 'X': X_combined}
    
    return results

# Example usage with synthetic gene expression data
np.random.seed(42)
n_samples = 200
n_genes = 1000

# Generate synthetic gene expression data
X = np.random.randn(n_samples, n_genes)
# Add some informative genes
informative_genes = np.random.choice(n_genes, 50, replace=False)
X[:100, informative_genes] += 2
y = np.array([1] * 100 + [0] * 100)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Apply feature selection
results = gene_expression_feature_selection(X_train, y_train)

# Evaluate each method
for method, data in results.items():
    clf = RandomForestClassifier(random_state=42)
    clf.fit(data['X'], y_train)
    y_pred_proba = clf.predict_proba(X_test[:, data['mask']])[:, 1]
    auc = roc_auc_score(y_test, y_pred_proba)
    print(f"\n{method} AUC-ROC: {auc:.4f}")
    print(f"Selected features: {sum(data['mask'])}")
```

Slide 12: Feature Selection Performance Metrics

This implementation provides a comprehensive suite of metrics to evaluate feature selection methods, including stability, relevance, and redundancy measures across different selection techniques.

```python
import numpy as np
from sklearn.metrics import mutual_info_score
from sklearn.model_selection import cross_val_score

def evaluate_feature_selection(X, y, selected_features, 
                             original_features=None):
    """
    Comprehensive evaluation of feature selection results
    """
    metrics = {}
    
    # 1. Stability (Jaccard similarity between feature subsets)
    def calculate_stability(feature_sets):
        n_sets = len(feature_sets)
        stability = 0
        for i in range(n_sets):
            for j in range(i + 1, n_sets):
                intersection = len(set(feature_sets[i]) & 
                                 set(feature_sets[j]))
                union = len(set(feature_sets[i]) | 
                           set(feature_sets[j]))
                stability += intersection / union
        return 2 * stability / (n_sets * (n_sets - 1))
    
    # 2. Relevance (Mutual Information with target)
    def calculate_relevance(X, y, features):
        relevance = np.mean([
            mutual_info_score(X[:, f], y) for f in features
        ])
        return relevance
    
    # 3. Redundancy (Average mutual information between features)
    def calculate_redundancy(X, features):
        redundancy = 0
        n_features = len(features)
        if n_features < 2:
            return 0
        
        for i in range(n_features):
            for j in range(i + 1, n_features):
                redundancy += mutual_info_score(
                    X[:, features[i]], X[:, features[j]])
        return 2 * redundancy / (n_features * (n_features - 1))
    
    # Calculate metrics
    metrics['n_selected'] = len(selected_features)
    metrics['relevance'] = calculate_relevance(X, y, selected_features)
    metrics['redundancy'] = calculate_redundancy(X, selected_features)
    
    # Calculate prediction performance
    clf = RandomForestClassifier(random_state=42)
    scores = cross_val_score(clf, X[:, selected_features], y, cv=5)
    metrics['cv_score_mean'] = np.mean(scores)
    metrics['cv_score_std'] = np.std(scores)
    
    # Compare with original features if provided
    if original_features is not None:
        original_scores = cross_val_score(
            clf, X[:, original_features], y, cv=5)
        metrics['original_cv_score'] = np.mean(original_scores)
        metrics['feature_reduction'] = (
            1 - len(selected_features) / len(original_features))
    
    return metrics

# Example usage
X, y = make_classification(n_samples=1000, n_features=20,
                         n_informative=10, random_state=42)

# Apply different feature selection methods
methods = {
    'variance': variance_threshold_selector,
    'correlation': correlation_selector,
    'mutual_info': information_based_selection
}

results = {}
for name, method in methods.items():
    _, selected_features, _ = method(X, y)
    results[name] = evaluate_feature_selection(
        X, y, selected_features, range(X.shape[1]))
    
    print(f"\nMetrics for {name}:")
    for metric, value in results[name].items():
        print(f"{metric}: {value:.4f}")
```

Slide 13: Ensemble Feature Selection

This implementation combines multiple feature selection methods through a voting mechanism to create a more robust and reliable feature selection process, reducing the bias of individual methods.

```python
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif, f_classif
from sklearn.linear_model import LassoCV

class EnsembleFeatureSelector:
    def __init__(self, methods=None, voting_threshold=0.5):
        self.methods = methods if methods else {
            'rf': self._rf_importance,
            'mutual_info': self._mutual_info,
            'lasso': self._lasso_selection,
            'f_score': self._f_score
        }
        self.voting_threshold = voting_threshold
        self.feature_votes = None
        
    def _rf_importance(self, X, y):
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X, y)
        return rf.feature_importances_ > np.mean(rf.feature_importances_)
    
    def _mutual_info(self, X, y):
        mi_scores = mutual_info_classif(X, y)
        return mi_scores > np.mean(mi_scores)
    
    def _lasso_selection(self, X, y):
        lasso = LassoCV(cv=5, random_state=42)
        lasso.fit(X, y)
        return np.abs(lasso.coef_) > 0
    
    def _f_score(self, X, y):
        f_scores, _ = f_classif(X, y)
        return f_scores > np.mean(f_scores)
    
    def fit(self, X, y):
        self.feature_votes = np.zeros(X.shape[1])
        
        # Apply each method and collect votes
        for method in self.methods.values():
            self.feature_votes += method(X, y)
        
        # Normalize votes
        self.feature_votes /= len(self.methods)
        
        return self
    
    def transform(self, X):
        selected = self.feature_votes >= self.voting_threshold
        return X[:, selected]
    
    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)

# Example usage
X, y = make_classification(n_samples=1000, n_features=20,
                         n_informative=10, random_state=42)

# Apply ensemble feature selection
selector = EnsembleFeatureSelector()
X_selected = selector.fit_transform(X, y)

print(f"Original features: {X.shape[1]}")
print(f"Selected features: {X_selected.shape[1]}")
print(f"Feature votes:\n{selector.feature_votes}")
```

Slide 14: Feature Selection Visualization

This implementation provides comprehensive visualization tools for analyzing feature selection results, including feature importance distributions, stability plots, and performance comparisons.

```python
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc

def visualize_feature_selection(X, y, results_dict, 
                              output_format='png'):
    """
    Comprehensive visualization of feature selection results
    """
    plt.style.use('seaborn')
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Feature Importance Distribution
    plt.subplot(2, 2, 1)
    for method, result in results_dict.items():
        if 'importance' in result:
            sns.kdeplot(result['importance'], label=method)
    plt.title('Feature Importance Distribution')
    plt.xlabel('Importance Score')
    plt.ylabel('Density')
    plt.legend()
    
    # 2. Selected Features Heatmap
    plt.subplot(2, 2, 2)
    selection_matrix = np.array([result['mask'] 
                               for result in results_dict.values()])
    sns.heatmap(selection_matrix, 
                yticklabels=list(results_dict.keys()),
                cmap='YlOrRd')
    plt.title('Feature Selection Comparison')
    plt.xlabel('Feature Index')
    
    # 3. Performance Comparison
    plt.subplot(2, 2, 3)
    performance_data = []
    for method, result in results_dict.items():
        if 'cv_scores' in result:
            performance_data.append({
                'method': method,
                'score': result['cv_scores'].mean(),
                'std': result['cv_scores'].std()
            })
    
    performance_df = pd.DataFrame(performance_data)
    sns.barplot(data=performance_df, x='method', y='score',
                yerr=performance_df['std'])
    plt.title('Performance Comparison')
    plt.ylabel('Cross-validation Score')
    
    # 4. ROC Curves
    plt.subplot(2, 2, 4)
    for method, result in results_dict.items():
        if 'predictions' in result:
            fpr, tpr, _ = roc_curve(y, result['predictions'])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{method} (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend(loc="lower right")
    
    plt.tight_layout()
    return fig

# Example usage
X, y = make_classification(n_samples=1000, n_features=20,
                         n_informative=10, random_state=42)

# Apply different feature selection methods and collect results
results = {}
methods = {
    'RF': RandomForestClassifier(n_estimators=100),
    'Lasso': LassoCV(),
    'MutualInfo': SelectKBest(mutual_info_classif, k=10)
}

for name, method in methods.items():
    if hasattr(method, 'fit_transform'):
        X_selected = method.fit_transform(X, y)
    else:
        method.fit(X, y)
        X_selected = method.transform(X)
    
    results[name] = {
        'mask': np.where(X_selected.any(axis=0))[0],
        'importance': method.feature_importances_ 
                     if hasattr(method, 'feature_importances_') 
                     else None,
        'cv_scores': cross_val_score(
            RandomForestClassifier(), X_selected, y, cv=5),
        'predictions': cross_val_predict(
            RandomForestClassifier(), X_selected, y, cv=5,
            method='predict_proba')[:, 1]
    }

# Generate visualization
fig = visualize_feature_selection(X, y, results)
plt.show()
```

Slide 15: Additional Resources

*   "An Introduction to Variable and Feature Selection" - [https://www.jmlr.org/papers/volume3/guyon03a/guyon03a.pdf](https://www.jmlr.org/papers/volume3/guyon03a/guyon03a.pdf)
*   "Feature Selection for High-Dimensional Data: A Fast Correlation-Based Filter Solution" - Search on Google Scholar
*   "Stability Selection" by Nicolai Meinshausen and Peter Bühlmann - [https://arxiv.org/abs/0809.2932](https://arxiv.org/abs/0809.2932)
*   "Feature Selection with Ensemble Methods" - Search on Google Scholar for recent publications
*   "A Review of Feature Selection Methods for Machine Learning" - [https://arxiv.org/abs/1905.13525](https://arxiv.org/abs/1905.13525)
*   "Deep Feature Selection: Theory and Application to Identifying Compounds" - Search on Google Scholar

