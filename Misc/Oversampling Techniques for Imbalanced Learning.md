## Oversampling Techniques for Imbalanced Learning
Slide 1: Understanding Data Imbalance

Data imbalance occurs when class distributions in a dataset are significantly skewed. This fundamental challenge in machine learning can severely impact model performance, as algorithms tend to be biased towards the majority class, leading to poor predictive accuracy for minority classes.

```python
# Example of imbalanced dataset creation and visualization
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification

# Generate imbalanced dataset
X, y = make_classification(
    n_samples=1000,
    n_classes=2,
    weights=[0.9, 0.1],  # 90% majority, 10% minority
    random_state=42
)

# Display class distribution
unique, counts = np.unique(y, return_counts=True)
plt.bar(['Majority', 'Minority'], counts)
plt.title('Class Distribution')
plt.ylabel('Number of Samples')
plt.show()

print(f"Class distribution:\nMajority: {counts[0]}\nMinority: {counts[1]}")
```

Slide 2: Random Oversampling Implementation

Random oversampling duplicates minority class instances randomly until reaching balanced distribution. While simple, this approach carries the risk of overfitting as it creates exact copies of existing samples without introducing new information.

```python
import numpy as np
from sklearn.model_selection import train_test_split

def random_oversample(X, y):
    # Get indices of each class
    majority_indices = np.where(y == 0)[0]
    minority_indices = np.where(y == 1)[0]
    
    # Calculate number of samples to generate
    n_samples = len(majority_indices) - len(minority_indices)
    
    # Randomly sample with replacement from minority class
    minority_indices_resampled = np.random.choice(
        minority_indices,
        size=n_samples,
        replace=True
    )
    
    # Combine indices and sort them
    all_indices = np.concatenate([
        majority_indices,
        minority_indices,
        minority_indices_resampled
    ])
    
    return X[all_indices], y[all_indices]

# Example usage
X_resampled, y_resampled = random_oversample(X, y)
print(f"Original distribution: {np.bincount(y)}")
print(f"Resampled distribution: {np.bincount(y_resampled)}")
```

Slide 3: SMOTE Algorithm Core Concepts

Synthetic Minority Over-sampling Technique (SMOTE) generates synthetic samples by interpolating between minority class instances. It uses k-nearest neighbors to create new samples along the feature space lines connecting minority samples.

```python
def calculate_smote_formula():
    formula = """
    # SMOTE synthetic sample generation formula
    $$x_{new} = x_i + \lambda \cdot (x_{knn} - x_i)$$
    
    Where:
    $$x_i$$ is the selected minority instance
    $$x_{knn}$$ is one of its k-nearest neighbors
    $$\lambda$$ is a random number between 0 and 1
    """
    return formula

print(calculate_smote_formula())
```

Slide 4: SMOTE Implementation from Scratch

```python
import numpy as np
from sklearn.neighbors import NearestNeighbors

def smote_oversample(X_minority, n_samples, k_neighbors=5):
    # Initialize nearest neighbors
    neigh = NearestNeighbors(n_neighbors=k_neighbors + 1)
    neigh.fit(X_minority)
    
    # Find k-nearest neighbors for all minority samples
    distances, indices = neigh.kneighbors(X_minority)
    
    # Generate synthetic samples
    synthetic_samples = []
    for i in range(len(X_minority)):
        # Get k-nearest neighbors (excluding the sample itself)
        nn_indices = indices[i, 1:]
        
        # Generate n_samples/len(X_minority) samples for each minority instance
        n_synthetic = int(n_samples / len(X_minority))
        
        for _ in range(n_synthetic):
            # Select random neighbor
            nn_idx = np.random.choice(nn_indices)
            
            # Calculate synthetic sample
            diff = X_minority[nn_idx] - X_minority[i]
            synthetic = X_minority[i] + np.random.random() * diff
            synthetic_samples.append(synthetic)
    
    return np.vstack([X_minority, np.array(synthetic_samples)])

# Example usage
X_minority = X[y == 1]
n_samples_needed = sum(y == 0) - sum(y == 1)
X_synthetic = smote_oversample(X_minority, n_samples_needed)
print(f"Original minority samples: {len(X_minority)}")
print(f"Synthetic samples generated: {len(X_synthetic) - len(X_minority)}")
```

Slide 5: ADASYN Algorithm Principles

ADASYN (Adaptive Synthetic) improves upon SMOTE by focusing on harder-to-learn examples. It uses a density distribution to determine the number of synthetic samples needed for each minority instance, giving more weight to instances that are harder to learn.

```python
def calculate_adasyn_formula():
    formula = """
    # ADASYN density distribution formula
    $$r_i = \frac{\Delta_i}{Z}$$
    
    Where:
    $$\Delta_i$$ is the number of majority samples in k-neighbors
    $$Z$$ is a normalization constant
    
    # Number of synthetic samples formula
    $$g_i = r_i \cdot G$$
    
    Where:
    $$G$$ is the total number of synthetic samples needed
    """
    return formula

print(calculate_adasyn_formula())
```

Slide 6: ADASYN Implementation from Scratch

ADASYN generates synthetic samples by focusing on minority instances that are harder to learn. This implementation includes the density distribution calculation and adaptive synthetic sample generation based on the difficulty level of each minority instance.

```python
def adasyn_oversample(X, y, k_neighbors=5, beta=1.0):
    # Identify minority and majority classes
    minority_class = 1
    X_minority = X[y == minority_class]
    
    # Calculate number of samples to generate
    minority_count = sum(y == minority_class)
    majority_count = sum(y == 1 - minority_class)
    G = (majority_count - minority_count) * beta
    
    # Initialize nearest neighbors
    neigh = NearestNeighbors(n_neighbors=k_neighbors + 1)
    neigh.fit(X)
    
    # Calculate r_i (density ratio) for each minority instance
    r_i = []
    for x_i in X_minority:
        indices = neigh.kneighbors([x_i], return_distance=False)[0][1:]
        delta_i = sum(y[indices] != minority_class) / k_neighbors
        r_i.append(delta_i)
    
    # Normalize r_i
    if sum(r_i) == 0:
        r_i = np.ones(len(r_i)) / len(r_i)
    else:
        r_i = r_i / sum(r_i)
    
    # Calculate g_i (number of synthetic samples for each minority instance)
    g_i = np.round(r_i * G).astype(int)
    
    # Generate synthetic samples
    synthetic_samples = []
    for i, x_i in enumerate(X_minority):
        if g_i[i] == 0:
            continue
            
        # Find k-nearest minority neighbors
        minority_indices = np.where(y == minority_class)[0]
        neigh.fit(X[minority_indices])
        neighbors = neigh.kneighbors([x_i], n_neighbors=k_neighbors+1, return_distance=False)[0][1:]
        
        # Generate g_i synthetic samples
        for _ in range(g_i[i]):
            nn_idx = np.random.choice(neighbors)
            x_nn = X[minority_indices[nn_idx]]
            
            # Generate synthetic sample
            lambda_val = np.random.random()
            synthetic = x_i + lambda_val * (x_nn - x_i)
            synthetic_samples.append(synthetic)
    
    return np.vstack([X_minority, np.array(synthetic_samples)])

# Example usage
X_resampled = adasyn_oversample(X, y)
print(f"Original minority samples: {sum(y == 1)}")
print(f"Synthetic samples generated: {len(X_resampled) - sum(y == 1)}")
```

Slide 7: Real-world Application - Credit Card Fraud Detection

Credit card fraud detection presents a classic imbalanced learning problem where fraudulent transactions represent a tiny fraction of all transactions. This implementation demonstrates the application of oversampling techniques in a real-world scenario.

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pandas as pd

# Load and preprocess credit card data
def prepare_credit_card_data():
    # Synthetic credit card dataset for demonstration
    np.random.seed(42)
    n_samples = 10000
    
    # Generate legitimate transactions (99.7%)
    X_legitimate = np.random.normal(loc=0, scale=1, size=(int(n_samples * 0.997), 2))
    
    # Generate fraudulent transactions (0.3%)
    X_fraudulent = np.random.normal(loc=2, scale=1, size=(int(n_samples * 0.003), 2))
    
    X = np.vstack([X_legitimate, X_fraudulent])
    y = np.hstack([np.zeros(len(X_legitimate)), np.ones(len(X_fraudulent))])
    
    return X, y

# Prepare data
X, y = prepare_credit_card_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train models with different sampling techniques
def evaluate_model(X_train_resampled, y_train_resampled, X_test, y_test, method_name):
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train_resampled, y_train_resampled)
    y_pred = clf.predict(X_test)
    
    print(f"\nResults for {method_name}:")
    print(classification_report(y_test, y_pred))

# Original (imbalanced) data
evaluate_model(X_train, y_train, X_test, y_test, "Original Data")

# SMOTE
X_train_smote = smote_oversample(X_train[y_train == 1], sum(y_train == 0) - sum(y_train == 1))
y_train_smote = np.hstack([np.ones(len(X_train_smote))])
evaluate_model(X_train_smote, y_train_smote, X_test, y_test, "SMOTE")

# ADASYN
X_train_adasyn = adasyn_oversample(X_train, y_train)
y_train_adasyn = np.hstack([np.ones(len(X_train_adasyn))])
evaluate_model(X_train_adasyn, y_train_adasyn, X_test, y_test, "ADASYN")
```

Slide 8: Performance Visualization and Analysis

This slide focuses on visualizing and comparing the performance metrics of different oversampling techniques using comprehensive evaluation metrics and intuitive visualizations.

```python
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve

def plot_performance_comparison(models_dict, X_test, y_test):
    plt.figure(figsize=(15, 5))
    
    # ROC Curve
    plt.subplot(1, 2, 1)
    for name, model in models_dict.items():
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend()
    
    # Precision-Recall Curve
    plt.subplot(1, 2, 2)
    for name, model in models_dict.items():
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        pr_auc = auc(recall, precision)
        
        plt.plot(recall, precision, label=f'{name} (AUC = {pr_auc:.2f})')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# Create models dictionary
models_dict = {
    'Original': clf_original,
    'SMOTE': clf_smote,
    'ADASYN': clf_adasyn
}

plot_performance_comparison(models_dict, X_test, y_test)
```

Slide 9: Borderline-SMOTE Enhancement

Borderline-SMOTE enhances traditional SMOTE by focusing on minority instances near the decision boundary. This approach identifies borderline samples using the ratio of majority neighbors to total neighbors, creating synthetic samples only for these critical instances.

```python
def borderline_smote(X, y, k_neighbors=5, m_neighbors=10):
    minority_class = 1
    X_minority = X[y == minority_class]
    X_majority = X[y != minority_class]
    
    # Initialize nearest neighbors for both minority and majority
    neigh_m = NearestNeighbors(n_neighbors=m_neighbors)
    neigh_m.fit(X_majority)
    
    # Find borderline samples
    borderline_samples = []
    for x_i in X_minority:
        # Find m nearest majority neighbors
        distances, _ = neigh_m.kneighbors([x_i])
        majority_ratio = len([d for d in distances[0] if d <= np.mean(distances)]) / m_neighbors
        
        # Classify as DANGER if ratio is between 0.3 and 0.7
        if 0.3 <= majority_ratio <= 0.7:
            borderline_samples.append(x_i)
    
    borderline_samples = np.array(borderline_samples)
    
    # Apply SMOTE only to borderline samples
    if len(borderline_samples) > 0:
        synthetic_samples = smote_oversample(
            borderline_samples, 
            n_samples=len(X_majority) - len(X_minority),
            k_neighbors=min(k_neighbors, len(borderline_samples)-1)
        )
        return synthetic_samples
    
    return X_minority

# Example usage
X_borderline = borderline_smote(X, y)
print(f"Original minority samples: {sum(y == 1)}")
print(f"Borderline-SMOTE samples: {len(X_borderline)}")
```

Slide 10: Ensemble Oversampling Strategy

Combining multiple oversampling techniques can provide more robust synthetic samples. This implementation creates an ensemble approach that leverages the strengths of different oversampling methods while mitigating their individual weaknesses.

```python
def ensemble_oversample(X, y, methods=['random', 'smote', 'adasyn'], weights=[0.3, 0.4, 0.3]):
    assert len(methods) == len(weights) and sum(weights) == 1
    
    synthetic_samples = []
    minority_class = 1
    n_samples_needed = sum(y == 0) - sum(y == 1)
    
    for method, weight in zip(methods, weights):
        n_method_samples = int(n_samples_needed * weight)
        
        if method == 'random':
            X_synthetic = random_oversample(X[y == minority_class], 
                                         n_samples=n_method_samples)
        elif method == 'smote':
            X_synthetic = smote_oversample(X[y == minority_class], 
                                         n_samples=n_method_samples)
        elif method == 'adasyn':
            X_synthetic = adasyn_oversample(X, y, 
                                          beta=n_method_samples/sum(y == minority_class))
            
        synthetic_samples.append(X_synthetic)
    
    # Combine all synthetic samples
    X_combined = np.vstack(synthetic_samples)
    
    return X_combined

# Example usage with evaluation
X_ensemble = ensemble_oversample(X, y)
y_ensemble = np.ones(len(X_ensemble))

# Train and evaluate
clf_ensemble = RandomForestClassifier(random_state=42)
clf_ensemble.fit(np.vstack([X[y == 0], X_ensemble]), 
                np.hstack([np.zeros(sum(y == 0)), y_ensemble]))

print("Ensemble Oversampling Results:")
y_pred = clf_ensemble.predict(X_test)
print(classification_report(y_test, y_pred))
```

Slide 11: Oversampling Validation Strategy

Proper validation is crucial when working with oversampled data to prevent data leakage. This implementation demonstrates the correct way to validate models trained on oversampled data using stratified cross-validation.

```python
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

def validate_oversampling(X, y, oversample_func, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        # Split data
        X_train_fold = X[train_idx]
        y_train_fold = y[train_idx]
        X_val_fold = X[val_idx]
        y_val_fold = y[val_idx]
        
        # Apply oversampling only to training data
        X_train_resampled = oversample_func(X_train_fold, y_train_fold)
        y_train_resampled = np.ones(len(X_train_resampled))
        
        # Train model
        clf = RandomForestClassifier(random_state=42)
        clf.fit(np.vstack([X_train_fold[y_train_fold == 0], X_train_resampled]),
               np.hstack([np.zeros(sum(y_train_fold == 0)), y_train_resampled]))
        
        # Evaluate
        y_pred = clf.predict(X_val_fold)
        score = f1_score(y_val_fold, y_pred)
        scores.append(score)
        
        print(f"Fold {fold+1} F1-Score: {score:.3f}")
    
    print(f"\nMean F1-Score: {np.mean(scores):.3f} (+/- {np.std(scores):.3f})")
    return scores

# Example usage for different methods
methods = {
    'SMOTE': lambda X, y: smote_oversample(X[y == 1], sum(y == 0) - sum(y == 1)),
    'ADASYN': lambda X, y: adasyn_oversample(X, y),
    'Borderline-SMOTE': lambda X, y: borderline_smote(X, y),
    'Ensemble': lambda X, y: ensemble_oversample(X, y)
}

for name, method in methods.items():
    print(f"\nValidating {name}:")
    validate_oversampling(X, y, method)
```

Slide 12: Handling Mixed Data Types in Oversampling

Oversampling becomes more complex when dealing with datasets containing both numerical and categorical features. This implementation introduces a sophisticated approach to handle mixed data types during synthetic sample generation.

```python
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def mixed_data_oversample(X, y, categorical_features):
    # Separate numerical and categorical features
    X_num = X[:, ~np.isin(np.arange(X.shape[1]), categorical_features)]
    X_cat = X[:, categorical_features]
    
    # Initialize label encoders for categorical features
    label_encoders = [LabelEncoder() for _ in range(len(categorical_features))]
    X_cat_encoded = np.zeros_like(X_cat)
    
    # Encode categorical features
    for i, le in enumerate(label_encoders):
        X_cat_encoded[:, i] = le.fit_transform(X_cat[:, i])
    
    def generate_synthetic_sample(x1, x2, lambda_val):
        # Generate numerical features
        num_features = x1[~np.isin(np.arange(len(x1)), categorical_features)]
        num_neighbor = x2[~np.isin(np.arange(len(x2)), categorical_features)]
        synthetic_num = num_features + lambda_val * (num_neighbor - num_features)
        
        # Generate categorical features
        synthetic_cat = []
        for i in range(len(categorical_features)):
            # Randomly select category from either parent
            if np.random.random() > 0.5:
                synthetic_cat.append(x1[categorical_features[i]])
            else:
                synthetic_cat.append(x2[categorical_features[i]])
        
        # Combine features
        synthetic = np.zeros(len(x1))
        synthetic[~np.isin(np.arange(len(x1)), categorical_features)] = synthetic_num
        synthetic[categorical_features] = synthetic_cat
        
        return synthetic
    
    def mixed_smote(X_minority, n_samples, k_neighbors=5):
        synthetic_samples = []
        neigh = NearestNeighbors(n_neighbors=k_neighbors + 1)
        neigh.fit(X_cat_encoded[y == 1])
        
        for i in range(len(X_minority)):
            nn_indices = neigh.kneighbors([X_cat_encoded[y == 1][i]], 
                                        return_distance=False)[0][1:]
            
            for _ in range(int(n_samples / len(X_minority))):
                nn_idx = np.random.choice(nn_indices)
                lambda_val = np.random.random()
                
                synthetic = generate_synthetic_sample(
                    X_minority[i],
                    X_minority[nn_idx],
                    lambda_val
                )
                synthetic_samples.append(synthetic)
        
        return np.array(synthetic_samples)
    
    # Generate synthetic samples
    X_minority = X[y == 1]
    n_samples = sum(y == 0) - sum(y == 1)
    synthetic_samples = mixed_smote(X_minority, n_samples)
    
    return synthetic_samples

# Example usage with mixed data
def create_mixed_dataset():
    # Create synthetic dataset with mixed features
    np.random.seed(42)
    n_samples = 1000
    
    # Numerical features
    X_num = np.random.normal(size=(n_samples, 2))
    
    # Categorical features
    categories = ['A', 'B', 'C']
    X_cat = np.random.choice(categories, size=(n_samples, 2))
    
    # Combine features
    X = np.hstack([X_num, X_cat])
    
    # Create imbalanced labels
    y = np.zeros(n_samples)
    y[:100] = 1  # 10% minority class
    
    return X, y, [2, 3]  # indices of categorical features

# Test the implementation
X_mixed, y_mixed, categorical_features = create_mixed_dataset()
X_synthetic = mixed_data_oversample(X_mixed, y_mixed, categorical_features)
print(f"Original minority samples: {sum(y_mixed == 1)}")
print(f"Synthetic samples generated: {len(X_synthetic)}")
```

Slide 13: Cost-Sensitive Learning Integration

This advanced implementation combines oversampling techniques with cost-sensitive learning to create a more robust approach to imbalanced learning. The method adjusts both sample weights and synthetic sample generation based on misclassification costs.

```python
class CostSensitiveOversampler:
    def __init__(self, cost_matrix=None, oversampling_method='smote'):
        self.cost_matrix = cost_matrix if cost_matrix is not None else {
            (0, 1): 1.0,  # False Negative cost
            (1, 0): 5.0   # False Positive cost
        }
        self.oversampling_method = oversampling_method
    
    def compute_sample_weights(self, y):
        weights = np.ones(len(y))
        for i, yi in enumerate(y):
            weights[i] = sum(self.cost_matrix.get((yi, j), 0) 
                           for j in set(y) if j != yi)
        return weights
    
    def fit_resample(self, X, y):
        # Calculate initial sample weights
        sample_weights = self.compute_sample_weights(y)
        
        # Determine number of synthetic samples based on costs
        minority_cost = self.cost_matrix.get((1, 0), 1.0)
        majority_cost = self.cost_matrix.get((0, 1), 1.0)
        cost_ratio = minority_cost / majority_cost
        
        n_synthetic = int((sum(y == 0) * cost_ratio) - sum(y == 1))
        
        # Generate synthetic samples
        if self.oversampling_method == 'smote':
            X_synthetic = smote_oversample(X[y == 1], n_synthetic)
        elif self.oversampling_method == 'adasyn':
            X_synthetic = adasyn_oversample(X, y, beta=cost_ratio)
        else:
            raise ValueError("Unsupported oversampling method")
        
        # Combine original and synthetic samples
        X_resampled = np.vstack([X, X_synthetic])
        y_resampled = np.hstack([y, np.ones(len(X_synthetic))])
        
        # Update sample weights for combined dataset
        weights_resampled = np.hstack([
            sample_weights,
            np.ones(len(X_synthetic)) * self.cost_matrix.get((1, 0), 1.0)
        ])
        
        return X_resampled, y_resampled, weights_resampled

# Example usage
cost_matrix = {
    (0, 1): 1.0,   # Cost of misclassifying negative as positive
    (1, 0): 10.0   # Cost of misclassifying positive as negative
}

oversample = CostSensitiveOversampler(cost_matrix=cost_matrix)
X_resampled, y_resampled, sample_weights = oversample.fit_resample(X, y)

# Train cost-sensitive model
clf = RandomForestClassifier(random_state=42)
clf.fit(X_resampled, y_resampled, sample_weight=sample_weights)

# Evaluate
y_pred = clf.predict(X_test)
print("Cost-Sensitive Results:")
print(classification_report(y_test, y_pred))
```

Slide 14: Dynamic Oversampling Rate Adjustment

This implementation introduces an adaptive approach that dynamically adjusts oversampling rates based on model performance feedback during training. The method monitors validation metrics to optimize the synthetic sample generation process.

```python
class DynamicOversamplingAdjuster:
    def __init__(self, base_rate=1.0, adjustment_factor=0.1, 
                 min_rate=0.5, max_rate=2.0):
        self.base_rate = base_rate
        self.adjustment_factor = adjustment_factor
        self.min_rate = min_rate
        self.max_rate = max_rate
        self.performance_history = []
        
    def adjust_rate(self, current_performance):
        if len(self.performance_history) > 0:
            prev_performance = self.performance_history[-1]
            
            # Adjust rate based on performance trend
            if current_performance > prev_performance:
                self.base_rate *= (1 + self.adjustment_factor)
            else:
                self.base_rate *= (1 - self.adjustment_factor)
                
            # Ensure rate stays within bounds
            self.base_rate = np.clip(self.base_rate, 
                                   self.min_rate, 
                                   self.max_rate)
        
        self.performance_history.append(current_performance)
        return self.base_rate

    def generate_samples(self, X, y, method='smote'):
        minority_count = sum(y == 1)
        majority_count = sum(y == 0)
        
        # Calculate dynamic number of samples
        n_samples = int((majority_count - minority_count) * self.base_rate)
        
        if method == 'smote':
            return smote_oversample(X[y == 1], n_samples)
        elif method == 'adasyn':
            return adasyn_oversample(X, y, beta=self.base_rate)
        else:
            raise ValueError("Unsupported method")

def train_with_dynamic_oversampling(X, y, n_epochs=5):
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # Initialize adjuster and model
    adjuster = DynamicOversamplingAdjuster()
    clf = RandomForestClassifier(random_state=42)
    
    # Training loop
    for epoch in range(n_epochs):
        # Generate samples with current rate
        X_synthetic = adjuster.generate_samples(X_train, y_train)
        
        # Combine data
        X_combined = np.vstack([X_train, X_synthetic])
        y_combined = np.hstack([y_train, np.ones(len(X_synthetic))])
        
        # Train model
        clf.fit(X_combined, y_combined)
        
        # Evaluate performance
        y_val_pred = clf.predict(X_val)
        current_f1 = f1_score(y_val, y_val_pred)
        
        # Adjust sampling rate
        new_rate = adjuster.adjust_rate(current_f1)
        
        print(f"Epoch {epoch+1}:")
        print(f"F1-Score: {current_f1:.3f}")
        print(f"Sampling Rate: {new_rate:.2f}")
        print(f"Synthetic Samples: {len(X_synthetic)}\n")
    
    return clf, adjuster.performance_history

# Example usage
clf_dynamic, history = train_with_dynamic_oversampling(X, y)
```

Slide 15: Additional Resources

Here are relevant papers from ArXiv that provide deeper insights into oversampling techniques and imbalanced learning:

*   [https://arxiv.org/abs/1106.1813](https://arxiv.org/abs/1106.1813) "SMOTE: Synthetic Minority Over-sampling Technique"
*   [https://arxiv.org/abs/1505.01658](https://arxiv.org/abs/1505.01658) "Learning from Imbalanced Data: A Comprehensive Review"
*   [https://arxiv.org/abs/1608.06048](https://arxiv.org/abs/1608.06048) "A Survey of Predictive Modelling under Imbalanced Distributions"
*   [https://arxiv.org/abs/2105.02340](https://arxiv.org/abs/2105.02340) "Imbalanced Learning: Foundations, Algorithms, and Applications"
*   [https://arxiv.org/abs/1910.10352](https://arxiv.org/abs/1910.10352) "A Systematic Study of the Class Imbalance Problem in Convolutional Neural Networks"

