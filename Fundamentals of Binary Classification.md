## Fundamentals of Binary Classification
Slide 1: Binary Classification Fundamentals

Binary classification forms the foundation of many machine learning applications, where the goal is to categorize input data into one of two classes. This implementation demonstrates a basic binary classifier using logistic regression from scratch, showcasing the core mathematical principles and probability-based decision making.

```python
import numpy as np

class BinaryClassifier:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.lr = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Training loop
        for _ in range(self.epochs):
            linear_pred = np.dot(X, self.weights) + self.bias
            predictions = self.sigmoid(linear_pred)
            
            # Update weights and bias
            dw = (1/n_samples) * np.dot(X.T, (predictions - y))
            db = (1/n_samples) * np.sum(predictions - y)
            
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
    
    def predict(self, X, threshold=0.5):
        linear_pred = np.dot(X, self.weights) + self.bias
        probabilities = self.sigmoid(linear_pred)
        return (probabilities >= threshold).astype(int)
```

Slide 2: Data Preparation and Preprocessing

Effective binary classification requires careful data preparation. This implementation shows essential preprocessing steps including normalization, handling missing values, and train-test splitting, which are crucial for model performance.

```python
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def prepare_binary_data(X, y, test_size=0.2, random_state=42):
    # Handle missing values
    X = np.nan_to_num(X, nan=np.nanmean(X))
    
    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, 
        test_size=test_size, 
        random_state=random_state,
        stratify=y
    )
    
    # Check class balance
    train_balance = np.bincount(y_train) / len(y_train)
    print(f"Training set class balance: {train_balance}")
    
    return X_train, X_test, y_train, y_test, scaler
```

Slide 3: Performance Metrics Implementation

Understanding model performance requires comprehensive evaluation metrics. This implementation creates a metrics class that calculates accuracy, precision, recall, F1-score, and generates a confusion matrix for binary classification tasks.

```python
import numpy as np

class BinaryMetrics:
    def __init__(self, y_true, y_pred):
        self.y_true = y_true
        self.y_pred = y_pred
        self.tp = np.sum((y_true == 1) & (y_pred == 1))
        self.tn = np.sum((y_true == 0) & (y_pred == 0))
        self.fp = np.sum((y_true == 0) & (y_pred == 1))
        self.fn = np.sum((y_true == 1) & (y_pred == 0))
    
    def accuracy(self):
        return (self.tp + self.tn) / len(self.y_true)
    
    def precision(self):
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 0
    
    def recall(self):
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0
    
    def f1_score(self):
        prec = self.precision()
        rec = self.recall()
        return 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
    
    def confusion_matrix(self):
        return np.array([[self.tn, self.fp], 
                        [self.fn, self.tp]])
```

Slide 4: ROC Curve Implementation

The Receiver Operating Characteristic (ROC) curve is essential for evaluating binary classifier performance across different threshold values. This implementation creates a complete ROC curve visualization with AUC calculation.

```python
import numpy as np
import matplotlib.pyplot as plt

def plot_roc_curve(y_true, y_prob):
    thresholds = np.linspace(0, 1, 100)
    tpr_list = []
    fpr_list = []
    
    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        tpr_list.append(tpr)
        fpr_list.append(fpr)
    
    # Calculate AUC
    auc = np.trapz(tpr_list, fpr_list)
    
    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(fpr_list, tpr_list, label=f'ROC (AUC = {auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True)
    return auc
```

Slide 5: Cross-Validation Implementation

Cross-validation is crucial for assessing model generalization and preventing overfitting in binary classification. This implementation demonstrates stratified k-fold cross-validation with performance metrics tracking across folds.

```python
import numpy as np
from sklearn.model_selection import StratifiedKFold

class CrossValidator:
    def __init__(self, model, n_splits=5):
        self.model = model
        self.n_splits = n_splits
        self.metrics_history = []
    
    def validate(self, X, y):
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        fold_metrics = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_train_fold = X[train_idx]
            y_train_fold = y[train_idx]
            X_val_fold = X[val_idx]
            y_val_fold = y[val_idx]
            
            # Train and evaluate
            self.model.fit(X_train_fold, y_train_fold)
            y_pred = self.model.predict(X_val_fold)
            
            # Calculate metrics
            metrics = BinaryMetrics(y_val_fold, y_pred)
            fold_metrics.append({
                'fold': fold + 1,
                'accuracy': metrics.accuracy(),
                'precision': metrics.precision(),
                'recall': metrics.recall(),
                'f1': metrics.f1_score()
            })
        
        return fold_metrics
```

Slide 6: Decision Boundary Visualization

Understanding decision boundaries helps visualize how binary classifiers separate classes in feature space. This implementation creates a detailed visualization of decision boundaries with probability contours.

```python
import numpy as np
import matplotlib.pyplot as plt

def plot_decision_boundary(model, X, y, title='Decision Boundary'):
    # Create mesh grid
    h = 0.02  # Step size
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # Make predictions on mesh grid
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(title)
    plt.colorbar(label='Prediction Probability')
    return plt
```

Slide 7: Real-World Example - Email Spam Classification

This implementation demonstrates binary classification for email spam detection, including text preprocessing, feature extraction using TF-IDF, and model training with comprehensive evaluation.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np

class SpamClassifier:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.classifier = BinaryClassifier(learning_rate, epochs)
        self.vectorizer = TfidfVectorizer(max_features=1000)
        
    def preprocess_text(self, texts):
        # Convert to lowercase and remove special characters
        texts = [str(text).lower() for text in texts]
        return texts
    
    def fit(self, texts, labels):
        processed_texts = self.preprocess_text(texts)
        X = self.vectorizer.fit_transform(processed_texts).toarray()
        self.classifier.fit(X, labels)
    
    def predict(self, texts):
        processed_texts = self.preprocess_text(texts)
        X = self.vectorizer.transform(processed_texts).toarray()
        return self.classifier.predict(X)

# Example usage
texts = ["Free money now!", "Meeting at 3pm", "Win prizes click here"]
labels = np.array([1, 0, 1])  # 1 for spam, 0 for not spam
spam_classifier = SpamClassifier()
spam_classifier.fit(texts, labels)
```

Slide 8: Handling Class Imbalance

Class imbalance is a common challenge in binary classification where one class significantly outnumbers the other. This implementation shows various techniques to address this issue, including SMOTE and class weights.

```python
import numpy as np
from collections import Counter

class ImbalanceHandler:
    def __init__(self, strategy='smote'):
        self.strategy = strategy
        
    def smote_sampling(self, X, y):
        # Synthetic Minority Over-sampling Technique
        minority_class = X[y == 1]
        n_samples = len(X[y == 0]) - len(minority_class)
        
        synthetic_samples = []
        for i in range(n_samples):
            sample = minority_class[np.random.randint(0, len(minority_class))]
            neighbor = minority_class[np.random.randint(0, len(minority_class))]
            synthetic = sample + np.random.random() * (neighbor - sample)
            synthetic_samples.append(synthetic)
        
        X_balanced = np.vstack([X, synthetic_samples])
        y_balanced = np.hstack([y, np.ones(n_samples)])
        
        return X_balanced, y_balanced
    
    def class_weights(self, y):
        counter = Counter(y)
        majority = max(counter.values())
        return {cls: majority/count for cls, count in counter.items()}
    
    def fit_transform(self, X, y):
        if self.strategy == 'smote':
            return self.smote_sampling(X, y)
        elif self.strategy == 'weights':
            return X, y, self.class_weights(y)
```

Slide 9: Ensemble Binary Classifier

Ensemble methods combine multiple binary classifiers to improve overall performance. This implementation creates a voting ensemble with different base classifiers and weighted voting mechanisms.

```python
import numpy as np
from abc import ABC, abstractmethod

class BaseClassifier(ABC):
    @abstractmethod
    def fit(self, X, y):
        pass
    
    @abstractmethod
    def predict_proba(self, X):
        pass

class EnsembleClassifier:
    def __init__(self, classifiers, weights=None):
        self.classifiers = classifiers
        self.weights = weights if weights else [1/len(classifiers)]*len(classifiers)
    
    def fit(self, X, y):
        for classifier in self.classifiers:
            classifier.fit(X, y)
    
    def predict(self, X, threshold=0.5):
        probas = []
        for clf, weight in zip(self.classifiers, self.weights):
            prob = clf.predict_proba(X)[:, 1]
            probas.append(weight * prob)
        
        avg_probas = np.sum(probas, axis=0) / np.sum(self.weights)
        return (avg_probas >= threshold).astype(int)
    
    def predict_proba(self, X):
        weighted_probas = []
        for clf, weight in zip(self.classifiers, self.weights):
            prob = clf.predict_proba(X) * weight
            weighted_probas.append(prob)
        return np.mean(weighted_probas, axis=0)
```

Slide 10: Feature Selection for Binary Classification

Feature selection is crucial for improving model performance and reducing dimensionality. This implementation showcases different feature selection methods specifically designed for binary classification.

```python
import numpy as np
from scipy.stats import chi2_contingency

class BinaryFeatureSelector:
    def __init__(self, method='chi2', k_features=10):
        self.method = method
        self.k_features = k_features
        self.selected_features = None
    
    def chi_square_score(self, X, y):
        scores = []
        for i in range(X.shape[1]):
            contingency = pd.crosstab(y, pd.qcut(X[:, i], 10))
            chi2, _ = chi2_contingency(contingency)
            scores.append(chi2)
        return np.array(scores)
    
    def mutual_information(self, X, y):
        scores = []
        for i in range(X.shape[1]):
            hist_2d, x_edges, y_edges = np.histogram2d(X[:, i], y, bins=10)
            p_xy = hist_2d / np.sum(hist_2d)
            p_x = np.sum(p_xy, axis=1)
            p_y = np.sum(p_xy, axis=0)
            
            H_xy = -np.sum(p_xy * np.log2(p_xy + 1e-10))
            H_x = -np.sum(p_x * np.log2(p_x + 1e-10))
            H_y = -np.sum(p_y * np.log2(p_y + 1e-10))
            
            mi = H_x + H_y - H_xy
            scores.append(mi)
        return np.array(scores)
    
    def fit_transform(self, X, y):
        if self.method == 'chi2':
            scores = self.chi_square_score(X, y)
        else:
            scores = self.mutual_information(X, y)
        
        self.selected_features = np.argsort(scores)[-self.k_features:]
        return X[:, self.selected_features]
```

Slide 11: Early Stopping Implementation

Early stopping is a crucial regularization technique that prevents overfitting in binary classification by monitoring validation performance. This implementation demonstrates a flexible early stopping mechanism with various stopping criteria.

```python
import numpy as np

class EarlyStopping:
    def __init__(self, patience=5, min_delta=1e-4, monitor='val_loss'):
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.best_value = np.inf
        self.counter = 0
        self.should_stop = False
        self.best_weights = None
    
    def __call__(self, model, current_value):
        if current_value < self.best_value - self.min_delta:
            self.best_value = current_value
            self.counter = 0
            self.best_weights = model.weights.copy()
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                if self.best_weights is not None:
                    model.weights = self.best_weights
        
        return self.should_stop

class BinaryClassifierWithEarlyStopping(BinaryClassifier):
    def fit(self, X, y, X_val=None, y_val=None):
        early_stopping = EarlyStopping()
        
        for epoch in range(self.epochs):
            # Train step
            linear_pred = np.dot(X, self.weights) + self.bias
            predictions = self.sigmoid(linear_pred)
            
            # Validation step
            if X_val is not None and y_val is not None:
                val_pred = self.predict(X_val)
                val_loss = -np.mean(y_val * np.log(val_pred + 1e-10) + 
                                  (1 - y_val) * np.log(1 - val_pred + 1e-10))
                
                if early_stopping(self, val_loss):
                    break
```

Slide 12: Real-World Example - Credit Card Fraud Detection

This implementation showcases a complete binary classification pipeline for credit card fraud detection, including data preprocessing, feature engineering, and model evaluation with specific focus on handling highly imbalanced data.

```python
import numpy as np
from sklearn.preprocessing import RobustScaler

class FraudDetector:
    def __init__(self):
        self.scaler = RobustScaler()
        self.classifier = BinaryClassifier()
        self.feature_selector = BinaryFeatureSelector()
        self.imbalance_handler = ImbalanceHandler()
    
    def preprocess_features(self, X):
        # Scale amount and time features
        X_scaled = self.scaler.fit_transform(X)
        
        # Add interaction features
        X_interactions = np.zeros((X_scaled.shape[0], 3))
        X_interactions[:, 0] = X_scaled[:, 0] * X_scaled[:, 1]  # Time * Amount
        X_interactions[:, 1] = np.square(X_scaled[:, 0])        # Time^2
        X_interactions[:, 2] = np.square(X_scaled[:, 1])        # Amount^2
        
        return np.hstack([X_scaled, X_interactions])
    
    def fit(self, X, y):
        # Preprocess
        X_processed = self.preprocess_features(X)
        
        # Feature selection
        X_selected = self.feature_selector.fit_transform(X_processed, y)
        
        # Handle imbalance
        X_balanced, y_balanced = self.imbalance_handler.fit_transform(X_selected, y)
        
        # Train
        self.classifier.fit(X_balanced, y_balanced)
    
    def predict(self, X):
        X_processed = self.preprocess_features(X)
        X_selected = self.feature_selector.transform(X_processed)
        return self.classifier.predict(X_selected)
```

Slide 13: Additional Resources

*   Recent advances in binary classification with deep learning:
    *   [https://arxiv.org/abs/2304.12210](https://arxiv.org/abs/2304.12210)
    *   [https://arxiv.org/abs/2303.15681](https://arxiv.org/abs/2303.15681)
    *   [https://arxiv.org/abs/2302.09778](https://arxiv.org/abs/2302.09778)
*   Recommended resources for further learning:
    *   [https://www.sciencedirect.com/science/article/pii/S2352711023001723](https://www.sciencedirect.com/science/article/pii/S2352711023001723)
    *   Machine Learning Mastery: [https://machinelearningmastery.com/binary-classification-tutorial-with-the-keras-deep-learning-library/](https://machinelearningmastery.com/binary-classification-tutorial-with-the-keras-deep-learning-library/)
    *   Towards Data Science: [https://towardsdatascience.com/binary-classification-techniques-explained-a-visual-guide-to-12-basic-algorithms-46626c2e2159](https://towardsdatascience.com/binary-classification-techniques-explained-a-visual-guide-to-12-basic-algorithms-46626c2e2159)

