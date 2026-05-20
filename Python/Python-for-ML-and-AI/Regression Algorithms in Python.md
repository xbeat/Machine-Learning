## Regression Algorithms in Python
Slide 1: Introduction to K-Nearest Neighbors Classification

K-Nearest Neighbors (KNN) is a simple yet powerful non-parametric classification algorithm that makes predictions based on the majority class among the k closest training examples in the feature space, making it fundamentally different from regression-focused algorithms.

```python
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Generate synthetic classification dataset
X, y = make_classification(n_samples=1000, n_features=2, n_classes=2, random_state=42)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train KNN classifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Make predictions
y_pred = knn.predict(X_test)
accuracy = (y_pred == y_test).mean()
print(f"Classification accuracy: {accuracy:.2f}")
```

Slide 2: Mathematical Foundation of Linear Regression

Linear regression establishes a linear relationship between independent variables and a continuous target variable, forming the basis for many advanced regression techniques through the minimization of squared errors.

```python
# Mathematical representation of Linear Regression
'''
$$
y = \beta_0 + \beta_1x_1 + \beta_2x_2 + ... + \beta_nx_n + \epsilon
$$

Cost Function (Mean Squared Error):
$$
MSE = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2
$$

Parameter Estimation:
$$
\beta = (X^TX)^{-1}X^Ty
$$
'''

import numpy as np
from sklearn.linear_model import LinearRegression

# Generate synthetic data
X = np.random.randn(100, 1)
y = 2 * X + 1 + np.random.randn(100, 1) * 0.1

# Create and train model
model = LinearRegression()
model.fit(X, y)

print(f"Coefficient: {model.coef_[0][0]:.2f}")
print(f"Intercept: {model.intercept_[0]:.2f}")
```

Slide 3: Support Vector Classification (SVC)

Support Vector Classification constructs a hyperplane that maximizes the margin between different classes in the feature space, utilizing kernel tricks for non-linear classification problems while maintaining its fundamental classification nature.

```python
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

# Prepare data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Initialize and train SVM classifier
svc = SVC(kernel='rbf', C=1.0)
svc.fit(X_scaled, y.ravel())

# Evaluate model
accuracy = svc.score(X_scaled, y)
print(f"SVC Accuracy: {accuracy:.2f}")

# Kernel function (RBF):
'''
$$
K(x_i, x_j) = \exp\left(-\gamma ||x_i - x_j||^2\right)
$$
'''
```

Slide 4: Decision Trees for Classification

Decision trees partition the feature space into regions using recursive binary splitting, making them naturally suited for classification tasks through the optimization of metrics like Gini impurity or entropy.

```python
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

# Create and train decision tree classifier
dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(X_train, y_train)

# Visualization code
def plot_decision_boundary(clf, X, y):
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu)
    plt.show()

plot_decision_boundary(dt, X, y)
```

Slide 5: Random Forest Classification

Random Forest combines multiple decision trees through bootstrap aggregation (bagging) and random feature selection, creating a robust ensemble method specifically designed for classification tasks.

```python
from sklearn.ensemble import RandomForestClassifier

# Initialize and train Random Forest classifier
rf = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42)
rf.fit(X_train, y_train)

# Feature importance analysis
importances = rf.feature_importances_
features = [f"Feature {i}" for i in range(X.shape[1])]

# Plot feature importance
plt.figure(figsize=(10, 5))
plt.bar(features, importances)
plt.title("Feature Importance in Random Forest")
plt.show()

# Model evaluation
accuracy = rf.score(X_test, y_test)
print(f"Random Forest Accuracy: {accuracy:.2f}")
```

Slide 6: Gradient Boosting for Classification

Gradient Boosting builds an ensemble of weak learners sequentially, with each new tree focusing on the mistakes of the previous ones, making it particularly effective for classification through the optimization of classification-specific loss functions.

```python
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report

# Create and train Gradient Boosting classifier
gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3)
gb.fit(X_train, y_train)

# Make predictions
y_pred = gb.predict(X_test)

# Print detailed classification report
print(classification_report(y_test, y_pred))

# Loss function visualization
train_scores = []
for stage in range(100):
    train_scores.append(gb.train_score_[stage])

plt.figure(figsize=(10, 5))
plt.plot(range(100), train_scores)
plt.title('Training Score vs Boosting Iteration')
plt.xlabel('Boosting Iteration')
plt.ylabel('Training Score')
plt.show()
```

Slide 7: Neural Networks for Classification

Neural networks for classification tasks utilize specialized output layers with activation functions like softmax, fundamentally differing from regression networks that output continuous values through linear activation functions.

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import LabelBinarizer

# Convert labels to one-hot encoding
lb = LabelBinarizer()
y_train_encoded = lb.fit_transform(y_train)
y_test_encoded = lb.transform(y_test)

# Create neural network for classification
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile and train
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train_encoded, epochs=50, batch_size=32, 
                   validation_split=0.2, verbose=0)
```

Slide 8: Logistic Regression Implementation

Logistic regression transforms linear combinations of features through the sigmoid function to model probability distributions, making it inherently suited for classification rather than regression tasks.

```python
class LogisticRegression:
    def __init__(self, learning_rate=0.01, iterations=1000):
        self.lr = learning_rate
        self.iterations = iterations
        
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y):
        self.weights = np.zeros(X.shape[1])
        self.bias = 0
        
        for _ in range(self.iterations):
            # Forward pass
            z = np.dot(X, self.weights) + self.bias
            predictions = self.sigmoid(z)
            
            # Compute gradients
            dw = (1/X.shape[0]) * np.dot(X.T, (predictions - y))
            db = (1/X.shape[0]) * np.sum(predictions - y)
            
            # Update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
            
    def predict(self, X):
        z = np.dot(X, self.weights) + self.bias
        return (self.sigmoid(z) >= 0.5).astype(int)

# Test implementation
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
y_pred = log_reg.predict(X_test)
accuracy = np.mean(y_pred == y_test)
print(f"Logistic Regression Accuracy: {accuracy:.2f}")
```

Slide 9: Naive Bayes Classification

Naive Bayes applies Bayes' theorem with strong independence assumptions between features, making it a probabilistic classifier rather than a regression algorithm, particularly effective for high-dimensional classification problems.

```python
from sklearn.naive_bayes import GaussianNB
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Create and train Naive Bayes classifier
nb = GaussianNB()
nb.fit(X_train, y_train)

# Make predictions
y_pred = nb.predict(X_test)

# Create confusion matrix visualization
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix for Naive Bayes')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# Print model probabilities
print("Class probabilities:", nb.predict_proba(X_test[:5]))
```

Slide 10: XGBoost for Classification Tasks

XGBoost implements gradient boosting with specialized objective functions for classification, incorporating advanced features like tree pruning and handling class imbalance through scale\_pos\_weight parameter.

```python
import xgboost as xgb
from sklearn.metrics import roc_curve, auc

# Convert data to DMatrix format
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# Set parameters for classification
params = {
    'objective': 'binary:logistic',
    'max_depth': 3,
    'eta': 0.1,
    'eval_metric': 'logloss'
}

# Train model
model = xgb.train(params, dtrain, num_boost_round=100)

# Generate ROC curve
y_pred_proba = model.predict(dtest)
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for XGBoost Classifier')
plt.legend(loc="lower right")
plt.show()
```

Slide 11: LightGBM Classification Implementation

LightGBM employs a leaf-wise growth strategy optimized for classification tasks, incorporating gradient-based one-side sampling to handle class distribution effectively.

```python
import lightgbm as lgb
from sklearn.metrics import accuracy_score

# Create dataset format
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

# Set parameters
params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9
}

# Train model
model = lgb.train(params,
                 train_data,
                 num_boost_round=100,
                 valid_sets=[test_data],
                 early_stopping_rounds=10)

# Make predictions
y_pred = model.predict(X_test)
y_pred_binary = (y_pred > 0.5).astype(int)
accuracy = accuracy_score(y_test, y_pred_binary)
print(f"LightGBM Accuracy: {accuracy:.2f}")

# Feature importance visualization
lgb.plot_importance(model, figsize=(10, 6))
plt.title("Feature Importance in LightGBM")
plt.show()
```

Slide 12: Real-world Application: Credit Card Fraud Detection

Implementation of multiple classification algorithms for detecting fraudulent transactions, demonstrating the practical application of classification techniques in financial security.

```python
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_curve
import pandas as pd

# Simulate credit card transaction data
np.random.seed(42)
n_samples = 10000
n_features = 10

# Generate synthetic transaction data
X = np.random.randn(n_samples, n_features)
# Create imbalanced classes (99.7% normal, 0.3% fraudulent)
y = np.random.choice([0, 1], size=n_samples, p=[0.997, 0.003])

# Preprocess data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train multiple classifiers
classifiers = {
    'XGBoost': xgb.XGBClassifier(scale_pos_weight=300),
    'LightGBM': lgb.LGBMClassifier(is_unbalanced=True),
    'RandomForest': RandomForestClassifier(class_weight='balanced')
}

# Evaluate and compare results
results = {}
for name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = accuracy
    print(f"{name} Accuracy: {accuracy:.4f}")
```

Slide 13: Additional Resources

*   Machine Learning Classification Algorithms - A Comprehensive Survey [https://arxiv.org/abs/2106.04386](https://arxiv.org/abs/2106.04386)
*   XGBoost: A Scalable Tree Boosting System [https://arxiv.org/abs/1603.02754](https://arxiv.org/abs/1603.02754)
*   LightGBM: A Highly Efficient Gradient Boosting Decision Tree [https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree](https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree)
*   Recent Advances in Deep Learning for Classification Tasks [https://arxiv.org/abs/2012.03365](https://arxiv.org/abs/2012.03365)
*   Ensemble Methods in Machine Learning: A Modern Perspective [https://arxiv.org/abs/2012.00991](https://arxiv.org/abs/2012.00991)

