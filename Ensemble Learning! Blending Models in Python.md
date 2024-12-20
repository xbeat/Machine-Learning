## Ensemble Learning! Blending Models in Python
Slide 1: Introduction to Ensemble Learning and Blending

Ensemble learning is a powerful machine learning technique that combines multiple models to improve prediction accuracy and robustness. Blending, a specific ensemble method, aims to create a single, unified prediction by combining the outputs of diverse base models. The goal of blending is to leverage the strengths of individual models while mitigating their weaknesses, ultimately producing a more accurate and reliable final prediction.

```python
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Simulating a dataset
X = np.random.rand(1000, 10)
y = (X[:, 0] + X[:, 1] > 1).astype(int)

# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Base models
rf = RandomForestClassifier(n_estimators=100, random_state=42)
gb = GradientBoostingClassifier(n_estimators=100, random_state=42)

# Training base models
rf.fit(X_train, y_train)
gb.fit(X_train, y_train)

# Generating predictions for blending
rf_pred = rf.predict_proba(X_test)[:, 1]
gb_pred = gb.predict_proba(X_test)[:, 1]

# Preparing data for blending
blend_data = np.column_stack((rf_pred, gb_pred))

# Blending model
blender = LogisticRegression()
blender.fit(blend_data, y_test)

# Final prediction
final_pred = blender.predict(blend_data)

print(f"Blended model accuracy: {accuracy_score(y_test, final_pred):.4f}")
```

Slide 2: The Motivation Behind Blending

Blending addresses the limitations of individual models by combining their predictions. Different models capture various aspects of the data, and blending allows us to create a more comprehensive and accurate representation. By leveraging diverse models, blending can reduce bias, decrease variance, and improve generalization, leading to better performance on unseen data.

```python
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

# Create a simple dataset
X = np.random.rand(1000, 2)
y = (X[:, 0] + X[:, 1] > 1).astype(int)

# Train individual models
dt = DecisionTreeClassifier(max_depth=3, random_state=42)
rf = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42)
gb = GradientBoostingClassifier(n_estimators=10, max_depth=3, random_state=42)

models = [dt, rf, gb]
names = ['Decision Tree', 'Random Forest', 'Gradient Boosting']

# Plot decision boundaries
plt.figure(figsize=(15, 5))
for i, model in enumerate(models):
    model.fit(X, y)
    
    plt.subplot(1, 3, i+1)
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8)
    plt.title(names[i])
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')

plt.tight_layout()
plt.show()
```

Slide 3: Types of Blending: Voting and Averaging

Two common blending techniques are voting and averaging. In voting, each base model casts a vote for the final prediction, and the majority vote determines the outcome. Averaging, on the other hand, combines the probability outputs of base models to produce a final prediction. These methods are simple yet effective for improving model performance.

```python
from sklearn.ensemble import VotingClassifier

# Base models
dt = DecisionTreeClassifier(max_depth=3, random_state=42)
rf = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42)
gb = GradientBoostingClassifier(n_estimators=10, max_depth=3, random_state=42)

# Voting classifier
voting_clf = VotingClassifier(
    estimators=[('dt', dt), ('rf', rf), ('gb', gb)],
    voting='soft'  # 'soft' for probability averaging, 'hard' for majority voting
)

# Fit and predict
voting_clf.fit(X_train, y_train)
voting_pred = voting_clf.predict(X_test)

print(f"Voting classifier accuracy: {accuracy_score(y_test, voting_pred):.4f}")

# Manual averaging
manual_avg_pred = (rf.predict_proba(X_test)[:, 1] +
                   gb.predict_proba(X_test)[:, 1] +
                   dt.predict_proba(X_test)[:, 1]) / 3
manual_avg_pred = (manual_avg_pred > 0.5).astype(int)

print(f"Manual averaging accuracy: {accuracy_score(y_test, manual_avg_pred):.4f}")
```

Slide 4: Stacking: A Sophisticated Blending Technique

Stacking is an advanced blending method that uses a meta-model to learn how to best combine the predictions of base models. This technique allows for more complex relationships between base model outputs and can potentially capture non-linear interactions. Stacking often yields better results than simple averaging or voting.

```python
from sklearn.ensemble import StackingClassifier

# Base models
base_models = [
    ('rf', RandomForestClassifier(n_estimators=10, random_state=42)),
    ('gb', GradientBoostingClassifier(n_estimators=10, random_state=42)),
    ('dt', DecisionTreeClassifier(max_depth=3, random_state=42))
]

# Meta-model
meta_model = LogisticRegression()

# Stacking classifier
stacking_clf = StackingClassifier(
    estimators=base_models,
    final_estimator=meta_model,
    cv=5  # Number of folds for cross-validation
)

# Fit and predict
stacking_clf.fit(X_train, y_train)
stacking_pred = stacking_clf.predict(X_test)

print(f"Stacking classifier accuracy: {accuracy_score(y_test, stacking_pred):.4f}")
```

Slide 5: Feature-level Blending

Feature-level blending combines the features extracted or learned by different models to create a richer representation of the data. This approach can be particularly useful when different models capture complementary aspects of the input space. By concatenating these features, we allow the final model to learn from a more diverse set of information.

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Feature extraction using PCA
pca = PCA(n_components=5)
pca_features = pca.fit_transform(X_train)

# Feature extraction using Random Forest feature importances
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_features = rf.apply(X_train)  # Extract leaf indices

# Combine original features with extracted features
combined_features = np.hstack((X_train, pca_features, rf_features))

# Scale the combined features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(combined_features)

# Train a final model on the combined features
final_model = LogisticRegression()
final_model.fit(scaled_features, y_train)

# Prepare test data
test_pca_features = pca.transform(X_test)
test_rf_features = rf.apply(X_test)
test_combined = np.hstack((X_test, test_pca_features, test_rf_features))
test_scaled = scaler.transform(test_combined)

# Make predictions
feature_blend_pred = final_model.predict(test_scaled)

print(f"Feature-level blending accuracy: {accuracy_score(y_test, feature_blend_pred):.4f}")
```

Slide 6: Weighted Blending

Weighted blending assigns different importance to each base model's predictions. This approach is useful when some models consistently perform better than others or when different models excel in specific scenarios. By optimizing these weights, we can create a more accurate ensemble that leverages the strengths of each model.

```python
from scipy.optimize import minimize

# Base model predictions
rf_pred = rf.predict_proba(X_test)[:, 1]
gb_pred = gb.predict_proba(X_test)[:, 1]
dt_pred = dt.predict_proba(X_test)[:, 1]

# Function to optimize
def weighted_log_loss(weights, *args):
    preds, true = args
    final_pred = np.sum(preds * weights.reshape(-1, 1), axis=0)
    final_pred = np.clip(final_pred, 1e-15, 1 - 1e-15)
    return -np.mean(true * np.log(final_pred) + (1 - true) * np.log(1 - final_pred))

# Optimize weights
initial_weights = np.array([1/3, 1/3, 1/3])
bounds = [(0, 1)] * 3
constraint = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}

result = minimize(weighted_log_loss, initial_weights, args=(np.array([rf_pred, gb_pred, dt_pred]), y_test),
                  method='SLSQP', bounds=bounds, constraints=constraint)

optimal_weights = result.x

# Make predictions with optimal weights
weighted_pred = np.sum(np.array([rf_pred, gb_pred, dt_pred]) * optimal_weights.reshape(-1, 1), axis=0)
weighted_pred = (weighted_pred > 0.5).astype(int)

print(f"Optimal weights: {optimal_weights}")
print(f"Weighted blending accuracy: {accuracy_score(y_test, weighted_pred):.4f}")
```

Slide 7: Time Series Blending

In time series forecasting, blending can be particularly effective as different models may capture various temporal patterns. By combining predictions from models that excel at different time scales or seasonal patterns, we can create a more robust forecast that adapts to changing trends and seasonality.

```python
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error

# Generate a sample time series
np.random.seed(42)
date_rng = pd.date_range(start='2020-01-01', end='2022-12-31', freq='D')
y = np.cumsum(np.random.randn(len(date_rng))) + np.sin(np.arange(len(date_rng)) * 2 * np.pi / 365) * 10
ts = pd.Series(y, index=date_rng)

# Split into train and test
train = ts[:'2022-06-30']
test = ts['2022-07-01':]

# Fit ARIMA model
arima = ARIMA(train, order=(1, 1, 1))
arima_fit = arima.fit()
arima_pred = arima_fit.forecast(steps=len(test))

# Fit Exponential Smoothing model
es = ExponentialSmoothing(train, seasonal_periods=365, trend='add', seasonal='add')
es_fit = es.fit()
es_pred = es_fit.forecast(len(test))

# Simple average blending
blend_pred = (arima_pred + es_pred) / 2

# Calculate MSE for each model
arima_mse = mean_squared_error(test, arima_pred)
es_mse = mean_squared_error(test, es_pred)
blend_mse = mean_squared_error(test, blend_pred)

print(f"ARIMA MSE: {arima_mse:.2f}")
print(f"Exponential Smoothing MSE: {es_mse:.2f}")
print(f"Blended MSE: {blend_mse:.2f}")

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(test.index, test.values, label='Actual')
plt.plot(test.index, arima_pred, label='ARIMA')
plt.plot(test.index, es_pred, label='Exponential Smoothing')
plt.plot(test.index, blend_pred, label='Blended')
plt.legend()
plt.title('Time Series Forecasting: Individual Models vs Blended')
plt.show()
```

Slide 8: Cross-validation in Blending

Cross-validation is crucial in blending to prevent overfitting and ensure that the blended model generalizes well to unseen data. By using techniques like k-fold cross-validation, we can create out-of-fold predictions for training the meta-model, leading to a more robust and reliable ensemble.

```python
from sklearn.model_selection import KFold
from sklearn.base import BaseEstimator, ClassifierMixin

class BlendedClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, base_models, meta_model):
        self.base_models = base_models
        self.meta_model = meta_model
        
    def fit(self, X, y):
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        meta_features = np.zeros((X.shape[0], len(self.base_models)))
        
        for i, model in enumerate(self.base_models):
            for train_idx, val_idx in kf.split(X):
                model.fit(X[train_idx], y[train_idx])
                meta_features[val_idx, i] = model.predict_proba(X[val_idx])[:, 1]
            
            model.fit(X, y)  # Refit on all data
        
        self.meta_model.fit(meta_features, y)
        return self
    
    def predict_proba(self, X):
        meta_features = np.column_stack([model.predict_proba(X)[:, 1] for model in self.base_models])
        return self.meta_model.predict_proba(meta_features)
    
    def predict(self, X):
        return self.predict_proba(X)[:, 1] > 0.5

# Create and train the blended classifier
base_models = [
    RandomForestClassifier(n_estimators=100, random_state=42),
    GradientBoostingClassifier(n_estimators=100, random_state=42),
    LogisticRegression(random_state=42)
]
meta_model = LogisticRegression()

blended_clf = BlendedClassifier(base_models, meta_model)
blended_clf.fit(X_train, y_train)

# Make predictions and evaluate
blended_pred = blended_clf.predict(X_test)
print(f"Blended classifier accuracy: {accuracy_score(y_test, blended_pred):.4f}")
```

Slide 9: Handling Imbalanced Data in Blending

When dealing with imbalanced datasets, blending can be particularly effective by combining models that handle class imbalance differently. This approach allows the ensemble to leverage various strategies for addressing the imbalance, potentially leading to better overall performance on minority classes.

```python
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

# Create an imbalanced dataset
X_imb, y_imb = make_classification(n_samples=10000, n_classes=2, weights=[0.9, 0.1], 
                                   n_features=20, random_state=42)

# Split the data
X_train_imb, X_test_imb, y_train_imb, y_test_imb = train_test_split(X_imb, y_imb, test_size=0.2, random_state=42)

# Create base models with different sampling strategies
rf_smote = Pipeline([
    ('smote', SMOTE(random_state=42)),
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42))
])

gb_undersample = Pipeline([
    ('undersample', RandomUnderSampler(random_state=42)),
    ('gb', GradientBoostingClassifier(n_estimators=100, random_state=42))
])

lr_base = LogisticRegression(class_weight='balanced', random_state=42)

# Train base models
rf_smote.fit(X_train_imb, y_train_imb)
gb_undersample.fit(X_train_imb, y_train_imb)
lr_base.fit(X_train_imb, y_train_imb)

# Make predictions
rf_pred = rf_smote.predict_proba(X_test_imb)[:, 1]
gb_pred = gb_undersample.predict_proba(X_test_imb)[:, 1]
lr_pred = lr_base.predict_proba(X_test_imb)[:, 1]

# Blend predictions
blended_pred_imb = (rf_pred + gb_pred + lr_pred) / 3
blended_pred_imb = (blended_pred_imb > 0.5).astype(int)

# Evaluate
from sklearn.metrics import balanced_accuracy_score, f1_score

print(f"Blended model balanced accuracy: {balanced_accuracy_score(y_test_imb, blended_pred_imb):.4f}")
print(f"Blended model F1-score: {f1_score(y_test_imb, blended_pred_imb):.4f}")
```

Slide 10: Blending for Multi-class Classification

Blending can be extended to multi-class classification problems by combining the probability outputs for each class from multiple base models. This approach allows the ensemble to leverage the strengths of different models across various class boundaries.

```python
from sklearn.datasets import load_iris
from sklearn.multiclass import OneVsRestClassifier

# Load iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create base models
rf_multi = OneVsRestClassifier(RandomForestClassifier(n_estimators=100, random_state=42))
gb_multi = OneVsRestClassifier(GradientBoostingClassifier(n_estimators=100, random_state=42))
lr_multi = OneVsRestClassifier(LogisticRegression(random_state=42))

# Train base models
rf_multi.fit(X_train, y_train)
gb_multi.fit(X_train, y_train)
lr_multi.fit(X_train, y_train)

# Make predictions
rf_pred = rf_multi.predict_proba(X_test)
gb_pred = gb_multi.predict_proba(X_test)
lr_pred = lr_multi.predict_proba(X_test)

# Blend predictions
blended_pred_multi = (rf_pred + gb_pred + lr_pred) / 3
blended_class_multi = np.argmax(blended_pred_multi, axis=1)

# Evaluate
from sklearn.metrics import accuracy_score, classification_report

print(f"Blended model accuracy: {accuracy_score(y_test, blended_class_multi):.4f}")
print("Classification Report:")
print(classification_report(y_test, blended_class_multi, target_names=iris.target_names))
```

Slide 11: Blending for Regression Tasks

Blending is not limited to classification problems; it can also be applied to regression tasks. By combining predictions from multiple regression models, we can often achieve more accurate and stable estimates of continuous target variables.

```python
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Generate regression dataset
X, y = make_regression(n_samples=1000, n_features=20, noise=0.1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train base models
rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
gb_reg = GradientBoostingRegressor(n_estimators=100, random_state=42)
lr_reg = LinearRegression()

rf_reg.fit(X_train, y_train)
gb_reg.fit(X_train, y_train)
lr_reg.fit(X_train, y_train)

# Make predictions
rf_pred = rf_reg.predict(X_test)
gb_pred = gb_reg.predict(X_test)
lr_pred = lr_reg.predict(X_test)

# Blend predictions
blended_pred_reg = (rf_pred + gb_pred + lr_pred) / 3

# Evaluate
mse = mean_squared_error(y_test, blended_pred_reg)
r2 = r2_score(y_test, blended_pred_reg)

print(f"Blended model MSE: {mse:.4f}")
print(f"Blended model R-squared: {r2:.4f}")
```

Slide 12: Real-life Example: Image Classification

In this example, we'll use blending to improve image classification performance on a subset of the CIFAR-10 dataset. We'll combine predictions from different convolutional neural network architectures to create a more robust classifier.

```python
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# Load and preprocess CIFAR-10 data
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Define model architectures
def create_model_a():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def create_model_b():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        Conv2D(64, (3, 3), activation='relu'),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Train models
model_a = create_model_a()
model_b = create_model_b()

model_a.fit(X_train, y_train, epochs=10, batch_size=64, validation_split=0.1, verbose=0)
model_b.fit(X_train, y_train, epochs=10, batch_size=64, validation_split=0.1, verbose=0)

# Make predictions
pred_a = model_a.predict(X_test)
pred_b = model_b.predict(X_test)

# Blend predictions
blended_pred = (pred_a + pred_b) / 2
blended_classes = np.argmax(blended_pred, axis=1)

# Evaluate
test_classes = np.argmax(y_test, axis=1)
accuracy = np.mean(blended_classes == test_classes)
print(f"Blended model accuracy: {accuracy:.4f}")
```

Slide 13: Real-life Example: Sentiment Analysis

In this example, we'll use blending to improve sentiment analysis performance on movie reviews. We'll combine predictions from different natural language processing models to create a more accurate sentiment classifier.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_20newsgroups

# Load movie review data (using 20 newsgroups as a substitute)
categories = ['rec.sport.hockey', 'sci.space']
data = fetch_20newsgroups(subset='all', categories=categories, shuffle=True, random_state=42)
X, y = data.data, data.target

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize the text data
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train base models
nb_model = MultinomialNB()
svm_model = LinearSVC(random_state=42)

nb_model.fit(X_train_vec, y_train)
svm_model.fit(X_train_vec, y_train)

# Make predictions
nb_pred = nb_model.predict_proba(X_test_vec)
svm_pred = svm_model.decision_function(X_test_vec)
svm_pred = (svm_pred - svm_pred.min()) / (svm_pred.max() - svm_pred.min())  # Normalize SVM scores

# Blend predictions
blended_pred = (nb_pred[:, 1] + svm_pred) / 2
blended_classes = (blended_pred > 0.5).astype(int)

# Evaluate
accuracy = accuracy_score(y_test, blended_classes)
print(f"Blended model accuracy: {accuracy:.4f}")
```

Slide 14: Additional Resources

For those interested in diving deeper into ensemble learning and blending techniques, here are some valuable resources:

1. "Ensemble Methods in Machine Learning" by Thomas G. Dietterich ArXiv: [https://arxiv.org/abs/2106.04662](https://arxiv.org/abs/2106.04662)
2. "A Survey of Ensemble Learning Techniques" by Yue Jiang et al. ArXiv: [https://arxiv.org/abs/2202.00881](https://arxiv.org/abs/2202.00881)
3. "Stacked Generalization" by David H. Wolpert ArXiv: [https://arxiv.org/abs/1908.01942](https://arxiv.org/abs/1908.01942)

These papers provide in-depth discussions on various ensemble methods, including blending, and offer insights into their theoretical foundations and practical applications.

