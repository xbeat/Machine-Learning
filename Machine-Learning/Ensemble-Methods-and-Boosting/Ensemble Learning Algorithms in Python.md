## Ensemble Learning Algorithms in Python
Slide 1: Introduction to Ensemble Learning

Ensemble learning is a powerful machine learning technique that combines multiple models to improve prediction accuracy and robustness. This approach leverages the diversity of different models to overcome individual weaknesses and produce superior results. In this presentation, we'll explore various ensemble learning algorithms and their implementation in Python.

```python
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Sample data
X = np.random.rand(1000, 10)
y = (X[:, 0] + X[:, 1] > 1).astype(int)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train individual models
dt = DecisionTreeClassifier(random_state=42)
rf = RandomForestClassifier(n_estimators=10, random_state=42)
gb = GradientBoostingClassifier(n_estimators=10, random_state=42)

models = [dt, rf, gb]
for model in models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"{model.__class__.__name__} Accuracy: {accuracy_score(y_test, y_pred):.4f}")
```

Slide 2: Bagging (Bootstrap Aggregating)

Bagging is an ensemble technique that creates multiple subsets of the original dataset through random sampling with replacement. It then trains a model on each subset and combines their predictions through voting or averaging. This method reduces overfitting and variance in the final model.

```python
from sklearn.ensemble import BaggingClassifier

# Create and train a bagging classifier
bagging = BaggingClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=10, random_state=42)
bagging.fit(X_train, y_train)

# Make predictions and calculate accuracy
y_pred_bagging = bagging.predict(X_test)
accuracy_bagging = accuracy_score(y_test, y_pred_bagging)
print(f"Bagging Classifier Accuracy: {accuracy_bagging:.4f}")
```

Slide 3: Random Forests

Random Forests are an extension of bagging that introduces additional randomness by selecting a random subset of features at each split in the decision trees. This technique further reduces correlation between trees and improves generalization.

```python
from sklearn.ensemble import RandomForestClassifier

# Create and train a random forest classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Make predictions and calculate accuracy
y_pred_rf = rf.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f"Random Forest Classifier Accuracy: {accuracy_rf:.4f}")

# Feature importance
importances = rf.feature_importances_
for i, importance in enumerate(importances):
    print(f"Feature {i}: {importance:.4f}")
```

Slide 4: Boosting

Boosting is an iterative ensemble technique that builds models sequentially, with each new model focusing on the mistakes of the previous ones. This approach creates a strong learner from multiple weak learners, effectively reducing bias and variance.

```python
from sklearn.ensemble import AdaBoostClassifier

# Create and train an AdaBoost classifier
adaboost = AdaBoostClassifier(n_estimators=50, random_state=42)
adaboost.fit(X_train, y_train)

# Make predictions and calculate accuracy
y_pred_ada = adaboost.predict(X_test)
accuracy_ada = accuracy_score(y_test, y_pred_ada)
print(f"AdaBoost Classifier Accuracy: {accuracy_ada:.4f}")
```

Slide 5: Gradient Boosting

Gradient Boosting is a powerful boosting algorithm that builds trees to minimize the loss function's gradient. It combines weak learners into a single strong learner in an iterative fashion, offering high predictive power and flexibility.

```python
from sklearn.ensemble import GradientBoostingClassifier

# Create and train a Gradient Boosting classifier
gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
gb.fit(X_train, y_train)

# Make predictions and calculate accuracy
y_pred_gb = gb.predict(X_test)
accuracy_gb = accuracy_score(y_test, y_pred_gb)
print(f"Gradient Boosting Classifier Accuracy: {accuracy_gb:.4f}")
```

Slide 6: XGBoost

XGBoost is an optimized distributed gradient boosting library designed for efficiency, flexibility, and portability. It implements machine learning algorithms under the Gradient Boosting framework, providing a highly efficient, flexible, and portable distributed gradient boosting library.

```python
import xgboost as xgb

# Create and train an XGBoost classifier
xgb_clf = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
xgb_clf.fit(X_train, y_train)

# Make predictions and calculate accuracy
y_pred_xgb = xgb_clf.predict(X_test)
accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
print(f"XGBoost Classifier Accuracy: {accuracy_xgb:.4f}")
```

Slide 7: LightGBM

LightGBM is a gradient boosting framework that uses tree-based learning algorithms. It is designed for distributed and efficient training and has faster training speed and higher efficiency than other boosting algorithms.

```python
import lightgbm as lgb

# Create and train a LightGBM classifier
lgb_train = lgb.Dataset(X_train, y_train)
params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9
}
lgb_clf = lgb.train(params, lgb_train, num_boost_round=100)

# Make predictions and calculate accuracy
y_pred_lgb = lgb_clf.predict(X_test)
y_pred_lgb = [1 if y > 0.5 else 0 for y in y_pred_lgb]
accuracy_lgb = accuracy_score(y_test, y_pred_lgb)
print(f"LightGBM Classifier Accuracy: {accuracy_lgb:.4f}")
```

Slide 8: Stacking

Stacking is an ensemble learning technique that combines predictions from multiple models using another model, called a meta-learner. This method can capture complex relationships between base models and often yields better performance than individual models.

```python
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression

# Define base models
base_models = [
    ('rf', RandomForestClassifier(n_estimators=10, random_state=42)),
    ('gb', GradientBoostingClassifier(n_estimators=10, random_state=42)),
    ('xgb', xgb.XGBClassifier(n_estimators=10, random_state=42))
]

# Create and train a stacking classifier
stacking = StackingClassifier(
    estimators=base_models,
    final_estimator=LogisticRegression(),
    cv=5
)
stacking.fit(X_train, y_train)

# Make predictions and calculate accuracy
y_pred_stacking = stacking.predict(X_test)
accuracy_stacking = accuracy_score(y_test, y_pred_stacking)
print(f"Stacking Classifier Accuracy: {accuracy_stacking:.4f}")
```

Slide 9: Voting

Voting is a simple yet effective ensemble method that combines predictions from multiple models. It can be implemented as hard voting (majority vote) or soft voting (weighted average of probabilities).

```python
from sklearn.ensemble import VotingClassifier

# Define base models
base_models = [
    ('rf', RandomForestClassifier(n_estimators=10, random_state=42)),
    ('gb', GradientBoostingClassifier(n_estimators=10, random_state=42)),
    ('xgb', xgb.XGBClassifier(n_estimators=10, random_state=42))
]

# Create and train a voting classifier
voting = VotingClassifier(estimators=base_models, voting='soft')
voting.fit(X_train, y_train)

# Make predictions and calculate accuracy
y_pred_voting = voting.predict(X_test)
accuracy_voting = accuracy_score(y_test, y_pred_voting)
print(f"Voting Classifier Accuracy: {accuracy_voting:.4f}")
```

Slide 10: Hyperparameter Tuning for Ensemble Models

Hyperparameter tuning is crucial for optimizing ensemble models. We'll use RandomizedSearchCV to efficiently search for the best hyperparameters for a Random Forest classifier.

```python
from sklearn.model_selection import RandomizedSearchCV

# Define hyperparameter search space
param_dist = {
    'n_estimators': [10, 50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Create a random forest classifier
rf = RandomForestClassifier(random_state=42)

# Perform randomized search
random_search = RandomizedSearchCV(rf, param_distributions=param_dist, n_iter=20, cv=5, random_state=42)
random_search.fit(X_train, y_train)

# Print best parameters and score
print("Best parameters:", random_search.best_params_)
print("Best score:", random_search.best_score_)
```

Slide 11: Feature Importance in Ensemble Models

Ensemble models can provide valuable insights into feature importance. We'll demonstrate how to extract and visualize feature importance from a Random Forest classifier.

```python
import matplotlib.pyplot as plt

# Train a random forest classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Get feature importances
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

# Plot feature importances
plt.figure(figsize=(10, 6))
plt.title("Feature Importances")
plt.bar(range(X.shape[1]), importances[indices])
plt.xticks(range(X.shape[1]), [f"Feature {i}" for i in indices], rotation=90)
plt.tight_layout()
plt.show()
```

Slide 12: Real-Life Example: Image Classification

Ensemble learning is widely used in image classification tasks. Let's use a simple ensemble of convolutional neural networks for classifying handwritten digits from the MNIST dataset.

```python
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical

# Load and preprocess MNIST data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32') / 255
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32') / 255
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Define a simple CNN model
def create_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(100, activation='relu'),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Train multiple models
models = [create_model() for _ in range(3)]
for i, model in enumerate(models):
    print(f"Training model {i+1}")
    model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0)

# Make ensemble predictions
ensemble_preds = np.mean([model.predict(X_test) for model in models], axis=0)
ensemble_accuracy = np.mean(np.argmax(ensemble_preds, axis=1) == np.argmax(y_test, axis=1))
print(f"Ensemble Accuracy: {ensemble_accuracy:.4f}")
```

Slide 13: Real-Life Example: Weather Prediction

Ensemble methods are frequently used in weather forecasting. Let's create a simple ensemble model to predict temperature based on various weather features.

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

# Generate synthetic weather data
np.random.seed(42)
n_samples = 1000
data = pd.DataFrame({
    'humidity': np.random.uniform(30, 100, n_samples),
    'wind_speed': np.random.uniform(0, 30, n_samples),
    'pressure': np.random.uniform(980, 1050, n_samples),
    'cloudiness': np.random.uniform(0, 100, n_samples)
})
data['temperature'] = (
    25 + 0.1 * data['humidity'] - 0.3 * data['wind_speed'] +
    0.02 * (data['pressure'] - 1015) - 0.1 * data['cloudiness'] +
    np.random.normal(0, 2, n_samples)
)

# Prepare data
X = data.drop('temperature', axis=1)
y = data['temperature']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train models
rf = RandomForestRegressor(n_estimators=100, random_state=42)
gb = GradientBoostingRegressor(n_estimators=100, random_state=42)

rf.fit(X_train_scaled, y_train)
gb.fit(X_train_scaled, y_train)

# Make predictions
rf_pred = rf.predict(X_test_scaled)
gb_pred = gb.predict(X_test_scaled)

# Ensemble prediction (simple average)
ensemble_pred = (rf_pred + gb_pred) / 2

# Calculate RMSE
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
gb_rmse = np.sqrt(mean_squared_error(y_test, gb_pred))
ensemble_rmse = np.sqrt(mean_squared_error(y_test, ensemble_pred))

print(f"Random Forest RMSE: {rf_rmse:.4f}")
print(f"Gradient Boosting RMSE: {gb_rmse:.4f}")
print(f"Ensemble RMSE: {ensemble_rmse:.4f}")
```

Slide 14: Additional Resources

For those interested in diving deeper into ensemble learning algorithms and their implementations in Python, here are some valuable resources:

1. "Ensemble Methods: Foundations and Algorithms" by Zhi-Hua Zhou (CRC Press, 2012)
2. "Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow" by Aurélien Géron (O'Reilly Media, 2019)
3. Sci



