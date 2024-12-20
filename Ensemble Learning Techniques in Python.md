## Ensemble Learning Techniques in Python
Slide 1: Introduction to Ensemble Learning in Sequence

Ensemble learning is a powerful technique that combines multiple models to improve prediction accuracy and robustness. In this presentation, we'll focus on sequential ensemble methods using Python, where models are trained one after another, each learning from the errors of its predecessors.

```python
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Generate sample data
X = np.random.rand(1000, 5)
y = np.sum(X, axis=1) + np.random.normal(0, 0.1, 1000)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train a Gradient Boosting model
gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
gb_model.fit(X_train, y_train)

# Make predictions and calculate error
y_pred = gb_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.4f}")
```

Slide 2: Gradient Boosting: The Sequential Ensemble Champion

Gradient Boosting is a popular sequential ensemble method that builds weak learners (typically decision trees) in a stage-wise manner. Each new model focuses on correcting the errors made by the combined ensemble of previous models.

```python
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

class SimpleGradientBoosting:
    def __init__(self, n_estimators=100, learning_rate=0.1):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.models = []

    def fit(self, X, y):
        current_predictions = np.zeros_like(y)
        for _ in range(self.n_estimators):
            residuals = y - current_predictions
            model = DecisionTreeRegressor(max_depth=3)
            model.fit(X, residuals)
            self.models.append(model)
            current_predictions += self.learning_rate * model.predict(X)

    def predict(self, X):
        return sum(self.learning_rate * model.predict(X) for model in self.models)

# Use the SimpleGradientBoosting class
sgb = SimpleGradientBoosting()
sgb.fit(X_train, y_train)
y_pred_sgb = sgb.predict(X_test)
mse_sgb = mean_squared_error(y_test, y_pred_sgb)
print(f"Simple Gradient Boosting MSE: {mse_sgb:.4f}")
```

Slide 3: AdaBoost: Adaptive Boosting in Sequence

AdaBoost, short for Adaptive Boosting, is another sequential ensemble method. It works by giving more weight to misclassified instances in subsequent iterations, allowing the model to focus on harder examples.

```python
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor

# Create base estimator
base_estimator = DecisionTreeRegressor(max_depth=3)

# Create AdaBoost model
adaboost = AdaBoostRegressor(base_estimator=base_estimator, n_estimators=50, random_state=42)

# Fit the model
adaboost.fit(X_train, y_train)

# Make predictions
y_pred_ada = adaboost.predict(X_test)

# Calculate error
mse_ada = mean_squared_error(y_test, y_pred_ada)
print(f"AdaBoost MSE: {mse_ada:.4f}")
```

Slide 4: Stacking: Layering Models for Better Performance

Stacking is a method where multiple first-level models are trained independently, and their predictions are used as inputs to a second-level model (meta-learner) to make the final prediction.

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_predict

# First-level models
model1 = RandomForestRegressor(n_estimators=100, random_state=42)
model2 = GradientBoostingRegressor(n_estimators=100, random_state=42)

# Generate predictions from first-level models
pred1 = cross_val_predict(model1, X_train, y_train, cv=5)
pred2 = cross_val_predict(model2, X_train, y_train, cv=5)

# Prepare input for meta-learner
stacking_train = np.column_stack((pred1, pred2))

# Train meta-learner
meta_learner = LinearRegression()
meta_learner.fit(stacking_train, y_train)

# Make final predictions
model1.fit(X_train, y_train)
model2.fit(X_train, y_train)
stacking_test = np.column_stack((model1.predict(X_test), model2.predict(X_test)))
y_pred_stacking = meta_learner.predict(stacking_test)

mse_stacking = mean_squared_error(y_test, y_pred_stacking)
print(f"Stacking MSE: {mse_stacking:.4f}")
```

Slide 5: Bagging: Bootstrap Aggregating

While not strictly sequential, Bagging is worth mentioning as it's often used in combination with sequential methods. It involves training multiple instances of the same model on different subsets of the data and aggregating their predictions.

```python
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor

# Create base estimator
base_estimator = DecisionTreeRegressor(max_depth=3)

# Create Bagging model
bagging = BaggingRegressor(base_estimator=base_estimator, n_estimators=50, random_state=42)

# Fit the model
bagging.fit(X_train, y_train)

# Make predictions
y_pred_bagging = bagging.predict(X_test)

# Calculate error
mse_bagging = mean_squared_error(y_test, y_pred_bagging)
print(f"Bagging MSE: {mse_bagging:.4f}")
```

Slide 6: XGBoost: Extreme Gradient Boosting

XGBoost is an optimized implementation of gradient boosting that offers improved performance and speed. It uses a more regularized model formalization to control overfitting.

```python
from xgboost import XGBRegressor

# Create XGBoost model
xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)

# Fit the model
xgb_model.fit(X_train, y_train)

# Make predictions
y_pred_xgb = xgb_model.predict(X_test)

# Calculate error
mse_xgb = mean_squared_error(y_test, y_pred_xgb)
print(f"XGBoost MSE: {mse_xgb:.4f}")
```

Slide 7: LightGBM: Light Gradient Boosting Machine

LightGBM is another gradient boosting framework that uses tree-based learning algorithms. It's designed for distributed and efficient training, making it suitable for large datasets.

```python
from lightgbm import LGBMRegressor

# Create LightGBM model
lgbm_model = LGBMRegressor(n_estimators=100, learning_rate=0.1, random_state=42)

# Fit the model
lgbm_model.fit(X_train, y_train)

# Make predictions
y_pred_lgbm = lgbm_model.predict(X_test)

# Calculate error
mse_lgbm = mean_squared_error(y_test, y_pred_lgbm)
print(f"LightGBM MSE: {mse_lgbm:.4f}")
```

Slide 8: CatBoost: Handling Categorical Features

CatBoost is a gradient boosting library that efficiently handles categorical features. It uses a symmetric tree structure and ordered boosting to improve accuracy and reduce overfitting.

```python
from catboost import CatBoostRegressor

# Create CatBoost model
catboost_model = CatBoostRegressor(iterations=100, learning_rate=0.1, random_state=42)

# Fit the model
catboost_model.fit(X_train, y_train, verbose=False)

# Make predictions
y_pred_catboost = catboost_model.predict(X_test)

# Calculate error
mse_catboost = mean_squared_error(y_test, y_pred_catboost)
print(f"CatBoost MSE: {mse_catboost:.4f}")
```

Slide 9: Sequential Model-Based Optimization (SMBO)

SMBO is a technique used in hyperparameter optimization. It sequentially proposes new hyperparameter configurations based on the performance of previous configurations.

```python
from skopt import BayesSearchCV
from sklearn.model_selection import cross_val_score

# Define the search space
search_space = {
    'n_estimators': (10, 200),
    'learning_rate': (0.01, 0.3, 'log-uniform'),
    'max_depth': (3, 10)
}

# Create the BayesSearchCV object
bayes_search = BayesSearchCV(
    GradientBoostingRegressor(random_state=42),
    search_space,
    n_iter=50,
    cv=3,
    random_state=42
)

# Perform the search
bayes_search.fit(X_train, y_train)

# Print the best parameters and score
print("Best parameters:", bayes_search.best_params_)
print("Best cross-validation score:", bayes_search.best_score_)

# Evaluate on test set
best_model = bayes_search.best_estimator_
y_pred_best = best_model.predict(X_test)
mse_best = mean_squared_error(y_test, y_pred_best)
print(f"Best Model MSE: {mse_best:.4f}")
```

Slide 10: Real-Life Example: Weather Prediction

Weather prediction is a complex task that benefits from ensemble methods. We'll use a simplified example to demonstrate how ensemble methods can improve accuracy in predicting temperature.

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression

# Generate synthetic weather data
np.random.seed(42)
dates = pd.date_range(start='2020-01-01', end='2022-12-31', freq='D')
temp = 20 + 10 * np.sin(np.arange(len(dates)) * 2 * np.pi / 365) + np.random.normal(0, 3, len(dates))
humidity = 60 + 20 * np.sin(np.arange(len(dates)) * 2 * np.pi / 365 + np.pi/2) + np.random.normal(0, 5, len(dates))
wind_speed = 5 + 3 * np.sin(np.arange(len(dates)) * 2 * np.pi / 365 + np.pi/4) + np.random.normal(0, 1, len(dates))

df = pd.DataFrame({
    'date': dates,
    'temp': temp,
    'humidity': humidity,
    'wind_speed': wind_speed
})

# Feature engineering
df['day_of_year'] = df['date'].dt.dayofyear
df['month'] = df['date'].dt.month

# Prepare data for modeling
X = df[['day_of_year', 'month', 'humidity', 'wind_speed']]
y = df['temp']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train individual models
rf = RandomForestRegressor(n_estimators=100, random_state=42)
gb = GradientBoostingRegressor(n_estimators=100, random_state=42)

rf.fit(X_train_scaled, y_train)
gb.fit(X_train_scaled, y_train)

# Make predictions
y_pred_rf = rf.predict(X_test_scaled)
y_pred_gb = gb.predict(X_test_scaled)

# Combine predictions for stacking
stacking_features = np.column_stack((y_pred_rf, y_pred_gb))

# Train meta-learner
meta_learner = LinearRegression()
meta_learner.fit(stacking_features, y_test)

# Make final prediction
y_pred_stacking = meta_learner.predict(stacking_features)

# Calculate errors
mae_rf = mean_absolute_error(y_test, y_pred_rf)
mae_gb = mean_absolute_error(y_test, y_pred_gb)
mae_stacking = mean_absolute_error(y_test, y_pred_stacking)

print(f"Random Forest MAE: {mae_rf:.2f}")
print(f"Gradient Boosting MAE: {mae_gb:.2f}")
print(f"Stacking MAE: {mae_stacking:.2f}")
```

Slide 11: Real-Life Example: Image Classification Ensemble

Image classification is another area where ensemble methods excel. We'll create a simple ensemble of convolutional neural networks for image classification using the CIFAR-10 dataset.

```python
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical

# Load and preprocess data
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
X_train, X_test = X_train / 255.0, X_test / 255.0
y_train, y_test = to_categorical(y_train, 10), to_categorical(y_test, 10)

def create_model():
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

# Train multiple models
models = [create_model() for _ in range(3)]
for i, model in enumerate(models):
    print(f"Training model {i+1}")
    model.fit(X_train, y_train, epochs=5, batch_size=64, validation_split=0.2, verbose=0)

# Make ensemble predictions
ensemble_predictions = [model.predict(X_test) for model in models]
avg_predictions = np.mean(ensemble_predictions, axis=0)
ensemble_accuracy = np.mean(np.argmax(avg_predictions, axis=1) == np.argmax(y_test, axis=1))

print(f"Ensemble Accuracy: {ensemble_accuracy:.4f}")

# Compare with individual model accuracies
for i, model in enumerate(models):
    _, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Model {i+1} Accuracy: {accuracy:.4f}")
```

Slide 12: Boosting vs Bagging: Understanding the Differences

Boosting and Bagging are two fundamental ensemble techniques with distinct approaches. Boosting builds models sequentially, focusing on difficult examples, while Bagging creates independent models and combines their predictions.

```python
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Generate synthetic data
X, y = make_regression(n_samples=1000, n_features=20, noise=0.1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest (Bagging)
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_mse = mean_squared_error(y_test, rf_pred)

# Gradient Boosting
gb = GradientBoostingRegressor(n_estimators=100, random_state=42)
gb.fit(X_train, y_train)
gb_pred = gb.predict(X_test)
gb_mse = mean_squared_error(y_test, gb_pred)

print(f"Random Forest MSE: {rf_mse:.4f}")
print(f"Gradient Boosting MSE: {gb_mse:.4f}")

# Plot feature importances
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.bar(range(20), rf.feature_importances_, alpha=0.5, label='Random Forest')
plt.bar(range(20), gb.feature_importances_, alpha=0.5, label='Gradient Boosting')
plt.xlabel('Feature Index')
plt.ylabel('Feature Importance')
plt.legend()
plt.title('Feature Importances: Random Forest vs Gradient Boosting')
plt.show()
```

Slide 13: Ensemble Method Hyperparameter Tuning

Hyperparameter tuning is crucial for optimizing ensemble methods. We'll use RandomizedSearchCV to tune a Gradient Boosting Regressor.

```python
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from scipy.stats import randint, uniform

# Define the parameter space
param_dist = {
    'n_estimators': randint(50, 500),
    'max_depth': randint(1, 20),
    'learning_rate': uniform(0.01, 0.5),
    'subsample': uniform(0.5, 0.5),
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 10)
}

# Create the base model
gb = GradientBoostingRegressor(random_state=42)

# Set up RandomizedSearchCV
random_search = RandomizedSearchCV(
    gb, param_distributions=param_dist, n_iter=100, cv=5, 
    scoring='neg_mean_squared_error', random_state=42, n_jobs=-1
)

# Perform the random search
random_search.fit(X_train, y_train)

# Print the best parameters and score
print("Best parameters:", random_search.best_params_)
print("Best MSE:", -random_search.best_score_)

# Evaluate the best model on the test set
best_gb = random_search.best_estimator_
best_gb_pred = best_gb.predict(X_test)
best_gb_mse = mean_squared_error(y_test, best_gb_pred)
print(f"Best Gradient Boosting MSE on test set: {best_gb_mse:.4f}")
```

Slide 14: Ensemble Method Visualization

Visualizing the decision boundaries of ensemble methods can provide insights into their behavior. We'll create a simple visualization for a binary classification problem.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

# Generate moon-shaped data
X, y = make_moons(n_samples=1000, noise=0.3, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest and Gradient Boosting
rf = RandomForestClassifier(n_estimators=100, random_state=42)
gb = GradientBoostingClassifier(n_estimators=100, random_state=42)

rf.fit(X_train, y_train)
gb.fit(X_train, y_train)

# Create a mesh grid
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

# Plot decision boundaries
plt.figure(figsize=(12, 5))

for i, clf in enumerate((rf, gb)):
    plt.subplot(1, 2, i + 1)
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8)
    plt.title(clf.__class__.__name__)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")

plt.tight_layout()
plt.show()
```

Slide 15: Additional Resources

For those interested in diving deeper into ensemble methods and their sequential implementations, here are some valuable resources:

1. "Ensemble Methods: Foundations and Algorithms" by Zhi-Hua Zhou ArXiv: [https://arxiv.org/abs/1404.4502](https://arxiv.org/abs/1404.4502)
2. "XGBoost: A Scalable Tree Boosting System" by Tianqi Chen and Carlos Guestrin ArXiv: [https://arxiv.org/abs/1603.02754](https://arxiv.org/abs/1603.02754)
3. "LightGBM: A Highly Efficient Gradient Boosting Decision Tree" by Guolin Ke et al. ArXiv: [https://arxiv.org/abs/1703.09803](https://arxiv.org/abs/1703.09803)
4. "CatBoost: unbiased boosting with categorical features" by Liudmila Prokhorenkova et al. ArXiv: [https://arxiv.org/abs/1706.09516](https://arxiv.org/abs/1706.09516)

These papers provide in-depth explanations of various ensemble methods and their implementations, offering valuable insights for both beginners and advanced practitioners in the field of machine learning.

