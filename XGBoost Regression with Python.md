## XGBoost Regression with Python
Slide 1: Introduction to XGBoost Regression

XGBoost (Extreme Gradient Boosting) is a powerful machine learning algorithm for regression tasks. It's an optimized implementation of gradient boosting that offers high performance and accuracy.

```python
import xgboost as xgb
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

# Generate sample regression data
X, y = make_regression(n_samples=1000, n_features=10, noise=0.1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create XGBoost regressor
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)
```

Slide 2: XGBoost Features

XGBoost offers several advantages, including regularization, handling missing values, and parallel processing. It uses decision tree ensembles and gradient boosting to create a robust model.

```python
# XGBoost with custom parameters
xgb_model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8
)

# Train the model
xgb_model.fit(X_train, y_train)
```

Slide 3: Data Preparation

Before training an XGBoost model, it's crucial to prepare your data properly. This includes handling missing values, encoding categorical variables, and scaling features if necessary.

```python
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load sample data
data = pd.read_csv('sample_data.csv')

# Handle missing values
data.fillna(data.mean(), inplace=True)

# Encode categorical variables
le = LabelEncoder()
data['category'] = le.fit_transform(data['category'])

# Scale numerical features
scaler = StandardScaler()
data[['feature1', 'feature2']] = scaler.fit_transform(data[['feature1', 'feature2']])
```

Slide 4: Training an XGBoost Regressor

Training an XGBoost regressor involves fitting the model to your training data. You can use various parameters to control the training process and model complexity.

```python
# Prepare features and target
X = data.drop('target', axis=1)
y = data['target']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1)
xgb_model.fit(X_train, y_train)
```

Slide 5: Making Predictions

Once your XGBoost model is trained, you can use it to make predictions on new data. This is useful for both evaluating the model and applying it to real-world problems.

```python
# Make predictions on test data
y_pred = xgb_model.predict(X_test)

# Calculate Mean Squared Error
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Make a single prediction
new_data = [[5.1, 3.5, 1.4, 0.2]]  # Example features
prediction = xgb_model.predict(new_data)
print(f"Prediction for new data: {prediction[0]}")
```

Slide 6: Feature Importance

XGBoost allows you to assess the importance of each feature in your model. This can help you understand which variables have the most impact on your predictions.

```python
import matplotlib.pyplot as plt

# Get feature importance
importance = xgb_model.feature_importances_
feature_names = X.columns

# Plot feature importance
plt.figure(figsize=(10, 6))
plt.bar(range(len(importance)), importance)
plt.xticks(range(len(importance)), feature_names, rotation=90)
plt.title('Feature Importance')
plt.tight_layout()
plt.show()
```

Slide 7: Hyperparameter Tuning

Optimizing XGBoost's hyperparameters can significantly improve model performance. Grid search and random search are common methods for finding the best parameter combinations.

```python
from sklearn.model_selection import GridSearchCV

# Define parameter grid
param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.3],
    'n_estimators': [100, 200, 300],
    'subsample': [0.8, 1.0]
}

# Perform grid search
grid_search = GridSearchCV(xgb.XGBRegressor(objective='reg:squarederror'), param_grid, cv=3, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

# Print best parameters
print("Best parameters:", grid_search.best_params_)
```

Slide 8: Cross-Validation

Cross-validation helps assess how well your XGBoost model generalizes to unseen data. It's particularly useful when you have limited data.

```python
from sklearn.model_selection import cross_val_score

# Perform 5-fold cross-validation
cv_scores = cross_val_score(xgb_model, X, y, cv=5, scoring='neg_mean_squared_error')

# Convert scores to positive values
cv_scores = -cv_scores

print("Cross-validation scores:", cv_scores)
print("Mean CV score:", cv_scores.mean())
print("Standard deviation of CV scores:", cv_scores.std())
```

Slide 9: Early Stopping

Early stopping can prevent overfitting by stopping the training process when the model's performance on a validation set stops improving.

```python
from sklearn.model_selection import train_test_split

# Split data into train, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Train with early stopping
xgb_model = xgb.XGBRegressor(n_estimators=1000)
xgb_model.fit(X_train, y_train, 
              eval_set=[(X_val, y_val)], 
              early_stopping_rounds=10, 
              eval_metric='rmse', 
              verbose=False)

print("Best iteration:", xgb_model.best_iteration)
```

Slide 10: Handling Imbalanced Data

When dealing with imbalanced regression data, you can use sample weights to give more importance to underrepresented samples.

```python
import numpy as np

# Generate sample weights (inverse of target frequency)
target_counts = np.bincount(np.digitize(y_train, bins=10))
sample_weights = 1 / target_counts[np.digitize(y_train, bins=10)]

# Normalize weights
sample_weights /= np.sum(sample_weights)

# Train with sample weights
xgb_model = xgb.XGBRegressor(objective='reg:squarederror')
xgb_model.fit(X_train, y_train, sample_weight=sample_weights)
```

Slide 11: Real-Life Example: House Price Prediction

Let's use XGBoost to predict house prices based on various features like square footage, number of bedrooms, and location.

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Load house price data
house_data = pd.read_csv('house_prices.csv')

# Prepare features and target
X = house_data.drop('price', axis=1)
y = house_data['price']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train XGBoost model
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)
xgb_model.fit(X_train, y_train)

# Make predictions and evaluate
y_pred = xgb_model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error: ${mae:.2f}")
```

Slide 12: Real-Life Example: Stock Price Prediction

In this example, we'll use XGBoost to predict stock prices based on historical data and technical indicators.

```python
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

# Load stock data
stock_data = pd.read_csv('stock_data.csv')

# Calculate technical indicators (e.g., Moving Average)
stock_data['MA_7'] = stock_data['Close'].rolling(window=7).mean()
stock_data['MA_21'] = stock_data['Close'].rolling(window=21).mean()

# Prepare features and target
X = stock_data[['Open', 'High', 'Low', 'Volume', 'MA_7', 'MA_21']].dropna()
y = stock_data['Close'].dropna()

# Scale features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
split = int(0.8 * len(X_scaled))
X_train, X_test = X_scaled[:split], X_scaled[split:]
y_train, y_test = y[:split], y[split:]

# Train XGBoost model
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)
xgb_model.fit(X_train, y_train)

# Make predictions and evaluate
y_pred = xgb_model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Root Mean Squared Error: ${rmse:.2f}")
```

Slide 13: Saving and Loading XGBoost Models

After training your XGBoost model, you can save it for future use without retraining. This is particularly useful for deploying models in production environments.

```python
import joblib

# Save the model
joblib.dump(xgb_model, 'xgboost_model.joblib')

# Load the model
loaded_model = joblib.load('xgboost_model.joblib')

# Use the loaded model for predictions
new_data = [[5.1, 3.5, 1.4, 0.2]]  # Example features
prediction = loaded_model.predict(new_data)
print(f"Prediction using loaded model: {prediction[0]}")
```

Slide 14: Visualizing the Decision Trees

XGBoost models consist of multiple decision trees. Visualizing these trees can provide insights into how the model makes predictions.

```python
from xgboost import plot_tree
import matplotlib.pyplot as plt

# Plot the first tree
plt.figure(figsize=(20, 10))
plot_tree(xgb_model, num_trees=0)
plt.title('First Decision Tree in XGBoost Model')
plt.show()

# Plot feature importance
xgb.plot_importance(xgb_model)
plt.title('Feature Importance in XGBoost Model')
plt.show()
```

Slide 15: Additional Resources

For further learning about XGBoost regression:

1. XGBoost Documentation: [https://xgboost.readthedocs.io/](https://xgboost.readthedocs.io/)
2. "XGBoost: A Scalable Tree Boosting System" by Chen and Guestrin (2016): [https://arxiv.org/abs/1603.02754](https://arxiv.org/abs/1603.02754)
3. "An End-to-End Guide to Understand the Math behind XGBoost" by Aniruddha Bhandari: [https://www.analyticsvidhya.com/blog/2018/09/an-end-to-end-guide-to-understand-the-math-behind-xgboost/](https://www.analyticsvidhya.com/blog/2018/09/an-end-to-end-guide-to-understand-the-math-behind-xgboost/)

