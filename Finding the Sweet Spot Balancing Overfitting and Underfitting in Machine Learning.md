## Finding the Sweet Spot Balancing Overfitting and Underfitting in Machine Learning

Slide 1: Understanding Model Fitting in Machine Learning

Model fitting in machine learning is about finding the right balance between capturing patterns in data and avoiding excessive specialization. This balance is crucial for creating models that perform well on unseen data. Let's explore the concepts of overfitting, underfitting, and best fitting using Python examples.

```python
import matplotlib.pyplot as plt
import numpy as np

# Generate sample data
np.random.seed(42)
X = np.linspace(0, 10, 100)
y = 3 * X + 2 + np.random.normal(0, 2, 100)

# Plot the data
plt.scatter(X, y, color='blue', label='Data')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Sample Data for Model Fitting')
plt.legend()
plt.show()
```

Slide 2: Overfitting

Overfitting occurs when a model becomes too specialized in the training data, learning not just the patterns but also the noise. This leads to excellent performance on training data but poor generalization to new data.

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Create and fit an overfitted model (high-degree polynomial)
poly_features = PolynomialFeatures(degree=15)
X_poly = poly_features.fit_transform(X.reshape(-1, 1))
model_overfit = LinearRegression()
model_overfit.fit(X_poly, y)

# Generate predictions
X_pred = np.linspace(0, 10, 100).reshape(-1, 1)
X_pred_poly = poly_features.transform(X_pred)
y_pred_overfit = model_overfit.predict(X_pred_poly)

# Plot the results
plt.scatter(X, y, color='blue', label='Data')
plt.plot(X_pred, y_pred_overfit, color='red', label='Overfitted Model')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Overfitting Example')
plt.legend()
plt.show()

print(f"Mean Squared Error: {mean_squared_error(y, model_overfit.predict(X_poly)):.2f}")
```

Slide 3: Underfitting

Underfitting occurs when the model is too simple and doesn't capture the important patterns in the data. It results in poor performance on both training and test data because the model hasn't learned enough from the data.

```python
# Create and fit an underfitted model (simple linear regression)
model_underfit = LinearRegression()
model_underfit.fit(X.reshape(-1, 1), y)

# Generate predictions
y_pred_underfit = model_underfit.predict(X_pred)

# Plot the results
plt.scatter(X, y, color='blue', label='Data')
plt.plot(X_pred, y_pred_underfit, color='green', label='Underfitted Model')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Underfitting Example')
plt.legend()
plt.show()

print(f"Mean Squared Error: {mean_squared_error(y, model_underfit.predict(X.reshape(-1, 1))):.2f}")
```

Slide 4: Best Fitting

Best fitting is the sweet spot where the model captures the patterns in the data without overfitting or underfitting, leading to good performance on unseen data. Achieving this ensures the model can generalize well while still learning effectively from the training data.

```python
# Create and fit a well-balanced model (polynomial of degree 3)
poly_features_best = PolynomialFeatures(degree=3)
X_poly_best = poly_features_best.fit_transform(X.reshape(-1, 1))
model_best = LinearRegression()
model_best.fit(X_poly_best, y)

# Generate predictions
X_pred_poly_best = poly_features_best.transform(X_pred)
y_pred_best = model_best.predict(X_pred_poly_best)

# Plot the results
plt.scatter(X, y, color='blue', label='Data')
plt.plot(X_pred, y_pred_best, color='purple', label='Best Fitting Model')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Best Fitting Example')
plt.legend()
plt.show()

print(f"Mean Squared Error: {mean_squared_error(y, model_best.predict(X_poly_best)):.2f}")
```

Slide 5: Comparing Model Performance

Let's compare the performance of our three models: overfitted, underfitted, and best fitting. We'll use the Mean Squared Error (MSE) as our metric.

```python
# Calculate MSE for all models
mse_overfit = mean_squared_error(y, model_overfit.predict(X_poly))
mse_underfit = mean_squared_error(y, model_underfit.predict(X.reshape(-1, 1)))
mse_best = mean_squared_error(y, model_best.predict(X_poly_best))

# Create a bar plot to compare MSE
models = ['Overfitted', 'Underfitted', 'Best Fitting']
mse_values = [mse_overfit, mse_underfit, mse_best]

plt.figure(figsize=(10, 6))
plt.bar(models, mse_values, color=['red', 'green', 'purple'])
plt.ylabel('Mean Squared Error')
plt.title('Model Performance Comparison')
for i, v in enumerate(mse_values):
    plt.text(i, v + 0.1, f'{v:.2f}', ha='center')
plt.show()
```

Slide 6: Identifying Overfitting

Overfitting can be identified by comparing the model's performance on training and validation data. A large gap between training and validation performance is a sign of overfitting.

```python
from sklearn.model_selection import train_test_split

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Function to train and evaluate models
def train_and_evaluate(degree):
    poly_features = PolynomialFeatures(degree=degree)
    X_poly_train = poly_features.fit_transform(X_train.reshape(-1, 1))
    X_poly_val = poly_features.transform(X_val.reshape(-1, 1))
    
    model = LinearRegression()
    model.fit(X_poly_train, y_train)
    
    train_mse = mean_squared_error(y_train, model.predict(X_poly_train))
    val_mse = mean_squared_error(y_val, model.predict(X_poly_val))
    
    return train_mse, val_mse

# Train models with different polynomial degrees
degrees = range(1, 16)
train_mse = []
val_mse = []

for degree in degrees:
    train, val = train_and_evaluate(degree)
    train_mse.append(train)
    val_mse.append(val)

# Plot the results
plt.plot(degrees, train_mse, label='Training MSE')
plt.plot(degrees, val_mse, label='Validation MSE')
plt.xlabel('Polynomial Degree')
plt.ylabel('Mean Squared Error')
plt.title('Training vs Validation Performance')
plt.legend()
plt.show()
```

Slide 7: Techniques to Prevent Overfitting

There are several techniques to prevent overfitting. Let's explore regularization, which adds a penalty term to the loss function to discourage complex models.

```python
from sklearn.linear_model import Ridge

# Function to train and evaluate Ridge models
def train_and_evaluate_ridge(degree, alpha):
    poly_features = PolynomialFeatures(degree=degree)
    X_poly_train = poly_features.fit_transform(X_train.reshape(-1, 1))
    X_poly_val = poly_features.transform(X_val.reshape(-1, 1))
    
    model = Ridge(alpha=alpha)
    model.fit(X_poly_train, y_train)
    
    train_mse = mean_squared_error(y_train, model.predict(X_poly_train))
    val_mse = mean_squared_error(y_val, model.predict(X_poly_val))
    
    return train_mse, val_mse

# Train models with different regularization strengths
alphas = [0, 0.1, 1, 10, 100]
degree = 15  # High degree polynomial

for alpha in alphas:
    train_mse, val_mse = train_and_evaluate_ridge(degree, alpha)
    print(f"Alpha: {alpha}")
    print(f"Training MSE: {train_mse:.2f}")
    print(f"Validation MSE: {val_mse:.2f}")
    print()
```

Slide 8: Cross-validation for Model Selection

Cross-validation is a powerful technique for assessing model performance and selecting the best model. It helps in finding the right balance between underfitting and overfitting.

```python
from sklearn.model_selection import cross_val_score

# Function to perform cross-validation
def cross_validate_model(degree):
    poly_features = PolynomialFeatures(degree=degree)
    X_poly = poly_features.fit_transform(X.reshape(-1, 1))
    
    model = LinearRegression()
    scores = cross_val_score(model, X_poly, y, cv=5, scoring='neg_mean_squared_error')
    return -scores.mean()

# Perform cross-validation for different polynomial degrees
degrees = range(1, 16)
cv_scores = [cross_validate_model(degree) for degree in degrees]

# Plot the results
plt.plot(degrees, cv_scores, marker='o')
plt.xlabel('Polynomial Degree')
plt.ylabel('Cross-validation MSE')
plt.title('Cross-validation Scores for Different Model Complexities')
plt.show()

best_degree = degrees[cv_scores.index(min(cv_scores))]
print(f"Best polynomial degree according to cross-validation: {best_degree}")
```

Slide 9: Learning Curves

Learning curves show how the model's performance changes as the amount of training data increases. They can help identify if the model is overfitting, underfitting, or well-fitted.

```python
from sklearn.model_selection import learning_curve

# Function to plot learning curves
def plot_learning_curve(degree):
    poly_features = PolynomialFeatures(degree=degree)
    X_poly = poly_features.fit_transform(X.reshape(-1, 1))
    
    model = LinearRegression()
    
    train_sizes, train_scores, val_scores = learning_curve(
        model, X_poly, y, train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='neg_mean_squared_error', cv=5
    )
    
    train_scores_mean = -train_scores.mean(axis=1)
    val_scores_mean = -val_scores.mean(axis=1)
    
    plt.plot(train_sizes, train_scores_mean, label='Training MSE')
    plt.plot(train_sizes, val_scores_mean, label='Validation MSE')
    plt.xlabel('Training Set Size')
    plt.ylabel('Mean Squared Error')
    plt.title(f'Learning Curves (Polynomial Degree {degree})')
    plt.legend()
    plt.show()

# Plot learning curves for different model complexities
plot_learning_curve(1)  # Underfitting
plot_learning_curve(3)  # Good fit
plot_learning_curve(15)  # Overfitting
```

Slide 10: Feature Selection

Feature selection is another technique to prevent overfitting by choosing the most relevant features for the model. Let's use recursive feature elimination (RFE) as an example.

```python
from sklearn.feature_selection import RFE
from sklearn.preprocessing import PolynomialFeatures

# Generate more features
poly_features = PolynomialFeatures(degree=5)
X_poly = poly_features.fit_transform(X.reshape(-1, 1))

# Perform recursive feature elimination
model = LinearRegression()
rfe = RFE(estimator=model, n_features_to_select=3)
X_rfe = rfe.fit_transform(X_poly, y)

# Train a model with selected features
model.fit(X_rfe, y)

# Print selected features
feature_names = poly_features.get_feature_names_out()
selected_features = feature_names[rfe.support_]
print("Selected features:")
for feature in selected_features:
    print(feature)

# Evaluate the model
y_pred = model.predict(X_rfe)
mse = mean_squared_error(y, y_pred)
print(f"\nMean Squared Error with selected features: {mse:.2f}")
```

Slide 11: Early Stopping

Early stopping is a technique to prevent overfitting by stopping the training process when the model's performance on a validation set starts to degrade. Let's implement a simple version of early stopping.

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor

# Prepare the data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.reshape(-1, 1))
X_val_scaled = scaler.transform(X_val.reshape(-1, 1))

# Initialize the model
model = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)

# Training loop with early stopping
best_val_loss = float('inf')
patience = 10
no_improve = 0
train_losses = []
val_losses = []

for epoch in range(1000):
    model.partial_fit(X_train_scaled, y_train)
    
    train_loss = mean_squared_error(y_train, model.predict(X_train_scaled))
    val_loss = mean_squared_error(y_val, model.predict(X_val_scaled))
    
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        no_improve = 0
    else:
        no_improve += 1
    
    if no_improve >= patience:
        print(f"Early stopping at epoch {epoch}")
        break

# Plot the learning curves
plt.plot(train_losses, label='Training MSE')
plt.plot(val_losses, label='Validation MSE')
plt.xlabel('Epochs')
plt.ylabel('Mean Squared Error')
plt.title('Learning Curves with Early Stopping')
plt.legend()
plt.show()
```

Slide 12: Real-life Example: House Price Prediction

Let's apply our understanding of model fitting to a real-world scenario: predicting house prices based on various features.

```python
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score

# Load and prepare the data
housing = fetch_california_housing()
X, y = housing.data, housing.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train and evaluate models
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression (alpha=1)": Ridge(alpha=1),
    "Ridge Regression (alpha=10)": Ridge(alpha=10)
}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"{name}:")
    print(f"  MSE: {mse:.4f}")
    print(f"  R2 Score: {r2:.4f}\n")
```

Slide 13: Visualizing Model Performance in House Price Prediction

Let's visualize the performance of our models on the house price prediction task.

```python
import matplotlib.pyplot as plt
import numpy as np

# Prepare data for visualization
model_names = list(models.keys())
mse_scores = []
r2_scores = []

for model in models.values():
    y_pred = model.predict(X_test_scaled)
    mse_scores.append(mean_squared_error(y_test, y_pred))
    r2_scores.append(r2_score(y_test, y_pred))

# Create bar plots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

ax1.bar(model_names, mse_scores)
ax1.set_title('Mean Squared Error')
ax1.set_ylabel('MSE')
ax1.set_xticklabels(model_names, rotation=45, ha='right')

ax2.bar(model_names, r2_scores)
ax2.set_title('R2 Score')
ax2.set_ylabel('R2')
ax2.set_xticklabels(model_names, rotation=45, ha='right')

plt.tight_layout()
plt.show()
```

Slide 14: Interpreting Results and Model Selection

The visualization helps us compare the performance of different models. Lower MSE and higher R2 scores indicate better performance. We can observe how regularization (Ridge regression) affects the model's performance compared to simple linear regression.

```python
# Find the best performing model
best_model_index = np.argmin(mse_scores)
best_model_name = model_names[best_model_index]
best_mse = mse_scores[best_model_index]
best_r2 = r2_scores[best_model_index]

print(f"Best performing model: {best_model_name}")
print(f"MSE: {best_mse:.4f}")
print(f"R2 Score: {best_r2:.4f}")

# Compare feature importances for the best model
best_model = models[best_model_name]
feature_importance = np.abs(best_model.coef_)
feature_names = housing.feature_names

# Plot feature importances
plt.figure(figsize=(10, 6))
plt.bar(feature_names, feature_importance)
plt.title(f"Feature Importances for {best_model_name}")
plt.xlabel("Features")
plt.ylabel("Absolute Coefficient Value")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
```

Slide 15: Additional Resources

For more in-depth information on model fitting and machine learning techniques, consider exploring these resources:

1.  "Understanding Machine Learning: From Theory to Algorithms" by Shai Shalev-Shwartz and Shai Ben-David ([https://arxiv.org/abs/1406.0923](https://arxiv.org/abs/1406.0923))
2.  "A Few Useful Things to Know About Machine Learning" by Pedro Domingos ([https://arxiv.org/abs/1206.5533](https://arxiv.org/abs/1206.5533))
3.  "Regularization and variable selection via the elastic net" by Hui Zou and Trevor Hastie ([https://arxiv.org/abs/1108.5017](https://arxiv.org/abs/1108.5017))

These papers provide comprehensive insights into machine learning concepts, including model fitting, regularization, and feature selection.

