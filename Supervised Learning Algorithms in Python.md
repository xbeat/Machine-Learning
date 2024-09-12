## Supervised Learning Algorithms in Python
Slide 1: Introduction to Supervised Learning

Supervised learning is a fundamental machine learning paradigm where algorithms learn from labeled training data to make predictions or decisions. This approach involves training a model on input-output pairs, allowing it to generalize patterns and apply them to new, unseen data.

```python
import numpy as np
from sklearn.model_selection import train_test_split

# Generate sample data
X = np.random.rand(100, 1)
y = 2 * X + 1 + np.random.randn(100, 1) * 0.1

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

Slide 2: Linear Regression

Linear regression is a simple yet powerful supervised learning algorithm used for predicting continuous values. It models the relationship between input features and the target variable as a linear function.

```python
from sklearn.linear_model import LinearRegression

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse:.4f}")
print(f"R-squared Score: {r2:.4f}")
```

Slide 3: Logistic Regression

Logistic regression is a classification algorithm used for predicting binary outcomes. It estimates the probability of an instance belonging to a particular class using the logistic function.

```python
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Generate binary classification data
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
```

Slide 4: Decision Trees

Decision trees are versatile algorithms used for both classification and regression tasks. They make decisions by splitting the data based on feature values, creating a tree-like structure.

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris

# Load iris dataset
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = DecisionTreeClassifier(max_depth=3)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
```

Slide 5: Random Forest

Random Forest is an ensemble learning method that combines multiple decision trees to improve prediction accuracy and reduce overfitting. It works by creating a forest of uncorrelated trees and aggregating their predictions.

```python
from sklearn.ensemble import RandomForestClassifier

# Create and train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Feature importance
importances = model.feature_importances_
for i, importance in enumerate(importances):
    print(f"Feature {i}: {importance:.4f}")
```

Slide 6: Support Vector Machines (SVM)

Support Vector Machines are powerful algorithms used for classification and regression tasks. They work by finding the optimal hyperplane that separates different classes in high-dimensional space.

```python
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train the model
model = SVC(kernel='rbf', C=1.0, random_state=42)
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
```

Slide 7: K-Nearest Neighbors (KNN)

K-Nearest Neighbors is a simple yet effective algorithm for classification and regression. It makes predictions based on the majority class or average value of the K nearest neighbors in the feature space.

```python
from sklearn.neighbors import KNeighborsClassifier

# Create and train the model
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
```

Slide 8: Naive Bayes

Naive Bayes is a probabilistic algorithm based on Bayes' theorem. It assumes independence between features and is particularly useful for text classification and spam filtering tasks.

```python
from sklearn.naive_bayes import GaussianNB

# Create and train the model
model = GaussianNB()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
```

Slide 9: Gradient Boosting

Gradient Boosting is an ensemble learning technique that combines weak learners (usually decision trees) to create a strong predictor. It builds models sequentially, with each new model correcting the errors of the previous ones.

```python
from sklearn.ensemble import GradientBoostingClassifier

# Create and train the model
model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
```

Slide 10: Neural Networks

Neural Networks are versatile algorithms inspired by the human brain. They consist of interconnected layers of neurons and can learn complex patterns in data through backpropagation.

```python
from sklearn.neural_network import MLPClassifier

# Create and train the model
model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
```

Slide 11: Cross-Validation

Cross-validation is a technique used to assess the performance and generalization ability of a model. It involves splitting the data into multiple subsets and training the model on different combinations of these subsets.

```python
from sklearn.model_selection import cross_val_score

# Perform 5-fold cross-validation
model = RandomForestClassifier(n_estimators=100, random_state=42)
cv_scores = cross_val_score(model, X, y, cv=5)

# Print cross-validation results
print(f"Cross-validation scores: {cv_scores}")
print(f"Mean CV score: {cv_scores.mean():.4f}")
print(f"Standard deviation: {cv_scores.std():.4f}")
```

Slide 12: Hyperparameter Tuning

Hyperparameter tuning is the process of finding the optimal set of hyperparameters for a machine learning model. Grid search and random search are common techniques for exploring the hyperparameter space.

```python
from sklearn.model_selection import GridSearchCV

# Define hyperparameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}

# Perform grid search
model = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Print best parameters and score
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
```

Slide 13: Real-Life Example: Spam Email Classification

Spam email classification is a common application of supervised learning. We'll use a Naive Bayes classifier to distinguish between spam and non-spam emails based on their content.

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# Sample email data (content, label)
emails = [
    ("Get rich quick! Buy now!", "spam"),
    ("Meeting scheduled for tomorrow", "ham"),
    ("Congratulations! You've won a prize", "spam"),
    ("Project update: new features implemented", "ham")
]

# Separate content and labels
X, y = zip(*emails)

# Create a pipeline with text vectorization and Naive Bayes classifier
pipeline = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('classifier', MultinomialNB())
])

# Train the model
pipeline.fit(X, y)

# Classify new emails
new_emails = ["Free money! Click here!", "Weekly team meeting agenda"]
predictions = pipeline.predict(new_emails)

for email, prediction in zip(new_emails, predictions):
    print(f"Email: {email}")
    print(f"Prediction: {prediction}\n")
```

Slide 14: Real-Life Example: House Price Prediction

House price prediction is a common regression task in real estate. We'll use a Random Forest Regressor to predict house prices based on various features.

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Sample house data (area, bedrooms, age, price)
houses = [
    (1500, 3, 10, 250000),
    (2000, 4, 5, 350000),
    (1200, 2, 15, 180000),
    (1800, 3, 8, 300000)
]

# Separate features and target
X, y = zip(*[(house[:-1], house[-1]) for house in houses])

# Create a preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), [0, 1, 2])
    ])

# Create a pipeline with preprocessing and Random Forest Regressor
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Train the model
pipeline.fit(X, y)

# Predict prices for new houses
new_houses = [(1600, 3, 12), (2200, 4, 3)]
predictions = pipeline.predict(new_houses)

for house, prediction in zip(new_houses, predictions):
    print(f"House: {house}")
    print(f"Predicted price: ${prediction:.2f}\n")
```

Slide 15: Additional Resources

For further exploration of supervised learning algorithms and their implementations in Python, consider the following resources:

1. Scikit-learn Documentation: [https://scikit-learn.org/stable/](https://scikit-learn.org/stable/)
2. "An Introduction to Statistical Learning" by James, Witten, Hastie, and Tibshirani
3. "Pattern Recognition and Machine Learning" by Christopher Bishop
4. ArXiv paper: "A Survey of Deep Learning Techniques for Neural Machine Translation" ([https://arxiv.org/abs/1703.03906](https://arxiv.org/abs/1703.03906))
5. ArXiv paper: "XGBoost: A Scalable Tree Boosting System" ([https://arxiv.org/abs/1603.02754](https://arxiv.org/abs/1603.02754))

These resources provide in-depth explanations, mathematical foundations, and advanced techniques in supervised learning.

