## Common Classification Algorithms in Python
Slide 1: Introduction to Classification Algorithms

Classification is a supervised learning technique where the model learns to categorize data into predefined classes. It's widely used in various fields, from spam detection to medical diagnosis. In this slideshow, we'll explore common classification algorithms and their implementation in Python.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification

# Generate a sample dataset
X, y = make_classification(n_samples=100, n_features=2, n_informative=2, 
                           n_redundant=0, n_clusters_per_class=1, random_state=42)

# Visualize the dataset
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis')
plt.title('Sample Classification Dataset')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.colorbar(label='Class')
plt.show()
```

Slide 2: Logistic Regression

Logistic Regression is a simple yet powerful algorithm for binary classification. It models the probability of an instance belonging to a particular class using the logistic function.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = model.score(X_test, y_test)
print(f"Accuracy: {accuracy:.2f}")
```

Slide 3: Decision Trees

Decision Trees are versatile algorithms that can handle both classification and regression tasks. They make decisions by splitting the data based on feature values, creating a tree-like structure.

```python
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

# Create and train the decision tree model
dt_model = DecisionTreeClassifier(max_depth=3, random_state=42)
dt_model.fit(X_train, y_train)

# Visualize the decision tree
from sklearn.tree import plot_tree
plt.figure(figsize=(15, 10))
plot_tree(dt_model, feature_names=['Feature 1', 'Feature 2'], 
          class_names=['Class 0', 'Class 1'], filled=True, rounded=True)
plt.title('Decision Tree Visualization')
plt.show()
```

Slide 4: Random Forest

Random Forest is an ensemble learning method that combines multiple decision trees to create a more robust and accurate model. It helps reduce overfitting and improves generalization.

```python
from sklearn.ensemble import RandomForestClassifier

# Create and train the random forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions and calculate accuracy
rf_pred = rf_model.predict(X_test)
rf_accuracy = rf_model.score(X_test, y_test)
print(f"Random Forest Accuracy: {rf_accuracy:.2f}")

# Feature importance
importances = rf_model.feature_importances_
for i, importance in enumerate(importances):
    print(f"Feature {i+1} Importance: {importance:.4f}")
```

Slide 5: Support Vector Machines (SVM)

Support Vector Machines find the optimal hyperplane that separates different classes in high-dimensional space. They're effective for both linear and non-linear classification problems.

```python
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train the SVM model
svm_model = SVC(kernel='rbf', random_state=42)
svm_model.fit(X_train_scaled, y_train)

# Make predictions and calculate accuracy
svm_pred = svm_model.predict(X_test_scaled)
svm_accuracy = svm_model.score(X_test_scaled, y_test)
print(f"SVM Accuracy: {svm_accuracy:.2f}")
```

Slide 6: K-Nearest Neighbors (KNN)

K-Nearest Neighbors is a simple, instance-based learning algorithm. It classifies new data points based on the majority class of their k nearest neighbors in the feature space.

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Create and train the KNN model
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)

# Make predictions and calculate accuracy
knn_pred = knn_model.predict(X_test)
knn_accuracy = accuracy_score(y_test, knn_pred)
print(f"KNN Accuracy: {knn_accuracy:.2f}")

# Visualize KNN decision boundary
def plot_decision_boundary(model, X, y):
    h = .02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('KNN Decision Boundary')
    plt.show()

plot_decision_boundary(knn_model, X, y)
```

Slide 7: Naive Bayes

Naive Bayes is a probabilistic classifier based on Bayes' theorem. It assumes that features are independent, which simplifies the calculation but may not always hold true in real-world scenarios.

```python
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Create and train the Naive Bayes model
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)

# Make predictions and calculate accuracy
nb_pred = nb_model.predict(X_test)
nb_accuracy = accuracy_score(y_test, nb_pred)
print(f"Naive Bayes Accuracy: {nb_accuracy:.2f}")

# Create and plot confusion matrix
cm = confusion_matrix(y_test, nb_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix for Naive Bayes')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
```

Slide 8: Gradient Boosting

Gradient Boosting is an ensemble learning method that builds a series of weak learners (typically decision trees) sequentially, with each new model correcting the errors of the previous ones.

```python
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report

# Create and train the Gradient Boosting model
gb_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
gb_model.fit(X_train, y_train)

# Make predictions and calculate accuracy
gb_pred = gb_model.predict(X_test)
gb_accuracy = accuracy_score(y_test, gb_pred)
print(f"Gradient Boosting Accuracy: {gb_accuracy:.2f}")

# Print classification report
print(classification_report(y_test, gb_pred))
```

Slide 9: Neural Networks

Neural Networks are powerful models inspired by the human brain. They consist of interconnected layers of neurons and can learn complex patterns in data.

```python
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train the Neural Network model
nn_model = MLPClassifier(hidden_layer_sizes=(10, 5), max_iter=1000, random_state=42)
nn_model.fit(X_train_scaled, y_train)

# Make predictions and calculate accuracy
nn_pred = nn_model.predict(X_test_scaled)
nn_accuracy = accuracy_score(y_test, nn_pred)
print(f"Neural Network Accuracy: {nn_accuracy:.2f}")

# Plot learning curve
plt.plot(nn_model.loss_curve_)
plt.title('Neural Network Learning Curve')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.show()
```

Slide 10: Model Comparison

Let's compare the performance of different classification algorithms on our dataset to see which one performs best.

```python
import pandas as pd

# Create a dictionary of models
models = {
    'Logistic Regression': LogisticRegression(),
    'Decision Tree': DecisionTreeClassifier(max_depth=3),
    'Random Forest': RandomForestClassifier(n_estimators=100),
    'SVM': SVC(kernel='rbf'),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'Naive Bayes': GaussianNB(),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100),
    'Neural Network': MLPClassifier(hidden_layer_sizes=(10, 5), max_iter=1000)
}

# Train and evaluate each model
results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    results.append({'Model': name, 'Accuracy': accuracy})

# Create a DataFrame and sort by accuracy
results_df = pd.DataFrame(results).sort_values('Accuracy', ascending=False)
print(results_df)

# Plot the results
plt.figure(figsize=(10, 6))
plt.bar(results_df['Model'], results_df['Accuracy'])
plt.title('Model Comparison')
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
```

Slide 11: Cross-Validation

Cross-validation is a technique used to assess model performance and prevent overfitting. It involves splitting the data into multiple subsets and training the model on different combinations of these subsets.

```python
from sklearn.model_selection import cross_val_score

# Choose a model (e.g., Random Forest)
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Perform 5-fold cross-validation
cv_scores = cross_val_score(model, X, y, cv=5)

print("Cross-validation scores:", cv_scores)
print(f"Mean CV score: {cv_scores.mean():.2f}")
print(f"Standard deviation of CV scores: {cv_scores.std():.2f}")

# Plot cross-validation scores
plt.figure(figsize=(8, 6))
plt.boxplot(cv_scores)
plt.title('Cross-Validation Scores Distribution')
plt.ylabel('Accuracy')
plt.show()
```

Slide 12: Hyperparameter Tuning

Hyperparameter tuning is the process of finding the optimal set of hyperparameters for a machine learning model. Grid search is a common method for this task.

```python
from sklearn.model_selection import GridSearchCV

# Define the parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}

# Create a Random Forest model
rf = RandomForestClassifier(random_state=42)

# Perform grid search
grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Print the best parameters and score
print("Best parameters:", grid_search.best_params_)
print(f"Best cross-validation score: {grid_search.best_score_:.2f}")

# Evaluate the best model on the test set
best_model = grid_search.best_estimator_
test_accuracy = best_model.score(X_test, y_test)
print(f"Test accuracy with best model: {test_accuracy:.2f}")
```

Slide 13: Real-Life Example: Iris Flower Classification

The Iris dataset is a classic example in machine learning. It contains measurements of iris flowers and the task is to classify them into different species.

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# Visualize feature importance
feature_importance = model.feature_importances_
features = iris.feature_names

plt.figure(figsize=(10, 6))
plt.bar(features, feature_importance)
plt.title('Feature Importance in Iris Classification')
plt.xlabel('Features')
plt.ylabel('Importance')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

Slide 14: Real-Life Example: Handwritten Digit Recognition

Handwritten digit recognition is a common application of classification algorithms, used in postal services and document digitization.

```python
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load the digits dataset
digits = load_digits()
X, y = digits.data, digits.target

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = SVC(kernel='rbf', random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Plot confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix for Digit Recognition')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Display some example digits and predictions
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for i, ax in enumerate(axes.flat):
    ax.imshow(X_test[i].reshape(8, 8), cmap='gray')
    ax.set_title(f"True: {y_test[i]}, Pred: {y_pred[i]}")
    ax.axis('off')
plt.tight_layout()
plt.show()
```

Slide 15: Additional Resources

For those interested in delving deeper into classification algorithms and machine learning, here are some valuable resources:

1. ArXiv paper on Random Forest: "Random Forests" by Leo Breiman (2001) ArXiv URL: [https://arxiv.org/abs/stat/0110009](https://arxiv.org/abs/stat/0110009)
2. ArXiv paper on Support Vector Machines: "A Tutorial on Support Vector Machines for Pattern Recognition" by Christopher J.C. Burges (1998) ArXiv URL: [https://arxiv.org/abs/1011.1669](https://arxiv.org/abs/1011.1669)
3. ArXiv paper on Neural Networks: "Efficient BackProp" by Yann LeCun et al. (1998) ArXiv URL: [https://arxiv.org/abs/1011.1669v3](https://arxiv.org/abs/1011.1669v3)
4. Scikit-learn Documentation: [https://scikit-learn.org/stable/documentation.html](https://scikit-learn.org/stable/documentation.html)
5. Python Machine Learning (Book) by Sebastian Raschka and Vahid Mirjalili

These resources provide in-depth explanations and advanced techniques for classification algorithms and their implementations.

