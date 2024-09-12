## Predictive Modeling with Python
Slide 1: Introduction to Predictive Modeling

Predictive modeling is a statistical technique used to forecast future outcomes based on historical data. It involves analyzing patterns in existing data to make informed predictions about future events or behaviors. In this slideshow, we'll explore how to implement predictive models using Python, a versatile programming language with powerful libraries for data analysis and machine learning.

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load and prepare data
data = pd.read_csv('sales_data.csv')
X = data[['advertising_spend', 'previous_sales']]
y = data['future_sales']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared Score: {r2}")
```

Slide 2: Data Collection and Preprocessing

The first step in predictive modeling is gathering and preparing the data. This involves collecting relevant information from various sources, cleaning the data to remove inconsistencies or errors, and transforming it into a format suitable for analysis. Python's pandas library is excellent for these tasks, offering powerful tools for data manipulation and preprocessing.

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load data from CSV file
data = pd.read_csv('customer_data.csv')

# Handle missing values
data['age'].fillna(data['age'].mean(), inplace=True)
data['income'].fillna(data['income'].median(), inplace=True)

# Remove duplicates
data.drop_duplicates(inplace=True)

# Convert categorical variables to numerical
data = pd.get_dummies(data, columns=['gender', 'occupation'])

# Scale numerical features
scaler = StandardScaler()
numerical_features = ['age', 'income']
data[numerical_features] = scaler.fit_transform(data[numerical_features])

print(data.head())
print(data.info())
```

Slide 3: Feature Selection and Engineering

Feature selection involves choosing the most relevant variables for your predictive model, while feature engineering is the process of creating new features from existing data. These steps are crucial for improving model performance and reducing overfitting. Python offers various techniques and libraries to assist with these tasks.

```python
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import PolynomialFeatures

# Load preprocessed data
data = pd.read_csv('preprocessed_data.csv')
X = data.drop('target', axis=1)
y = data['target']

# Feature selection using correlation
correlation = X.corr()['target'].abs().sort_values(ascending=False)
selected_features = correlation[correlation > 0.5].index.tolist()

# Create polynomial features
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X[selected_features])

# Select top K features
selector = SelectKBest(score_func=f_regression, k=10)
X_selected = selector.fit_transform(X_poly, y)

print("Selected features:", X.columns[selector.get_support()].tolist())
print("Shape of selected features:", X_selected.shape)
```

Slide 4: Linear Regression

Linear regression is a fundamental predictive modeling technique used to establish a relationship between input variables and a continuous output variable. It assumes a linear relationship between the features and the target variable. Let's implement a simple linear regression model using Python's scikit-learn library.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Generate sample data
np.random.seed(42)
X = np.random.rand(100, 1)
y = 2 + 3 * X + np.random.randn(100, 1) * 0.1

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Coefficient: {model.coef_[0][0]:.2f}")
print(f"Intercept: {model.intercept_[0]:.2f}")
print(f"Mean Squared Error: {mse:.4f}")
print(f"R-squared Score: {r2:.4f}")

# Plot the results
plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.plot(X_test, y_pred, color='red', label='Predicted')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()
```

Slide 5: Logistic Regression

Logistic regression is a popular algorithm for binary classification problems. It predicts the probability of an instance belonging to a particular class. Despite its name, logistic regression is a classification algorithm, not a regression algorithm. Let's implement a logistic regression model for a binary classification task.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.datasets import make_classification

# Generate sample data
X, y = make_classification(n_samples=1000, n_features=2, n_redundant=0, n_informative=2,
                           n_clusters_per_class=1, random_state=42)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print("Confusion Matrix:")
print(conf_matrix)

# Plot decision boundary
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.4)
plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8)
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Logistic Regression Decision Boundary")
plt.show()
```

Slide 6: Decision Trees

Decision trees are versatile algorithms used for both classification and regression tasks. They make predictions by learning simple decision rules inferred from the data features. Decision trees are easy to interpret and can handle both numerical and categorical data. Let's implement a decision tree classifier using scikit-learn.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import load_iris

# Load the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# Visualize the decision tree
plt.figure(figsize=(20,10))
plot_tree(model, feature_names=iris.feature_names, class_names=iris.target_names, filled=True, rounded=True)
plt.show()

# Feature importance
importance = model.feature_importances_
for i, v in enumerate(importance):
    print(f"Feature: {iris.feature_names[i]}, Score: {v:.4f}")
```

Slide 7: Random Forests

Random forests are an ensemble learning method that constructs multiple decision trees and combines their predictions. This technique often results in better performance and reduced overfitting compared to individual decision trees. Let's implement a random forest classifier using scikit-learn.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.datasets import make_classification

# Generate sample data
X, y = make_classification(n_samples=1000, n_features=20, n_informative=10, n_classes=3, random_state=42)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print("Confusion Matrix:")
print(conf_matrix)

# Feature importance
importance = model.feature_importances_
indices = np.argsort(importance)[::-1]

plt.figure(figsize=(10, 6))
plt.title("Feature Importances")
plt.bar(range(X.shape[1]), importance[indices])
plt.xticks(range(X.shape[1]), [f"Feature {i}" for i in indices], rotation=90)
plt.tight_layout()
plt.show()

# Out-of-bag score
oob_score = model.oob_score_
print(f"Out-of-bag Score: {oob_score:.4f}")
```

Slide 8: Support Vector Machines (SVM)

Support Vector Machines are powerful algorithms used for classification and regression tasks. They work by finding the hyperplane that best separates different classes in high-dimensional space. SVMs are particularly effective in handling non-linearly separable data through the use of kernel functions. Let's implement an SVM classifier using scikit-learn.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons

# Generate sample data
X, y = make_moons(n_samples=1000, noise=0.15, random_state=42)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
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
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Plot decision boundary
def plot_decision_boundary(X, y, model, ax=None):
    ax = ax or plt.gca()
    
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    
    Z = model.predict(scaler.transform(np.c_[xx.ravel(), yy.ravel()]))
    Z = Z.reshape(xx.shape)
    
    ax.contourf(xx, yy, Z, alpha=0.4)
    ax.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8)
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    return ax

plt.figure(figsize=(10, 8))
plot_decision_boundary(X, y, model)
plt.title("SVM Decision Boundary (RBF Kernel)")
plt.show()
```

Slide 9: K-Nearest Neighbors (KNN)

K-Nearest Neighbors is a simple yet effective algorithm used for both classification and regression tasks. It makes predictions based on the majority class (for classification) or average value (for regression) of the K nearest neighbors in the feature space. KNN is intuitive and easy to implement, making it a good starting point for many machine learning problems.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris

# Load the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Plot the effect of K on accuracy
k_range = range(1, 31)
scores = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    scores.append(knn.score(X_test, y_test))

plt.figure(figsize=(10, 6))
plt.plot(k_range, scores)
plt.xlabel('Value of K')
plt.ylabel('Testing Accuracy')
plt.title('KNN: Effect of K on Accuracy')
plt.show()
```

Slide 10: Naive Bayes

Naive Bayes is a probabilistic classifier based on applying Bayes' theorem with strong independence assumptions between the features. Despite its simplicity, Naive Bayes often performs surprisingly well and is particularly useful for text classification tasks. Let's implement a Gaussian Naive Bayes classifier using scikit-learn.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.datasets import make_classification

# Generate sample data
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = GaussianNB()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print("Confusion Matrix:")
print(conf_matrix)

# Plot feature importance
feature_importance = np.abs(model.theta_[1] - model.theta_[0])
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5

plt.figure(figsize=(10, 6))
plt.barh(pos, feature_importance[sorted_idx], align='center')
plt.yticks(pos, [f'Feature {i}' for i in sorted_idx])
plt.xlabel('Absolute difference in mean')
plt.title('Feature Importance in Gaussian Naive Bayes')
plt.tight_layout()
plt.show()
```

Slide 11: Gradient Boosting

Gradient Boosting is an ensemble learning technique that builds a series of weak learners (typically decision trees) to create a strong predictor. It works by iteratively improving upon the previous model's errors. Gradient Boosting is known for its high performance and is widely used in various competitions and real-world applications.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import make_classification

# Generate sample data
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_classes=2, random_state=42)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Plot feature importance
feature_importance = model.feature_importances_
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5

plt.figure(figsize=(10, 6))
plt.barh(pos, feature_importance[sorted_idx], align='center')
plt.yticks(pos, [f'Feature {i}' for i in sorted_idx])
plt.xlabel('Feature Importance')
plt.title('Feature Importance in Gradient Boosting')
plt.tight_layout()
plt.show()
```

Slide 12: Model Evaluation and Validation

Model evaluation and validation are crucial steps in the predictive modeling process. They help us assess the performance of our models and ensure that they generalize well to unseen data. Common techniques include cross-validation, learning curves, and various performance metrics. Let's explore some of these methods using scikit-learn.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, learning_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

# Generate sample data
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_classes=2, random_state=42)

# Create the model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Perform cross-validation
cv_scores = cross_val_score(model, X, y, cv=5)
print(f"Cross-validation scores: {cv_scores}")
print(f"Mean CV score: {cv_scores.mean():.4f}")

# Generate learning curve
train_sizes, train_scores, test_scores = learning_curve(
    model, X, y, train_sizes=np.linspace(0.1, 1.0, 10), cv=5)

train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.figure(figsize=(10, 6))
plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.1, color="r")
plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.1, color="g")
plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
plt.xlabel("Training examples")
plt.ylabel("Score")
plt.title("Learning Curve")
plt.legend(loc="best")
plt.show()
```

Slide 13: Hyperparameter Tuning

Hyperparameter tuning is the process of finding the optimal set of hyperparameters for a machine learning model. This step is crucial for maximizing model performance. Two common approaches for hyperparameter tuning are Grid Search and Random Search. Let's implement these techniques using scikit-learn.

```python
import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Generate sample data
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_classes=2, random_state=42)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the model
model = RandomForestClassifier(random_state=42)

# Define the hyperparameter grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Perform Grid Search
grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)

print("Grid Search Results:")
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-validation score: {grid_search.best_score_:.4f}")

# Perform Random Search
random_search = RandomizedSearchCV(model, param_distributions=param_grid, n_iter=20, cv=5, random_state=42, n_jobs=-1)
random_search.fit(X_train, y_train)

print("\nRandom Search Results:")
print(f"Best parameters: {random_search.best_params_}")
print(f"Best cross-validation score: {random_search.best_score_:.4f}")

# Evaluate the best model on the test set
best_model = grid_search.best_estimator_
test_score = best_model.score(X_test, y_test)
print(f"\nTest set score with best model: {test_score:.4f}")
```

Slide 14: Real-life Example: House Price Prediction

Let's apply our predictive modeling skills to a real-world problem: predicting house prices. We'll use a dataset containing various features of houses and their corresponding prices. This example demonstrates the entire workflow, from data preprocessing to model evaluation.

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset (assuming we have a CSV file named 'house_prices.csv')
data = pd.read_csv('house_prices.csv')

# Separate features and target
X = data.drop('price', axis=1)
y = data['price']

# Handle missing values
X = X.fillna(X.mean())

# Encode categorical variables
X = pd.get_dummies(X)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared Score: {r2:.4f}")

# Feature importance
feature_importance = model.feature_importances_
feature_importance_df = pd.DataFrame({'feature': X.columns, 'importance': feature_importance})
feature_importance_df = feature_importance_df.sort_values('importance', ascending=False).head(10)

print("\nTop 10 Most Important Features:")
print(feature_importance_df)
```

Slide 15: Additional Resources

For those interested in delving deeper into predictive modeling and machine learning, here are some valuable resources:

1. ArXiv.org: A comprehensive repository of research papers on machine learning and predictive modeling. URL: [https://arxiv.org/list/stat.ML/recent](https://arxiv.org/list/stat.ML/recent)
2. Scikit-learn Documentation: Official documentation for the scikit-learn library, which provides extensive resources on machine learning algorithms and techniques.
3. Kaggle: A platform for data science competitions and a wealth of datasets for practice.
4. Machine Learning Mastery: A blog with practical tutorials and guides on various machine learning topics.
5. Coursera Machine Learning Course: A popular online course by Andrew Ng, covering fundamental concepts in machine learning.

Remember to verify the accuracy and relevance of these resources, as the field of machine learning is rapidly evolving.

