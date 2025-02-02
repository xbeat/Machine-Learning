## Part 1: Foundations of Machine Learning

### Chapter 1: Supervised vs Unsupervised Learning

#### Example 1
```python
import numpy as np  
from sklearn.datasets import make_classification  
from sklearn.model_selection import train_test_split  
from sklearn.linear_model import LogisticRegression  
from sklearn.metrics import accuracy_score  

# Create a synthetic dataset  
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)  

# Split the data into training and testing sets  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  
```

#### Example 2
```python
# Initialize and train the model  
model = LogisticRegression(random_state=42)  
model.fit(X_train, y_train)  

# Make predictions on the test set  
y_pred = model.predict(X_test)  

# Calculate the accuracy  
accuracy = accuracy_score(y_test, y_pred)  
print(f"Accuracy: {accuracy:.2f}")  
```

#### Example 3
```python
from sklearn.cluster import KMeans  
from sklearn.datasets import make_blobs  
import matplotlib.pyplot as plt  

# Create a synthetic dataset  
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=42)  

# Initialize and fit the K-means model  
kmeans = KMeans(n_clusters=4, random_state=42)  
kmeans.fit(X)  

# Get the cluster assignments and centroids  
labels = kmeans.labels_  
centroids = kmeans.cluster_centers_  

# Plot the results  
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')  
plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=200, linewidths=3, color='r')  
plt.title('K-means Clustering')  
plt.show()  
```

### Chapter 2: Regression, Classification, and Clustering  

#### Example 1  

```python
import numpy as np  
import pandas as pd  
from sklearn.model_selection import train_test_split  
from sklearn.linear_model import LinearRegression  
from sklearn.metrics import mean_squared_error, r2_score  
import matplotlib.pyplot as plt  

# Load the dataset  
data = pd.read_csv('house_prices.csv')  
X = data\['size'\].values.reshape(-1, 1)  
y = data\['price'\].values  

# Split the data into training and testing sets  
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

# Plot the results  
plt.scatter(X_test, y_test, color='blue', label='Actual')  
plt.plot(X_test, y_pred, color='red', label='Predicted')  
plt.xlabel('House Size')  
plt.ylabel('Price')  
plt.legend()  
plt.show()  
```

#### Example 2  

```python
from sklearn.datasets import load_iris  
from sklearn.model_selection import train_test_split  
from sklearn.linear_model import LogisticRegression  
from sklearn.metrics import accuracy_score, classification_report  
from sklearn.preprocessing import StandardScaler  

# Load the iris dataset  
iris = load_iris()  
X, y = iris.data, iris.target  

# Split the data into training and testing sets  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)  

# Scale the features  
scaler = StandardScaler()  
X_train_scaled = scaler.fit_transform(X_train)  
X_test_scaled = scaler.transform(X_test)  

# Create and train the model  
model = LogisticRegression(multi_class='ovr', random_state=42)  
model.fit(X_train_scaled, y_train)  

# Make predictions  
y_pred = model.predict(X_test_scaled)  

# Evaluate the model  
accuracy = accuracy_score(y_test, y_pred)  
print(f"Accuracy: {accuracy}")  
print("\nClassification Report:")  
print(classification_report(y_test, y_pred, target_names=iris.target_names))  
```

#### Example 3 

```python 
from sklearn.cluster import KMeans  
from sklearn.preprocessing import StandardScaler  
from sklearn.datasets import make_blobs  
import matplotlib.pyplot as plt  

# Generate synthetic data  
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=42)  

# Scale the features  
scaler = StandardScaler()  
X_scaled = scaler.fit_transform(X)  

# Perform K-means clustering  
kmeans = KMeans(n_clusters=4, random_state=42)  
cluster_labels = kmeans.fit_predict(X_scaled)  

# Visualize the results  
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=cluster_labels, cmap='viridis')  
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='x', color='red', s=200, label='Centroids')  
plt.title('K-means Clustering Results')  
plt.legend()  
plt.show()  
```

### Chapter 3: Bias-Variance Tradeoff  

#### Example 1  

```python
import numpy as np  
import matplotlib.pyplot as plt  
from sklearn.model_selection import train_test_split  
from sklearn.preprocessing import PolynomialFeatures  
from sklearn.linear_model import LinearRegression  
from sklearn.metrics import mean_squared_error  

# Generate synthetic data  
np.random.seed(0)  
X = np.sort(5 \* np.random.rand(80, 1), axis=0)  
y = np.sin(X).ravel() + np.random.normal(0, 0.1, X.shape\[0\])  

# Split the data into training and testing sets  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  
```

#### Example 2  

```python
def fit_polynomial_regression(X_train, y_train, X_test, y_test, degrees):  
  train_errors = []  
  test_errors = []  

  for degree in degrees:  
    poly_features = PolynomialFeatures(degree=degree, include_bias=False)  
    X_train_poly = poly_features.fit_transform(X_train)  
    X_test_poly = poly_features.transform(X_test)  
  
    model = LinearRegression()  
    model.fit(X_train_poly, y_train)  
    
    train_pred = model.predict(X_train_poly)  
    test_pred = model.predict(X_test_poly)  
    
    train_error = mean_squared_error(y_train, train_pred)  
    test_error = mean_squared_error(y_test, test_pred)  
    
    train_errors.append(train_error)  
    test_errors.append(test_error)  
  
  return train_errors, test_errors  
```

#### Example 3  

```python
degrees = range(1, 15)  
train_errors, test_errors = fit_polynomial_regression(X_train, y_train, X_test, y_test, degrees)  

plt.figure(figsize=(10, 6))  
plt.plot(degrees, train_errors, label='Training Error')  
plt.plot(degrees, test_errors, label='Testing Error')  
plt.xlabel('Polynomial Degree')  
plt.ylabel('Mean Squared Error')  
plt.title('Bias-Variance Tradeoff')  
plt.legend()  
plt.show()  
```

### Chapter 4: Overfitting vs Underfitting  

#### Example 1  

```python
import numpy as np  
import matplotlib.pyplot as plt  
from sklearn.model_selection import train_test_split  
from sklearn.preprocessing import PolynomialFeatures  
from sklearn.linear_model import LinearRegression  
from sklearn.metrics import mean_squared_error  

# Generate sample data  
np.random.seed(0)  
X = np.sort(5 \* np.random.rand(80, 1), axis=0)  
y = np.sin(X).ravel() + np.random.normal(0, 0.1, X.shape\[0\])  

# Split the data into training and testing sets  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)  
```

#### Example 2  

```python
def fit_polynomial_regression(X_train, y_train, X_test, y_test, degree):  
  poly_features = PolynomialFeatures(degree=degree, include_bias=False)  
  X_train_poly = poly_features.fit_transform(X_train)  
  X_test_poly = poly_features.transform(X_test)  
  
  model = LinearRegression()  
  model.fit(X_train_poly, y_train)  
  
  train_pred = model.predict(X_train_poly)  
  test_pred = model.predict(X_test_poly)  
  
  train_mse = mean_squared_error(y_train, train_pred)  
  test_mse = mean_squared_error(y_test, test_pred)  
  
  return model, poly_features, train_mse, test_mse  
```

#### Example 3  

```python
degrees = [1, 3, 15]  
plt.figure(figsize=(15, 5))  

for i, degree in enumerate(degrees):  
model, poly_features, train_mse, test_mse = fit_polynomial_regression(X_train, y_train, X_test, y_test, degree)  

X_plot = np.linspace(0, 5, 100).reshape(-1, 1)  
X_plot_poly = poly_features.transform(X_plot)  
y_plot = model.predict(X_plot_poly)  

plt.subplot(1, 3, i+1)  
plt.scatter(X_train, y_train, color='b', label='Training data')  
plt.scatter(X_test, y_test, color='r', label='Test data')  
plt.plot(X_plot, y_plot, color='g', label='Model prediction')  
plt.title(f'Degree {degree} Polynomial\nTrain MSE: {train_mse:.4f}, Test MSE: {test_mse:.4f}')  
plt.xlabel('X')  
plt.ylabel('y')  
plt.legend()  

plt.tight_layout()  
plt.show()  
```

#### Example 4  

```python
from sklearn.linear_model import Ridge  

def fit_polynomial_regression_with_regularization(X_train, y_train, X_test, y_test, degree, alpha):  
  poly_features = PolynomialFeatures(degree=degree, include_bias=False)  
  X_train_poly = poly_features.fit_transform(X_train)  
  X_test_poly = poly_features.transform(X_test)  

  model = Ridge(alpha=alpha)  
  model.fit(X_train_poly, y_train)  
  
  train_pred = model.predict(X_train_poly)  
  test_pred = model.predict(X_test_poly)  
  
  train_mse = mean_squared_error(y_train, train_pred)  
  test_mse = mean_squared_error(y_test, test_pred)  
  
  return model, poly_features, train_mse, test_mse  
  
# Fit regularized models  
alphas = [0, 0.1, 1]  
degree = 15  
plt.figure(figsize=(15, 5))  

for i, alpha in enumerate(alphas):  
  model, poly_features, train_mse, test_mse = fit_polynomial_regression_with_regularization(X_train, y_train, X_test, y_test, degree, alpha)  

  X_plot = np.linspace(0, 5, 100).reshape(-1, 1)  
  X_plot_poly = poly_features.transform(X_plot)  
  y_plot = model.predict(X_plot_poly)  
  
  plt.subplot(1, 3, i+1)  
  plt.scatter(X_train, y_train, color='b', label='Training data')  
  plt.scatter(X_test, y_test, color='r', label='Test data')  
  plt.plot(X_plot, y_plot, color='g', label='Model prediction')  
  plt.title(f'Degree {degree} Polynomial, Alpha: {alpha}\nTrain MSE: {train_mse:.4f}, Test MSE: {test_mse:.4f}')  
  plt.xlabel('X')  
  plt.ylabel('y')  
  plt.legend()  
  
plt.tight_layout()  
plt.show()  
```

#### Example 5  

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator  
from tensorflow.keras.models import Sequential  
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense  

# Create a data generator with augmentation  
datagen = ImageDataGenerator(  
rotation_range=20,  
width_shift_range=0.2,  
height_shift_range=0.2,  
horizontal_flip=True,  
zoom_range=0.2  
)  

# Create a simple CNN model  
model = Sequential([  
Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),  
MaxPooling2D((2, 2)),  
Conv2D(64, (3, 3), activation='relu'),  
MaxPooling2D((2, 2)),  
Conv2D(64, (3, 3), activation='relu'),  
Flatten(),  
Dense(64, activation='relu'),  
Dense(10, activation='softmax')  
])  

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])  

# Train the model using the data generator  
history = model.fit(  
datagen.flow(X_train, y_train, batch_size=32),  
steps_per_epoch=len(X_train) // 32,  
epochs=50,  
validation_data=(X_val, y_val)  
)  
```

### Chapter 5: Cross-Validation  
#### Example 1  

```python
import numpy as np  
from sklearn.model_selection import KFold  
from sklearn.linear_model import LinearRegression  
from sklearn.metrics import mean_squared_error  
import pandas as pd  

# Load the dataset (assuming we have a CSV file with our data)  
data = pd.read_csv('house_prices.csv')  
X = data\[\['size', 'bedrooms', 'location'\]\]  
y = data\['price'\]  
```

#### Example 2  

```python
# Set the number of folds  
k = 5  

# Initialize the KFold object  
kf = KFold(n_splits=k, shuffle=True, random_state=42)  

# Initialize lists to store our results  
mse_scores = []  
```

#### Example 3  

```python
for train_index, test_index in kf.split(X):  
# Split the data into training and testing sets  
X_train, X_test = X.iloc[train_index], X.iloc[test_index]  
y_train, y_test = y.iloc[train_index], y.iloc[test_index]  

# Initialize and train the model  
model = LinearRegression()  
model.fit(X_train, y_train)  

# Make predictions on the test set  
y_pred = model.predict(X_test)  

# Calculate the mean squared error and append to our list  
mse = mean_squared_error(y_test, y_pred)  
mse_scores.append(mse)  

# Calculate the average MSE across all folds  
average_mse = np.mean(mse_scores)  
print(f"Average Mean Squared Error: {average_mse}")  

from sklearn.model_selection import StratifiedKFold  

# Assuming 'y' is our target variable for a classification problem  
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)  

for train_index, test_index in skf.split(X, y):  
X_train, X_test = X.iloc[train_index], X.iloc[test_index]  
y_train, y_test = y.iloc[train_index], y.iloc[test_index]  
# ... proceed with model training and evaluation as before  
```

#### Example 4  

```python
from sklearn.model_selection import LeaveOneOut  

loo = LeaveOneOut()  

for train_index, test_index in loo.split(X):  
X_train, X_test = X.iloc[train_index], X.iloc[test_index]  
y_train, y_test = y.iloc[train_index], y.iloc[test_index]  
# ... proceed with model training and evaluation as before  
```

## Part 2: Popular Machine Learning Algorithms  

### Chapter 6: Linear and Logistic Regression  

#### Example 1  

```python
import numpy as np  
from sklearn.linear_model import LinearRegression  
import matplotlib.pyplot as plt  

# Generate sample data  
X = np.array(\[\[1\], \[2\], \[3\], \[4\], \[5\]\])  
y = np.array(\[2, 4, 5, 4, 5\])  

# Create and fit the model  
model = LinearRegression()  
model.fit(X, y)  

# Make predictions  
X_test = np.array(\[\[0\], \[6\]\])  
y_pred = model.predict(X_test)  

# Plot the results  
plt.scatter(X, y, color='blue')  
plt.plot(X_test, y_pred, color='red')  
plt.xlabel('X')  
plt.ylabel('y')  
plt.show()  

print(f"Intercept: {model.intercept\_}")  
print(f"Slope: {model.coef\_\[0\]}")  
```

#### Example 2  

```python
from sklearn.linear_model import LogisticRegression  
from sklearn.model_selection import train_test_split  
from sklearn.metrics import accuracy_score, confusion_matrix  
import numpy as np  

# Generate sample data  
X = np.random.randn(100, 2)  
y = (X[:, 0] + X[:, 1] > 0).astype(int)  

# Split the data into training and testing sets  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  

# Create and fit the model  
model = LogisticRegression()  
model.fit(X_train, y_train)  

# Make predictions  
y_pred = model.predict(X_test)  

# Evaluate the model  
accuracy = accuracy_score(y_test, y_pred)  
conf_matrix = confusion_matrix(y_test, y_pred)  

print(f"Accuracy: {accuracy}")  
print("Confusion Matrix:")  
print(conf_matrix)  
```

#### Example 3  

```python
from sklearn.preprocessing import StandardScaler  
import numpy as np  

# Sample data  
X = np.array([[1, 10, 100],  
[2, 20, 200],  
[3, 30, 300]])  

# Create and fit the scaler  
scaler = StandardScaler()  
X_scaled = scaler.fit_transform(X)  

print("Original data:")  
print(X)  
print("\nScaled data:")  
print(X_scaled)  
```

#### Example 4  

```python
import pandas as pd  
from sklearn.model_selection import train_test_split  
from sklearn.linear_model import LinearRegression  
from sklearn.metrics import mean_squared_error, r2_score  

# Load the data (assuming we have a CSV file with the relevant information)  
data = pd.read_csv('house_prices.csv')  

# Separate features and target variable  
X = data[['sqft', 'bedrooms', 'location']]  
y = data['price']  

# Split the data  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  

# Create and fit the model  
model = LinearRegression()  
model.fit(X_train, y_train)  

# Make predictions  
y_pred = model.predict(X_test)  

# Evaluate the model  
mse = mean_squared_error(y_test, y_pred)  
r2 = r2_score(y_test, y_pred)  

print(f"Mean Squared Error: {mse}")  
print(f"R-squared: {r2}")  

# Print coefficients  
for feature, coef in zip(X.columns, model.coef_):  
print(f"{feature}: {coef}")  
```

#### Example 5  

```python
from sklearn.preprocessing import PolynomialFeatures  
from sklearn.pipeline import make_pipeline  
from sklearn.linear_model import LinearRegression  
import numpy as np  
import matplotlib.pyplot as plt  

# Generate sample data  
X = np.sort(5 * np.random.rand(80, 1), axis=0)  
y = np.sin(X).ravel() + np.random.normal(0, 0.1, X.shape[0])  

# Create polynomial regression models of different degrees  
degrees = [1, 3, 5]  
plt.figure(figsize=(14, 5))  

for i, degree in enumerate(degrees):  
ax = plt.subplot(1, 3, i + 1)  
model = make_pipeline(PolynomialFeatures(degree), LinearRegression())  
model.fit(X, y)  

X_test = np.linspace(0, 5, 100)[:, np.newaxis]  
plt.scatter(X, y, color='blue', s=30, alpha=0.5)  
plt.plot(X_test, model.predict(X_test), color='red')  
plt.title(f'Degree {degree}')  
plt.xlabel('X')  
plt.ylabel('y')  

plt.tight_layout()  
plt.show()  
```

#### Example 6  

```python
from sklearn.linear_model import Ridge, Lasso  
from sklearn.model_selection import train_test_split  
from sklearn.metrics import mean_squared_error  
import numpy as np  

# Generate sample data  
np.random.seed(42)  
X = np.random.randn(100, 20)  
y = np.random.randn(100)  

# Split the data  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  

# Ridge Regression  
ridge = Ridge(alpha=1.0)  
ridge.fit(X_train, y_train)  
y_pred_ridge = ridge.predict(X_test)  
mse_ridge = mean_squared_error(y_test, y_pred_ridge)  

# Lasso Regression  
lasso = Lasso(alpha=1.0)  
lasso.fit(X_train, y_train)  
y_pred_lasso = lasso.predict(X_test)  
mse_lasso = mean_squared_error(y_test, y_pred_lasso)  

print(f"Ridge MSE: {mse_ridge}")  
print(f"Lasso MSE: {mse_lasso}")  

# Compare coefficients  
print("\nNumber of non-zero coefficients:")  
print(f"Ridge: {np.sum(ridge.coef_ != 0)}")  
print(f"Lasso: {np.sum(lasso.coef\_ != 0)}") 
```

### Chapter 7: Decision Trees and Ensemble Methods  

#### Example 1  

```python
from sklearn.datasets import load_iris  
from sklearn.model_selection import train_test_split  
from sklearn.tree import DecisionTreeClassifier  
from sklearn.ensemble import RandomForestClassifier  
from sklearn.metrics import accuracy_score, classification_report  

# Load the iris dataset  
iris = load_iris()  
X, y = iris.data, iris.target  

# Split the data into training and testing sets  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  
```

#### Example 2  

```python
# Create and train the decision tree classifier  
dt_clf = DecisionTreeClassifier(random_state=42)  
dt_clf.fit(X_train, y_train)  

# Make predictions on the test set  
dt_predictions = dt_clf.predict(X_test)  

# Evaluate the model  
dt_accuracy = accuracy_score(y_test, dt_predictions)  
print("Decision Tree Accuracy:", dt_accuracy)  
print("Classification Report:")  
print(classification_report(y_test, dt_predictions, target_names=iris.target_names))  
```

#### Example 3  

```python
# Create and train the random forest classifier  
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)  
rf_clf.fit(X_train, y_train)  

# Make predictions on the test set  
rf_predictions = rf_clf.predict(X_test)  

# Evaluate the model  
rf_accuracy = accuracy_score(y_test, rf_predictions)  
print("Random Forest Accuracy:", rf_accuracy)  
print("Classification Report:")  
print(classification_report(y_test, rf_predictions, target_names=iris.target_names))  
```

#### Example 4  

```python
# Print feature importances for the random forest classifier  
feature_importances = rf_clf.feature_importances_  
for feature, importance in zip(iris.feature_names, feature_importances):  
print(f"{feature}: {importance}")  
```

#### Example 5  

```python
from sklearn.model_selection import GridSearchCV  

# Define the parameter grid  
param_grid = {  
'n_estimators': [50, 100, 200],  
'max_depth': [None, 10, 20],  
'min_samples_split': [2, 5, 10],  
'min_samples_leaf': [1, 2, 4]  
}  

# Create the grid search object  
grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5)  

# Perform the grid search  
grid_search.fit(X_train, y_train)  

# Print the best parameters and score  
print("Best parameters:", grid_search.best_params_)  
print("Best cross-validation score:", grid_search.best_score\_)  
```

### Chapter 8: Support Vector Machines (SVMs)  

#### Example 1  

```python
import numpy as np  
import matplotlib.pyplot as plt  
from sklearn import svm, datasets  
from sklearn.model_selection import train_test_split  
from sklearn.preprocessing import StandardScaler  
from sklearn.metrics import accuracy_score, classification_report  

# Load the iris dataset  
iris = datasets.load_iris()  
X = iris.data\[:, \[2, 3\]\] # We'll use petal length and width  
y = iris.target  

# Split the data into training and testing sets  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)  

# Standardize the features  
scaler = StandardScaler()  
X_train_scaled = scaler.fit_transform(X_train)  
X_test_scaled = scaler.transform(X_test)  

# Create and train the SVM classifier  
svm_classifier = svm.SVC(kernel='rbf', C=1.0, random_state=42)  
svm_classifier.fit(X_train_scaled, y_train)  

# Make predictions on the test set  
y_pred = svm_classifier.predict(X_test_scaled)  

# Evaluate the model  
accuracy = accuracy_score(y_test, y_pred)  
print(f"Accuracy: {accuracy:.2f}")  
print("\nClassification Report:")  
print(classification_report(y_test, y_pred, target_names=iris.target_names))  

# Visualize the decision boundaries  
x_min, x_max = X\[:, 0\].min() - 1, X\[:, 0\].max() + 1  
y_min, y_max = X\[:, 1\].min() - 1, X\[:, 1\].max() + 1  
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),  
np.arange(y_min, y_max, 0.02))  
Z = svm_classifier.predict(scaler.transform(np.c\_\[xx.ravel(), yy.ravel()\]))  
Z = Z.reshape(xx.shape)  

plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdYlBu)  
plt.scatter(X\[:, 0\], X\[:, 1\], c=y, cmap=plt.cm.RdYlBu, edgecolor='black')  
plt.xlabel('Petal length')  
plt.ylabel('Petal width')  
plt.title('SVM Decision Boundaries')  
plt.show()  
```

#### Example 2  

```python
from sklearn import svm  
from sklearn.multiclass import OneVsRestClassifier  
from sklearn.preprocessing import StandardScaler  
from sklearn.model_selection import train_test_split  
from sklearn.metrics import accuracy_score, classification_report  
from sklearn.datasets import load_iris  

# Load the iris dataset  
iris = load_iris()  
X, y = iris.data, iris.target  

# Split the data into training and testing sets  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)  

# Standardize the features  
scaler = StandardScaler()  
X_train_scaled = scaler.fit_transform(X_train)  
X_test_scaled = scaler.transform(X_test)  

# Create and train the multi-class SVM classifier  
svm_classifier = OneVsRestClassifier(svm.SVC(kernel='rbf', C=1.0, random_state=42))  
svm_classifier.fit(X_train_scaled, y_train)  

# Make predictions on the test set  
y_pred = svm_classifier.predict(X_test_scaled)  

# Evaluate the model  
accuracy = accuracy_score(y_test, y_pred)  
print(f"Accuracy: {accuracy:.2f}")  
print("\nClassification Report:")  
print(classification_report(y_test, y_pred, target_names=iris.target_names))  
```

### Chapter 9: K-Means Clustering and Principal Component Analysis (PCA)  

#### Example 1  

```python
import numpy as np  
import matplotlib.pyplot as plt  
from sklearn.cluster import KMeans  
from sklearn.preprocessing import StandardScaler  

# Generate sample data  
np.random.seed(42)  
X = np.random.randn(300, 2)  
X\[:100, :\] += 2  
X\[100:200, :\] -= 2  

# Preprocess the data  
scaler = StandardScaler()  
X_scaled = scaler.fit_transform(X)  

# Perform K-Means clustering  
kmeans = KMeans(n_clusters=3, random_state=42)  
kmeans.fit(X_scaled)  

# Visualize the results  
plt.figure(figsize=(10, 8))  
plt.scatter(X_scaled\[:, 0\], X_scaled\[:, 1\], c=kmeans.labels\_, cmap='viridis')  
plt.scatter(kmeans.cluster_centers\_\[:, 0\], kmeans.cluster_centers\_\[:, 1\], marker='x', s=200, linewidths=3, color='r')  
plt.title('K-Means Clustering Results')  
plt.xlabel('Feature 1')  
plt.ylabel('Feature 2')  
plt.show()  
```

#### Example 2  

```python
import numpy as np  
import matplotlib.pyplot as plt  
from sklearn.decomposition import PCA  
from sklearn.preprocessing import StandardScaler  
from sklearn.datasets import load_iris  

# Load the Iris dataset  
iris = load_iris()  
X = iris.data  
y = iris.target  

# Preprocess the data  
scaler = StandardScaler()  
X_scaled = scaler.fit_transform(X)  

# Apply PCA  
pca = PCA()  
X_pca = pca.fit_transform(X_scaled)  

# Plot the cumulative explained variance ratio  
plt.figure(figsize=(10, 6))  
plt.plot(np.cumsum(pca.explained_variance_ratio_))  
plt.xlabel('Number of Components')  
plt.ylabel('Cumulative Explained Variance Ratio')  
plt.title('Explained Variance Ratio vs. Number of Components')  
plt.grid(True)  
plt.show()  

# Visualize the first two principal components  
plt.figure(figsize=(10, 8))  
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis')  
plt.xlabel('First Principal Component')  
plt.ylabel('Second Principal Component')  
plt.title('Iris Dataset - First Two Principal Components')  
plt.colorbar(label='Target Class')  
plt.show()  

# Print the explained variance ratio  
print("Explained Variance Ratio:")  
print(pca.explained_variance_ratio\_)  
```

### Chapter 10: Neural Networks: CNNs and RNNs  

#### Example 1  

```python
import tensorflow as tf  
from tensorflow.keras import layers, models  
from tensorflow.keras.datasets import mnist  

# Load and preprocess the MNIST dataset  
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()  
train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255  
test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255  

# Define the CNN model  
model = models.Sequential(\[  
layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),  
layers.MaxPooling2D((2, 2)),  
layers.Conv2D(64, (3, 3), activation='relu'),  
layers.MaxPooling2D((2, 2)),  
layers.Conv2D(64, (3, 3), activation='relu'),  
layers.Flatten(),  
layers.Dense(64, activation='relu'),  
layers.Dense(10, activation='softmax')  
\])  

# Compile the model  
model.compile(optimizer='adam',  
loss='sparse_categorical_crossentropy',  
metrics=\['accuracy'\])  

# Train the model  
history = model.fit(train_images, train_labels, epochs=5,  
validation_data=(test_images, test_labels))  

# Evaluate the model  
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)  
print(f'Test accuracy: {test_acc}')  
```

#### Example 2  

```python
import tensorflow as tf  
from tensorflow.keras import layers, models  
from tensorflow.keras.datasets import imdb  
from tensorflow.keras.preprocessing import sequence  

# Load and preprocess the IMDB dataset  
max_features = 10000  
maxlen = 500  
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)  
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)  
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)  

# Define the RNN model  
model = models.Sequential([  
layers.Embedding(max_features, 32),  
layers.SimpleRNN(32),  
layers.Dense(1, activation='sigmoid')  
])  

# Compile the model  
model.compile(optimizer='rmsprop',  
loss='binary_crossentropy',  
metrics=['accuracy'])  

# Train the model  
history = model.fit(x_train, y_train,  
epochs=10,  
batch_size=128,  
validation_split=0.2)  

# Evaluate the model  
test_loss, test_acc = model.evaluate(x_test, y_test)  
print(f'Test accuracy: {test_acc}')  
```

## Part 3: Practical Machine Learning with Python  

### Chapter 11: Data Preprocessing with Pandas and NumPy  

#### Example 1  

```python
import pandas as pd  
import numpy as np  
from sklearn.preprocessing import StandardScaler, OneHotEncoder  
from sklearn.impute import SimpleImputer  

# Load the data  
df = pd.read_csv('sample_data.csv')  

# Display basic information about the dataset  
print(df.info())  
print(df.describe())  

# Check for missing values  
print(df.isnull().sum())  

# Handle missing values  
numeric_columns = df.select_dtypes(include=\[np.number\]).columns  
imputer = SimpleImputer(strategy='mean')  
df\[numeric_columns\] = imputer.fit_transform(df\[numeric_columns\])  

# Identify and handle outliers (using IQR method for numeric columns)  
for column in numeric_columns:  
Q1 = df\[column\].quantile(0.25)  
Q3 = df\[column\].quantile(0.75)  
IQR = Q3 - Q1  
lower_bound = Q1 - 1.5 \* IQR  
upper_bound = Q3 + 1.5 \* IQR  
df\[column\] = np.where(df\[column\] \> upper_bound, upper_bound,  
np.where(df\[column\] \< lower_bound, lower_bound, df\[column\]))  

# Scale numeric features  
scaler = StandardScaler()  
df\[numeric_columns\] = scaler.fit_transform(df\[numeric_columns\])  

# Encode categorical variables  
categorical_columns = df.select_dtypes(include=\['object'\]).columns  
encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')  
encoded_cats = encoder.fit_transform(df\[categorical_columns\])  
encoded_df = pd.DataFrame(encoded_cats, columns=encoder.get_feature_names(categorical_columns))  

# Combine encoded categorical variables with numeric variables  
preprocessed_df = pd.concat(\[df\[numeric_columns\], encoded_df\], axis=1)  

print(preprocessed_df.head())  
```

#### Example 2  

```python
from sklearn.feature_selection import mutual_info_classif  
from sklearn.decomposition import PCA  
from imblearn.over_sampling import SMOTE  

# Feature selection using mutual information  
mi_scores = mutual_info_classif(preprocessed_df, target)  
mi_scores = pd.Series(mi_scores, name="MI Scores", index=preprocessed_df.columns)  
top_features = mi_scores.sort_values(ascending=False).head(10).index.tolist()  
selected_df = preprocessed_df[top_features]  

# Dimensionality reduction using PCA  
pca = PCA(n_components=5)  
pca_features = pca.fit_transform(selected_df)  
pca_df = pd.DataFrame(pca_features, columns=[f'PC{i+1}' for i in range(5)])  

# Handle imbalanced dataset using SMOTE  
smote = SMOTE(random_state=42)  
X_resampled, y_resampled = smote.fit_resample(pca_df, target)  

print(X_resampled.shape, y_resampled.shape)  
```

### Chapter 12: Model Training and Evaluation with Scikit-learn  

#### Example 1  

```python
from sklearn.model_selection import train_test_split  
from sklearn.linear_model import LinearRegression  
from sklearn.preprocessing import StandardScaler  
import numpy as np  
import pandas as pd  

# Load and prepare the data  
data = pd.read_csv('house_prices.csv')  
X = data\[\['sqft', 'bedrooms'\]\]  
y = data\['price'\]  

# Split the data into training and testing sets  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  

# Scale the features  
scaler = StandardScaler()  
X_train_scaled = scaler.fit_transform(X_train)  
X_test_scaled = scaler.transform(X_test)  

# Train the model  
model = LinearRegression()  
model.fit(X_train_scaled, y_train)  

# Make predictions  
y_pred = model.predict(X_test_scaled)  
```

#### Example 2  

```python
from sklearn.metrics import mean_squared_error, r2_score  

# Calculate MSE and RMSE  
mse = mean_squared_error(y_test, y_pred)  
rmse = np.sqrt(mse)  

# Calculate R-squared score  
r2 = r2_score(y_test, y_pred)  

print(f"Mean Squared Error: {mse}")  
print(f"Root Mean Squared Error: {rmse}")  
print(f"R-squared Score: {r2}")  
```

#### Example 3  

```python
from sklearn.model_selection import cross_val_score  

# Perform 5-fold cross-validation  
cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='neg_mean_squared_error')  

# Convert MSE to RMSE  
rmse_scores = np.sqrt(-cv_scores)  

print(f"Cross-validation RMSE scores: {rmse_scores}")  
print(f"Mean RMSE: {np.mean(rmse_scores)}")  
print(f"Standard deviation of RMSE: {np.std(rmse_scores)}")  
```

#### Example 4  

```python
from sklearn.ensemble import RandomForestClassifier  

# Create a Random Forest classifier with balanced class weights  
rf_classifier = RandomForestClassifier(class_weight='balanced', random_state=42)  

# Train the model  
rf_classifier.fit(X_train, y_train)  
```

#### Example 5 

```python 
from sklearn.linear_model import LogisticRegression  

# Create a logistic regression model with L2 regularization  
logistic_model = LogisticRegression(C=0.1, random_state=42)  

# Train the model  
logistic_model.fit(X_train, y_train)  
```

#### Example 6  


```python
from sklearn.compose import ColumnTransformer  
from sklearn.preprocessing import OneHotEncoder, StandardScaler  
from sklearn.pipeline import Pipeline  

# Define the preprocessing steps  
preprocessor = ColumnTransformer(  
transformers=[  
('num', StandardScaler(), ['age', 'income']),  
('cat', OneHotEncoder(drop='first'), ['education', 'occupation'])  
])  

# Create a pipeline that includes preprocessing and the model  
full_pipeline = Pipeline([  
('preprocessor', preprocessor),  
('classifier', RandomForestClassifier(random_state=42))  
])  

# Fit the pipeline to the data  
full_pipeline.fit(X_train, y_train)  
```

#### Example 7  

```python
from sklearn.feature_selection import RFECV  
from sklearn.ensemble import RandomForestRegressor  

# Create the RFE object and compute a cross-validated score for each number of features  
rfecv = RFECV(estimator=RandomForestRegressor(random_state=42), step=1, cv=5, scoring='neg_mean_squared_error')  

# Fit RFECV  
rfecv.fit(X_train, y_train)  

# Get the optimal number of features  
optimal_features = rfecv.n_features_  

print(f"Optimal number of features: {optimal_features}")  
```

#### Example 8  

```python
from sklearn.model_selection import RandomizedSearchCV  
from sklearn.ensemble import RandomForestClassifier  
from scipy.stats import randint  

# Define the parameter distribution for random search  
param_dist = {  
'n_estimators': randint(100, 500),  
'max_depth': randint(5, 20),  
'min_samples_split': randint(2, 11),  
'min_samples_leaf': randint(1, 11)  
}  

# Create a random forest classifier  
rf = RandomForestClassifier(random_state=42)  

# Set up the random search with cross-validation  
random_search = RandomizedSearchCV(rf, param_distributions=param_dist, n_iter=100, cv=5, random_state=42)  

# Fit the random search object to the data  
random_search.fit(X_train, y_train)  

# Print the best parameters and score  
print("Best parameters:", random_search.best_params_)  
print("Best cross-validation score:", random_search.best_score\_)  
```

### Chapter 13: Deep Learning with TensorFlow and PyTorch  

#### Example 1  

```python
import tensorflow as tf  
from tensorflow.keras import layers, models  
from tensorflow.keras.datasets import cifar10  

# Load and preprocess the CIFAR-10 dataset  
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()  
train_images, test_images = train_images / 255.0, test_images / 255.0  

# Define the CNN architecture  
model = models.Sequential(\[  
layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),  
layers.MaxPooling2D((2, 2)),  
layers.Conv2D(64, (3, 3), activation='relu'),  
layers.MaxPooling2D((2, 2)),  
layers.Conv2D(64, (3, 3), activation='relu'),  
layers.Flatten(),  
layers.Dense(64, activation='relu'),  
layers.Dense(10)  
\])  

# Compile the model  
model.compile(optimizer='adam',  
loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),  
metrics=\['accuracy'\])  

# Train the model  
history = model.fit(train_images, train_labels, epochs=10,  
validation_data=(test_images, test_labels))  

# Evaluate the model  
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)  
print(f"Test accuracy: {test_acc}")  
```

#### Example 2  


```python
import torch  
import torch.nn as nn  
import torch.optim as optim  
import torchvision  
import torchvision.transforms as transforms  

# Define the CNN architecture  
class Net(nn.Module):  
def __init__(self):  
super(Net, self).__init__()  
self.conv1 = nn.Conv2d(3, 32, 3)  
self.pool = nn.MaxPool2d(2, 2)  
self.conv2 = nn.Conv2d(32, 64, 3)  
self.conv3 = nn.Conv2d(64, 64, 3)  
self.fc1 = nn.Linear(64 * 4 * 4, 64)  
self.fc2 = nn.Linear(64, 10)  

def forward(self, x):  
x = self.pool(torch.relu(self.conv1(x)))  
x = self.pool(torch.relu(self.conv2(x)))  
x = torch.relu(self.conv3(x))  
x = x.view(-1, 64 * 4 * 4)  
x = torch.relu(self.fc1(x))  
x = self.fc2(x)  
return x  

# Load and preprocess the CIFAR-10 dataset  
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])  
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)  
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)  

# Initialize the network, loss function, and optimizer  
net = Net()  
criterion = nn.CrossEntropyLoss()  
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)  

# Train the network  
for epoch in range(10):  
running_loss = 0.0  
for i, data in enumerate(trainloader, 0):  
inputs, labels = data  
optimizer.zero_grad()  
outputs = net(inputs)  
loss = criterion(outputs, labels)  
loss.backward()  
optimizer.step()  
running_loss += loss.item()  
if i % 2000 == 1999:  
print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')  
running_loss = 0.0  

print('Finished Training')  
```

#### Example 3  


```python
import torch  
import torch.nn as nn  
import torchvision.models as models  
import torchvision.transforms as transforms  
from torch.utils.data import DataLoader  
from torchvision.datasets import ImageFolder  

# Load pre-trained ResNet model  
model = models.resnet18(pretrained=True)  

# Freeze all layers  
for param in model.parameters():  
param.requires_grad = False  

# Replace the last fully connected layer  
num_ftrs = model.fc.in_features  
model.fc = nn.Linear(num_ftrs, 10)  # 10 is the number of classes in your new task  

# Define transforms  
transform = transforms.Compose([  
transforms.Resize(256),  
transforms.CenterCrop(224),  
transforms.ToTensor(),  
transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  
])  

# Load your custom dataset  
train_dataset = ImageFolder('path/to/train/data', transform=transform)  
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)  

# Define loss function and optimizer  
criterion = nn.CrossEntropyLoss()  
optimizer = torch.optim.SGD(model.fc.parameters(), lr=0.001, momentum=0.9)  

# Train the model  
num_epochs = 10  
for epoch in range(num_epochs):  
for inputs, labels in train_loader:  
optimizer.zero_grad()  
outputs = model(inputs)  
loss = criterion(outputs, labels)  
loss.backward()  
optimizer.step()  
print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}')  

print('Finished Training')  
```

### Chapter 14: Feature Engineering Techniques 

#### Example 1  

```python
import pandas as pd  
import numpy as np  
from sklearn.preprocessing import StandardScaler, OneHotEncoder  
from sklearn.impute import SimpleImputer  
from sklearn.compose import ColumnTransformer  
from sklearn.pipeline import Pipeline  

# Load the dataset  
df = pd.read_csv('house_prices.csv')  
```

#### Example 2  

```python
# 1. Handling missing values  
df['LotFrontage'] = df['LotFrontage'].fillna(df['LotFrontage'].mean())  

# 2. Creating interaction features  
df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']  

# 3. Binning continuous variables  
df['AgeBin'] = pd.cut(df['YearBuilt'], bins=5, labels=['Very Old', 'Old', 'Medium', 'New', 'Very New'])  

# 4. Logarithmic transformation  
df['LogSalePrice'] = np.log(df['SalePrice'])  

# 5. One-hot encoding for categorical variables  
categorical_features = ['MSZoning', 'Street', 'LandContour', 'LotConfig', 'Neighborhood']  
df_encoded = pd.get_dummies(df, columns=categorical_features)  

# 6. Scaling numerical features  
numerical_features = ['LotArea', 'TotalSF', 'OverallQual', 'OverallCond']  
scaler = StandardScaler()  
df_encoded[numerical_features] = scaler.fit_transform(df_encoded[numerical_features])  
```

#### Example 3  

```python
# Define preprocessing steps  
numeric_transformer = Pipeline(steps=[  
('imputer', SimpleImputer(strategy='mean')),  
('scaler', StandardScaler())  
])  

categorical_transformer = Pipeline(steps=[  
('imputer', SimpleImputer(strategy='constant', fill_value='missing')),  
('onehot', OneHotEncoder(handle_unknown='ignore'))  
])  

preprocessor = ColumnTransformer(  
transformers=[  
('num', numeric_transformer, numerical_features),  
('cat', categorical_transformer, categorical_features)  
])  

# Create and fit the pipeline  
pipeline = Pipeline(steps=[('preprocessor', preprocessor)])  
X_transformed = pipeline.fit_transform(df)  
```

#### Example 4  

```python
import featuretools as ft  

# Assuming we have a pandas DataFrame 'df' with our data  
es = ft.EntitySet(id="house_prices")  
es = es.add_dataframe(dataframe_name="houses", dataframe=df, index="Id")  

# Generate features automatically  
feature_matrix, feature_defs = ft.dfs(entityset=es, target_dataframe_name="houses",  
trans_primitives=\["add_numeric", "multiply_numeric"\])  
Part 4: Evaluating Model Performance  
```

### Chapter 15: Classification Metrics: Confusion Matrix, Accuracy, Precision, and Recall  

#### Example 1  

```python
import numpy as np  
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score  
from sklearn.model_selection import train_test_split  
from sklearn.ensemble import RandomForestClassifier  
from sklearn.datasets import make_classification  

# Generate a sample dataset  
X, y = make_classification(n_samples=1000, n_classes=2, n_features=20, random_state=42)  

# Split the data into training and testing sets  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  

# Train a random forest classifier  
clf = RandomForestClassifier(n_estimators=100, random_state=42)  
clf.fit(X_train, y_train)  

# Make predictions on the test set  
y_pred = clf.predict(X_test)  

# Calculate the confusion matrix  
cm = confusion_matrix(y_test, y_pred)  
print("Confusion Matrix:")  
print(cm)  

# Calculate accuracy  
accuracy = accuracy_score(y_test, y_pred)  
print(f"Accuracy: {accuracy:.4f}")  

# Calculate precision  
precision = precision_score(y_test, y_pred)  
print(f"Precision: {precision:.4f}")  

# Calculate recall  
recall = recall_score(y_test, y_pred)  
print(f"Recall: {recall:.4f}")  
```

#### Example 2  

```python
from sklearn.metrics import precision_recall_curve  
import matplotlib.pyplot as plt  

# Assuming we have y_test and y_pred_proba (predicted probabilities) from a classifier  
precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)  

# Plot the Precision-Recall curve  
plt.figure(figsize=(10, 8))  
plt.plot(recall, precision, marker='.')  
plt.xlabel('Recall')  
plt.ylabel('Precision')  
plt.title('Precision-Recall Curve')  
plt.show()  

# Calculate the area under the Precision-Recall curve  
from sklearn.metrics import auc  
pr_auc = auc(recall, precision)  
print(f"Area under the Precision-Recall curve: {pr_auc:.4f}")  
```

#### Example 3  

```python
from sklearn.metrics import make_scorer  
from sklearn.model_selection import cross_val_score  

def custom_metric(y_true, y_pred):  
precision = precision_score(y_true, y_pred)  
recall = recall_score(y_true, y_pred)  
#### Example: Weighted harmonic mean of precision and recall  
beta = 0.5  # Adjust beta to change the weight of precision vs. recall  
return (1 + beta2) * (precision * recall) / ((beta2 * precision) + recall)  

# Create a scorer from the custom metric  
custom_scorer = make_scorer(custom_metric)  

# Use the custom scorer in cross-validation  
cv_scores = cross_val_score(clf, X, y, cv=5, scoring=custom_scorer)  
print(f"Cross-validation scores: {cv_scores}")  
print(f"Mean CV score: {cv_scores.mean():.4f}")  
```

### Chapter 16: ROC-AUC and F1-Score  

#### Example 1  

```python
import numpy as np  
from sklearn.metrics import roc_curve, auc, f1_score  
from sklearn.model_selection import train_test_split  
from sklearn.ensemble import RandomForestClassifier  
from sklearn.datasets import make_classification  
import matplotlib.pyplot as plt  

# Generate a random binary classification problem  
X, y = make_classification(n_samples=1000, n_classes=2, n_features=20, random_state=42)  

# Split the data into training and testing sets  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)  

# Train a random forest classifier  
clf = RandomForestClassifier(n_estimators=100, random_state=42)  
clf.fit(X_train, y_train)  

# Get predicted probabilities for the positive class  
y_pred_proba = clf.predict_proba(X_test)\[:, 1\]  

# Calculate ROC curve and ROC AUC  
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)  
roc_auc = auc(fpr, tpr)  

# Plot ROC curve  
plt.figure()  
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')  
plt.plot(\[0, 1\], \[0, 1\], color='navy', lw=2, linestyle='--')  
plt.xlim(\[0.0, 1.0\])  
plt.ylim(\[0.0, 1.05\])  
plt.xlabel('False Positive Rate')  
plt.ylabel('True Positive Rate')  
plt.title('Receiver Operating Characteristic (ROC) Curve')  
plt.legend(loc="lower right")  
plt.show()  

# Calculate F1-Score  
y_pred = clf.predict(X_test)  
f1 = f1_score(y_test, y_pred)  
print(f"F1-Score: {f1:.2f}")  
```

#### Example 2  

```python
from sklearn.calibration import CalibratedClassifierCV  
from sklearn.metrics import roc_auc_score, f1_score  
from sklearn.model_selection import StratifiedKFold  

# Use stratified k-fold cross-validation  
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)  

# Lists to store the results  
roc_auc_scores = []  
f1_scores = []  

for train_index, test_index in skf.split(X, y):  
X_train, X_test = X[train_index], X[test_index]  
y_train, y_test = y[train_index], y[test_index]  

# Train the model  
clf = RandomForestClassifier(n_estimators=100, random_state=42)  

# Use probability calibration  
calibrated_clf = CalibratedClassifierCV(clf, cv=5, method='sigmoid')  
calibrated_clf.fit(X_train, y_train)  

# Get calibrated predictions  
y_pred_proba = calibrated_clf.predict_proba(X_test)[:, 1]  
y_pred = calibrated_clf.predict(X_test)  

# Calculate and store ROC-AUC and F1-Score  
roc_auc_scores.append(roc_auc_score(y_test, y_pred_proba))  
f1_scores.append(f1_score(y_test, y_pred))  

# Print average scores  
print(f"Average ROC-AUC: {np.mean(roc_auc_scores):.2f} (+/- {np.std(roc_auc_scores):.2f})")  
print(f"Average F1-Score: {np.mean(f1_scores):.2f} (+/- {np.std(f1_scores):.2f})")  
```

### Chapter 17: Regression Metrics: MSE and RMSE  

#### Example 1  

```python
import numpy as np  
from sklearn.metrics import mean_squared_error  

# Sample data  
y_true = np.array(\[100, 150, 200, 250, 300\]) # Actual values  
y_pred = np.array(\[110, 155, 195, 240, 310\]) # Predicted values  

# Calculate MSE  
mse = mean_squared_error(y_true, y_pred)  

# Calculate RMSE  
rmse = np.sqrt(mse)  

print(f"Mean Squared Error: {mse}")  
print(f"Root Mean Squared Error: {rmse}")  
```

#### Example 2  

```python
import numpy as np  
from sklearn.metrics import mean_squared_error  

def weighted_mse(y_true, y_pred, weights):  
return np.average((y_true - y_pred)  2, weights=weights)  

# Sample data  
y_true = np.array([100, 150, 200, 250, 300])  
y_pred = np.array([110, 155, 195, 240, 310])  
weights = np.array([1, 2, 1, 0.5, 0.5])  #### Example weights  

# Calculate weighted MSE  
wmse = weighted_mse(y_true, y_pred, weights)  

# Calculate weighted RMSE  
wrmse = np.sqrt(wmse)  

print(f"Weighted Mean Squared Error: {wmse}")  
print(f"Weighted Root Mean Squared Error: {wrmse}")  

# Compare with standard MSE and RMSE  
mse = mean_squared_error(y_true, y_pred)  
rmse = np.sqrt(mse)  

print(f"Standard Mean Squared Error: {mse}")  
print(f"Standard Root Mean Squared Error: {rmse}")  
```

## Part 5: Statistics and Probability for Machine Learning  

### Chapter 18: Probability Distributions  

#### Example 1  

```python
import numpy as np  
import matplotlib.pyplot as plt  
from scipy import stats  

# Generate 10000 random numbers from a normal distribution  
mu, sigma = 0, 1 # mean and standard deviation  
data = np.random.normal(mu, sigma, 10000)  

# Plot histogram  
plt.hist(data, bins=50, density=True, alpha=0.7, color='skyblue')  

# Plot the theoretical PDF  
xmin, xmax = plt.xlim()  
x = np.linspace(xmin, xmax, 100)  
p = stats.norm.pdf(x, mu, sigma)  
plt.plot(x, p, 'k', linewidth=2)  

plt.title("Normal Distribution")  
plt.xlabel("Value")  
plt.ylabel("Frequency")  
plt.show()  
```

#### Example 2  

```python
from scipy.stats import norm  
import numpy as np  
import matplotlib.pyplot as plt  

def gaussian_mixture(x, mu1, sigma1, mu2, sigma2, w):  
return w * norm.pdf(x, mu1, sigma1) + (1 - w) * norm.pdf(x, mu2, sigma2)  

# Parameters for two Gaussian components  
mu1, sigma1 = 0, 1  
mu2, sigma2 = 3, 0.5  
w = 0.6  # weight of the first component  

# Generate x values  
x = np.linspace(-3, 6, 1000)  

# Calculate PDF of the mixture  
y = gaussian_mixture(x, mu1, sigma1, mu2, sigma2, w)  

# Plot the mixture and its components  
plt.figure(figsize=(10, 6))  
plt.plot(x, y, 'k', linewidth=2, label='Mixture')  
plt.plot(x, w * norm.pdf(x, mu1, sigma1), 'r--', label='Component 1')  
plt.plot(x, (1 - w) * norm.pdf(x, mu2, sigma2), 'b--', label='Component 2')  
plt.legend()  
plt.title("Gaussian Mixture Model")  
plt.xlabel("x")  
plt.ylabel("Probability Density")  
plt.show()  

# Sample from the mixture  
n_samples = 10000  
component = np.random.choice(2, size=n_samples, p=[w, 1-w])  
samples = np.where(component == 0,   
np.random.normal(mu1, sigma1, n_samples),  
np.random.normal(mu2, sigma2, n_samples))  

plt.figure(figsize=(10, 6))  
plt.hist(samples, bins=50, density=True, alpha=0.7)  
plt.plot(x, y, 'r', linewidth=2)  
plt.title("Samples from Gaussian Mixture Model")  
plt.xlabel("x")  
plt.ylabel("Frequency")  
plt.show()  
```

#### Example 3  

```python
import numpy as np  
from scipy import stats  
import matplotlib.pyplot as plt  

# Generate sample data from a gamma distribution  
true_shape, true_scale = 2.0, 2.0  
data = np.random.gamma(true_shape, true_scale, 10000)  

# Fit gamma distribution to the data using MLE  
fit_shape, fit_loc, fit_scale = stats.gamma.fit(data)  

# Plot the results  
x = np.linspace(0, 20, 200)  
plt.figure(figsize=(10, 6))  
plt.hist(data, bins=50, density=True, alpha=0.7, label='Data')  
plt.plot(x, stats.gamma.pdf(x, true_shape, scale=true_scale),   
'r-', lw=2, label='True Distribution')  
plt.plot(x, stats.gamma.pdf(x, fit_shape, loc=fit_loc, scale=fit_scale),   
'g--', lw=2, label='Fitted Distribution')  
plt.legend()  
plt.title("Fitting Gamma Distribution")  
plt.xlabel("x")  
plt.ylabel("Probability Density")  
plt.show()  

print(f"True parameters: shape={true_shape}, scale={true_scale}")  
print(f"Fitted parameters: shape={fit_shape:.2f}, scale={fit_scale:.2f}")  
```

### Chapter 19: Hypothesis Testing  

#### Example 1  

```python
import numpy as np  
from scipy import stats  

# Sample data: daily sales before and after the campaign  
before_campaign = \[120, 115, 130, 140, 110, 125, 135\]  
after_campaign = \[145, 155, 150, 160, 165, 140, 155\]  

# Perform two-sample t-test  
t_statistic, p_value = stats.ttest_ind(before_campaign, after_campaign)  

# Print results  
print(f"T-statistic: {t_statistic}")  
print(f"P-value: {p_value}")  

# Interpret results  
alpha = 0.05  
if p_value \< alpha:  
print("Reject the null hypothesis. There is a significant difference in sales.")  
else:  
print("Fail to reject the null hypothesis. There is not enough evidence to conclude a significant difference.")  
```

#### Example 2  

```python
import numpy as np  
from scipy import stats  

def bootstrap_mean_diff(sample1, sample2, num_bootstrap=10000):  
n1, n2 = len(sample1), len(sample2)  
combined = np.concatenate([sample1, sample2])  
mean_diffs = []  

for _ in range(num_bootstrap):  
boot_sample = np.random.choice(combined, size=n1+n2, replace=True)  
boot_sample1 = boot_sample[:n1]  
boot_sample2 = boot_sample[n1:]  
mean_diff = np.mean(boot_sample2) - np.mean(boot_sample1)  
mean_diffs.append(mean_diff)  

return mean_diffs  

# Using the same data as before  
before_campaign = [120, 115, 130, 140, 110, 125, 135]  
after_campaign = [145, 155, 150, 160, 165, 140, 155]  

# Perform bootstrap  
bootstrap_diffs = bootstrap_mean_diff(before_campaign, after_campaign)  

# Calculate confidence interval  
confidence_interval = np.percentile(bootstrap_diffs, [2.5, 97.5])  

# Calculate p-value  
observed_diff = np.mean(after_campaign) - np.mean(before_campaign)  
p_value = np.mean(np.abs(bootstrap_diffs) >= np.abs(observed_diff))  

print(f"95% Confidence Interval: {confidence_interval}")  
print(f"Bootstrap p-value: {p_value}")  
```

### Chapter 20: Bayes' Theorem  

#### Example 1  

```python
import numpy as np  
from scipy import stats  

# Sample data: daily sales before and after the campaign  
before_campaign = \[120, 115, 130, 140, 110, 125, 135\]  
after_campaign = \[145, 155, 150, 160, 165, 140, 155\]  

# Perform two-sample t-test  
t_statistic, p_value = stats.ttest_ind(before_campaign, after_campaign)  

# Print results  
print(f"T-statistic: {t_statistic}")  
print(f"P-value: {p_value}")  

# Interpret results  
alpha = 0.05  
if p_value \< alpha:  
print("Reject the null hypothesis. There is a significant difference in sales.")  
else:  
print("Fail to reject the null hypothesis. There is not enough evidence to conclude a significant difference.")  
```

#### Example 2  

```python
import numpy as np  
from scipy import stats  

def bootstrap_mean_diff(sample1, sample2, num_bootstrap=10000):  
n1, n2 = len(sample1), len(sample2)  
combined = np.concatenate([sample1, sample2])  
mean_diffs = []  

for _ in range(num_bootstrap):  
boot_sample = np.random.choice(combined, size=n1+n2, replace=True)  
boot_sample1 = boot_sample[:n1]  
boot_sample2 = boot_sample[n1:]  
mean_diff = np.mean(boot_sample2) - np.mean(boot_sample1)  
mean_diffs.append(mean_diff)  

return mean_diffs  

# Using the same data as before  
before_campaign = [120, 115, 130, 140, 110, 125, 135]  
after_campaign = [145, 155, 150, 160, 165, 140, 155]  

# Perform bootstrap  
bootstrap_diffs = bootstrap_mean_diff(before_campaign, after_campaign)  

# Calculate confidence interval  
confidence_interval = np.percentile(bootstrap_diffs, [2.5, 97.5])  

# Calculate p-value  
observed_diff = np.mean(after_campaign) - np.mean(before_campaign)  
p_value = np.mean(np.abs(bootstrap_diffs) >= np.abs(observed_diff))  

print(f"95% Confidence Interval: {confidence_interval}")  
print(f"Bootstrap p-value: {p_value}")  
```

### Chapter 21: P-values and Statistical Significance  

#### Example 1  

```python
import numpy as np  
from scipy import stats  

# Generate sample data for two groups  
np.random.seed(42)  
group1 = np.random.normal(loc=10, scale=2, size=100)  
group2 = np.random.normal(loc=11, scale=2, size=100)  

# Perform independent t-test  
t_statistic, p_value = stats.ttest_ind(group1, group2)  

# Print results  
print(f"T-statistic: {t_statistic}")  
print(f"P-value: {p_value}")  

# Check for statistical significance  
alpha = 0.05  
if p_value \< alpha:  
print("The difference between the groups is statistically significant.")  
else:  
print("There is no statistically significant difference between the groups.")  
```

#### Example 2  

```python
import numpy as np  
from scipy import stats  

def permutation_test(group1, group2, num_permutations=10000):  
combined = np.concatenate([group1, group2])  
observed_diff = np.mean(group1) - np.mean(group2)  

diffs = []  
for _ in range(num_permutations):  
perm = np.random.permutation(combined)  
perm_group1 = perm[:len(group1)]  
perm_group2 = perm[len(group1):]  
diffs.append(np.mean(perm_group1) - np.mean(perm_group2))  

p_value = np.sum(np.abs(diffs) >= np.abs(observed_diff)) / num_permutations  
return p_value  

# Generate sample data  
np.random.seed(42)  
group1 = np.random.normal(loc=10, scale=2, size=100)  
group2 = np.random.normal(loc=11, scale=2, size=100)  

# Perform permutation test  
p_value = permutation_test(group1, group2)  
print(f"Permutation test p-value: {p_value}")  
```

## Part 6: Real-world Machine Learning Applications  

### Chapter 22: Case Studies in Machine Learning  

#### Example 1  

```python
import pandas as pd  
import numpy as np  
from sklearn.model_selection import train_test_split  
from sklearn.preprocessing import StandardScaler  
from sklearn.ensemble import RandomForestClassifier  
from sklearn.metrics import accuracy_score, classification_report  

# Load the dataset  
data = pd.read_csv('patient_data.csv')  

# Preprocess the data  
X = data.drop('readmitted', axis=1)  
y = data\['readmitted'\]  

# Split the data into training and testing sets  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  

# Scale the features  
scaler = StandardScaler()  
X_train_scaled = scaler.fit_transform(X_train)  
X_test_scaled = scaler.transform(X_test)  

# Train a Random Forest classifier  
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)  
rf_classifier.fit(X_train_scaled, y_train)  

# Make predictions on the test set  
y_pred = rf_classifier.predict(X_test_scaled)  

# Evaluate the model  
accuracy = accuracy_score(y_test, y_pred)  
print(f"Accuracy: {accuracy:.2f}")  
print(classification_report(y_test, y_pred))  
```

#### Example 2  

```python
import tensorflow as tf  
from tensorflow.keras.models import Sequential  
from tensorflow.keras.layers import Dense, Dropout  

model = Sequential([  
Dense(64, activation='relu', input_shape=(n_features,)),  
Dropout(0.5),  
Dense(32, activation='relu'),  
Dropout(0.5),  
Dense(1, activation='sigmoid')  
])  

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])  
```

#### Example 3  

```python
import xgboost as xgb  
from sklearn.model_selection import train_test_split  
from sklearn.metrics import accuracy_score  

# Prepare the data  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  

# Create DMatrix for XGBoost  
dtrain = xgb.DMatrix(X_train, label=y_train)  
dtest = xgb.DMatrix(X_test, label=y_test)  

# Set parameters  
params = {  
'max_depth': 3,  
'eta': 0.1,  
'objective': 'binary:logistic',  
'eval_metric': 'logloss'  
}  

# Train the model  
num_rounds = 100  
model = xgb.train(params, dtrain, num_rounds)  

# Make predictions  
y_pred = model.predict(dtest)  
y_pred_binary = [1 if p > 0.5 else 0 for p in y_pred]  

# Evaluate the model  
accuracy = accuracy_score(y_test, y_pred_binary)  
print(f"Accuracy: {accuracy:.2f}")  
```

### Chapter 23: Handling Data Challenges  

#### Example 1  

```python
import pandas as pd  
import numpy as np  
from sklearn.impute import SimpleImputer  
from sklearn.preprocessing import StandardScaler  

# Load the dataset  
df = pd.read_csv('raw_data.csv')  

# Display basic information about the dataset  
print(df.info())  
print(df.describe())  
```

#### Example 2  

```python
# Identify columns with missing values  
columns_with_missing = df.columns[df.isnull().any()].tolist()  

# Impute missing values using mean strategy  
imputer = SimpleImputer(strategy='mean')  
df[columns_with_missing] = imputer.fit_transform(df[columns_with_missing])  

# Verify that missing values have been handled  
print(df.isnull().sum())  
```

#### Example 3  

```python
# Convert 'date' column to datetime format  
df['date'] = pd.to_datetime(df['date'], errors='coerce')  

# Extract relevant features from the date  
df['year'] = df['date'].dt.year  
df['month'] = df['date'].dt.month  
df['day'] = df['date'].dt.day  

# Drop the original 'date' column  
df = df.drop('date', axis=1)  
```

#### Example 4  

```python
def handle_outliers(df, column):  
Q1 = df[column].quantile(0.25)  
Q3 = df[column].quantile(0.75)  
IQR = Q3 - Q1  
lower_bound = Q1 - 1.5 * IQR  
upper_bound = Q3 + 1.5 * IQR  

df[column] = np.where(df[column] > upper_bound, upper_bound,  
np.where(df[column] < lower_bound, lower_bound, df[column]))  
return df  

# Apply outlier handling to numerical columns  
numerical_columns = df.select_dtypes(include=[np.number]).columns  
for column in numerical_columns:  
df = handle_outliers(df, column)  
```

#### Example 5  

```python
from sklearn.decomposition import PCA  

# Apply PCA to reduce dimensionality  
pca = PCA(n_components=0.95)  # Preserve 95% of variance  
df_pca = pca.fit_transform(df[numerical_columns])  

# Create a new DataFrame with the reduced features  
df_reduced = pd.DataFrame(df_pca, columns=[f'PC_{i+1}' for i in range(df_pca.shape[1])])  

print(f"Original number of features: {len(numerical_columns)}")  
print(f"Number of features after PCA: {df_reduced.shape\[1\]}")  
```

### Chapter 24: Improving Model Accuracy  

#### Example 1  

```python
import pandas as pd  
import numpy as np  
from sklearn.model_selection import train_test_split, cross_val_score  
from sklearn.preprocessing import StandardScaler  
from sklearn.ensemble import RandomForestClassifier  
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report  
from sklearn.model_selection import GridSearchCV  
import matplotlib.pyplot as plt  
import seaborn as sns  

# Load the dataset  
data = pd.read_csv('telecom_churn_data.csv')  

# Display basic information about the dataset  
print(data.info())  
print(data.describe())  
```

#### Example 2  

```python
# Handle missing values  
data = data.dropna()  

# Encode categorical variables  
data = pd.get_dummies(data, columns=['gender', 'contract_type'])  

# Scale numerical features  
scaler = StandardScaler()  
numerical_features = ['tenure', 'monthly_charges', 'total_charges']  
data[numerical_features] = scaler.fit_transform(data[numerical_features])  

# Split the data into features and target  
X = data.drop('churn', axis=1)  
y = data['churn']  

# Split the data into training and testing sets  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  
```

#### Example 3  

```python
# Train a baseline Random Forest model  
rf_baseline = RandomForestClassifier(random_state=42)  
rf_baseline.fit(X_train, y_train)  

# Make predictions on the test set  
y_pred = rf_baseline.predict(X_test)  

# Evaluate the baseline model  
print("Baseline Model Accuracy:", accuracy_score(y_test, y_pred))  
print("\nClassification Report:\n", classification_report(y_test, y_pred))  
```

#### Example 4  

```python
# Get feature importances  
importances = rf_baseline.feature_importances_  
feature_importances = pd.Series(importances, index=X.columns).sort_values(ascending=False)  

# Plot feature importances  
plt.figure(figsize=(10, 6))  
feature_importances.plot(kind='bar')  
plt.title("Feature Importances")  
plt.xlabel("Features")  
plt.ylabel("Importance")  
plt.tight_layout()  
plt.show()  
```

#### Example 5  

```python
# Select top features  
top_features = feature_importances.index[:10]  
X_train_top = X_train[top_features]  
X_test_top = X_test[top_features]  

# Train model with selected features  
rf_top_features = RandomForestClassifier(random_state=42)  
rf_top_features.fit(X_train_top, y_train)  

# Evaluate model with selected features  
y_pred_top = rf_top_features.predict(X_test_top)  
print("Model Accuracy with Top Features:", accuracy_score(y_test, y_pred_top))  
```    

#### Example 6  

```python
# Define hyperparameter grid  
param_grid = {  
'n_estimators': [100, 200, 300],  
'max_depth': [5, 10, 15, None],  
'min_samples_split': [2, 5, 10],  
'min_samples_leaf': [1, 2, 4]  
}  

# Perform grid search  
grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, n_jobs=-1)  
grid_search.fit(X_train_top, y_train)  

# Get best model  
best_rf = grid_search.best_estimator_  

# Evaluate best model  
y_pred_best = best_rf.predict(X_test_top)  
print("Best Model Accuracy:", accuracy_score(y_test, y_pred_best))  
print("\nBest Hyperparameters:", grid_search.best_params\_)  
```

### Chapter 25: Ethical Considerations in ML Projects  

#### Example 1  

```python
import shap  
import numpy as np  
from sklearn.model_selection import train_test_split  
from sklearn.ensemble import RandomForestClassifier  

# Assume we have a dataset X and target variable y  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  

# Train a random forest classifier  
model = RandomForestClassifier(n_estimators=100, random_state=42)  
model.fit(X_train, y_train)  

# Explain the model's predictions using SHAP  
explainer = shap.TreeExplainer(model)  
shap_values = explainer.shap_values(X_test)  

# Visualize the results  
shap.summary_plot(shap_values, X_test, plot_type="bar")  
```

#### Example 2  

```python
import numpy as np  
from diffprivlib import mechanisms  

# Assume we have a sensitive dataset  
sensitive_data = np.array([1, 2, 3, 4, 5])  

# Define the privacy parameter epsilon (lower values provide stronger privacy guarantees)  
epsilon = 0.1  

# Create a Laplace mechanism for adding noise  
mech = mechanisms.Laplace(epsilon=epsilon, sensitivity=1)  

# Add noise to the data  
private_data = np.array([mech.randomise(x) for x in sensitive_data])  

print("Original data:", sensitive_data)  
print("Private data:", private_data)  
```

#### Example 3  

```python
from aif360.datasets import BinaryLabelDataset  
from aif360.metrics import BinaryLabelDatasetMetric  

# Assume we have predictions and true labels for different demographic groups  
predictions = [...]  
true_labels = [...]  
protected_attribute = [...]  # e.g., 0 for group A, 1 for group B  

# Create a BinaryLabelDataset  
dataset = BinaryLabelDataset(favorable_label=1, unfavorable_label=0,   
df=pd.DataFrame({'predictions': predictions,   
'true_labels': true_labels,   
'protected_attribute': protected_attribute}),  
label_names=['true_labels'],  
protected_attribute_names=['protected_attribute'])  

# Calculate fairness metrics  
metric = BinaryLabelDatasetMetric(dataset, unprivileged_groups=[{'protected_attribute': 0}],   
privileged_groups=[{'protected_attribute': 1}])  

print("Disparate Impact:", metric.disparate_impact())  
print("Statistical Parity Difference:", metric.statistical_parity_difference())  
print("Equal Opportunity Difference:", metric.equal_opportunity_difference())  
```

#### Example 4  

```python
import mlflow  
from sklearn.ensemble import RandomForestClassifier  
from sklearn.metrics import accuracy_score  

# Start an MLflow run  
with mlflow.start_run():  
# Train your model  
model = RandomForestClassifier(n_estimators=100, random_state=42)  
model.fit(X_train, y_train)  

# Log model parameters  
mlflow.log_param("n_estimators", 100)  
mlflow.log_param("random_state", 42)  

# Log model performance  
y_pred = model.predict(X_test)  
accuracy = accuracy_score(y_test, y_pred)  
mlflow.log_metric("accuracy", accuracy)  

# Save the model  
mlflow.sklearn.log_model(model, "random_forest_model")  
Part 7: Emerging Trends in Machine Learning  
```
### Chapter 26: AutoML: Automated Machine Learning  

#### Example 1  
!pip install auto-sklearn  

#### Example 2  
```python
import autosklearn.classification  
import sklearn.model_selection  
import sklearn.datasets  
import sklearn.metrics  

# Load a sample dataset (in this case, the breast cancer dataset)  
X, y = sklearn.datasets.load_breast_cancer(return_X_y=True)  

# Split the data into training and test sets  
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(  
X, y, random_state=1  
)  

# Create and configure the AutoML classifier  
automl = autosklearn.classification.AutoSklearnClassifier(  
time_left_for_this_task=300,  
per_run_time_limit=30,  
tmp_folder='/tmp/autosklearn_classification_example',  
output_folder='/tmp/autosklearn_classification_example_out',  
delete_tmp_folder_after_terminate=True,  
ensemble_size=50,  
initial_configurations_via_metalearning=25,  
seed=1,  
)  

# Fit the AutoML classifier  
automl.fit(X_train, y_train)  

# Make predictions on the test set  
y_pred = automl.predict(X_test)  

# Print the accuracy score  
print("Accuracy score:", sklearn.metrics.accuracy_score(y_test, y_pred))  

# Print the best model found by AutoML  
print(automl.show_models())  
```

### Chapter 27: Transfer Learning  

#### Example 1  

```python
import torch  
import torchvision  
from torchvision import transforms  
from torch.utils.data import DataLoader  
from torchvision.datasets import ImageFolder  
import torch.nn as nn  
import torch.optim as optim  

# Define data transforms  
data_transforms = transforms.Compose(\[  
transforms.Resize((224, 224)),  
transforms.ToTensor(),  
transforms.Normalize(mean=\[0.485, 0.456, 0.406\], std=\[0.229, 0.224, 0.225\])  
\])  

# Load and prepare the dataset  
train_dataset = ImageFolder('path/to/train/data', transform=data_transforms)  
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)  

# Load a pre-trained ResNet model  
model = torchvision.models.resnet50(pretrained=True)  

# Freeze all layers  
for param in model.parameters():  
param.requires_grad = False  

# Replace the final fully connected layer  
num_ftrs = model.fc.in_features  
model.fc = nn.Linear(num_ftrs, 2) # 2 classes: cats and dogs  

# Define loss function and optimizer  
criterion = nn.CrossEntropyLoss()  
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)  

# Move model to GPU if available  
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  
model = model.to(device)  
```

#### Example 2  

```python
num_epochs = 10  

for epoch in range(num_epochs):  
model.train()  
running_loss = 0.0  

for inputs, labels in train_loader:  
inputs = inputs.to(device)  
labels = labels.to(device)  

optimizer.zero_grad()  

outputs = model(inputs)  
loss = criterion(outputs, labels)  
loss.backward()  
optimizer.step()  

running_loss += loss.item() * inputs.size(0)  

epoch_loss = running_loss / len(train_loader.dataset)  
print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")  

print("Training complete!")  
```

### Chapter 28: Explainable AI (XAI)  

#### Example 1  

```python
import shap  
from sklearn.ensemble import RandomForestClassifier  
from sklearn.datasets import load_iris  

# Load data and train model  
X, y = load_iris(return_X_y=True)  
model = RandomForestClassifier(n_estimators=100, random_state=42)  
model.fit(X, y)  

# Explain the model's predictions using SHAP  
explainer = shap.TreeExplainer(model)  
shap_values = explainer.shap_values(X)  

# Visualize the first prediction's explanation  
shap.force_plot(explainer.expected_value\[0\], shap_values\[0\]\[0\], X\[0\])  

# Visualize feature importance  
shap.summary_plot(shap_values, X, plot_type="bar")  
```

#### Example 2  

```python
import numpy as np  
from sklearn.feature_extraction.text import TfidfVectorizer  
from sklearn.pipeline import make_pipeline  
from sklearn.naive_bayes import MultinomialNB  
from lime.lime_text import LimeTextExplainer  

# Prepare data and train model  
texts = ["This is a positive review", "This movie was terrible", "I love this product"]  
labels = [1, 0, 1]  # 1 for positive, 0 for negative  
vectorizer = TfidfVectorizer()  
classifier = MultinomialNB()  
pipeline = make_pipeline(vectorizer, classifier)  
pipeline.fit(texts, labels)  

# Create a LIME explainer  
explainer = LimeTextExplainer(class_names=["negative", "positive"])  

# Explain a prediction  
idx = 0  
exp = explainer.explain_instance(texts[idx], pipeline.predict_proba, num_features=6)  
exp.show_in_notebook()  
Part 8: Designing Machine Learning Systems  
```

### Chapter 29: Scalability in Machine Learning  

#### Example 1  

```python
from pyspark.sql import SparkSession  
from pyspark.ml.classification import LogisticRegression  
from pyspark.ml.feature import VectorAssembler  
from pyspark.ml.evaluation import BinaryClassificationEvaluator  

# Create a Spark session  
spark = SparkSession.builder.appName("ScalableML").getOrCreate()  

# Load the dataset (assuming it's in CSV format)  
data = spark.read.csv("large_dataset.csv", header=True, inferSchema=True)  

# Prepare the features and label  
feature_columns = \["feature1", "feature2", "feature3", "feature4"\]  
assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")  
data_assembled = assembler.transform(data)  

# Split the data into training and testing sets  
train_data, test_data = data_assembled.randomSplit(\[0.8, 0.2\], seed=42)  

# Create and train the logistic regression model  
lr = LogisticRegression(featuresCol="features", labelCol="label", maxIter=10)  
model = lr.fit(train_data)  

# Make predictions on the test data  
predictions = model.transform(test_data)  

# Evaluate the model  
evaluator = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction")  
auc = evaluator.evaluate(predictions)  
print(f"AUC: {auc}")  

# Save the model for future use  
model.save("scalable_lr_model")  

# Stop the Spark session  
spark.stop()  
```

#### Example 2  

```python
from pyspark.ml.feature import HashingTF, IDF  

# Assume 'text' is a column containing text data  
hashing_tf = HashingTF(inputCol="text", outputCol="raw_features", numFeatures=10000)  
featurized_data = hashing_tf.transform(data)  

idf = IDF(inputCol="raw_features", outputCol="features")  
idf_model = idf.fit(featurized_data)  
rescaled_data = idf_model.transform(featurized_data)  
```

### Chapter 30: Model Deployment Strategies  

#### Example 1  

```python
import joblib  
from sklearn.datasets import load_iris  
from sklearn.model_selection import train_test_split  
from sklearn.ensemble import RandomForestClassifier  

# Load data  
iris = load_iris()  
X, y = iris.data, iris.target  

# Split the data  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  

# Train the model  
model = RandomForestClassifier(n_estimators=100, random_state=42)  
model.fit(X_train, y_train)  

# Save the model  
joblib.dump(model, 'iris_model.joblib')  
```

#### Example 2  

```python
from flask import Flask, request, jsonify  
import joblib  
import numpy as np  

app = Flask(__name__)  

# Load the model  
model = joblib.load('iris_model.joblib')  

@app.route('/predict', methods=['POST'])  
def predict():  
# Get the data from the POST request  
data = request.json['data']  

# Make prediction using model  
prediction = model.predict(np.array(data).reshape(1, -1))  

# Return the prediction  
return jsonify({'prediction': int(prediction[0])})  

if __name__ == '__main__':  
app.run(debug=True)  
```

### Chapter 31: Handling Large-Scale Data  

#### Example 1  

```python
from pyspark.sql import SparkSession  
from pyspark.sql.functions import col, sum, avg  
import matplotlib.pyplot as plt  

# Initialize a Spark session  
spark = SparkSession.builder \\  
.appName("LargeScaleDataProcessing") \\  
.getOrCreate()  

# Load the large dataset  
transactions = spark.read.csv("path/to/large/dataset.csv", header=True, inferSchema=True)  
```

#### Example 2  

```python
# Filter and aggregate data  
daily_sales = transactions.groupBy("date") \  
.agg(sum("amount").alias("total_sales"),   
avg("amount").alias("average_sale"))  

# Calculate total revenue  
total_revenue = daily_sales.agg(sum("total_sales")).collect()[0][0]  

# Find the top 10 days with highest sales  
top_sales_days = daily_sales.orderBy(col("total_sales").desc()).limit(10)  

# Convert Spark DataFrame to Pandas for visualization  
pandas_df = top_sales_days.toPandas()  

# Create a bar plot of top sales days  
plt.figure(figsize=(12, 6))  
plt.bar(pandas_df["date"], pandas_df["total_sales"])  
plt.title("Top 10 Sales Days")  
plt.xlabel("Date")  
plt.ylabel("Total Sales")  
plt.xticks(rotation=45)  
plt.tight_layout()  
plt.show()  

print(f"Total Revenue: ${total_revenue:,.2f}")  
```

#### Example 3  


```python
from pyspark.sql.window import Window  
from pyspark.sql.functions import rank, dense_rank  

# Define a window specification  
window_spec = Window.partitionBy("category").orderBy(col("amount").desc())  

# Apply window functions  
ranked_transactions = transactions.withColumn("rank", rank().over(window_spec)) \  
.withColumn("dense_rank", dense_rank().over(window_spec))  

# Show top 5 transactions by rank within each category  
ranked_transactions.filter(col("rank") <= 5).show()  
```

#### Example 4  


```python
from pyspark.sql.functions import pandas_udf  
from pyspark.sql.types import DoubleType  

@pandas_udf(DoubleType())  
def complex_calculation(amount: pd.Series) -> pd.Series:  
return amount * 2 + amount.mean()  

# Apply the Pandas UDF to the dataset  
result = transactions.withColumn("calculated_value", complex_calculation(col("amount")))
```
