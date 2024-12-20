## Improving Model Accuracy Through Feature Selection
Slide 1: Feature Selection: The Key to Model Performance

Feature selection is a crucial step in machine learning that involves identifying the most relevant features for a given task. By focusing on key features, we can improve model accuracy, reduce complexity, and enhance efficiency. Let's explore techniques and strategies for effective feature selection.

```python
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.datasets import load_iris

# Load the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Apply SelectKBest to choose top 2 features
selector = SelectKBest(f_classif, k=2)
X_selected = selector.fit_transform(X, y)

# Display selected features
selected_features = iris.feature_names[selector.get_support()]
print("Selected features:", selected_features)
```

Slide 2: Correlation Analysis

One of the simplest methods for feature selection is correlation analysis. By examining the correlation between features and the target variable, we can identify which features are most strongly related to the outcome we're trying to predict.

```python
import seaborn as sns
import matplotlib.pyplot as plt

# Create a DataFrame from the Iris dataset
df = pd.DataFrame(X, columns=iris.feature_names)
df['target'] = y

# Calculate correlation matrix
corr_matrix = df.corr()

# Plot heatmap of correlations
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap of Iris Dataset')
plt.show()
```

Slide 3: Feature Importance with Random Forests

Random Forests provide a built-in method for assessing feature importance. By analyzing how much each feature contributes to the accuracy of the model, we can rank features by their importance and select the most influential ones.

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

# Get feature importances
importances = rf_classifier.feature_importances_
feature_importances = pd.Series(importances, index=iris.feature_names).sort_values(ascending=False)

# Plot feature importances
plt.figure(figsize=(10, 6))
feature_importances.plot(kind='bar')
plt.title('Feature Importances in Iris Dataset')
plt.xlabel('Features')
plt.ylabel('Importance')
plt.tight_layout()
plt.show()
```

Slide 4: Recursive Feature Elimination (RFE)

Recursive Feature Elimination is an iterative technique that starts with all features and progressively eliminates the least important ones. This method helps identify a subset of features that contributes most to the model's performance.

```python
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

# Create a logistic regression model
logistic = LogisticRegression(random_state=42)

# Create the RFE object and specify the number of features to select
rfe = RFE(estimator=logistic, n_features_to_select=2)

# Fit RFE
rfe = rfe.fit(X, y)

# Get the selected features
selected_features = [feature for feature, selected in zip(iris.feature_names, rfe.support_) if selected]
print("Selected features:", selected_features)

# Plot feature ranking
plt.figure(figsize=(10, 6))
plt.bar(range(len(rfe.ranking_)), rfe.ranking_)
plt.title('Feature Ranking (lower = more important)')
plt.xlabel('Features')
plt.ylabel('Ranking')
plt.xticks(range(len(rfe.ranking_)), iris.feature_names, rotation=45)
plt.tight_layout()
plt.show()
```

Slide 5: Lasso Regularization for Feature Selection

Lasso (L1) regularization can be used for feature selection by shrinking the coefficients of less important features to zero. This technique is particularly useful when dealing with high-dimensional datasets.

```python
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Fit Lasso model
lasso = Lasso(alpha=0.1)
lasso.fit(X_scaled, y)

# Get feature coefficients
feature_coefficients = pd.Series(lasso.coef_, index=iris.feature_names).sort_values(key=abs, ascending=False)

# Plot feature coefficients
plt.figure(figsize=(10, 6))
feature_coefficients.plot(kind='bar')
plt.title('Lasso Coefficients for Iris Dataset')
plt.xlabel('Features')
plt.ylabel('Coefficient')
plt.tight_layout()
plt.show()
```

Slide 6: Principal Component Analysis (PCA)

PCA is a dimensionality reduction technique that can be used for feature selection. It transforms the original features into a new set of uncorrelated features called principal components, which capture the most variance in the data.

```python
from sklearn.decomposition import PCA

# Apply PCA
pca = PCA()
X_pca = pca.fit_transform(X)

# Calculate explained variance ratio
explained_variance_ratio = pca.explained_variance_ratio_

# Plot cumulative explained variance ratio
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(explained_variance_ratio) + 1), np.cumsum(explained_variance_ratio))
plt.title('Cumulative Explained Variance Ratio')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance Ratio')
plt.grid(True)
plt.show()

# Select top 2 principal components
pca_2 = PCA(n_components=2)
X_pca_2 = pca_2.fit_transform(X)

# Plot data points in 2D space
plt.figure(figsize=(10, 6))
scatter = plt.scatter(X_pca_2[:, 0], X_pca_2[:, 1], c=y, cmap='viridis')
plt.title('Iris Dataset in 2D PCA Space')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.colorbar(scatter)
plt.show()
```

Slide 7: Mutual Information for Feature Selection

Mutual Information is a powerful technique for measuring the relationship between features and the target variable. It can capture both linear and non-linear dependencies, making it useful for a wide range of datasets.

```python
from sklearn.feature_selection import mutual_info_classif

# Calculate mutual information scores
mi_scores = mutual_info_classif(X, y)

# Create a Series of feature names and their mutual information scores
mi_series = pd.Series(mi_scores, index=iris.feature_names).sort_values(ascending=False)

# Plot mutual information scores
plt.figure(figsize=(10, 6))
mi_series.plot(kind='bar')
plt.title('Mutual Information Scores for Iris Dataset Features')
plt.xlabel('Features')
plt.ylabel('Mutual Information Score')
plt.tight_layout()
plt.show()
```

Slide 8: Cross-Validation for Feature Selection

Cross-validation can be used to assess the performance of different feature subsets and select the most robust combination. This technique helps prevent overfitting and ensures that the selected features generalize well to unseen data.

```python
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.svm import SVC

# Create an SVM classifier
svm = SVC(kernel='linear', random_state=42)

# Perform forward feature selection
sfs = SequentialFeatureSelector(svm, n_features_to_select=2, direction='forward', cv=5)
sfs.fit(X, y)

# Get selected features
selected_features = [iris.feature_names[i] for i in sfs.get_support(indices=True)]
print("Selected features:", selected_features)

# Evaluate model performance with selected features
scores = cross_val_score(svm, X[:, sfs.get_support()], y, cv=5)
print("Cross-validation scores:", scores)
print("Mean score:", scores.mean())
```

Slide 9: Handling Multicollinearity

Multicollinearity occurs when features are highly correlated with each other. Addressing this issue can improve model stability and interpretability. Let's explore techniques to detect and handle multicollinearity.

```python
import numpy as np
from sklearn.preprocessing import StandardScaler

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Calculate the correlation matrix
corr_matrix = np.corrcoef(X_scaled.T)

# Plot the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', xticklabels=iris.feature_names, yticklabels=iris.feature_names)
plt.title('Correlation Matrix of Iris Dataset Features')
plt.tight_layout()
plt.show()

# Calculate Variance Inflation Factor (VIF)
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif_data = pd.DataFrame()
vif_data["feature"] = iris.feature_names
vif_data["VIF"] = [variance_inflation_factor(X_scaled, i) for i in range(X_scaled.shape[1])]
print(vif_data)
```

Slide 10: Feature Engineering

Feature engineering involves creating new features or transforming existing ones to capture more meaningful information. This process can lead to improved model performance and insights.

```python
# Create interaction features
X_engineered = X.()
X_engineered[:, 4] = X[:, 0] * X[:, 1]  # Interaction between sepal length and sepal width
X_engineered[:, 5] = X[:, 2] * X[:, 3]  # Interaction between petal length and petal width

# Create polynomial features
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)

# Compare original and engineered feature spaces
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

ax1.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis')
ax1.set_title('Original Feature Space')
ax1.set_xlabel('Sepal Length')
ax1.set_ylabel('Sepal Width')

ax2.scatter(X_engineered[:, 4], X_engineered[:, 5], c=y, cmap='viridis')
ax2.set_title('Engineered Feature Space')
ax2.set_xlabel('Sepal Length * Sepal Width')
ax2.set_ylabel('Petal Length * Petal Width')

plt.tight_layout()
plt.show()
```

Slide 11: Automated Feature Selection

Automated feature selection techniques can help streamline the process of identifying important features. Let's explore an example using the Boruta algorithm, which is based on Random Forests.

```python
from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier

# Create and fit Boruta feature selector
rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5)
boruta_selector = BorutaPy(rf, n_estimators='auto', verbose=2, random_state=42)
boruta_selector.fit(X, y)

# Get selected features
selected_features = [feature for feature, selected in zip(iris.feature_names, boruta_selector.support_) if selected]
print("Selected features:", selected_features)

# Plot feature importances
feature_ranks = pd.Series(boruta_selector.ranking_, index=iris.feature_names)
plt.figure(figsize=(10, 6))
feature_ranks.sort_values().plot(kind='bar')
plt.title('Feature Importance Rankings by Boruta')
plt.xlabel('Features')
plt.ylabel('Rank (lower is better)')
plt.tight_layout()
plt.show()
```

Slide 12: Ensemble Methods for Feature Selection

Ensemble methods combine multiple feature selection techniques to provide more robust and reliable results. Let's implement a simple ensemble approach using different feature selection methods.

```python
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier

# Define feature selection methods
methods = [
    ('f_classif', SelectKBest(f_classif, k=2)),
    ('mutual_info', SelectKBest(mutual_info_classif, k=2)),
    ('random_forest', SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=42), max_features=2))
]

# Apply each method and collect selected features
selected_features = {}
for name, method in methods:
    method.fit(X, y)
    selected = [feature for feature, selected in zip(iris.feature_names, method.get_support()) if selected]
    selected_features[name] = selected

# Count how many times each feature was selected
feature_counts = {feature: sum(feature in selected for selected in selected_features.values()) for feature in iris.feature_names}

# Plot feature selection frequency
plt.figure(figsize=(10, 6))
plt.bar(feature_counts.keys(), feature_counts.values())
plt.title('Feature Selection Frequency in Ensemble')
plt.xlabel('Features')
plt.ylabel('Number of Times Selected')
plt.tight_layout()
plt.show()

print("Selected features by each method:")
for name, features in selected_features.items():
    print(f"{name}: {features}")
```

Slide 13: Real-Life Example: Predicting Customer Churn

Let's apply feature selection techniques to a real-world problem: predicting customer churn in a telecommunication company. We'll use a subset of features from the Telco Customer Churn dataset.

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectFromModel

# Load and preprocess the data
data = pd.read_csv('telco_customer_churn.csv')
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
data = data.dropna()

# Select relevant features and target variable
features = ['tenure', 'MonthlyCharges', 'TotalCharges', 'InternetService', 'Contract', 'PaymentMethod']
X = pd.get_dummies(data[features], drop_first=True)
y = (data['Churn'] == 'Yes').astype(int)

# Split the data and scale features
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a Random Forest classifier and select important features
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_scaled, y_train)
selector = SelectFromModel(rf, prefit=True)
X_train_selected = selector.transform(X_train_scaled)
X_test_selected = selector.transform(X_test_scaled)

# Train models with all features and selected features
rf_all = RandomForestClassifier(n_estimators=100, random_state=42)
rf_all.fit(X_train_scaled, y_train)
rf_selected = RandomForestClassifier(n_estimators=100, random_state=42)
rf_selected.fit(X_train_selected, y_train)

# Evaluate the models
print("Accuracy with all features:", accuracy_score(y_test, rf_all.predict(X_test_scaled)))
print("Accuracy with selected features:", accuracy_score(y_test, rf_selected.predict(X_test_selected)))

# Display selected features
selected_features = X.columns[selector.get_support()].tolist()
print("Selected features:", selected_features)
```

Slide 14: Real-Life Example: Image Classification Feature Extraction

In this example, we'll explore feature extraction for image classification using a pre-trained convolutional neural network (CNN) as a feature extractor. We'll use a small subset of the CIFAR-10 dataset for demonstration purposes.

```python
import numpy as np
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load pre-trained VGG16 model
base_model = VGG16(weights='imagenet', include_top=False)

# Function to extract features from an image
def extract_features(img_path):
    img = load_img(img_path, target_size=(224, 224))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = base_model.predict(x)
    return features.flatten()

# Load and preprocess a small subset of CIFAR-10 images (assuming you have the data)
images = []
labels = []
for i in range(1000):  # Load 1000 images for demonstration
    img_path = f'path/to/cifar10/images/image_{i}.png'
    label = ...  # Get the corresponding label
    features = extract_features(img_path)
    images.append(features)
    labels.append(label)

X = np.array(images)
y = np.array(labels)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train an SVM classifier
svm = SVC(kernel='rbf', random_state=42)
svm.fit(X_train, y_train)

# Evaluate the model
y_pred = svm.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
```

Slide 15: Additional Resources

For those interested in diving deeper into feature selection and model optimization techniques, here are some valuable resources:

1. "Feature Engineering and Selection: A Practical Approach for Predictive Models" by Max Kuhn and Kjell Johnson ArXiv: [https://arxiv.org/abs/2001.03994](https://arxiv.org/abs/2001.03994)
2. "An Introduction to Variable and Feature Selection" by Isabelle Guyon and André Elisseeff ArXiv: [https://arxiv.org/abs/cs/0307015](https://arxiv.org/abs/cs/0307015)
3. "Feature Selection for Machine Learning: The Complete Guide" by Sebastian Raschka ArXiv: [https://arxiv.org/abs/1811.10404](https://arxiv.org/abs/1811.10404)
4. "A Survey of Deep Learning Techniques for Feature Selection" by Ren et al. ArXiv: [https://arxiv.org/abs/2007.08168](https://arxiv.org/abs/2007.08168)
5. "Automated Machine Learning: Methods, Systems, Challenges" edited by Frank Hutter, Lars Kotthoff, and Joaquin Vanschoren ArXiv: [https://arxiv.org/abs/1904.12054](https://arxiv.org/abs/1904.12054)

These resources provide in-depth discussions on various feature selection techniques, their applications, and best practices for model optimization.
