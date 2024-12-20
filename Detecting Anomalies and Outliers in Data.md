## Detecting Anomalies and Outliers in Data
Slide 1: Flag Outliers

Outliers are data points that significantly differ from other observations. Flagging outliers is crucial for data analysis and model performance.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Generate sample data
np.random.seed(42)
data = np.random.normal(0, 1, 1000)
outliers = np.array([5, 7, -8, 10])
data = np.concatenate([data, outliers])

# Calculate z-scores
z_scores = np.abs(stats.zscore(data))

# Flag outliers (z-score > 3)
outliers = data[z_scores > 3]

# Visualize data and outliers
plt.figure(figsize=(10, 6))
plt.scatter(range(len(data)), data, c='blue', alpha=0.5)
plt.scatter(np.where(z_scores > 3)[0], outliers, c='red', s=100)
plt.title('Data with Flagged Outliers')
plt.xlabel('Index')
plt.ylabel('Value')
plt.show()

print(f"Number of outliers detected: {len(outliers)}")
```

Slide 2: Find Label Errors

Label errors occur when data points are incorrectly labeled, potentially affecting model performance. Identifying and correcting these errors is essential for maintaining data quality.

```python
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict

# Generate sample data with intentional label errors
np.random.seed(42)
X = np.random.rand(1000, 5)
y = (X[:, 0] + X[:, 1] > 1).astype(int)
y[np.random.choice(1000, 50, replace=False)] = 1 - y[np.random.choice(1000, 50, replace=False)]

# Train a Random Forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)

# Get cross-validated predictions
y_pred = cross_val_predict(clf, X, y, cv=5, method='predict_proba')

# Calculate confidence scores
confidence_scores = np.max(y_pred, axis=1)

# Identify potential label errors
threshold = 0.9
potential_errors = np.where((confidence_scores > threshold) & (np.argmax(y_pred, axis=1) != y))[0]

print(f"Number of potential label errors: {len(potential_errors)}")
print("Indices of potential label errors:", potential_errors)
```

Slide 3: Identify Near Duplicates

Near duplicates are data points that are very similar but not exactly identical. Identifying them helps in data cleaning and reducing redundancy.

```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Generate sample data
np.random.seed(42)
data = np.random.rand(1000, 10)

# Add near duplicates
data[50] = data[0] + np.random.normal(0, 0.01, 10)
data[100] = data[1] + np.random.normal(0, 0.01, 10)

# Calculate cosine similarity
similarity_matrix = cosine_similarity(data)

# Set diagonal to 0 to ignore self-similarity
np.fill_diagonal(similarity_matrix, 0)

# Find pairs with high similarity
threshold = 0.99
near_duplicates = np.where(similarity_matrix > threshold)

print("Near duplicate pairs:")
for i, j in zip(near_duplicates[0], near_duplicates[1]):
    if i < j:  # Avoid printing duplicate pairs
        print(f"Pair ({i}, {j}): Similarity = {similarity_matrix[i, j]:.4f}")
```

Slide 4: Perform Active Learning

Active learning is a machine learning approach where the algorithm selects the most informative samples for labeling, reducing the amount of labeled data needed for training.

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Generate sample data
X, y = make_classification(n_samples=1000, n_features=20, n_informative=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)

# Active learning loop
n_initial = 10
n_queries = 5
n_instances = 10

# Start with a small labeled dataset
labeled_indices = np.random.choice(len(X_train), n_initial, replace=False)
X_labeled = X_train[labeled_indices]
y_labeled = y_train[labeled_indices]

for i in range(n_queries):
    # Train the model on the labeled data
    clf.fit(X_labeled, y_labeled)
    
    # Get predictions and uncertainty scores for unlabeled data
    unlabeled_indices = np.setdiff1d(range(len(X_train)), labeled_indices)
    X_unlabeled = X_train[unlabeled_indices]
    y_pred_proba = clf.predict_proba(X_unlabeled)
    uncertainty = 1 - np.max(y_pred_proba, axis=1)
    
    # Select the most uncertain instances
    query_indices = unlabeled_indices[np.argsort(uncertainty)[-n_instances:]]
    
    # Add newly labeled instances to the labeled dataset
    labeled_indices = np.concatenate([labeled_indices, query_indices])
    X_labeled = X_train[labeled_indices]
    y_labeled = y_train[labeled_indices]
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, clf.predict(X_test))
    print(f"Query {i+1}: Accuracy = {accuracy:.4f}, Labeled samples = {len(labeled_indices)}")
```

Slide 5: Find Out-of-Distribution Samples

Out-of-distribution (OOD) samples are data points that differ significantly from the training distribution. Detecting them is crucial for ensuring model reliability and safety.

```python
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.datasets import make_blobs

# Generate in-distribution data
X_in, _ = make_blobs(n_samples=1000, n_features=2, centers=3, random_state=42)

# Generate out-of-distribution data
X_out, _ = make_blobs(n_samples=100, n_features=2, centers=1, center_box=(10, 15), random_state=42)

# Combine in-distribution and out-of-distribution data
X_combined = np.vstack((X_in, X_out))

# Train Isolation Forest
clf = IsolationForest(contamination=0.1, random_state=42)
clf.fit(X_in)

# Predict anomaly scores
scores = clf.decision_function(X_combined)

# Identify OOD samples
threshold = np.percentile(scores[:len(X_in)], 5)  # Use 5th percentile of in-distribution scores as threshold
ood_mask = scores < threshold

# Visualize results
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.scatter(X_combined[~ood_mask, 0], X_combined[~ood_mask, 1], c='blue', label='In-distribution')
plt.scatter(X_combined[ood_mask, 0], X_combined[ood_mask, 1], c='red', label='Out-of-distribution')
plt.title('In-distribution vs Out-of-distribution Samples')
plt.legend()
plt.show()

print(f"Number of OOD samples detected: {np.sum(ood_mask)}")
```

Slide 6: Real-Life Example - Detecting Anomalies in Sensor Data

In industrial settings, detecting anomalies in sensor data is crucial for predictive maintenance and ensuring equipment reliability.

```python
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

# Generate synthetic sensor data
np.random.seed(42)
timestamps = pd.date_range(start='2023-01-01', periods=1000, freq='H')
temperature = np.random.normal(25, 5, 1000)
pressure = np.random.normal(100, 10, 1000)

# Introduce anomalies
temperature[500:520] += 20  # Sudden temperature spike
pressure[700:720] -= 30     # Sudden pressure drop

# Create DataFrame
df = pd.DataFrame({'timestamp': timestamps, 'temperature': temperature, 'pressure': pressure})

# Train Isolation Forest
clf = IsolationForest(contamination=0.05, random_state=42)
clf.fit(df[['temperature', 'pressure']])

# Predict anomalies
df['anomaly'] = clf.predict(df[['temperature', 'pressure']])

# Visualize results
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(df['timestamp'], df['temperature'], label='Temperature')
plt.scatter(df[df['anomaly'] == -1]['timestamp'], df[df['anomaly'] == -1]['temperature'], color='red', label='Anomaly')
plt.legend()
plt.title('Temperature Anomalies')

plt.subplot(2, 1, 2)
plt.plot(df['timestamp'], df['pressure'], label='Pressure')
plt.scatter(df[df['anomaly'] == -1]['timestamp'], df[df['anomaly'] == -1]['pressure'], color='red', label='Anomaly')
plt.legend()
plt.title('Pressure Anomalies')

plt.tight_layout()
plt.show()

print(f"Number of anomalies detected: {sum(df['anomaly'] == -1)}")
```

Slide 7: Real-Life Example - Identifying Near-Duplicate Images

In content moderation systems, identifying near-duplicate images helps in reducing redundancy and improving processing efficiency.

```python
import numpy as np
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

# Function to load and preprocess image
def load_image(path):
    img = Image.open(path).resize((224, 224))
    return np.array(img).flatten()

# Load sample images (replace with actual image paths)
image_paths = ['image1.jpg', 'image2.jpg', 'image3.jpg', 'image4.jpg']
images = [load_image(path) for path in image_paths]

# Calculate cosine similarity
similarity_matrix = cosine_similarity(images)

# Find near-duplicate pairs
threshold = 0.95
near_duplicates = np.where(similarity_matrix > threshold)

# Visualize results
plt.figure(figsize=(10, 10))
for i, (img1, img2) in enumerate(zip(near_duplicates[0], near_duplicates[1])):
    if img1 < img2:
        plt.subplot(2, 2, i+1)
        plt.imshow(np.reshape(images[img1], (224, 224, 3)))
        plt.title(f"Image {img1}")
        plt.axis('off')
        
        plt.subplot(2, 2, i+2)
        plt.imshow(np.reshape(images[img2], (224, 224, 3)))
        plt.title(f"Image {img2}")
        plt.axis('off')
        
        print(f"Near-duplicate pair: Image {img1} and Image {img2}")
        print(f"Similarity: {similarity_matrix[img1, img2]:.4f}")

plt.tight_layout()
plt.show()
```

Slide 8: Handling Imbalanced Datasets

Imbalanced datasets can lead to biased models. Techniques like oversampling, undersampling, and SMOTE help address this issue.

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE

# Generate imbalanced dataset
X, y = make_classification(n_samples=1000, n_classes=2, weights=[0.9, 0.1],
                           n_informative=3, n_redundant=1, flip_y=0,
                           n_features=20, n_clusters_per_class=1,
                           n_repeated=0, random_state=42)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model on imbalanced data
clf_imbalanced = RandomForestClassifier(random_state=42)
clf_imbalanced.fit(X_train, y_train)

# Apply SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Train model on balanced data
clf_balanced = RandomForestClassifier(random_state=42)
clf_balanced.fit(X_train_resampled, y_train_resampled)

# Evaluate models
print("Imbalanced Dataset Results:")
print(classification_report(y_test, clf_imbalanced.predict(X_test)))

print("\nBalanced Dataset Results:")
print(classification_report(y_test, clf_balanced.predict(X_test)))
```

Slide 9: Feature Importance Analysis

Understanding which features contribute most to a model's predictions helps in feature selection and model interpretation.

```python
import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

# Load dataset
boston = load_boston()
X, y = boston.data, boston.target

# Train Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X, y)

# Get feature importance
feature_importance = rf_model.feature_importances_

# Sort features by importance
feature_importance_sorted = sorted(zip(feature_importance, boston.feature_names), reverse=True)

# Visualize feature importance
plt.figure(figsize=(10, 6))
plt.bar(range(len(feature_importance_sorted)), [imp for imp, _ in feature_importance_sorted])
plt.xticks(range(len(feature_importance_sorted)), [name for _, name in feature_importance_sorted], rotation=90)
plt.title('Feature Importance in Boston Housing Dataset')
plt.xlabel('Features')
plt.ylabel('Importance')
plt.tight_layout()
plt.show()

# Print top 5 important features
print("Top 5 Important Features:")
for imp, name in feature_importance_sorted[:5]:
    print(f"{name}: {imp:.4f}")
```

Slide 10: Cross-Validation Strategies

Cross-validation helps in assessing model performance and preventing overfitting. Different strategies are suitable for different scenarios.

```python
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold, TimeSeriesSplit
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Generate sample data
X, y = make_classification(n_samples=1000, n_features=20, n_informative=10, random_state=42)

# Initialize classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)

# K-Fold Cross-Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
kf_scores = cross_val_score(clf, X, y, cv=kf)

# Stratified K-Fold Cross-Validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
skf_scores = cross_val_score(clf, X, y, cv=skf)

# Time Series Split (for time-series data)
tscv = TimeSeriesSplit(n_splits=5)
ts_scores = cross_val_score(clf, X, y, cv=tscv)

print("K-Fold CV Scores:", kf_scores)
print("Stratified K-Fold CV Scores:", skf_scores)
print("Time Series Split Scores:", ts_scores)

print("\nMean Scores:")
print("K-Fold:", np.mean(kf_scores))
print("Stratified K-Fold:", np.mean(skf_scores))
print("Time Series Split:", np.mean(ts_scores))
```

Slide 11: Hyperparameter Tuning

Optimizing model hyperparameters can significantly improve performance. Grid search and random search are common techniques for hyperparameter tuning.

```python
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Generate sample data
X, y = make_classification(n_samples=1000, n_features=20, n_informative=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Initialize base classifier
rf = RandomForestClassifier(random_state=42)

# Grid Search
grid_search = GridSearchCV(rf, param_grid, cv=5, n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

print("Best parameters (Grid Search):", grid_search.best_params_)
print("Best score (Grid Search):", grid_search.best_score_)

# Random Search
random_search = RandomizedSearchCV(rf, param_grid, n_iter=20, cv=5, n_jobs=-1, verbose=1, random_state=42)
random_search.fit(X_train, y_train)

print("\nBest parameters (Random Search):", random_search.best_params_)
print("Best score (Random Search):", random_search.best_score_)
```

Slide 12: Model Interpretability with SHAP Values

SHAP (SHapley Additive exPlanations) values help explain individual predictions and overall feature importance in machine learning models.

```python
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import load_boston

# Load Boston Housing dataset
X, y = shap.datasets.boston()
feature_names = X.columns

# Train a Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Compute SHAP values
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# Plot summary
shap.summary_plot(shap_values, X, plot_type="bar", feature_names=feature_names)

# Explain a single prediction
sample_idx = 0
shap.force_plot(explainer.expected_value, shap_values[sample_idx], X.iloc[sample_idx])

print("SHAP values for the first sample:")
for feature, value in zip(feature_names, shap_values[sample_idx]):
    print(f"{feature}: {value:.4f}")
```

Slide 13: Ensemble Methods

Ensemble methods combine multiple models to improve prediction accuracy and robustness.

```python
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Generate sample data
X, y = make_classification(n_samples=1000, n_features=20, n_informative=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Individual classifiers
rf = RandomForestClassifier(n_estimators=100, random_state=42)
gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
lr = LogisticRegression(random_state=42)

# Voting Classifier (Ensemble)
voting_clf = VotingClassifier(
    estimators=[('rf', rf), ('gb', gb), ('lr', lr)],
    voting='soft'
)

# Train and evaluate individual classifiers
for clf in (rf, gb, lr):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(f"{clf.__class__.__name__} Accuracy: {accuracy_score(y_test, y_pred):.4f}")

# Train and evaluate ensemble
voting_clf.fit(X_train, y_train)
y_pred_ensemble = voting_clf.predict(X_test)
print(f"Ensemble Accuracy: {accuracy_score(y_test, y_pred_ensemble):.4f}")
```

Slide 14: Additional Resources

For further exploration of machine learning techniques and best practices, consider the following resources:

1. "Machine Learning Yearning" by Andrew Ng ArXiv: [https://arxiv.org/abs/1803.07282](https://arxiv.org/abs/1803.07282)
2. "A Survey of Deep Learning Techniques for Neural Machine Translation" ArXiv: [https://arxiv.org/abs/1703.03906](https://arxiv.org/abs/1703.03906)
3. "XGBoost: A Scalable Tree Boosting System" ArXiv: [https://arxiv.org/abs/1603.02754](https://arxiv.org/abs/1603.02754)
4. "Practical Recommendations for Gradient-Based Training of Deep Architectures" ArXiv: [https://arxiv.org/abs/1206.5533](https://arxiv.org/abs/1206.5533)
5. "Deep Learning: A Critical Appraisal" ArXiv: [https://arxiv.org/abs/1801.00631](https://arxiv.org/abs/1801.00631)

These resources provide in-depth insights into various aspects of machine learning, from practical tips to advanced techniques and critical perspectives on the field.

