## Handling Imbalanced Datasets in Tabular Classification

Slide 1: Imbalanced Datasets in Classification

Imbalanced datasets are a common challenge in tabular classification tasks. They occur when one class significantly outnumbers the other classes, leading to biased models that perform poorly on minority classes. This imbalance is often inherent in real-world data, such as fraud detection or rare disease diagnosis. Understanding and addressing this issue is crucial for developing effective classification models.

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate an imbalanced dataset
np.random.seed(42)
majority_class = np.random.normal(0, 1, (1000, 2))
minority_class = np.random.normal(3, 1, (100, 2))

# Visualize the imbalanced dataset
plt.figure(figsize=(10, 6))
plt.scatter(majority_class[:, 0], majority_class[:, 1], label='Majority Class', alpha=0.5)
plt.scatter(minority_class[:, 0], minority_class[:, 1], label='Minority Class', alpha=0.5)
plt.legend()
plt.title('Imbalanced Dataset Example')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()
```

Slide 2: Oversampling Techniques

Oversampling is a popular approach to address class imbalance. It involves increasing the number of instances in the minority class to balance the dataset. Various oversampling techniques exist, including random oversampling and more sophisticated methods like SMOTE (Synthetic Minority Over-sampling Technique). These techniques aim to improve model performance on minority classes without losing information from the majority class.

```python
from sklearn.datasets import make_classification
from imblearn.over_sampling import RandomOverSampler, SMOTE

# Generate an imbalanced dataset
X, y = make_classification(n_samples=1000, n_classes=2, weights=[0.9, 0.1], random_state=42)

# Apply random oversampling
ros = RandomOverSampler(random_state=42)
X_ros, y_ros = ros.fit_resample(X, y)

# Apply SMOTE
smote = SMOTE(random_state=42)
X_smote, y_smote = smote.fit_resample(X, y)

print(f"Original dataset shape: {dict(zip(*np.unique(y, return_counts=True)))}")
print(f"Random oversampled shape: {dict(zip(*np.unique(y_ros, return_counts=True)))}")
print(f"SMOTE oversampled shape: {dict(zip(*np.unique(y_smote, return_counts=True)))}")
```

Slide 3: SMOTE (Synthetic Minority Over-sampling Technique)

SMOTE is an advanced oversampling method that creates synthetic examples in the feature space. It works by selecting minority class instances and interpolating new instances between them and their nearest neighbors. This approach aims to create more diverse and representative samples of the minority class, potentially improving the model's ability to generalize.

```python
import numpy as np
from sklearn.neighbors import NearestNeighbors

def smote(X, y, k=5, n_samples=100):
    minority_class = X[y == 1]
    nn = NearestNeighbors(n_neighbors=k+1).fit(minority_class)
    
    synthetic_samples = []
    for _ in range(n_samples):
        idx = np.random.randint(0, len(minority_class))
        sample = minority_class[idx]
        neighbors = nn.kneighbors([sample], return_distance=False)[0][1:]
        nn_idx = np.random.choice(neighbors)
        nn_sample = minority_class[nn_idx]
        
        alpha = np.random.random()
        new_sample = sample + alpha * (nn_sample - sample)
        synthetic_samples.append(new_sample)
    
    return np.vstack([X, synthetic_samples]), np.hstack([y, np.ones(n_samples)])

# Example usage
X = np.random.randn(100, 2)
y = np.hstack([np.zeros(90), np.ones(10)])
X_resampled, y_resampled = smote(X, y, n_samples=90)
print(f"Original shape: {X.shape}, Resampled shape: {X_resampled.shape}")
```

Slide 4: Benefits of SMOTE

SMOTE offers several advantages in handling imbalanced datasets. By creating synthetic examples, it increases the diversity of the minority class, which can lead to better decision boundaries and improved generalization. SMOTE can help prevent overfitting to the majority class and enhance the model's ability to recognize patterns in the minority class. This technique is particularly useful when the minority class is underrepresented and additional real-world data is difficult or expensive to obtain.

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE

# Generate imbalanced dataset
X, y = make_classification(n_samples=1000, n_classes=2, weights=[0.9, 0.1], random_state=42)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train without SMOTE
clf_no_smote = RandomForestClassifier(random_state=42)
clf_no_smote.fit(X_train, y_train)

# Apply SMOTE and train
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
clf_smote = RandomForestClassifier(random_state=42)
clf_smote.fit(X_train_smote, y_train_smote)

# Compare results
print("Without SMOTE:")
print(classification_report(y_test, clf_no_smote.predict(X_test)))
print("\nWith SMOTE:")
print(classification_report(y_test, clf_smote.predict(X_test)))
```

Slide 5: Potential Drawbacks of SMOTE

While SMOTE can be beneficial, it's not always the optimal solution. SMOTE may introduce noise or create unrealistic synthetic examples, especially in high-dimensional spaces or with complex data distributions. This can lead to overfitting or the creation of artificial patterns that don't exist in the real data. Additionally, SMOTE assumes that the feature space is continuous and that interpolation between examples is meaningful, which may not hold for all types of data.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from imblearn.over_sampling import SMOTE

# Generate imbalanced, non-linear dataset
X, y = make_moons(n_samples=1000, noise=0.1, random_state=42)
X_minority = X[y == 1]
X_majority = X[y == 0][:100]
X_imbalanced = np.vstack([X_majority, X_minority])
y_imbalanced = np.hstack([np.zeros(100), np.ones(len(X_minority))])

# Apply SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_imbalanced, y_imbalanced)

# Visualize original and SMOTE-resampled data
plt.figure(figsize=(12, 5))
plt.subplot(121)
plt.scatter(X_imbalanced[y_imbalanced == 0][:, 0], X_imbalanced[y_imbalanced == 0][:, 1], label='Majority')
plt.scatter(X_imbalanced[y_imbalanced == 1][:, 0], X_imbalanced[y_imbalanced == 1][:, 1], label='Minority')
plt.title('Original Imbalanced Dataset')
plt.legend()

plt.subplot(122)
plt.scatter(X_resampled[y_resampled == 0][:, 0], X_resampled[y_resampled == 0][:, 1], label='Majority')
plt.scatter(X_resampled[y_resampled == 1][:, 0], X_resampled[y_resampled == 1][:, 1], label='Minority (SMOTE)')
plt.title('SMOTE-Resampled Dataset')
plt.legend()

plt.tight_layout()
plt.show()
```

Slide 6: SMOTE and Noise Introduction

SMOTE can inadvertently introduce noise into the dataset. This occurs when synthetic samples are generated in regions that don't accurately represent the true distribution of the minority class. For instance, in datasets with overlapping classes or complex decision boundaries, SMOTE might create synthetic samples that fall into the majority class region, leading to increased confusion for the classifier.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE

# Generate an imbalanced dataset with overlapping classes
X, y = make_classification(n_samples=1000, n_classes=2, weights=[0.9, 0.1], 
                           n_clusters_per_class=1, class_sep=0.5, random_state=42)

# Apply SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Visualize original and SMOTE-resampled data
plt.figure(figsize=(12, 5))
plt.subplot(121)
plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], label='Majority', alpha=0.5)
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], label='Minority', alpha=0.5)
plt.title('Original Imbalanced Dataset')
plt.legend()

plt.subplot(122)
plt.scatter(X_resampled[y_resampled == 0][:, 0], X_resampled[y_resampled == 0][:, 1], label='Majority', alpha=0.5)
plt.scatter(X_resampled[y_resampled == 1][:, 0], X_resampled[y_resampled == 1][:, 1], label='Minority (SMOTE)', alpha=0.5)
plt.title('SMOTE-Resampled Dataset')
plt.legend()

plt.tight_layout()
plt.show()
```

Slide 7: Real-Life Example: Rare Disease Detection

Consider a rare disease detection scenario where only 1% of patients have the disease. SMOTE can be applied to balance the dataset, but it may introduce noise by creating synthetic patients with unrealistic combinations of symptoms. This could lead to false positives in the model's predictions, potentially causing unnecessary stress and further testing for healthy individuals.

```python
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE

# Generate synthetic patient data
np.random.seed(42)
n_samples = 10000
n_features = 10

X = np.random.randn(n_samples, n_features)
y = np.zeros(n_samples)
y[:100] = 1  # 1% of patients have the rare disease

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train without SMOTE
clf_no_smote = RandomForestClassifier(random_state=42)
clf_no_smote.fit(X_train, y_train)

# Apply SMOTE and train
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
clf_smote = RandomForestClassifier(random_state=42)
clf_smote.fit(X_train_smote, y_train_smote)

# Compare results
print("Without SMOTE:")
print(classification_report(y_test, clf_no_smote.predict(X_test)))
print("\nWith SMOTE:")
print(classification_report(y_test, clf_smote.predict(X_test)))
```

Slide 8: Real-Life Example: Image Classification

In image classification tasks, such as identifying rare objects in satellite imagery, SMOTE can be problematic. Generating synthetic images by interpolating between existing ones may create unrealistic or nonsensical images. This can lead to poor generalization and decreased model performance when applied to real-world data.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from imblearn.over_sampling import SMOTE

# Load digit dataset and select two classes
digits = load_digits()
X = digits.data[(digits.target == 0) | (digits.target == 1)]
y = digits.target[(digits.target == 0) | (digits.target == 1)]

# Make it imbalanced by reducing class 1
X_imbalanced = np.vstack([X[y == 0], X[y == 1][:10]])
y_imbalanced = np.hstack([y[y == 0], y[y == 1][:10]])

# Apply SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_imbalanced, y_imbalanced)

# Visualize original and synthetic images
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
for i, ax in enumerate(axes[0]):
    ax.imshow(X_imbalanced[y_imbalanced == 1][i].reshape(8, 8), cmap='gray')
    ax.set_title(f"Original {i+1}")
    ax.axis('off')

for i, ax in enumerate(axes[1]):
    synthetic_idx = np.where((y_resampled == 1) & (y_imbalanced != 1))[0][i]
    ax.imshow(X_resampled[synthetic_idx].reshape(8, 8), cmap='gray')
    ax.set_title(f"Synthetic {i+1}")
    ax.axis('off')

plt.tight_layout()
plt.show()
```

Slide 9: Alternatives to SMOTE

While SMOTE can be effective, other techniques may be more suitable depending on the specific dataset and problem. Undersampling methods, such as Random Undersampling or Tomek Links, reduce the majority class instead of increasing the minority class. Ensemble methods like BalancedRandomForestClassifier combine multiple models to handle imbalance. Additionally, adjusting class weights or using specialized loss functions can address imbalance without modifying the dataset.

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier

# Generate imbalanced dataset
X, y = make_classification(n_samples=10000, n_classes=2, weights=[0.99, 0.01], random_state=42)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest with class weights
rf_weighted = RandomForestClassifier(class_weight='balanced', random_state=42)
rf_weighted.fit(X_train, y_train)

# Evaluate
y_pred = rf_weighted.predict(X_test)
print("Random Forest with class weights:")
print(classification_report(y_test, y_pred))
```

Slide 10: Evaluating the Need for SMOTE

Before applying SMOTE, it's crucial to assess whether it's necessary and beneficial for your specific problem. Evaluate the dataset's characteristics, such as class distribution and feature relationships. Consider the problem domain and the consequences of false positives versus false negatives. Sometimes, the natural imbalance in the data reflects real-world distributions and shouldn't be altered.

```python
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

def evaluate_smote_necessity(X, y, cv=5):
    clf = RandomForestClassifier(random_state=42)
    
    # Evaluate without SMOTE
    scores_no_smote = cross_val_score(clf, X, y, cv=cv, scoring='f1')
    
    # Evaluate with SMOTE
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    scores_smote = cross_val_score(clf, X_resampled, y_resampled, cv=cv, scoring='f1')
    
    print(f"Mean F1-score without SMOTE: {np.mean(scores_no_smote):.3f}")
    print(f"Mean F1-score with SMOTE: {np.mean(scores_smote):.3f}")
    
    if np.mean(scores_smote) > np.mean(scores_no_smote):
        print("SMOTE appears to be beneficial for this dataset.")
    else:
        print("SMOTE does not seem to improve performance significantly.")

# Example usage
X, y = make_classification(n_samples=1000, n_classes=2, weights=[0.9, 0.1], random_state=42)
evaluate_smote_necessity(X, y)
```

Slide 11: SMOTE Hyperparameter Tuning

When using SMOTE, careful tuning of its hyperparameters is essential to maximize its effectiveness while minimizing potential drawbacks. Key parameters include the sampling strategy (determining the desired ratio of minority to majority samples) and the number of nearest neighbors used for interpolation. Grid search with cross-validation can help find optimal parameters for your specific dataset.

```python
from sklearn.model_selection import GridSearchCV
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier

# Create imbalanced dataset
X, y = make_classification(n_samples=1000, n_classes=2, weights=[0.9, 0.1], random_state=42)

# Define pipeline and parameters
pipeline = Pipeline([
    ('smote', SMOTE(random_state=42)),
    ('classifier', RandomForestClassifier(random_state=42))
])

param_grid = {
    'smote__sampling_strategy': [0.1, 0.2, 0.5, 0.75, 1.0],
    'smote__k_neighbors': [3, 5, 7],
    'classifier__n_estimators': [50, 100, 200]
}

# Perform grid search
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='f1', n_jobs=-1)
grid_search.fit(X, y)

print("Best parameters:", grid_search.best_params_)
print("Best F1-score:", grid_search.best_score_)
```

Slide 12: Combining SMOTE with Other Techniques

To address SMOTE's limitations, consider combining it with other techniques. For example, SMOTEENN (SMOTE with Edited Nearest Neighbors) or SMOTETomek (SMOTE with Tomek Links) apply SMOTE followed by undersampling to remove noisy samples. These hybrid approaches can help create more balanced datasets while reducing the risk of introducing noise or unrealistic synthetic samples.

```python
from imblearn.combine import SMOTETomek, SMOTEENN
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

# Create imbalanced dataset
X, y = make_classification(n_samples=1000, n_classes=2, weights=[0.9, 0.1], random_state=42)

# Initialize resampling methods
smote_tomek = SMOTETomek(random_state=42)
smote_enn = SMOTEENN(random_state=42)

# Resample the dataset
X_resampled_tomek, y_resampled_tomek = smote_tomek.fit_resample(X, y)
X_resampled_enn, y_resampled_enn = smote_enn.fit_resample(X, y)

# Evaluate using cross-validation
clf = RandomForestClassifier(random_state=42)

scores_original = cross_val_score(clf, X, y, cv=5, scoring='f1')
scores_tomek = cross_val_score(clf, X_resampled_tomek, y_resampled_tomek, cv=5, scoring='f1')
scores_enn = cross_val_score(clf, X_resampled_enn, y_resampled_enn, cv=5, scoring='f1')

print(f"Mean F1-score (Original): {scores_original.mean():.3f}")
print(f"Mean F1-score (SMOTETomek): {scores_tomek.mean():.3f}")
print(f"Mean F1-score (SMOTEENN): {scores_enn.mean():.3f}")
```

Slide 13: Monitoring and Validating SMOTE Results

After applying SMOTE, it's crucial to monitor and validate the results to ensure the synthetic samples are meaningful and beneficial. Techniques like t-SNE or UMAP can help visualize high-dimensional data before and after SMOTE. Additionally, comparing performance metrics on both the original and SMOTE-resampled datasets using cross-validation can provide insights into the effectiveness of the technique.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from imblearn.over_sampling import SMOTE

def visualize_smote_results(X, y, X_resampled, y_resampled):
    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X)
    X_resampled_tsne = tsne.fit_transform(X_resampled)
    
    # Plot original and resampled data
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    ax1.scatter(X_tsne[y == 0, 0], X_tsne[y == 0, 1], label='Majority', alpha=0.5)
    ax1.scatter(X_tsne[y == 1, 0], X_tsne[y == 1, 1], label='Minority', alpha=0.5)
    ax1.set_title('Original Data')
    ax1.legend()
    
    ax2.scatter(X_resampled_tsne[y_resampled == 0, 0], X_resampled_tsne[y_resampled == 0, 1], label='Majority', alpha=0.5)
    ax2.scatter(X_resampled_tsne[y_resampled == 1, 0], X_resampled_tsne[y_resampled == 1, 1], label='Minority (SMOTE)', alpha=0.5)
    ax2.set_title('SMOTE-Resampled Data')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

# Example usage
X, y = make_classification(n_samples=1000, n_classes=2, weights=[0.9, 0.1], n_clusters_per_class=1, n_features=20, random_state=42)
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

visualize_smote_results(X, y, X_resampled, y_resampled)
```

Slide 14: Additional Resources

For those interested in diving deeper into the topic of imbalanced datasets and SMOTE, here are some valuable resources:

1.  Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: Synthetic Minority Over-sampling Technique. Journal of Artificial Intelligence Research, 16, 321-357. ArXiv: [https://arxiv.org/abs/1106.1813](https://arxiv.org/abs/1106.1813)
2.  He, H., & Garcia, E. A. (2009). Learning from Imbalanced Data. IEEE Transactions on Knowledge and Data Engineering, 21(9), 1263-1284. DOI: 10.1109/TKDE.2008.239
3.  Lemaitre, G., Nogueira, F., & Aridas, C. K. (2017). Imbalanced-learn: A Python Toolbox to Tackle the Curse of Imbalanced Datasets in Machine Learning. Journal of Machine Learning Research, 18(17), 1-5. ArXiv: [https://arxiv.org/abs/1609.06570](https://arxiv.org/abs/1609.06570)

These papers provide in-depth discussions on imbalanced datasets, SMOTE, and various other techniques for handling class imbalance in machine learning.

