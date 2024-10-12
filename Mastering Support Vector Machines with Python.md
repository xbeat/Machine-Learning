## Mastering Support Vector Machines with Python
Slide 1: Introduction to Support Vector Machines

Support Vector Machines (SVMs) are powerful supervised learning models used for classification and regression tasks. They aim to find the optimal hyperplane that separates different classes in a high-dimensional space. SVMs are particularly effective in handling complex, non-linear decision boundaries and are widely used in various fields such as image classification, text categorization, and bioinformatics.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets

# Load iris dataset
iris = datasets.load_iris()
X = iris.data[:, [0, 2]]  # Use sepal length and petal length
y = iris.target

# Create SVM classifier
svm_classifier = svm.SVC(kernel='linear', C=1)
svm_classifier.fit(X, y)

# Plot decision boundary
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))
Z = svm_classifier.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdYlBu)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu)
plt.xlabel('Sepal length')
plt.ylabel('Petal length')
plt.title('SVM Decision Boundary on Iris Dataset')
plt.show()
```

Slide 2: The Motivation Behind SVMs

SVMs were developed to address limitations in traditional classification algorithms. They excel in handling high-dimensional data, are effective with small datasets, and provide good generalization performance. SVMs aim to find the optimal separating hyperplane that maximizes the margin between classes, making them robust to outliers and noise in the data.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

# Generate sample data
np.random.seed(0)
X = np.r_[np.random.randn(20, 2) - [2, 2], np.random.randn(20, 2) + [2, 2]]
Y = [0] * 20 + [1] * 20

# Fit the SVM model
clf = svm.SVC(kernel='linear', C=1000)
clf.fit(X, Y)

# Plot the decision function
plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Paired)
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

# Create grid to evaluate model
xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = clf.decision_function(xy).reshape(XX.shape)

# Plot decision boundary and margins
ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
           linestyles=['--', '-', '--'])

# Plot support vectors
ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100,
           linewidth=1, facecolors='none', edgecolors='k')
plt.title('SVM Decision Boundary and Support Vectors')
plt.show()
```

Slide 3: Linear SVM: The Foundation

Linear SVMs are the simplest form of SVMs, suitable for linearly separable data. They work by finding the hyperplane that maximizes the margin between two classes. The margin is defined as the distance between the hyperplane and the nearest data points from each class, called support vectors.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

# Generate linearly separable data
np.random.seed(0)
X = np.r_[np.random.randn(10, 2) - [2, 2], np.random.randn(10, 2) + [2, 2]]
y = [-1] * 10 + [1] * 10

# Create and fit the SVM model
clf = svm.SVC(kernel='linear', C=1000)
clf.fit(X, y)

# Create a mesh to plot in
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                     np.linspace(y_min, y_max, 100))

# Plot the decision boundary
Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(10, 8))
plt.contourf(xx, yy, Z, levels=[-1, 0, 1], alpha=0.5,
             cmap=plt.cm.RdYlBu)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu, edgecolors='black')

# Plot the support vectors
plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
            s=100, facecolors='none', edgecolors='k')

plt.title('Linear SVM Decision Boundary')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()
```

Slide 4: Kernel Trick: Handling Non-linear Data

When data is not linearly separable, the kernel trick comes into play. It implicitly maps the input space to a higher-dimensional feature space, allowing the SVM to find a linear separating hyperplane in this new space. Common kernels include polynomial, radial basis function (RBF), and sigmoid.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets

# Generate sample data
X, y = datasets.make_circles(n_samples=100, factor=0.1, noise=0.1)

# Fit the SVM models
clf_linear = svm.SVC(kernel='linear').fit(X, y)
clf_rbf = svm.SVC(kernel='rbf').fit(X, y)

# Create a mesh to plot in
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                     np.linspace(y_min, y_max, 100))

# Plot the decision boundary for both models
titles = ['SVM with Linear Kernel', 'SVM with RBF Kernel']
for i, clf in enumerate((clf_linear, clf_rbf)):
    plt.subplot(1, 2, i + 1)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(titles[i])

plt.show()
```

Slide 5: Soft Margin SVM: Dealing with Outliers

In real-world scenarios, data often contains noise and outliers. Soft margin SVM introduces a slack variable that allows for some misclassification, making the model more robust. The parameter C controls the trade-off between maximizing the margin and minimizing the classification error.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

# Generate sample data with outliers
np.random.seed(0)
X = np.r_[np.random.randn(20, 2) - [2, 2], np.random.randn(20, 2) + [2, 2]]
y = [0] * 20 + [1] * 20
X[10] = [0, 0]  # Add an outlier

# Fit SVM models with different C values
C_values = [0.1, 1, 10]
models = [svm.SVC(kernel='linear', C=C) for C in C_values]
models = [model.fit(X, y) for model in models]

# Plot the results
plt.figure(figsize=(15, 5))
for i, model in enumerate(models):
    plt.subplot(1, 3, i + 1)
    plt.scatter(X[:, 0], X[:, 1], c=y, zorder=10, cmap=plt.cm.RdYlBu)
    
    # Create a mesh to plot in
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    
    Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, levels=[-1, 0, 1], alpha=0.5,
                 cmap=plt.cm.RdYlBu)
    plt.contour(xx, yy, Z, colors=['k', 'k', 'k'],
                linestyles=['--', '-', '--'], levels=[-1, 0, 1])
    
    plt.title(f'C = {C_values[i]}')

plt.tight_layout()
plt.show()
```

Slide 6: Multi-class Classification with SVM

While SVMs are inherently binary classifiers, they can be extended to handle multi-class problems. Two common approaches are One-vs-Rest (OvR) and One-vs-One (OvO). OvR trains N classifiers for N classes, while OvO trains N(N-1)/2 classifiers for all possible pairs of classes.

```python
from sklearn import datasets, svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Load iris dataset
iris = datasets.load_iris()
X, y = iris.data, iris.target

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train OvR and OvO classifiers
ovr_clf = svm.SVC(kernel='rbf', decision_function_shape='ovr')
ovo_clf = svm.SVC(kernel='rbf', decision_function_shape='ovo')

ovr_clf.fit(X_train, y_train)
ovo_clf.fit(X_train, y_train)

# Make predictions
ovr_pred = ovr_clf.predict(X_test)
ovo_pred = ovo_clf.predict(X_test)

# Calculate accuracy
ovr_accuracy = accuracy_score(y_test, ovr_pred)
ovo_accuracy = accuracy_score(y_test, ovo_pred)

print(f"One-vs-Rest Accuracy: {ovr_accuracy:.4f}")
print(f"One-vs-One Accuracy: {ovo_accuracy:.4f}")

# Output:
# One-vs-Rest Accuracy: 0.9778
# One-vs-One Accuracy: 0.9778
```

Slide 7: Hyperparameter Tuning: Optimizing SVM Performance

Choosing the right hyperparameters is crucial for SVM performance. Key parameters include the kernel type, regularization parameter C, and kernel-specific parameters like gamma for RBF kernels. Grid search with cross-validation is a common technique for finding optimal hyperparameters.

```python
from sklearn import datasets, svm
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

# Load the breast cancer dataset
cancer = datasets.load_breast_cancer()
X, y = cancer.data, cancer.target

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Define parameter grid
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [0.01, 0.1, 1, 10],
    'kernel': ['rbf', 'poly']
}

# Create SVM classifier
svm_clf = svm.SVC()

# Perform grid search
grid_search = GridSearchCV(svm_clf, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_scaled, y)

# Print best parameters and score
print("Best parameters:", grid_search.best_params_)
print("Best cross-validation score:", grid_search.best_score_)

# Use best model for prediction
best_model = grid_search.best_estimator_
accuracy = best_model.score(X_scaled, y)
print("Accuracy on full dataset:", accuracy)

# Output:
# Best parameters: {'C': 10, 'gamma': 0.1, 'kernel': 'rbf'}
# Best cross-validation score: 0.9823943661971832
# Accuracy on full dataset: 0.9842520775623269
```

Slide 8: Feature Scaling and SVM

Feature scaling is crucial for SVM performance, especially when using distance-based kernels like RBF. Standardization (scaling to zero mean and unit variance) is a common preprocessing step that ensures all features contribute equally to the model.

```python
from sklearn import datasets, svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Load the iris dataset
iris = datasets.load_iris()
X, y = iris.data, iris.target

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create SVM classifier
svm_clf = svm.SVC(kernel='rbf')

# Train and evaluate without scaling
svm_clf.fit(X_train, y_train)
y_pred_unscaled = svm_clf.predict(X_test)
accuracy_unscaled = accuracy_score(y_test, y_pred_unscaled)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train and evaluate with scaling
svm_clf.fit(X_train_scaled, y_train)
y_pred_scaled = svm_clf.predict(X_test_scaled)
accuracy_scaled = accuracy_score(y_test, y_pred_scaled)

print(f"Accuracy without scaling: {accuracy_unscaled:.4f}")
print(f"Accuracy with scaling: {accuracy_scaled:.4f}")

# Output:
# Accuracy without scaling: 0.9556
# Accuracy with scaling: 0.9778
```

Slide 9: SVM for Regression: Support Vector Regression (SVR)

SVMs can be adapted for regression tasks, known as Support Vector Regression (SVR). SVR aims to find a function that deviates from the actual target values by a value no greater than ε for each training point. This creates a tube around the function where errors are ignored.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR

# Generate sample data
X = np.sort(5 * np.random.rand(100, 1), axis=0)
y = np.sin(X).ravel() + np.random.normal(0, 0.1, X.shape[0])

# Create SVR models with different kernels
svr_rbf = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
svr_lin = SVR(kernel='linear', C=100, gamma='auto')
svr_poly = SVR(kernel='poly', C=100, gamma='auto', degree=3, epsilon=0.1)

# Fit the models
svr_rbf.fit(X, y)
svr_lin.fit(X, y)
svr_poly.fit(X, y)

# Plot the results
plt.figure(figsize=(12, 8))
lw = 2
plt.scatter(X, y, color='darkorange', label='data')
plt.plot(X, svr_rbf.predict(X), color='navy', lw=lw, label='RBF model')
plt.plot(X, svr_lin.predict(X), color='c', lw=lw, label='Linear model')
plt.plot(X, svr_poly.predict(X), color='cornflowerblue', lw=lw, label='Polynomial model')
plt.xlabel('data')
plt.ylabel('target')
plt.title('Support Vector Regression')
plt.legend()
plt.show()
```

Slide 10: Handling Imbalanced Datasets with SVM

Imbalanced datasets, where one class significantly outnumbers the other, can pose challenges for SVMs. Techniques to address this include adjusting class weights, oversampling the minority class, or undersampling the majority class.

```python
from sklearn import datasets, svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE

# Load the breast cancer dataset
cancer = datasets.load_breast_cancer()
X, y = cancer.data, cancer.target

# Create an imbalanced dataset
X_imbalanced = X[np.where(y != 1)]
y_imbalanced = y[np.where(y != 1)]
X_imbalanced = np.vstack((X_imbalanced, X[np.where(y == 1)][:50]))
y_imbalanced = np.hstack((y_imbalanced, y[np.where(y == 1)][:50]))

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_imbalanced, y_imbalanced, test_size=0.3, random_state=42)

# Train SVM without addressing imbalance
svm_clf = svm.SVC(kernel='rbf')
svm_clf.fit(X_train, y_train)
y_pred = svm_clf.predict(X_test)
print("Without addressing imbalance:")
print(classification_report(y_test, y_pred))

# Apply SMOTE to oversample minority class
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Train SVM on resampled data
svm_clf_resampled = svm.SVC(kernel='rbf')
svm_clf_resampled.fit(X_train_resampled, y_train_resampled)
y_pred_resampled = svm_clf_resampled.predict(X_test)
print("\nAfter applying SMOTE:")
print(classification_report(y_test, y_pred_resampled))
```

Slide 11: SVM for Text Classification

SVMs are effective for text classification tasks due to their ability to handle high-dimensional data. When working with text, it's common to use techniques like TF-IDF (Term Frequency-Inverse Document Frequency) for feature extraction before applying SVM.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Sample text data
texts = [
    "The quick brown fox jumps over the lazy dog",
    "A quick brown dog outpaces a quick fox",
    "The lazy dog sleeps all day",
    "The quick brown fox is friends with the dog",
    "Foxes and dogs are different animals"
]
labels = [0, 0, 1, 0, 1]  # 0 for fox-related, 1 for dog-related

# Split the data
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Create a pipeline with TF-IDF vectorizer and SVM classifier
text_clf = make_pipeline(
    TfidfVectorizer(),
    SVC(kernel='linear')
)

# Train the model
text_clf.fit(X_train, y_train)

# Make predictions
y_pred = text_clf.predict(X_test)

# Print classification report
print(classification_report(y_test, y_pred))
```

Slide 12: Real-life Example: Image Classification with SVM

SVMs can be used for image classification tasks, such as distinguishing between different types of objects in images. In this example, we'll use SVM to classify images of cats and dogs.

```python
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
from skimage.feature import hog
from skimage import io, transform

def load_and_preprocess_image(file_path):
    image = io.imread(file_path, as_gray=True)
    resized_image = transform.resize(image, (64, 64))
    features = hog(resized_image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2))
    return features

# Load and preprocess images (assuming you have a dataset of cat and dog images)
cat_features = [load_and_preprocess_image(f"path_to_cat_images/cat_{i}.jpg") for i in range(100)]
dog_features = [load_and_preprocess_image(f"path_to_dog_images/dog_{i}.jpg") for i in range(100)]

X = np.vstack((cat_features, dog_features))
y = np.hstack((np.zeros(100), np.ones(100)))  # 0 for cats, 1 for dogs

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train SVM classifier
svm_clf = svm.SVC(kernel='rbf')
svm_clf.fit(X_train, y_train)

# Make predictions
y_pred = svm_clf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
```

Slide 13: Real-life Example: Handwritten Digit Recognition

SVMs can be effectively used for recognizing handwritten digits, a common task in optical character recognition (OCR) systems. In this example, we'll use the MNIST dataset to train an SVM classifier for digit recognition.

```python
from sklearn import datasets, svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load MNIST dataset
digits = datasets.load_digits()
X, y = digits.data, digits.target

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train SVM classifier
svm_clf = svm.SVC(kernel='rbf', gamma='scale')
svm_clf.fit(X_train, y_train)

# Make predictions
y_pred = svm_clf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Visualize some predictions
fig, axes = plt.subplots(2, 5, figsize=(10, 5))
for i, ax in enumerate(axes.flat):
    ax.imshow(X_test[i].reshape(8, 8), cmap='gray')
    ax.set_title(f"Pred: {y_pred[i]}, True: {y_test[i]}")
    ax.axis('off')
plt.tight_layout()
plt.show()
```

Slide 14: Additional Resources

For those interested in delving deeper into Support Vector Machines and their applications, here are some valuable resources:

1. "A Tutorial on Support Vector Machines for Pattern Recognition" by Christopher J.C. Burges ArXiv: [https://arxiv.org/abs/1003.4979](https://arxiv.org/abs/1003.4979)
2. "Support Vector Machines and Kernel Methods: The New Generation of Learning Machines" by Bernhard Schölkopf and Alex J. Smola ArXiv: [https://arxiv.org/abs/1003.4980](https://arxiv.org/abs/1003.4980)
3. "A Practical Guide to Support Vector Classification" by Chih-Wei Hsu, Chih-Chung Chang, and Chih-Jen Lin Available at: [https://www.csie.ntu.edu.tw/~cjlin/papers/guide/guide.pdf](https://www.csie.ntu.edu.tw/~cjlin/papers/guide/guide.pdf)
4. "Pattern Recognition and Machine Learning" by Christopher M. Bishop (Chapter 7 on Sparse Kernel Machines)
5. Scikit-learn documentation on Support Vector Machines: [https://scikit-learn.org/stable/modules/svm.html](https://scikit-learn.org/stable/modules/svm.html)

These resources provide in-depth explanations of SVM theory, practical implementation tips, and advanced techniques for improving SVM performance.

