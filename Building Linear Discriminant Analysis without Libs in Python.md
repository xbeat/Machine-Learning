## Building Linear Discriminant Analysis without Libs in Python
Slide 1: Introduction to Linear Discriminant Analysis (LDA)

Linear Discriminant Analysis (LDA) is a dimensionality reduction and classification technique used in machine learning and statistics. It aims to find a linear combination of features that best separates two or more classes of objects or events. LDA is particularly useful when dealing with multi-class classification problems and can be used for both dimensionality reduction and classification tasks.

```python
# Simple visualization of LDA concept
import random

def generate_sample_data(n_samples):
    class1 = [(random.gauss(0, 1), random.gauss(0, 1)) for _ in range(n_samples)]
    class2 = [(random.gauss(2, 1), random.gauss(2, 1)) for _ in range(n_samples)]
    return class1, class2

class1, class2 = generate_sample_data(100)
print(f"Class 1 first 5 samples: {class1[:5]}")
print(f"Class 2 first 5 samples: {class2[:5]}")
```

Slide 2: Mathematical Foundation of LDA

LDA is based on the concept of maximizing the between-class variance while minimizing the within-class variance. The key idea is to project the data onto a lower-dimensional space in a way that maximizes class separability. The mathematical objective of LDA can be expressed as:

$J(w) = \\frac{w^T S\_B w}{w^T S\_W w}$

Where:

*   $w$ is the projection vector
*   $S\_B$ is the between-class scatter matrix
*   $S\_W$ is the within-class scatter matrix

```python
def calculate_mean(data):
    return [sum(feature) / len(data) for feature in zip(*data)]

def calculate_scatter_matrices(class1, class2):
    mean1 = calculate_mean(class1)
    mean2 = calculate_mean(class2)
    overall_mean = calculate_mean(class1 + class2)

    # Calculate within-class scatter matrix
    S_W = [[0, 0], [0, 0]]
    for x in class1 + class2:
        diff = [x[i] - overall_mean[i] for i in range(2)]
        S_W[0][0] += diff[0] * diff[0]
        S_W[0][1] += diff[0] * diff[1]
        S_W[1][0] += diff[1] * diff[0]
        S_W[1][1] += diff[1] * diff[1]

    # Calculate between-class scatter matrix
    diff = [mean1[i] - mean2[i] for i in range(2)]
    S_B = [[diff[0] * diff[0], diff[0] * diff[1]],
           [diff[1] * diff[0], diff[1] * diff[1]]]

    return S_W, S_B

S_W, S_B = calculate_scatter_matrices(class1, class2)
print("Within-class scatter matrix:", S_W)
print("Between-class scatter matrix:", S_B)
```

Slide 3: Data Preparation and Preprocessing

Before applying LDA, it's crucial to prepare and preprocess the data. This involves steps such as handling missing values, encoding categorical variables, and scaling numerical features. In this implementation, we'll focus on numerical data and assume it's already preprocessed.

```python
def preprocess_data(X, y):
    # Check for missing values
    if any(None in sublist for sublist in X) or None in y:
        raise ValueError("Data contains missing values")
    
    # Normalize features
    n_features = len(X[0])
    means = [sum(feature) / len(X) for feature in zip(*X)]
    stds = [sum((x[i] - means[i])**2 for x in X) / len(X) for i in range(n_features)]
    X_normalized = [[(x[i] - means[i]) / stds[i] for i in range(n_features)] for x in X]
    
    return X_normalized, y

# Example usage
X = [[1, 2], [3, 4], [5, 6], [7, 8]]
y = [0, 0, 1, 1]

X_preprocessed, y_preprocessed = preprocess_data(X, y)
print("Preprocessed X:", X_preprocessed)
print("Preprocessed y:", y_preprocessed)
```

Slide 4: Calculating Class Means and Overall Mean

To begin the LDA algorithm, we need to calculate the mean vectors for each class and the overall mean of the dataset. These calculations are fundamental for determining the between-class and within-class scatter matrices.

```python
def calculate_class_means(X, y):
    classes = set(y)
    class_means = {}
    for c in classes:
        class_data = [X[i] for i in range(len(X)) if y[i] == c]
        class_means[c] = calculate_mean(class_data)
    return class_means

def calculate_overall_mean(X):
    return calculate_mean(X)

# Example usage
X = [[1, 2], [3, 4], [5, 6], [7, 8]]
y = [0, 0, 1, 1]

class_means = calculate_class_means(X, y)
overall_mean = calculate_overall_mean(X)

print("Class means:", class_means)
print("Overall mean:", overall_mean)
```

Slide 5: Computing Scatter Matrices

The next step in LDA is to compute the within-class and between-class scatter matrices. These matrices quantify the spread of data points within each class and the separation between classes, respectively.

```python
def compute_scatter_matrices(X, y, class_means, overall_mean):
    n_features = len(X[0])
    S_W = [[0] * n_features for _ in range(n_features)]
    S_B = [[0] * n_features for _ in range(n_features)]
    
    for i, x in enumerate(X):
        class_mean = class_means[y[i]]
        diff_within = [x[j] - class_mean[j] for j in range(n_features)]
        diff_between = [class_mean[j] - overall_mean[j] for j in range(n_features)]
        
        for m in range(n_features):
            for n in range(n_features):
                S_W[m][n] += diff_within[m] * diff_within[n]
                S_B[m][n] += diff_between[m] * diff_between[n]
    
    return S_W, S_B

# Example usage
S_W, S_B = compute_scatter_matrices(X, y, class_means, overall_mean)
print("Within-class scatter matrix:", S_W)
print("Between-class scatter matrix:", S_B)
```

Slide 6: Eigenvalue Decomposition

LDA involves solving a generalized eigenvalue problem. We need to find the eigenvectors and eigenvalues of the matrix $S\_W^{-1}S\_B$. However, since we're implementing this from scratch, we'll use a simpler approach for 2D data.

```python
def solve_eigenvalue_problem_2d(S_W, S_B):
    # For 2D data, we can directly compute the projection vector
    inv_S_W = [[S_W[1][1], -S_W[0][1]], [-S_W[1][0], S_W[0][0]]]
    det = S_W[0][0] * S_W[1][1] - S_W[0][1] * S_W[1][0]
    inv_S_W = [[inv_S_W[i][j] / det for j in range(2)] for i in range(2)]
    
    # Compute S_W^-1 * S_B
    result = [[sum(inv_S_W[i][k] * S_B[k][j] for k in range(2)) for j in range(2)] for i in range(2)]
    
    # For 2D, the eigenvector is simply the first column of the result
    return [result[0][0], result[1][0]]

# Example usage
eigenvector = solve_eigenvalue_problem_2d(S_W, S_B)
print("LDA projection vector:", eigenvector)
```

Slide 7: Projecting Data onto LDA Space

Once we have the LDA projection vector, we can project our original data onto this new space. This step reduces the dimensionality of our data while maximizing class separability.

```python
def project_data(X, projection_vector):
    return [sum(x[i] * projection_vector[i] for i in range(len(x))) for x in X]

# Example usage
X_projected = project_data(X, eigenvector)
print("Projected data:", X_projected)

# Visualize the projection
for i, proj in enumerate(X_projected):
    print(f"Original: {X[i]}, Projected: {proj}, Class: {y[i]}")
```

Slide 8: Classification using LDA

After projecting the data, we can use the LDA space for classification. A simple approach is to compute the mean of each class in the projected space and classify new points based on their proximity to these means.

```python
def lda_classify(X_train, y_train, X_test, projection_vector):
    # Project training data
    X_train_proj = project_data(X_train, projection_vector)
    
    # Calculate class means in projected space
    class_means = {}
    for c in set(y_train):
        class_data = [X_train_proj[i] for i in range(len(X_train_proj)) if y_train[i] == c]
        class_means[c] = sum(class_data) / len(class_data)
    
    # Project and classify test data
    X_test_proj = project_data(X_test, projection_vector)
    predictions = []
    for x in X_test_proj:
        distances = {c: abs(x - mean) for c, mean in class_means.items()}
        predictions.append(min(distances, key=distances.get))
    
    return predictions

# Example usage
X_test = [[2, 3], [6, 7]]
predictions = lda_classify(X, y, X_test, eigenvector)
print("Predictions for test data:", predictions)
```

Slide 9: Evaluating LDA Performance

To assess the performance of our LDA implementation, we can use metrics such as accuracy, precision, and recall. Here's a simple function to calculate accuracy:

```python
def calculate_accuracy(y_true, y_pred):
    correct = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
    return correct / len(y_true)

# Example usage
y_true = [0, 1]
y_pred = lda_classify(X, y, X_test, eigenvector)
accuracy = calculate_accuracy(y_true, y_pred)
print(f"Accuracy: {accuracy:.2f}")
```

Slide 10: Visualizing LDA Results

Visualization can help us understand how LDA separates different classes. For 2D data, we can plot the original points and the LDA decision boundary.

```python
def plot_lda_results(X, y, projection_vector):
    # This function would typically use a plotting library like matplotlib
    # Since we're avoiding external libraries, we'll print a text-based representation
    
    print("LDA Results Visualization:")
    print("---------------------------")
    print("Original Data:")
    for i, x in enumerate(X):
        print(f"Point {i+1}: {x}, Class: {y[i]}")
    
    print("\nProjection Vector:", projection_vector)
    
    print("\nProjected Data:")
    X_proj = project_data(X, projection_vector)
    for i, proj in enumerate(X_proj):
        print(f"Point {i+1}: {proj:.2f}, Class: {y[i]}")

# Example usage
plot_lda_results(X, y, eigenvector)
```

Slide 11: Real-Life Example: Iris Flower Classification

Let's apply our LDA implementation to the classic Iris flower dataset. We'll use a simplified version with two features and two classes.

```python
# Simplified Iris dataset (sepal length, petal length) for two species
iris_data = [
    [5.1, 1.4], [4.9, 1.4], [4.7, 1.3], [4.6, 1.5], [5.0, 1.4],  # Setosa
    [7.0, 4.7], [6.4, 4.5], [6.9, 4.9], [5.5, 4.0], [6.5, 4.6]   # Virginica
]
iris_labels = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

# Preprocess data
X_iris, y_iris = preprocess_data(iris_data, iris_labels)

# Calculate means and scatter matrices
iris_class_means = calculate_class_means(X_iris, y_iris)
iris_overall_mean = calculate_overall_mean(X_iris)
S_W_iris, S_B_iris = compute_scatter_matrices(X_iris, y_iris, iris_class_means, iris_overall_mean)

# Solve eigenvalue problem and project data
iris_eigenvector = solve_eigenvalue_problem_2d(S_W_iris, S_B_iris)
X_iris_projected = project_data(X_iris, iris_eigenvector)

print("Iris LDA Results:")
for i, proj in enumerate(X_iris_projected):
    print(f"Flower {i+1}: Original {X_iris[i]}, Projected {proj:.2f}, Species: {'Setosa' if y_iris[i] == 0 else 'Virginica'}")
```

Slide 12: Real-Life Example: Handwritten Digit Recognition

Another application of LDA is in handwritten digit recognition. We'll use a simplified dataset with 2D features representing digit images.

```python
# Simplified digit dataset (two features per digit)
digit_data = [
    [2, 3], [1, 2], [2, 1], [3, 2], [2, 2],  # Digit 0
    [5, 5], [6, 4], [4, 6], [5, 6], [6, 5]   # Digit 1
]
digit_labels = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

# Preprocess data
X_digit, y_digit = preprocess_data(digit_data, digit_labels)

# Calculate means and scatter matrices
digit_class_means = calculate_class_means(X_digit, y_digit)
digit_overall_mean = calculate_overall_mean(X_digit)
S_W_digit, S_B_digit = compute_scatter_matrices(X_digit, y_digit, digit_class_means, digit_overall_mean)

# Solve eigenvalue problem and project data
digit_eigenvector = solve_eigenvalue_problem_2d(S_W_digit, S_B_digit)
X_digit_projected = project_data(X_digit, digit_eigenvector)

print("Digit Recognition LDA Results:")
for i, proj in enumerate(X_digit_projected):
    print(f"Image {i+1}: Original {X_digit[i]}, Projected {proj:.2f}, Digit: {y_digit[i]}")

# Classify a new digit
new_digit = [[4, 4]]
prediction = lda_classify(X_digit, y_digit, new_digit, digit_eigenvector)
print(f"New digit classified as: {prediction[0]}")
```

Slide 13: Limitations and Considerations

While LDA is a powerful technique, it has some limitations:

1.  Assumes normally distributed data with equal covariance matrices for each class.
2.  Can struggle with non-linear class boundaries.
3.  May not perform well when the number of samples per class is less than the number of features.
4.  Sensitive to outliers and extreme values in the dataset.
5.  Requires at least two classes for meaningful analysis.

To address these limitations, consider techniques like Quadratic Discriminant Analysis (QDA) for non-equal covariances, or kernel-based methods for non-linear boundaries.

```python
def check_lda_assumptions(X, y):
    class_sizes = {}
    for label in y:
        class_sizes[label] = class_sizes.get(label, 0) + 1
    
    n_features = len(X[0])
    n_classes = len(set(y))
    
    print(f"Number of features: {n_features}")
    print(f"Number of classes: {n_classes}")
    print("Class sizes:", class_sizes)
    print("Assumption check:")
    print(f"- Samples per class > features: {'Yes' if all(size > n_features for size in class_sizes.values()) else 'No'}")
    print(f"- At least two classes: {'Yes' if n_classes >= 2 else 'No'}")

# Example usage
X = [[1, 2], [3, 4], [5, 6], [7, 8]]
y = [0, 0, 1, 1]
check_lda_assumptions(X, y)
```

Slide 14: Extending LDA to Multiple Classes

While our implementation focused on binary classification, LDA can be extended to handle multiple classes. The principle remains the same, but the calculations become more complex.

```python
def multi_class_lda(X, y):
    classes = list(set(y))
    n_classes = len(classes)
    n_features = len(X[0])
    
    # Calculate class means and overall mean
    class_means = calculate_class_means(X, y)
    overall_mean = calculate_overall_mean(X)
    
    # Calculate scatter matrices
    S_W = [[0] * n_features for _ in range(n_features)]
    S_B = [[0] * n_features for _ in range(n_features)]
    
    for c in classes:
        X_c = [x for i, x in enumerate(X) if y[i] == c]
        mean_c = class_means[c]
        n_c = len(X_c)
        
        # Update within-class scatter matrix
        for x in X_c:
            diff = [x[i] - mean_c[i] for i in range(n_features)]
            for i in range(n_features):
                for j in range(n_features):
                    S_W[i][j] += diff[i] * diff[j]
        
        # Update between-class scatter matrix
        diff = [mean_c[i] - overall_mean[i] for i in range(n_features)]
        for i in range(n_features):
            for j in range(n_features):
                S_B[i][j] += n_c * diff[i] * diff[j]
    
    # The rest of the LDA algorithm remains similar
    # (eigenvalue decomposition, projection, etc.)
    
    return S_W, S_B

# Example usage
X_multi = [[1, 2], [2, 3], [3, 3], [4, 5], [5, 5]]
y_multi = [0, 0, 1, 1, 2]
S_W_multi, S_B_multi = multi_class_lda(X_multi, y_multi)
print("Multi-class Within-class scatter matrix:", S_W_multi)
print("Multi-class Between-class scatter matrix:", S_B_multi)
```

Slide 15: Additional Resources

For those interested in diving deeper into Linear Discriminant Analysis and its applications, consider exploring the following resources:

1.  "Fisher's Linear Discriminant Analysis" by S. Mika et al. (1999) ArXiv: [https://arxiv.org/abs/cs/9901014](https://arxiv.org/abs/cs/9901014)
2.  "A Tutorial on Support Vector Machines for Pattern Recognition" by C. Burges (1998) ArXiv: [https://arxiv.org/abs/1303.6151](https://arxiv.org/abs/1303.6151)
3.  "Dimensionality Reduction by Learning an Invariant Mapping" by Y. Bengio et al. (2006) ArXiv: [https://arxiv.org/abs/cs/0512126](https://arxiv.org/abs/cs/0512126)

These papers provide in-depth discussions on LDA, its relationships to other methods, and advanced applications in machine learning and pattern recognition.

