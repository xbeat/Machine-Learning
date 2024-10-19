## Understanding Data through Statistics

Slide 1: Understanding Data through Statistics

Data and statistics are inseparable in the realm of machine learning and artificial intelligence. Statistics provide the foundation for interpreting and analyzing data, enabling us to extract meaningful insights and make informed decisions. Let's explore this relationship with a simple example of calculating the mean and standard deviation of a dataset.

```python
import random

# Generate a random dataset
data = [random.randint(1, 100) for _ in range(50)]

# Calculate mean
mean = sum(data) / len(data)

# Calculate standard deviation
variance = sum((x - mean) ** 2 for x in data) / len(data)
std_dev = variance ** 0.5

print(f"Dataset: {data}")
print(f"Mean: {mean:.2f}")
print(f"Standard Deviation: {std_dev:.2f}")
```

Slide 2: Probability Theory in AI

Probability theory forms the backbone of many AI algorithms. It allows us to model uncertainty and make predictions based on incomplete information. Let's implement a simple Naive Bayes classifier to demonstrate how probability is used in machine learning.

```python
def naive_bayes_classifier(features, labels, new_sample):
    classes = set(labels)
    class_counts = {c: labels.count(c) for c in classes}
    
    # Calculate prior probabilities
    priors = {c: count / len(labels) for c, count in class_counts.items()}
    
    # Calculate likelihood
    likelihoods = {c: 1 for c in classes}
    for c in classes:
        for i, feature in enumerate(new_sample):
            feature_count = sum(1 for j, label in enumerate(labels) if label == c and features[j][i] == feature)
            likelihoods[c] *= (feature_count + 1) / (class_counts[c] + len(set(f[i] for f in features)))
    
    # Calculate posterior probabilities
    posteriors = {c: priors[c] * likelihoods[c] for c in classes}
    
    # Return the class with the highest posterior probability
    return max(posteriors, key=posteriors.get)

# Example usage
features = [
    ['Sunny', 'Hot', 'High', 'Weak'],
    ['Sunny', 'Hot', 'High', 'Strong'],
    ['Overcast', 'Hot', 'High', 'Weak'],
    ['Rain', 'Mild', 'High', 'Weak'],
    ['Rain', 'Cool', 'Normal', 'Weak'],
    ['Rain', 'Cool', 'Normal', 'Strong'],
    ['Overcast', 'Cool', 'Normal', 'Strong'],
    ['Sunny', 'Mild', 'High', 'Weak'],
    ['Sunny', 'Cool', 'Normal', 'Weak'],
    ['Rain', 'Mild', 'Normal', 'Weak'],
    ['Sunny', 'Mild', 'Normal', 'Strong'],
    ['Overcast', 'Mild', 'High', 'Strong'],
    ['Overcast', 'Hot', 'Normal', 'Weak'],
    ['Rain', 'Mild', 'High', 'Strong']
]
labels = ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']

new_sample = ['Sunny', 'Cool', 'High', 'Strong']
prediction = naive_bayes_classifier(features, labels, new_sample)
print(f"Prediction for {new_sample}: {prediction}")
```

Slide 3: Hypothesis Testing in Machine Learning

Hypothesis testing is crucial for validating model assumptions and ensuring reliable inferences. Let's implement a simple t-test to compare two groups of data.

```python
import math

def t_test(group1, group2):
    n1, n2 = len(group1), len(group2)
    mean1, mean2 = sum(group1) / n1, sum(group2) / n2
    var1 = sum((x - mean1) ** 2 for x in group1) / (n1 - 1)
    var2 = sum((x - mean2) ** 2 for x in group2) / (n2 - 1)
    
    # Pooled standard deviation
    pooled_std = math.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    
    # T-statistic
    t_stat = (mean1 - mean2) / (pooled_std * math.sqrt(2 / n1))
    
    # Degrees of freedom
    df = n1 + n2 - 2
    
    return t_stat, df

# Example usage
group1 = [75, 82, 78, 80, 79]
group2 = [70, 72, 76, 74, 73]

t_stat, df = t_test(group1, group2)
print(f"T-statistic: {t_stat:.4f}")
print(f"Degrees of freedom: {df}")
```

Slide 4: Regression Models from Scratch

Regression models are fundamental in machine learning for predicting continuous outcomes. Let's implement a simple linear regression model from scratch.

```python
def linear_regression(X, y):
    n = len(X)
    mean_x, mean_y = sum(X) / n, sum(y) / n
    
    # Calculate slope (b1) and intercept (b0)
    numerator = sum((X[i] - mean_x) * (y[i] - mean_y) for i in range(n))
    denominator = sum((X[i] - mean_x) ** 2 for i in range(n))
    b1 = numerator / denominator
    b0 = mean_y - b1 * mean_x
    
    return b0, b1

def predict(X, b0, b1):
    return [b0 + b1 * x for x in X]

# Example data
X = [1, 2, 3, 4, 5]
y = [2, 4, 5, 4, 5]

# Fit the model
b0, b1 = linear_regression(X, y)
print(f"Intercept (b0): {b0:.2f}")
print(f"Slope (b1): {b1:.2f}")

# Make predictions
predictions = predict(X, b0, b1)
print("Predictions:", [f"{p:.2f}" for p in predictions])
```

Slide 5: Optimization Techniques

Optimization is at the heart of many machine learning algorithms. Let's implement a simple gradient descent algorithm to find the minimum of a function.

```python
def gradient_descent(f, df, initial_x, learning_rate, num_iterations):
    x = initial_x
    history = [x]
    
    for _ in range(num_iterations):
        gradient = df(x)
        x = x - learning_rate * gradient
        history.append(x)
    
    return x, history

# Example: Find the minimum of f(x) = x^2 + 2x + 1
def f(x):
    return x**2 + 2*x + 1

def df(x):
    return 2*x + 2

initial_x = 5
learning_rate = 0.1
num_iterations = 50

minimum, history = gradient_descent(f, df, initial_x, learning_rate, num_iterations)

print(f"Minimum found at x = {minimum:.4f}")
print(f"f(x) at minimum = {f(minimum):.4f}")
```

Slide 6: Overfitting and Underfitting

Understanding overfitting and underfitting is crucial for building effective machine learning models. Let's implement a simple polynomial regression to demonstrate these concepts.

```python
import random

def generate_data(n, degree, noise=0.1):
    X = [random.uniform(-1, 1) for _ in range(n)]
    y = [sum(i**d for d in range(degree+1)) + random.gauss(0, noise) for i in X]
    return X, y

def polynomial_features(X, degree):
    return [[x**i for i in range(degree+1)] for x in X]

def fit_polynomial(X, y, degree):
    X_poly = polynomial_features(X, degree)
    n = len(X)
    X_t = list(zip(*X_poly))
    
    # Calculate coefficients using normal equation
    X_t_X = [[sum(a*b for a, b in zip(row1, row2)) for row2 in X_t] for row1 in X_t]
    X_t_y = [sum(x*y for x, y in zip(row, y)) for row in X_t]
    
    # Solve the system of linear equations
    coeffs = gaussian_elimination(X_t_X, X_t_y)
    
    return coeffs

def gaussian_elimination(A, b):
    n = len(A)
    for i in range(n):
        max_element = abs(A[i][i])
        max_row = i
        for k in range(i + 1, n):
            if abs(A[k][i]) > max_element:
                max_element = abs(A[k][i])
                max_row = k
        A[i], A[max_row] = A[max_row], A[i]
        b[i], b[max_row] = b[max_row], b[i]

        for k in range(i + 1, n):
            c = -A[k][i] / A[i][i]
            for j in range(i, n):
                if i == j:
                    A[k][j] = 0
                else:
                    A[k][j] += c * A[i][j]
            b[k] += c * b[i]

    x = [0] * n
    for i in range(n - 1, -1, -1):
        x[i] = b[i] / A[i][i]
        for k in range(i - 1, -1, -1):
            b[k] -= A[k][i] * x[i]
    return x

# Generate data
X, y = generate_data(50, 3, noise=0.5)

# Fit polynomials of different degrees
degrees = [1, 3, 10]
models = [fit_polynomial(X, y, degree) for degree in degrees]

# Print coefficients
for degree, coeffs in zip(degrees, models):
    print(f"Degree {degree} coefficients: {[f'{c:.4f}' for c in coeffs]}")
```

Slide 7: Feature Selection

Feature selection is crucial for improving model performance and reducing overfitting. Let's implement a simple correlation-based feature selection method.

```python
def correlation(x, y):
    n = len(x)
    mean_x, mean_y = sum(x) / n, sum(y) / n
    var_x = sum((xi - mean_x) ** 2 for xi in x) / n
    var_y = sum((yi - mean_y) ** 2 for yi in y) / n
    cov_xy = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y)) / n
    return cov_xy / (var_x * var_y) ** 0.5

def feature_selection(X, y, threshold=0.5):
    selected_features = []
    for i, feature in enumerate(zip(*X)):
        corr = abs(correlation(feature, y))
        if corr > threshold:
            selected_features.append(i)
    return selected_features

# Example usage
X = [
    [1, 2, 3, 4],
    [2, 3, 4, 5],
    [3, 4, 5, 6],
    [4, 5, 6, 7],
    [5, 6, 7, 8]
]
y = [10, 12, 14, 16, 18]

selected = feature_selection(X, y, threshold=0.9)
print(f"Selected features: {selected}")
```

Slide 8: Model Evaluation

Evaluating model performance is crucial for understanding how well our models generalize to unseen data. Let's implement some common evaluation metrics for classification problems.

```python
def confusion_matrix(y_true, y_pred):
    classes = sorted(set(y_true) | set(y_pred))
    matrix = {(actual, pred): 0 for actual in classes for pred in classes}
    for true, pred in zip(y_true, y_pred):
        matrix[(true, pred)] += 1
    return matrix

def accuracy(y_true, y_pred):
    return sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)

def precision_recall_f1(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    classes = sorted(set(y_true) | set(y_pred))
    
    metrics = {}
    for c in classes:
        tp = cm[(c, c)]
        fp = sum(cm[(actual, c)] for actual in classes if actual != c)
        fn = sum(cm[(c, pred)] for pred in classes if pred != c)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics[c] = {"precision": precision, "recall": recall, "f1": f1}
    
    return metrics

# Example usage
y_true = ['cat', 'dog', 'cat', 'cat', 'dog', 'bird']
y_pred = ['cat', 'dog', 'cat', 'dog', 'dog', 'cat']

print(f"Confusion Matrix: {confusion_matrix(y_true, y_pred)}")
print(f"Accuracy: {accuracy(y_true, y_pred):.2f}")
print("Precision, Recall, F1:")
for class_, metrics in precision_recall_f1(y_true, y_pred).items():
    print(f"  {class_}: {metrics}")
```

Slide 9: Uncertainty Quantification

Quantifying uncertainty in predictions is crucial for making informed decisions. Let's implement a simple bootstrap method to estimate confidence intervals for a mean.

```python
import random

def bootstrap_mean_ci(data, num_bootstrap_samples=1000, ci=0.95):
    means = []
    n = len(data)
    
    for _ in range(num_bootstrap_samples):
        sample = [random.choice(data) for _ in range(n)]
        means.append(sum(sample) / n)
    
    means.sort()
    lower_percentile = (1 - ci) / 2
    upper_percentile = 1 - lower_percentile
    
    lower_bound = means[int(lower_percentile * num_bootstrap_samples)]
    upper_bound = means[int(upper_percentile * num_bootstrap_samples)]
    
    return lower_bound, upper_bound

# Example usage
data = [random.gauss(100, 20) for _ in range(100)]
mean = sum(data) / len(data)
ci_lower, ci_upper = bootstrap_mean_ci(data)

print(f"Sample mean: {mean:.2f}")
print(f"95% Confidence Interval: ({ci_lower:.2f}, {ci_upper:.2f})")
```

Slide 10: Real-Life Example: Weather Prediction

Let's apply our statistical knowledge to a real-life scenario: predicting whether it will rain tomorrow based on today's weather conditions. We'll implement a simple logistic regression model for this purpose.

```python
import math

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def weather_prediction(temperature, humidity, pressure, wind_speed):
    # Coefficients (normally learned from data)
    w0, w1, w2, w3, w4 = -10, 0.2, 0.3, 0.1, 0.1
    
    # Logistic regression
    z = w0 + w1*temperature + w2*humidity + w3*pressure + w4*wind_speed
    probability = sigmoid(z)
    
    return probability

# Example usage
temp, humid, press, wind = 25, 70, 1015, 10
rain_probability = weather_prediction(temp, humid, press, wind)
print(f"Probability of rain tomorrow: {rain_probability:.2%}")
```

Slide 11: Real-Life Example: Image Classification

Let's explore another real-life application of statistics in machine learning: a simple image classification system. We'll implement a basic nearest neighbor classifier for digit recognition.

```python
def euclidean_distance(img1, img2):
    return sum((p1 - p2) ** 2 for p1, p2 in zip(img1, img2)) ** 0.5

def nearest_neighbor_classifier(train_images, train_labels, test_image):
    distances = [(euclidean_distance(test_image, img), label) 
                 for img, label in zip(train_images, train_labels)]
    return min(distances, key=lambda x: x[0])[1]

# Example usage (with simplified 5x5 digit images)
train_images = [
    [1,1,1,1,1, 1,0,0,0,1, 1,0,0,0,1, 1,0,0,0,1, 1,1,1,1,1],  # 0
    [0,0,1,0,0, 0,1,1,0,0, 0,0,1,0,0, 0,0,1,0,0, 0,1,1,1,0],  # 1
    [1,1,1,1,1, 0,0,0,0,1, 1,1,1,1,1, 1,0,0,0,0, 1,1,1,1,1]   # 2
]
train_labels = [0, 1, 2]

test_image = [1,1,1,1,1, 1,0,0,0,1, 1,0,0,0,1, 1,0,0,0,1, 1,1,1,1,0]

prediction = nearest_neighbor_classifier(train_images, train_labels, test_image)
print(f"Predicted digit: {prediction}")
```

Slide 12: Statistical Hypothesis Testing in Practice

Statistical hypothesis testing is crucial in scientific research and data analysis. Let's implement a simple chi-square test for independence to demonstrate its application.

```python
import math

def chi_square_test(observed, expected):
    chi_square = sum((o - e) ** 2 / e for o, e in zip(observed, expected))
    df = len(observed) - 1
    return chi_square, df

def p_value(chi_square, df):
    # This is a simplified approximation
    return 1 - (math.erf(math.sqrt(chi_square / 2)) - 
                math.erf(math.sqrt((chi_square - 1) / 2)))

# Example: Testing if a die is fair
rolls = [150, 142, 144, 158, 152, 154]  # Observed frequencies
expected = [150] * 6  # Expected frequencies for a fair die

chi_square, df = chi_square_test(rolls, expected)
p = p_value(chi_square, df)

print(f"Chi-square statistic: {chi_square:.4f}")
print(f"Degrees of freedom: {df}")
print(f"P-value: {p:.4f}")
```

Slide 13: Time Series Analysis

Time series analysis is essential in many fields, from finance to environmental science. Let's implement a simple moving average model to demonstrate basic time series concepts.

```python
def moving_average(data, window_size):
    return [sum(data[i:i+window_size]) / window_size 
            for i in range(len(data) - window_size + 1)]

def forecast(data, window_size, steps):
    ma = moving_average(data, window_size)
    forecast = ma[-1:] * steps
    return forecast

# Example: Monthly sales data
sales_data = [100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300, 320]
window_size = 3
forecast_steps = 3

ma = moving_average(sales_data, window_size)
future_forecast = forecast(sales_data, window_size, forecast_steps)

print("Moving Average:")
print(ma)
print("\nForecast for next 3 months:")
print(future_forecast)
```

Slide 14: Additional Resources

For those interested in diving deeper into statistics and its applications in machine learning, here are some valuable resources:

1.  "An Introduction to Statistical Learning" by James, Witten, Hastie, and Tibshirani (2013) ArXiv: [https://arxiv.org/abs/1301.3781](https://arxiv.org/abs/1301.3781)
2.  "The Elements of Statistical Learning" by Hastie, Tibshirani, and Friedman (2009) Available at: [https://web.stanford.edu/~hastie/ElemStatLearn/](https://web.stanford.edu/~hastie/ElemStatLearn/)
3.  "Pattern Recognition and Machine Learning" by Christopher Bishop (2006)
4.  "Bayesian Reasoning and Machine Learning" by David Barber (2012) ArXiv: [https://arxiv.org/abs/1011.1669](https://arxiv.org/abs/1011.1669)
5.  "Information Theory, Inference, and Learning Algorithms" by David MacKay (2003) Available at: [http://www.inference.org.uk/mackay/itila/](http://www.inference.org.uk/mackay/itila/)

These resources provide in-depth coverage of statistical concepts and their applications in machine learning, offering both theoretical foundations and practical insights.

