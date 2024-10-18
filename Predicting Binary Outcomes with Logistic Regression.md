## Predicting Binary Outcomes with Logistic Regression

Slide 1: Introduction to Logistic Regression

Logistic regression is a powerful statistical method used for predicting binary outcomes. It's particularly useful when we need to classify data into two distinct categories, such as determining whether a tumor is malignant or benign based on its size. Unlike linear regression, which can produce unbounded predictions, logistic regression uses the sigmoid function to constrain outputs between 0 and 1, making it ideal for binary classification tasks.

```python
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

z = np.linspace(-10, 10, 100)
y = sigmoid(z)

plt.figure(figsize=(10, 6))
plt.plot(z, y)
plt.title('Sigmoid Function')
plt.xlabel('z')
plt.ylabel('sigmoid(z)')
plt.grid(True)
plt.show()
```

Slide 2: The Logistic Function

The logistic function, also known as the sigmoid function, is the core of logistic regression. It maps any real-valued number to a value between 0 and 1, which can be interpreted as a probability. The function is defined as:

f(z) = 1 / (1 + e^(-z))

Where z is typically a linear combination of input features and their corresponding weights.

```python
    z = np.dot(w, x) + b
    return 1 / (1 + np.exp(-z))

# Example usage
x = np.array([2, 3])  # Input features
w = np.array([0.5, -0.5])  # Weights
b = 1  # Bias

probability = logistic_function(x, w, b)
print(f"Probability: {probability:.4f}")
```

Slide 3: Logistic Regression Model

In logistic regression, we model the probability of an input belonging to a particular class. The model takes the form:

P(y=1|x; w, b) = 1 / (1 + e^(-(w · x + b)))

Where:

* x is the input feature vector
* w is the weight vector
* b is the bias term
* y is the binary output (0 or 1)

```python
    z = np.dot(X, w) + b
    return 1 / (1 + np.exp(-z))

# Generate some example data
np.random.seed(42)
X = np.random.randn(100, 2)
w = np.array([1, -1])
b = 0

# Calculate probabilities
probabilities = logistic_regression_model(X, w, b)

plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1], c=probabilities, cmap='coolwarm')
plt.colorbar(label='Probability')
plt.title('Logistic Regression Decision Boundary')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()
```

Slide 4: Decision Boundary

The decision boundary in logistic regression is the line (or hyperplane in higher dimensions) that separates the two classes. It's defined as the set of points where the model predicts equal probabilities for both classes, typically at P(y=1|x) = 0.5. This occurs when w · x + b = 0.

```python
    x1 = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
    x2 = -(w[0] * x1 + b) / w[1]  # Solve for x2 when w · x + b = 0
    
    plt.figure(figsize=(10, 6))
    plt.scatter(X[:, 0], X[:, 1], c=logistic_regression_model(X, w, b), cmap='coolwarm')
    plt.plot(x1, x2, 'g--', label='Decision Boundary')
    plt.colorbar(label='Probability')
    plt.legend()
    plt.title('Logistic Regression Decision Boundary')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()

plot_decision_boundary(X, w, b)
```

Slide 5: Cost Function

The cost function for logistic regression measures how well the model's predictions match the actual labels. We use the log loss (also known as cross-entropy loss) function:

J(w, b) = -1/m \* Σ\[y^(i) \* log(h(x^(i))) + (1 - y^(i)) \* log(1 - h(x^(i)))\]

Where:

* m is the number of training examples
* y^(i) is the true label for the i-th example
* h(x^(i)) is the model's prediction for the i-th example

```python
    epsilon = 1e-15  # Small value to avoid log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# Generate example data
np.random.seed(42)
X = np.random.randn(1000, 2)
y_true = (X[:, 0] + X[:, 1] > 0).astype(int)
w = np.array([0.5, 0.5])
b = 0

y_pred = logistic_regression_model(X, w, b)
loss = log_loss(y_true, y_pred)
print(f"Log Loss: {loss:.4f}")
```

Slide 6: Gradient Descent

Gradient descent is an optimization algorithm used to find the optimal weights and bias that minimize the cost function. The update rules for the parameters are:

w := w - α \* ∂J/∂w b := b - α \* ∂J/∂b

Where α is the learning rate, and ∂J/∂w and ∂J/∂b are the partial derivatives of the cost function with respect to w and b.

```python
    m = len(y)
    for _ in range(num_iterations):
        y_pred = logistic_regression_model(X, w, b)
        dw = (1/m) * np.dot(X.T, (y_pred - y))
        db = (1/m) * np.sum(y_pred - y)
        w -= learning_rate * dw
        b -= learning_rate * db
    return w, b

# Train the model
w, b = np.random.randn(2), 0
w, b = gradient_descent(X, y_true, w, b, learning_rate=0.1, num_iterations=1000)

# Plot the results
plot_decision_boundary(X, w, b)
```

Slide 7: Model Evaluation

To evaluate the performance of a logistic regression model, we typically use metrics such as accuracy, precision, recall, and F1-score. These metrics help us understand how well the model is performing on both the training and test datasets.

```python

def evaluate_model(X, y_true, w, b):
    y_pred = (logistic_regression_model(X, w, b) >= 0.5).astype(int)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")

evaluate_model(X, y_true, w, b)
```

Slide 8: Real-Life Example: Email Spam Classification

Logistic regression can be used to classify emails as spam or not spam based on various features such as the presence of certain words, the sender's domain, or the time of day the email was sent.

```python
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Sample email data (you would typically have more data)
emails = [
    ("Get rich quick! Limited time offer!", 1),
    ("Meeting at 3 PM in the conference room", 0),
    ("Viagra for sale! Discount prices!", 1),
    ("Project deadline reminder: submit by Friday", 0),
    ("You've won a free iPhone! Click here", 1)
]

df = pd.DataFrame(emails, columns=['text', 'is_spam'])

# Prepare features and target
X = df['text']
y = df['is_spam']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize the text
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Train the model
model = LogisticRegression()
model.fit(X_train_vectorized, y_train)

# Make predictions
y_pred = model.predict(X_test_vectorized)

# Evaluate the model
print(classification_report(y_test, y_pred))
```

Slide 9: Real-Life Example: Medical Diagnosis

Logistic regression can be applied in medical diagnosis to predict the likelihood of a patient having a certain condition based on various symptoms and test results.

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Sample medical data (you would typically have more data)
data = {
    'age': [45, 62, 35, 58, 40, 53, 47, 55, 61, 39],
    'blood_pressure': [130, 140, 120, 145, 125, 135, 130, 140, 150, 120],
    'cholesterol': [220, 260, 180, 240, 200, 230, 210, 250, 270, 190],
    'has_heart_disease': [0, 1, 0, 1, 0, 1, 0, 1, 1, 0]
}

df = pd.DataFrame(data)

# Prepare features and target
X = df[['age', 'blood_pressure', 'cholesterol']]
y = df['has_heart_disease']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print("Confusion Matrix:")
print(conf_matrix)
```

Slide 10: Regularization in Logistic Regression

Regularization helps prevent overfitting by adding a penalty term to the cost function. Two common types of regularization are L1 (Lasso) and L2 (Ridge). L1 regularization can lead to sparse models by driving some coefficients to zero, while L2 regularization prevents any single feature from having too much influence.

```python
from sklearn.model_selection import cross_val_score
import numpy as np

# Generate some example data
np.random.seed(42)
X = np.random.randn(1000, 20)
y = (X[:, 0] + X[:, 1] + np.random.randn(1000) * 0.1 > 0).astype(int)

# Compare different regularization types
models = {
    'No regularization': LogisticRegression(penalty='none'),
    'L1 regularization': LogisticRegression(penalty='l1', solver='liblinear'),
    'L2 regularization': LogisticRegression(penalty='l2')
}

for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=5)
    print(f"{name}: Mean accuracy = {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")

# Plot feature importances for L1 regularization
l1_model = LogisticRegression(penalty='l1', solver='liblinear')
l1_model.fit(X, y)

plt.figure(figsize=(10, 6))
plt.bar(range(20), l1_model.coef_[0])
plt.title('Feature Importances (L1 Regularization)')
plt.xlabel('Feature Index')
plt.ylabel('Coefficient Value')
plt.show()
```

Slide 11: Multiclass Logistic Regression

Multiclass logistic regression extends binary classification to handle multiple classes. Two common approaches are One-vs-Rest (OvR) and Multinomial Logistic Regression. OvR trains a separate binary classifier for each class, while Multinomial fits a single model for all classes simultaneously.

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train One-vs-Rest model
ovr_model = LogisticRegression(multi_class='ovr')
ovr_model.fit(X_train, y_train)

# Train Multinomial model
multi_model = LogisticRegression(multi_class='multinomial', solver='lbfgs')
multi_model.fit(X_train, y_train)

# Make predictions
ovr_pred = ovr_model.predict(X_test)
multi_pred = multi_model.predict(X_test)

# Evaluate models
print("One-vs-Rest:")
print(classification_report(y_test, ovr_pred, target_names=iris.target_names))
print("\nMultinomial:")
print(classification_report(y_test, multi_pred, target_names=iris.target_names))
```

Slide 12: Handling Imbalanced Datasets

In many real-world scenarios, datasets can be imbalanced, with one class significantly outnumbering the others. This can lead to biased models that perform poorly on minority classes. Techniques to address this include oversampling, undersampling, and adjusting class weights.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report

# Generate imbalanced dataset
X, y = make_classification(n_samples=1000, n_classes=2, weights=[0.9, 0.1], random_state=42)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train standard logistic regression
standard_model = LogisticRegression()
standard_model.fit(X_train, y_train)

# Train weighted logistic regression
weighted_model = LogisticRegression(class_weight='balanced')
weighted_model.fit(X_train, y_train)

# Apply SMOTE oversampling
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
smote_model = LogisticRegression()
smote_model.fit(X_train_smote, y_train_smote)

# Evaluate models
print("Standard Model:")
print(classification_report(y_test, standard_model.predict(X_test)))
print("\nWeighted Model:")
print(classification_report(y_test, weighted_model.predict(X_test)))
print("\nSMOTE Model:")
print(classification_report(y_test, smote_model.predict(X_test)))
```

Slide 13: Feature Selection and Engineering

Feature selection and engineering are crucial steps in improving model performance. Techniques include correlation analysis, mutual information, and dimensionality reduction methods like PCA. Here's an example of using mutual information for feature selection:

```python
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np

# Load the breast cancer dataset
data = load_breast_cancer()
X, y = data.data, data.target

# Calculate mutual information scores
mi_scores = mutual_info_classif(X, y)

# Select top 10 features
top_10_features = np.argsort(mi_scores)[-10:]
X_selected = X[:, top_10_features]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

# Train and evaluate model
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(f"Accuracy with top 10 features: {accuracy_score(y_test, y_pred):.4f}")

# Feature importance plot
plt.figure(figsize=(10, 6))
plt.bar(range(10), model.coef_[0])
plt.title('Feature Importance (Top 10 Features)')
plt.xlabel('Feature Index')
plt.ylabel('Coefficient Value')
plt.show()
```

Slide 14: Interpreting Logistic Regression Results

Interpreting logistic regression results involves understanding coefficients, odds ratios, and confidence intervals. The coefficients represent the change in log-odds for a one-unit increase in the corresponding feature, while odds ratios provide a more intuitive interpretation.

```python
from sklearn.datasets import load_breast_cancer

# Load the breast cancer dataset
data = load_breast_cancer()
X, y = data.data, data.target

# Add constant term to the features
X = sm.add_constant(X)

# Fit the model
model = sm.Logit(y, X)
results = model.fit()

# Print summary
print(results.summary())

# Calculate odds ratios and confidence intervals
odds_ratios = np.exp(results.params)
conf_int = np.exp(results.conf_int())

# Create a DataFrame with results
summary_df = pd.DataFrame({
    'Odds Ratio': odds_ratios,
    'Lower CI': conf_int[0],
    'Upper CI': conf_int[1]
}, index=data.feature_names)

print("\nOdds Ratios and Confidence Intervals:")
print(summary_df)
```

Slide 15: Additional Resources

For those interested in diving deeper into logistic regression and its applications, here are some valuable resources:

1. "Logistic Regression: From Basic to Advanced" by Gareth James et al. (2013) - An in-depth exploration of logistic regression techniques.
2. "Pattern Recognition and Machine Learning" by Christopher Bishop (2006) - Covers logistic regression in the context of broader machine learning concepts.
3. ArXiv paper: "A Survey of Logistic Regression Techniques for Prediction of Student Academic Performance" by Alaa M. El-Halees (2019) URL: [https://arxiv.org/abs/1904.04904](https://arxiv.org/abs/1904.04904)
4. Scikit-learn documentation on Logistic Regression: [https://scikit-learn.org/stable/modules/linear\_model.html#logistic-regression](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression)
5. StatQuest YouTube channel - Logistic Regression playlist: [https://www.youtube.com/playlist?list=PLblh5JKOoLUKxzEP5HA2d-Li7IJkHfXSe](https://www.youtube.com/playlist?list=PLblh5JKOoLUKxzEP5HA2d-Li7IJkHfXSe)

These resources provide a mix of theoretical foundations and practical implementations to enhance your understanding of logistic regression.


