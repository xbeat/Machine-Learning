## Platt Scaling for Model Calibration in Python
Slide 1: Introduction to Platt Scaling

Platt scaling is a technique used to calibrate machine learning models, particularly for binary classification problems. It transforms the raw output scores of a classifier into well-calibrated probability estimates. This method is especially useful when the model's predictions are not inherently probabilistic or when the model's output scores are not well-calibrated.

```python
import numpy as np
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.datasets import make_classification

# Generate a sample dataset
X, y = make_classification(n_samples=1000, n_classes=2, random_state=42)

# Train an SVM classifier
svm = SVC(kernel='rbf', random_state=42)
svm.fit(X, y)

# Apply Platt scaling
platt_svm = CalibratedClassifierCV(svm, method='sigmoid', cv=5)
platt_svm.fit(X, y)

# Compare raw SVM scores with Platt-scaled probabilities
svm_score = svm.decision_function(X[:1])
platt_prob = platt_svm.predict_proba(X[:1])

print(f"Raw SVM score: {svm_score}")
print(f"Platt-scaled probability: {platt_prob}")
```

Slide 2: The Need for Calibration

Uncalibrated classifiers may produce biased or unreliable probability estimates. This can lead to poor decision-making in applications where accurate probabilities are crucial. Platt scaling addresses this issue by learning a mapping from raw scores to well-calibrated probabilities.

```python
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve

# Generate predictions for uncalibrated and calibrated models
y_pred_uncalibrated = svm.decision_function(X)
y_pred_calibrated = platt_svm.predict_proba(X)[:, 1]

# Plot calibration curves
plt.figure(figsize=(10, 6))
plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly calibrated')
prob_true, prob_pred = calibration_curve(y, y_pred_uncalibrated, n_bins=10)
plt.plot(prob_pred, prob_true, marker='o', label='Uncalibrated')
prob_true, prob_pred = calibration_curve(y, y_pred_calibrated, n_bins=10)
plt.plot(prob_pred, prob_true, marker='o', label='Platt-scaled')
plt.xlabel('Mean predicted probability')
plt.ylabel('Fraction of positives')
plt.title('Calibration curves')
plt.legend()
plt.show()
```

Slide 3: Platt Scaling Algorithm

Platt scaling works by fitting a logistic regression model to the classifier's scores. The algorithm learns parameters A and B to transform the original scores (f) into calibrated probabilities (p) using the formula: p = 1 / (1 + exp(A\*f + B)).

```python
import scipy.optimize as optimize

def platt_scale(scores, labels):
    def sigmoid(x, A, B):
        return 1 / (1 + np.exp(A*x + B))
    
    def nll(AB, scores, labels):
        A, B = AB
        p = sigmoid(scores, A, B)
        return -np.sum(labels * np.log(p) + (1 - labels) * np.log(1 - p))
    
    A, B = optimize.minimize(nll, [1, 0], args=(scores, labels)).x
    return lambda x: sigmoid(x, A, B)

# Apply custom Platt scaling
scores = svm.decision_function(X)
platt_func = platt_scale(scores, y)
calibrated_probs = platt_func(scores)

print(f"Custom Platt-scaled probability for first sample: {calibrated_probs[0]}")
```

Slide 4: Implementing Platt Scaling

To implement Platt scaling, we typically split our data into training and validation sets. The classifier is trained on the training set, and its scores on the validation set are used to learn the Platt scaling parameters.

```python
from sklearn.model_selection import train_test_split

# Split the data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the classifier
svm = SVC(kernel='rbf', random_state=42)
svm.fit(X_train, y_train)

# Get scores on validation set
val_scores = svm.decision_function(X_val)

# Learn Platt scaling parameters
platt_func = platt_scale(val_scores, y_val)

# Apply Platt scaling to new data
X_new = X[:5]  # Using first 5 samples as new data
new_scores = svm.decision_function(X_new)
calibrated_probs = platt_func(new_scores)

print("Calibrated probabilities for new data:")
print(calibrated_probs)
```

Slide 5: Cross-Validation in Platt Scaling

To make the most of our data, we can use cross-validation when learning the Platt scaling parameters. This approach helps prevent overfitting and provides more robust calibration.

```python
from sklearn.model_selection import cross_val_predict

# Perform cross-validated Platt scaling
cv_scores = cross_val_predict(svm, X, y, cv=5, method='decision_function')
platt_func = platt_scale(cv_scores, y)

# Apply to the entire dataset
all_scores = svm.decision_function(X)
calibrated_probs = platt_func(all_scores)

# Visualize the distribution of calibrated probabilities
plt.figure(figsize=(10, 6))
plt.hist(calibrated_probs, bins=50, edgecolor='black')
plt.title('Distribution of Calibrated Probabilities')
plt.xlabel('Probability')
plt.ylabel('Frequency')
plt.show()
```

Slide 6: Comparing Calibration Methods

Platt scaling is one of several calibration methods. Let's compare it with isotonic regression, another popular calibration technique.

```python
from sklearn.isotonic import IsotonicRegression

# Platt scaling
platt_svm = CalibratedClassifierCV(svm, method='sigmoid', cv=5)
platt_svm.fit(X, y)

# Isotonic regression
iso_svm = CalibratedClassifierCV(svm, method='isotonic', cv=5)
iso_svm.fit(X, y)

# Generate predictions
X_test = X[:100]  # Using first 100 samples as test data
y_test = y[:100]

platt_probs = platt_svm.predict_proba(X_test)[:, 1]
iso_probs = iso_svm.predict_proba(X_test)[:, 1]

# Plot calibration curves
plt.figure(figsize=(10, 6))
plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly calibrated')
for probs, name in [(platt_probs, 'Platt Scaling'), (iso_probs, 'Isotonic Regression')]:
    prob_true, prob_pred = calibration_curve(y_test, probs, n_bins=10)
    plt.plot(prob_pred, prob_true, marker='o', label=name)

plt.xlabel('Mean predicted probability')
plt.ylabel('Fraction of positives')
plt.title('Calibration Curve Comparison')
plt.legend()
plt.show()
```

Slide 7: Handling Multi-class Problems

While Platt scaling was originally designed for binary classification, it can be extended to multi-class problems using the one-vs-rest (OvR) or one-vs-one (OvO) approach.

```python
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize

# Generate multi-class data
X_multi, y_multi = make_classification(n_samples=1000, n_classes=3, n_informative=3, random_state=42)

# Binarize the labels
y_bin = label_binarize(y_multi, classes=[0, 1, 2])

# Train a multi-class SVM
svm_multi = OneVsRestClassifier(SVC(kernel='rbf', probability=True))
svm_multi.fit(X_multi, y_multi)

# Apply Platt scaling
platt_svm_multi = CalibratedClassifierCV(svm_multi, method='sigmoid', cv=5)
platt_svm_multi.fit(X_multi, y_multi)

# Compare probabilities
raw_probs = svm_multi.predict_proba(X_multi[:1])
platt_probs = platt_svm_multi.predict_proba(X_multi[:1])

print("Raw probabilities:", raw_probs)
print("Platt-scaled probabilities:", platt_probs)
```

Slide 8: Evaluating Calibration Performance

To assess the quality of calibration, we can use metrics such as Brier score, log loss, or reliability diagrams. These metrics help quantify how well the predicted probabilities match the true probabilities.

```python
from sklearn.metrics import brier_score_loss, log_loss

# Generate predictions
y_pred_uncalibrated = svm.decision_function(X)
y_pred_calibrated = platt_svm.predict_proba(X)[:, 1]

# Calculate Brier score
brier_uncalibrated = brier_score_loss(y, (y_pred_uncalibrated - y_pred_uncalibrated.min()) / (y_pred_uncalibrated.max() - y_pred_uncalibrated.min()))
brier_calibrated = brier_score_loss(y, y_pred_calibrated)

# Calculate log loss
ll_uncalibrated = log_loss(y, (y_pred_uncalibrated - y_pred_uncalibrated.min()) / (y_pred_uncalibrated.max() - y_pred_uncalibrated.min()))
ll_calibrated = log_loss(y, y_pred_calibrated)

print(f"Brier score (uncalibrated): {brier_uncalibrated:.4f}")
print(f"Brier score (calibrated): {brier_calibrated:.4f}")
print(f"Log loss (uncalibrated): {ll_uncalibrated:.4f}")
print(f"Log loss (calibrated): {ll_calibrated:.4f}")
```

Slide 9: Real-life Example: Medical Diagnosis

Consider a machine learning model used to predict the likelihood of a patient having a certain disease based on their symptoms. Accurate probability estimates are crucial for making informed decisions about further tests or treatments.

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Simulated medical data
data = {
    'age': np.random.normal(50, 15, 1000),
    'blood_pressure': np.random.normal(120, 20, 1000),
    'cholesterol': np.random.normal(200, 30, 1000),
    'has_disease': np.random.choice([0, 1], 1000, p=[0.9, 0.1])
}
df = pd.DataFrame(data)

# Prepare the data
X = df[['age', 'blood_pressure', 'cholesterol']]
y = df['has_disease']

# Split and scale the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train and calibrate the model
svm = SVC(kernel='rbf', random_state=42)
platt_svm = CalibratedClassifierCV(svm, method='sigmoid', cv=5)
platt_svm.fit(X_train_scaled, y_train)

# Predict probabilities for a new patient
new_patient = scaler.transform([[55, 130, 220]])
disease_prob = platt_svm.predict_proba(new_patient)[0, 1]

print(f"Probability of disease for the new patient: {disease_prob:.2f}")
```

Slide 10: Real-life Example: Spam Detection

Email spam detection is another area where well-calibrated probabilities are important. A miscalibrated model might incorrectly classify important emails as spam or let too many spam emails through.

```python
from sklearn.feature_extraction.text import CountVectorizer

# Simulated email data
emails = [
    "Get rich quick! Buy now!", "Meeting at 3pm", "Free offer inside",
    "Project deadline reminder", "You've won a prize!", "Lunch tomorrow?",
    "Important: account security", "Discount on all products", "Team update",
    "Your package has shipped"
]
labels = [1, 0, 1, 0, 1, 0, 0, 1, 0, 0]  # 1 for spam, 0 for not spam

# Prepare the data
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(emails)

# Train and calibrate the model
svm = SVC(kernel='linear', random_state=42)
platt_svm = CalibratedClassifierCV(svm, method='sigmoid', cv=3)
platt_svm.fit(X, labels)

# Predict spam probability for a new email
new_email = ["Congratulations! You've been selected for a free trial"]
new_email_vectorized = vectorizer.transform(new_email)
spam_prob = platt_svm.predict_proba(new_email_vectorized)[0, 1]

print(f"Probability of the new email being spam: {spam_prob:.2f}")
```

Slide 11: Limitations and Considerations

While Platt scaling is effective, it's not without limitations. It assumes that the sigmoid function is an appropriate transformation, which may not always be true. Additionally, Platt scaling can be sensitive to outliers and may struggle with extremely imbalanced datasets.

```python
import matplotlib.pyplot as plt
import numpy as np

def generate_scores(n_samples, mean, std):
    return np.random.normal(mean, std, n_samples)

# Generate scores for two classes
scores_class0 = generate_scores(1000, -2, 1)
scores_class1 = generate_scores(1000, 2, 1)

# Plot the score distributions
plt.figure(figsize=(10, 6))
plt.hist(scores_class0, bins=50, alpha=0.5, label='Class 0')
plt.hist(scores_class1, bins=50, alpha=0.5, label='Class 1')
plt.title('Distribution of Scores for Two Classes')
plt.xlabel('Score')
plt.ylabel('Frequency')
plt.legend()
plt.show()

# Demonstrate the effect of outliers
outlier_score = 10
scores_class1_with_outlier = np.append(scores_class1, outlier_score)

all_scores = np.concatenate([scores_class0, scores_class1_with_outlier])
all_labels = np.concatenate([np.zeros_like(scores_class0), np.ones_like(scores_class1_with_outlier)])

platt_func = platt_scale(all_scores, all_labels)
calibrated_probs = platt_func(all_scores)

plt.figure(figsize=(10, 6))
plt.scatter(all_scores, calibrated_probs, alpha=0.5)
plt.title('Platt Scaling with Outlier')
plt.xlabel('Original Score')
plt.ylabel('Calibrated Probability')
plt.show()
```

Slide 12: Alternative Calibration Techniques

While Platt scaling is popular, other calibration methods exist, such as isotonic regression and temperature scaling. Each method has its strengths and may be more suitable depending on the specific problem and data characteristics.

```python
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import calibration_curve

# Generate sample data
np.random.seed(42)
scores = np.random.rand(1000)
true_probs = 1 / (1 + np.exp(-10 * (scores - 0.5)))
labels = np.random.binomial(1, true_probs)

# Platt scaling
platt_func = platt_scale(scores, labels)
platt_probs = platt_func(scores)

# Isotonic regression
iso_reg = IsotonicRegression(out_of_bounds='clip')
iso_probs = iso_reg.fit_transform(scores, labels)

# Temperature scaling (simplified version)
def temperature_scale(scores, temperature):
    return 1 / (1 + np.exp(-scores / temperature))

# Find optimal temperature (simplified)
temp = np.linspace(0.1, 2, 20)
temp_probs = [temperature_scale(scores, t) for t in temp]
temp_losses = [log_loss(labels, p) for p in temp_probs]
best_temp = temp[np.argmin(temp_losses)]
temp_probs = temperature_scale(scores, best_temp)

# Plot calibration curves
plt.figure(figsize=(10, 6))
for probs, name in [(scores, 'Uncalibrated'), 
                    (platt_probs, 'Platt Scaling'),
                    (iso_probs, 'Isotonic Regression'),
                    (temp_probs, 'Temperature Scaling')]:
    prob_true, prob_pred = calibration_curve(labels, probs, n_bins=10)
    plt.plot(prob_pred, prob_true, marker='o', label=name)

plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly calibrated')
plt.xlabel('Mean predicted probability')
plt.ylabel('Fraction of positives')
plt.title('Calibration Curve Comparison')
plt.legend()
plt.show()
```

Slide 13: Implementing Platt Scaling in Production

When deploying a model with Platt scaling in a production environment, it's important to consider factors such as computational efficiency, model updates, and monitoring calibration quality over time.

```python
import joblib

# Train and calibrate the model
svm = SVC(kernel='rbf', random_state=42)
platt_svm = CalibratedClassifierCV(svm, method='sigmoid', cv=5)
platt_svm.fit(X, y)

# Save the calibrated model
joblib.dump(platt_svm, 'calibrated_model.joblib')

# In production:
def predict_proba(X_new):
    # Load the model (only once in practice)
    model = joblib.load('calibrated_model.joblib')
    
    # Make predictions
    probabilities = model.predict_proba(X_new)
    
    return probabilities

# Simulate new data
X_new = np.random.rand(5, X.shape[1])

# Get predictions
predictions = predict_proba(X_new)
print("Predictions for new data:")
print(predictions)

# Monitor calibration quality
def monitor_calibration(y_true, y_pred):
    brier_score = brier_score_loss(y_true, y_pred)
    log_loss_score = log_loss(y_true, y_pred)
    
    print(f"Brier Score: {brier_score:.4f}")
    print(f"Log Loss: {log_loss_score:.4f}")
    
    # If scores exceed thresholds, trigger recalibration
    if brier_score > 0.1 or log_loss_score > 0.3:
        print("Recalibration recommended")

# Simulate monitoring
y_true = np.random.randint(0, 2, 100)
y_pred = np.random.rand(100)
monitor_calibration(y_true, y_pred)
```

Slide 14: Conclusion and Best Practices

Platt scaling is a powerful technique for improving the calibration of machine learning models. To make the most of it:

1. Always evaluate the need for calibration before applying Platt scaling.
2. Use cross-validation when learning the scaling parameters to improve robustness.
3. Regularly monitor the calibration quality of your model in production.
4. Consider alternative calibration methods if Platt scaling doesn't perform well for your specific problem.
5. Be aware of the limitations, such as sensitivity to outliers and assumptions about the sigmoid function.

By following these practices, you can ensure that your models provide well-calibrated probability estimates, leading to more reliable and interpretable predictions in various applications.

Slide 15: Additional Resources

For those interested in diving deeper into Platt scaling and model calibration, here are some valuable resources:

1. Platt, J. (1999). "Probabilistic Outputs for Support Vector Machines and Comparisons to Regularized Likelihood Methods". Advances in Large Margin Classifiers. ArXiv: [https://arxiv.org/abs/1109.2378](https://arxiv.org/abs/1109.2378)
2. Niculescu-Mizil, A., & Caruana, R. (2005). "Predicting Good Probabilities with Supervised Learning". Proceedings of the 22nd International Conference on Machine Learning. ArXiv: [https://arxiv.org/abs/1206.6393](https://arxiv.org/abs/1206.6393)
3. Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). "On Calibration of Modern Neural Networks". ArXiv: [https://arxiv.org/abs/1706.04599](https://arxiv.org/abs/1706.04599)

These papers provide in-depth discussions on the theory and practical aspects of model calibration, including Platt scaling and other techniques.

