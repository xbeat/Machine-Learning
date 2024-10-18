## Identifying Harmful Features in Classification Models with Python

Slide 1: Introduction to Harmful Features in Classification Models

Classification models are powerful tools in machine learning, but they can sometimes rely on harmful or biased features. This presentation will explore techniques to identify and mitigate such features using Python, ensuring our models are fair and ethical.

```python
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Load a sample dataset
data = pd.read_csv('sample_data.csv')
X = data.drop('target', axis=1)
y = data['target']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a simple logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")
```

Slide 2: Understanding Feature Importance

Feature importance helps us identify which features have the most impact on our model's predictions. For linear models like logistic regression, we can use the coefficients to gauge feature importance.

```python
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': abs(model.coef_[0])
})
feature_importance = feature_importance.sort_values('importance', ascending=False)

print("Top 5 most important features:")
print(feature_importance.head())

# Visualize feature importance
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.bar(feature_importance['feature'][:10], feature_importance['importance'][:10])
plt.title("Top 10 Feature Importance")
plt.xlabel("Features")
plt.ylabel("Importance")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
```

Slide 3: Correlation Analysis

Correlation analysis helps identify relationships between features and the target variable. High correlation with sensitive attributes may indicate potential bias.

```python
correlation_matrix = data.corr()

# Visualize correlation matrix
plt.figure(figsize=(12, 10))
plt.imshow(correlation_matrix, cmap='coolwarm', aspect='auto')
plt.colorbar()
plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns, rotation=90)
plt.yticks(range(len(correlation_matrix.columns)), correlation_matrix.columns)
plt.title("Correlation Matrix")
plt.tight_layout()
plt.show()

# Print correlation with target variable
print("Correlation with target variable:")
print(correlation_matrix['target'].sort_values(ascending=False))
```

Slide 4: Identifying Proxy Variables

Proxy variables can inadvertently introduce bias by serving as substitutes for sensitive attributes. We'll examine how to detect these variables through correlation and feature importance analysis.

```python
sensitive_attribute = 'gender'
threshold = 0.7

proxy_candidates = correlation_matrix[sensitive_attribute][
    (abs(correlation_matrix[sensitive_attribute]) > threshold) &
    (correlation_matrix.index != sensitive_attribute)
]

print("Potential proxy variables for", sensitive_attribute)
print(proxy_candidates)

# Check importance of proxy candidates
proxy_importance = feature_importance[feature_importance['feature'].isin(proxy_candidates.index)]
print("\nImportance of potential proxy variables:")
print(proxy_importance)
```

Slide 5: Real-Life Example: Job Application Screening

Consider a job application screening model that predicts whether a candidate should be interviewed. We'll identify potentially harmful features in this context.

```python
job_data = pd.DataFrame({
    'years_experience': [2, 5, 3, 8, 1, 4],
    'education_level': [16, 18, 16, 20, 14, 16],
    'age': [25, 35, 28, 40, 22, 30],
    'interview': [0, 1, 0, 1, 0, 1]
})

# Train a logistic regression model
X_job = job_data.drop('interview', axis=1)
y_job = job_data['interview']
job_model = LogisticRegression()
job_model.fit(X_job, y_job)

# Analyze feature importance
job_importance = pd.DataFrame({
    'feature': X_job.columns,
    'importance': abs(job_model.coef_[0])
})
job_importance = job_importance.sort_values('importance', ascending=False)

print("Feature importance in job screening model:")
print(job_importance)

# Check correlation with age (potentially sensitive attribute)
print("\nCorrelation with age:")
print(job_data.corr()['age'].sort_values(ascending=False))
```

Slide 6: Mitigating Harmful Features: Feature Selection

One approach to mitigate harmful features is through careful feature selection. We'll demonstrate how to remove potentially biased features and retrain the model.

```python
harmful_features = ['age']
X_job_filtered = X_job.drop(columns=harmful_features)

# Retrain the model
job_model_filtered = LogisticRegression()
job_model_filtered.fit(X_job_filtered, y_job)

# Compare feature importance
job_importance_filtered = pd.DataFrame({
    'feature': X_job_filtered.columns,
    'importance': abs(job_model_filtered.coef_[0])
})
job_importance_filtered = job_importance_filtered.sort_values('importance', ascending=False)

print("Feature importance after removing harmful features:")
print(job_importance_filtered)

# Compare model performance (in practice, use a separate test set)
original_accuracy = job_model.score(X_job, y_job)
filtered_accuracy = job_model_filtered.score(X_job_filtered, y_job)

print(f"\nOriginal model accuracy: {original_accuracy:.2f}")
print(f"Filtered model accuracy: {filtered_accuracy:.2f}")
```

Slide 7: Fairness Metrics

Fairness metrics help assess whether a model treats different groups equally. We'll introduce some common fairness metrics and how to calculate them.

```python

def calculate_fairness_metrics(y_true, y_pred, sensitive_feature):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Calculate metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    
    # Calculate group-specific metrics
    group_metrics = {}
    for group in sensitive_feature.unique():
        mask = sensitive_feature == group
        group_cm = confusion_matrix(y_true[mask], y_pred[mask])
        group_tn, group_fp, group_fn, group_tp = group_cm.ravel()
        
        group_metrics[group] = {
            'accuracy': (group_tp + group_tn) / (group_tp + group_tn + group_fp + group_fn),
            'precision': group_tp / (group_tp + group_fp) if (group_tp + group_fp) > 0 else 0,
            'recall': group_tp / (group_tp + group_fn) if (group_tp + group_fn) > 0 else 0
        }
    
    return accuracy, precision, recall, group_metrics

# Example usage
sensitive_attribute = pd.Series([0, 1, 0, 1, 0, 1])  # 0: Group A, 1: Group B
y_true = np.array([0, 1, 1, 1, 0, 1])
y_pred = np.array([0, 1, 0, 1, 0, 1])

accuracy, precision, recall, group_metrics = calculate_fairness_metrics(y_true, y_pred, sensitive_attribute)

print(f"Overall Accuracy: {accuracy:.2f}")
print(f"Overall Precision: {precision:.2f}")
print(f"Overall Recall: {recall:.2f}")

for group, metrics in group_metrics.items():
    print(f"\nGroup {group} metrics:")
    for metric, value in metrics.items():
        print(f"{metric.capitalize()}: {value:.2f}")
```

Slide 8: Disparate Impact Analysis

Disparate impact measures whether a model's decisions disproportionately affect different groups. We'll implement a function to calculate and visualize disparate impact.

```python

def calculate_disparate_impact(y_pred, sensitive_feature, privileged_group):
    groups = sensitive_feature.unique()
    group_positive_rates = {}
    
    for group in groups:
        mask = sensitive_feature == group
        group_positive_rate = y_pred[mask].mean()
        group_positive_rates[group] = group_positive_rate
    
    disparate_impact = group_positive_rates[privileged_group] / max(
        [rate for group, rate in group_positive_rates.items() if group != privileged_group]
    )
    
    return disparate_impact, group_positive_rates

# Example usage
sensitive_attribute = pd.Series([0, 1, 0, 1, 0, 1, 0, 1])  # 0: Group A, 1: Group B
y_pred = np.array([1, 1, 0, 1, 0, 1, 1, 0])
privileged_group = 1  # Group B is considered privileged

di, group_rates = calculate_disparate_impact(y_pred, sensitive_attribute, privileged_group)

print(f"Disparate Impact: {di:.2f}")
print("Group Positive Rates:")
for group, rate in group_rates.items():
    print(f"Group {group}: {rate:.2f}")

# Visualize group positive rates
plt.figure(figsize=(8, 6))
plt.bar(group_rates.keys(), group_rates.values())
plt.title("Positive Prediction Rates by Group")
plt.xlabel("Group")
plt.ylabel("Positive Prediction Rate")
plt.ylim(0, 1)
for i, v in enumerate(group_rates.values()):
    plt.text(i, v + 0.01, f'{v:.2f}', ha='center')
plt.show()
```

Slide 9: Real-Life Example: Medical Diagnosis

Let's examine a medical diagnosis model for a hypothetical disease and identify potential biases in the features used.

```python
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Generate synthetic medical data
np.random.seed(42)
n_samples = 1000

data = pd.DataFrame({
    'age': np.random.randint(18, 80, n_samples),
    'blood_pressure': np.random.randint(90, 180, n_samples),
    'cholesterol': np.random.randint(120, 300, n_samples),
    'bmi': np.random.uniform(18, 35, n_samples),
    'smoking': np.random.choice([0, 1], n_samples),
    'exercise': np.random.choice([0, 1, 2], n_samples),  # 0: low, 1: moderate, 2: high
    'gender': np.random.choice([0, 1], n_samples),  # 0: female, 1: male
    'ethnicity': np.random.choice([0, 1, 2, 3], n_samples)  # 0, 1, 2, 3 represent different ethnicities
})

# Generate target variable (diagnosis) with some bias
data['diagnosis'] = (
    (data['age'] > 50) * 0.3 +
    (data['blood_pressure'] > 140) * 0.2 +
    (data['cholesterol'] > 200) * 0.2 +
    (data['bmi'] > 30) * 0.1 +
    (data['smoking'] == 1) * 0.1 +
    (data['exercise'] == 0) * 0.1 +
    (data['gender'] == 1) * 0.05 +  # Slight gender bias
    (data['ethnicity'] == 2) * 0.05  # Slight ethnic bias
)
data['diagnosis'] = (data['diagnosis'] > np.random.uniform(0, 1, n_samples)).astype(int)

# Split the data
X = data.drop('diagnosis', axis=1)
y = data['diagnosis']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a random forest classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Analyze feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
})
feature_importance = feature_importance.sort_values('importance', ascending=False)

print("\nFeature Importance:")
print(feature_importance)

# Check for potential biases
sensitive_attributes = ['gender', 'ethnicity']
for attr in sensitive_attributes:
    print(f"\nCorrelation with {attr}:")
    print(data.corr()[attr].sort_values(ascending=False))
```

Slide 10: Addressing Algorithmic Bias

To address algorithmic bias, we can employ techniques such as resampling, adjusting class weights, or using fairness-aware algorithms. Let's demonstrate how to use class weights to mitigate bias.

```python

# Compute class weights
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(zip(np.unique(y_train), class_weights))

# Train a new model with class weights
model_weighted = RandomForestClassifier(n_estimators=100, random_state=42, class_weight=class_weight_dict)
model_weighted.fit(X_train, y_train)

# Evaluate the weighted model
y_pred_weighted = model_weighted.predict(X_test)
accuracy_weighted = accuracy_score(y_test, y_pred_weighted)
print(f"Weighted Model Accuracy: {accuracy_weighted:.2f}")

# Compare feature importance
feature_importance_weighted = pd.DataFrame({
    'feature': X.columns,
    'importance': model_weighted.feature_importances_
})
feature_importance_weighted = feature_importance_weighted.sort_values('importance', ascending=False)

print("\nFeature Importance (Weighted Model):")
print(feature_importance_weighted)

# Calculate and compare fairness metrics for both models
for attr in sensitive_attributes:
    print(f"\nFairness metrics for {attr}:")
    
    # Original model
    _, _, _, group_metrics_original = calculate_fairness_metrics(y_test, y_pred, X_test[attr])
    print("Original model:")
    for group, metrics in group_metrics_original.items():
        print(f"Group {group}: Accuracy = {metrics['accuracy']:.2f}, Precision = {metrics['precision']:.2f}, Recall = {metrics['recall']:.2f}")
    
    # Weighted model
    _, _, _, group_metrics_weighted = calculate_fairness_metrics(y_test, y_pred_weighted, X_test[attr])
    print("Weighted model:")
    for group, metrics in group_metrics_weighted.items():
        print(f"Group {group}: Accuracy = {metrics['accuracy']:.2f}, Precision = {metrics['precision']:.2f}, Recall = {metrics['recall']:.2f}")
```

Slide 11: Regularization for Feature Selection

Regularization techniques can help reduce the impact of potentially harmful features. L1 regularization (Lasso) can be particularly useful for feature selection.

```python

# Train a logistic regression model with L1 regularization
lasso_model = LogisticRegression(penalty='l1', solver='liblinear', C=0.1)
lasso_model.fit(X_train, y_train)

# Get feature coefficients
feature_coef = pd.DataFrame({
    'feature': X.columns,
    'coefficient': lasso_model.coef_[0]
})
feature_coef = feature_coef.sort_values('coefficient', key=abs, ascending=False)

print("Feature coefficients after L1 regularization:")
print(feature_coef)

# Identify features with non-zero coefficients
selected_features = feature_coef[feature_coef['coefficient'] != 0]['feature'].tolist()

print("\nSelected features:")
print(selected_features)

# Train a new model using only selected features
X_train_selected = X_train[selected_features]
X_test_selected = X_test[selected_features]

model_selected = RandomForestClassifier(n_estimators=100, random_state=42)
model_selected.fit(X_train_selected, y_train)

# Evaluate the model with selected features
y_pred_selected = model_selected.predict(X_test_selected)
accuracy_selected = accuracy_score(y_test, y_pred_selected)
print(f"\nModel Accuracy (Selected Features): {accuracy_selected:.2f}")
```

Slide 12: Interpreting Model Decisions

Understanding individual model decisions is crucial for identifying potential biases. We'll use SHAP (SHapley Additive exPlanations) values to interpret our model's predictions.

```python

# Create a SHAP explainer
explainer = shap.TreeExplainer(model)

# Calculate SHAP values for the test set
shap_values = explainer.shap_values(X_test)

# Visualize SHAP values for a single prediction
sample_idx = 0
shap.force_plot(explainer.expected_value[1], shap_values[1][sample_idx], X_test.iloc[sample_idx])

# Visualize feature importance based on SHAP values
shap.summary_plot(shap_values[1], X_test)

# Analyze SHAP values for sensitive attributes
sensitive_attrs = ['gender', 'ethnicity']
for attr in sensitive_attrs:
    shap.dependence_plot(attr, shap_values[1], X_test)
```

Slide 13: Continuous Monitoring and Feedback Loop

Identifying harmful features is an ongoing process. Implement a monitoring system to track model performance and fairness metrics over time.

```python

def monitor_model_performance(model, X, y, sensitive_attributes):
    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred)
    
    fairness_metrics = {}
    for attr in sensitive_attributes:
        _, _, _, group_metrics = calculate_fairness_metrics(y, y_pred, X[attr])
        fairness_metrics[attr] = group_metrics
    
    return {
        'timestamp': datetime.datetime.now(),
        'accuracy': accuracy,
        'fairness_metrics': fairness_metrics
    }

# Simulate monitoring over time
monitoring_results = []
for _ in range(5):  # Simulate 5 time periods
    result = monitor_model_performance(model, X_test, y_test, sensitive_attributes)
    monitoring_results.append(result)
    
    # In a real scenario, you would wait for a certain time period and collect new data
    # Here, we're just shuffling the test data to simulate changes
    X_test, y_test = shuffle(X_test, y_test, random_state=np.random.randint(0, 1000))

# Analyze monitoring results
for result in monitoring_results:
    print(f"\nTimestamp: {result['timestamp']}")
    print(f"Accuracy: {result['accuracy']:.2f}")
    for attr, metrics in result['fairness_metrics'].items():
        print(f"Fairness metrics for {attr}:")
        for group, group_metrics in metrics.items():
            print(f"  Group {group}: Accuracy = {group_metrics['accuracy']:.2f}, "
                  f"Precision = {group_metrics['precision']:.2f}, "
                  f"Recall = {group_metrics['recall']:.2f}")
```

Slide 14: Conclusion and Best Practices

Identifying harmful features in classification models is crucial for building fair and ethical AI systems. Key takeaways include:

1. Regularly analyze feature importance and correlation with sensitive attributes.
2. Use fairness metrics to assess model performance across different groups.
3. Employ techniques like feature selection, regularization, and fairness-aware algorithms to mitigate bias.
4. Interpret individual model decisions using tools like SHAP values.
5. Implement continuous monitoring to track model performance and fairness over time.
6. Foster a diverse and inclusive team to bring varied perspectives to the model development process.
7. Stay informed about the latest research and best practices in AI ethics and fairness.

By following these practices, we can work towards creating more equitable and reliable classification models.

Slide 15: Additional Resources

For further reading on identifying harmful features and building fair classification models, consider the following resources:

1. "A Survey on Bias and Fairness in Machine Learning" by Mehrabi et al. (2019) ArXiv: [https://arxiv.org/abs/1908.09635](https://arxiv.org/abs/1908.09635)
2. "What-If Tool: An Interactive Visual Interface for Model Understanding" by Wexler et al. (2020) ArXiv: [https://arxiv.org/abs/1907.04135](https://arxiv.org/abs/1907.04135)
3. "Fairness and Machine Learning: Limitations and Opportunities" by Barocas, Hardt, and Narayanan (2019) Available at: [https://fairmlbook.org/](https://fairmlbook.org/)
4. "Interpreting Machine Learning Models: An Overview" by Molnar et al. (2020) ArXiv: [https://arxiv.org/abs/2010.13251](https://arxiv.org/abs/2010.13251)

These resources provide in-depth discussions on fairness in machine learning, interpretability techniques, and practical tools for identifying and mitigating harmful features in classification models.

