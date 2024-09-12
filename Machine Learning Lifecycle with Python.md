## Machine Learning Lifecycle with Python
Slide 1: Introduction to Machine Learning Lifecycle

The machine learning lifecycle encompasses the entire process of developing, deploying, and maintaining machine learning models. It includes data collection, preprocessing, model selection, training, evaluation, deployment, and monitoring. Understanding this lifecycle is crucial for effectively managing machine learning projects using Python.

```python
import matplotlib.pyplot as plt

lifecycle_stages = ['Data Collection', 'Preprocessing', 'Model Selection', 
                    'Training', 'Evaluation', 'Deployment', 'Monitoring']
stage_durations = [10, 15, 5, 20, 10, 5, 35]

plt.figure(figsize=(12, 6))
plt.pie(stage_durations, labels=lifecycle_stages, autopct='%1.1f%%')
plt.title('Typical Time Distribution in ML Lifecycle')
plt.axis('equal')
plt.show()
```

Slide 2: Data Collection and Preprocessing

Data collection involves gathering relevant data from various sources. Preprocessing is the crucial step of cleaning and preparing the data for model training. This includes handling missing values, encoding categorical variables, and scaling numerical features.

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load and preprocess data
data = pd.read_csv('raw_data.csv')
data['age'].fillna(data['age'].mean(), inplace=True)
data['gender'] = pd.get_dummies(data['gender'], drop_first=True)

scaler = StandardScaler()
data['income'] = scaler.fit_transform(data[['income']])

print(data.head())
```

Slide 3: Exploratory Data Analysis (EDA)

EDA is a critical step in understanding the dataset's characteristics, distributions, and relationships between variables. It helps in feature selection and guides the choice of appropriate models.

```python
import seaborn as sns
import matplotlib.pyplot as plt

# Pairplot for visualizing relationships
sns.pairplot(data, vars=['age', 'income', 'education_years'], hue='gender')
plt.show()

# Correlation heatmap
correlation_matrix = data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()
```

Slide 4: Feature Selection and Engineering

Feature selection involves choosing the most relevant features for the model, while feature engineering creates new features to improve model performance. These steps are crucial for building effective machine learning models.

```python
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import PolynomialFeatures

# Feature selection
X = data.drop('target', axis=1)
y = data['target']
selector = SelectKBest(f_classif, k=5)
X_selected = selector.fit_transform(X, y)

print("Selected features:", X.columns[selector.get_support()].tolist())

# Feature engineering
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X_selected)
print("Engineered features shape:", X_poly.shape)
```

Slide 5: Model Selection

Choosing the right model depends on the problem type, dataset characteristics, and project requirements. It's often beneficial to try multiple models and compare their performance.

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    'Logistic Regression': LogisticRegression(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier()
}

for name, model in models.items():
    model.fit(X_train, y_train)
    print(f"{name} accuracy: {model.score(X_test, y_test):.4f}")
```

Slide 6: Model Training and Hyperparameter Tuning

Training involves fitting the model to the data, while hyperparameter tuning optimizes the model's performance. Grid search and random search are common techniques for finding the best hyperparameters.

```python
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5, 10]
}

rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

print("Best parameters:", grid_search.best_params_)
print("Best cross-validation score:", grid_search.best_score_)
```

Slide 7: Model Evaluation

Evaluation metrics help assess model performance. The choice of metrics depends on the problem type (classification, regression) and specific project requirements.

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

y_pred = grid_search.best_estimator_.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, average='weighted'))
print("Recall:", recall_score(y_test, y_pred, average='weighted'))
print("F1-score:", f1_score(y_test, y_pred, average='weighted'))
```

Slide 8: Cross-Validation

Cross-validation helps assess model performance more robustly by using multiple train-test splits. It provides a better estimate of how the model will perform on unseen data.

```python
from sklearn.model_selection import cross_val_score

best_model = grid_search.best_estimator_
cv_scores = cross_val_score(best_model, X, y, cv=5)

print("Cross-validation scores:", cv_scores)
print("Mean CV score:", cv_scores.mean())
print("Standard deviation of CV scores:", cv_scores.std())
```

Slide 9: Model Interpretation

Understanding how a model makes decisions is crucial for trust and debugging. Techniques like feature importance and SHAP values can provide insights into model behavior.

```python
import shap

explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_test)

shap.summary_plot(shap_values, X_test, plot_type="bar")
plt.show()

shap.summary_plot(shap_values, X_test)
plt.show()
```

Slide 10: Model Deployment

Deploying a model involves making it available for use in production environments. This can be done through various methods, such as REST APIs or batch processing systems.

```python
import joblib
from flask import Flask, request, jsonify

# Save the model
joblib.dump(best_model, 'model.joblib')

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    model = joblib.load('model.joblib')
    prediction = model.predict(data['features'])
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True)
```

Slide 11: Model Monitoring and Maintenance

Continuous monitoring of model performance in production is essential. This includes tracking prediction quality, data drift, and retraining the model when necessary.

```python
import numpy as np
from scipy.stats import ks_2samp

def detect_data_drift(reference_data, new_data, threshold=0.1):
    drift_detected = False
    for column in reference_data.columns:
        ks_statistic, p_value = ks_2samp(reference_data[column], new_data[column])
        if p_value < threshold:
            print(f"Drift detected in feature {column}: KS statistic = {ks_statistic}, p-value = {p_value}")
            drift_detected = True
    return drift_detected

# Simulate new data
new_data = X.()
new_data['age'] += np.random.normal(0, 5, size=len(new_data))

if detect_data_drift(X, new_data):
    print("Consider retraining the model with new data")
else:
    print("No significant data drift detected")
```

Slide 12: Real-Life Example: Predictive Maintenance

In this example, we'll use machine learning to predict equipment failures in a manufacturing plant. This application of the ML lifecycle can help reduce downtime and maintenance costs.

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Load and preprocess data
data = pd.read_csv('equipment_data.csv')
X = data.drop('failure', axis=1)
y = data['failure']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Feature importance
importance = pd.DataFrame({'feature': X.columns, 'importance': model.feature_importances_})
importance = importance.sort_values('importance', ascending=False)
print(importance.head(10))
```

Slide 13: Real-Life Example: Customer Churn Prediction

In this example, we'll apply the ML lifecycle to predict customer churn for a subscription-based service. This can help businesses retain customers and improve customer satisfaction.

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
import shap

# Load and preprocess data
data = pd.read_csv('customer_data.csv')
X = data.drop('churned', axis=1)
y = data['churned']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = GradientBoostingClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Evaluate model
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
print("ROC AUC Score:", roc_auc_score(y_test, y_pred_proba))

# Model interpretation
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test_scaled)
shap.summary_plot(shap_values, X_test)
```

Slide 14: Additional Resources

For further exploration of machine learning lifecycle management using Python, consider the following resources:

1. "A Survey of Data Management in Machine Learning Systems" by Schelter et al. (2020) ArXiv: [https://arxiv.org/abs/2010.00821](https://arxiv.org/abs/2010.00821)
2. "MLOps: Continuous Delivery and Automation Pipelines in Machine Learning" by Treveil et al. (2020) O'Reilly Media
3. "Towards MLOps: A Framework and Maturity Model" by Shankar et al. (2021) ArXiv: [https://arxiv.org/abs/2103.07974](https://arxiv.org/abs/2103.07974)
4. Scikit-learn Documentation: [https://scikit-learn.org/stable/](https://scikit-learn.org/stable/)
5. TensorFlow Extended (TFX) Documentation: [https://www.tensorflow.org/tfx](https://www.tensorflow.org/tfx)

