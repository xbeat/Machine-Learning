## Breast Cancer Survival Prediction with Machine Learning
Slide 1: Introduction to Breast Cancer Survival Prediction

Breast cancer survival prediction using gene expression and clinical data is a crucial area of research in oncology. This presentation will guide you through the process of preparing data and applying machine learning algorithms for survival prediction.

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sklearn.linear_model import LogisticRegression
import xgboost as xgb

# Load sample data (replace with actual data loading)
data = pd.read_csv('breast_cancer_data.csv')
X = data.drop(['survival_time', 'event'], axis=1)
y = np.array([(data['event'][i], data['survival_time'][i]) for i in range(len(data))], 
             dtype=[('event', bool), ('time', float)])
```

Slide 2: Data Preparation

Data preparation is a critical step in survival prediction. We need to handle missing values, encode categorical variables, and normalize numerical features.

```python
# Handle missing values
X = X.fillna(X.mean())

# Encode categorical variables
X = pd.get_dummies(X, columns=['ER_status', 'PR_status', 'HER2_status'])

# Normalize numerical features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

print(X_scaled.head())
```

Slide 3: Train-Test Split

Splitting the data into training and testing sets is crucial for evaluating model performance.

```python
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

print(f"Training set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}")
```

Slide 4: Baseline Survival Forecasting Benchmark

We'll use the Kaplan-Meier estimator as our baseline benchmark for survival prediction.

```python
from sksurv.nonparametric import kaplan_meier_estimator

time, survival_prob, _ = kaplan_meier_estimator(y_train['event'], y_train['time'])

import matplotlib.pyplot as plt

plt.step(time, survival_prob, where="post")
plt.ylabel("est. probability of survival $\hat{S}(t)$")
plt.xlabel("time $t$")
plt.title("Kaplan-Meier Baseline Survival Curve")
plt.grid(True)
plt.show()
```

Slide 5: Scikit-Survival Gradient Boosting

Gradient Boosting is a powerful ensemble method that can be applied to survival analysis.

```python
gb_model = GradientBoostingSurvivalAnalysis(n_estimators=100, learning_rate=0.1, random_state=42)
gb_model.fit(X_train, y_train)

gb_pred = gb_model.predict(X_test)
print(f"Gradient Boosting prediction shape: {gb_pred.shape}")
```

Slide 6: Scikit-Survival Cox Proportional Hazards

The Cox Proportional Hazards model is a semi-parametric model widely used in survival analysis.

```python
cph_model = CoxPHSurvivalAnalysis()
cph_model.fit(X_train, y_train)

cph_pred = cph_model.predict(X_test)
print(f"Cox PH prediction shape: {cph_pred.shape}")
```

Slide 7: Logistic Regression for Binary Classification

While not a survival model per se, logistic regression can be used to predict the probability of an event occurring within a specific time frame.

```python
# Convert survival data to binary classification problem
threshold_time = y['time'].median()
y_binary = (y['time'] <= threshold_time) & y['event']

X_train_lr, X_test_lr, y_train_lr, y_test_lr = train_test_split(X_scaled, y_binary, test_size=0.2, random_state=42)

lr_model = LogisticRegression(random_state=42)
lr_model.fit(X_train_lr, y_train_lr)

lr_pred = lr_model.predict_proba(X_test_lr)[:, 1]
print(f"Logistic Regression prediction shape: {lr_pred.shape}")
```

Slide 8: XGBoost for Survival Analysis

XGBoost can be adapted for survival analysis by treating it as a ranking problem.

```python
def xgb_objective(y_true, y_pred):
    d = y_true.get_label()
    theta = 1.0 / (1.0 + np.exp(-y_pred)) - 0.5
    return (d * theta - np.log(1.0 + np.exp(y_pred))).sum()

xgb_model = xgb.XGBRegressor(objective=xgb_objective, n_estimators=100, learning_rate=0.1, random_state=42)
xgb_model.fit(X_train, y_train['time'])

xgb_pred = xgb_model.predict(X_test)
print(f"XGBoost prediction shape: {xgb_pred.shape}")
```

Slide 9: Model Evaluation

Evaluating survival models requires specialized metrics such as Harrell's C-index and the Brier score.

```python
from sksurv.metrics import concordance_index_censored, brier_score

def evaluate_model(y_true, y_pred):
    c_index = concordance_index_censored(y_true['event'], y_true['time'], y_pred)[0]
    brier = brier_score(y_true, y_pred, y_train['time'], y_train['event'])
    return c_index, brier.mean()

models = {
    "Gradient Boosting": gb_pred,
    "Cox PH": cph_pred,
    "XGBoost": xgb_pred
}

for name, preds in models.items():
    c_index, brier = evaluate_model(y_test, preds)
    print(f"{name}: C-index = {c_index:.3f}, Brier Score = {brier:.3f}")
```

Slide 10: Feature Importance

Understanding which features are most important for survival prediction can provide valuable insights.

```python
import shap

explainer = shap.TreeExplainer(gb_model)
shap_values = explainer.shap_values(X_test)

shap.summary_plot(shap_values, X_test, plot_type="bar")
plt.title("Feature Importance (SHAP values)")
plt.show()
```

Slide 11: Survival Curves for Individual Patients

Visualizing survival curves for individual patients can help in personalized treatment planning.

```python
def plot_individual_survival(model, patient_data):
    survival_funcs = model.predict_survival_function(patient_data)
    for i, sf in enumerate(survival_funcs):
        plt.step(sf.x, sf.y, where="post", label=f"Patient {i+1}")
    plt.ylabel("Survival probability")
    plt.xlabel("Time")
    plt.title("Individual Patient Survival Curves")
    plt.legend()
    plt.grid(True)
    plt.show()

sample_patients = X_test.iloc[:3]
plot_individual_survival(gb_model, sample_patients)
```

Slide 12: Real-Life Example: Predicting Survival in Clinical Trials

In clinical trials for new breast cancer treatments, survival prediction models can help researchers estimate the potential benefit of the treatment.

```python
# Simulating clinical trial data
np.random.seed(42)
n_patients = 1000
treatment = np.random.choice([0, 1], size=n_patients)
age = np.random.normal(55, 10, n_patients)
stage = np.random.choice([1, 2, 3, 4], size=n_patients)
gene_expr = np.random.normal(0, 1, (n_patients, 10))

X_trial = pd.DataFrame({
    'treatment': treatment,
    'age': age,
    'stage': stage,
    **{f'gene_{i}': gene_expr[:, i] for i in range(10)}
})

# Simulate survival times
baseline_hazard = 0.05
treatment_effect = 0.7
survival_times = np.random.exponential(
    1 / (baseline_hazard * np.exp(-treatment_effect * treatment + 0.02 * (age - 55) + 0.2 * (stage - 1)))
)
censoring_times = np.random.uniform(0, 10, n_patients)
observed_times = np.minimum(survival_times, censoring_times)
events = (survival_times <= censoring_times).astype(bool)

y_trial = np.array(list(zip(events, observed_times)), dtype=[('event', bool), ('time', float)])

# Train and evaluate model
X_train, X_test, y_train, y_test = train_test_split(X_trial, y_trial, test_size=0.2, random_state=42)
model = CoxPHSurvivalAnalysis().fit(X_train, y_train)

c_index = concordance_index_censored(y_test['event'], y_test['time'], model.predict(X_test))[0]
print(f"C-index on test set: {c_index:.3f}")

# Visualize treatment effect
treatment_effect = model.coef_[0]
plt.bar(['Treatment Effect'], [treatment_effect])
plt.title("Estimated Treatment Effect")
plt.ylabel("Coefficient")
plt.show()
```

Slide 13: Real-Life Example: Personalized Treatment Recommendations

Survival prediction models can assist oncologists in making personalized treatment recommendations based on a patient's individual characteristics.

```python
# Simulating patient data
patient_data = pd.DataFrame({
    'age': [45, 60, 55],
    'stage': [2, 3, 1],
    'ER_status': ['positive', 'negative', 'positive'],
    'PR_status': ['positive', 'negative', 'positive'],
    'HER2_status': ['negative', 'positive', 'negative'],
    'tumor_size': [2.5, 4.0, 1.8],
    'grade': [2, 3, 1]
})

# Preprocess patient data (assuming we have the preprocessor from earlier)
patient_data_processed = preprocess_data(patient_data)

# Predict survival probabilities
survival_probs = gb_model.predict_survival_function(patient_data_processed)

# Plot survival curves
for i, sf in enumerate(survival_probs):
    plt.step(sf.x, sf.y, where="post", label=f"Patient {i+1}")

plt.xlabel("Time (years)")
plt.ylabel("Survival Probability")
plt.title("Personalized Survival Curves")
plt.legend()
plt.grid(True)
plt.show()

# Recommend treatment based on 5-year survival probability
for i, sf in enumerate(survival_probs):
    five_year_prob = sf(5)
    if five_year_prob > 0.8:
        recommendation = "Standard treatment"
    elif five_year_prob > 0.5:
        recommendation = "Consider more aggressive treatment"
    else:
        recommendation = "Recommend aggressive treatment and close monitoring"
    print(f"Patient {i+1}: 5-year survival probability = {five_year_prob:.2f}, Recommendation: {recommendation}")
```

Slide 14: Additional Resources

For further exploration of breast cancer survival prediction using machine learning:

1. "Machine Learning for Survival Analysis: A Survey" by Wang et al. (2019) ArXiv: [https://arxiv.org/abs/1708.04649](https://arxiv.org/abs/1708.04649)
2. "Deep Learning for Survival Analysis: A Review" by Ren et al. (2022) ArXiv: [https://arxiv.org/abs/2110.12158](https://arxiv.org/abs/2110.12158)
3. "Integrating Genomic Data and Clinical Outcomes for Precision Oncology" by Yousefi et al. (2021) ArXiv: [https://arxiv.org/abs/2103.14704](https://arxiv.org/abs/2103.14704)

These resources provide in-depth discussions on advanced techniques and current research in survival analysis and precision oncology.
