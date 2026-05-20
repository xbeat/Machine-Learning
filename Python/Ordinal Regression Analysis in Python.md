## Ordinal Regression Analysis in Python
Slide 1: Ordinal Regression in Python

Ordinal regression is a statistical method used when the dependent variable is ordinal, meaning it has categories with a natural order. This type of regression is crucial for analyzing data where the outcome has a clear ranking but the distances between categories may not be equal. In this presentation, we'll explore different approaches to ordinal regression using Python, focusing on practical implementations and real-world applications.

```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

# Example of ordinal data
satisfaction_levels = ['Low', 'Medium', 'High']
data = pd.DataFrame({'Satisfaction': np.random.choice(satisfaction_levels, 100)})

# Encoding ordinal data
encoder = OrdinalEncoder(categories=[satisfaction_levels])
data['Satisfaction_Encoded'] = encoder.fit_transform(data[['Satisfaction']])

print(data.head())
```

Slide 2: Proportional Odds Model

The Proportional Odds Model, also known as the Cumulative Logit Model, is a common approach for ordinal regression. It assumes that the effect of independent variables is consistent across different thresholds of the ordinal outcome. This model extends logistic regression to handle ordinal outcomes by estimating the cumulative probability of being at or below a certain category.

```python
import statsmodels.formula.api as smf

# Simulating data
np.random.seed(42)
X = np.random.randn(1000, 2)
y = np.dot(X, [1, 2]) + np.random.randn(1000)
y_ordinal = pd.cut(y, bins=3, labels=['Low', 'Medium', 'High'])

data = pd.DataFrame({'X1': X[:, 0], 'X2': X[:, 1], 'Y': y_ordinal})

# Fitting the Proportional Odds Model
model = smf.ordinal_gam(formula='Y ~ X1 + X2', data=data)
results = model.fit()

print(results.summary())
```

Slide 3: Ordered Logit Model

The Ordered Logit Model is another approach to ordinal regression. It's similar to the Proportional Odds Model but uses the logistic function to model the cumulative probabilities. This model is particularly useful when you want to interpret the odds ratios of moving from one category to another.

```python
from mord import LogisticIT

# Preparing data
X = data[['X1', 'X2']]
y = data['Y'].cat.codes  # Convert categories to numeric codes

# Fitting the Ordered Logit Model
model = LogisticIT()
model.fit(X, y)

# Predicting probabilities
probs = model.predict_proba(X)

print("Coefficients:", model.coef_)
print("Intercepts:", model.theta_)
print("Predicted probabilities shape:", probs.shape)
```

Slide 4: Ordinal Regression with scikit-learn

While scikit-learn doesn't have a built-in ordinal regression model, we can adapt existing models to handle ordinal outcomes. One approach is to use a series of binary classifiers, known as the one-vs-rest strategy, combined with an ordinal encoder.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OrdinalEncoder
from sklearn.multiclass import OneVsRestClassifier

# Prepare data
X = data[['X1', 'X2']]
y = data['Y']

# Encode target variable
encoder = OrdinalEncoder()
y_encoded = encoder.fit_transform(y.values.reshape(-1, 1)).ravel()

# Fit the model
model = OneVsRestClassifier(LogisticRegression())
model.fit(X, y_encoded)

# Predict
predictions = model.predict(X)
probabilities = model.predict_proba(X)

print("Predictions shape:", predictions.shape)
print("Probabilities shape:", probabilities.shape)
```

Slide 5: Ordinal Regression with MORD

MORD (Multiclass Ordinal Regression) is a Python package specifically designed for ordinal regression. It provides implementations of various ordinal regression models, including the Logistic Ordinal Regression and the All-Thresholds variant.

```python
from mord import LogisticAT

# Prepare data
X = data[['X1', 'X2']]
y = data['Y'].cat.codes

# Fit the model
model = LogisticAT()
model.fit(X, y)

# Predict
predictions = model.predict(X)
probabilities = model.predict_proba(X)

print("Coefficients:", model.coef_)
print("Intercepts:", model.theta_)
print("Predictions shape:", predictions.shape)
print("Probabilities shape:", probabilities.shape)
```

Slide 6: Feature Engineering for Ordinal Regression

Feature engineering plays a crucial role in improving the performance of ordinal regression models. We can create interaction terms, polynomial features, or apply transformations to capture non-linear relationships between predictors and the ordinal outcome.

```python
from sklearn.preprocessing import PolynomialFeatures

# Create polynomial features
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)

# Fit the model with polynomial features
model = LogisticIT()
model.fit(X_poly, y)

# Predict using polynomial features
predictions = model.predict(X_poly)

print("Original feature shape:", X.shape)
print("Polynomial feature shape:", X_poly.shape)
print("Predictions shape:", predictions.shape)
```

Slide 7: Model Evaluation for Ordinal Regression

Evaluating ordinal regression models requires metrics that account for the ordered nature of the outcome. Common metrics include Mean Absolute Error (MAE), Kendall's Tau, and the Quadratic Weighted Kappa.

```python
from sklearn.metrics import mean_absolute_error, cohen_kappa_score
from scipy.stats import kendalltau

# Assuming we have true labels (y_true) and predictions (y_pred)
y_true = y
y_pred = model.predict(X)

# Calculate metrics
mae = mean_absolute_error(y_true, y_pred)
tau, _ = kendalltau(y_true, y_pred)
qwk = cohen_kappa_score(y_true, y_pred, weights='quadratic')

print(f"Mean Absolute Error: {mae:.4f}")
print(f"Kendall's Tau: {tau:.4f}")
print(f"Quadratic Weighted Kappa: {qwk:.4f}")
```

Slide 8: Cross-Validation for Ordinal Regression

Cross-validation is essential for assessing the generalization performance of ordinal regression models. We can use stratified k-fold cross-validation to maintain the distribution of ordinal categories across folds.

```python
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import make_scorer

# Define a custom scorer for ordinal regression
def ordinal_scorer(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    tau, _ = kendalltau(y_true, y_pred)
    qwk = cohen_kappa_score(y_true, y_pred, weights='quadratic')
    return (1 / mae) * (tau + qwk) / 2  # Higher is better

# Perform cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scorer = make_scorer(ordinal_scorer)

scores = cross_val_score(model, X, y, cv=cv, scoring=scorer)

print(f"Cross-validation scores: {scores}")
print(f"Mean score: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
```

Slide 9: Handling Class Imbalance in Ordinal Regression

Class imbalance can be a challenge in ordinal regression, especially when certain categories are underrepresented. We can address this using techniques like oversampling, undersampling, or adjusting class weights.

```python
from imblearn.over_sampling import SMOTENC

# Check class distribution
print("Original class distribution:")
print(y.value_counts(normalize=True))

# Apply SMOTEENC (SMOTE for Nominal and Continuous features)
smote = SMOTEENC(random_state=42, categorical_features=[False, False])
X_resampled, y_resampled = smote.fit_resample(X, y)

# Check new class distribution
print("\nResampled class distribution:")
print(pd.Series(y_resampled).value_counts(normalize=True))

# Fit model on resampled data
model = LogisticIT()
model.fit(X_resampled, y_resampled)

# Evaluate on original data
predictions = model.predict(X)
mae = mean_absolute_error(y, predictions)
print(f"\nMean Absolute Error on original data: {mae:.4f}")
```

Slide 10: Interpreting Ordinal Regression Results

Interpreting the results of ordinal regression models is crucial for gaining insights from the analysis. We can examine coefficients, odds ratios, and predicted probabilities to understand the impact of predictors on the ordinal outcome.

```python
import matplotlib.pyplot as plt

# Assuming we have fitted a LogisticIT model
coefficients = model.coef_
feature_names = X.columns

# Plot coefficients
plt.figure(figsize=(10, 6))
plt.bar(feature_names, coefficients)
plt.title("Feature Coefficients")
plt.xlabel("Features")
plt.ylabel("Coefficient Value")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Calculate and print odds ratios
odds_ratios = np.exp(coefficients)
for feature, odds_ratio in zip(feature_names, odds_ratios):
    print(f"Odds Ratio for {feature}: {odds_ratio:.4f}")

# Plot predicted probabilities for a single observation
single_obs = X.iloc[0]
probs = model.predict_proba(single_obs.values.reshape(1, -1))[0]
categories = encoder.categories_[0]

plt.figure(figsize=(8, 6))
plt.bar(categories, probs)
plt.title("Predicted Probabilities for Single Observation")
plt.xlabel("Categories")
plt.ylabel("Probability")
plt.tight_layout()
plt.show()
```

Slide 11: Real-Life Example: Customer Satisfaction Prediction

Let's apply ordinal regression to predict customer satisfaction levels (Low, Medium, High) based on various factors such as product quality and customer service rating.

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from mord import LogisticIT

# Generate sample data
np.random.seed(42)
n_samples = 1000

data = pd.DataFrame({
    'product_quality': np.random.randint(1, 6, n_samples),
    'customer_service': np.random.randint(1, 6, n_samples),
    'delivery_time': np.random.randint(1, 5, n_samples)
})

# Generate ordinal outcome
satisfaction = 0.5 * data['product_quality'] + 0.3 * data['customer_service'] + 0.2 * data['delivery_time'] + np.random.normal(0, 0.5, n_samples)
data['satisfaction'] = pd.cut(satisfaction, bins=3, labels=['Low', 'Medium', 'High'])

# Prepare data for modeling
X = data[['product_quality', 'customer_service', 'delivery_time']]
y = data['satisfaction'].cat.codes

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit model
model = LogisticIT()
model.fit(X_train, y_train)

# Evaluate model
train_accuracy = model.score(X_train, y_train)
test_accuracy = model.score(X_test, y_test)

print(f"Train Accuracy: {train_accuracy:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# Print coefficients
for feature, coef in zip(X.columns, model.coef_):
    print(f"{feature}: {coef:.4f}")
```

Slide 12: Real-Life Example: Education Level Prediction

In this example, we'll use ordinal regression to predict a person's education level (High School, Bachelor's, Master's, PhD) based on factors such as study hours, age, and work experience.

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from mord import LogisticAT
from sklearn.preprocessing import StandardScaler

# Generate sample data
np.random.seed(42)
n_samples = 1000

data = pd.DataFrame({
    'study_hours': np.random.randint(0, 80, n_samples),
    'age': np.random.randint(18, 60, n_samples),
    'work_experience': np.random.randint(0, 30, n_samples)
})

# Generate ordinal outcome
education_score = 0.4 * data['study_hours'] + 0.3 * data['age'] + 0.3 * data['work_experience'] + np.random.normal(0, 10, n_samples)
data['education_level'] = pd.cut(education_score, bins=4, labels=['High School', "Bachelor's", "Master's", 'PhD'])

# Prepare data for modeling
X = data[['study_hours', 'age', 'work_experience']]
y = data['education_level'].cat.codes

# Split and scale data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Fit model
model = LogisticAT()
model.fit(X_train_scaled, y_train)

# Evaluate model
train_accuracy = model.score(X_train_scaled, y_train)
test_accuracy = model.score(X_test_scaled, y_test)

print(f"Train Accuracy: {train_accuracy:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# Print coefficients
for feature, coef in zip(X.columns, model.coef_):
    print(f"{feature}: {coef:.4f}")

# Predict for a new observation
new_person = np.array([[50, 30, 5]])  # 50 study hours, 30 years old, 5 years work experience
new_person_scaled = scaler.transform(new_person)
predicted_level = model.predict(new_person_scaled)
education_levels = ['High School', "Bachelor's", "Master's", 'PhD']
print(f"Predicted Education Level: {education_levels[predicted_level[0]]}")
```

Slide 13: Challenges and Limitations of Ordinal Regression

While ordinal regression is a powerful tool for analyzing ordered categorical data, it comes with its own set of challenges and limitations. Understanding these is crucial for proper application and interpretation of results. The proportional odds assumption in many ordinal regression models may not always hold in real-world data. This assumption states that the effect of predictors is consistent across all levels of the outcome. Violation of this assumption can lead to biased results. Another challenge is the potential for overfitting, especially with small sample sizes or when using complex models. This can result in poor generalization to new data. Interpreting results can also be challenging, as the coefficients in ordinal regression models are not as straightforward to interpret as in linear regression.

Slide 14: Challenges and Limitations of Ordinal Regression

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Simulate data that violates the proportional odds assumption
np.random.seed(42)
x = np.linspace(0, 10, 1000)
y1 = stats.norm.cdf(x, loc=3, scale=1)
y2 = stats.norm.cdf(x, loc=5, scale=1.5)
y3 = stats.norm.cdf(x, loc=7, scale=2)

plt.figure(figsize=(10, 6))
plt.plot(x, y1, label='Category 1')
plt.plot(x, y2, label='Category 2')
plt.plot(x, y3, label='Category 3')
plt.title('Violation of Proportional Odds Assumption')
plt.xlabel('Predictor')
plt.ylabel('Cumulative Probability')
plt.legend()
plt.show()

# Example of overfitting
np.random.seed(42)
X = np.random.rand(20, 1)
y = np.random.randint(0, 3, 20)

from sklearn.preprocessing import PolynomialFeatures
from mord import LogisticIT

poly = PolynomialFeatures(degree=10)
X_poly = poly.fit_transform(X)

model = LogisticIT()
model.fit(X_poly, y)

X_test = np.linspace(0, 1, 100).reshape(-1, 1)
X_test_poly = poly.transform(X_test)
y_pred = model.predict(X_test_poly)

plt.figure(figsize=(10, 6))
plt.scatter(X, y, c='r', label='Training data')
plt.plot(X_test, y_pred, label='Prediction')
plt.title('Overfitting in Ordinal Regression')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()
```

Slide 15: Addressing Challenges in Ordinal Regression

To address the challenges in ordinal regression, we can employ various techniques. For the proportional odds assumption, we can use tests like the Brant test or visualize the assumption. To combat overfitting, regularization techniques or simpler models can be used. For interpretation, we can focus on odds ratios or predicted probabilities instead of raw coefficients.

Slide 16: Addressing Challenges in Ordinal Regression

```python
from sklearn.model_selection import train_test_split
from mord import LogisticIT
from sklearn.metrics import mean_absolute_error

# Generate sample data
np.random.seed(42)
X = np.random.rand(1000, 2)
y = np.dot(X, [1, 2]) + np.random.randn(1000)
y = pd.cut(y, bins=3, labels=[0, 1, 2])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit models with different regularization strengths
alphas = [0.001, 0.01, 0.1, 1, 10]
mae_scores = []

for alpha in alphas:
    model = LogisticIT(alpha=alpha)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mae_scores.append(mae)

# Plot regularization effect
plt.figure(figsize=(10, 6))
plt.plot(alphas, mae_scores, marker='o')
plt.xscale('log')
plt.title('Effect of Regularization on Model Performance')
plt.xlabel('Regularization Strength (alpha)')
plt.ylabel('Mean Absolute Error')
plt.show()

# Interpret results using odds ratios
best_model = LogisticIT(alpha=alphas[np.argmin(mae_scores)])
best_model.fit(X_train, y_train)

odds_ratios = np.exp(best_model.coef_)
for i, odds_ratio in enumerate(odds_ratios):
    print(f"Odds Ratio for feature {i+1}: {odds_ratio:.4f}")
```

Slide 17: Additional Resources

For those interested in delving deeper into ordinal regression and its implementation in Python, here are some valuable resources:

1. "Ordinal Regression Models in Psychology: A Tutorial" by Christensen, R. H. B. (2019). Available at: [https://arxiv.org/abs/1907.09ordinal](https://arxiv.org/abs/1907.09ordinal)
2. "Regression Models for Ordinal Data" by McCullagh, P. (1980). Journal of the Royal Statistical Society. Series B (Methodological), 42(2), 109-142.
3. "mord: Ordinal Regression in Python" documentation: [https://pythonhosted.org/mord/](https://pythonhosted.org/mord/)
4. "Ordinal Regression" chapter in "An Introduction to Statistical Learning" by James, G., Witten, D., Hastie, T., & Tibshirani, R.

These resources provide a mix of theoretical foundations and practical implementations to further your understanding of ordinal regression techniques and their applications in various fields.


