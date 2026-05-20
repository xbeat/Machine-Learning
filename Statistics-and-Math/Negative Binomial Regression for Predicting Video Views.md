## Negative Binomial Regression for Predicting Video Views

Slide 1: Introduction to Negative Binomial Regression

Negative Binomial Regression is a statistical model used for count data with overdispersion. It's an extension of Poisson regression that allows for greater variance in the data. This model is particularly useful when dealing with count data where the variance exceeds the mean, a common occurrence in real-world scenarios.

```python
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Generate sample data
np.random.seed(0)
X = np.random.randn(100, 1)
y = np.random.negative_binomial(n=10, p=0.5, size=100)

# Fit Negative Binomial Regression model
model = sm.NegativeBinomial(y, X)
results = model.fit()

# Plot the results
plt.scatter(X, y)
plt.plot(X, results.predict(X), color='red')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Negative Binomial Regression')
plt.show()

print(results.summary())
```

Slide 2: The Negative Binomial Distribution

The Negative Binomial distribution is a discrete probability distribution that models the number of successes in a sequence of independent Bernoulli trials before a specified number of failures occur. It's characterized by two parameters: the number of failures until the experiment stops (r) and the probability of success in each trial (p).

```python
import numpy as np
import matplotlib.pyplot as plt

# Set parameters
r, p = 5, 0.5
x = np.arange(0, 20)

# Calculate PMF
pmf = nbinom.pmf(x, r, p)

# Plot PMF
plt.bar(x, pmf)
plt.xlabel('Number of successes')
plt.ylabel('Probability')
plt.title(f'Negative Binomial Distribution (r={r}, p={p})')
plt.show()
```

Slide 3: Overdispersion and Its Importance

Overdispersion occurs when the variance of the data is greater than what would be expected under a Poisson distribution. This is common in real-world count data and can lead to underestimated standard errors and overestimated significance if not accounted for. Negative Binomial Regression addresses this issue by introducing an additional parameter to model the extra variance.

```python
import matplotlib.pyplot as plt
from scipy.stats import poisson, nbinom

# Generate overdispersed data
np.random.seed(0)
lambda_param = 5
alpha = 0.5
size = 1000

poisson_data = np.random.poisson(lambda_param, size)
nb_data = np.random.negative_binomial(n=1/alpha, p=1/(1+alpha*lambda_param), size=size)

# Plot histograms
plt.figure(figsize=(12, 5))
plt.subplot(121)
plt.hist(poisson_data, bins=20, alpha=0.5, density=True)
plt.title('Poisson Distribution')
plt.subplot(122)
plt.hist(nb_data, bins=20, alpha=0.5, density=True)
plt.title('Negative Binomial Distribution')
plt.tight_layout()
plt.show()

print(f"Poisson variance: {np.var(poisson_data):.2f}")
print(f"Negative Binomial variance: {np.var(nb_data):.2f}")
```

Slide 4: Model Specification

In Negative Binomial Regression, we model the relationship between the predictor variables and the count outcome. The model assumes that the logarithm of the expected value of the outcome is a linear combination of the predictor variables. The dispersion parameter α accounts for the extra variance in the data.

```python
import numpy as np

# Generate sample data
np.random.seed(0)
X = np.column_stack((np.ones(100), np.random.randn(100, 2)))
y = np.random.negative_binomial(n=10, p=0.5, size=100)

# Fit Negative Binomial Regression model
model = sm.NegativeBinomial(y, X)
results = model.fit()

print(results.summary())

# Access model parameters
print("Coefficients:", results.params)
print("Dispersion parameter:", results.alpha)
```

Slide 5: Maximum Likelihood Estimation

Negative Binomial Regression parameters are typically estimated using Maximum Likelihood Estimation (MLE). This method finds the parameter values that maximize the likelihood of observing the given data. In practice, we often work with the log-likelihood for computational convenience.

```python
from scipy.optimize import minimize
from scipy.stats import nbinom

def neg_log_likelihood(params, X, y):
    beta, alpha = params[:-1], params[-1]
    mu = np.exp(X @ beta)
    return -np.sum(nbinom.logpmf(y, n=1/alpha, p=1/(1+alpha*mu)))

# Generate sample data
np.random.seed(0)
X = np.column_stack((np.ones(100), np.random.randn(100, 2)))
y = np.random.negative_binomial(n=10, p=0.5, size=100)

# Initial guess for parameters
initial_params = np.zeros(X.shape[1] + 1)

# Perform MLE
result = minimize(neg_log_likelihood, initial_params, args=(X, y), method='BFGS')

print("Estimated parameters:", result.x)
print("Log-likelihood:", -result.fun)
```

Slide 6: Interpreting Model Coefficients

In Negative Binomial Regression, the coefficients represent the change in the log of the expected count for a one-unit increase in the corresponding predictor variable, holding other variables constant. To interpret these in terms of the actual counts, we need to exponentiate the coefficients.

```python
import numpy as np
import pandas as pd

# Generate sample data
np.random.seed(0)
X = np.random.randn(100, 2)
y = np.random.negative_binomial(n=10, p=0.5, size=100)

# Fit Negative Binomial Regression model
model = sm.NegativeBinomial(y, sm.add_constant(X))
results = model.fit()

# Create a DataFrame for easier interpretation
coef_df = pd.DataFrame({
    'Coefficient': results.params,
    'Std Error': results.bse,
    'z-value': results.tvalues,
    'p-value': results.pvalues,
    'Exp(Coefficient)': np.exp(results.params)
})

print(coef_df)

# Interpret a specific coefficient
feature_index = 1
change = np.exp(results.params[feature_index]) - 1
print(f"A one-unit increase in X{feature_index} is associated with a {change:.2%} change in the expected count.")
```

Slide 7: Model Diagnostics

Model diagnostics are crucial for assessing the fit and assumptions of the Negative Binomial Regression. Common diagnostics include residual plots, Q-Q plots, and tests for overdispersion. These help identify potential issues such as non-linearity, heteroscedasticity, or influential observations.

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate sample data and fit model
np.random.seed(0)
X = sm.add_constant(np.random.randn(100, 2))
y = np.random.negative_binomial(n=10, p=0.5, size=100)
model = sm.NegativeBinomial(y, X).fit()

# Residual plot
plt.figure(figsize=(12, 4))
plt.subplot(131)
plt.scatter(model.predict(), model.resid_pearson)
plt.xlabel('Predicted values')
plt.ylabel('Pearson residuals')
plt.title('Residual Plot')

# Q-Q plot
plt.subplot(132)
sm.qqplot(model.resid_deviance, line='s')
plt.title('Q-Q Plot')

# Influence plot
plt.subplot(133)
sm.graphics.influence_plot(model)
plt.title('Influence Plot')

plt.tight_layout()
plt.show()

# Test for overdispersion
dispersion = model.pearson_chi2 / model.df_resid
print(f"Dispersion parameter: {dispersion:.2f}")
```

Slide 8: Akaike Information Criterion (AIC)

The Akaike Information Criterion (AIC) is a measure used for model selection, balancing model fit against complexity. Lower AIC values indicate better models. When comparing models, a difference in AIC of 2 or more is considered meaningful, with the lower AIC being preferred.

```python
import numpy as np

# Generate sample data
np.random.seed(0)
X = sm.add_constant(np.random.randn(100, 2))
y = np.random.negative_binomial(n=10, p=0.5, size=100)

# Fit models
poisson_model = sm.Poisson(y, X).fit()
nb_model = sm.NegativeBinomial(y, X).fit()

# Compare AIC
print("Poisson AIC:", poisson_model.aic)
print("Negative Binomial AIC:", nb_model.aic)

# Calculate AIC difference
aic_diff = poisson_model.aic - nb_model.aic
print(f"AIC difference: {aic_diff:.2f}")

if aic_diff > 2:
    print("The Negative Binomial model is preferred.")
elif aic_diff < -2:
    print("The Poisson model is preferred.")
else:
    print("There's no strong preference between the models.")
```

Slide 9: Predicting Views Based on First 2 Hours

To predict views based on the first 2 hours using Negative Binomial Regression, we'll create a model that uses early engagement metrics as predictors. This approach can be particularly useful for content creators and marketers to estimate the potential popularity of their content.

```python
import statsmodels.api as sm
import numpy as np

# Generate sample data
np.random.seed(0)
data = pd.DataFrame({
    'views_2h': np.random.poisson(lam=100, size=1000),
    'likes_2h': np.random.poisson(lam=10, size=1000),
    'comments_2h': np.random.poisson(lam=5, size=1000),
    'total_views': np.random.negative_binomial(n=5, p=0.1, size=1000)
})

# Prepare features and target
X = sm.add_constant(data[['views_2h', 'likes_2h', 'comments_2h']])
y = data['total_views']

# Fit Negative Binomial Regression model
model = sm.NegativeBinomial(y, X).fit()

# Print summary
print(model.summary())

# Make predictions for new data
new_data = pd.DataFrame({
    'views_2h': [150, 200],
    'likes_2h': [15, 20],
    'comments_2h': [8, 10]
})
X_new = sm.add_constant(new_data)
predictions = model.predict(X_new)

print("\nPredictions:")
print(pd.DataFrame({'Input': new_data.to_dict('records'), 'Predicted Views': predictions.round().astype(int)}))
```

Slide 10: Cross-Validation for Model Evaluation

Cross-validation helps assess how well our model generalizes to unseen data. We'll use k-fold cross-validation to evaluate our Negative Binomial Regression model for predicting views based on early engagement metrics.

```python
from sklearn.metrics import mean_squared_error
import numpy as np
import statsmodels.api as sm

# Generate sample data
np.random.seed(0)
X = np.column_stack((
    np.random.poisson(lam=100, size=1000),  # views_2h
    np.random.poisson(lam=10, size=1000),   # likes_2h
    np.random.poisson(lam=5, size=1000)     # comments_2h
))
y = np.random.negative_binomial(n=5, p=0.1, size=1000)  # total_views

# Perform k-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
mse_scores = []

for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    model = sm.NegativeBinomial(y_train, sm.add_constant(X_train)).fit()
    y_pred = model.predict(sm.add_constant(X_test))
    
    mse = mean_squared_error(y_test, y_pred)
    mse_scores.append(mse)

print(f"Mean MSE: {np.mean(mse_scores):.2f}")
print(f"Standard deviation of MSE: {np.std(mse_scores):.2f}")
```

Slide 11: Feature Importance and Selection

Understanding which features are most important for predicting views can help content creators focus on the most impactful factors. We'll use a simple feature importance technique based on the absolute values of the standardized coefficients.

```python
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt

# Generate sample data
np.random.seed(0)
data = pd.DataFrame({
    'views_2h': np.random.poisson(lam=100, size=1000),
    'likes_2h': np.random.poisson(lam=10, size=1000),
    'comments_2h': np.random.poisson(lam=5, size=1000),
    'shares_2h': np.random.poisson(lam=2, size=1000),
    'total_views': np.random.negative_binomial(n=5, p=0.1, size=1000)
})

# Prepare features and target
X = data[['views_2h', 'likes_2h', 'comments_2h', 'shares_2h']]
y = data['total_views']

# Fit Negative Binomial Regression model
model = sm.NegativeBinomial(y, sm.add_constant(X)).fit()

# Calculate feature importance
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': np.abs(model.params[1:] * X.std())
})
feature_importance = feature_importance.sort_values('Importance', ascending=False)

# Plot feature importance
plt.figure(figsize=(10, 6))
plt.bar(feature_importance['Feature'], feature_importance['Importance'])
plt.title('Feature Importance')
plt.xlabel('Features')
plt.ylabel('Importance')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print(feature_importance)
```

Slide 12: Handling Zero-Inflation

Zero-inflated count data occurs when there's an excess of zeros beyond what's expected from a Negative Binomial distribution. This scenario can be addressed using Zero-Inflated Negative Binomial Regression, which combines a binary model for predicting zeros with a Negative Binomial model for the count data.

```python
import statsmodels.api as sm

# Generate zero-inflated data
np.random.seed(0)
n_samples = 1000
X = np.random.randn(n_samples, 2)
zero_prob = 0.3
count_data = np.random.negative_binomial(n=5, p=0.5, size=n_samples)
zero_mask = np.random.binomial(1, zero_prob, size=n_samples)
y = count_data * (1 - zero_mask)

# Fit Zero-Inflated Negative Binomial model
zinb_model = sm.ZeroInflatedNegativeBinomialP(y, X).fit()

print(zinb_model.summary())

# Predict
X_new = np.array([[0.5, 1.0], [-0.5, -1.0]])
predictions = zinb_model.predict(X_new)
print("Predictions:", predictions)
```

Slide 13: Real-Life Example: Predicting Video Views

Let's apply Negative Binomial Regression to predict the total views of online videos based on their performance in the first two hours. This example is relevant for content creators and marketers who want to estimate a video's potential reach.

```python
import pandas as pd
import statsmodels.api as sm

# Simulate video data
np.random.seed(42)
n_videos = 1000

data = pd.DataFrame({
    'views_2h': np.random.poisson(100, n_videos),
    'likes_2h': np.random.poisson(10, n_videos),
    'comments_2h': np.random.poisson(5, n_videos),
    'shares_2h': np.random.poisson(2, n_videos)
})

# Generate total views with some randomness
data['total_views'] = np.random.negative_binomial(
    n=data['views_2h'] * 0.1,
    p=0.1,
    size=n_videos
)

# Fit the model
X = sm.add_constant(data[['views_2h', 'likes_2h', 'comments_2h', 'shares_2h']])
y = data['total_views']
model = sm.NegativeBinomial(y, X).fit()

print(model.summary())

# Predict for a new video
new_video = pd.DataFrame({
    'views_2h': [500],
    'likes_2h': [50],
    'comments_2h': [20],
    'shares_2h': [10]
})
X_new = sm.add_constant(new_video)
prediction = model.predict(X_new)

print(f"Predicted total views: {prediction[0]:.0f}")
```

Slide 14: Real-Life Example: Predicting Insect Counts

Ecologists often use Negative Binomial Regression to model the abundance of insects in different habitats. This example demonstrates how to predict the number of insects based on environmental factors.

```python
import pandas as pd
import statsmodels.api as sm

# Simulate insect count data
np.random.seed(42)
n_sites = 500

data = pd.DataFrame({
    'temperature': np.random.normal(25, 5, n_sites),
    'humidity': np.random.normal(60, 10, n_sites),
    'vegetation_density': np.random.uniform(0, 1, n_sites)
})

# Generate insect counts
lambda_ = np.exp(2 + 0.1 * data['temperature'] + 
                 0.05 * data['humidity'] + 
                 1.5 * data['vegetation_density'])
data['insect_count'] = np.random.negative_binomial(n=5, p=5/(5+lambda_))

# Fit the model
X = sm.add_constant(data[['temperature', 'humidity', 'vegetation_density']])
y = data['insect_count']
model = sm.NegativeBinomial(y, X).fit()

print(model.summary())

# Predict for a new site
new_site = pd.DataFrame({
    'temperature': [28],
    'humidity': [65],
    'vegetation_density': [0.7]
})
X_new = sm.add_constant(new_site)
prediction = model.predict(X_new)

print(f"Predicted insect count: {prediction[0]:.0f}")
```

Slide 15: Additional Resources

For those interested in diving deeper into Negative Binomial Regression and AIC model fit, here are some valuable resources:

1. "Regression Models for Count Data in R" by Zeileis et al. (2008) ArXiv URL: [https://arxiv.org/abs/0804.2899](https://arxiv.org/abs/0804.2899)
2. "Model Selection and Multimodel Inference: A Practical Information-Theoretic Approach" by Burnham and Anderson (2002) (This is a book, not available on ArXiv)
3. "Negative Binomial Regression" by Hilbe (2011) (This is a book, not available on ArXiv)

These resources provide in-depth discussions on the theoretical foundations and practical applications of Negative Binomial Regression and model selection using AIC.


