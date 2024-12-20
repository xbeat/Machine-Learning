## Linear Mixed Models in Python
Slide 1: Introduction to Linear Mixed Models

Linear Mixed Models (LMM) are an extension of linear regression that allow for both fixed and random effects. They're particularly useful for analyzing hierarchical or grouped data, such as repeated measures or longitudinal studies.

```python
import numpy as np
import pandas as pd
from statsmodels.formula.api import mixedlm

# Example data
data = pd.DataFrame({
    'subject': np.repeat(['A', 'B', 'C', 'D'], 5),
    'time': np.tile(range(5), 4),
    'treatment': np.repeat(['control', 'treatment'], 10),
    'response': np.random.normal(0, 1, 20)
})

# Fit a basic LMM
model = mixedlm("response ~ time + treatment", data, groups="subject")
results = model.fit()
print(results.summary())
```

Slide 2: Fixed vs. Random Effects

Fixed effects are parameters associated with an entire population or certain repeatable levels of factors. Random effects are associated with individual experimental units drawn at random from a population.

```python
# Fixed effects model
fixed_model = mixedlm("response ~ time + treatment", data, groups="subject")
fixed_results = fixed_model.fit()

# Random effects model (random intercept for subject)
random_model = mixedlm("response ~ time + treatment", data, groups="subject", 
                       re_formula="~1")
random_results = random_model.fit()

print("Fixed effects AIC:", fixed_results.aic)
print("Random effects AIC:", random_results.aic)
```

Slide 3: Model Structure

The general form of a linear mixed model is: y = Xβ + Zu + ε Where y is the response variable, X is the fixed effects design matrix, β are the fixed effects parameters, Z is the random effects design matrix, u are the random effects, and ε is the error term.

```python
import statsmodels.api as sm

# Manually construct design matrices
X = sm.add_constant(data[['time', 'treatment']])
Z = pd.get_dummies(data['subject'])

# Fit the model
model = sm.MixedLM(data['response'], X, groups=data['subject'], exog_re=Z)
results = model.fit()
print(results.summary())
```

Slide 4: Variance Components

In LMMs, the total variance is partitioned into components: variance due to random effects and residual variance. This partitioning allows for more accurate estimation of fixed effects and their standard errors.

```python
# Fit a model with random intercept and slope
model = mixedlm("response ~ time + treatment", data, groups="subject", 
                re_formula="~time")
results = model.fit()

# Extract variance components
var_intercept = results.cov_re.iloc[0, 0]
var_slope = results.cov_re.iloc[1, 1]
var_residual = results.scale

print(f"Variance of random intercept: {var_intercept}")
print(f"Variance of random slope: {var_slope}")
print(f"Residual variance: {var_residual}")
```

Slide 5: Model Diagnostics

Assessing model fit is crucial in LMMs. Common diagnostics include residual plots, Q-Q plots, and information criteria like AIC or BIC.

```python
import matplotlib.pyplot as plt
from scipy import stats

# Residual plot
plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.scatter(results.fittedvalues, results.resid)
plt.xlabel("Fitted values")
plt.ylabel("Residuals")
plt.title("Residual Plot")

# Q-Q plot
plt.subplot(122)
stats.probplot(results.resid, dist="norm", plot=plt)
plt.title("Q-Q Plot")

plt.tight_layout()
plt.show()

print(f"AIC: {results.aic}")
print(f"BIC: {results.bic}")
```

Slide 6: Random Intercepts vs. Random Slopes

LMMs can include random intercepts, random slopes, or both. Random intercepts allow the baseline level to vary across groups, while random slopes allow the effect of predictors to vary.

```python
# Random intercept model
model_intercept = mixedlm("response ~ time + treatment", data, groups="subject", 
                          re_formula="~1")
results_intercept = model_intercept.fit()

# Random slope model
model_slope = mixedlm("response ~ time + treatment", data, groups="subject", 
                      re_formula="~time")
results_slope = model_slope.fit()

print("Random Intercept AIC:", results_intercept.aic)
print("Random Slope AIC:", results_slope.aic)
```

Slide 7: Crossed vs. Nested Random Effects

LMMs can handle various data structures, including crossed random effects (e.g., items and subjects in a psycholinguistic experiment) and nested random effects (e.g., students nested within schools).

```python
# Simulating data with crossed random effects
np.random.seed(42)
n_subjects = 50
n_items = 30
n_obs = n_subjects * n_items

data = pd.DataFrame({
    'subject': np.repeat(range(n_subjects), n_items),
    'item': np.tile(range(n_items), n_subjects),
    'condition': np.random.choice(['A', 'B'], n_obs),
    'response': np.random.normal(0, 1, n_obs)
})

# Fit model with crossed random effects
model = mixedlm("response ~ condition", data, groups="subject", 
                re_formula="~1|item")
results = model.fit()
print(results.summary())
```

Slide 8: Handling Missing Data

LMMs are robust to missing data under certain conditions (Missing At Random). They can provide unbiased estimates even with unbalanced designs or dropout in longitudinal studies.

```python
# Introduce some missing data
data.loc[np.random.choice(data.index, 100), 'response'] = np.nan

# Fit model with missing data
model = mixedlm("response ~ condition", data, groups="subject", 
                re_formula="~1|item")
results = model.fit()
print(results.summary())

# Compare number of observations
print(f"Total observations: {len(data)}")
print(f"Observations used in model: {results.nobs}")
```

Slide 9: Interaction Effects

LMMs can incorporate interaction effects between fixed factors, between random factors, or between fixed and random factors.

```python
# Add an interaction term
data['time_treatment'] = data['time'] * (data['treatment'] == 'treatment')

# Fit model with interaction
model = mixedlm("response ~ time + treatment + time_treatment", data, 
                groups="subject", re_formula="~1")
results = model.fit()
print(results.summary())
```

Slide 10: Model Comparison

Comparing different models is a crucial step in LMM analysis. We can use likelihood ratio tests for nested models or information criteria for non-nested models.

```python
from scipy.stats import chi2

# Fit two nested models
model1 = mixedlm("response ~ time + treatment", data, groups="subject", 
                 re_formula="~1")
results1 = model1.fit()

model2 = mixedlm("response ~ time + treatment + time_treatment", data, 
                 groups="subject", re_formula="~1")
results2 = model2.fit()

# Likelihood ratio test
LR_statistic = -2 * (results1.llf - results2.llf)
df = results2.df_modelwc - results1.df_modelwc
p_value = chi2.sf(LR_statistic, df)

print(f"LR statistic: {LR_statistic}")
print(f"p-value: {p_value}")
```

Slide 11: Power Analysis for LMMs

Power analysis for LMMs is complex due to the multiple sources of variance. Simulation-based approaches are often used to estimate power for different sample sizes and effect sizes.

```python
import numpy as np
from scipy import stats

def simulate_lmm_data(n_subjects, n_obs_per_subject, fixed_effect, random_effect_sd):
    subjects = np.repeat(range(n_subjects), n_obs_per_subject)
    X = np.random.normal(0, 1, n_subjects * n_obs_per_subject)
    random_effects = np.random.normal(0, random_effect_sd, n_subjects)
    y = fixed_effect * X + np.repeat(random_effects, n_obs_per_subject) + np.random.normal(0, 1, n_subjects * n_obs_per_subject)
    return pd.DataFrame({'subject': subjects, 'X': X, 'y': y})

def run_power_analysis(n_subjects, n_obs_per_subject, fixed_effect, random_effect_sd, n_simulations=1000):
    significant_results = 0
    for _ in range(n_simulations):
        data = simulate_lmm_data(n_subjects, n_obs_per_subject, fixed_effect, random_effect_sd)
        model = mixedlm("y ~ X", data, groups="subject")
        results = model.fit()
        if results.pvalues['X'] < 0.05:
            significant_results += 1
    return significant_results / n_simulations

power = run_power_analysis(n_subjects=20, n_obs_per_subject=5, fixed_effect=0.3, random_effect_sd=0.5)
print(f"Estimated power: {power}")
```

Slide 12: Assumptions and Diagnostics

LMMs have several assumptions, including linearity, normality of residuals, and homoscedasticity. Checking these assumptions is crucial for valid inference.

```python
import seaborn as sns

# Fit a model
model = mixedlm("response ~ time + treatment", data, groups="subject")
results = model.fit()

# Residual vs. fitted plot
plt.figure(figsize=(10, 5))
plt.subplot(121)
sns.scatterplot(x=results.fittedvalues, y=results.resid)
plt.xlabel("Fitted values")
plt.ylabel("Residuals")
plt.title("Residuals vs. Fitted")

# Q-Q plot
plt.subplot(122)
stats.probplot(results.resid, dist="norm", plot=plt)
plt.title("Q-Q Plot")

plt.tight_layout()
plt.show()

# Shapiro-Wilk test for normality
_, p_value = stats.shapiro(results.resid)
print(f"Shapiro-Wilk test p-value: {p_value}")
```

Slide 13: Interpreting LMM Results

Interpreting LMM results requires understanding both fixed and random effects estimates, as well as their standard errors and significance.

```python
# Fit a model
model = mixedlm("response ~ time + treatment", data, groups="subject", 
                re_formula="~time")
results = model.fit()

# Print summary
print(results.summary())

# Extract and interpret key results
fixed_effects = results.fe_params
random_effects = results.random_effects
print("\nFixed Effects:")
print(fixed_effects)
print("\nRandom Effects (first 3 subjects):")
print(random_effects[:3])

# Calculate confidence intervals
conf_int = results.conf_int()
print("\nConfidence Intervals:")
print(conf_int)
```

Slide 14: Additional Resources

For more in-depth understanding of Linear Mixed Models, consider exploring these resources:

1. "Linear Mixed-Effects Models using R: A Step-by-Step Approach" by Andrzej Gałecki and Tomasz Burzykowski ArXiv link: [https://arxiv.org/abs/1406.5823](https://arxiv.org/abs/1406.5823)
2. "Mixed-Effects Models in S and S-PLUS" by José Pinheiro and Douglas Bates Reference: Pinheiro, J. C., & Bates, D. M. (2000). Mixed-effects models in S and S-PLUS. Springer, New York.
3. "Data Analysis Using Regression and Multilevel/Hierarchical Models" by Andrew Gelman and Jennifer Hill Reference: Gelman, A., & Hill, J. (2006). Data analysis using regression and multilevel/hierarchical models. Cambridge University Press.

