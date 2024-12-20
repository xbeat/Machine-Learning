## Extending ANOVA Beyond the Basics
Slide 1: Understanding AN\[C\]OVA and Beyond

AN\[C\]OVA (Analysis of \[Co\]Variance) is commonly used to assess main and interaction effects in the general linear model. However, this concept can be extended to other statistical models, providing a broader perspective on group comparisons and effect analysis.

```python
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols

# Generate sample data
np.random.seed(0)
group = np.repeat(['A', 'B', 'C'], 50)
covariate = np.random.normal(0, 1, 150)
outcome = 2 * (group == 'A') + 3 * (group == 'B') + covariate + np.random.normal(0, 1, 150)

# Perform ANCOVA
model = ols('outcome ~ C(group) + covariate', data={'outcome': outcome, 'group': group, 'covariate': covariate}).fit()
print(model.summary())
```

Slide 2: General Linear Model in AN\[C\]OVA

The general linear model underpins classic AN\[C\]OVA, assuming a linear relationship between predictors and the outcome. It allows for the assessment of group differences while controlling for covariates.

```python
import matplotlib.pyplot as plt

# Visualize the data and fitted lines
plt.figure(figsize=(10, 6))
for g in ['A', 'B', 'C']:
    mask = group == g
    plt.scatter(covariate[mask], outcome[mask], label=f'Group {g}')
    plt.plot(covariate[mask], model.fittedvalues[mask], linewidth=2)

plt.xlabel('Covariate')
plt.ylabel('Outcome')
plt.legend()
plt.title('ANCOVA: Group Differences with Covariate')
plt.show()
```

Slide 3: Beyond the General Linear Model

The concept of assessing main and interaction effects can be extended to other statistical models. This allows for more flexible analysis, particularly when data doesn't meet the assumptions of the general linear model.

```python
from statsmodels.formula.api import glm
import statsmodels.api as sm

# Generate binary outcome data
binary_outcome = (outcome > np.median(outcome)).astype(int)

# Fit logistic regression model
logistic_model = glm('binary_outcome ~ C(group) + covariate', 
                     data={'binary_outcome': binary_outcome, 'group': group, 'covariate': covariate}, 
                     family=sm.families.Binomial()).fit()
print(logistic_model.summary())
```

Slide 4: Quantile Regression: Beyond Means

Quantile regression allows us to examine group differences at various quantiles of the outcome distribution, not just the mean. This is particularly useful when the relationship between predictors and the outcome varies across the distribution.

```python
from statsmodels.formula.api import quantreg

# Fit quantile regression model (median)
quantile_model = quantreg('outcome ~ C(group) + covariate', 
                          data={'outcome': outcome, 'group': group, 'covariate': covariate}).fit(q=0.5)
print(quantile_model.summary())
```

Slide 5: Interpreting Quantile Regression Results

Quantile regression provides group quantiles (e.g., medians) instead of means. This allows for a more comprehensive understanding of group differences across the entire distribution of the outcome variable.

```python
# Compare OLS and Quantile Regression coefficients
ols_coef = model.params
quant_coef = quantile_model.params

print("OLS vs Quantile Regression Coefficients:")
for param in ols_coef.index:
    print(f"{param}: OLS = {ols_coef[param]:.3f}, Quantile = {quant_coef[param]:.3f}")
```

Slide 6: Generalized Linear Models (GLM)

GLMs extend the linear model to handle non-normal outcomes through link functions. They maintain the concept of assessing equality of conditional expectations, but the interpretation depends on the conditional distribution and the link function used.

```python
# Poisson regression example
count_outcome = np.random.poisson(np.exp(outcome / 10))
poisson_model = glm('count_outcome ~ C(group) + covariate', 
                    data={'count_outcome': count_outcome, 'group': group, 'covariate': covariate}, 
                    family=sm.families.Poisson()).fit()
print(poisson_model.summary())
```

Slide 7: Natural vs. Response Scale in GLMs

In GLMs, we can perform analysis on the natural scale (linear predictor) or the response scale. This choice affects interpretation and has consequences when using Wald's test for hypothesis testing.

```python
# Calculate effects on both scales for logistic regression
log_odds = logistic_model.params
odds_ratios = np.exp(log_odds)
probabilities = odds_ratios / (1 + odds_ratios)

print("Effects on different scales:")
for param, lo, or_, prob in zip(log_odds.index, log_odds, odds_ratios, probabilities):
    print(f"{param}: Log-odds = {lo:.3f}, Odds Ratio = {or_:.3f}, Probability = {prob:.3f}")
```

Slide 8: Logistic Regression for Proportion Comparisons

Logistic regression is commonly used in clinical trials with binary endpoints to test hypotheses about the equality of proportions across groups.

```python
# Simulate clinical trial data
np.random.seed(42)
treatment = np.random.choice(['Control', 'Treatment A', 'Treatment B'], size=300)
age = np.random.normal(50, 10, 300)
success = (0.3 + 0.2 * (treatment == 'Treatment A') + 0.4 * (treatment == 'Treatment B') + 0.01 * (age - 50) + 
           np.random.normal(0, 0.1, 300)) > 0.5

# Fit logistic regression
clinical_model = glm('success ~ C(treatment) + age', 
                     data={'success': success, 'treatment': treatment, 'age': age}, 
                     family=sm.families.Binomial()).fit()
print(clinical_model.summary())
```

Slide 9: Interpreting Logistic Regression Results

Logistic regression results can be interpreted in terms of log-odds, odds ratios, or probabilities. Each scale provides a different perspective on the treatment effects.

```python
# Calculate and print odds ratios
odds_ratios = np.exp(clinical_model.params)
conf_int = np.exp(clinical_model.conf_int())

print("Odds Ratios and 95% Confidence Intervals:")
for param, or_, ci in zip(odds_ratios.index, odds_ratios, conf_int.values):
    print(f"{param}: OR = {or_:.3f} (95% CI: {ci[0]:.3f} - {ci[1]:.3f})")
```

Slide 10: Real-life Example: Education Study

Consider a study examining the effect of teaching methods on student performance, controlling for prior academic achievement.

```python
# Simulate education study data
np.random.seed(123)
teaching_method = np.random.choice(['Traditional', 'Interactive', 'Online'], size=200)
prior_achievement = np.random.normal(70, 10, 200)
performance = (75 + 5 * (teaching_method == 'Interactive') + 3 * (teaching_method == 'Online') + 
               0.5 * (prior_achievement - 70) + np.random.normal(0, 5, 200))

# Perform ANCOVA
edu_model = ols('performance ~ C(teaching_method) + prior_achievement', 
                data={'performance': performance, 'teaching_method': teaching_method, 
                      'prior_achievement': prior_achievement}).fit()
print(edu_model.summary())
```

Slide 11: Visualizing Education Study Results

Visualizing the results helps in understanding the effects of different teaching methods while accounting for prior achievement.

```python
import seaborn as sns

plt.figure(figsize=(10, 6))
sns.scatterplot(x=prior_achievement, y=performance, hue=teaching_method)
for method in ['Traditional', 'Interactive', 'Online']:
    mask = teaching_method == method
    sns.regplot(x=prior_achievement[mask], y=performance[mask], scatter=False, label=f'{method} (fitted)')
plt.xlabel('Prior Achievement')
plt.ylabel('Performance')
plt.title('Effect of Teaching Methods on Performance')
plt.legend()
plt.show()
```

Slide 12: Real-life Example: Environmental Study

An environmental study examining the impact of pollution levels on plant growth across different soil types.

```python
# Simulate environmental study data
np.random.seed(456)
soil_type = np.random.choice(['Sandy', 'Clay', 'Loam'], size=150)
pollution_level = np.random.uniform(0, 100, 150)
plant_growth = (20 - 0.1 * pollution_level + 5 * (soil_type == 'Clay') + 3 * (soil_type == 'Loam') + 
                np.random.normal(0, 2, 150))

# Perform ANCOVA
env_model = ols('plant_growth ~ C(soil_type) + pollution_level', 
                data={'plant_growth': plant_growth, 'soil_type': soil_type, 
                      'pollution_level': pollution_level}).fit()
print(env_model.summary())
```

Slide 13: Visualizing Environmental Study Results

Visualizing the environmental study results helps in understanding the complex interactions between soil types and pollution levels on plant growth.

```python
plt.figure(figsize=(10, 6))
sns.scatterplot(x=pollution_level, y=plant_growth, hue=soil_type)
for soil in ['Sandy', 'Clay', 'Loam']:
    mask = soil_type == soil
    sns.regplot(x=pollution_level[mask], y=plant_growth[mask], scatter=False, label=f'{soil} (fitted)')
plt.xlabel('Pollution Level')
plt.ylabel('Plant Growth')
plt.title('Effect of Soil Type and Pollution on Plant Growth')
plt.legend()
plt.show()
```

Slide 14: Additional Resources

For further exploration of advanced statistical models and their applications:

1. "Extending the Linear Model with R" by Julian J. Faraway ArXiv: [https://arxiv.org/abs/math/0409046](https://arxiv.org/abs/math/0409046)
2. "Quantile Regression" by Roger Koenker ArXiv: [https://arxiv.org/abs/2106.08597](https://arxiv.org/abs/2106.08597)
3. "Generalized Linear Models" by P. McCullagh and J.A. Nelder (Classic textbook, not available on ArXiv)

These resources provide in-depth coverage of the topics discussed in this presentation, offering more advanced techniques and theoretical foundations.

