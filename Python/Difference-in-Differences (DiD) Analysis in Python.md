## Difference-in-Differences (DiD) Analysis in Python
Slide 1: Introduction to Difference-in-Differences (DiD)

Difference-in-Differences (DiD) is a statistical method used to estimate the causal effect of a treatment or intervention by comparing the changes in outcomes over time between a treatment group and a control group. This technique is particularly useful when randomized experiments are not feasible or ethical.

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate sample data
np.random.seed(42)
time = np.array([0, 1])
control = np.array([50, 52]) + np.random.normal(0, 2, 2)
treatment = np.array([50, 58]) + np.random.normal(0, 2, 2)

# Plot DiD
plt.figure(figsize=(10, 6))
plt.plot(time, control, 'b-', label='Control Group')
plt.plot(time, treatment, 'r-', label='Treatment Group')
plt.xlabel('Time')
plt.ylabel('Outcome')
plt.title('Difference-in-Differences Illustration')
plt.legend()
plt.xticks([0, 1], ['Before', 'After'])
plt.show()
```

Slide 2: Key Assumptions of DiD

The DiD method relies on several key assumptions to ensure valid causal inference. These include the parallel trends assumption, which states that in the absence of treatment, the difference between the treatment and control groups would remain constant over time. Additionally, we assume that the composition of the groups remains stable and that there are no spillover effects between groups.

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate data with parallel trends
np.random.seed(42)
time = np.arange(5)
control = 50 + 2 * time + np.random.normal(0, 1, 5)
treatment = 48 + 2 * time + np.random.normal(0, 1, 5)
treatment_effect = np.array([0, 0, 5, 5, 5])

plt.figure(figsize=(10, 6))
plt.plot(time, control, 'b-', label='Control Group')
plt.plot(time, treatment, 'r--', label='Treatment Group (Counterfactual)')
plt.plot(time, treatment + treatment_effect, 'r-', label='Treatment Group (Observed)')
plt.axvline(x=2, color='gray', linestyle='--', label='Intervention')
plt.xlabel('Time')
plt.ylabel('Outcome')
plt.title('Parallel Trends Assumption')
plt.legend()
plt.show()
```

Slide 3: Setting Up the DiD Model

To set up a DiD model, we need to organize our data into treatment and control groups, as well as pre- and post-intervention periods. We'll use a pandas DataFrame to structure our data and prepare it for analysis.

```python
import pandas as pd
import numpy as np

# Generate sample data
np.random.seed(42)
n = 1000
data = pd.DataFrame({
    'id': range(n),
    'treatment': np.random.choice([0, 1], n),
    'time': np.random.choice([0, 1], n),
    'outcome': np.random.normal(50, 10, n)
})

# Add treatment effect
data.loc[(data['treatment'] == 1) & (data['time'] == 1), 'outcome'] += 5

print(data.head(10))
print("\nData summary:")
print(data.groupby(['treatment', 'time'])['outcome'].mean())
```

Slide 4: Implementing DiD Using Ordinary Least Squares (OLS)

We can implement the DiD model using Ordinary Least Squares (OLS) regression. This approach allows us to estimate the treatment effect while controlling for group-specific and time-specific effects.

```python
import statsmodels.api as sm

# Prepare the data for regression
data['treatment_time'] = data['treatment'] * data['time']
X = sm.add_constant(data[['treatment', 'time', 'treatment_time']])
y = data['outcome']

# Fit the OLS model
model = sm.OLS(y, X).fit()

# Print the results
print(model.summary())

# Extract the DiD estimator (coefficient of treatment_time)
did_estimate = model.params['treatment_time']
print(f"\nEstimated treatment effect: {did_estimate:.4f}")
```

Slide 5: Visualizing DiD Results

Visualizing the results of a DiD analysis can help in understanding and communicating the findings. We'll create a plot that shows the mean outcomes for treatment and control groups before and after the intervention.

```python
import matplotlib.pyplot as plt

# Calculate mean outcomes for each group and time period
means = data.groupby(['treatment', 'time'])['outcome'].mean().unstack()

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot([0, 1], means.loc[0], 'b-o', label='Control Group')
plt.plot([0, 1], means.loc[1], 'r-o', label='Treatment Group')
plt.xlabel('Time')
plt.ylabel('Mean Outcome')
plt.title('Difference-in-Differences Results')
plt.legend()
plt.xticks([0, 1], ['Before', 'After'])
plt.show()

# Print the DiD estimate
did_manual = (means.loc[1, 1] - means.loc[1, 0]) - (means.loc[0, 1] - means.loc[0, 0])
print(f"Manual DiD estimate: {did_manual:.4f}")
```

Slide 6: Handling Time-Varying Covariates

In practice, we often need to control for time-varying covariates that might affect the outcome. We can incorporate these into our DiD model to improve the accuracy of our estimates.

```python
import pandas as pd
import numpy as np
import statsmodels.api as sm

# Generate sample data with a time-varying covariate
np.random.seed(42)
n = 1000
data = pd.DataFrame({
    'id': range(n),
    'treatment': np.random.choice([0, 1], n),
    'time': np.random.choice([0, 1], n),
    'covariate': np.random.normal(0, 1, n),
    'outcome': np.random.normal(50, 10, n)
})

# Add treatment effect and covariate effect
data.loc[(data['treatment'] == 1) & (data['time'] == 1), 'outcome'] += 5
data['outcome'] += 2 * data['covariate']

# Prepare the data for regression
data['treatment_time'] = data['treatment'] * data['time']
X = sm.add_constant(data[['treatment', 'time', 'treatment_time', 'covariate']])
y = data['outcome']

# Fit the OLS model
model = sm.OLS(y, X).fit()

# Print the results
print(model.summary())

# Extract the DiD estimator (coefficient of treatment_time)
did_estimate = model.params['treatment_time']
print(f"\nEstimated treatment effect: {did_estimate:.4f}")
```

Slide 7: Checking Parallel Trends Assumption

The parallel trends assumption is crucial for the validity of DiD. We can visually inspect this assumption by plotting pre-treatment trends for both groups.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Generate sample data with multiple pre-treatment periods
np.random.seed(42)
n = 1000
periods = 5
data = pd.DataFrame({
    'id': np.repeat(range(n), periods),
    'treatment': np.repeat(np.random.choice([0, 1], n), periods),
    'time': np.tile(range(periods), n),
    'outcome': np.random.normal(50, 10, n * periods)
})

# Add treatment effect in the last period for the treatment group
data.loc[(data['treatment'] == 1) & (data['time'] == periods-1), 'outcome'] += 5

# Calculate mean outcomes for each group and time period
means = data.groupby(['treatment', 'time'])['outcome'].mean().unstack()

# Plot pre-treatment trends
plt.figure(figsize=(10, 6))
plt.plot(range(periods-1), means.loc[0, :periods-1], 'b-o', label='Control Group')
plt.plot(range(periods-1), means.loc[1, :periods-1], 'r-o', label='Treatment Group')
plt.xlabel('Time')
plt.ylabel('Mean Outcome')
plt.title('Pre-treatment Trends')
plt.legend()
plt.show()

# Test for parallel trends
from statsmodels.formula.api import ols

pre_treatment = data[data['time'] < periods-1]
model = ols('outcome ~ treatment * time', data=pre_treatment).fit()
print(model.summary())
```

Slide 8: DiD with Multiple Time Periods

DiD can be extended to settings with multiple time periods, allowing for more robust estimation of treatment effects over time.

```python
import pandas as pd
import numpy as np
import statsmodels.api as sm

# Generate sample data with multiple time periods
np.random.seed(42)
n = 1000
periods = 5
data = pd.DataFrame({
    'id': np.repeat(range(n), periods),
    'treatment': np.repeat(np.random.choice([0, 1], n), periods),
    'time': np.tile(range(periods), n),
    'outcome': np.random.normal(50, 10, n * periods)
})

# Add treatment effect starting from period 3
data.loc[(data['treatment'] == 1) & (data['time'] >= 3), 'outcome'] += 5

# Create dummy variables for each time period
for t in range(1, periods):
    data[f'time_{t}'] = (data['time'] == t).astype(int)

# Create interaction terms
for t in range(1, periods):
    data[f'treat_time_{t}'] = data['treatment'] * data[f'time_{t}']

# Prepare the data for regression
X = sm.add_constant(data[['treatment'] + [f'time_{t}' for t in range(1, periods)] + 
                         [f'treat_time_{t}' for t in range(1, periods)]])
y = data['outcome']

# Fit the OLS model
model = sm.OLS(y, X).fit()

# Print the results
print(model.summary())

# Plot the treatment effects over time
effects = [model.params[f'treat_time_{t}'] for t in range(1, periods)]
plt.figure(figsize=(10, 6))
plt.plot(range(1, periods), effects, 'bo-')
plt.xlabel('Time Period')
plt.ylabel('Treatment Effect')
plt.title('Treatment Effects Over Time')
plt.axhline(y=0, color='r', linestyle='--')
plt.show()
```

Slide 9: Synthetic Control Method

The Synthetic Control Method is an extension of DiD that creates a synthetic control unit by weighting control units to match the pre-treatment characteristics of the treated unit.

```python
import pandas as pd
import numpy as np
from scipy.optimize import minimize

# Generate sample data
np.random.seed(42)
n_units = 20
n_periods = 10
data = pd.DataFrame({
    'unit': np.repeat(range(n_units), n_periods),
    'time': np.tile(range(n_periods), n_units),
    'outcome': np.random.normal(50, 10, n_units * n_periods)
})

# Add treatment effect for unit 0 in the last 5 periods
data.loc[(data['unit'] == 0) & (data['time'] >= 5), 'outcome'] += 10

# Separate treated and control units
treated = data[data['unit'] == 0]
control = data[data['unit'] != 0]

# Define the objective function to minimize
def objective(weights, control, treated, pre_treatment):
    synthetic = (control.pivot(index='time', columns='unit', values='outcome') * weights).sum(axis=1)
    return np.sum((treated.set_index('time')['outcome'] - synthetic)**2)

# Optimize weights
pre_treatment = data['time'] < 5
control_pre = control[pre_treatment].pivot(index='time', columns='unit', values='outcome')
treated_pre = treated[pre_treatment]

result = minimize(
    objective, 
    x0=np.ones(n_units-1) / (n_units-1),
    args=(control_pre, treated_pre, pre_treatment),
    method='SLSQP',
    constraints={'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
    bounds=[(0, 1) for _ in range(n_units-1)]
)

# Calculate synthetic control
synthetic = (control.pivot(index='time', columns='unit', values='outcome') * result.x).sum(axis=1)

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(treated.set_index('time')['outcome'], label='Treated Unit')
plt.plot(synthetic, label='Synthetic Control')
plt.axvline(x=4.5, color='r', linestyle='--', label='Treatment Start')
plt.xlabel('Time')
plt.ylabel('Outcome')
plt.title('Synthetic Control Method')
plt.legend()
plt.show()
```

Slide 10: Difference-in-Differences with Propensity Score Matching

Combining DiD with propensity score matching can help address potential selection bias in observational studies. This approach first matches treated and control units based on their propensity to receive treatment, then applies the DiD method to the matched sample.

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from scipy.stats import ttest_ind

# Generate sample data
np.random.seed(42)
n = 1000
data = pd.DataFrame({
    'age': np.random.normal(40, 10, n),
    'income': np.random.normal(50000, 10000, n),
    'education': np.random.choice(['low', 'medium', 'high'], n),
    'treatment': np.random.choice([0, 1], n),
    'time': np.random.choice([0, 1], n),
    'outcome': np.random.normal(50, 10, n)
})

# Add treatment effect
data.loc[(data['treatment'] == 1) & (data['time'] == 1), 'outcome'] += 5

# Estimate propensity scores
X = pd.get_dummies(data[['age', 'income', 'education']], drop_first=True)
y = data['treatment']
ps_model = LogisticRegression()
ps_model.fit(X, y)
data['propensity_score'] = ps_model.predict_proba(X)[:, 1]

# Perform matching (simplified nearest neighbor matching)
data['matched'] = False
for treated in data[data['treatment'] == 1].index:
    control = data[(data['treatment'] == 0) & (~data['matched'])]['propensity_score'].sub(data.loc[treated, 'propensity_score']).abs().idxmin()
    data.loc[[treated, control], 'matched'] = True

# Perform DiD on matched sample
matched_data = data[data['matched']]
did_model = sm.OLS.from_formula('outcome ~ treatment + time + treatment:time', data=matched_data).fit()

print(did_model.summary())
```

Slide 11: Event Study Design

An event study design extends the DiD framework by examining the treatment effect over multiple time periods before and after the intervention. This approach helps visualize pre-treatment trends and dynamic treatment effects.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Generate sample data
np.random.seed(42)
n_units = 100
n_periods = 10
treatment_start = 5

data = pd.DataFrame({
    'unit': np.repeat(range(n_units), n_periods),
    'time': np.tile(range(n_periods), n_units),
    'treatment': np.repeat(np.random.choice([0, 1], n_units), n_periods),
    'outcome': np.random.normal(50, 10, n_units * n_periods)
})

# Add treatment effect
data.loc[(data['treatment'] == 1) & (data['time'] >= treatment_start), 'outcome'] += 5

# Create relative time variable
data['rel_time'] = data['time'] - treatment_start
data['post'] = (data['rel_time'] >= 0).astype(int)

# Create dummy variables for each relative time period
for t in range(-treatment_start + 1, n_periods - treatment_start):
    if t != -1:  # Omit -1 as the reference category
        data[f'rel_time_{t}'] = (data['rel_time'] == t).astype(int)

# Estimate event study model
formula = 'outcome ~ ' + ' + '.join([f'rel_time_{t}' for t in range(-treatment_start + 1, n_periods - treatment_start) if t != -1])
model = sm.OLS.from_formula(formula, data=data[data['treatment'] == 1]).fit()

# Plot event study results
coef = model.params[1:]
ci = model.conf_int().iloc[1:]
plt.figure(figsize=(12, 6))
plt.plot(range(-treatment_start + 1, n_periods - treatment_start), coef, marker='o')
plt.fill_between(range(-treatment_start + 1, n_periods - treatment_start), ci[0], ci[1], alpha=0.2)
plt.axvline(x=0, color='r', linestyle='--')
plt.axhline(y=0, color='k', linestyle='-')
plt.xlabel('Relative Time')
plt.ylabel('Treatment Effect')
plt.title('Event Study Results')
plt.show()
```

Slide 12: Heterogeneous Treatment Effects

Exploring heterogeneous treatment effects allows us to understand how the impact of an intervention varies across different subgroups or characteristics of the population.

```python
import numpy as np
import pandas as pd
import statsmodels.api as sm

# Generate sample data
np.random.seed(42)
n = 1000
data = pd.DataFrame({
    'age': np.random.uniform(20, 60, n),
    'treatment': np.random.choice([0, 1], n),
    'time': np.random.choice([0, 1], n),
    'outcome': np.random.normal(50, 10, n)
})

# Add heterogeneous treatment effect
data['outcome'] += 5 * data['treatment'] * data['time']  # Base effect
data['outcome'] += 0.1 * data['age'] * data['treatment'] * data['time']  # Age interaction

# Estimate heterogeneous treatment effects
formula = 'outcome ~ treatment + time + treatment:time + age + age:treatment:time'
model = sm.OLS.from_formula(formula, data=data).fit()

print(model.summary())

# Calculate treatment effect at different ages
ages = np.linspace(20, 60, 5)
effects = model.params['treatment:time'] + model.params['age:treatment:time'] * ages

# Plot heterogeneous treatment effects
plt.figure(figsize=(10, 6))
plt.plot(ages, effects, marker='o')
plt.xlabel('Age')
plt.ylabel('Treatment Effect')
plt.title('Heterogeneous Treatment Effects by Age')
plt.show()
```

Slide 13: Difference-in-Differences with Staggered Adoption

In many real-world scenarios, treatments are adopted at different times by different units. This staggered adoption design requires special consideration in DiD analysis.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Generate sample data with staggered adoption
np.random.seed(42)
n_units = 50
n_periods = 10
data = pd.DataFrame({
    'unit': np.repeat(range(n_units), n_periods),
    'time': np.tile(range(n_periods), n_units),
    'adoption_time': np.repeat(np.random.choice(range(2, 8), n_units), n_periods),
    'outcome': np.random.normal(50, 10, n_units * n_periods)
})

# Add treatment effect
data['treated'] = (data['time'] >= data['adoption_time']).astype(int)
data.loc[data['treated'] == 1, 'outcome'] += 5

# Estimate staggered DiD model
formula = 'outcome ~ treated + C(unit) + C(time)'
model = sm.OLS.from_formula(formula, data=data).fit()

print(model.summary())

# Visualize staggered adoption
plt.figure(figsize=(12, 6))
for unit in range(n_units):
    unit_data = data[data['unit'] == unit]
    plt.plot(unit_data['time'], unit_data['outcome'], alpha=0.3)
    plt.axvline(x=unit_data['adoption_time'].iloc[0], color='r', linestyle='--', alpha=0.2)

plt.xlabel('Time')
plt.ylabel('Outcome')
plt.title('Staggered Adoption of Treatment')
plt.show()
```

Slide 14: Robustness Checks and Sensitivity Analysis

Conducting robustness checks and sensitivity analyses is crucial to ensure the validity and reliability of DiD estimates. These techniques help assess the stability of results under different assumptions or specifications.

```python
import numpy as np
import pandas as pd
import statsmodels.api as sm

# Generate sample data
np.random.seed(42)
n = 1000
data = pd.DataFrame({
    'treatment': np.random.choice([0, 1], n),
    'time': np.random.choice([0, 1], n),
    'outcome': np.random.normal(50, 10, n),
    'covariate': np.random.normal(0, 1, n)
})

# Add treatment effect
data.loc[(data['treatment'] == 1) & (data['time'] == 1), 'outcome'] += 5

# Base DiD model
base_model = sm.OLS.from_formula('outcome ~ treatment + time + treatment:time', data=data).fit()

# DiD model with covariate
covariate_model = sm.OLS.from_formula('outcome ~ treatment + time + treatment:time + covariate', data=data).fit()

# Placebo test (fake treatment)
data['fake_treatment'] = np.random.choice([0, 1], n)
placebo_model = sm.OLS.from_formula('outcome ~ fake_treatment + time + fake_treatment:time', data=data).fit()

# Print results
print("Base DiD Model:")
print(base_model.summary().tables[1])

print("\nDiD Model with Covariate:")
print(covariate_model.summary().tables[1])

print("\nPlacebo Test:")
print(placebo_model.summary().tables[1])
```

Slide 15: Additional Resources

For those interested in delving deeper into Difference-in-Differences methodology and its applications, here are some valuable resources:

1. Abadie, A. (2005). Semiparametric Difference-in-Differences Estimators. Review of Economic Studies, 72(1), 1-19. ArXiv: [https://arxiv.org/abs/2007.01124](https://arxiv.org/abs/2007.01124)
2. Athey, S., & Imbens, G. W. (2006). Identification and Inference in Nonlinear Difference-in-Differences Models. Econometrica, 74(2), 431-497. ArXiv: [https://arxiv.org/abs/1604.03544](https://arxiv.org/abs/1604.03544)
3. Goodman-Bacon, A. (2021). Difference-in-Differences with Variation in Treatment Timing. Journal of Econometrics, 225(2), 254-277. ArXiv: [https://arxiv.org/abs/1806.01221](https://arxiv.org/abs/1806.01221)

These papers provide advanced theoretical foundations and methodological innovations in DiD analysis. They offer insights into addressing various challenges and extensions of the basic DiD framework.

