## Causal Inference in Machine Learning in Python
Slide 1: Introduction to Causal Inference

Causal inference is the process of determining the cause-and-effect relationships between variables in a dataset. It plays a crucial role in many fields, including machine learning and data science, as it helps to understand the underlying mechanisms driving observed data patterns and make accurate predictions about the effects of interventions or policy changes.

Code:

```python
# No code for the introduction slide
```

Slide 2: Correlation vs. Causation

Correlation between variables does not necessarily imply causation. Causal inference aims to move beyond mere correlations by establishing causal relationships, which are essential for making reliable predictions and decisions.

Code:

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate random data
x = np.random.rand(100)
y = 2 * x + np.random.randn(100)

# Plot the data
plt.scatter(x, y)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Correlation between X and Y')
plt.show()
```

Slide 3: Potential Outcomes Framework

The potential outcomes framework, also known as the Rubin Causal Model, is a fundamental concept in causal inference. It defines the potential outcomes for each individual under different treatment conditions and forms the basis for many causal inference methods.

Code:

```python
import numpy as np

def potential_outcomes(treatment, outcome, treatment_effect):
    potential_outcomes = []
    for t in [0, 1]:
        potential_outcomes.append(outcome + t * treatment_effect)
    return potential_outcomes

# Example
treatment = np.array([0, 1, 0, 1])
outcome = np.array([10, 12, 8, 15])
treatment_effect = 3

potential_outcomes_list = [potential_outcomes(t, y, treatment_effect) for t, y in zip(treatment, outcome)]
print(potential_outcomes_list)
```

Slide 4: Randomized Controlled Trials (RCTs)

Randomized controlled trials (RCTs) are considered the gold standard for establishing causal relationships. They involve randomly assigning individuals to treatment and control groups, ensuring that any observed differences in outcomes can be attributed to the treatment itself.

Code:

```python
import numpy as np
from scipy.stats import ttest_ind

def run_rct(treatment, control):
    t_stat, p_value = ttest_ind(treatment, control)
    return p_value

# Example
treatment_group = np.random.normal(loc=5, scale=2, size=100)
control_group = np.random.normal(loc=3, scale=2, size=100)

p_value = run_rct(treatment_group, control_group)
print(f'P-value: {p_value:.4f}')
```

Slide 5: Confounding Variables

Confounding variables are factors that influence both the treatment and the outcome, leading to spurious associations or biased estimates of causal effects. Proper handling of confounding variables is crucial in causal inference.

Code:

```python
import numpy as np
import pandas as pd

# Generate synthetic data
np.random.seed(42)
n = 1000
age = np.random.uniform(20, 80, n)
sex = np.random.binomial(1, 0.5, n)
treatment = np.random.binomial(1, 0.5, n)
outcome = 2 * treatment + 0.5 * age + 0.2 * sex + np.random.normal(0, 1, n)

data = pd.DataFrame({'age': age, 'sex': sex, 'treatment': treatment, 'outcome': outcome})

# Analyze the data
print(data.corr())
```

Slide 6: Propensity Score Matching

Propensity score matching is a technique used to estimate causal effects in observational studies. It involves matching treated and control units based on their propensity scores (the probability of receiving treatment given the observed covariates) to mimic a randomized experiment.

Code:

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors

def propensity_score_matching(X, treatment, outcome):
    model = LogisticRegression().fit(X, treatment)
    propensity_scores = model.predict_proba(X)[:, 1]
    
    treated = X[treatment == 1]
    control = X[treatment == 0]
    
    neighbors = NearestNeighbors(n_neighbors=1).fit(control)
    distances, indices = neighbors.kneighbors(treated)
    matched_control = control.iloc[indices.ravel()]
    
    treated_outcome = outcome[treatment == 1]
    matched_control_outcome = outcome[treatment == 0].iloc[indices.ravel()]
    
    return np.mean(treated_outcome) - np.mean(matched_control_outcome)

# Example usage
# ...
```

Slide 7: Instrumental Variables (IV)

Instrumental variables (IV) are used to estimate causal effects in the presence of unmeasured confounding variables. An instrumental variable is a variable that is associated with the treatment but not directly with the outcome, except through its effect on the treatment.

Code:

```python
import numpy as np
import statsmodels.api as sm

def iv_estimation(data, treatment, outcome, instrument):
    model = sm.IV2SLS(data[outcome], data[treatment], data[instrument])
    results = model.fit()
    return results.params[treatment]

# Example
# ...
```

Slide 8: Causal Graphs and Structural Equation Models

Causal graphs, also known as directed acyclic graphs (DAGs), are graphical representations of causal relationships among variables. Structural equation models (SEMs) are statistical models that combine causal graphs with equations to represent and estimate causal effects.

Code:

```python
import numpy as np
from causalnex.structure.notears import from_pandas
from causalnex.plots import plot_structure

# Generate synthetic data
# ...

# Learn the causal graph
sm = from_pandas(data)
plot_structure(sm, node_names=data.columns, graph_attributes={"scale": 2})
```

Slide 9: Mediation Analysis

Mediation analysis is a statistical technique used to investigate the mechanisms by which an independent variable (treatment) affects a dependent variable (outcome) through one or more intervening variables (mediators).

Code:

```python
import numpy as np
from statsmodels.formula.api import ols

def mediation_analysis(data, treatment, outcome, mediator):
    # Estimate the total effect
    total_model = ols(f'{outcome} ~ {treatment}', data=data).fit()
    total_effect = total_model.params[treatment]
    
    # Estimate the indirect effect through the mediator
    mediator_model = ols(f'{mediator} ~ {treatment}', data=data).fit()
    outcome_model = ols(f'{outcome} ~ {treatment} + {mediator}', data=data).fit()
    indirect_effect = mediator_model.params[treatment] * outcome_model.params[mediator]
    
    # Calculate the direct effect
    direct_effect = total_effect - indirect_effect
    
    return total_effect, direct_effect, indirect_effect

# Example usage
# ...
```

Slide 10: Causal Discovery Algorithms

Causal discovery algorithms aim to learn causal relationships from observational data without prior knowledge or assumptions about the underlying causal structure. These algorithms use various techniques, such as constraint-based methods (e.g., PC algorithm) or score-based methods (e.g., GES algorithm).

Code:

```python
import numpy as np
import pandas as pd
from causalnex.structure.notears import from_pandas
from causalnex.plots import plot_structure

# Generate synthetic data
# ...

# Learn the causal graph
sm = from_pandas(data)
plot_structure(sm, node_names=data.columns, graph_attributes={"scale": 2})
```

Slide 11: Causal Inference with Machine Learning

Machine learning techniques, such as neural networks and tree-based models, can be used in causal inference tasks. These methods can capture complex nonlinear relationships and interactions between variables, making them powerful tools for estimating causal effects in high-dimensional and complex datasets.

Code:

```python
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor

def ml_causal_inference(X, treatment, outcome):
    # Split data into treated and control groups
    X_treated = X[treatment == 1]
    X_control = X[treatment == 0]
    y_treated = outcome[treatment == 1]
    y_control = outcome[treatment == 0]
    
    # Train machine learning models
    mlp = MLPRegressor().fit(X_treated, y_treated)
    gbr = GradientBoostingRegressor().fit(X_control, y_control)
    
    # Estimate the causal effect
    y_treated_pred = mlp.predict(X_control)
    y_control_pred = gbr.predict(X_treated)
    
    causal_effect = np.mean(y_treated) - np.mean(y_control_pred)
    
    return causal_effect

# Example usage
# ...
```

Slide 12: Doubly Robust Estimation

Doubly robust estimation is a technique that combines propensity score modeling and outcome regression modeling. It provides consistent estimates of causal effects even if one of the two models (propensity score or outcome regression) is misspecified, making it a robust approach in various scenarios.

Code:

```python
import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression

def doubly_robust_estimation(X, treatment, outcome):
    # Estimate propensity scores
    propensity_model = LogisticRegression().fit(X, treatment)
    propensity_scores = propensity_model.predict_proba(X)[:, 1]
    
    # Estimate outcome regression
    outcome_model = LinearRegression().fit(X, outcome)
    
    # Compute the doubly robust estimate
    treated = treatment == 1
    control = treatment == 0
    
    mu_treated = np.mean(outcome[treated] - outcome_model.predict(X[treated]) / propensity_scores[treated])
    mu_control = np.mean((outcome_model.predict(X[control]) / (1 - propensity_scores[control])) * (1 - propensity_scores[control]))
    
    causal_effect = mu_treated - mu_control
    
    return causal_effect

# Example usage
# ...
```

Slide 13: Causal Inference in Practice

Causal inference in practice often involves careful data collection, variable selection, model specification, and assumption checking. It's crucial to consider potential biases, unmeasured confounding variables, and the limitations of the available data and methods.

Code:

```python
# Example pseudocode
# 1. Collect data and identify relevant variables
# 2. Check for missing data and handle it appropriately
# 3. Explore the data and visualize relationships
# 4. Select appropriate causal inference method(s)
# 5. Check assumptions and potential biases
# 6. Estimate causal effects and interpret results
# 7. Validate findings and assess robustness
# 8. Communicate results and limitations
```

Slide 14 (Additional Resources): Additional Resources

For further learning and exploration of causal inference in machine learning and data science using Python, here are some recommended resources:

* "Causal Inference in Statistics: A Primer" by Judea Pearl, Madelyn Glymour, and Nicholas P. Jewell
* "Causal Inference: What If" by Miguel A. Hernan and James M. Robins
* "Causal Inference: The Mixtape" by Cunningham, Scott ([https://arxiv.org/abs/2002.02631](https://arxiv.org/abs/2002.02631))
* "CausalNex" Python library ([https://causalnex.readthedocs.io/en/latest/](https://causalnex.readthedocs.io/en/latest/))
* "DoWhy" Python library ([https://py-why.github.io/dowhy/](https://py-why.github.io/dowhy/))
* "EconML" Python library ([https://econml.azurewebsites.net/](https://econml.azurewebsites.net/))

These resources cover theoretical foundations, practical applications, and Python libraries for causal inference tasks. Please note that the provided links and references are subject to availability and might change over time.

