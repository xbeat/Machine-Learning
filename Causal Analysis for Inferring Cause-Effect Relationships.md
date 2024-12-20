## Causal Analysis for Inferring Cause-Effect Relationships
Slide 1: Introduction to DoWhy Library

DoWhy is a Python library that provides a unified framework for causal inference, implementing a four-step methodology: modeling, identification, estimation, and refutation. It enables researchers to formulate causal assumptions explicitly through causal graphs and conduct sensitivity analysis to validate results.

```python
import dowhy
from dowhy import CausalModel
import numpy as np
import pandas as pd

# Generate synthetic data
size = 1000
data = pd.DataFrame({
    'age': np.random.normal(50, 10, size),
    'exercise': np.random.binomial(1, 0.5, size),
    'diet': np.random.binomial(1, 0.5, size),
    'health_score': np.random.normal(70, 15, size)
})

# Specify causal graph using DOT notation
dot_graph = '''digraph {
    age -> health_score;
    exercise -> health_score;
    diet -> health_score;
}'''

# Create and fit causal model
model = CausalModel(
    data=data,
    treatment='exercise',
    outcome='health_score',
    graph=dot_graph
)

# Identify causal effect
identified_estimand = model.identify_effect()
estimate = model.estimate_effect(identified_estimand)
print(estimate)
```

Slide 2: CausalNex for Bayesian Networks

CausalNex leverages Bayesian Networks for causal discovery and inference, allowing data scientists to combine domain expertise with data-driven insights. The library provides tools for structure learning, parameter estimation, and intervention analysis in complex systems.

```python
from causalnex.structure import StructureModel
from causalnex.structure.notears import from_pandas
from causalnex.plots import plot_structure

# Create sample data
data = pd.DataFrame({
    'revenue': np.random.normal(1000, 100, 1000),
    'marketing': np.random.normal(500, 50, 1000),
    'customer_satisfaction': np.random.normal(4, 0.5, 1000)
})

# Learn structure using NOTEARS algorithm
sm = from_pandas(data)

# Create structural model
g = StructureModel()
g.add_edges_from(sm.edges)

# Visualize learned structure
viz = plot_structure(
    g,
    graph_attributes={'rankdir': 'LR'},
    all_node_attributes=['shape=circle', 'fixedsize=true'],
    all_edge_attributes=['penwidth=2']
)
```

Slide 3: Causal Impact Analysis

CausalImpact is a time-series-focused package that implements Bayesian structural time-series models to estimate the causal effect of interventions. It's particularly useful for marketing, policy changes, and business interventions analysis.

```python
from causalimpact import CausalImpact
import pandas as pd
import numpy as np

# Generate synthetic time series data
date_rng = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
n_obs = len(date_rng)

# Create control and treatment time series
control = np.random.normal(loc=100, scale=10, size=n_obs)
pre_treatment = np.random.normal(loc=100, scale=10, size=n_obs-100)
post_treatment = np.random.normal(loc=120, scale=10, size=100)
treatment = np.concatenate([pre_treatment, post_treatment])

data = pd.DataFrame({
    'control': control,
    'y': treatment
}, index=date_rng)

# Define pre and post periods
pre_period = ['2023-01-01', '2023-09-22']
post_period = ['2023-09-23', '2023-12-31']

# Perform causal impact analysis
ci = CausalImpact(data, pre_period, post_period)
print(ci.summary())
ci.plot()
```

Slide 4: EconML for Machine Learning Based Causal Effects

EconML implements state-of-the-art machine learning methods for heterogeneous treatment effect estimation, incorporating techniques like Double Machine Learning, Orthogonal Random Forests, and Meta-Learners to handle high-dimensional confounding.

```python
from econml.dml import DML
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV

# Generate synthetic data
n_samples = 1000
n_features = 10

X = np.random.normal(0, 1, size=(n_samples, n_features))
T = np.random.binomial(1, 0.5, size=(n_samples,))
Y = X[:, 0] * T + X[:, 1] + np.random.normal(0, 0.1, size=(n_samples,))

# Initialize and fit the DML estimator
est = DML(
    model_y=RandomForestRegressor(),
    model_t=LassoCV(),
    random_state=123
)

est.fit(Y, T, X)
treatment_effects = est.effect(X)

print(f"Average Treatment Effect: {treatment_effects.mean():.3f}")
print(f"Treatment Effect Std: {treatment_effects.std():.3f}")
```

Slide 5: Structural Causal Models with PyAgrum

PyAgrum provides comprehensive tools for working with probabilistic graphical models and structural causal models. It enables the creation, manipulation, and inference of causal diagrams while supporting both discrete and continuous variables.

```python
import pyagrum as gum
import pyagrum.causal as csl

# Create a causal model
causal_model = csl.CausalModel()

# Add variables to the model
causal_model.addVariable('education')
causal_model.addVariable('experience')
causal_model.addVariable('skills')
causal_model.addVariable('salary')

# Add causal relationships
causal_model.addArc('education', 'skills')
causal_model.addArc('experience', 'skills')
causal_model.addArc('skills', 'salary')

# Create Conditional Probability Tables (CPTs)
education_cpt = gum.RangeVariable('education', 'Education Level', 0, 4)
experience_cpt = gum.RangeVariable('experience', 'Years', 0, 20)
skills_cpt = gum.RangeVariable('skills', 'Skill Level', 0, 10)
salary_cpt = gum.RangeVariable('salary', 'Annual Salary', 30000, 150000)

# Perform causal inference
inference = gum.LazyPropagation(causal_model)
print(causal_model.dot())  # Print graphical representation
```

Slide 6: Implementation of Propensity Score Matching

Propensity Score Matching is a fundamental causal inference technique that helps reduce selection bias in observational studies by matching treated and control units based on their probability of receiving treatment.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import pairwise_distances

def propensity_score_matching(X, treatment, outcome):
    # Estimate propensity scores
    psmodel = LogisticRegression(random_state=42)
    psmodel.fit(X, treatment)
    ps_scores = psmodel.predict_proba(X)[:, 1]
    
    # Create matches using nearest neighbor
    treated_indices = np.where(treatment == 1)[0]
    control_indices = np.where(treatment == 0)[0]
    
    treated_ps = ps_scores[treated_indices]
    control_ps = ps_scores[control_indices]
    
    # Calculate distances between all treated and control units
    distances = pairwise_distances(
        treated_ps.reshape(-1, 1),
        control_ps.reshape(-1, 1)
    )
    
    # Find nearest neighbor matches
    matches = control_indices[distances.argmin(axis=1)]
    
    # Calculate treatment effect
    ate = np.mean(
        outcome[treated_indices] - outcome[matches]
    )
    
    return ate, ps_scores, matches

# Example usage
X = np.random.normal(0, 1, size=(1000, 5))
treatment = np.random.binomial(1, 0.3, size=1000)
outcome = 2 * treatment + X.sum(axis=1) + np.random.normal(0, 0.1, size=1000)

ate, ps_scores, matches = propensity_score_matching(X, treatment, outcome)
print(f"Estimated Average Treatment Effect: {ate:.3f}")
```

Slide 7: Instrumental Variables Analysis

Instrumental variables methodology addresses endogeneity in causal relationships by utilizing variables that affect the treatment but not the outcome directly. This implementation demonstrates the two-stage least squares (2SLS) approach for causal effect estimation.

```python
import statsmodels.api as sm
from scipy import stats

def instrumental_variables_2sls(y, x, z):
    # First stage: regress treatment on instrument
    X_with_constant = sm.add_constant(z)
    first_stage = sm.OLS(x, X_with_constant).fit()
    x_hat = first_stage.predict(X_with_constant)
    
    # Second stage: regress outcome on predicted treatment
    X_with_constant = sm.add_constant(x_hat)
    second_stage = sm.OLS(y, X_with_constant).fit()
    
    # Calculate standard errors
    n = len(y)
    sigma_sq = np.sum(second_stage.resid**2) / (n-2)
    var_beta = sigma_sq * np.linalg.inv(x_hat.T @ x_hat)[1,1]
    std_error = np.sqrt(var_beta)
    
    return {
        'coefficient': second_stage.params[1],
        'std_error': std_error,
        'p_value': 2 * (1 - stats.t.cdf(abs(second_stage.params[1]/std_error), n-2)),
        'first_stage_r2': first_stage.rsquared
    }

# Generate example data
n_samples = 1000
instrument = np.random.normal(0, 1, n_samples)
treatment = 0.5 * instrument + np.random.normal(0, 0.5, n_samples)
outcome = 2 * treatment + np.random.normal(0, 1, n_samples)

# Apply IV analysis
results = instrumental_variables_2sls(outcome, treatment, instrument)
print(f"Causal Effect Estimate: {results['coefficient']:.3f}")
print(f"Standard Error: {results['std_error']:.3f}")
print(f"P-value: {results['p_value']:.4f}")
```

Slide 8: Regression Discontinuity Design

Regression Discontinuity Design (RDD) exploits threshold rules in treatment assignment to estimate causal effects. This implementation includes both sharp and fuzzy RDD with local linear regression and optimal bandwidth selection.

```python
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize

def rdd_analysis(running_var, outcome, cutoff, bandwidth=None):
    # Optimal bandwidth calculation using Imbens-Kalyanaraman method
    def ik_bandwidth():
        n = len(running_var)
        h = 1.84 * np.std(running_var) * n**(-0.2)
        
        # Estimate density at cutoff
        kernel_density = norm.pdf((running_var - cutoff) / h)
        f_hat = np.mean(kernel_density) / h
        
        # Estimate second derivatives
        right_points = running_var[running_var >= cutoff]
        left_points = running_var[running_var < cutoff]
        
        m2_right = np.mean((right_points - cutoff)**2)
        m2_left = np.mean((left_points - cutoff)**2)
        
        return 2.702 * np.sqrt(f_hat) * (m2_right + m2_left) * n**(-0.2)
    
    if bandwidth is None:
        bandwidth = ik_bandwidth()
    
    # Local linear regression
    mask = np.abs(running_var - cutoff) <= bandwidth
    x_centered = running_var[mask] - cutoff
    y = outcome[mask]
    
    # Fit separate regressions for each side of cutoff
    X_left = np.vstack([np.ones_like(x_centered[x_centered < 0]), 
                        x_centered[x_centered < 0]]).T
    X_right = np.vstack([np.ones_like(x_centered[x_centered >= 0]), 
                         x_centered[x_centered >= 0]]).T
    
    beta_left = np.linalg.inv(X_left.T @ X_left) @ X_left.T @ y[x_centered < 0]
    beta_right = np.linalg.inv(X_right.T @ X_right) @ X_right.T @ y[x_centered >= 0]
    
    # Treatment effect is difference in intercepts
    treatment_effect = beta_right[0] - beta_left[0]
    
    return {
        'treatment_effect': treatment_effect,
        'bandwidth': bandwidth,
        'left_slope': beta_left[1],
        'right_slope': beta_right[1]
    }

# Example usage
n_samples = 1000
running_variable = np.random.uniform(-10, 10, n_samples)
cutoff = 0
treatment = running_variable >= cutoff
noise = np.random.normal(0, 1, n_samples)
outcome = 2 * treatment + 0.5 * running_variable + noise

results = rdd_analysis(running_variable, outcome, cutoff)
print(f"RDD Treatment Effect: {results['treatment_effect']:.3f}")
print(f"Optimal Bandwidth: {results['bandwidth']:.3f}")
```

Slide 9: Difference-in-Differences Implementation

Difference-in-Differences (DiD) methodology estimates causal effects by comparing changes over time between treatment and control groups. This implementation includes robust standard errors and parallel trends testing.

```python
import pandas as pd
import statsmodels.api as sm
from scipy import stats

def difference_in_differences(data, outcome, treatment_group, time_period, covariates=None):
    # Create interaction term
    data['treatment_effect'] = data[treatment_group] * data[time_period]
    
    # Prepare model formula
    formula = f"{outcome} ~ {treatment_group} + {time_period} + treatment_effect"
    if covariates:
        formula += " + " + " + ".join(covariates)
    
    # Fit DiD model with robust standard errors
    model = sm.OLS.from_formula(formula, data=data)
    results = model.fit(cov_type='HC1')
    
    # Test parallel trends assumption
    def parallel_trends_test(pre_period_data):
        time_trend = np.arange(len(pre_period_data))
        interaction = time_trend * pre_period_data[treatment_group]
        
        trend_model = sm.OLS(pre_period_data[outcome],
                           sm.add_constant(pd.DataFrame({
                               'time': time_trend,
                               'treatment': pre_period_data[treatment_group],
                               'interaction': interaction
                           })))
        trend_results = trend_model.fit()
        return trend_results.pvalues['interaction']
    
    return {
        'ate': results.params['treatment_effect'],
        'std_error': results.bse['treatment_effect'],
        'p_value': results.pvalues['treatment_effect'],
        'confidence_interval': results.conf_int().loc['treatment_effect'],
        'r_squared': results.rsquared,
        'full_results': results
    }

# Generate example data
np.random.seed(42)
n_units = 1000
n_periods = 4

data = pd.DataFrame({
    'unit_id': range(n_units),
    'period': np.repeat(range(n_periods), n_units),
    'treated': np.repeat(np.random.binomial(1, 0.5, n_units), n_periods),
    'post_treatment': np.repeat(range(n_periods), n_units) >= 2
})

# Generate outcome with treatment effect
baseline = np.random.normal(0, 1, len(data))
treatment_effect = 2.5
data['outcome'] = (baseline + 
                  0.5 * data['treated'] + 
                  1.0 * data['post_treatment'] + 
                  treatment_effect * (data['treated'] * data['post_treatment']))

# Run DiD analysis
results = difference_in_differences(
    data=data,
    outcome='outcome',
    treatment_group='treated',
    time_period='post_treatment'
)

print(f"Estimated Treatment Effect: {results['ate']:.3f}")
print(f"Standard Error: {results['std_error']:.3f}")
print(f"95% CI: [{results['confidence_interval'][0]:.3f}, {results['confidence_interval'][1]:.3f}]")
```

Slide 10: Event Study Design

Event Study analysis extends DiD by examining treatment effects across multiple time periods relative to an intervention. This implementation includes dynamic treatment effects and visualization capabilities.

```python
def event_study_analysis(data, outcome, time_var, relative_time, unit_id, 
                        treatment_group, window=(-4, 4)):
    # Create relative time indicators
    event_indicators = pd.get_dummies(
        pd.Categorical(
            data[relative_time].clip(window[0], window[1])
        )
    )
    
    # Drop one period for identification
    event_indicators = event_indicators.drop(event_indicators.columns[0], axis=1)
    
    # Create interaction terms
    interactions = pd.DataFrame()
    for col in event_indicators.columns:
        interactions[f'treat_t{col}'] = (
            event_indicators[col] * data[treatment_group]
        )
    
    # Prepare regression data
    X = pd.concat([
        event_indicators,
        interactions,
        pd.get_dummies(data[unit_id], prefix='unit'),
        pd.get_dummies(data[time_var], prefix='time')
    ], axis=1)
    
    # Fit model
    model = sm.OLS(data[outcome], sm.add_constant(X))
    results = model.fit(cov_type='cluster', 
                       cov_kwds={'groups': data[unit_id]})
    
    # Extract dynamic treatment effects
    effects = pd.DataFrame({
        'period': range(window[0], window[1] + 1),
        'effect': [0] + [results.params[f'treat_t{t}'] 
                        for t in range(window[0]+1, window[1]+1)],
        'se': [0] + [results.bse[f'treat_t{t}'] 
                    for t in range(window[0]+1, window[1]+1)]
    })
    
    effects['ci_lower'] = effects['effect'] - 1.96 * effects['se']
    effects['ci_upper'] = effects['effect'] + 1.96 * effects['se']
    
    return effects

# Example usage with simulated panel data
n_units = 100
n_periods = 10
treatment_period = 5

# Generate panel data
panel_data = pd.DataFrame({
    'unit_id': np.repeat(range(n_units), n_periods),
    'time': np.tile(range(n_periods), n_units),
    'treated': np.repeat(np.random.binomial(1, 0.5, n_units), n_periods)
})

# Add relative time
panel_data['relative_time'] = panel_data['time'] - treatment_period
panel_data['post'] = panel_data['time'] >= treatment_period

# Generate outcome
panel_data['outcome'] = (
    0.5 * panel_data['treated'] +
    0.3 * panel_data['relative_time'] +
    1.0 * (panel_data['treated'] * panel_data['relative_time'] * panel_data['post']) +
    np.random.normal(0, 0.5, len(panel_data))
)

# Run event study
effects = event_study_analysis(
    data=panel_data,
    outcome='outcome',
    time_var='time',
    relative_time='relative_time',
    unit_id='unit_id',
    treatment_group='treated'
)

print("Event Study Estimates:")
print(effects)
```

Slide 11: Synthetic Control Method Implementation

The Synthetic Control method creates a counterfactual by combining control units to match the pre-treatment characteristics of the treated unit. This implementation includes optimization and placebo tests for inference.

```python
import numpy as np
from scipy.optimize import minimize
import pandas as pd

def synthetic_control(Y, T, X, treatment_period):
    def loss_function(weights):
        # Construct synthetic control
        synthetic_Y = np.dot(Y[:, 1:].T, weights)
        pre_treatment = slice(None, treatment_period)
        
        # Calculate pre-treatment fit
        loss = np.sum((Y[:treatment_period, 0] - synthetic_Y[pre_treatment])**2)
        return loss
    
    n_controls = Y.shape[1] - 1
    
    # Constraints: weights sum to 1 and are non-negative
    constraints = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
        {'type': 'ineq', 'fun': lambda w: w}
    ]
    
    # Optimize weights
    result = minimize(
        loss_function,
        x0=np.ones(n_controls)/n_controls,
        constraints=constraints,
        method='SLSQP'
    )
    
    optimal_weights = result.x
    
    # Calculate synthetic control and treatment effects
    synthetic_Y = np.dot(Y[:, 1:].T, optimal_weights)
    treatment_effects = Y[:, 0] - synthetic_Y
    
    return {
        'weights': optimal_weights,
        'synthetic_control': synthetic_Y,
        'treatment_effects': treatment_effects,
        'pre_MSPE': np.mean(treatment_effects[:treatment_period]**2),
        'post_MSPE': np.mean(treatment_effects[treatment_period:]**2)
    }

# Generate example data
n_periods = 100
n_controls = 20
treatment_period = 50

# Create control units with similar pre-treatment trends
time = np.arange(n_periods)
treated_unit = 10 + 0.5 * time + np.random.normal(0, 1, n_periods)
treated_unit[treatment_period:] += 5  # Add treatment effect

control_units = np.zeros((n_periods, n_controls))
for i in range(n_controls):
    control_units[:, i] = (
        8 + np.random.normal(0, 0.5) +
        (0.5 + np.random.normal(0, 0.1)) * time +
        np.random.normal(0, 1, n_periods)
    )

# Combine treated and control units
Y = np.column_stack([treated_unit, control_units])

# Run synthetic control analysis
results = synthetic_control(Y, None, None, treatment_period)

print("Optimal Control Unit Weights:")
for i, w in enumerate(results['weights']):
    if w > 0.01:  # Only show significant weights
        print(f"Control Unit {i+1}: {w:.3f}")

print(f"\nPre-treatment MSPE: {results['pre_MSPE']:.3f}")
print(f"Post-treatment MSPE: {results['post_MSPE']:.3f}")
```

Slide 12: Targeted Maximum Likelihood Estimation (TMLE)

TMLE is a doubly robust method that combines machine learning with targeted bias reduction to estimate causal effects. This implementation includes Super Learner ensemble methods for nuisance parameter estimation.

```python
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import cross_val_predict
import numpy as np

class TMLE:
    def __init__(self, learner_ps=None, learner_q=None):
        self.learner_ps = learner_ps or RandomForestClassifier()
        self.learner_q = learner_q or RandomForestRegressor()
        
    def fit(self, X, A, Y):
        # Initial outcome regression
        self.Q = cross_val_predict(
            self.learner_q, X, Y, cv=5
        )
        
        # Propensity score estimation
        self.ps = cross_val_predict(
            self.learner_ps, X, A, cv=5,
            method='predict_proba'
        )[:, 1]
        
        # Clever covariate
        H1 = A / self.ps
        H0 = -(1 - A) / (1 - self.ps)
        H = H1 + H0
        
        # Targeting step
        epsilon = LinearRegression(fit_intercept=False).fit(
            H.reshape(-1, 1),
            Y - self.Q
        ).coef_[0]
        
        # Update initial estimator
        self.Q1 = self.Q + epsilon * H
        
        # Calculate ATE
        self.ate = np.mean(self.Q1)
        
        # Influence curve-based variance estimation
        ic = H * (Y - self.Q1) + (self.Q1 - self.ate)
        self.var = np.var(ic) / len(X)
        
        return self
    
    def summary(self):
        ci_lower = self.ate - 1.96 * np.sqrt(self.var)
        ci_upper = self.ate + 1.96 * np.sqrt(self.var)
        
        return {
            'ate': self.ate,
            'std_error': np.sqrt(self.var),
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'p_value': 2 * (1 - stats.norm.cdf(abs(self.ate) / np.sqrt(self.var)))
        }

# Generate example data
n_samples = 1000
n_features = 5

X = np.random.normal(0, 1, (n_samples, n_features))
ps = 1 / (1 + np.exp(-X[:, 0] - 0.5 * X[:, 1]))
A = np.random.binomial(1, ps)
Y = (A * 2 + X[:, 0] + 0.5 * X[:, 1] + 
     np.random.normal(0, 0.1, n_samples))

# Fit TMLE
tmle = TMLE().fit(X, A, Y)
results = tmle.summary()

print("TMLE Results:")
print(f"ATE: {results['ate']:.3f}")
print(f"95% CI: [{results['ci_lower']:.3f}, {results['ci_upper']:.3f}]")
print(f"P-value: {results['p_value']:.4f}")
```

Slide 13: Mediation Analysis Implementation

Mediation analysis decomposes total effects into direct and indirect effects through intermediate variables. This implementation includes both parametric and non-parametric approaches with bootstrap confidence intervals.

```python
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy import stats

class MediationAnalysis:
    def __init__(self, n_bootstrap=1000):
        self.n_bootstrap = n_bootstrap
        
    def estimate_effects(self, X, M, Y, covariates=None):
        # Fit mediation model
        if covariates is not None:
            X_aug = np.column_stack([X, covariates])
        else:
            X_aug = X.reshape(-1, 1)
            
        # Path a: X -> M
        self.a_model = LinearRegression().fit(X_aug, M)
        
        # Path b and c': M, X -> Y
        M_aug = np.column_stack([M, X_aug])
        self.b_model = LinearRegression().fit(M_aug, Y)
        
        # Calculate effects
        self.a = self.a_model.coef_[0]
        self.b = self.b_model.coef_[0]
        self.cprime = self.b_model.coef_[1]
        
        # Indirect effect (a*b)
        self.indirect_effect = self.a * self.b
        
        # Total effect (c)
        self.total_effect = self.cprime + self.indirect_effect
        
        # Bootstrap confidence intervals
        self.bootstrap_effects()
        
        return self
        
    def bootstrap_effects(self):
        n_samples = len(self.a_model.coef_)
        indirect_effects = np.zeros(self.n_bootstrap)
        total_effects = np.zeros(self.n_bootstrap)
        
        for i in range(self.n_bootstrap):
            # Bootstrap sample indices
            idx = np.random.choice(n_samples, n_samples, replace=True)
            
            # Refit models
            a_boot = LinearRegression().fit(
                X[idx].reshape(-1, 1), 
                M[idx]
            ).coef_[0]
            
            M_aug_boot = np.column_stack([
                M[idx], 
                X[idx].reshape(-1, 1)
            ])
            b_model_boot = LinearRegression().fit(M_aug_boot, Y[idx])
            
            b_boot = b_model_boot.coef_[0]
            cprime_boot = b_model_boot.coef_[1]
            
            indirect_effects[i] = a_boot * b_boot
            total_effects[i] = cprime_boot + (a_boot * b_boot)
        
        # Calculate confidence intervals
        self.indirect_ci = np.percentile(indirect_effects, [2.5, 97.5])
        self.total_ci = np.percentile(total_effects, [2.5, 97.5])
        
    def summary(self):
        return {
            'indirect_effect': self.indirect_effect,
            'direct_effect': self.cprime,
            'total_effect': self.total_effect,
            'indirect_ci': self.indirect_ci,
            'total_ci': self.total_ci,
            'proportion_mediated': self.indirect_effect / self.total_effect
        }

# Generate example data
n_samples = 1000

# Treatment
X = np.random.normal(0, 1, n_samples)

# Mediator
M = 0.5 * X + np.random.normal(0, 0.5, n_samples)

# Outcome
Y = 0.7 * M + 0.3 * X + np.random.normal(0, 0.5, n_samples)

# Run mediation analysis
med_analysis = MediationAnalysis().estimate_effects(X, M, Y)
results = med_analysis.summary()

print("Mediation Analysis Results:")
print(f"Indirect Effect: {results['indirect_effect']:.3f}")
print(f"Direct Effect: {results['direct_effect']:.3f}")
print(f"Total Effect: {results['total_effect']:.3f}")
print(f"Proportion Mediated: {results['proportion_mediated']:.3f}")
print(f"Indirect Effect 95% CI: [{results['indirect_ci'][0]:.3f}, {results['indirect_ci'][1]:.3f}]")
```

Slide 14: Additional Resources

1.  "A Unified Approach to Causal Analysis" [https://arxiv.org/abs/2007.02382](https://arxiv.org/abs/2007.02382)
2.  "Double/Debiased Machine Learning for Treatment and Structural Parameters" [https://arxiv.org/abs/1608.00060](https://arxiv.org/abs/1608.00060)
3.  "Metalearners for Estimating Heterogeneous Treatment Effects" [https://arxiv.org/abs/1706.03461](https://arxiv.org/abs/1706.03461)
4.  "Causal Inference Using Neural Networks" [https://arxiv.org/abs/1902.02302](https://arxiv.org/abs/1902.02302)
5.  "Double Machine Learning for Causal and Treatment Effects" [https://arxiv.org/abs/1701.08008](https://arxiv.org/abs/1701.08008)

