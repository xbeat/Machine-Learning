## Correlation vs Causation Understanding the Difference
Slide 1: Understanding Correlation

Correlation measures the statistical relationship between two variables, indicating how they move together. It quantifies the strength and direction of their linear relationship, producing values between -1 and +1, where -1 indicates perfect negative correlation and +1 perfect positive correlation.

```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Generate correlated data
np.random.seed(42)
x = np.random.normal(0, 1, 1000)
y = 0.8 * x + np.random.normal(0, 0.5, 1000)  # Strong positive correlation

# Calculate Pearson correlation
correlation = np.corrcoef(x, y)[0, 1]

# Visualize
plt.figure(figsize=(10, 6))
plt.scatter(x, y, alpha=0.5)
plt.title(f'Correlation Example (r = {correlation:.2f})')
plt.xlabel('Variable X')
plt.ylabel('Variable Y')
plt.show()
```

Slide 2: Mathematical Foundation of Correlation

The Pearson correlation coefficient, commonly denoted as r, is calculated using covariance and standard deviations. This statistical measure helps quantify the linear relationship between variables while normalizing for their individual scales and variances.

```python
def pearson_correlation(x, y):
    # Calculate means
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    
    # Calculate numerator (covariance)
    numerator = np.sum((x - x_mean) * (y - y_mean))
    
    # Calculate denominator (product of standard deviations)
    denominator = np.sqrt(np.sum((x - x_mean)**2) * np.sum((y - y_mean)**2))
    
    # Calculate correlation coefficient
    r = numerator / denominator
    
    return r

# Mathematical formula (LaTeX notation)
print("Pearson Correlation Formula:")
print("$$r = \\frac{\\sum_{i=1}^{n} (x_i - \\bar{x})(y_i - \\bar{y})}{\\sqrt{\\sum_{i=1}^{n} (x_i - \\bar{x})^2 \\sum_{i=1}^{n} (y_i - \\bar{y})^2}}$$")
```

Slide 3: Understanding Causation

Causation implies that changes in one variable directly influence changes in another variable, establishing a cause-and-effect relationship. This is fundamentally different from correlation, as it requires experimental evidence and controlled studies to validate.

```python
import pandas as pd
from scipy import stats

# Simulate data for ice cream sales and crime rates
np.random.seed(42)
n_samples = 100

# Temperature affects both variables (hidden common cause)
temperature = np.random.normal(25, 5, n_samples)

# Ice cream sales influenced by temperature
ice_cream_sales = 100 + 2 * temperature + np.random.normal(0, 10, n_samples)

# Crime rates influenced by temperature
crime_rates = 50 + 1.5 * temperature + np.random.normal(0, 8, n_samples)

# Calculate correlation
correlation = stats.pearsonr(ice_cream_sales, crime_rates)[0]
print(f"Correlation between ice cream sales and crime rates: {correlation:.2f}")
```

Slide 4: Spurious Correlations

Spurious correlations occur when two variables appear to be related but have no logical or causal connection. These relationships often arise by chance or due to hidden confounding variables, highlighting why correlation alone cannot prove causation.

```python
# Simulate spurious correlation example
years = np.arange(2000, 2020)
internet_users = 100 * (1 / (1 + np.exp(-0.5 * (years - 2010)))) + np.random.normal(0, 2, 20)
organic_food_sales = 50 * (1 / (1 + np.exp(-0.4 * (years - 2008)))) + np.random.normal(0, 3, 20)

# Calculate and plot correlation
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.plot(years, internet_users, 'b-', label='Internet Users')
plt.plot(years, organic_food_sales, 'g-', label='Organic Food Sales')
plt.legend()
plt.title('Trends Over Time')

plt.subplot(122)
plt.scatter(internet_users, organic_food_sales)
plt.xlabel('Internet Users')
plt.ylabel('Organic Food Sales')
plt.title(f'Correlation: {np.corrcoef(internet_users, organic_food_sales)[0,1]:.2f}')
plt.tight_layout()
plt.show()
```

Slide 5: Testing for Causality

Testing for causality requires rigorous experimental design and statistical methods beyond simple correlation analysis. The Bradford Hill criteria and randomized controlled trials are essential tools for establishing causal relationships in scientific research.

```python
def granger_causality_test(x, y, max_lag=5):
    """
    Implement simple Granger causality test
    """
    from statsmodels.tsa.stattools import grangercausalitytests
    
    # Create DataFrame
    data = pd.DataFrame({'x': x, 'y': y})
    
    # Perform Granger causality test
    result = grangercausalitytests(data[['y', 'x']], maxlag=max_lag, verbose=False)
    
    # Extract p-values for each lag
    p_values = [result[i+1][0]['ssr_chi2test'][1] for i in range(max_lag)]
    
    return p_values

# Example usage with simulated time series data
t = np.linspace(0, 100, 1000)
x = np.sin(0.1 * t) + np.random.normal(0, 0.1, 1000)
y = np.roll(x, 20) + np.random.normal(0, 0.1, 1000)  # y is caused by x with lag

p_values = granger_causality_test(x, y)
print("Granger causality p-values for different lags:", p_values)
```

Slide 6: Regression Analysis and Causation

Regression analysis helps explore relationships between variables while controlling for confounding factors. However, even sophisticated regression models cannot definitively prove causation without proper experimental design and domain knowledge to support causal inference.

```python
import statsmodels.api as sm

# Simulate data with multiple variables
np.random.seed(42)
n_samples = 1000

# Independent variables
X1 = np.random.normal(0, 1, n_samples)  # Direct cause
X2 = np.random.normal(0, 1, n_samples)  # Confounding variable
X3 = 0.5 * X1 + np.random.normal(0, 0.1, n_samples)  # Correlated with X1

# Dependent variable
Y = 2 * X1 + 3 * X2 + np.random.normal(0, 1, n_samples)

# Create regression model
X = sm.add_constant(np.column_stack((X1, X2, X3)))
model = sm.OLS(Y, X).fit()

# Print results
print(model.summary().tables[1])

# Visualization of partial relationships
plt.figure(figsize=(15, 5))
for i, (x, name) in enumerate(zip([X1, X2, X3], ['X1', 'X2', 'X3'])):
    plt.subplot(1, 3, i+1)
    plt.scatter(x, Y, alpha=0.5)
    plt.xlabel(name)
    plt.ylabel('Y')
    plt.title(f'Partial relationship: {name} vs Y')
plt.tight_layout()
plt.show()
```

Slide 7: Establishing Causality through Experimentation

A carefully designed experiment with random assignment, control groups, and manipulation of independent variables provides stronger evidence for causality than observational studies. This slide demonstrates implementing a basic A/B testing framework.

```python
class ABTest:
    def __init__(self, control_size, treatment_size):
        np.random.seed(42)
        # Simulate control group
        self.control = np.random.normal(10, 2, control_size)
        # Simulate treatment group with causal effect
        self.treatment = np.random.normal(11, 2, treatment_size)
        
    def calculate_statistics(self):
        # Calculate basic statistics
        control_mean = np.mean(self.control)
        treatment_mean = np.mean(self.treatment)
        
        # Perform t-test
        t_stat, p_value = stats.ttest_ind(self.control, self.treatment)
        
        # Calculate effect size (Cohen's d)
        pooled_std = np.sqrt((np.var(self.control) + np.var(self.treatment)) / 2)
        cohens_d = (treatment_mean - control_mean) / pooled_std
        
        return {
            'control_mean': control_mean,
            'treatment_mean': treatment_mean,
            'p_value': p_value,
            'effect_size': cohens_d
        }

# Run experiment
experiment = ABTest(1000, 1000)
results = experiment.calculate_statistics()

# Visualize results
plt.figure(figsize=(10, 6))
plt.hist(experiment.control, alpha=0.5, label='Control')
plt.hist(experiment.treatment, alpha=0.5, label='Treatment')
plt.legend()
plt.title('Distribution of Outcomes in Control vs Treatment Groups')
plt.xlabel('Value')
plt.ylabel('Frequency')

print(f"Results:\n"
      f"Control Mean: {results['control_mean']:.2f}\n"
      f"Treatment Mean: {results['treatment_mean']:.2f}\n"
      f"P-value: {results['p_value']:.4f}\n"
      f"Effect Size: {results['effect_size']:.2f}")
```

Slide 8: Handling Time Series Causality

Time series analysis requires special consideration when investigating causal relationships. This implementation demonstrates the Granger Causality test and time series cross-correlation to identify potential temporal causal relationships.

```python
def time_series_causality_analysis(x, y, lags=5):
    # Calculate cross-correlation
    cross_corr = np.correlate(x - np.mean(x), 
                             y - np.mean(y), 
                             mode='full') / (len(x) * np.std(x) * np.std(y))
    
    # Center lags array
    lags_array = np.arange(-(len(x)-1), len(x))
    
    # Create time series DataFrame
    df = pd.DataFrame({
        'x': x,
        'y': y
    })
    
    # Calculate lagged correlations
    lagged_corrs = []
    for i in range(1, lags + 1):
        corr = df['x'].corr(df['y'].shift(-i))
        lagged_corrs.append(corr)
    
    return {
        'cross_correlation': cross_corr,
        'lags': lags_array,
        'lagged_correlations': lagged_corrs
    }

# Generate example time series with causal relationship
t = np.linspace(0, 100, 1000)
x = np.sin(0.1 * t) + np.random.normal(0, 0.1, 1000)
y = np.roll(x, 50) + np.random.normal(0, 0.1, 1000)  # y follows x with lag

# Analyze causality
results = time_series_causality_analysis(x, y)

# Plot results
plt.figure(figsize=(12, 6))
plt.subplot(211)
plt.plot(results['lags'], results['cross_correlation'])
plt.title('Cross-correlation Function')
plt.xlabel('Lag')
plt.ylabel('Correlation')

plt.subplot(212)
plt.plot(range(1, len(results['lagged_correlations']) + 1), 
         results['lagged_correlations'], 
         marker='o')
plt.title('Lagged Correlations')
plt.xlabel('Lag')
plt.ylabel('Correlation')
plt.tight_layout()
plt.show()
```

Slide 9: Real-World Example - Weather and Ice Cream Sales

This example demonstrates how to analyze the relationship between temperature and ice cream sales, illustrating how correlation might suggest causation but requires additional evidence and control variables to establish a causal link.

```python
# Generate realistic weather and sales data
np.random.seed(42)
n_days = 365

# Temperature data with seasonal pattern
time = np.linspace(0, 2*np.pi, n_days)
temperature = 20 + 15 * np.sin(time) + np.random.normal(0, 2, n_days)

# Ice cream sales with temperature dependency and random noise
base_sales = 100
temperature_effect = 5
random_variation = np.random.normal(0, 10, n_days)
ice_cream_sales = base_sales + temperature_effect * (temperature - temperature.mean())/temperature.std() + random_variation

# Create analysis model
def analyze_sales_weather_relationship(temp, sales):
    # Create DataFrame
    df = pd.DataFrame({
        'Temperature': temp,
        'Sales': sales
    })
    
    # Calculate correlation
    correlation = df['Temperature'].corr(df['Sales'])
    
    # Fit linear regression
    X = sm.add_constant(df['Temperature'])
    model = sm.OLS(df['Sales'], X).fit()
    
    # Plot relationships
    plt.figure(figsize=(15, 5))
    
    # Time series plot
    plt.subplot(131)
    plt.plot(df.index, df['Temperature'], label='Temperature')
    plt.plot(df.index, df['Sales'], label='Sales')
    plt.legend()
    plt.title('Time Series View')
    
    # Scatter plot
    plt.subplot(132)
    plt.scatter(df['Temperature'], df['Sales'], alpha=0.5)
    plt.plot(df['Temperature'], model.predict(X), 'r-', label='Regression Line')
    plt.xlabel('Temperature')
    plt.ylabel('Sales')
    plt.title(f'Correlation: {correlation:.2f}')
    
    # Residual plot
    plt.subplot(133)
    plt.scatter(model.predict(X), model.resid, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Sales')
    plt.ylabel('Residuals')
    plt.title('Residual Analysis')
    
    plt.tight_layout()
    plt.show()
    
    return model.summary()

# Run analysis
analysis_results = analyze_sales_weather_relationship(temperature, ice_cream_sales)
print(analysis_results)
```

Slide 10: Real-World Example - Education and Income

This implementation explores the relationship between education levels and income, demonstrating how to handle potential confounding variables and interpret results in a causal framework.

```python
# Generate synthetic dataset with confounding variables
n_samples = 1000

# Generate education years (12-20 years)
education_years = np.random.normal(16, 2, n_samples)
education_years = np.clip(education_years, 12, 20)

# Generate confounding variables
ability_score = np.random.normal(100, 15, n_samples)
family_background = np.random.normal(0, 1, n_samples)

# Generate income with multiple influences
base_income = 30000
education_effect = 5000
ability_effect = 200
family_effect = 3000

income = (base_income + 
         education_effect * (education_years - 12) +
         ability_effect * (ability_score - 100)/15 +
         family_effect * family_background +
         np.random.normal(0, 5000, n_samples))

def causal_analysis(education, income, ability, family):
    # Create DataFrame
    df = pd.DataFrame({
        'Education': education,
        'Income': income,
        'Ability': ability,
        'Family': family
    })
    
    # Simple correlation
    simple_corr = df['Education'].corr(df['Income'])
    
    # Multiple regression
    X = sm.add_constant(df[['Education', 'Ability', 'Family']])
    model = sm.OLS(df['Income'], X).fit()
    
    # Visualization
    plt.figure(figsize=(15, 5))
    
    # Raw relationship
    plt.subplot(131)
    plt.scatter(df['Education'], df['Income'], alpha=0.5)
    plt.xlabel('Education Years')
    plt.ylabel('Income')
    plt.title(f'Raw Correlation: {simple_corr:.2f}')
    
    # Partial residual plot for education
    residuals = df['Income'] - (model.params['Ability'] * df['Ability'] + 
                               model.params['Family'] * df['Family'] + 
                               model.params['const'])
    plt.subplot(132)
    plt.scatter(df['Education'], residuals, alpha=0.5)
    plt.xlabel('Education Years')
    plt.ylabel('Partial Residuals')
    plt.title('Education Effect (Controlling for Confounders)')
    
    # Coefficient plot
    plt.subplot(133)
    coef = model.params[1:]
    err = model.bse[1:]
    plt.errorbar(coef, range(len(coef)), xerr=1.96*err, fmt='o')
    plt.yticks(range(len(coef)), ['Education', 'Ability', 'Family'])
    plt.title('Coefficient Estimates with 95% CI')
    
    plt.tight_layout()
    plt.show()
    
    return model.summary()

# Run analysis
causal_results = causal_analysis(education_years, income, ability_score, family_background)
print(causal_results)
```

Slide 11: Mediation Analysis

Mediation analysis helps understand the mechanisms through which one variable affects another by identifying intermediate variables (mediators) in the causal pathway. This implementation demonstrates how to conduct a basic mediation analysis.

```python
class MediationAnalysis:
    def __init__(self, X, M, Y):
        self.X = X  # Independent variable
        self.M = M  # Mediator
        self.Y = Y  # Dependent variable
        
    def analyze(self):
        # Path a: X -> M
        X_const = sm.add_constant(self.X)
        model_a = sm.OLS(self.M, X_const).fit()
        a_path = model_a.params[1]
        
        # Path b and c': M, X -> Y
        XM_const = sm.add_constant(np.column_stack((self.X, self.M)))
        model_b = sm.OLS(self.Y, XM_const).fit()
        b_path = model_b.params[2]
        direct_effect = model_b.params[1]
        
        # Total effect (c): X -> Y
        model_c = sm.OLS(self.Y, X_const).fit()
        total_effect = model_c.params[1]
        
        # Indirect effect (a*b)
        indirect_effect = a_path * b_path
        
        # Proportion mediated
        prop_mediated = indirect_effect / total_effect
        
        return {
            'a_path': a_path,
            'b_path': b_path,
            'direct_effect': direct_effect,
            'indirect_effect': indirect_effect,
            'total_effect': total_effect,
            'prop_mediated': prop_mediated
        }

# Generate example data
n = 1000
X = np.random.normal(0, 1, n)  # Independent variable
M = 0.5 * X + np.random.normal(0, 1, n)  # Mediator
Y = 0.3 * X + 0.4 * M + np.random.normal(0, 1, n)  # Dependent variable

# Run mediation analysis
mediation = MediationAnalysis(X, M, Y)
results = mediation.analyze()

# Visualize results
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.scatter(X, M, alpha=0.5)
plt.xlabel('X (Independent Variable)')
plt.ylabel('M (Mediator)')
plt.title(f'Path a: {results["a_path"]:.3f}')

plt.subplot(122)
plt.scatter(M, Y, alpha=0.5)
plt.xlabel('M (Mediator)')
plt.ylabel('Y (Dependent Variable)')
plt.title(f'Path b: {results["b_path"]:.3f}')

print(f"Direct Effect: {results['direct_effect']:.3f}")
print(f"Indirect Effect: {results['indirect_effect']:.3f}")
print(f"Total Effect: {results['total_effect']:.3f}")
print(f"Proportion Mediated: {results['prop_mediated']:.3f}")
```

Slide 12: Instrumental Variables Analysis

Instrumental variables (IV) analysis is a powerful technique for establishing causality when randomization is impossible and confounding variables are present. This implementation shows how to perform basic IV regression.

```python
class InstrumentalVariablesAnalysis:
    def __init__(self, Z, X, Y):
        self.Z = Z  # Instrument
        self.X = X  # Endogenous variable
        self.Y = Y  # Outcome
        
    def two_stage_least_squares(self):
        # First stage: regress X on Z
        Z_const = sm.add_constant(self.Z)
        first_stage = sm.OLS(self.X, Z_const).fit()
        X_hat = first_stage.predict(Z_const)
        
        # Second stage: regress Y on predicted X
        X_hat_const = sm.add_constant(X_hat)
        second_stage = sm.OLS(self.Y, X_hat_const).fit()
        
        # Compare with naive OLS
        X_const = sm.add_constant(self.X)
        naive_ols = sm.OLS(self.Y, X_const).fit()
        
        return {
            'iv_estimate': second_stage.params[1],
            'iv_std_error': second_stage.bse[1],
            'naive_estimate': naive_ols.params[1],
            'first_stage_f': first_stage.fvalue,
            'first_stage_r2': first_stage.rsquared
        }

# Generate example data with confounding
n = 1000
# Instrument (e.g., policy change)
Z = np.random.normal(0, 1, n)
# Unobserved confounder
U = np.random.normal(0, 1, n)
# Endogenous variable affected by instrument and confounder
X = 0.5 * Z + 0.5 * U + np.random.normal(0, 1, n)
# Outcome affected by endogenous variable and confounder
Y = 0.7 * X + 0.3 * U + np.random.normal(0, 1, n)

# Run IV analysis
iv_analysis = InstrumentalVariablesAnalysis(Z, X, Y)
results = iv_analysis.two_stage_least_squares()

# Visualize results
plt.figure(figsize=(15, 5))
plt.subplot(131)
plt.scatter(Z, X, alpha=0.5)
plt.xlabel('Instrument (Z)')
plt.ylabel('Endogenous Variable (X)')
plt.title('First Stage Relationship')

plt.subplot(132)
plt.scatter(X, Y, alpha=0.5)
plt.xlabel('Endogenous Variable (X)')
plt.ylabel('Outcome (Y)')
plt.title('Naive Relationship')

plt.subplot(133)
Z_const = sm.add_constant(Z)
X_hat = sm.OLS(X, Z_const).fit().predict(Z_const)
plt.scatter(X_hat, Y, alpha=0.5)
plt.xlabel('Predicted X from Z')
plt.ylabel('Outcome (Y)')
plt.title('IV Relationship')

print(f"IV Estimate: {results['iv_estimate']:.3f}")
print(f"Naive Estimate: {results['naive_estimate']:.3f}")
print(f"First Stage F-statistic: {results['first_stage_f']:.3f}")
print(f"First Stage R-squared: {results['first_stage_r2']:.3f}")
```

Slide 13: Propensity Score Matching

Propensity score matching helps estimate causal effects in observational studies by matching treated and control units based on their probability of receiving treatment. This implementation demonstrates the technique with a practical example.

```python
from sklearn.linear_model import LogisticRegression

class PropensityScoreMatching:
    def __init__(self, X, treatment, outcome):
        self.X = X  # Covariates
        self.treatment = treatment  # Treatment assignment
        self.outcome = outcome  # Outcome variable
        
    def estimate_propensity_scores(self):
        # Fit logistic regression to estimate propensity scores
        model = LogisticRegression(random_state=42)
        model.fit(self.X, self.treatment)
        return model.predict_proba(self.X)[:, 1]
    
    def match_samples(self, propensity_scores, caliper=0.2):
        treated_indices = np.where(self.treatment == 1)[0]
        control_indices = np.where(self.treatment == 0)[0]
        
        # Initialize matching arrays
        matches = []
        used_control = set()
        
        for t_idx in treated_indices:
            t_score = propensity_scores[t_idx]
            
            # Find closest control unit
            min_diff = float('inf')
            best_c_idx = None
            
            for c_idx in control_indices:
                if c_idx in used_control:
                    continue
                    
                c_score = propensity_scores[c_idx]
                diff = abs(t_score - c_score)
                
                if diff < min_diff and diff < caliper:
                    min_diff = diff
                    best_c_idx = c_idx
            
            if best_c_idx is not None:
                matches.append((t_idx, best_c_idx))
                used_control.add(best_c_idx)
        
        return matches
    
    def estimate_ate(self, matches):
        treatment_effects = []
        for t_idx, c_idx in matches:
            effect = self.outcome[t_idx] - self.outcome[c_idx]
            treatment_effects.append(effect)
        
        ate = np.mean(treatment_effects)
        ate_se = np.std(treatment_effects) / np.sqrt(len(treatment_effects))
        
        return ate, ate_se

# Generate example data
n_samples = 1000
n_features = 3

# Generate covariates
X = np.random.normal(0, 1, (n_samples, n_features))

# Generate treatment assignment based on covariates
propensity = 1 / (1 + np.exp(-0.5 * X.sum(axis=1)))
treatment = np.random.binomial(1, propensity)

# Generate outcome with treatment effect
true_effect = 2
outcome = X.sum(axis=1) + true_effect * treatment + np.random.normal(0, 1, n_samples)

# Perform matching analysis
psm = PropensityScoreMatching(X, treatment, outcome)
propensity_scores = psm.estimate_propensity_scores()
matches = psm.match_samples(propensity_scores)
ate, ate_se = psm.estimate_ate(matches)

# Visualize results
plt.figure(figsize=(15, 5))

# Propensity score distributions
plt.subplot(131)
plt.hist(propensity_scores[treatment == 1], alpha=0.5, label='Treated', density=True)
plt.hist(propensity_scores[treatment == 0], alpha=0.5, label='Control', density=True)
plt.xlabel('Propensity Score')
plt.ylabel('Density')
plt.legend()
plt.title('Propensity Score Distributions')

# Outcome distributions before matching
plt.subplot(132)
plt.hist(outcome[treatment == 1], alpha=0.5, label='Treated', density=True)
plt.hist(outcome[treatment == 0], alpha=0.5, label='Control', density=True)
plt.xlabel('Outcome')
plt.ylabel('Density')
plt.legend()
plt.title('Outcome Distributions (Before Matching)')

# Matched pairs
plt.subplot(133)
matched_treated = [outcome[t] for t, c in matches]
matched_control = [outcome[c] for t, c in matches]
plt.scatter(matched_treated, matched_control, alpha=0.5)
plt.plot([min(outcome), max(outcome)], [min(outcome), max(outcome)], 'r--')
plt.xlabel('Treated Outcome')
plt.ylabel('Matched Control Outcome')
plt.title('Matched Pairs Comparison')

plt.tight_layout()
plt.show()

print(f"Estimated Average Treatment Effect: {ate:.3f} ± {1.96*ate_se:.3f}")
print(f"True Treatment Effect: {true_effect}")
```

Slide 14: Additional Resources

*   "Causal Inference in Statistics: A Primer" - ArXiv URL: [https://arxiv.org/abs/2011.02595](https://arxiv.org/abs/2011.02595)
*   "A Survey of Causal Inference Applications in Machine Learning" - ArXiv URL: [https://arxiv.org/abs/2009.11162](https://arxiv.org/abs/2009.11162)
*   "Causality in Machine Learning: A Survey and Critical Review" - ArXiv URL: [https://arxiv.org/abs/2102.02878](https://arxiv.org/abs/2102.02878)
*   "Tutorial on Causal Inference and Counterfactual Reasoning" - Search on Google Scholar for comprehensive tutorials
*   "Modern Methods for Causal Inference: A Practical Guide" - Visit [https://www.hsph.harvard.edu/miguel-hernan/causal-inference-book/](https://www.hsph.harvard.edu/miguel-hernan/causal-inference-book/) for detailed resources

