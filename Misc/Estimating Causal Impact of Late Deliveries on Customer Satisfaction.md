## Estimating Causal Impact of Late Deliveries on Customer Satisfaction
Slide 1: Understanding Propensity Score Matching Fundamentals

Propensity Score Matching (PSM) estimates treatment effects by creating matched pairs of treated and control subjects with similar characteristics. The propensity score represents the probability of receiving treatment based on observed covariates, helping reduce selection bias in observational studies.

```python
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

def calculate_propensity_scores(X, treatment):
    # Fit logistic regression to estimate propensity scores
    model = LogisticRegression(random_state=42)
    model.fit(X, treatment)
    
    # Calculate propensity scores
    propensity_scores = model.predict_proba(X)[:, 1]
    return propensity_scores
```

Slide 2: Data Preparation for Late Delivery Analysis

In our e-commerce scenario, we'll analyze how late deliveries impact customer satisfaction. We'll prepare a dataset containing customer features, delivery times, and satisfaction scores to demonstrate the practical application of PSM.

```python
# Generate synthetic e-commerce data
np.random.seed(42)
n_samples = 1000

data = {
    'customer_age': np.random.normal(35, 10, n_samples),
    'purchase_amount': np.random.normal(100, 30, n_samples),
    'distance_km': np.random.normal(50, 20, n_samples),
    'is_prime_member': np.random.binomial(1, 0.3, n_samples),
    'late_delivery': np.random.binomial(1, 0.2, n_samples)
}

df = pd.DataFrame(data)
# Simulate satisfaction scores (1-5) with lower scores for late deliveries
df['satisfaction'] = np.where(
    df['late_delivery'] == 1,
    np.random.normal(3, 0.5, n_samples),
    np.random.normal(4, 0.5, n_samples)
).clip(1, 5)
```

Slide 3: Propensity Score Estimation

The first step in PSM involves estimating propensity scores using logistic regression. We'll use customer characteristics to predict the probability of experiencing a late delivery, creating a balanced comparison between affected and unaffected customers.

```python
# Prepare features for propensity score calculation
features = ['customer_age', 'purchase_amount', 'distance_km', 'is_prime_member']
X = df[features]
treatment = df['late_delivery']

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=features)

# Calculate propensity scores
ps_scores = calculate_propensity_scores(X_scaled, treatment)
df['propensity_score'] = ps_scores

print("Propensity Score Summary:")
print(pd.DataFrame({'Propensity Scores': ps_scores}).describe())
```

Slide 4: Nearest Neighbor Matching Implementation

This implementation uses the nearest neighbor algorithm to match treated and control units based on propensity scores. We'll create a function that finds the closest match for each treated unit within a specified caliper distance.

```python
def nearest_neighbor_matching(prop_scores, treatment, caliper=0.2):
    treated_indices = np.where(treatment == 1)[0]
    control_indices = np.where(treatment == 0)[0]
    
    # Calculate standard deviation of propensity scores
    prop_scores_std = np.std(prop_scores)
    caliper_distance = caliper * prop_scores_std
    
    matches = []
    used_control = set()
    
    for t_idx in treated_indices:
        treated_score = prop_scores[t_idx]
        
        # Calculate distances to all control units
        distances = np.abs(prop_scores[control_indices] - treated_score)
        
        # Find best match within caliper
        valid_matches = control_indices[distances < caliper_distance]
        valid_distances = distances[distances < caliper_distance]
        
        if len(valid_matches) > 0:
            # Find closest unused control unit
            unused_mask = ~np.isin(valid_matches, list(used_control))
            if any(unused_mask):
                best_idx = valid_matches[unused_mask][np.argmin(valid_distances[unused_mask])]
                matches.append((t_idx, best_idx))
                used_control.add(best_idx)
    
    return matches
```

Slide 5: Implementing Matching and Balance Assessment

After matching, we need to assess the quality of our matches by comparing the distribution of covariates between treated and control groups before and after matching.

```python
def assess_balance(df, features, matches, treatment_col):
    matched_treated = [m[0] for m in matches]
    matched_control = [m[1] for m in matches]
    
    # Calculate standardized differences before and after matching
    results = []
    
    for feature in features:
        # Before matching
        treated_before = df[df[treatment_col] == 1][feature]
        control_before = df[df[treatment_col] == 0][feature]
        
        # After matching
        treated_after = df.iloc[matched_treated][feature]
        control_after = df.iloc[matched_control][feature]
        
        # Calculate standardized differences
        std_diff_before = (treated_before.mean() - control_before.mean()) / \
                         np.sqrt((treated_before.var() + control_before.var()) / 2)
        
        std_diff_after = (treated_after.mean() - control_after.mean()) / \
                        np.sqrt((treated_after.var() + control_after.var()) / 2)
        
        results.append({
            'Feature': feature,
            'Std_Diff_Before': std_diff_before,
            'Std_Diff_After': std_diff_after
        })
    
    return pd.DataFrame(results)
```

Slide 6: Average Treatment Effect Calculation

The Average Treatment Effect (ATE) represents the mean difference in outcomes between treated and control groups after matching. We'll implement a function to calculate this effect and its statistical significance.

```python
def calculate_ate(df, matches, outcome_col):
    treated_outcomes = df.iloc[[m[0] for m in matches]][outcome_col]
    control_outcomes = df.iloc[[m[1] for m in matches]][outcome_col]
    
    # Calculate average treatment effect
    ate = np.mean(treated_outcomes - control_outcomes)
    
    # Calculate standard error and confidence interval
    se = np.std(treated_outcomes - control_outcomes) / np.sqrt(len(matches))
    ci_lower = ate - 1.96 * se
    ci_upper = ate + 1.96 * se
    
    # Perform t-test
    from scipy import stats
    t_stat, p_value = stats.ttest_ind(treated_outcomes, control_outcomes)
    
    return {
        'ATE': ate,
        'SE': se,
        'CI_Lower': ci_lower,
        'CI_Upper': ci_upper,
        'P_Value': p_value
    }
```

Slide 7: Implementation of Full PSM Analysis Pipeline

The complete pipeline combines all previous components into a cohesive analysis workflow. This implementation demonstrates the end-to-end process of propensity score matching for analyzing late delivery impacts on customer satisfaction.

```python
def run_psm_analysis(df, features, treatment_col, outcome_col, caliper=0.2):
    # 1. Prepare data
    X = df[features]
    treatment = df[treatment_col]
    
    # 2. Calculate propensity scores
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    ps_scores = calculate_propensity_scores(pd.DataFrame(X_scaled, columns=features), treatment)
    
    # 3. Perform matching
    matches = nearest_neighbor_matching(ps_scores, treatment, caliper)
    
    # 4. Assess balance
    balance_stats = assess_balance(df, features, matches, treatment_col)
    
    # 5. Calculate treatment effect
    treatment_effect = calculate_ate(df, matches, outcome_col)
    
    return {
        'matches': matches,
        'balance_stats': balance_stats,
        'treatment_effect': treatment_effect,
        'propensity_scores': ps_scores
    }
```

Slide 8: Real-world Example: E-commerce Late Delivery Analysis

We'll analyze the impact of late deliveries on customer satisfaction using our synthetic e-commerce dataset. This example demonstrates the practical application of PSM in a business context.

```python
# Run complete PSM analysis
features = ['customer_age', 'purchase_amount', 'distance_km', 'is_prime_member']
results = run_psm_analysis(
    df=df,
    features=features,
    treatment_col='late_delivery',
    outcome_col='satisfaction',
    caliper=0.2
)

# Display results
print("Number of matched pairs:", len(results['matches']))
print("\nBalance Statistics:")
print(results['balance_stats'])
print("\nTreatment Effect Results:")
print(pd.DataFrame([results['treatment_effect']]))
```

Slide 9: Visualization of Matching Quality

Understanding the quality of matches is crucial for validating PSM results. We'll create visualizations to compare propensity score distributions and covariate balance before and after matching.

```python
import matplotlib.pyplot as plt
import seaborn as sns

def plot_matching_diagnostics(df, ps_scores, matches, features):
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    
    # Plot propensity score distributions
    treated_ps = ps_scores[df['late_delivery'] == 1]
    control_ps = ps_scores[df['late_delivery'] == 0]
    
    sns.kdeplot(data=treated_ps, ax=ax1, label='Treated')
    sns.kdeplot(data=control_ps, ax=ax1, label='Control')
    ax1.set_title('Propensity Score Distributions')
    ax1.set_xlabel('Propensity Score')
    ax1.legend()
    
    # Plot standardized differences
    balance_stats = assess_balance(df, features, matches, 'late_delivery')
    
    balance_stats.plot(
        x='Feature',
        y=['Std_Diff_Before', 'Std_Diff_After'],
        kind='bar',
        ax=ax2
    )
    ax2.set_title('Standardized Differences Before and After Matching')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_xticklabels(balance_stats['Feature'], rotation=45)
    
    plt.tight_layout()
    plt.show()
```

Slide 10: Sensitivity Analysis Implementation

We'll implement Rosenbaum bounds sensitivity analysis to assess how robust our findings are to potential unmeasured confounding variables.

```python
def sensitivity_analysis(df, matches, outcome_col, gamma_range=np.arange(1, 2.1, 0.1)):
    results = []
    
    for gamma in gamma_range:
        # Calculate Wilcoxon signed-rank test statistic
        treated_outcomes = df.iloc[[m[0] for m in matches]][outcome_col]
        control_outcomes = df.iloc[[m[1] for m in matches]][outcome_col]
        
        differences = treated_outcomes.values - control_outcomes.values
        abs_diff = np.abs(differences)
        ranks = stats.rankdata(abs_diff)
        
        # Calculate bounds
        p_upper = 1 / (1 + 1/gamma)
        p_lower = 1 / (1 + gamma)
        
        # Store results
        results.append({
            'Gamma': gamma,
            'P_Lower': p_lower,
            'P_Upper': p_upper
        })
    
    return pd.DataFrame(results)

# Example usage
sensitivity_results = sensitivity_analysis(df, results['matches'], 'satisfaction')
print("Sensitivity Analysis Results:")
print(sensitivity_results)
```

Slide 11: Bootstrapping PSM for Uncertainty Estimation

Bootstrapping allows us to estimate the uncertainty in our treatment effect estimates by resampling our matched pairs. This implementation provides confidence intervals and distribution of treatment effects.

```python
def bootstrap_psm(df, features, treatment_col, outcome_col, n_iterations=1000):
    treatment_effects = []
    
    for _ in range(n_iterations):
        # Sample with replacement
        bootstrap_indices = np.random.choice(
            len(df), size=len(df), replace=True
        )
        bootstrap_df = df.iloc[bootstrap_indices].reset_index(drop=True)
        
        # Run PSM on bootstrap sample
        results = run_psm_analysis(
            bootstrap_df, 
            features, 
            treatment_col, 
            outcome_col
        )
        
        treatment_effects.append(results['treatment_effect']['ATE'])
    
    # Calculate confidence intervals
    ci_lower = np.percentile(treatment_effects, 2.5)
    ci_upper = np.percentile(treatment_effects, 97.5)
    
    return {
        'mean_effect': np.mean(treatment_effects),
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'effects_distribution': treatment_effects
    }
```

Slide 12: Advanced Matching Strategies

This implementation extends basic PSM with optimal matching and genetic matching algorithms to improve covariate balance and treatment effect estimation accuracy.

```python
def optimal_matching(prop_scores, treatment, max_ratio=1):
    from scipy.optimize import linear_sum_assignment
    
    treated_indices = np.where(treatment == 1)[0]
    control_indices = np.where(treatment == 0)[0]
    
    # Create cost matrix
    cost_matrix = np.zeros((len(treated_indices), len(control_indices)))
    for i, t_idx in enumerate(treated_indices):
        for j, c_idx in enumerate(control_indices):
            cost_matrix[i, j] = abs(prop_scores[t_idx] - prop_scores[c_idx])
    
    # Solve assignment problem
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    # Create matches
    matches = list(zip(treated_indices[row_ind], control_indices[col_ind]))
    
    return matches
```

Slide 13: Results and Statistical Inference

Implementation of comprehensive statistical testing and inference procedures for PSM results, including multiple hypothesis testing corrections and effect size calculations.

```python
def statistical_inference(df, matches, outcome_col, covariates):
    from scipy import stats
    import statsmodels.api as sm
    
    # Extract matched samples
    treated_indices = [m[0] for m in matches]
    control_indices = [m[1] for m in matches]
    
    # Prepare data for regression
    X = df.iloc[treated_indices + control_indices][covariates]
    y = df.iloc[treated_indices + control_indices][outcome_col]
    treatment = np.concatenate([
        np.ones(len(treated_indices)),
        np.zeros(len(control_indices))
    ])
    
    # Add treatment indicator to covariates
    X = sm.add_constant(X)
    X = pd.concat([X, pd.Series(treatment, name='treatment')], axis=1)
    
    # Fit regression model
    model = sm.OLS(y, X).fit()
    
    # Calculate effect size (Cohen's d)
    treated_outcomes = df.iloc[treated_indices][outcome_col]
    control_outcomes = df.iloc[control_indices][outcome_col]
    
    pooled_std = np.sqrt(
        (np.var(treated_outcomes) + np.var(control_outcomes)) / 2
    )
    cohens_d = (np.mean(treated_outcomes) - np.mean(control_outcomes)) / pooled_std
    
    return {
        'regression_results': model.summary(),
        'effect_size': cohens_d,
        'treatment_coef': model.params['treatment'],
        'p_value': model.pvalues['treatment']
    }
```

Slide 14: Additional Resources

*   Matching Methods for Causal Inference: A Review and a Look Forward [https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2943670/](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2943670/)
*   An Introduction to Propensity Score Methods for Reducing the Effects of Confounding in Observational Studies [https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3144483/](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3144483/)
*   Guidelines for Conducting Propensity Score Matched Analyses [https://www.tandfonline.com/doi/full/10.1080/00273171.2011.568786](https://www.tandfonline.com/doi/full/10.1080/00273171.2011.568786)
*   Tutorial on Causal Inference and Propensity Score Methods Search on Google Scholar with keywords: "Tutorial Propensity Score Matching Methodology"
*   Advanced Topics in Propensity Score Methods Search on Google Scholar with keywords: "Advanced Propensity Score Matching Techniques"

