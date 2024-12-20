## Boosting Coffee Shop Sales with Statistical Power Analysis
Slide 1: Statistical Power Analysis Setup

Statistical power analysis requires careful setup of experiment parameters and hypotheses testing. We'll create a class to handle the basic calculations for the Bean of Ice coffee shop case study, including effect size, sample size, and power calculations.

```python
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

class StatisticalPowerAnalysis:
    def __init__(self, alpha=0.05, beta=0.20, effect_size=0.8):
        self.alpha = alpha  # Type I error rate
        self.beta = beta    # Type II error rate
        self.power = 1 - beta
        self.effect_size = effect_size
        
    def calculate_sample_size(self, std_dev=1):
        # Calculate required sample size for desired power
        z_alpha = stats.norm.ppf(1 - self.alpha/2)
        z_beta = stats.norm.ppf(1 - self.beta)
        n = ((z_alpha + z_beta) * std_dev / self.effect_size) ** 2
        return np.ceil(n)

# Example usage
power_analysis = StatisticalPowerAnalysis()
required_sample = power_analysis.calculate_sample_size()
print(f"Required sample size: {required_sample}")
```

Slide 2: Data Generation for Coffee Shop Sales

For the Bean of Ice case study, we'll generate synthetic daily sales data that represents both control (no campaign) and treatment (with campaign) groups using real-world-like parameters and normal distribution.

```python
def generate_sales_data(n_days, base_sales=100, campaign_effect=20, std_dev=25, seed=42):
    np.random.seed(seed)
    
    # Generate control group sales
    control_sales = np.random.normal(base_sales, std_dev, n_days)
    
    # Generate treatment group sales (with campaign effect)
    treatment_sales = np.random.normal(base_sales + campaign_effect, std_dev, n_days)
    
    return control_sales, treatment_sales

# Generate 30 days of sales data
n_days = 30
control, treatment = generate_sales_data(n_days)

print("Control group mean:", np.mean(control))
print("Treatment group mean:", np.mean(treatment))
```

Slide 3: Visualization of Sales Distribution

Creating visualizations to understand the distribution of sales data before and after the marketing campaign helps in identifying the effect size and potential overlap between distributions.

```python
def plot_sales_distributions(control_data, treatment_data):
    plt.figure(figsize=(10, 6))
    
    # Plot distributions
    plt.hist(control_data, alpha=0.5, label='No Campaign', bins=15)
    plt.hist(treatment_data, alpha=0.5, label='With Campaign', bins=15)
    
    plt.title('Distribution of Daily Sales: Control vs Treatment')
    plt.xlabel('Daily Sales ($)')
    plt.ylabel('Frequency')
    plt.legend()
    
    # Calculate and display means
    plt.axvline(np.mean(control_data), color='blue', linestyle='--', alpha=0.5)
    plt.axvline(np.mean(treatment_data), color='orange', linestyle='--', alpha=0.5)
    
    plt.show()

# Visualize the distributions
plot_sales_distributions(control, treatment)
```

Slide 4: Effect Size Calculation

The effect size calculation is crucial for determining the practical significance of the marketing campaign. We'll implement Cohen's d calculation to quantify the standardized difference between control and treatment groups.

```python
def calculate_cohens_d(control_data, treatment_data):
    # Calculate means
    mean_control = np.mean(control_data)
    mean_treatment = np.mean(treatment_data)
    
    # Calculate pooled standard deviation
    n1, n2 = len(control_data), len(treatment_data)
    var1, var2 = np.var(control_data, ddof=1), np.var(treatment_data, ddof=1)
    pooled_sd = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1 + n2 - 2))
    
    # Calculate Cohen's d
    cohens_d = (mean_treatment - mean_control) / pooled_sd
    
    return cohens_d

# Calculate effect size
effect_size = calculate_cohens_d(control, treatment)
print(f"Cohen's d effect size: {effect_size:.3f}")
```

Slide 5: Power Analysis Simulation

We'll implement a Monte Carlo simulation to estimate the statistical power of our experiment, helping Bean of Ice understand the probability of detecting a true effect of their marketing campaign.

```python
def power_simulation(n_simulations=10000, sample_size=30, 
                    effect_size=0.8, alpha=0.05):
    significant_tests = 0
    
    for _ in range(n_simulations):
        # Generate data with known effect
        control = np.random.normal(100, 25, sample_size)
        treatment = np.random.normal(100 + effect_size*25, 25, sample_size)
        
        # Perform t-test
        _, p_value = stats.ttest_ind(control, treatment)
        
        if p_value < alpha:
            significant_tests += 1
    
    return significant_tests / n_simulations

# Run simulation
estimated_power = power_simulation()
print(f"Estimated statistical power: {estimated_power:.3f}")
```

Slide 6: Hypothesis Testing Implementation

This implementation provides a comprehensive framework for conducting the hypothesis test for Bean of Ice's marketing campaign effectiveness, including both parametric and non-parametric testing options.

```python
class CampaignEffectivenessTest:
    def __init__(self, control_data, treatment_data, alpha=0.05):
        self.control = control_data
        self.treatment = treatment_data
        self.alpha = alpha
        
    def run_parametric_test(self):
        # Perform t-test
        t_stat, p_value = stats.ttest_ind(self.control, self.treatment)
        
        # Calculate confidence interval
        control_mean = np.mean(self.control)
        treatment_mean = np.mean(self.treatment)
        diff_mean = treatment_mean - control_mean
        
        se = np.sqrt(np.var(self.control)/len(self.control) + 
                    np.var(self.treatment)/len(self.treatment))
        ci = stats.t.interval(1-self.alpha, 
                            len(self.control)+len(self.treatment)-2, 
                            diff_mean, 
                            se)
        
        return {
            't_statistic': t_stat,
            'p_value': p_value,
            'confidence_interval': ci,
            'mean_difference': diff_mean
        }

# Example usage
test = CampaignEffectivenessTest(control, treatment)
results = test.run_parametric_test()
print(f"P-value: {results['p_value']:.4f}")
print(f"95% CI: {results['confidence_interval']}")
```

Slide 7: Sample Size Determination

For Bean of Ice to make informed decisions about their experiment duration, we'll implement a sample size calculator that considers various practical constraints and statistical requirements.

```python
def calculate_optimal_sample_size(desired_effect=0.8, power=0.8, 
                                alpha=0.05, ratio=1.0):
    # Calculate required sample size for two-sample t-test
    def compute_size():
        z_a = stats.norm.ppf(1 - alpha/2)
        z_b = stats.norm.ppf(power)
        
        n = ((z_a + z_b)**2 * (1 + 1/ratio)) / desired_effect**2
        return np.ceil(n)
    
    n = compute_size()
    
    # Calculate group sizes
    n1 = np.ceil(n)
    n2 = np.ceil(n * ratio)
    
    return {
        'control_size': int(n1),
        'treatment_size': int(n2),
        'total_size': int(n1 + n2)
    }

# Calculate required sample sizes
sizes = calculate_optimal_sample_size()
print("Required sample sizes:")
print(f"Control group: {sizes['control_size']}")
print(f"Treatment group: {sizes['treatment_size']}")
print(f"Total: {sizes['total_size']}")
```

Slide 8: Time Series Analysis of Sales

To account for temporal patterns in Bean of Ice's sales, we'll implement time series analysis to detect campaign effects while controlling for seasonal and trend components.

```python
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose

def analyze_sales_time_series(dates, sales_data, campaign_start_date):
    # Create time series
    ts = pd.Series(sales_data, index=pd.DatetimeIndex(dates))
    
    # Decompose series
    decomposition = seasonal_decompose(ts, period=7)  # Weekly seasonality
    
    # Calculate campaign effect
    pre_campaign = ts[ts.index < campaign_start_date].mean()
    post_campaign = ts[ts.index >= campaign_start_date].mean()
    
    return {
        'trend': decomposition.trend,
        'seasonal': decomposition.seasonal,
        'residual': decomposition.resid,
        'pre_campaign_mean': pre_campaign,
        'post_campaign_mean': post_campaign,
        'percent_change': ((post_campaign - pre_campaign) / pre_campaign) * 100
    }

# Example usage
dates = pd.date_range(start='2024-01-01', periods=len(control))
campaign_start = '2024-01-15'
results = analyze_sales_time_series(dates, treatment, campaign_start)
print(f"Sales change: {results['percent_change']:.2f}%")
```

Slide 9: ROI Calculator for Marketing Campaign

This implementation helps Bean of Ice calculate the return on investment for their marketing campaign, considering both statistical significance and financial metrics.

```python
class MarketingROICalculator:
    def __init__(self, control_sales, treatment_sales, 
                 campaign_cost, profit_margin=0.3):
        self.control = control_sales
        self.treatment = treatment_sales
        self.campaign_cost = campaign_cost
        self.profit_margin = profit_margin
        
    def calculate_roi(self):
        # Calculate average daily increase
        avg_control = np.mean(self.control)
        avg_treatment = np.mean(self.treatment)
        daily_increase = avg_treatment - avg_control
        
        # Calculate additional profit
        additional_revenue = daily_increase * len(self.treatment)
        additional_profit = additional_revenue * self.profit_margin
        
        # Calculate ROI
        roi = ((additional_profit - self.campaign_cost) / 
               self.campaign_cost) * 100
        
        return {
            'daily_increase': daily_increase,
            'additional_revenue': additional_revenue,
            'additional_profit': additional_profit,
            'roi_percentage': roi
        }

# Example usage
calculator = MarketingROICalculator(control, treatment, campaign_cost=5000)
roi_results = calculator.calculate_roi()
print(f"Campaign ROI: {roi_results['roi_percentage']:.2f}%")
```

Slide 10: Statistical Power Visualization

Creating visual representations of statistical power helps Bean of Ice understand the relationship between sample size, effect size, and the probability of detecting true campaign effects.

```python
def plot_power_curves(effect_sizes=[0.2, 0.5, 0.8], 
                     max_n=100, alpha=0.05):
    plt.figure(figsize=(12, 8))
    sample_sizes = np.arange(5, max_n, 5)
    
    for effect_size in effect_sizes:
        powers = []
        for n in sample_sizes:
            # Calculate power for each sample size
            ncp = effect_size * np.sqrt(n/2)  # Non-centrality parameter
            critical_value = stats.t.ppf(1-alpha/2, 2*n-2)
            power = 1 - stats.nct.cdf(critical_value, 2*n-2, ncp)
            powers.append(power)
        
        plt.plot(sample_sizes, powers, 
                label=f'Effect Size = {effect_size}')
    
    plt.axhline(y=0.8, color='r', linestyle='--', 
                label='Target Power (0.8)')
    plt.xlabel('Sample Size (per group)')
    plt.ylabel('Statistical Power')
    plt.title('Power Curves for Different Effect Sizes')
    plt.legend()
    plt.grid(True)
    plt.show()

# Generate power curves
plot_power_curves()
```

Slide 11: A/B Test Implementation for Sales Data

A comprehensive A/B testing framework for Bean of Ice to evaluate their marketing campaign's effectiveness with robust statistical analysis and validation.

```python
class ABTestAnalysis:
    def __init__(self, control_data, treatment_data):
        self.control = np.array(control_data)
        self.treatment = np.array(treatment_data)
        
    def compute_statistics(self):
        # Basic statistics
        stats_dict = {
            'control_mean': np.mean(self.control),
            'treatment_mean': np.mean(self.treatment),
            'control_std': np.std(self.control),
            'treatment_std': np.std(self.treatment),
            'relative_lift': ((np.mean(self.treatment) - 
                             np.mean(self.control)) / 
                             np.mean(self.control) * 100)
        }
        
        # Effect size
        pooled_std = np.sqrt(
            ((len(self.control) - 1) * stats_dict['control_std']**2 +
             (len(self.treatment) - 1) * stats_dict['treatment_std']**2) /
            (len(self.control) + len(self.treatment) - 2)
        )
        
        stats_dict['cohens_d'] = ((stats_dict['treatment_mean'] - 
                                  stats_dict['control_mean']) / 
                                 pooled_std)
        
        return stats_dict

# Example usage
ab_test = ABTestAnalysis(control, treatment)
ab_stats = ab_test.compute_statistics()
print(f"Relative lift: {ab_stats['relative_lift']:.2f}%")
print(f"Cohen's d: {ab_stats['cohens_d']:.3f}")
```

Slide 12: Bayesian Analysis for Campaign Effect

Implementing Bayesian analysis provides Bean of Ice with a more nuanced understanding of the probability that their marketing campaign had a positive effect on sales.

```python
import scipy.stats as stats
import numpy as np

class BayesianAnalysis:
    def __init__(self, control_data, treatment_data):
        self.control = control_data
        self.treatment = treatment_data
        
    def calculate_posterior_probability(self, n_samples=10000):
        # Calculate means and standard errors
        control_mean = np.mean(self.control)
        treatment_mean = np.mean(self.treatment)
        control_se = np.std(self.control) / np.sqrt(len(self.control))
        treatment_se = np.std(self.treatment) / np.sqrt(len(self.treatment))
        
        # Generate posterior samples
        control_posterior = np.random.normal(
            control_mean, control_se, n_samples)
        treatment_posterior = np.random.normal(
            treatment_mean, treatment_se, n_samples)
        
        # Calculate probability of positive effect
        diff_posterior = treatment_posterior - control_posterior
        prob_positive = np.mean(diff_posterior > 0)
        
        return {
            'prob_positive': prob_positive,
            'mean_effect': np.mean(diff_posterior),
            'ci_95': np.percentile(diff_posterior, [2.5, 97.5])
        }

# Example usage
bayesian = BayesianAnalysis(control, treatment)
results = bayesian.calculate_posterior_probability()
print(f"Probability of positive effect: {results['prob_positive']:.3f}")
print(f"95% Credible Interval: {results['ci_95']}")
```

Slide 13: Additional Resources

*   Analyzing Marketing Campaign Effectiveness:
    *   [https://arxiv.org/abs/2203.16346](https://arxiv.org/abs/2203.16346)
    *   [https://arxiv.org/abs/2105.07577](https://arxiv.org/abs/2105.07577)
    *   [https://arxiv.org/abs/1906.00195](https://arxiv.org/abs/1906.00195)
*   Power Analysis Tools and Methods:
    *   [https://www.jmir.org/2021/3/e21851](https://www.jmir.org/2021/3/e21851)
    *   [https://www.sciencedirect.com/science/article/pii/S2590238520301879](https://www.sciencedirect.com/science/article/pii/S2590238520301879)
*   Advanced A/B Testing Frameworks:
    *   Search "Bayesian A/B Testing Frameworks" on Google Scholar
    *   Visit [https://towardsdatascience.com](https://towardsdatascience.com) for practical implementations
    *   Explore [https://machinelearningmastery.com](https://machinelearningmastery.com) for detailed tutorials

