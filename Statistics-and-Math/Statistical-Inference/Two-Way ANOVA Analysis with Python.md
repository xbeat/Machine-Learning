## Two-Way ANOVA Analysis with Python
Slide 1: Introduction to Two-Way ANOVA

Two-way ANOVA extends one-way ANOVA by examining the influence of two independent categorical variables on a dependent variable. It evaluates main effects of each factor and their interaction effect, providing deeper insights into complex relationships within the data.

```python
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.anova import anova_lm
from statsmodels.formula.api import ols

# Mathematical representation of Two-Way ANOVA model
# $$Y_{ijk} = \mu + \alpha_i + \beta_j + (\alpha\beta)_{ij} + \epsilon_{ijk}$$
```

Slide 2: Data Preparation and Assumptions

Two-way ANOVA requires careful data preparation and validation of key assumptions: normality, homogeneity of variances, and independence of observations. We'll create a synthetic marketing dataset to demonstrate these concepts and perform necessary checks.

```python
# Generate synthetic marketing data
np.random.seed(42)
n_samples = 200

# Create factors
channels = np.random.choice(['Email', 'Social', 'Search'], n_samples)
locations = np.random.choice(['Urban', 'Suburban', 'Rural'], n_samples)
conversion_rates = np.random.normal(0.15, 0.05, n_samples)

# Create DataFrame
data = pd.DataFrame({
    'Channel': channels,
    'Location': locations,
    'Conversion_Rate': conversion_rates
})

# Check normality using Shapiro-Wilk test
stat, p_value = stats.shapiro(data['Conversion_Rate'])
print(f"Shapiro-Wilk test: statistic={stat:.4f}, p-value={p_value:.4f}")
```

Slide 3: Exploratory Data Analysis

Before conducting ANOVA, it's crucial to understand data distribution and relationships between variables. We'll create visualizations to examine the distribution of conversion rates across different marketing channels and locations.

```python
# Create visualization
plt.figure(figsize=(12, 6))
sns.boxplot(x='Channel', y='Conversion_Rate', hue='Location', data=data)
plt.title('Conversion Rates by Channel and Location')
plt.xlabel('Marketing Channel')
plt.ylabel('Conversion Rate')
plt.show()

# Calculate summary statistics
summary_stats = data.groupby(['Channel', 'Location'])['Conversion_Rate'].agg(['mean', 'std'])
print("\nSummary Statistics:")
print(summary_stats)
```

Slide 4: Two-Way ANOVA Implementation

The implementation of two-way ANOVA involves fitting a linear model and calculating the F-statistics for main effects and interaction. We use statsmodels to perform the analysis and obtain comprehensive results.

```python
# Fit the two-way ANOVA model
model = ols('Conversion_Rate ~ C(Channel) + C(Location) + C(Channel):C(Location)', data=data).fit()

# Perform Type II ANOVA
anova_table = anova_lm(model, typ=2)

# Display results
print("Two-Way ANOVA Results:")
print(anova_table)

# Calculate effect sizes (Partial Eta-Squared)
def partial_eta_squared(aov):
    aov['pes'] = aov['sum_sq'] / (aov['sum_sq'] + aov['sum_sq'].sum())
    return aov

anova_table_with_pes = partial_eta_squared(anova_table)
print("\nEffect Sizes (Partial Eta-Squared):")
print(anova_table_with_pes['pes'])
```

Slide 5: Post-Hoc Analysis

Following significant ANOVA results, post-hoc tests help identify specific group differences. We'll implement Tukey's HSD test to perform pairwise comparisons while controlling for multiple testing.

```python
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Perform Tukey's HSD test for Channel
tukey_channel = pairwise_tukeyhsd(data['Conversion_Rate'], 
                                 data['Channel'],
                                 alpha=0.05)

# Perform Tukey's HSD test for Location
tukey_location = pairwise_tukeyhsd(data['Conversion_Rate'], 
                                  data['Location'],
                                  alpha=0.05)

print("Tukey's HSD Results for Channel:")
print(tukey_channel)
print("\nTukey's HSD Results for Location:")
print(tukey_location)
```

Slide 6: Interaction Effects Analysis

Interaction effects reveal how the impact of one factor depends on the level of another factor. Understanding these relationships is crucial for marketing strategy optimization and decision-making processes in multi-channel campaigns.

```python
# Create interaction plot
plt.figure(figsize=(12, 6))
means = data.groupby(['Channel', 'Location'])['Conversion_Rate'].mean().unstack()

# Plot interactions with error bars
for location in means.columns:
    plt.plot(means.index, means[location], marker='o', label=location)
    
plt.title('Interaction Plot: Channel x Location')
plt.xlabel('Marketing Channel')
plt.ylabel('Mean Conversion Rate')
plt.legend(title='Location')
plt.grid(True)
plt.show()
```

Slide 7: Effect Size Calculations

Effect size measurements quantify the strength of relationships between variables, providing practical significance beyond statistical significance. We'll calculate and interpret multiple effect size metrics.

```python
def calculate_effect_sizes(anova_table):
    # Calculate Partial Eta Squared
    anova_table['partial_eta_sq'] = (anova_table['sum_sq'] / 
                                    (anova_table['sum_sq'] + 
                                     anova_table['sum_sq'].sum()))
    
    # Calculate Cohen's f
    anova_table['cohens_f'] = np.sqrt(anova_table['partial_eta_sq'] / 
                                     (1 - anova_table['partial_eta_sq']))
    
    return anova_table

# Calculate effect sizes
effects = calculate_effect_sizes(anova_table.copy())
print("Effect Size Results:")
print(effects[['partial_eta_sq', 'cohens_f']])
```

Slide 8: Model Diagnostics

Comprehensive model diagnostics ensure the validity of our ANOVA results. We'll examine residuals for normality, homoscedasticity, and independence assumptions through visual and statistical methods.

```python
# Get model residuals
residuals = model.resid
fitted_values = model.fittedvalues

# Create diagnostic plots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Q-Q plot
stats.probplot(residuals, dist="norm", plot=axes[0,0])
axes[0,0].set_title("Q-Q Plot")

# Residuals vs Fitted
axes[0,1].scatter(fitted_values, residuals)
axes[0,1].axhline(y=0, color='r', linestyle='--')
axes[0,1].set_xlabel("Fitted values")
axes[0,1].set_ylabel("Residuals")
axes[0,1].set_title("Residuals vs Fitted")

# Residuals distribution
axes[1,0].hist(residuals, bins=30)
axes[1,0].set_title("Residuals Distribution")

# Scale-Location plot
axes[1,1].scatter(fitted_values, np.sqrt(np.abs(residuals)))
axes[1,1].set_xlabel("Fitted values")
axes[1,1].set_ylabel("Sqrt(|Residuals|)")
axes[1,1].set_title("Scale-Location Plot")

plt.tight_layout()
plt.show()
```

Slide 9: Real-world Example: E-commerce Campaign Analysis

Implementation of two-way ANOVA on an e-commerce dataset analyzing the effects of advertisement placement and time of day on click-through rates, demonstrating practical application in digital marketing.

```python
# Generate realistic e-commerce data
np.random.seed(42)
n_samples = 300

# Create factors
ad_placement = np.random.choice(['Header', 'Sidebar', 'Footer'], n_samples)
time_of_day = np.random.choice(['Morning', 'Afternoon', 'Evening'], n_samples)

# Generate click-through rates with realistic interactions
base_ctr = 0.05
placement_effect = {'Header': 0.02, 'Sidebar': 0.01, 'Footer': 0}
time_effect = {'Morning': 0.01, 'Afternoon': 0.02, 'Evening': 0.015}

ctr = np.array([base_ctr + placement_effect[p] + time_effect[t] + 
                np.random.normal(0, 0.01) for p, t in zip(ad_placement, time_of_day)])

# Create DataFrame
ecommerce_data = pd.DataFrame({
    'Ad_Placement': ad_placement,
    'Time_of_Day': time_of_day,
    'CTR': ctr
})

# Fit ANOVA model
ecommerce_model = ols('CTR ~ C(Ad_Placement) + C(Time_of_Day) + C(Ad_Placement):C(Time_of_Day)', 
                     data=ecommerce_data).fit()
print(anova_lm(ecommerce_model, typ=2))
```

Slide 10: Power Analysis and Sample Size Determination

Understanding statistical power and determining appropriate sample sizes are crucial for robust ANOVA analyses. We'll implement power calculations and sample size estimation methods.

```python
from statsmodels.stats.power import FTestAnovaPower
import matplotlib.pyplot as plt

# Power analysis function
def power_analysis(effect_sizes, n_groups, alpha=0.05):
    power_calculator = FTestAnovaPower()
    sample_sizes = np.arange(5, 100, 5)
    powers = []
    
    for effect_size in effect_sizes:
        power_per_size = [power_calculator.power(
            effect_size=effect_size,
            nobs=n * n_groups,
            alpha=alpha
        ) for n in sample_sizes]
        powers.append(power_per_size)
    
    return sample_sizes, powers

# Calculate power for different effect sizes
effect_sizes = [0.1, 0.25, 0.4]  # Small, medium, large effects
n_groups = 9  # 3x3 design
sample_sizes, powers = power_analysis(effect_sizes, n_groups)

# Plot power analysis results
plt.figure(figsize=(10, 6))
for i, effect_size in enumerate(effect_sizes):
    plt.plot(sample_sizes, powers[i], 
             label=f'Effect size = {effect_size}')

plt.axhline(y=0.8, color='r', linestyle='--', label='0.8 power threshold')
plt.xlabel('Sample size per group')
plt.ylabel('Statistical Power')
plt.title('Power Analysis for Two-Way ANOVA')
plt.legend()
plt.grid(True)
plt.show()
```

Slide 11: Handling Unbalanced Designs

Unbalanced designs occur when group sizes are unequal, requiring special consideration in ANOVA calculations. We'll implement methods to handle unbalanced data and compare different types of sums of squares.

```python
# Generate unbalanced dataset
np.random.seed(42)
n_samples = [80, 120, 100]  # Unequal group sizes

# Create unbalanced data
channels = np.concatenate([np.repeat(['Email', 'Social', 'Search'], n) 
                          for n in n_samples])
locations = np.random.choice(['Urban', 'Suburban', 'Rural'], 
                           sum(n_samples))
conversion_rates = np.random.normal(0.15, 0.05, sum(n_samples))

unbalanced_data = pd.DataFrame({
    'Channel': channels,
    'Location': locations,
    'Conversion_Rate': conversion_rates
})

# Compare Type I, II, and III SS
types = [1, 2, 3]
results = {}

for typ in types:
    model = ols('Conversion_Rate ~ C(Channel) + C(Location) + C(Channel):C(Location)', 
                data=unbalanced_data).fit()
    results[f'Type {typ}'] = anova_lm(model, typ=typ)
    print(f"\nType {typ} SS Results:")
    print(results[f'Type {typ}'])
```

Slide 12: Robust ANOVA Alternatives

When ANOVA assumptions are violated, robust alternatives provide more reliable results. We'll implement non-parametric and robust ANOVA variations for comparison.

```python
from scipy.stats import kruskal
from scipy.stats import friedmanchisquare

# Kruskal-Wallis H-test (non-parametric alternative)
def robust_analysis(data):
    # Kruskal-Wallis for Channel effect
    channel_groups = [group['Conversion_Rate'].values 
                     for name, group in data.groupby('Channel')]
    kw_stat, kw_p = kruskal(*channel_groups)
    
    # Aligned Ranks Transform ANOVA (ART)
    from scipy.stats import rankdata
    
    def aligned_ranks(df, dv, between):
        # Calculate cell means
        cell_means = df.groupby(between)[dv].transform('mean')
        # Calculate grand mean
        grand_mean = df[dv].mean()
        # Align the data
        aligned_data = df[dv] - cell_means + grand_mean
        # Rank the aligned data
        ranked_data = rankdata(aligned_data)
        return ranked_data
    
    # Perform ART
    data['ranked_dv'] = aligned_ranks(data, 'Conversion_Rate', 'Channel')
    
    # Fit ANOVA on ranked data
    robust_model = ols('ranked_dv ~ C(Channel) + C(Location) + C(Channel):C(Location)', 
                      data=data).fit()
    
    return {
        'Kruskal-Wallis': {'statistic': kw_stat, 'p-value': kw_p},
        'ART-ANOVA': anova_lm(robust_model, typ=2)
    }

# Run robust analysis
robust_results = robust_analysis(unbalanced_data)
print("\nRobust Analysis Results:")
print("\nKruskal-Wallis Test:")
print(robust_results['Kruskal-Wallis'])
print("\nART-ANOVA Results:")
print(robust_results['ART-ANOVA'])
```

Slide 13: Additional Resources

*   A Comprehensive Guide to Two-Way ANOVA in Python [https://arxiv.org/abs/2108.13005](https://arxiv.org/abs/2108.13005)
*   Statistical Power Analysis for the Behavioral Sciences: Guidelines and Best Practices [https://arxiv.org/abs/1912.08878](https://arxiv.org/abs/1912.08878)
*   Robust Alternatives to Traditional ANOVA Methods [https://arxiv.org/abs/2003.09742](https://arxiv.org/abs/2003.09742)
*   Modern Methods for Handling Missing Data in ANOVA Designs [https://www.google.com/search?q=modern+methods+for+handling+missing+data+in+anova+designs](https://www.google.com/search?q=modern+methods+for+handling+missing+data+in+anova+designs)
*   Recent Advances in Non-parametric ANOVA Techniques [https://www.google.com/search?q=recent+advances+in+non-parametric+anova+techniques](https://www.google.com/search?q=recent+advances+in+non-parametric+anova+techniques)

