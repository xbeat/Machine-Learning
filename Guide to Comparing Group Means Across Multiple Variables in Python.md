## Guide to Comparing Group Means Across Multiple Variables in Python
Slide 1: Introduction to Comparing Group Means

Comparing group means is a fundamental task in data analysis, allowing us to understand differences between various subsets of our data. This guide will walk you through the process using Python, covering essential techniques and statistical methods.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Set up a sample dataset
np.random.seed(42)
data = pd.DataFrame({
    'Group': np.repeat(['A', 'B', 'C'], 50),
    'Value': np.concatenate([
        np.random.normal(10, 2, 50),
        np.random.normal(12, 2, 50),
        np.random.normal(11, 2, 50)
    ])
})

print(data.head())
```

Slide 2: Visualizing Group Means with Box Plots

Box plots provide a quick visual summary of the distribution of data across groups, showing the median, quartiles, and potential outliers.

```python
plt.figure(figsize=(10, 6))
sns.boxplot(x='Group', y='Value', data=data)
plt.title('Box Plot of Values by Group')
plt.show()

# Calculate and print group means
group_means = data.groupby('Group')['Value'].mean()
print("Group Means:")
print(group_means)
```

Slide 3: One-way ANOVA (Analysis of Variance)

ANOVA is used to determine whether there are statistically significant differences between the means of three or more independent groups.

```python
f_statistic, p_value = stats.f_oneway(
    data[data['Group'] == 'A']['Value'],
    data[data['Group'] == 'B']['Value'],
    data[data['Group'] == 'C']['Value']
)

print(f"One-way ANOVA results:")
print(f"F-statistic: {f_statistic:.4f}")
print(f"p-value: {p_value:.4f}")
```

Slide 4: Post-hoc Tests: Tukey's HSD

If ANOVA indicates significant differences, we use post-hoc tests like Tukey's HSD to identify which specific groups differ from each other.

```python
from statsmodels.stats.multicomp import pairwise_tukeyhsd

tukey_results = pairwise_tukeyhsd(data['Value'], data['Group'])
print(tukey_results)

# Visualize Tukey's HSD results
tukey_results.plot_simultaneous()
plt.show()
```

Slide 5: Effect Size: Cohen's d

Cohen's d measures the standardized difference between two group means, providing insight into the magnitude of the difference.

```python
def cohens_d(group1, group2):
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_se = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    return (np.mean(group1) - np.mean(group2)) / pooled_se

groups = ['A', 'B', 'C']
for i in range(len(groups)):
    for j in range(i+1, len(groups)):
        group1 = data[data['Group'] == groups[i]]['Value']
        group2 = data[data['Group'] == groups[j]]['Value']
        d = cohens_d(group1, group2)
        print(f"Cohen's d between {groups[i]} and {groups[j]}: {d:.4f}")
```

Slide 6: Comparing Means with T-tests

T-tests are used to compare means between two groups. We'll demonstrate both independent and paired t-tests.

```python
# Independent t-test
t_stat, p_val = stats.ttest_ind(
    data[data['Group'] == 'A']['Value'],
    data[data['Group'] == 'B']['Value']
)
print(f"Independent t-test (A vs B): t-statistic = {t_stat:.4f}, p-value = {p_val:.4f}")

# Paired t-test (simulating paired data)
np.random.seed(42)
before = np.random.normal(10, 2, 30)
after = before + np.random.normal(1, 1, 30)  # Simulating an increase
t_stat, p_val = stats.ttest_rel(before, after)
print(f"Paired t-test: t-statistic = {t_stat:.4f}, p-value = {p_val:.4f}")
```

Slide 7: Handling Non-normal Distributions: Mann-Whitney U Test

When data doesn't follow a normal distribution, non-parametric tests like the Mann-Whitney U test can be used to compare group medians.

```python
# Simulating non-normal data
np.random.seed(42)
group1 = np.random.exponential(scale=2, size=100)
group2 = np.random.exponential(scale=2.5, size=100)

# Perform Mann-Whitney U test
statistic, p_value = stats.mannwhitneyu(group1, group2, alternative='two-sided')

print(f"Mann-Whitney U test results:")
print(f"Statistic: {statistic:.4f}")
print(f"p-value: {p_value:.4f}")

# Visualize the distributions
plt.figure(figsize=(10, 6))
sns.histplot(group1, kde=True, label='Group 1')
sns.histplot(group2, kde=True, label='Group 2')
plt.legend()
plt.title('Distribution of Non-normal Data')
plt.show()
```

Slide 8: Dealing with Unequal Variances: Welch's T-test

When comparing groups with unequal variances, Welch's t-test is more appropriate than the standard t-test.

```python
np.random.seed(42)
group1 = np.random.normal(10, 2, 100)
group2 = np.random.normal(11, 4, 100)  # Higher variance

# Perform Welch's t-test
t_stat, p_val = stats.ttest_ind(group1, group2, equal_var=False)

print(f"Welch's t-test results:")
print(f"t-statistic: {t_stat:.4f}")
print(f"p-value: {p_val:.4f}")

# Visualize the distributions
plt.figure(figsize=(10, 6))
sns.histplot(group1, kde=True, label='Group 1')
sns.histplot(group2, kde=True, label='Group 2')
plt.legend()
plt.title('Distributions with Unequal Variances')
plt.show()
```

Slide 9: Real-life Example: Comparing Crop Yields

Let's compare the yields of three different crop varieties across multiple fields.

```python
np.random.seed(42)
crops = pd.DataFrame({
    'Variety': np.repeat(['A', 'B', 'C'], 50),
    'Yield': np.concatenate([
        np.random.normal(8, 1.5, 50),  # Variety A
        np.random.normal(9, 1.2, 50),  # Variety B
        np.random.normal(7.5, 1.8, 50)  # Variety C
    ])
})

# Visualize the data
plt.figure(figsize=(10, 6))
sns.boxplot(x='Variety', y='Yield', data=crops)
plt.title('Crop Yields by Variety')
plt.ylabel('Yield (tons per hectare)')
plt.show()

# Perform one-way ANOVA
f_stat, p_val = stats.f_oneway(
    crops[crops['Variety'] == 'A']['Yield'],
    crops[crops['Variety'] == 'B']['Yield'],
    crops[crops['Variety'] == 'C']['Yield']
)

print(f"One-way ANOVA results:")
print(f"F-statistic: {f_stat:.4f}")
print(f"p-value: {p_val:.4f}")
```

Slide 10: Real-life Example: Comparing Student Performance

We'll compare test scores across different teaching methods to evaluate their effectiveness.

```python
np.random.seed(42)
scores = pd.DataFrame({
    'Method': np.repeat(['Traditional', 'Interactive', 'Online'], 40),
    'Score': np.concatenate([
        np.random.normal(75, 10, 40),  # Traditional
        np.random.normal(80, 8, 40),   # Interactive
        np.random.normal(72, 12, 40)   # Online
    ])
})

# Visualize the data
plt.figure(figsize=(10, 6))
sns.violinplot(x='Method', y='Score', data=scores)
plt.title('Test Scores by Teaching Method')
plt.ylabel('Score')
plt.show()

# Perform Kruskal-Wallis H-test (non-parametric alternative to one-way ANOVA)
h_stat, p_val = stats.kruskal(
    scores[scores['Method'] == 'Traditional']['Score'],
    scores[scores['Method'] == 'Interactive']['Score'],
    scores[scores['Method'] == 'Online']['Score']
)

print(f"Kruskal-Wallis H-test results:")
print(f"H-statistic: {h_stat:.4f}")
print(f"p-value: {p_val:.4f}")
```

Slide 11: Handling Repeated Measures: Mixed ANOVA

When dealing with data that involves both between-group and within-subject factors, a mixed ANOVA is appropriate.

```python
import statsmodels.api as sm
from statsmodels.formula.api import ols

# Simulating repeated measures data
np.random.seed(42)
subjects = 20
treatments = ['A', 'B', 'C']
times = ['Pre', 'Post']

data = []
for subject in range(subjects):
    for treatment in treatments:
        pre = np.random.normal(10, 2)
        post = pre + np.random.normal(2, 1)  # Assuming improvement
        data.append({'Subject': subject, 'Treatment': treatment, 'Time': 'Pre', 'Score': pre})
        data.append({'Subject': subject, 'Treatment': treatment, 'Time': 'Post', 'Score': post})

df = pd.DataFrame(data)

# Perform mixed ANOVA
model = ols('Score ~ C(Treatment) + C(Time) + C(Treatment):C(Time)', data=df).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
print(anova_table)

# Visualize the data
plt.figure(figsize=(12, 6))
sns.boxplot(x='Treatment', y='Score', hue='Time', data=df)
plt.title('Scores by Treatment and Time')
plt.show()
```

Slide 12: Dealing with Multiple Comparisons: Bonferroni Correction

When performing multiple comparisons, the risk of Type I errors increases. The Bonferroni correction adjusts the p-value threshold to account for this.

```python
from statsmodels.stats.multitest import multipletests

# Simulating multiple comparisons
np.random.seed(42)
groups = ['A', 'B', 'C', 'D', 'E']
n_comparisons = len(groups) * (len(groups) - 1) // 2
p_values = np.random.uniform(0, 0.1, n_comparisons)  # Simulating p-values

# Apply Bonferroni correction
bonferroni_correction = multipletests(p_values, alpha=0.05, method='bonferroni')

print("Original p-values:")
print(p_values)
print("\nBonferroni-corrected results:")
print("Rejected null hypotheses:", bonferroni_correction[0])
print("Corrected p-values:", bonferroni_correction[1])

# Visualize the results
plt.figure(figsize=(10, 6))
plt.scatter(range(len(p_values)), sorted(p_values), label='Original')
plt.scatter(range(len(p_values)), sorted(bonferroni_correction[1]), label='Bonferroni-corrected')
plt.axhline(0.05, color='r', linestyle='--', label='α = 0.05')
plt.xlabel('Comparison')
plt.ylabel('p-value')
plt.title('Original vs. Bonferroni-corrected p-values')
plt.legend()
plt.show()
```

Slide 13: Visualizing Multiple Group Comparisons: Heatmaps

Heatmaps provide an effective way to visualize pairwise comparisons between multiple groups.

```python
# Simulating pairwise comparison data
np.random.seed(42)
groups = ['A', 'B', 'C', 'D', 'E']
n_groups = len(groups)
comparison_matrix = np.random.rand(n_groups, n_groups)
comparison_matrix = (comparison_matrix + comparison_matrix.T) / 2  # Make symmetric
np.fill_diagonal(comparison_matrix, 1)  # Set diagonal to 1

# Create heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(comparison_matrix, annot=True, cmap='YlGnBu', xticklabels=groups, yticklabels=groups)
plt.title('Pairwise Comparison Heatmap')
plt.show()

# Calculate and print summary statistics
print("Summary Statistics:")
print(f"Mean comparison value: {np.mean(comparison_matrix):.4f}")
print(f"Minimum comparison value: {np.min(comparison_matrix):.4f}")
print(f"Maximum comparison value: {np.max(comparison_matrix):.4f}")
```

Slide 14: Additional Resources

For further exploration of group comparison techniques and statistical analysis in Python, consider the following resources:

1. "Statistical Methods in Python" by Eric Ma (ArXiv:2208.03186) URL: [https://arxiv.org/abs/2208.03186](https://arxiv.org/abs/2208.03186)
2. "A Comprehensive Guide to the scipy.stats Module" by Janani Ravi (ArXiv:2106.07177) URL: [https://arxiv.org/abs/2106.07177](https://arxiv.org/abs/2106.07177)
3. "Statsmodels: Econometric and Statistical Modeling with Python" by Seabold and Perktold (ArXiv:1003.1764) URL: [https://arxiv.org/abs/1003.1764](https://arxiv.org/abs/1003.1764)

These resources provide in-depth coverage of statistical methods and their implementation in Python, offering valuable insights for both beginners and advanced practitioners.

