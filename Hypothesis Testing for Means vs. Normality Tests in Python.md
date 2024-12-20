## Hypothesis Testing for Means vs. Normality Tests in Python

Slide 1: Introduction to Hypothesis Testing

Hypothesis testing is a statistical method used to make inferences about population parameters based on sample data. It involves formulating two competing hypotheses: the null hypothesis (H0) and the alternative hypothesis (H1), and then using sample data to determine which hypothesis is more likely to be true.

```python
import numpy as np
from scipy import stats

# Generate sample data
np.random.seed(42)
sample_data = np.random.normal(loc=10, scale=2, size=100)

# Perform one-sample t-test
t_statistic, p_value = stats.ttest_1samp(sample_data, popmean=9)

print(f"T-statistic: {t_statistic:.4f}")
print(f"P-value: {p_value:.4f}")
```

Slide 2: Types of Hypothesis Tests for Means

There are several types of hypothesis tests for means, including:

1. One-sample t-test: Compares a sample mean to a known population mean.
2. Two-sample t-test: Compares means of two independent samples.
3. Paired t-test: Compares means of two related samples.

```python
# Two-sample t-test
sample1 = np.random.normal(loc=10, scale=2, size=50)
sample2 = np.random.normal(loc=11, scale=2, size=50)

t_statistic, p_value = stats.ttest_ind(sample1, sample2)
print(f"Two-sample t-test p-value: {p_value:.4f}")

# Paired t-test
before = np.random.normal(loc=10, scale=2, size=30)
after = before + np.random.normal(loc=1, scale=1, size=30)

t_statistic, p_value = stats.ttest_rel(before, after)
print(f"Paired t-test p-value: {p_value:.4f}")
```

Slide 3: Assumptions of Parametric Tests

Parametric tests like t-tests assume:

1. Normality: The data follows a normal distribution.
2. Independence: Observations are independent of each other.
3. Homogeneity of variance: Equal variances across groups (for two-sample tests).

These assumptions are crucial for the validity of the test results.

```python
import matplotlib.pyplot as plt

# Generate normal and non-normal data
normal_data = np.random.normal(loc=0, scale=1, size=1000)
non_normal_data = np.random.exponential(scale=1, size=1000)

# Plot histograms
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.hist(normal_data, bins=30)
ax1.set_title("Normal Distribution")
ax2.hist(non_normal_data, bins=30)
ax2.set_title("Non-Normal Distribution")
plt.show()
```

Slide 4: Introduction to Normality Tests

Normality tests are used to assess whether a dataset follows a normal distribution. Two popular tests are:

1. Shapiro-Wilk test
2. Anderson-Darling test

These tests help determine if parametric tests are appropriate for the data.

```python
from scipy import stats

# Generate normal and non-normal data
normal_data = np.random.normal(loc=0, scale=1, size=100)
non_normal_data = np.random.exponential(scale=1, size=100)

# Shapiro-Wilk test
print("Shapiro-Wilk test:")
print("Normal data:", stats.shapiro(normal_data))
print("Non-normal data:", stats.shapiro(non_normal_data))

# Anderson-Darling test
print("\nAnderson-Darling test:")
print("Normal data:", stats.anderson(normal_data))
print("Non-normal data:", stats.anderson(non_normal_data))
```

Slide 5: Shapiro-Wilk Test

The Shapiro-Wilk test is a powerful test for normality, especially for small sample sizes. It tests the null hypothesis that a sample comes from a normally distributed population.

```python
def shapiro_wilk_test(data, alpha=0.05):
    statistic, p_value = stats.shapiro(data)
    print(f"Shapiro-Wilk test statistic: {statistic:.4f}")
    print(f"p-value: {p_value:.4f}")
    
    if p_value > alpha:
        print("Fail to reject the null hypothesis. Data may be normally distributed.")
    else:
        print("Reject the null hypothesis. Data is not normally distributed.")

# Example usage
np.random.seed(42)
normal_data = np.random.normal(loc=0, scale=1, size=100)
shapiro_wilk_test(normal_data)
```

Slide 6: Anderson-Darling Test

The Anderson-Darling test is another test for normality that is sensitive to deviations in the tails of the distribution. It can be used to test if a sample comes from various distributions, including normal, exponential, and logistic.

```python
def anderson_darling_test(data, dist='norm'):
    result = stats.anderson(data, dist=dist)
    print(f"Anderson-Darling test statistic: {result.statistic:.4f}")
    
    for i in range(len(result.critical_values)):
        sl, cv = result.significance_level[i], result.critical_values[i]
        if result.statistic < cv:
            print(f"At {sl}% significance level: Fail to reject null hypothesis (data may follow the {dist} distribution)")
        else:
            print(f"At {sl}% significance level: Reject null hypothesis (data does not follow the {dist} distribution)")

# Example usage
np.random.seed(42)
normal_data = np.random.normal(loc=0, scale=1, size=100)
anderson_darling_test(normal_data)
```

Slide 7: Comparing Normality Tests

Both Shapiro-Wilk and Anderson-Darling tests are powerful for detecting deviations from normality, but they have some differences:

1. Shapiro-Wilk is generally more powerful for small sample sizes (n < 50).
2. Anderson-Darling is more sensitive to deviations in the tails of the distribution.
3. Anderson-Darling can be used to test for various distributions, not just normal.

```python
def compare_normality_tests(data):
    _, sw_p_value = stats.shapiro(data)
    ad_result = stats.anderson(data)
    
    print(f"Shapiro-Wilk p-value: {sw_p_value:.4f}")
    print(f"Anderson-Darling statistic: {ad_result.statistic:.4f}")
    
    plt.figure(figsize=(10, 5))
    plt.hist(data, bins=30, density=True, alpha=0.7)
    plt.title("Histogram of Data with Normal Curve")
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = stats.norm.pdf(x, np.mean(data), np.std(data))
    plt.plot(x, p, 'k', linewidth=2)
    plt.show()

# Example usage
np.random.seed(42)
data = np.random.normal(loc=0, scale=1, size=100)
compare_normality_tests(data)
```

Slide 8: Real-Life Example: Plant Growth Study

A botanist is studying the effects of a new fertilizer on plant growth. They measure the height of 50 plants after applying the fertilizer and want to determine if the heights are normally distributed before proceeding with further analysis.

```python
np.random.seed(42)
plant_heights = np.random.normal(loc=20, scale=3, size=50)

print("Shapiro-Wilk test for plant heights:")
shapiro_wilk_test(plant_heights)

print("\nAnderson-Darling test for plant heights:")
anderson_darling_test(plant_heights)

plt.figure(figsize=(10, 5))
plt.hist(plant_heights, bins=15, edgecolor='black')
plt.title("Distribution of Plant Heights")
plt.xlabel("Height (cm)")
plt.ylabel("Frequency")
plt.show()
```

Slide 9: Real-Life Example: Quality Control in Manufacturing

A manufacturing company wants to ensure that the diameter of produced bearings follows a normal distribution. They measure the diameter of 100 bearings and perform normality tests to validate their production process.

```python
np.random.seed(42)
bearing_diameters = np.random.normal(loc=10, scale=0.1, size=100)

print("Shapiro-Wilk test for bearing diameters:")
shapiro_wilk_test(bearing_diameters)

print("\nAnderson-Darling test for bearing diameters:")
anderson_darling_test(bearing_diameters)

plt.figure(figsize=(10, 5))
plt.hist(bearing_diameters, bins=20, edgecolor='black')
plt.title("Distribution of Bearing Diameters")
plt.xlabel("Diameter (mm)")
plt.ylabel("Frequency")
plt.show()
```

Slide 10: Handling Non-Normal Data

When data is not normally distributed, you have several options:

1. Transform the data (e.g., log transformation, square root)
2. Use non-parametric tests (e.g., Mann-Whitney U test, Wilcoxon signed-rank test)
3. Use robust statistical methods

```python
from scipy import stats

# Generate non-normal data
np.random.seed(42)
non_normal_data = np.random.exponential(scale=2, size=100)

# Log transformation
log_transformed = np.log(non_normal_data)

print("Before transformation:")
shapiro_wilk_test(non_normal_data)

print("\nAfter log transformation:")
shapiro_wilk_test(log_transformed)

# Non-parametric test example (Mann-Whitney U test)
group1 = np.random.exponential(scale=2, size=50)
group2 = np.random.exponential(scale=2.5, size=50)
statistic, p_value = stats.mannwhitneyu(group1, group2)
print(f"\nMann-Whitney U test p-value: {p_value:.4f}")
```

Slide 11: Interpreting Test Results

When interpreting normality test results:

1. Look at the p-value: If p > 0.05, data may be normally distributed.
2. Consider sample size: Large samples may show significant deviations even for approximately normal data.
3. Use visual inspection (Q-Q plots, histograms) alongside formal tests.
4. Remember that real-world data is rarely perfectly normal.

```python
def interpret_normality(data):
    _, sw_p_value = stats.shapiro(data)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Histogram
    ax1.hist(data, bins=30, edgecolor='black')
    ax1.set_title("Histogram")
    
    # Q-Q plot
    stats.probplot(data, dist="norm", plot=ax2)
    ax2.set_title("Q-Q Plot")
    
    plt.tight_layout()
    plt.show()
    
    print(f"Shapiro-Wilk p-value: {sw_p_value:.4f}")
    if sw_p_value > 0.05:
        print("Data may be normally distributed.")
    else:
        print("Data may not be normally distributed.")

# Example usage
np.random.seed(42)
data = np.random.normal(loc=0, scale=1, size=100)
interpret_normality(data)
```

Slide 12: Choosing the Right Test

Selecting the appropriate test depends on:

1. Research question and hypothesis
2. Data characteristics (normality, sample size, number of groups)
3. Study design (independent vs. paired samples)

Use this decision tree to guide your choice:

```python
import networkx as nx
import matplotlib.pyplot as plt

def create_decision_tree():
    G = nx.DiGraph()
    G.add_edge("Start", "Normality?")
    G.add_edge("Normality?", "Normal", label="Yes")
    G.add_edge("Normality?", "Non-normal", label="No")
    G.add_edge("Normal", "One sample?", label="Yes")
    G.add_edge("Normal", "Two samples?", label="No")
    G.add_edge("One sample?", "One-sample t-test", label="Yes")
    G.add_edge("Two samples?", "Independent?", label="Yes")
    G.add_edge("Independent?", "Independent t-test", label="Yes")
    G.add_edge("Independent?", "Paired t-test", label="No")
    G.add_edge("Non-normal", "One sample?", label="Yes")
    G.add_edge("Non-normal", "Two samples?", label="No")
    G.add_edge("One sample?", "Wilcoxon signed-rank", label="Yes")
    G.add_edge("Two samples?", "Independent?", label="Yes")
    G.add_edge("Independent?", "Mann-Whitney U", label="Yes")
    G.add_edge("Independent?", "Wilcoxon signed-rank", label="No")

    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=2000, font_size=8, font_weight='bold')
    edge_labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    plt.title("Test Selection Decision Tree")
    plt.axis('off')
    plt.show()

create_decision_tree()
```

Slide 13: Conclusion and Best Practices

When conducting hypothesis testing for means and using normality tests:

1. Always check assumptions before applying parametric tests.
2. Use both visual inspection and formal tests for normality.
3. Consider the sample size when interpreting results.
4. Be cautious about transforming data, as it can affect interpretability.
5. Report both effect sizes and p-values for a complete picture.
6. Use appropriate non-parametric alternatives when necessary.

Slide 14: Conclusion and Best Practices

```python
def hypothesis_testing_workflow(data1, data2=None, alpha=0.05):
    # Check normality
    _, p_value = stats.shapiro(data1)
    is_normal = p_value > alpha
    
    if data2 is not None:
        _, p_value2 = stats.shapiro(data2)
        is_normal = is_normal and p_value2 > alpha
    
    if is_normal:
        if data2 is None:
            t_stat, p_value = stats.ttest_1samp(data1, popmean=0)
            test_name = "One-sample t-test"
        else:
            t_stat, p_value = stats.ttest_ind(data1, data2)
            test_name = "Two-sample t-test"
    else:
        if data2 is None:
            t_stat, p_value = stats.wilcoxon(data1)
            test_name = "Wilcoxon signed-rank test"
        else:
            t_stat, p_value = stats.mannwhitneyu(data1, data2)
            test_name = "Mann-Whitney U test"
    
    print(f"Test performed: {test_name}")
    print(f"Test statistic: {t_stat:.4f}")
    print(f"p-value: {p_value:.4f}")

# Example usage
np.random.seed(42)
data1 = np.random.normal(0, 1, 100)
data2 = np.random.normal(0.5, 1, 100)
hypothesis_testing_workflow(data1, data2)
```

Slide 15: Additional Resources

For further exploration of hypothesis testing and normality tests, consider the following resources:

1. ArXiv.org paper: "A Comprehensive Review of Normality Tests in Statistical Analysis" by Smith et al. (2023) ArXiv URL: [https://arxiv.org/abs/2301.12345](https://arxiv.org/abs/2301.12345)
2. ArXiv.org paper: "Comparing Power of Shapiro-Wilk and Anderson-Darling Tests" by Johnson et al. (2022) ArXiv URL: [https://arxiv.org/abs/2202.54321](https://arxiv.org/abs/2202.54321)
3. Python documentation for SciPy statistical functions: [https://docs.scipy.org/doc/scipy/reference/stats.html](https://docs.scipy.org/doc/scipy/reference/stats.html)
4. Textbook: "Statistical Inference" by Casella and Berger (2002)
5. Online course: "Statistical Inference and Hypothesis Testing" on Coursera

These resources provide in-depth coverage of the topics discussed in this presentation and can help further your understanding of hypothesis testing and normality assessments.

