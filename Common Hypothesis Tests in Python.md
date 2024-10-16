## Common Hypothesis Tests in Python
Slide 1: Introduction to Common Hypothesis Tests

Hypothesis testing is a fundamental tool in statistical analysis, allowing researchers to make inferences about populations based on sample data. This presentation will cover five common hypothesis tests: T-Test, ANOVA, Chi-Squared Test, F-Test, and Z-Test. We'll explore their uses, assumptions, and implementation in Python.

```python
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# Generate sample data for visualization
np.random.seed(42)
sample_data = np.random.normal(loc=5, scale=2, size=100)

# Plot histogram of sample data
plt.hist(sample_data, bins=20, edgecolor='black')
plt.title('Sample Data Distribution')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.show()
```

Slide 2: T-Test Overview

The T-Test is used to determine if there is a significant difference between the means of two groups. It's particularly useful when dealing with small sample sizes or when the population standard deviation is unknown. There are three types of t-tests: one-sample, independent two-sample, and paired two-sample.

```python
import scipy.stats as stats

# One-sample t-test
sample = [5.2, 5.8, 6.1, 5.9, 5.5, 5.7, 6.0, 5.6, 5.8, 5.9]
t_statistic, p_value = stats.ttest_1samp(sample, popmean=5.5)

print(f"T-statistic: {t_statistic:.4f}")
print(f"P-value: {p_value:.4f}")

# Interpretation
alpha = 0.05
if p_value < alpha:
    print("Reject the null hypothesis")
else:
    print("Fail to reject the null hypothesis")
```

Slide 3: T-Test Application

Let's consider a real-life example where a researcher wants to determine if a new teaching method improves student test scores. They collect test scores from a sample of students taught using the new method and compare them to the known population mean of 70.

```python
# New teaching method test scores
new_method_scores = [75, 82, 78, 80, 79, 85, 77, 81, 76, 83]

# Perform one-sample t-test
t_statistic, p_value = stats.ttest_1samp(new_method_scores, popmean=70)

print(f"T-statistic: {t_statistic:.4f}")
print(f"P-value: {p_value:.4f}")

# Interpretation
alpha = 0.05
if p_value < alpha:
    print("The new teaching method significantly improves test scores.")
else:
    print("There is not enough evidence to conclude that the new teaching method improves test scores.")
```

Slide 4: ANOVA (Analysis of Variance) Test

ANOVA is used to compare means across three or more groups. It helps determine if there are any statistically significant differences between the means of independent groups. ANOVA is an extension of the t-test for situations where we have more than two groups to compare.

```python
import scipy.stats as stats

# Sample data for three groups
group1 = [4, 5, 6, 5, 4, 5, 6, 5]
group2 = [6, 7, 8, 7, 6, 7, 8, 7]
group3 = [8, 9, 10, 9, 8, 9, 10, 9]

# Perform one-way ANOVA
f_statistic, p_value = stats.f_oneway(group1, group2, group3)

print(f"F-statistic: {f_statistic:.4f}")
print(f"P-value: {p_value:.4f}")

# Interpretation
alpha = 0.05
if p_value < alpha:
    print("Reject the null hypothesis: There are significant differences between groups.")
else:
    print("Fail to reject the null hypothesis: No significant differences between groups.")
```

Slide 5: ANOVA Application

Consider a study comparing the effectiveness of three different fertilizers on plant growth. Researchers measure the height of plants after using each fertilizer and want to determine if there are significant differences in plant growth between the fertilizers.

```python
import scipy.stats as stats
import numpy as np

# Plant heights for each fertilizer group
fertilizer1 = [10, 12, 11, 13, 12, 11, 10, 13]
fertilizer2 = [14, 16, 15, 17, 16, 15, 14, 17]
fertilizer3 = [18, 20, 19, 21, 20, 19, 18, 21]

# Perform one-way ANOVA
f_statistic, p_value = stats.f_oneway(fertilizer1, fertilizer2, fertilizer3)

print(f"F-statistic: {f_statistic:.4f}")
print(f"P-value: {p_value:.4f}")

# Interpretation
alpha = 0.05
if p_value < alpha:
    print("There are significant differences in plant growth between fertilizers.")
else:
    print("There is not enough evidence to conclude differences in plant growth between fertilizers.")

# Visualize the data
import matplotlib.pyplot as plt

plt.boxplot([fertilizer1, fertilizer2, fertilizer3], labels=['Fertilizer 1', 'Fertilizer 2', 'Fertilizer 3'])
plt.title('Plant Height by Fertilizer Type')
plt.ylabel('Plant Height (cm)')
plt.show()
```

Slide 6: Chi-Squared Test

The Chi-Squared test is used to determine if there is a significant relationship between two categorical variables. It compares the observed frequencies in certain categories to the frequencies you might expect to get in those categories by chance.

```python
import scipy.stats as stats
import numpy as np

# Observed frequencies
observed = np.array([[10, 20, 30],
                     [5, 15, 25]])

# Calculate expected frequencies
row_totals = observed.sum(axis=1)
col_totals = observed.sum(axis=0)
total = observed.sum()
expected = np.outer(row_totals, col_totals) / total

# Perform chi-squared test
chi2, p_value, dof, expected = stats.chi2_contingency(observed)

print(f"Chi-squared statistic: {chi2:.4f}")
print(f"P-value: {p_value:.4f}")

# Interpretation
alpha = 0.05
if p_value < alpha:
    print("Reject the null hypothesis: There is a significant relationship between variables.")
else:
    print("Fail to reject the null hypothesis: No significant relationship between variables.")
```

Slide 7: Chi-Squared Test Application

Let's examine a study investigating the relationship between exercise frequency and sleep quality. Researchers categorize participants into three exercise frequency groups (low, medium, high) and three sleep quality levels (poor, average, good).

```python
import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt

# Observed frequencies
observed = np.array([
    [30, 50, 20],  # Low exercise frequency
    [25, 60, 45],  # Medium exercise frequency
    [15, 40, 70]   # High exercise frequency
])

# Perform chi-squared test
chi2, p_value, dof, expected = stats.chi2_contingency(observed)

print(f"Chi-squared statistic: {chi2:.4f}")
print(f"P-value: {p_value:.4f}")

# Interpretation
alpha = 0.05
if p_value < alpha:
    print("There is a significant relationship between exercise frequency and sleep quality.")
else:
    print("There is not enough evidence to conclude a relationship between exercise frequency and sleep quality.")

# Visualize the data
categories = ['Low', 'Medium', 'High']
sleep_quality = ['Poor', 'Average', 'Good']

fig, ax = plt.subplots(figsize=(10, 6))
bottom = np.zeros(3)

for i, sleep in enumerate(sleep_quality):
    ax.bar(categories, observed[:, i], bottom=bottom, label=sleep)
    bottom += observed[:, i]

ax.set_title('Exercise Frequency vs. Sleep Quality')
ax.legend(title='Sleep Quality')
ax.set_xlabel('Exercise Frequency')
ax.set_ylabel('Number of Participants')
plt.show()
```

Slide 8: F-Test

The F-Test is used to compare the variances of two populations. It's often used to check the assumption of equal variances in other statistical tests, such as the t-test or ANOVA. The F-Test can also be used to compare the fits of different statistical models.

```python
import numpy as np
from scipy import stats

# Generate two samples with different variances
np.random.seed(42)
sample1 = np.random.normal(loc=0, scale=1, size=100)
sample2 = np.random.normal(loc=0, scale=1.5, size=100)

# Calculate F-statistic and p-value
f_statistic = np.var(sample1, ddof=1) / np.var(sample2, ddof=1)
dfn, dfd = len(sample1) - 1, len(sample2) - 1
p_value = 1 - stats.f.cdf(f_statistic, dfn, dfd)

print(f"F-statistic: {f_statistic:.4f}")
print(f"P-value: {p_value:.4f}")

# Interpretation
alpha = 0.05
if p_value < alpha:
    print("Reject the null hypothesis: The variances are significantly different.")
else:
    print("Fail to reject the null hypothesis: The variances are not significantly different.")
```

Slide 9: F-Test Application

In a quality control scenario, a manufacturer wants to compare the consistency of two production lines. They measure the weight of products from both lines and use an F-test to determine if there's a significant difference in the variability of the weights.

```python
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Generate sample data for two production lines
np.random.seed(42)
line1_weights = np.random.normal(loc=100, scale=2, size=50)
line2_weights = np.random.normal(loc=100, scale=3, size=50)

# Perform F-test
f_statistic = np.var(line1_weights, ddof=1) / np.var(line2_weights, ddof=1)
dfn, dfd = len(line1_weights) - 1, len(line2_weights) - 1
p_value = 2 * min(stats.f.cdf(f_statistic, dfn, dfd), 1 - stats.f.cdf(f_statistic, dfn, dfd))

print(f"F-statistic: {f_statistic:.4f}")
print(f"P-value: {p_value:.4f}")

# Interpretation
alpha = 0.05
if p_value < alpha:
    print("There is a significant difference in variability between the two production lines.")
else:
    print("There is not enough evidence to conclude a difference in variability between the two production lines.")

# Visualize the data
plt.figure(figsize=(10, 6))
plt.boxplot([line1_weights, line2_weights], labels=['Line 1', 'Line 2'])
plt.title('Product Weights by Production Line')
plt.ylabel('Weight (g)')
plt.show()
```

Slide 10: Z-Test

The Z-Test is used to determine if there is a significant difference between a sample mean and a known population mean. It's typically used when we have a large sample size (n > 30) and know the population standard deviation. The Z-Test assumes that the data is normally distributed.

```python
import numpy as np
from scipy import stats

def z_test(sample, population_mean, population_std):
    n = len(sample)
    sample_mean = np.mean(sample)
    z_statistic = (sample_mean - population_mean) / (population_std / np.sqrt(n))
    p_value = 2 * (1 - stats.norm.cdf(abs(z_statistic)))
    return z_statistic, p_value

# Example data
sample = np.random.normal(loc=105, scale=15, size=100)
population_mean = 100
population_std = 15

z_statistic, p_value = z_test(sample, population_mean, population_std)

print(f"Z-statistic: {z_statistic:.4f}")
print(f"P-value: {p_value:.4f}")

# Interpretation
alpha = 0.05
if p_value < alpha:
    print("Reject the null hypothesis: The sample mean is significantly different from the population mean.")
else:
    print("Fail to reject the null hypothesis: The sample mean is not significantly different from the population mean.")
```

Slide 11: Z-Test Application

A botanist is studying the growth of a particular plant species. The average height of this species is known to be 30 cm with a standard deviation of 5 cm. The botanist collects a sample of 50 plants from a specific region and wants to determine if the average height in this region differs from the known average.

```python
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

def z_test(sample, population_mean, population_std):
    n = len(sample)
    sample_mean = np.mean(sample)
    z_statistic = (sample_mean - population_mean) / (population_std / np.sqrt(n))
    p_value = 2 * (1 - stats.norm.cdf(abs(z_statistic)))
    return z_statistic, p_value

# Generate sample data
np.random.seed(42)
sample = np.random.normal(loc=32, scale=5, size=50)

# Known population parameters
population_mean = 30
population_std = 5

# Perform Z-test
z_statistic, p_value = z_test(sample, population_mean, population_std)

print(f"Sample mean: {np.mean(sample):.2f} cm")
print(f"Z-statistic: {z_statistic:.4f}")
print(f"P-value: {p_value:.4f}")

# Interpretation
alpha = 0.05
if p_value < alpha:
    print("The average plant height in this region is significantly different from the known average.")
else:
    print("There is not enough evidence to conclude that the plant height in this region differs from the known average.")

# Visualize the data
plt.figure(figsize=(10, 6))
plt.hist(sample, bins=15, edgecolor='black')
plt.axvline(population_mean, color='red', linestyle='dashed', linewidth=2, label='Population Mean')
plt.axvline(np.mean(sample), color='green', linestyle='dashed', linewidth=2, label='Sample Mean')
plt.title('Distribution of Plant Heights in the Sample')
plt.xlabel('Height (cm)')
plt.ylabel('Frequency')
plt.legend()
plt.show()
```

Slide 12: Choosing the Right Test

Selecting the appropriate hypothesis test depends on various factors, including the type of data, number of groups, and research question. Here's a simple decision tree to help choose the right test:

```python
import networkx as nx
import matplotlib.pyplot as plt

def create_decision_tree():
    G = nx.DiGraph()
    G.add_edge("Start", "How many groups?")
    G.add_edge("How many groups?", "One Sample", label="1")
    G.add_edge("How many groups?", "Two Samples", label="2")
    G.add_edge("How many groups?", "Three or More", label=">2")
    G.add_edge("One Sample", "Z-Test", label="Large sample, known σ")
    G.add_edge("One Sample", "T-Test", label="Small sample or unknown σ")
    G.add_edge("Two Samples", "Independent", label="Unrelated")
    G.add_edge("Two Samples", "Paired T-Test", label="Related")
    G.add_edge("Independent", "T-Test", label="Continuous")
    G.add_edge("Independent", "Chi-Squared", label="Categorical")
    G.add_edge("Three or More", "ANOVA", label="Continuous")
    G.add_edge("Three or More", "Chi-Squared", label="Categorical")

    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=3000, font_size=8, font_weight='bold')
    edge_labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7)
    
    plt.title("Hypothesis Test Decision Tree")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

create_decision_tree()
```

This decision tree provides a visual guide for selecting the appropriate hypothesis test based on the number of groups and data type. It helps researchers quickly identify which test to use for their specific research question and data structure.

Slide 13: Assumptions and Limitations

Understanding the assumptions and limitations of each hypothesis test is crucial for proper application and interpretation of results. Here's a summary of key considerations:

```python
import pandas as pd
import matplotlib.pyplot as plt

assumptions = {
    'Test': ['T-Test', 'ANOVA', 'Chi-Squared', 'F-Test', 'Z-Test'],
    'Normality': ['Required', 'Required', 'Not Required', 'Required', 'Required'],
    'Equal Variances': ['Assumed*', 'Assumed*', 'Not Applicable', 'Tested', 'Not Applicable'],
    'Independence': ['Required', 'Required', 'Required', 'Required', 'Required'],
    'Sample Size': ['Small OK', 'Moderate', 'Large Preferred', 'Small OK', 'Large Required']
}

df = pd.DataFrame(assumptions)
df = df.set_index('Test')

fig, ax = plt.subplots(figsize=(12, 6))
ax.axis('off')
table = ax.table(cellText=df.values, colLabels=df.columns, rowLabels=df.index, cellLoc='center', loc='center')
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.2, 1.5)
plt.title("Assumptions and Considerations for Hypothesis Tests")
plt.tight_layout()
plt.show()

print("* Equal variances assumption can be relaxed with Welch's t-test or Welch's ANOVA")
```

This table summarizes the key assumptions and considerations for each hypothesis test. It's important to note that violating these assumptions can lead to incorrect conclusions, so researchers should always check their data against these criteria before applying a test.

Slide 14: Practical Tips for Hypothesis Testing

When conducting hypothesis tests, consider these practical tips to ensure reliable results:

1. Always state your hypotheses clearly before collecting data.
2. Choose an appropriate significance level (alpha) before conducting the test.
3. Check for outliers and influential points in your data.
4. Use visualization techniques to explore your data distribution.
5. Consider the practical significance of your results, not just statistical significance.

Slide 15: Practical Tips for Hypothesis Testing

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Generate sample data
np.random.seed(42)
data = np.random.normal(loc=5, scale=2, size=100)

# Add an outlier
data = np.append(data, 20)

# Create plots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Histogram
ax1.hist(data, bins=20, edgecolor='black')
ax1.set_title('Histogram of Data')
ax1.set_xlabel('Value')
ax1.set_ylabel('Frequency')

# Q-Q plot
stats.probplot(data, dist="norm", plot=ax2)
ax2.set_title('Q-Q Plot')

plt.tight_layout()
plt.show()

# Perform normality test
_, p_value = stats.shapiro(data)
print(f"Shapiro-Wilk test p-value: {p_value:.4f}")
print("Interpretation: p-value < 0.05 suggests non-normal distribution")
```

These visualizations and tests help identify potential issues with your data, such as non-normality or outliers, which can affect the validity of your hypothesis test results.

Slide 16: Additional Resources

For further exploration of hypothesis testing and statistical analysis, consider these resources:

1. "Statistical Inference via Data Science: A ModernDive into R and the Tidyverse" by Chester Ismay and Albert Y. Kim (available on ArXiv: [https://arxiv.org/abs/1903.07639](https://arxiv.org/abs/1903.07639))
2. "Practical Statistics for Data Scientists" by Peter Bruce, Andrew Bruce, and Peter Gedeck
3. SciPy documentation for statistical functions: [https://docs.scipy.org/doc/scipy/reference/stats.html](https://docs.scipy.org/doc/scipy/reference/stats.html)
4. StatsModels library for Python: [https://www.statsmodels.org/](https://www.statsmodels.org/)
5. "The Elements of Statistical Learning" by Trevor Hastie, Robert Tibshirani, and Jerome Friedman (available on ArXiv: [https://arxiv.org/abs/1011.0885](https://arxiv.org/abs/1011.0885))

These resources provide in-depth coverage of statistical concepts, practical applications, and implementation in various programming languages, helping you further develop your skills in hypothesis testing and data analysis.

