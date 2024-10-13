## Essentials of Probability and Statistics
Slide 1: Introduction to Probability

Probability is the branch of mathematics that deals with the likelihood of events occurring. It forms the foundation for statistical analysis and decision-making under uncertainty.

```python
import random

# Simulating a coin flip
coin = ['Heads', 'Tails']
flips = 1000
results = [random.choice(coin) for _ in range(flips)]

heads_count = results.count('Heads')
probability_heads = heads_count / flips

print(f"Probability of getting Heads: {probability_height:.2f}")
```

Slide 2: Fundamental Concepts of Probability

The three axioms of probability define its mathematical foundation: non-negativity, normalization, and additivity. These principles ensure that probabilities are always between 0 and 1, and the sum of all possible outcomes equals 1.

```python
def check_probability_axioms(probabilities):
    non_negativity = all(p >= 0 for p in probabilities)
    normalization = sum(probabilities) == 1
    additivity = sum(probabilities) == sum(set(probabilities))
    
    return non_negativity and normalization and additivity

# Example probabilities
event_probabilities = [0.2, 0.3, 0.5]
print(f"Probabilities satisfy axioms: {check_probability_axioms(event_probabilities)}")
```

Slide 3: Key Probability Descriptors

Probability mass functions (PMF) for discrete variables and probability density functions (PDF) for continuous variables describe the likelihood of different outcomes. Cumulative distribution functions (CDF) give the probability of a value being less than or equal to a given point.

```python
import numpy as np
import matplotlib.pyplot as plt

# PMF for a discrete uniform distribution
x = np.arange(1, 7)
pmf = np.ones_like(x) / len(x)

plt.bar(x, pmf)
plt.title("PMF of a Fair Die Roll")
plt.xlabel("Outcome")
plt.ylabel("Probability")
plt.show()
```

Slide 4: Central Tendency Metrics

Measures of central tendency include the mean, median, and mode. They provide different perspectives on the typical or central value in a dataset.

```python
import numpy as np

data = [2, 3, 3, 4, 5, 5, 5, 6, 7]

mean = np.mean(data)
median = np.median(data)
mode = max(set(data), key=data.count)

print(f"Mean: {mean}")
print(f"Median: {median}")
print(f"Mode: {mode}")
```

Slide 5: Probability Distributions

Probability distributions describe the likelihood of different outcomes for a random variable. Common distributions include normal, binomial, and Poisson distributions.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

x = np.linspace(-4, 4, 100)
y = norm.pdf(x, 0, 1)

plt.plot(x, y)
plt.title("Standard Normal Distribution")
plt.xlabel("Value")
plt.ylabel("Probability Density")
plt.show()
```

Slide 6: Similarity and Correlation Measures

Correlation coefficients measure the strength and direction of relationships between variables. Common measures include Pearson's correlation for linear relationships and Spearman's rank correlation for monotonic relationships.

```python
import numpy as np

x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 5, 4, 5])

pearson_corr = np.corrcoef(x, y)[0, 1]
print(f"Pearson correlation: {pearson_corr:.2f}")
```

Slide 7: Introduction to Statistics

Statistics involves collecting, analyzing, interpreting, and presenting data. It allows us to make inferences about populations based on sample data.

```python
import numpy as np

population = np.random.normal(loc=100, scale=15, size=10000)
sample = np.random.choice(population, size=100, replace=False)

population_mean = np.mean(population)
sample_mean = np.mean(sample)

print(f"Population mean: {population_mean:.2f}")
print(f"Sample mean: {sample_mean:.2f}")
```

Slide 8: Hypothesis Testing

Hypothesis testing is a statistical method used to make inferences about a population parameter based on sample data. It involves formulating null and alternative hypotheses and using statistical tests to decide whether to reject the null hypothesis.

```python
from scipy import stats

# Example: Testing if a coin is fair
flips = 100
heads = 60

# Perform binomial test
p_value = stats.binom_test(heads, n=flips, p=0.5, alternative='two-sided')

print(f"P-value: {p_value:.4f}")
print(f"{'Reject' if p_value < 0.05 else 'Fail to reject'} the null hypothesis")
```

Slide 9: Z-test

The Z-test is used when the population standard deviation is known and the sample size is large. It compares a sample mean to a known population mean using the standard normal distribution.

```python
from scipy import stats
import numpy as np

population_mean = 100
population_std = 15
sample_size = 30

sample = np.random.normal(loc=105, scale=population_std, size=sample_size)
sample_mean = np.mean(sample)

z_statistic = (sample_mean - population_mean) / (population_std / np.sqrt(sample_size))
p_value = 2 * (1 - stats.norm.cdf(abs(z_statistic)))

print(f"Z-statistic: {z_statistic:.2f}")
print(f"P-value: {p_value:.4f}")
```

Slide 10: t-test

The t-test is used when the population standard deviation is unknown and the sample size is small. It compares means between two groups or a sample mean to a known value.

```python
from scipy import stats
import numpy as np

group1 = np.random.normal(loc=100, scale=15, size=20)
group2 = np.random.normal(loc=110, scale=15, size=20)

t_statistic, p_value = stats.ttest_ind(group1, group2)

print(f"T-statistic: {t_statistic:.2f}")
print(f"P-value: {p_value:.4f}")
```

Slide 11: Chi-Square Test

The Chi-Square test is used to determine if there is a significant association between categorical variables or to test the goodness of fit of observed data to expected distributions.

```python
from scipy.stats import chi2_contingency

observed = np.array([[10, 20, 30],
                     [15, 25, 20]])

chi2, p_value, dof, expected = chi2_contingency(observed)

print(f"Chi-square statistic: {chi2:.2f}")
print(f"P-value: {p_value:.4f}")
```

Slide 12: Analysis of Variance (ANOVA)

ANOVA is used to compare means across three or more groups. It helps determine if there are statistically significant differences between group means.

```python
import numpy as np
from scipy import stats

group1 = np.random.normal(loc=10, scale=2, size=30)
group2 = np.random.normal(loc=12, scale=2, size=30)
group3 = np.random.normal(loc=11, scale=2, size=30)

f_statistic, p_value = stats.f_oneway(group1, group2, group3)

print(f"F-statistic: {f_statistic:.2f}")
print(f"P-value: {p_value:.4f}")
```

Slide 13: Multiple Comparisons

When conducting multiple statistical tests, the chance of a Type I error (false positive) increases. Multiple comparison procedures, such as Bonferroni correction or False Discovery Rate, adjust p-values to control this error rate.

```python
from statsmodels.stats.multitest import multipletests
import numpy as np

# Simulating p-values from multiple tests
p_values = np.random.uniform(0, 1, 10)

# Bonferroni correction
bonferroni_corrected = multipletests(p_values, method='bonferroni')

print("Original p-values:", p_values)
print("Bonferroni corrected p-values:", bonferroni_corrected[1])
```

Slide 14: Factor Analysis

Factor analysis is a statistical method used to describe variability among observed, correlated variables in terms of a potentially lower number of unobserved variables called factors.

```python
from factor_analyzer import FactorAnalyzer
import pandas as pd
import numpy as np

# Generate sample data
np.random.seed(0)
data = pd.DataFrame(np.random.rand(100, 5), columns=['V1', 'V2', 'V3', 'V4', 'V5'])

# Perform factor analysis
fa = FactorAnalyzer(rotation=None, n_factors=2)
fa.fit(data)

# Get factor loadings
loadings = pd.DataFrame(fa.loadings_, columns=['Factor1', 'Factor2'], index=data.columns)
print(loadings)
```

Slide 15: Additional Resources

For further exploration of probability and statistics, consider the following resources:

1. "Introduction to Probability" by Blitzstein and Hwang (arXiv:1302.1281)
2. "Statistical Inference" by Casella and Berger
3. "The Elements of Statistical Learning" by Hastie, Tibshirani, and Friedman (arXiv:1011.0933)
4. Online courses on platforms like Coursera, edX, or MIT OpenCourseWare
5. Statistical software documentation (e.g., Python's SciPy and statsmodels libraries)

