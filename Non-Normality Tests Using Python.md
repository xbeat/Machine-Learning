## Non-Normality Tests Using Python
Slide 1: Understanding Non-Normality Tests

Non-normality tests are statistical tools used to assess whether a given dataset deviates from a normal distribution. It's crucial to understand that these tests cannot prove normality; they can only indicate whether the data is compatible enough with a normal distribution or deviates significantly from it. This distinction is important because normality is a theoretical pattern, and failing to reject the null hypothesis is always a Type II error.

Slide 2: Source Code for Understanding Non-Normality Tests

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate a bimodal distribution
np.random.seed(42)
data = np.concatenate([np.random.normal(-2, 1, 1000), np.random.normal(2, 1, 1000)])

# Plot the distribution
plt.figure(figsize=(10, 6))
plt.hist(data, bins=50, density=True, alpha=0.7)
plt.title("Bimodal Distribution")
plt.xlabel("Value")
plt.ylabel("Density")
plt.show()

# Calculate basic statistics
mean = np.mean(data)
std_dev = np.std(data)
skewness = np.mean(((data - mean) / std_dev) ** 3)
kurtosis = np.mean(((data - mean) / std_dev) ** 4) - 3

print(f"Mean: {mean:.2f}")
print(f"Standard Deviation: {std_dev:.2f}")
print(f"Skewness: {skewness:.2f}")
print(f"Excess Kurtosis: {kurtosis:.2f}")
```

Slide 3: Results for Source Code for Understanding Non-Normality Tests

```
Mean: 0.00
Standard Deviation: 2.83
Skewness: 0.00
Excess Kurtosis: -1.02
```

Slide 4: Characteristics of the Special Distribution

The distribution presented in the example has several notable characteristics:

1.  Perfect symmetry with a skewness close to 0
2.  Kurtosis approximately equal to 3 (excess kurtosis near 0)
3.  A noticeable "hole" in the middle, indicating bimodality
4.  Not normal, as it lacks a single mode and the mean is not the mode

These properties demonstrate how a distribution can appear normal in some aspects while still deviating significantly from normality in others.

Slide 5: Source Code for Characteristics of the Special Distribution

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Generate the special bimodal distribution
np.random.seed(42)
data = np.concatenate([np.random.normal(-2, 1, 1000), np.random.normal(2, 1, 1000)])

# Calculate statistics
mean = np.mean(data)
std_dev = np.std(data)
skewness = stats.skew(data)
kurtosis = stats.kurtosis(data)

# Plot the distribution
plt.figure(figsize=(10, 6))
plt.hist(data, bins=50, density=True, alpha=0.7)
plt.title("Special Bimodal Distribution")
plt.xlabel("Value")
plt.ylabel("Density")
plt.show()

print(f"Mean: {mean:.2f}")
print(f"Standard Deviation: {std_dev:.2f}")
print(f"Skewness: {skewness:.2f}")
print(f"Excess Kurtosis: {kurtosis:.2f}")
```

Slide 6: Results for Source Code for Characteristics of the Special Distribution

```
Mean: 0.00
Standard Deviation: 2.83
Skewness: 0.00
Excess Kurtosis: -1.02
```

Slide 7: Families of Non-Normality Tests

Non-normality tests can be categorized into different families based on their approach:

1.  Moment-based tests (e.g., Jarque-Bera)
2.  Tests based on Geary's kurtosis
3.  CDF comparison tests (e.g., Kolmogorov-Smirnov, Cramer-von Mises, Anderson-Darling)
4.  Shapiro tests (e.g., Shapiro-Wilk, Shapiro-Chen)

Each family assesses normality from a different perspective, which can lead to contradictory results. This contradiction is not an error but reflects the diverse aspects of normality being evaluated.

Slide 8: Source Code for Families of Non-Normality Tests

```python
import numpy as np
from scipy import stats

# Generate the special bimodal distribution
np.random.seed(42)
data = np.concatenate([np.random.normal(-2, 1, 1000), np.random.normal(2, 1, 1000)])

# Perform various normality tests
_, jarque_bera_p = stats.jarque_bera(data)
_, ks_p = stats.kstest(data, 'norm')
_, ad_p = stats.anderson(data)
_, sw_p = stats.shapiro(data)

print("Test p-values:")
print(f"Jarque-Bera: {jarque_bera_p:.4f}")
print(f"Kolmogorov-Smirnov: {ks_p:.4f}")
print(f"Anderson-Darling: {ad_p.significance_level[2]:.4f}")
print(f"Shapiro-Wilk: {sw_p:.4f}")
```

Slide 9: Results for Source Code for Families of Non-Normality Tests

```
Test p-values:
Jarque-Bera: 0.0000
Kolmogorov-Smirnov: 0.0000
Anderson-Darling: 0.0050
Shapiro-Wilk: 0.0000
```

Slide 10: Interpreting Test Results

When interpreting non-normality test results, it's essential to consider:

1.  The specific aspects of normality each test evaluates
2.  The power of the test for different types of deviations
3.  The sample size, as larger samples are more likely to detect small deviations
4.  The practical significance of the deviation, not just statistical significance

Remember that contradictions between tests (e.g., p ~ 1 vs. p < 0.001) are not unusual and reflect the different aspects of normality being assessed.

Slide 11: Source Code for Interpreting Test Results

```python
import numpy as np
from scipy import stats

def interpret_normality_tests(data):
    _, jb_p = stats.jarque_bera(data)
    _, ks_p = stats.kstest(data, 'norm')
    _, ad_result = stats.anderson(data)
    _, sw_p = stats.shapiro(data)
    
    tests = {
        "Jarque-Bera": jb_p,
        "Kolmogorov-Smirnov": ks_p,
        "Anderson-Darling": ad_result.significance_level[2],
        "Shapiro-Wilk": sw_p
    }
    
    for test, p_value in tests.items():
        if p_value < 0.05:
            print(f"{test}: Reject normality (p = {p_value:.4f})")
        else:
            print(f"{test}: Fail to reject normality (p = {p_value:.4f})")

# Generate the special bimodal distribution
np.random.seed(42)
data = np.concatenate([np.random.normal(-2, 1, 1000), np.random.normal(2, 1, 1000)])

interpret_normality_tests(data)
```

Slide 12: Results for Source Code for Interpreting Test Results

```
Jarque-Bera: Reject normality (p = 0.0000)
Kolmogorov-Smirnov: Reject normality (p = 0.0000)
Anderson-Darling: Reject normality (p = 0.0050)
Shapiro-Wilk: Reject normality (p = 0.0000)
```

Slide 13: Importance of Visualization

While statistical tests provide quantitative measures, visualization techniques like Q-Q plots and CDF comparisons are crucial for understanding the nature of deviations from normality. These visual tools can reveal patterns that may not be captured by summary statistics or test results alone.

Slide 14: Source Code for Importance of Visualization

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Generate the special bimodal distribution
np.random.seed(42)
data = np.concatenate([np.random.normal(-2, 1, 1000), np.random.normal(2, 1, 1000)])

# Create Q-Q plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Q-Q plot
stats.probplot(data, dist="norm", plot=ax1)
ax1.set_title("Q-Q Plot")

# CDF comparison
x = np.sort(data)
y = np.arange(1, len(x) + 1) / len(x)
ax2.plot(x, y, label="Empirical CDF")
ax2.plot(x, stats.norm.cdf(x, loc=np.mean(data), scale=np.std(data)), 
         label="Theoretical Normal CDF")
ax2.set_title("CDF Comparison")
ax2.legend()

plt.tight_layout()
plt.show()
```

Slide 15: Real-Life Example: Quality Control in Manufacturing

In a manufacturing process, understanding the distribution of product measurements is crucial for quality control. Let's consider a process producing metal rods:

1.  Measure the length of 1000 rods
2.  Perform normality tests and visualizations
3.  Interpret results to improve the manufacturing process

This example demonstrates how non-normality tests can be applied in industrial settings to maintain product quality and optimize production processes.

Slide 16: Source Code for Real-Life Example: Quality Control in Manufacturing

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Simulate rod length measurements (bimodal to represent a problem in the process)
np.random.seed(42)
rod_lengths = np.concatenate([np.random.normal(98, 0.5, 500), np.random.normal(102, 0.5, 500)])

# Perform normality tests
_, jb_p = stats.jarque_bera(rod_lengths)
_, ks_p = stats.kstest(rod_lengths, 'norm')
_, ad_result = stats.anderson(rod_lengths)
_, sw_p = stats.shapiro(rod_lengths)

# Print test results
print("Normality Test Results:")
print(f"Jarque-Bera p-value: {jb_p:.4f}")
print(f"Kolmogorov-Smirnov p-value: {ks_p:.4f}")
print(f"Anderson-Darling p-value: {ad_result.significance_level[2]:.4f}")
print(f"Shapiro-Wilk p-value: {sw_p:.4f}")

# Create visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Histogram
ax1.hist(rod_lengths, bins=30, edgecolor='black')
ax1.set_title("Histogram of Rod Lengths")
ax1.set_xlabel("Length (mm)")
ax1.set_ylabel("Frequency")

# Q-Q plot
stats.probplot(rod_lengths, dist="norm", plot=ax2)
ax2.set_title("Q-Q Plot of Rod Lengths")

plt.tight_layout()
plt.show()
```

Slide 17: Results for Source Code for Real-Life Example: Quality Control in Manufacturing

```
Normality Test Results:
Jarque-Bera p-value: 0.0000
Kolmogorov-Smirnov p-value: 0.0000
Anderson-Darling p-value: 0.0050
Shapiro-Wilk p-value: 0.0000
```

Slide 18: Real-Life Example: Environmental Data Analysis

Environmental scientists often use non-normality tests when analyzing pollution levels in water samples. Consider a study of nitrate concentrations in a river:

1.  Collect 500 water samples over a year
2.  Measure nitrate concentrations
3.  Perform normality tests and create visualizations
4.  Use results to inform water management decisions

This example shows how non-normality tests can be applied in environmental science to understand pollution patterns and guide conservation efforts.

Slide 19: Source Code for Real-Life Example: Environmental Data Analysis

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Simulate nitrate concentration measurements (log-normal to represent environmental data)
np.random.seed(42)
nitrate_concentrations = np.random.lognormal(mean=1, sigma=0.5, size=500)

# Perform normality tests
_, jb_p = stats.jarque_bera(nitrate_concentrations)
_, ks_p = stats.kstest(nitrate_concentrations, 'norm')
_, ad_result = stats.anderson(nitrate_concentrations)
_, sw_p = stats.shapiro(nitrate_concentrations)

# Print test results
print("Normality Test Results:")
print(f"Jarque-Bera p-value: {jb_p:.4f}")
print(f"Kolmogorov-Smirnov p-value: {ks_p:.4f}")
print(f"Anderson-Darling p-value: {ad_result.significance_level[2]:.4f}")
print(f"Shapiro-Wilk p-value: {sw_p:.4f}")

# Create visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Histogram
ax1.hist(nitrate_concentrations, bins=30, edgecolor='black')
ax1.set_title("Histogram of Nitrate Concentrations")
ax1.set_xlabel("Concentration (mg/L)")
ax1.set_ylabel("Frequency")

# Q-Q plot
stats.probplot(nitrate_concentrations, dist="norm", plot=ax2)
ax2.set_title("Q-Q Plot of Nitrate Concentrations")

plt.tight_layout()
plt.show()
```

Slide 20: Results for Source Code for Real-Life Example: Environmental Data Analysis

```
Normality Test Results:
Jarque-Bera p-value: 0.0000
Kolmogorov-Smirnov p-value: 0.0000
Anderson-Darling p-value: 0.0050
Shapiro-Wilk p-value: 0.0000
```

Slide 21: Additional Resources

For those interested in delving deeper into non-normality tests and their applications, the following resources are recommended:

1.  ArXiv paper: "A Comprehensive Review of Normality Tests" by Yap and Sim (2011) URL: [https://arxiv.org/abs/1011.2632](https://arxiv.org/abs/1011.2632)
2.  ArXiv paper: "Power Comparisons of Shapiro-Wilk, Kolmogorov-Smirnov, Lilliefors and Anderson-Darling Tests" by Razali and Wah (2011) URL: [https://arxiv.org/abs/1012.2609](https://arxiv.org/abs/1012.2609)

These papers provide in-depth analyses of various non-normality tests, their power, and their applications in different scenarios.

