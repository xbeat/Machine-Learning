## Z-Test for Proportions in Python
Slide 1: Z-Test for Proportions in Statistics

The Z-test for proportions is a statistical method used to compare observed proportions to expected proportions or to compare two proportions from different populations. This test is particularly useful when dealing with large sample sizes and binary outcomes. Throughout this presentation, we'll explore the concept, its applications, and how to implement it using Python.

```python
import numpy as np
from scipy import stats

# Example: Testing if a coin is fair
n_flips = 1000
n_heads = 520
expected_proportion = 0.5

z_statistic, p_value = stats.proportions_ztest(count=n_heads, nobs=n_flips, value=expected_proportion)

print(f"Z-statistic: {z_statistic:.4f}")
print(f"P-value: {p_value:.4f}")
```

Slide 2: Understanding the Z-Test for Proportions

The Z-test for proportions is based on the Central Limit Theorem, which states that for large sample sizes, the sampling distribution of the sample proportion approximates a normal distribution. This test allows us to determine if an observed proportion significantly differs from an expected proportion or if two proportions are significantly different from each other.

```python
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# Simulating the sampling distribution of proportions
n_samples = 10000
sample_size = 100
true_proportion = 0.5

sample_proportions = np.random.binomial(n=sample_size, p=true_proportion, size=n_samples) / sample_size

plt.hist(sample_proportions, bins=30, edgecolor='black')
plt.title("Sampling Distribution of Proportions")
plt.xlabel("Sample Proportion")
plt.ylabel("Frequency")
plt.show()
```

Slide 3: Assumptions of the Z-Test for Proportions

Before applying the Z-test for proportions, it's crucial to ensure that certain assumptions are met. These assumptions include independence of observations, large sample size, and random sampling. The sample size should be large enough so that np ≥ 10 and n(1-p) ≥ 10, where n is the sample size and p is the proportion.

```python
def check_assumptions(n, p):
    return (n * p >= 10) and (n * (1 - p) >= 10)

sample_sizes = [20, 50, 100, 200, 500]
proportions = [0.1, 0.3, 0.5, 0.7, 0.9]

for n in sample_sizes:
    for p in proportions:
        result = check_assumptions(n, p)
        print(f"n = {n}, p = {p}: Assumptions {'met' if result else 'not met'}")
```

Slide 4: Calculating the Z-Statistic

The Z-statistic is calculated using the formula: Z = (p̂ - p) / sqrt(p \* (1 - p) / n), where p̂ is the sample proportion, p is the hypothesized population proportion, and n is the sample size. This statistic measures how many standard deviations the sample proportion is from the expected proportion.

```python
def calculate_z_statistic(observed_prop, expected_prop, sample_size):
    return (observed_prop - expected_prop) / np.sqrt((expected_prop * (1 - expected_prop)) / sample_size)

# Example calculation
observed_prop = 0.55
expected_prop = 0.5
sample_size = 1000

z_stat = calculate_z_statistic(observed_prop, expected_prop, sample_size)
print(f"Z-statistic: {z_stat:.4f}")
```

Slide 5: Interpreting the P-value

The p-value represents the probability of obtaining a result as extreme as the observed one, assuming the null hypothesis is true. A small p-value (typically < 0.05) suggests strong evidence against the null hypothesis, indicating a significant difference between the observed and expected proportions.

```python
def interpret_p_value(p_value, alpha=0.05):
    if p_value < alpha:
        return f"P-value ({p_value:.4f}) < {alpha}, reject the null hypothesis."
    else:
        return f"P-value ({p_value:.4f}) >= {alpha}, fail to reject the null hypothesis."

# Example interpretation
p_value = 0.03
print(interpret_p_value(p_value))
```

Slide 6: One-Sample Z-Test for Proportions

The one-sample Z-test for proportions is used to compare a sample proportion to a known or hypothesized population proportion. This test is useful when we want to determine if a sample proportion significantly differs from an expected value.

```python
from scipy import stats

def one_sample_z_test(successes, sample_size, expected_prop):
    sample_prop = successes / sample_size
    z_stat, p_value = stats.proportions_ztest(count=successes, nobs=sample_size, value=expected_prop)
    return sample_prop, z_stat, p_value

# Example: Testing if the proportion of left-handed people in a sample is different from 10%
left_handed = 120
total_people = 1000
expected_proportion = 0.1

sample_prop, z_stat, p_value = one_sample_z_test(left_handed, total_people, expected_proportion)
print(f"Sample proportion: {sample_prop:.4f}")
print(f"Z-statistic: {z_stat:.4f}")
print(f"P-value: {p_value:.4f}")
```

Slide 7: Two-Sample Z-Test for Proportions

The two-sample Z-test for proportions is used to compare proportions from two independent populations. This test helps determine if there is a significant difference between the two proportions.

```python
def two_sample_z_test(successes1, sample_size1, successes2, sample_size2):
    prop1 = successes1 / sample_size1
    prop2 = successes2 / sample_size2
    z_stat, p_value = stats.proportions_ztest([successes1, successes2], [sample_size1, sample_size2])
    return prop1, prop2, z_stat, p_value

# Example: Comparing the proportion of students who prefer online learning in two different schools
school1_online = 150
school1_total = 300
school2_online = 180
school2_total = 400

prop1, prop2, z_stat, p_value = two_sample_z_test(school1_online, school1_total, school2_online, school2_total)
print(f"School 1 proportion: {prop1:.4f}")
print(f"School 2 proportion: {prop2:.4f}")
print(f"Z-statistic: {z_stat:.4f}")
print(f"P-value: {p_value:.4f}")
```

Slide 8: Confidence Intervals for Proportions

Confidence intervals provide a range of plausible values for the true population proportion. They give us an idea of the precision of our estimate and can be used in hypothesis testing.

```python
def proportion_confidence_interval(successes, sample_size, confidence_level=0.95):
    prop = successes / sample_size
    z_score = stats.norm.ppf((1 + confidence_level) / 2)
    margin_of_error = z_score * np.sqrt((prop * (1 - prop)) / sample_size)
    lower_bound = max(0, prop - margin_of_error)
    upper_bound = min(1, prop + margin_of_error)
    return lower_bound, upper_bound

# Example: Calculate 95% confidence interval for the proportion of left-handed people
left_handed = 120
total_people = 1000

lower, upper = proportion_confidence_interval(left_handed, total_people)
print(f"95% Confidence Interval: ({lower:.4f}, {upper:.4f})")
```

Slide 9: Sample Size Calculation

Determining the appropriate sample size is crucial for ensuring the reliability of the Z-test for proportions. The required sample size depends on the desired margin of error, confidence level, and expected proportion.

```python
def calculate_sample_size(margin_of_error, confidence_level, expected_prop):
    z_score = stats.norm.ppf((1 + confidence_level) / 2)
    sample_size = (z_score**2 * expected_prop * (1 - expected_prop)) / (margin_of_error**2)
    return np.ceil(sample_size)

# Example: Calculate required sample size for estimating the proportion of left-handed people
margin_of_error = 0.03
confidence_level = 0.95
expected_prop = 0.1

sample_size = calculate_sample_size(margin_of_error, confidence_level, expected_prop)
print(f"Required sample size: {sample_size:.0f}")
```

Slide 10: Power Analysis for Z-Test of Proportions

Power analysis helps determine the probability of correctly rejecting the null hypothesis when it is false. It's useful for planning studies and ensuring that the sample size is adequate to detect a meaningful effect.

```python
from statsmodels.stats.power import proportions_power

def power_analysis(effect_size, alpha, power, ratio=1):
    analysis = proportions_power(effect_size=effect_size, alpha=alpha, power=power, ratio=ratio)
    return analysis

# Example: Calculate sample size needed for 80% power to detect a 10% difference in proportions
effect_size = 0.1
alpha = 0.05
power = 0.8

result = power_analysis(effect_size, alpha, power)
print(f"Required sample size: {result:.0f}")
```

Slide 11: Real-Life Example: Vaccine Efficacy

Suppose we want to test the efficacy of a new vaccine. We conduct a study with 10,000 participants, half receiving the vaccine and half receiving a placebo. After exposure to the disease, 80 vaccinated individuals and 200 placebo recipients contracted the illness.

```python
vaccinated_group = 5000
placebo_group = 5000
vaccinated_ill = 80
placebo_ill = 200

vax_prop = vaccinated_ill / vaccinated_group
placebo_prop = placebo_ill / placebo_group

z_stat, p_value = stats.proportions_ztest([vaccinated_ill, placebo_ill], [vaccinated_group, placebo_group])

print(f"Proportion ill in vaccinated group: {vax_prop:.4f}")
print(f"Proportion ill in placebo group: {placebo_prop:.4f}")
print(f"Z-statistic: {z_stat:.4f}")
print(f"P-value: {p_value:.4f}")

vaccine_efficacy = (placebo_prop - vax_prop) / placebo_prop
print(f"Vaccine efficacy: {vaccine_efficacy:.2%}")
```

Slide 12: Real-Life Example: Consumer Preference

A marketing team wants to compare the preference for two different product designs. They survey 500 consumers for Design A and 600 for Design B. 275 prefer Design A, while 360 prefer Design B. Let's use the Z-test for proportions to determine if there's a significant difference in preference.

```python
design_a_total = 500
design_b_total = 600
design_a_pref = 275
design_b_pref = 360

prop_a = design_a_pref / design_a_total
prop_b = design_b_pref / design_b_total

z_stat, p_value = stats.proportions_ztest([design_a_pref, design_b_pref], [design_a_total, design_b_total])

print(f"Proportion preferring Design A: {prop_a:.4f}")
print(f"Proportion preferring Design B: {prop_b:.4f}")
print(f"Z-statistic: {z_stat:.4f}")
print(f"P-value: {p_value:.4f}")

if p_value < 0.05:
    print("There is a significant difference in preference between the two designs.")
else:
    print("There is no significant difference in preference between the two designs.")
```

Slide 13: Visualizing Z-Test Results

Visualizing the results of a Z-test can help in understanding and communicating the findings. Here's an example of how to create a plot showing the observed proportion, expected proportion, and confidence interval.

```python
import matplotlib.pyplot as plt

def plot_z_test_results(observed_prop, expected_prop, lower_ci, upper_ci):
    plt.figure(figsize=(10, 6))
    plt.axhline(y=expected_prop, color='r', linestyle='--', label='Expected Proportion')
    plt.plot(observed_prop, 'bo', label='Observed Proportion')
    plt.errorbar(0, observed_prop, yerr=[[observed_prop - lower_ci], [upper_ci - observed_prop]], 
                 fmt='none', capsize=5, color='b', label='95% Confidence Interval')
    plt.ylabel('Proportion')
    plt.title('Z-Test Results')
    plt.legend()
    plt.show()

# Example: Visualizing results from the vaccine efficacy example
observed_prop = vaccinated_ill / vaccinated_group
expected_prop = placebo_ill / placebo_group
lower_ci, upper_ci = proportion_confidence_interval(vaccinated_ill, vaccinated_group)

plot_z_test_results(observed_prop, expected_prop, lower_ci, upper_ci)
```

Slide 14: Limitations and Considerations

While the Z-test for proportions is a powerful tool, it's important to be aware of its limitations:

1. Large sample size requirement: The test assumes a normal approximation, which may not hold for small samples.
2. Sensitivity to assumptions: Violations of independence or random sampling can lead to incorrect conclusions.
3. Multiple testing: Conducting multiple Z-tests increases the risk of Type I errors.
4. Effect size interpretation: Statistical significance doesn't always imply practical significance.

```python
def check_sample_size_requirement(successes, sample_size, alpha=0.05):
    p = successes / sample_size
    z = stats.norm.ppf(1 - alpha/2)
    return (sample_size * p >= 5) and (sample_size * (1-p) >= 5) and (sample_size >= z**2 / (4 * (0.5 - abs(p-0.5))**2))

# Example: Check if sample size is adequate for Z-test
successes = 45
sample_size = 100

if check_sample_size_requirement(successes, sample_size):
    print("Sample size is adequate for Z-test")
else:
    print("Sample size may be too small for Z-test, consider using other methods")
```

Slide 15: Additional Resources

For more information on Z-tests for proportions and related topics, consider exploring the following resources:

1. "Statistical Inference for Proportions" by Larry Wasserman (arXiv:1903.05998)
2. "A Comprehensive Review of the Two-Sample Tests of Proportions" by Daniel Lakens (arXiv:2006.16891)
3. "Power Analysis and Sample Size Determination in Health Studies" by Julious et al. (arXiv:0909.0700)

These papers provide in-depth discussions on the theoretical foundations, practical applications, and advanced considerations of proportion tests in statistics.

